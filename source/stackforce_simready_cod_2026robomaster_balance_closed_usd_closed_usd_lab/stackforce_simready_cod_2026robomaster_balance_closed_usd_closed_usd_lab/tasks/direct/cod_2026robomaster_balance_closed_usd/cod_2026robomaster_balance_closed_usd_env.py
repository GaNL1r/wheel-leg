from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaacsim.core.utils.stage import get_current_stage
from pxr import Sdf, UsdPhysics

from .custom_rewards import compute_custom_reward_terms
from .cod_2026robomaster_balance_closed_usd_env_cfg import Cod2026robomasterBalanceClosedUsdEnvCfg


class Cod2026robomasterBalanceClosedUsdEnv(DirectRLEnv):
    cfg: Cod2026robomasterBalanceClosedUsdEnvCfg

    def __init__(self, cfg: Cod2026robomasterBalanceClosedUsdEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._actuated_joint_ids, self._actuated_joint_names = self._robot.find_joints(
            self.cfg.actuated_joint_names, preserve_order=True
        )
        if len(self._actuated_joint_ids) != gym.spaces.flatdim(self.single_action_space):
            raise RuntimeError(
                "Closed-chain USD actuator mismatch: "
                f"configured action_space={gym.spaces.flatdim(self.single_action_space)}, "
                f"matched_joints={len(self._actuated_joint_ids)}, "
                f"matched_names={self._actuated_joint_names}"
            )
        self._capture_usd_default_joint_state()
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in self.cfg.reward_scales.keys()
        }
        self._commands = torch.zeros(self.num_envs, 2, device=self.device)
        self._command_time_left = torch.zeros(self.num_envs, device=self.device)
        self._left_right_pairs = self._build_left_right_pairs()

    def _capture_usd_default_joint_state(self):
        # Closed-chain USD assets often rely on authored passive joint coordinates.
        # Isaac Lab's config defaults every joint to zero unless we explicitly preserve
        # the parsed PhysX state before the first reset.
        self._robot.update(0.0)
        joint_pos = self._robot.data.joint_pos.clone()
        joint_vel = torch.zeros_like(self._robot.data.default_joint_vel)
        self._robot.data.default_joint_pos[:] = joint_pos
        self._robot.data.default_joint_vel[:] = joint_vel
        self._robot.data.joint_pos_target[:] = joint_pos
        self._robot.data.joint_vel_target[:] = joint_vel

    def _resample_commands(self, env_ids):
        lin_min, lin_max = self.cfg.commanded_lin_vel_x_range
        ang_min, ang_max = self.cfg.commanded_ang_vel_z_range
        self._commands[env_ids, 0] = torch.empty(len(env_ids), device=self.device).uniform_(lin_min, lin_max)
        self._commands[env_ids, 1] = torch.empty(len(env_ids), device=self.device).uniform_(ang_min, ang_max)
        self._command_time_left[env_ids] = self.cfg.command_resample_time

    def _build_left_right_pairs(self):
        name_to_idx = {name: idx for idx, name in enumerate(self._actuated_joint_names)}
        pairs = []
        for name, idx in name_to_idx.items():
            counterpart = name.replace("Left_", "Right_") if "Left_" in name else name.replace("Right_", "Left_")
            right_idx = name_to_idx.get(counterpart)
            if right_idx is not None and idx < right_idx:
                pairs.append((idx, right_idx))
        return pairs

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._strip_embedded_ground_prims()
        self._patch_projected_loop_joints()
        self._patch_missing_rigid_body_collisions()
        self.scene.articulations["robot"] = self._robot
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light_cfg = sim_utils.DomeLightCfg(intensity=2400.0, color=(0.78, 0.82, 0.9))
        light_cfg.func("/World/Light", light_cfg)

    def _patch_projected_loop_joints(self):
        # Some closed-chain USD files model the closure joint as a regular
        # PhysX joint excluded from the reduced-coordinate articulation. Enable
        # projection on those closure joints so their anchors stay coincident.
        loop_joint_names = set(getattr(self.cfg, "projected_loop_joint_names", []))
        if not loop_joint_names:
            return
        stage = get_current_stage()
        patched = []
        for prim in stage.TraverseAll():
            if prim.GetName() not in loop_joint_names:
                continue
            prim.CreateAttribute("physxJoint:enableProjection", Sdf.ValueTypeNames.Bool).Set(True)
            prim.CreateAttribute("physxJoint:projectionLinearTolerance", Sdf.ValueTypeNames.Float).Set(0.002)
            prim.CreateAttribute("physxJoint:projectionAngularTolerance", Sdf.ValueTypeNames.Float).Set(0.05)
            patched.append(str(prim.GetPath()))
        patched_names = {path.rsplit("/", 1)[-1] for path in patched}
        missing = sorted(loop_joint_names - patched_names)
        if missing:
            raise RuntimeError(f"Missing projected closed-chain loop joints: {missing}; patched={patched}")

    def _strip_embedded_ground_prims(self):
        # Some shared USD examples include a demo GroundPlane inside the robot
        # asset. When Isaac Lab spawns the asset as an Articulation, that plane
        # moves with the robot root and can hide the real terrain or collide at
        # the robot spawn height. Remove those demo-only ground prims at runtime.
        if not getattr(self.cfg, "strip_embedded_ground_prims", False):
            return
        stage = get_current_stage()
        to_remove = []
        for prim in stage.TraverseAll():
            path = str(prim.GetPath())
            if not path.startswith("/World/envs/") or "/Robot/" not in path:
                continue
            name = prim.GetName().lower()
            if name in {"groundplane", "ground_plane"} or (name.startswith("ground") and prim.GetTypeName() == "Plane"):
                to_remove.append(prim.GetPath())
        for path in to_remove:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                prim.SetActive(False)
        if to_remove:
            print(f"[StackForce] Deactivated embedded demo ground prims: {[str(path) for path in to_remove]}", flush=True)

    def _patch_missing_rigid_body_collisions(self):
        # Some USD examples contain rigid-body visual meshes but only partial
        # collision coverage. Only patch child-link bodies (passive leg links)
        # to avoid expensive convex-hull computation on irrelevant geometry.
        if not getattr(self.cfg, "auto_collision_from_visuals", False):
            return
        stage = get_current_stage()
        patched = []
        # collect all prims once
        all_prims = list(stage.TraverseAll())
        for prim in all_prims:
            prim_name = prim.GetName().lower()
            if "child" not in prim_name and "link" not in prim_name:
                continue
            schemas = {str(s) for s in prim.GetAppliedSchemas()}
            if "PhysicsRigidBodyAPI" not in schemas:
                continue
            descendants = [p for p in all_prims if p.GetPath().HasPrefix(prim.GetPath()) and p != prim]
            has_collision = any(
                "PhysicsCollisionAPI" in {str(s) for s in d.GetAppliedSchemas()} for d in descendants
            )
            if has_collision:
                continue
            for item in descendants:
                if item.GetTypeName() != "Mesh":
                    continue
                UsdPhysics.CollisionAPI.Apply(item)
                mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(item)
                mesh_collision.CreateApproximationAttr().Set("convexHull")
                patched.append(str(item.GetPath()))
        if patched:
            print(f"[StackForce] Added convex-hull collision to {len(patched)} visual meshes without colliders.", flush=True)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = torch.clamp(actions.clone(), -self.cfg.action_clip, self.cfg.action_clip)
        if self.cfg.action_control_mode == "effort":
            self._processed_actions = self.cfg.action_scale * self._actions
        else:
            default_pos = self._robot.data.default_joint_pos[:, self._actuated_joint_ids]
            self._processed_actions = self.cfg.action_scale * self._actions + default_pos
        self._command_time_left -= self.step_dt
        overdue = (self._command_time_left <= 0).nonzero(as_tuple=False).flatten()
        if len(overdue) > 0:
            self._resample_commands(overdue)

    def _apply_action(self):
        if self.cfg.action_control_mode == "effort":
            self._robot.set_joint_effort_target(self._processed_actions, joint_ids=self._actuated_joint_ids)
        else:
            self._robot.set_joint_position_target(self._robot.data.default_joint_pos)
            self._robot.set_joint_velocity_target(self._robot.data.default_joint_vel)
            self._robot.set_joint_position_target(self._processed_actions, joint_ids=self._actuated_joint_ids)

    def _get_observations(self) -> dict:
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                self._robot.data.joint_pos[:, self._actuated_joint_ids]
                - self._robot.data.default_joint_pos[:, self._actuated_joint_ids],
                self._robot.data.joint_vel[:, self._actuated_joint_ids],
                self._actions,
                self._commands,
            ],
            dim=-1,
        )
        self._previous_actions = self._actions.clone()
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        scales = self.cfg.reward_scales
        root_height = self._robot.data.root_pos_w[:, 2] - self._terrain.env_origins[:, 2]
        base_height_error = torch.square(root_height - self.cfg.base_height_target)
        upright_error = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)
        lin_vel_z = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        ang_vel_xy = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        joint_vel = torch.sum(torch.square(self._robot.data.joint_vel[:, self._actuated_joint_ids]), dim=1)
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        lin_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 0] - self._commands[:, 0])
        ang_vel_error = torch.square(self._robot.data.root_ang_vel_b[:, 2] - self._commands[:, 1])
        joint_symmetry = torch.zeros(self.num_envs, device=self.device)
        for li, ri in self._left_right_pairs:
            joint_symmetry += torch.square(
                self._robot.data.joint_pos[:, self._actuated_joint_ids[li]]
                - self._robot.data.joint_pos[:, self._actuated_joint_ids[ri]]
            )
        rewards = {
            "alive": torch.ones(self.num_envs, device=self.device) * scales.get("alive", 0.0),
            "upright": torch.exp(-upright_error / 0.25) * scales.get("upright", 0.0),
            "base_height": base_height_error * scales.get("base_height", 0.0),
            "lin_vel_z": lin_vel_z * scales.get("lin_vel_z", 0.0),
            "ang_vel_xy": ang_vel_xy * scales.get("ang_vel_xy", 0.0),
            "joint_vel": joint_vel * scales.get("joint_vel", 0.0),
            "action_rate": action_rate * scales.get("action_rate", 0.0),
            "tracking_lin_vel": torch.exp(-lin_vel_error / 0.25) * scales.get("tracking_lin_vel", 0.0),
            "tracking_ang_vel": torch.exp(-ang_vel_error / 0.25) * scales.get("tracking_ang_vel", 0.0),
            "joint_symmetry": joint_symmetry * scales.get("joint_symmetry", 0.0),
        }
        for key, value in compute_custom_reward_terms(self).items():
            rewards[key] = rewards.get(key, torch.zeros_like(value)) + value * scales.get(key, 0.0)
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0) * self.step_dt
        for key, value in rewards.items():
            if key in self._episode_sums:
                self._episode_sums[key] += value * self.step_dt
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if getattr(self.cfg, "visual_disable_resets", False):
            no_reset = torch.zeros_like(time_out)
            return no_reset, no_reset
        root_height = self._robot.data.root_pos_w[:, 2] - self._terrain.env_origins[:, 2]
        upright_error = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)
        died = (root_height < 0.16) | (upright_error > 0.5)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        self._resample_commands(env_ids)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.set_joint_velocity_target(joint_vel, env_ids=env_ids)
        for key in self._episode_sums.keys():
            self._episode_sums[key][env_ids] = 0.0
