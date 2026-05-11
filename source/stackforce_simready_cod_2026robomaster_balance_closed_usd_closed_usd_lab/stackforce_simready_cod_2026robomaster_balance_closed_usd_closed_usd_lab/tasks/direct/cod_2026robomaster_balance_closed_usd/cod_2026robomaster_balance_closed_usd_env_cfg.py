from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


ASSET_DIR = Path(__file__).resolve().parents[3] / "assets" / "robots" / "cod_2026robomaster_balance_closed_usd" / "usd"
USD_PATH = ASSET_DIR / "COD-2026RoboMaster-Balance.usd"


@configclass
class Cod2026robomasterBalanceClosedUsdEnvCfg(DirectRLEnvCfg):
    episode_length_s = 20.0
    decimation = 4
    action_scale = 1
    action_control_mode = "position"
    action_space = 6
    observation_space = 29
    state_space = 0

    commanded_lin_vel_x_range = (-0.5, 0.5)
    commanded_ang_vel_z_range = (-1.0, 1.0)
    command_resample_time = 10.0
    viewer = ViewerCfg(
        eye=(3.0, -4.0, 2.0),
        lookat=(0.0, 0.0, 0.45),
        origin_type="world",
        resolution=(1280, 720),
    )

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.28, 0.30, 0.32)),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=4.0, replicate_physics=False)

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(USD_PATH),
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=16,
                sleep_threshold=0.0,
                stabilization_threshold=0.0001,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=16,
                sleep_threshold=0.0,
                stabilization_threshold=0.0001,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.26),
            rot=(1, 0, 0, 0),
            joint_pos={
                ".*_front_joint$": -0.3,
                ".*_rear_joint$": -0.3,
                ".*_Wheel_joint$": 0.0,
            },
            joint_vel={},
        ),
        actuators={
            "leg_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*_front_joint$", ".*_rear_joint$"],
                stiffness=60,
                damping=3,
                effort_limit_sim=30,
                velocity_limit_sim=40.0,
            ),
            "wheel_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*_Wheel_joint$"],
                stiffness=30,
                damping=2,
                effort_limit_sim=3,
                velocity_limit_sim=40.0,
            ),
        },
        soft_joint_pos_limit_factor=0.95,
    )

    actuated_joint_names = [".*_front_joint$", ".*_rear_joint$", ".*_Wheel_joint$"]
    projected_loop_joint_names = ["Right_Closure_Joint1", "Right_Closure_Joint2", "Left_Closure_Joint1", "Left_Closure_Joint2"]
    auto_collision_from_visuals = True
    strip_embedded_ground_prims = False
    base_height_target = 0.26
    action_clip = 1
    visual_disable_resets = False
    reward_scales = {
        "alive": 0.1,
        "upright": 0.25,
        "base_height": 5.0,
        "lin_vel_z": -1.0,
        "ang_vel_xy": -0.05,
        "joint_vel": -0.001,
        "action_rate": -0.01,
        "tracking_lin_vel": 0.5,
        "tracking_ang_vel": 0.2,
        "joint_symmetry": -1.0,
        "custom_reward": 0.0,
    }
