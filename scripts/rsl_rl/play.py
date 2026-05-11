import argparse
import inspect
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained StackForce closed-chain USD policy.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--checkpoint", type=str, default="model_.*.pt")
parser.add_argument("--load_run", type=str, default=".*")
parser.add_argument("--num_steps", type=int, default=500, help="Number of steps to run. Use 0 to run until the window is closed.")
parser.add_argument(
    "--disable_resets",
    action="store_true",
    default=False,
    help="Disable environment reset during visual play so short or unstable policies do not instantly jump back to the start pose.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import stackforce_simready_cod_2026robomaster_balance_closed_usd_closed_usd_lab.tasks  # noqa: F401


class LegacyRslRlVecEnvWrapper:
    def __init__(self, env, clip_actions=None):
        self.env = env
        self.clip_actions = clip_actions
        self.num_envs = env.unwrapped.num_envs
        self.device = env.unwrapped.device
        self.max_episode_length = env.unwrapped.max_episode_length
        self.num_actions = gym.spaces.flatdim(env.unwrapped.single_action_space)
        obs_dict, extras = self.env.reset()
        self.obs_buf = obs_dict["policy"]
        self.privileged_obs_buf = obs_dict.get("critic")
        self.num_obs = self.obs_buf.shape[-1]
        self.num_privileged_obs = self.privileged_obs_buf.shape[-1] if self.privileged_obs_buf is not None else None
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_length_buf = env.unwrapped.episode_length_buf
        self.extras = extras

    def _obs_as_dict(self):
        result = {"policy": self.obs_buf}
        if self.privileged_obs_buf is not None:
            result["critic"] = self.privileged_obs_buf
        return TensorDict(result, batch_size=self.num_envs, device=self.device)

    def get_observations(self):
        return self._obs_as_dict()

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset(self, env_ids=None):
        del env_ids
        obs_dict, extras = self.env.reset()
        self.obs_buf = obs_dict["policy"]
        self.privileged_obs_buf = obs_dict.get("critic")
        self.extras = extras
        return self._obs_as_dict(), self.privileged_obs_buf

    def step(self, actions):
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        obs_dict, rewards, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)
        if not self.env.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        if "log" in extras and "episode" not in extras:
            extras["episode"] = extras["log"]
        self.obs_buf = obs_dict["policy"]
        self.privileged_obs_buf = obs_dict.get("critic")
        self.rew_buf = rewards
        self.reset_buf = dones
        self.extras = extras
        return self._obs_as_dict(), rewards, dones, extras

    def close(self):
        return self.env.close()


def _runner_uses_nested_class_name():
    try:
        source = inspect.getsource(OnPolicyRunner)
    except OSError:
        return False
    return 'algorithm"]["class_name' in source or "resolve_callable" in source


def to_compatible_rsl_rl_cfg(agent_cfg):
    data = agent_cfg.to_dict() if hasattr(agent_cfg, "to_dict") else dict(agent_cfg)
    allowed_policy_keys = {"actor_hidden_dims", "critic_hidden_dims", "activation", "init_noise_std", "clip_actions"}
    allowed_algorithm_keys = {
        "num_learning_epochs", "num_mini_batches", "clip_param", "gamma", "lam", "value_loss_coef",
        "entropy_coef", "learning_rate", "max_grad_norm", "use_clipped_value_loss", "schedule", "desired_kl", "use_spo"
    }
    policy_cfg = {key: value for key, value in dict(data["policy"]).items() if key in allowed_policy_keys}
    algorithm_cfg = {key: value for key, value in dict(data["algorithm"]).items() if key in allowed_algorithm_keys}
    runner_cfg = {key: value for key, value in data.items() if key not in {"policy", "algorithm", "class_name"}}
    if _runner_uses_nested_class_name():
        policy_cfg.setdefault("class_name", "ActorCritic")
        algorithm_cfg.setdefault("class_name", "PPO")
        return {"runner": runner_cfg, "policy": policy_cfg, "algorithm": algorithm_cfg}
    else:
        policy_cfg.setdefault("class_name", "ActorCritic")
        algorithm_cfg.setdefault("class_name", "PPO")
        result = dict(runner_cfg)
        result["policy"] = policy_cfg
        result["algorithm"] = algorithm_cfg
        return result


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.disable_resets and hasattr(env_cfg, "visual_disable_resets"):
        env_cfg.visual_disable_resets = True
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    checkpoint_arg = args_cli.checkpoint
    resume_path = os.path.abspath(checkpoint_arg) if os.path.isfile(checkpoint_arg) else get_checkpoint_path(
        log_root_path, args_cli.load_run, checkpoint_arg
    )
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    env = gym.make(args_cli.task, cfg=env_cfg)
    wrapped_env = LegacyRslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
    runner = OnPolicyRunner(wrapped_env, to_compatible_rsl_rl_cfg(agent_cfg), log_dir=None, device=env.unwrapped.device)
    runner.load(resume_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    env.reset()
    obs = wrapped_env.get_observations()
    steps = 0
    with torch.inference_mode():
        while simulation_app.is_running():
            actions = policy(obs)
            obs, _, _, _ = wrapped_env.step(actions)
            steps += 1
            if args_cli.num_steps > 0 and steps >= args_cli.num_steps:
                break
    print(f"PLAY_COMPLETED steps={steps} checkpoint={resume_path}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
