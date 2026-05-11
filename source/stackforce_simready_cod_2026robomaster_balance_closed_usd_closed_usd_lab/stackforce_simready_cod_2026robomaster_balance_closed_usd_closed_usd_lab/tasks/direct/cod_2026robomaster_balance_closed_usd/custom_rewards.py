from __future__ import annotations

import torch


def compute_custom_reward_terms(env) -> dict[str, torch.Tensor]:
    """Return additional unscaled reward terms for this closed-chain USD task."""

    # --- wheel_air penalty: punish retracted legs (wheels likely off ground) ---
    leg_mask = torch.tensor(
        ["Wheel" not in name for name in env._actuated_joint_names],
        device=env.device,
    )
    leg_joint_pos = env._robot.data.joint_pos[:, env._actuated_joint_ids][:, leg_mask]
    mean_leg_pos = leg_joint_pos.mean(dim=1)
    # Punish when mean leg joint position falls below 0.1 rad (legs retracted)
    wheel_air = torch.square(torch.clamp(0.1 - mean_leg_pos, min=0))

    return {"wheel_air": wheel_air}
