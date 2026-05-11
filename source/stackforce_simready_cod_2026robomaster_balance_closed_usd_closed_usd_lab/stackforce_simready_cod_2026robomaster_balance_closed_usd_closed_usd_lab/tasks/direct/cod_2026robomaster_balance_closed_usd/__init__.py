import gymnasium as gym

from . import agents


gym.register(
    id="StackForce-Cod2026robomasterBalanceClosedUsd-ClosedUsd-v0",
    entry_point=f"{__name__}.cod_2026robomaster_balance_closed_usd_env:Cod2026robomasterBalanceClosedUsdEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cod_2026robomaster_balance_closed_usd_env_cfg:Cod2026robomasterBalanceClosedUsdEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Cod2026robomasterBalanceClosedUsdPPORunnerCfg",
    },
)
