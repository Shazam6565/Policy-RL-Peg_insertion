# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Shaurya-ForcePegInsert-Direct-v0",
    entry_point=f"{__name__}.forge_env:ForgeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.forge_env_cfg:ForgeTaskPegInsertCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Shaurya-ForcePegInsert-PolicyA-Direct-v0",
    entry_point=f"{__name__}.forge_env:ForgeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.forge_env_cfg:ForgeTaskPegInsertPolicyACfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Shaurya-ForcePegInsert-PolicyB-Direct-v0",
    entry_point=f"{__name__}.forge_env:ForgeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.forge_env_cfg:ForgeTaskPegInsertPolicyBCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Shaurya-ForcePegInsert-PolicyC-Direct-v0",
    entry_point=f"{__name__}.forge_env:ForgeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.forge_env_cfg:ForgeTaskPegInsertPolicyCCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

# EXPERIMENTAL — diagnostic only, not an ablation arm. See ForgeEnv._get_dones().
gym.register(
    id="Shaurya-ForcePegInsert-EarlyTerm-Direct-v0",
    entry_point=f"{__name__}.forge_env:ForgeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.forge_env_cfg:ForgeTaskPegInsertEarlyTermCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
