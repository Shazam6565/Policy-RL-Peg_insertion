# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.direct.factory.factory_tasks_cfg import FactoryTask, GearMesh, NutThread, PegInsert


@configclass
class ForgeTask(FactoryTask):
    action_penalty_ee_scale: float = 0.0
    action_penalty_asset_scale: float = 0.001
    action_grad_penalty_scale: float = 0.1
    contact_penalty_scale: float = 0.05
    delay_until_ratio: float = 0.25
    contact_penalty_threshold_range = [5.0, 10.0]
    # Hard force limit for early termination (§9.6) — distinct from the soft
    # contact_penalty_threshold_range above, which only shapes reward.
    #
    # Kept at the original 50.0 N placeholder (5x the soft [5, 10] N band) rather
    # than the 15.0 N figure measured from a 20-iteration near-random checkpoint
    # (peak 11.0 N / mean 6.8 N over 9 episodes) — that data point isn't
    # representative of a trained policy's force profile and shouldn't silently
    # change what Policy A's real training run (step 7) is evaluated against.
    # Revisit once a properly trained checkpoint exists to measure against instead.
    force_limit_threshold: float = 50.0


@configclass
class ForgePegInsert(PegInsert, ForgeTask):
    contact_penalty_scale: float = 0.2


@configclass
class ForgeGearMesh(GearMesh, ForgeTask):
    contact_penalty_scale: float = 0.05


@configclass
class ForgeNutThread(NutThread, ForgeTask):
    contact_penalty_scale: float = 0.05
