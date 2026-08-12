# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.direct.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, CtrlCfg, FactoryEnvCfg, ObsRandCfg

from .forge_events import randomize_dead_zone
from .forge_tasks_cfg import ForgeGearMesh, ForgeNutThread, ForgePegInsert, ForgeTask

OBS_DIM_CFG.update({"force_threshold": 1, "ft_force": 3})

STATE_DIM_CFG.update({"force_threshold": 1, "ft_force": 3})


@configclass
class ForgeCtrlCfg(CtrlCfg):
    ema_factor_range = [0.025, 0.1]
    default_task_prop_gains = [565.0, 565.0, 565.0, 28.0, 28.0, 28.0]
    task_prop_gains_noise_level = [0.41, 0.41, 0.41, 0.41, 0.41, 0.41]
    pos_threshold_noise_level = [0.25, 0.25, 0.25]
    rot_threshold_noise_level = [0.29, 0.29, 0.29]
    default_dead_zone = [5.0, 5.0, 5.0, 1.0, 1.0, 1.0]


@configclass
class ForgeObsRandCfg(ObsRandCfg):
    fingertip_pos = 0.00025
    fingertip_rot_deg = 0.1
    ft_force = 1.0


@configclass
class EventCfg:
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("held_asset"),
            "mass_distribution_params": (-0.005, 0.005),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    held_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("held_asset"),
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )

    fixed_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("fixed_asset"),
            "static_friction_range": (0.25, 1.25),  # TODO: Set these values based on asset type.
            "dynamic_friction_range": (0.25, 0.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 128,
        },
    )

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )

    dead_zone_thresholds = EventTerm(
        func=randomize_dead_zone,
        mode="interval",
        interval_range_s=(2.0, 2.0),  # (0.25, 0.25)
    )


@configclass
class ForgeEnvCfg(FactoryEnvCfg):
    action_space: int = 7
    obs_rand: ForgeObsRandCfg = ForgeObsRandCfg()
    ctrl: ForgeCtrlCfg = ForgeCtrlCfg()
    task: ForgeTask = ForgeTask()
    events: EventCfg = EventCfg()

    ft_smoothing_factor: float = 0.25

    # Ablation toggles (Policy A-D, see force_aware_peg_insertion_project.md §12).
    # Both default True so Shaurya-ForcePegInsert-Direct-v0's behavior is unchanged.
    use_force_obs: bool = True
    use_force_penalty: bool = True

    # EXPERIMENTAL (§9.6). Default False keeps every existing task on the
    # synchronized timeout-only resets they were trained with. See ForgeEnv._get_dones()
    # for why this is not simply switched on: FactoryEnv's reset drives the *shared*
    # PhysX world (global gravity toggle, un-indexed sim writes, ~1000+ substeps via
    # step_sim_no_action), so a partial reset perturbs the envs that are still running.
    use_early_termination: bool = False

    obs_order: list = [
        "fingertip_pos_rel_fixed",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "ft_force",
        "force_threshold",
    ]
    state_order: list = [
        "fingertip_pos",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "joint_pos",
        "held_pos",
        "held_pos_rel_fixed",
        "held_quat",
        "fixed_pos",
        "fixed_quat",
        "task_prop_gains",
        "ema_factor",
        "ft_force",
        "pos_threshold",
        "rot_threshold",
        "force_threshold",
    ]

    def __post_init__(self):
        super().__post_init__()
        if not self.use_force_obs:
            self.obs_order = [o for o in self.obs_order if o not in ("ft_force", "force_threshold")]
            self.state_order = [s for s in self.state_order if s not in ("ft_force", "force_threshold")]


@configclass
class ForgeTaskPegInsertCfg(ForgeEnvCfg):
    task_name = "peg_insert"
    task = ForgePegInsert()
    episode_length_s = 10.0


@configclass
class ForgeTaskPegInsertPolicyACfg(ForgeTaskPegInsertCfg):
    """Policy A ablation: geometry-only observations, no force penalty (§12)."""

    use_force_obs: bool = False
    use_force_penalty: bool = False


@configclass
class ForgeTaskPegInsertPolicyBCfg(ForgeTaskPegInsertCfg):
    """Policy B ablation: force observations, no force penalty (§12).

    use_early_termination stays at the base False default (same synchronized-timeout
    resets Policy A trained/evaluated with) for apples-to-apples comparability across
    the ablation, not the §9.6 experimental variant.
    """

    use_force_obs: bool = True
    use_force_penalty: bool = False


@configclass
class ForgeTaskPegInsertPolicyCCfg(ForgeTaskPegInsertCfg):
    """Policy C ablation: no force observations, force penalty active (§12).

    use_early_termination stays at the base False default — see PolicyBCfg docstring.
    """

    use_force_obs: bool = False
    use_force_penalty: bool = True


@configclass
class ForgeTaskPegInsertEarlyTermCfg(ForgeTaskPegInsertCfg):
    """EXPERIMENTAL: force-aware baseline with §9.6 early termination enabled.

    Exists to measure what a partial (non-synchronized) reset actually does to the
    envs that are still running — see ForgeEnv._get_dones(). Not intended for
    producing results; the ablation runs use the two configs above.
    """

    use_early_termination: bool = True


@configclass
class ForgeTaskGearMeshCfg(ForgeEnvCfg):
    task_name = "gear_mesh"
    task = ForgeGearMesh()
    episode_length_s = 20.0


@configclass
class ForgeTaskNutThreadCfg(ForgeEnvCfg):
    task_name = "nut_thread"
    task = ForgeNutThread()
    episode_length_s = 30.0
