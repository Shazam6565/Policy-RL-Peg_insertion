# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import torch
import warp as wp

from isaaclab.utils.math import (
    axis_angle_from_quat,
    euler_xyz_from_quat,
    quat_conjugate,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_mul,
)

from isaaclab_tasks.direct.factory import factory_utils
from isaaclab_tasks.direct.factory.factory_env import FactoryEnv

from . import forge_utils
from .forge_env_cfg import ForgeEnvCfg


class ForgeEnv(FactoryEnv):
    cfg: ForgeEnvCfg

    def __init__(self, cfg: ForgeEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize additional randomization and logging tensors."""
        super().__init__(cfg, render_mode, **kwargs)

        # Success prediction.
        self.success_pred_scale = 0.0
        self.first_pred_success_tx = {}
        for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
            self.first_pred_success_tx[thresh] = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # Flip quaternions.
        self.flip_quats = torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)

        # Force sensor information.
        self.force_sensor_body_idx = self._robot.body_names.index("force_sensor")
        self.force_sensor_smooth = torch.zeros((self.num_envs, 6), device=self.device)
        self.force_sensor_world_smooth = torch.zeros((self.num_envs, 6), device=self.device)

        # Set nominal dynamics parameters for randomization.
        self.default_gains = torch.tensor(self.cfg.ctrl.default_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.default_pos_threshold = torch.tensor(self.cfg.ctrl.pos_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.default_rot_threshold = torch.tensor(self.cfg.ctrl.rot_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.default_dead_zone = torch.tensor(self.cfg.ctrl.default_dead_zone, device=self.device).repeat(
            (self.num_envs, 1)
        )

        self.pos_threshold = self.default_pos_threshold.clone()
        self.rot_threshold = self.default_rot_threshold.clone()

        # Per-episode accumulators for evaluation (§11.4 episode-level logging schema).
        # Computed here rather than read back from outside because DirectRLEnv.step()
        # resets terminated envs' pose data before returning control to the caller —
        # by the time an external eval script sees post-step state, the episode that
        # just ended has already been overwritten by the next one.
        self.ep_max_contact_force = torch.zeros(self.num_envs, device=self.device)
        self.ep_sum_contact_force = torch.zeros(self.num_envs, device=self.device)
        self.ep_force_above_threshold_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.ep_step_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.ep_initial_xy_offset = torch.zeros(self.num_envs, device=self.device)
        self.ep_initial_orientation_error = torch.zeros(self.num_envs, device=self.device)
        # Sticky within an episode — see _get_dones(). Deliberately NOT named
        # ep_succeeded: the base FactoryEnv already owns that as a long buffer for
        # success-time logging (factory_env.py:81) and resets it itself.
        self.ep_success_latched = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.ep_force_limit_hit = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    def _compute_intermediate_values(self, dt):
        """Add noise to observations for force sensing."""
        super()._compute_intermediate_values(dt)

        # Add noise to fingertip position.
        pos_noise_level, rot_noise_level_deg = self.cfg.obs_rand.fingertip_pos, self.cfg.obs_rand.fingertip_rot_deg
        fingertip_pos_noise = torch.randn((self.num_envs, 3), dtype=torch.float32, device=self.device)
        fingertip_pos_noise = fingertip_pos_noise @ torch.diag(
            torch.tensor([pos_noise_level, pos_noise_level, pos_noise_level], dtype=torch.float32, device=self.device)
        )
        self.noisy_fingertip_pos = self.fingertip_midpoint_pos + fingertip_pos_noise

        rot_noise_axis = torch.randn((self.num_envs, 3), dtype=torch.float32, device=self.device)
        rot_noise_axis /= torch.linalg.norm(rot_noise_axis, dim=1, keepdim=True)
        rot_noise_angle = torch.randn((self.num_envs,), dtype=torch.float32, device=self.device) * np.deg2rad(
            rot_noise_level_deg
        )
        self.noisy_fingertip_quat = quat_mul(
            self.fingertip_midpoint_quat, quat_from_angle_axis(rot_noise_angle, rot_noise_axis)
        )
        self.noisy_fingertip_quat[:, [0, 3]] = 0.0
        self.noisy_fingertip_quat = self.noisy_fingertip_quat * self.flip_quats.unsqueeze(-1)

        # Repeat finite differencing with noisy fingertip positions.
        self.ee_linvel_fd = (self.noisy_fingertip_pos - self.prev_fingertip_pos) / dt
        self.prev_fingertip_pos = self.noisy_fingertip_pos.clone()

        # Add state differences if velocity isn't being added.
        rot_diff_quat = quat_mul(self.noisy_fingertip_quat, quat_conjugate(self.prev_fingertip_quat))
        rot_diff_quat *= torch.sign(rot_diff_quat[:, 3]).unsqueeze(-1)  # W component is at index 3 in XYZW format
        rot_diff_aa = axis_angle_from_quat(rot_diff_quat)
        self.ee_angvel_fd = rot_diff_aa / dt
        self.ee_angvel_fd[:, 0:2] = 0.0
        self.prev_fingertip_quat = self.noisy_fingertip_quat.clone()

        # Update and smooth force values.
        self.force_sensor_world = wp.to_torch(self._robot.root_view.get_link_incoming_joint_force())[
            :, self.force_sensor_body_idx
        ]

        alpha = self.cfg.ft_smoothing_factor
        self.force_sensor_world_smooth = alpha * self.force_sensor_world + (1 - alpha) * self.force_sensor_world_smooth

        self.force_sensor_smooth = torch.zeros_like(self.force_sensor_world)
        identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.force_sensor_smooth[:, :3], self.force_sensor_smooth[:, 3:6] = forge_utils.change_FT_frame(
            self.force_sensor_world_smooth[:, 0:3],
            self.force_sensor_world_smooth[:, 3:6],
            (identity_quat, torch.zeros((self.num_envs, 3), device=self.device)),
            (identity_quat, self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise),
        )

        # Compute noisy force values.
        force_noise = torch.randn((self.num_envs, 3), dtype=torch.float32, device=self.device)
        force_noise *= self.cfg.obs_rand.ft_force
        self.noisy_force = self.force_sensor_smooth[:, 0:3] + force_noise

    def _get_observations(self):
        """Add additional FORGE observations."""
        obs_dict, state_dict = self._get_factory_obs_state_dict()

        noisy_fixed_pos = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        prev_actions = self.actions.clone()
        prev_actions[:, 3:5] = 0.0

        obs_dict.update(
            {
                "fingertip_pos": self.noisy_fingertip_pos,
                "fingertip_pos_rel_fixed": self.noisy_fingertip_pos - noisy_fixed_pos,
                "fingertip_quat": self.noisy_fingertip_quat,
                "force_threshold": self.contact_penalty_thresholds[:, None],
                "ft_force": self.noisy_force,
                "prev_actions": prev_actions,
            }
        )

        state_dict.update(
            {
                "ema_factor": self.ema_factor,
                "ft_force": self.force_sensor_smooth[:, 0:3],
                "force_threshold": self.contact_penalty_thresholds[:, None],
                "prev_actions": prev_actions,
            }
        )

        obs_tensors = factory_utils.collapse_obs_dict(obs_dict, self.cfg.obs_order + ["prev_actions"])
        state_tensors = factory_utils.collapse_obs_dict(state_dict, self.cfg.state_order + ["prev_actions"])
        return {"policy": obs_tensors, "critic": state_tensors}

    def _apply_action(self):
        """FORGE actions are defined as targets relative to the fixed asset."""
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        # Step (0): Scale actions to allowed range.
        pos_actions = self.actions[:, 0:3]
        pos_actions = pos_actions @ torch.diag(torch.tensor(self.cfg.ctrl.pos_action_bounds, device=self.device))

        rot_actions = self.actions[:, 3:6]
        rot_actions = rot_actions @ torch.diag(torch.tensor(self.cfg.ctrl.rot_action_bounds, device=self.device))

        # Step (1): Compute desired pose targets in EE frame.
        # (1.a) Position. Action frame is assumed to be the top of the bolt (noisy estimate).
        fixed_pos_action_frame = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        ctrl_target_fingertip_preclipped_pos = fixed_pos_action_frame + pos_actions
        # (1.b) Enforce rotation action constraints.
        rot_actions[:, 0:2] = 0.0

        # Assumes joint limit is in (+x, -y)-quadrant of world frame.
        rot_actions[:, 2] = np.deg2rad(-180.0) + np.deg2rad(270.0) * (rot_actions[:, 2] + 1.0) / 2.0  # Joint limit.
        # (1.c) Get desired orientation target.
        bolt_frame_quat = quat_from_euler_xyz(roll=rot_actions[:, 0], pitch=rot_actions[:, 1], yaw=rot_actions[:, 2])

        rot_180_euler = torch.tensor([np.pi, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        quat_bolt_to_ee = quat_from_euler_xyz(
            roll=rot_180_euler[:, 0], pitch=rot_180_euler[:, 1], yaw=rot_180_euler[:, 2]
        )

        ctrl_target_fingertip_preclipped_quat = quat_mul(quat_bolt_to_ee, bolt_frame_quat)

        # Step (2): Clip targets if they are too far from current EE pose.
        # (2.a): Clip position targets.
        self.delta_pos = ctrl_target_fingertip_preclipped_pos - self.fingertip_midpoint_pos  # Used for action_penalty.
        pos_error_clipped = torch.clip(self.delta_pos, -self.pos_threshold, self.pos_threshold)
        ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_error_clipped

        # (2.b) Clip orientation targets. Use Euler angles. We assume we are near upright, so
        # clipping yaw will effectively cause slow motions. When we clip, we also need to make
        # sure we avoid the joint limit.

        # (2.b.i) Get current and desired Euler angles.
        curr_roll, curr_pitch, curr_yaw = euler_xyz_from_quat(self.fingertip_midpoint_quat)
        desired_roll, desired_pitch, desired_yaw = euler_xyz_from_quat(ctrl_target_fingertip_preclipped_quat)
        desired_xyz = torch.stack([desired_roll, desired_pitch, desired_yaw], dim=1)

        # (2.b.ii) Correct the direction of motion to avoid joint limit.
        # Map yaws between [-125, 235] degrees
        # (so that angles appear on a continuous span uninterrupted by the joint limit)
        curr_yaw = factory_utils.wrap_yaw(curr_yaw)
        desired_yaw = factory_utils.wrap_yaw(desired_yaw)

        # (2.b.iii) Clip motion in the correct direction.
        self.delta_yaw = desired_yaw - curr_yaw  # Used later for action_penalty.
        clipped_yaw = torch.clip(self.delta_yaw, -self.rot_threshold[:, 2], self.rot_threshold[:, 2])
        desired_xyz[:, 2] = curr_yaw + clipped_yaw

        # (2.b.iv) Clip roll and pitch.
        desired_roll = torch.where(desired_roll < 0.0, desired_roll + 2 * torch.pi, desired_roll)
        desired_pitch = torch.where(desired_pitch < 0.0, desired_pitch + 2 * torch.pi, desired_pitch)

        delta_roll = desired_roll - curr_roll
        clipped_roll = torch.clip(delta_roll, -self.rot_threshold[:, 0], self.rot_threshold[:, 0])
        desired_xyz[:, 0] = curr_roll + clipped_roll

        curr_pitch = torch.where(curr_pitch > torch.pi, curr_pitch - 2 * torch.pi, curr_pitch)
        desired_pitch = torch.where(desired_pitch > torch.pi, desired_pitch - 2 * torch.pi, desired_pitch)

        delta_pitch = desired_pitch - curr_pitch
        clipped_pitch = torch.clip(delta_pitch, -self.rot_threshold[:, 1], self.rot_threshold[:, 1])
        desired_xyz[:, 1] = curr_pitch + clipped_pitch

        ctrl_target_fingertip_midpoint_quat = quat_from_euler_xyz(
            roll=desired_xyz[:, 0], pitch=desired_xyz[:, 1], yaw=desired_xyz[:, 2]
        )

        self.generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            ctrl_target_gripper_dof_pos=0.0,
        )

    def _get_dones(self):
        """Label per-env termination reason (§9.6); early termination is opt-in and experimental.

        By default (`use_early_termination=False`) `terminated` is all-false, so
        DirectRLEnv resets only on timeout and every env resets together. Success and
        force_limit are still computed and latched per env, but as *labels* rather than
        as episode-ending events.

        Why it is not simply enabled — the real reason, established by measurement after
        two earlier wrong explanations were recorded and retracted (see
        docs/experiment_log.md): FactoryEnv.randomize_initial_state() is a scripted
        ~1-2 s IK + grasp routine that drives the **shared** PhysX world. Resetting a
        subset of envs therefore disturbs the ones still running:

        - factory_env.py:624/831 — set_gravity(0,0,0) then restore. Global to the whole
          scene, so every running env free-falls for the duration of the reset.
        - factory_env.py:549-551 — write_joint_position_to_sim_index /
          write_joint_velocity_to_sim_index / set_joint_position_target_index are called
          with **no env_ids**, hard-teleporting every robot once per IK iteration.
        - factory_env.py:792-793 — write_root_pose_to_sim_index(root_pose=held_pose) with
          **no env_ids**, teleporting every env's peg into its gripper.
        - step_sim_no_action() runs 4+ times plus inside two loops (~1000+ physics
          substeps) with no policy action applied.

        Indexing fixes alone cannot address the gravity toggle or the shared-sim
        stepping. Corroborating evidence: every IsaacLab task that *does* per-env
        termination (quadcopter, anymal_c, cartpole, franka_cabinet, ...) has a
        _reset_idx built purely from env_ids-indexed buffer writes and
        write_*(..., env_ids=env_ids), and none of them step the simulator inside reset.
        Factory/Forge/AutoMate are the only FactoryEnv subclasses and all three use the
        synchronized timeout-only pattern.

        The flag exists to *measure* that perturbation rather than argue about it. The
        debug/n_* counters below report exactly which conditions fired each step, so a
        reset count no longer has to be inferred from a traceback.

        peg_dropped/workspace_violation/joint_limit/jammed are not implemented — no
        existing signal or documented threshold for them exists on this fork, so they
        fall through to the timeout label rather than guessing at physical bounds.
        """
        self._compute_intermediate_values(dt=self.physics_dt)

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        check_rot = self.cfg_task.name == "nut_thread"
        succeeded = self._get_curr_successes(success_threshold=self.cfg_task.success_threshold, check_rot=check_rot)

        contact_force = torch.linalg.norm(self.force_sensor_smooth[:, 0:3], ord=2, dim=-1, keepdim=False)
        force_limit_exceeded = contact_force >= self.cfg_task.force_limit_threshold

        # Sticky per-episode flags: because resets are synchronized on time_out only
        # (see docstring), a success or force-limit event at step 50 must still be
        # visible in the label when the episode actually ends at the timeout. Without
        # latching, a peg that seated at step 50 and drifted by step 149 would be
        # labelled "timeout", losing the success.
        self.ep_success_latched |= succeeded
        self.ep_force_limit_hit |= force_limit_exceeded

        self.termination_reason = np.full(self.num_envs, "timeout", dtype=object)
        self.termination_reason[self.ep_force_limit_hit.cpu().numpy()] = "force_limit"
        self.termination_reason[self.ep_success_latched.cpu().numpy()] = "success"  # highest priority
        self.extras["termination_reason"] = self.termination_reason.copy()

        # Per-episode accumulators for the evaluation CSV schema (§11.4). Must be
        # updated and read out here, before _reset_idx() overwrites pose/force state
        # for any env whose episode is ending this step.
        # NOTE: all updates here are in-place (torch.maximum(..., out=) rather than
        # rebinding). Rebinding inside torch.inference_mode() — which is how the eval
        # script drives stepping — would replace these with inference tensors, and the
        # in-place reset in _reset_idx() then fails outside that context with
        # "Inplace update to inference tensor outside InferenceMode is not allowed".
        self.ep_step_count += 1
        torch.maximum(self.ep_max_contact_force, contact_force, out=self.ep_max_contact_force)
        self.ep_sum_contact_force += contact_force
        self.ep_force_above_threshold_steps += (contact_force >= self.contact_penalty_thresholds).long()

        held_base_pos, _ = factory_utils.get_held_base_pose(
            self.held_pos, self.held_quat, self.cfg_task.name, self.cfg_task.fixed_asset_cfg, self.num_envs, self.device
        )
        target_held_base_pos, _ = factory_utils.get_target_held_base_pose(
            self.fixed_pos, self.fixed_quat, self.cfg_task.name, self.cfg_task.fixed_asset_cfg, self.num_envs, self.device
        )
        lateral_error = torch.linalg.vector_norm(target_held_base_pos[:, 0:2] - held_base_pos[:, 0:2], dim=1)
        insertion_depth = held_base_pos[:, 2] - target_held_base_pos[:, 2]
        _, _, curr_yaw = euler_xyz_from_quat(self.fingertip_midpoint_quat)
        orientation_error = factory_utils.wrap_yaw(curr_yaw)

        if self.cfg.use_early_termination:
            terminated = succeeded | force_limit_exceeded
        else:
            terminated = torch.zeros_like(time_out)

        # An episode has genuinely ENDED when DirectRLEnv is about to reset it, i.e.
        # terminated | time_out. With early termination off this is time_out alone;
        # anything else would flag envs that keep running, and would keep flagging them
        # every subsequent step, so an evaluation script consuming this would write many
        # duplicate rows for a single episode.
        any_done = terminated | time_out

        # Per-step condition counts, so "why did N envs reset" is answered directly
        # rather than inferred from a traceback. Cheap (5 scalars/step) and the whole
        # point of the use_early_termination experiment.
        self.extras["debug/n_succeeded"] = succeeded.sum()
        self.extras["debug/n_force_limit"] = force_limit_exceeded.sum()
        self.extras["debug/n_time_out"] = time_out.sum()
        self.extras["debug/n_terminated"] = terminated.sum()
        self.extras["debug/n_reset"] = any_done.sum()
        self.extras["eval/episode_steps"] = self.ep_step_count.clone()
        self.extras["eval/max_contact_force"] = self.ep_max_contact_force.clone()
        self.extras["eval/mean_contact_force"] = self.ep_sum_contact_force / self.ep_step_count.clamp(min=1)
        self.extras["eval/force_above_threshold_duration"] = (
            self.ep_force_above_threshold_steps.float() * self.physics_dt
        )
        self.extras["eval/lateral_error_final"] = lateral_error
        self.extras["eval/insertion_depth_final"] = insertion_depth
        self.extras["eval/orientation_error_final"] = orientation_error
        self.extras["eval/initial_xy_offset"] = self.ep_initial_xy_offset.clone()
        self.extras["eval/initial_orientation_error"] = self.ep_initial_orientation_error.clone()
        self.extras["eval/any_done"] = any_done

        return terminated, time_out

    def _get_rewards(self):
        """FORGE reward includes a contact penalty and success prediction error."""
        # Use same base rewards as Factory.
        rew_buf = super()._get_rewards()

        rew_dict, rew_scales = {}, {}
        # Calculate action penalty for the asset-relative action space.
        pos_error = torch.linalg.norm(self.delta_pos, ord=2, dim=-1) / self.cfg.ctrl.pos_action_threshold[0]
        rot_error = torch.abs(self.delta_yaw) / self.cfg.ctrl.rot_action_threshold[0]
        # Contact penalty.
        contact_force = torch.linalg.norm(self.force_sensor_smooth[:, 0:3], ord=2, dim=-1, keepdim=False)
        if self.cfg.use_force_penalty:
            contact_penalty = torch.nn.functional.relu(contact_force - self.contact_penalty_thresholds)
        else:
            # Zero the raw term (not just its scale) so logs_rew_contact_penalty logs flat/zero —
            # proof the ablation removed the signal, not just its weight in rew_buf.
            contact_penalty = torch.zeros_like(contact_force)
        # Add success prediction rewards.
        check_rot = self.cfg_task.name == "nut_thread"
        true_successes = self._get_curr_successes(
            success_threshold=self.cfg_task.success_threshold, check_rot=check_rot
        )
        policy_success_pred = (self.actions[:, 6] + 1) / 2  # rescale from [-1, 1] to [0, 1]
        success_pred_error = (true_successes.float() - policy_success_pred).abs()
        # Delay success prediction penalty until some successes have occurred.
        if true_successes.float().mean() >= self.cfg_task.delay_until_ratio:
            self.success_pred_scale = 1.0

        # Add new FORGE reward terms.
        rew_dict = {
            "action_penalty_asset": pos_error + rot_error,
            "contact_penalty": contact_penalty,
            "success_pred_error": success_pred_error,
        }
        rew_scales = {
            "action_penalty_asset": -self.cfg_task.action_penalty_asset_scale,
            "contact_penalty": -self.cfg_task.contact_penalty_scale,
            "success_pred_error": -self.success_pred_scale,
        }
        for rew_name, rew in rew_dict.items():
            rew_buf += rew_dict[rew_name] * rew_scales[rew_name]

        self._log_forge_metrics(rew_dict, policy_success_pred)
        return rew_buf

    def _reset_idx(self, env_ids):
        """Perform additional randomizations."""
        # Upstream factory_env.py:659 does `self.init_fixed_pos_obs_noise[:] = <(len(env_ids),3)>`.
        # That full-slice assignment only works when len(env_ids) is num_envs (exact) or
        # 1 (broadcasts, silently corrupting all rows) — any other partial reset raises
        # "expanded size (num_envs) must match existing size (len(env_ids))".
        #
        # The buffer must stay num_envs-shaped throughout: randomize_initial_state calls
        # step_sim_no_action(), which runs _compute_intermediate_values() over ALL envs
        # mid-reset and reads this buffer. (Swapping in a smaller temporary was tried and
        # crashed there instead — "size of tensor a (64) must match tensor b (2)".)
        #
        # There is no way to satisfy line 659 from outside for a partial batch: it raises
        # for every len(env_ids) except num_envs and 1. Overriding randomize_initial_state()
        # wholesale would mean forking ~200 lines of upstream IK/grasp logic. So under
        # early termination we upgrade a partial reset to a full one — every env resets
        # together. This preserves the synchronized-reset invariant the upstream routine
        # requires, at the cost of ending some episodes prematurely; the debug/n_* counters
        # in _get_dones() record how often that happens so the cost is measurable.
        if self.cfg.use_early_termination and env_ids is not None and len(env_ids) != self.num_envs:
            self.extras["debug/n_partial_reset_upgraded"] = torch.tensor(len(env_ids), device=self.device)
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=env_ids.dtype)

        super()._reset_idx(env_ids)

        # Reset per-episode evaluation accumulators (see _get_dones()) for the envs
        # actually being reset.
        self.ep_max_contact_force[env_ids] = 0.0
        self.ep_sum_contact_force[env_ids] = 0.0
        self.ep_force_above_threshold_steps[env_ids] = 0
        self.ep_step_count[env_ids] = 0
        self.ep_success_latched[env_ids] = False
        self.ep_force_limit_hit[env_ids] = False

        # Capture the freshly-randomized initial pose (§11.4 initial_xy_offset /
        # initial_orientation_error) — super()._reset_idx() above already called
        # randomize_initial_state(env_ids), so self.held_pos/self.fixed_pos now
        # reflect this new episode's sampled starting pose, not the old one.
        held_base_pos, _ = factory_utils.get_held_base_pose(
            self.held_pos, self.held_quat, self.cfg_task.name, self.cfg_task.fixed_asset_cfg, self.num_envs, self.device
        )
        target_held_base_pos, _ = factory_utils.get_target_held_base_pose(
            self.fixed_pos, self.fixed_quat, self.cfg_task.name, self.cfg_task.fixed_asset_cfg, self.num_envs, self.device
        )
        initial_xy_offset = torch.linalg.vector_norm(
            target_held_base_pos[:, 0:2] - held_base_pos[:, 0:2], dim=1
        )
        _, _, initial_yaw = euler_xyz_from_quat(self.fingertip_midpoint_quat)
        self.ep_initial_xy_offset[env_ids] = initial_xy_offset[env_ids]
        self.ep_initial_orientation_error[env_ids] = factory_utils.wrap_yaw(initial_yaw)[env_ids]

        # Compute initial action for correct EMA computation.
        fixed_pos_action_frame = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        pos_actions = self.fingertip_midpoint_pos - fixed_pos_action_frame
        pos_action_bounds = torch.tensor(self.cfg.ctrl.pos_action_bounds, device=self.device)
        pos_actions = pos_actions @ torch.diag(1.0 / pos_action_bounds)
        self.actions[:, 0:3] = self.prev_actions[:, 0:3] = pos_actions

        # Relative yaw to bolt.
        unrot_180_euler = torch.tensor([-np.pi, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        unrot_quat = quat_from_euler_xyz(
            roll=unrot_180_euler[:, 0], pitch=unrot_180_euler[:, 1], yaw=unrot_180_euler[:, 2]
        )

        fingertip_quat_rel_bolt = quat_mul(unrot_quat, self.fingertip_midpoint_quat)
        fingertip_yaw_bolt = euler_xyz_from_quat(fingertip_quat_rel_bolt)[-1]
        fingertip_yaw_bolt = torch.where(
            fingertip_yaw_bolt > torch.pi / 2, fingertip_yaw_bolt - 2 * torch.pi, fingertip_yaw_bolt
        )
        fingertip_yaw_bolt = torch.where(
            fingertip_yaw_bolt < -torch.pi, fingertip_yaw_bolt + 2 * torch.pi, fingertip_yaw_bolt
        )

        yaw_action = (fingertip_yaw_bolt + np.deg2rad(180.0)) / np.deg2rad(270.0) * 2.0 - 1.0
        self.actions[:, 5] = self.prev_actions[:, 5] = yaw_action
        self.actions[:, 6] = self.prev_actions[:, 6] = -1.0

        # EMA randomization.
        ema_rand = torch.rand((self.num_envs, 1), dtype=torch.float32, device=self.device)
        ema_lower, ema_upper = self.cfg.ctrl.ema_factor_range
        self.ema_factor = ema_lower + ema_rand * (ema_upper - ema_lower)

        # Set initial gains for the episode.
        prop_gains = self.default_gains.clone()
        self.pos_threshold = self.default_pos_threshold.clone()
        self.rot_threshold = self.default_rot_threshold.clone()
        prop_gains = forge_utils.get_random_prop_gains(
            prop_gains, self.cfg.ctrl.task_prop_gains_noise_level, self.num_envs, self.device
        )
        self.pos_threshold = forge_utils.get_random_prop_gains(
            self.pos_threshold, self.cfg.ctrl.pos_threshold_noise_level, self.num_envs, self.device
        )
        self.rot_threshold = forge_utils.get_random_prop_gains(
            self.rot_threshold, self.cfg.ctrl.rot_threshold_noise_level, self.num_envs, self.device
        )
        self.task_prop_gains = prop_gains
        self.task_deriv_gains = factory_utils.get_deriv_gains(prop_gains)

        contact_rand = torch.rand((self.num_envs,), dtype=torch.float32, device=self.device)
        contact_lower, contact_upper = self.cfg.task.contact_penalty_threshold_range
        self.contact_penalty_thresholds = contact_lower + contact_rand * (contact_upper - contact_lower)

        self.dead_zone_thresholds = (
            torch.rand((self.num_envs, 6), dtype=torch.float32, device=self.device) * self.default_dead_zone
        )

        self.force_sensor_world_smooth[:, :] = 0.0

        self.flip_quats = torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)
        rand_flips = torch.rand(self.num_envs) > 0.5
        self.flip_quats[rand_flips] = -1.0

    def _reset_buffers(self, env_ids):
        """Reset additional logging metrics."""
        super()._reset_buffers(env_ids)
        # Reset success pred metrics.
        for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
            self.first_pred_success_tx[thresh][env_ids] = 0

    def _log_forge_metrics(self, rew_dict, policy_success_pred):
        """Log metrics to evaluate success prediction performance."""
        for rew_name, rew in rew_dict.items():
            self.extras[f"logs_rew_{rew_name}"] = rew.mean()

        for thresh, first_success_tx in self.first_pred_success_tx.items():
            curr_predicted_success = policy_success_pred > thresh
            first_success_idxs = torch.logical_and(curr_predicted_success, first_success_tx == 0)

            first_success_tx[:] = torch.where(first_success_idxs, self.episode_length_buf, first_success_tx)

            # Only log at the end.
            if torch.any(self.reset_buf):
                # Log prediction delay.
                delay_ids = torch.logical_and(self.ep_success_times != 0, first_success_tx != 0)
                delay_times = (first_success_tx[delay_ids] - self.ep_success_times[delay_ids]).sum() / delay_ids.sum()
                if delay_ids.sum().item() > 0:
                    self.extras[f"early_term_delay_all/{thresh}"] = delay_times

                correct_delay_ids = torch.logical_and(delay_ids, first_success_tx > self.ep_success_times)
                correct_delay_times = (
                    first_success_tx[correct_delay_ids] - self.ep_success_times[correct_delay_ids]
                ).sum() / correct_delay_ids.sum()
                if correct_delay_ids.sum().item() > 0:
                    self.extras[f"early_term_delay_correct/{thresh}"] = correct_delay_times.item()

                # Log early-term success rate (for all episodes we have "stopped", did we succeed?).
                pred_success_idxs = first_success_tx != 0  # Episodes which we have predicted success.

                true_success_preds = torch.logical_and(
                    self.ep_success_times[pred_success_idxs] > 0,  # Success has actually occurred.
                    self.ep_success_times[pred_success_idxs]
                    < first_success_tx[pred_success_idxs],  # Success occurred before we predicted it.
                )

                num_pred_success = pred_success_idxs.sum().item()
                et_prec = true_success_preds.sum() / num_pred_success
                if num_pred_success > 0:
                    self.extras[f"early_term_precision/{thresh}"] = et_prec

                true_success_idxs = self.ep_success_times > 0
                num_true_success = true_success_idxs.sum().item()
                et_recall = true_success_preds.sum() / num_true_success
                if num_true_success > 0:
                    self.extras[f"early_term_recall/{thresh}"] = et_recall
