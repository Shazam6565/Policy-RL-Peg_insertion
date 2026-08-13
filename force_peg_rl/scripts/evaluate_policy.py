# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal deterministic evaluation script (next_10_steps.md step 5, project §17).

Loads a checkpoint, disables stochasticity, runs a fixed number of episodes with a
named evaluation suite's seed(s), and writes one CSV row per completed episode
matching the §11.4 schema:

    run_id, checkpoint, training_seed, evaluation_suite, episode_seed, success,
    termination_reason, episode_steps, episode_return, final_insertion_depth,
    max_contact_force, mean_contact_force, force_above_threshold_duration,
    lateral_error_final, orientation_error_final, peg_mass, peg_friction,
    socket_friction, initial_xy_offset, initial_orientation_error

Known, documented gaps against the literal §11.4 spec (see docs/experiment_log.md
for the fuller discussion):

- ``episode_seed`` is the seed of the *batch* an episode belongs to, not a true
  independently-tracked per-episode seed — IsaacLab does not expose the latter.
  The suite's ``seeds`` list is honoured by splitting the episode budget into one
  re-seeded batch per seed, so re-running reproduces each batch deterministically,
  but not one arbitrary episode within a batch.
- ``peg_friction``/``socket_friction`` are read via
  ``asset.root_view.get_material_properties()`` — the same internal PhysX tensor
  accessor IsaacLab's own ``randomize_rigid_body_material`` event uses to write
  these values, not a documented public API. This is confirmed working against
  the IsaacLab version pinned in this repo (v3.0.0-beta2.patch1) but has no
  stability guarantee across IsaacLab releases.
- Fields marked "eval/*" in ForgeEnv.extras (episode_steps, max/mean_contact_force,
  force_above_threshold_duration, lateral/insertion/orientation error,
  initial_xy_offset, initial_orientation_error) are computed inside
  ForgeEnv._get_dones() itself, not recomputed here — DirectRLEnv.step() resets
  a terminated env's pose/force state before returning control to the caller, so
  by the time this script sees post-step data for an env whose episode just
  ended, that data already belongs to the *next* episode.

Example:

    python scripts/evaluate_policy.py \\
        --task Shaurya-ForcePegInsert-PolicyA-Direct-v0 \\
        --checkpoint logs/rl_games/Forge/<run>/nn/Forge.pth \\
        --suite combined_ood \\
        --output results/raw/policy_a_seed_0_combined_ood.csv

``episodes`` and ``seeds`` come from configs/evaluation_suites.yaml for the named
suite; ``--episodes`` / ``--seeds`` override them for quick smoke runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import torch
import yaml

from evaluation_config import apply_suite_to_env_cfg, load_named_yaml, validate_suite
from evaluation_results import grade_summary, summarize_rows

# -- argparse (kept minimal; simulator/launcher args added below) -------------
parser = argparse.ArgumentParser(description="Deterministic policy evaluation (§17).")
parser.add_argument("--task", type=str, required=True, help="Name of the registered task to evaluate.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the RL-Games checkpoint (.pth) to load.")
parser.add_argument(
    "--suite",
    type=str,
    default="nominal",
    help="Evaluation suite name from configs/evaluation_suites.yaml (§11.3).",
)
parser.add_argument(
    "--episodes",
    type=int,
    default=None,
    help="Override the suite's `episodes` count. Defaults to whatever the suite defines.",
)
parser.add_argument("--output", type=str, required=True, help="Path to write the per-episode CSV to.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel envs to evaluate with.")
parser.add_argument(
    "--seeds",
    type=int,
    nargs="+",
    default=None,
    help="Override the suite's `seeds` list. Defaults to whatever the suite defines.",
)
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
from isaaclab_tasks.utils import add_launcher_args  # noqa: E402

add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

from isaaclab_tasks.utils import launch_simulation, resolve_task_config  # noqa: E402

import force_peg_rl.tasks  # noqa: F401,E402

EVAL_SUITES_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "evaluation_suites.yaml")
EVAL_RUBRIC_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "evaluation_rubric.yaml")


def load_suite(suite_name: str) -> dict:
    """Load a named suite from configs/evaluation_suites.yaml (§11.3)."""
    suite = load_named_yaml(EVAL_SUITES_PATH, suite_name, kind="evaluation suite")
    return validate_suite(suite_name, suite)


def read_friction(asset) -> torch.Tensor:
    """Read back per-env static friction via the internal PhysX material accessor.

    Same private-by-convention method IsaacLab's own randomize_rigid_body_material
    event uses internally to re-sample (isaaclab/envs/mdp/events.py). Returns the
    mean static friction across all shapes on the asset, per env — the material
    buffer is (num_envs, num_shapes, 3) = [static_friction, dynamic_friction,
    restitution]; we only need static_friction, index 0 of the last dim.
    """
    import warp as wp

    materials = wp.to_torch(asset.root_view.get_material_properties())
    return materials[:, :, 0].mean(dim=1)


def main():
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent)

    with launch_simulation(env_cfg, args_cli):
        from rl_games.common.player import BasePlayer
        from rl_games.torch_runner import Runner

        from isaaclab.utils.assets import retrieve_file_path

        from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
        from rl_games.common import env_configurations, vecenv

        import gymnasium as gym
        import math

        suite = load_suite(args_cli.suite)

        # The suite file is authoritative (§11.3); CLI flags are explicit overrides.
        total_episodes = args_cli.episodes if args_cli.episodes is not None else suite["episodes"]
        seeds = args_cli.seeds if args_cli.seeds is not None else suite["seeds"]

        # Split the episode budget across the suite's seeds, giving the remainder to
        # the first batches so the totals add up exactly to `total_episodes`.
        base, extra = divmod(total_episodes, len(seeds))
        episodes_per_seed = [base + (1 if k < extra else 0) for k in range(len(seeds))]

        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = seeds[0]
        apply_suite_to_env_cfg(env_cfg, suite)

        resume_path = retrieve_file_path(args_cli.checkpoint)
        run_dir = os.path.dirname(os.path.dirname(resume_path))  # .../nn/x.pth -> run dir

        # Recover the seed the checkpoint was actually trained with, from the run's
        # own dumped agent config (§11.4 training_seed) — distinct from --seed above,
        # which is this *evaluation* run's env seed.
        training_seed = None
        agent_yaml_path = os.path.join(run_dir, "params", "agent.yaml")
        if os.path.isfile(agent_yaml_path):
            with open(agent_yaml_path) as f:
                training_seed = yaml.safe_load(f).get("params", {}).get("seed")

        rl_device = agent_cfg["params"]["config"]["device"]
        clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
        clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
        obs_groups = agent_cfg["params"]["env"].get("obs_groups")
        concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

        vecenv.register(
            "IsaacRlgWrapper",
            lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
        )
        env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
        print(f"[INFO] Loading checkpoint: {resume_path}")

        runner = Runner()
        runner.load(agent_cfg)
        agent: BasePlayer = runner.create_player()
        agent.restore(resume_path)
        agent.reset()
        agent.is_deterministic = True  # disable training-time stochasticity (§17 step 2)

        unwrapped = env.unwrapped
        num_envs = unwrapped.num_envs
        run_id = os.path.basename(run_dir)

        held_asset = getattr(unwrapped, "_held_asset", None)
        fixed_asset = getattr(unwrapped, "_fixed_asset", None)

        rows = []

        print(f"Evaluation suite: {args_cli.suite}")
        print(f"Episodes: {total_episodes} across seeds {seeds} ({episodes_per_seed})")

        for seed, seed_budget in zip(seeds, episodes_per_seed):
            if seed_budget == 0:
                continue
            # Re-seed pose/mass resets and re-sample startup-style friction so each
            # batch is an independent deterministic block rather than all three
            # suite seeds sharing the first batch's per-env friction assignment.
            env.seed(seed)
            if hasattr(unwrapped, "resample_evaluation_friction"):
                unwrapped.resample_evaluation_friction()
            obs = env.reset()
            if isinstance(obs, dict):
                obs = obs["obs"]
            _ = agent.get_batch_size(obs, 1)
            if agent.is_rnn:
                agent.init_rnn()
            # Allocated fresh per batch: accumulating inside torch.inference_mode()
            # marks the tensor as an inference tensor, which then cannot be mutated
            # (e.g. .zero_()) outside that context.
            episode_return = torch.zeros(num_envs, device=rl_device)

            seed_rows = 0
            print(f"[INFO] seed {seed}: collecting {seed_budget} episodes")

            while seed_rows < seed_budget:
                with torch.inference_mode():
                    obs_t = agent.obs_to_torch(obs)
                    actions = agent.get_action(obs_t, is_deterministic=True)
                    obs, rew, dones, extras = env.step(actions)

                    episode_return += rew.to(rl_device)

                    any_done = extras.get("eval/any_done")
                    if any_done is None:
                        # Fallback: our _get_dones() always returns terminated=False, so
                        # `dones` from the wrapper is really just time_out here — still
                        # correct as a per-episode-boundary signal even without the
                        # eval/* extras (e.g. baseline task predating this script).
                        any_done = dones.bool()

                    done_ids = any_done.nonzero(as_tuple=False).squeeze(-1).tolist()
                    if done_ids:
                        # Prefer the environment's pre-reset snapshot. DirectRLEnv
                        # resets done envs before env.step() returns, so a live mass
                        # read here can belong to the next episode.
                        peg_mass_all = extras.get("eval/peg_mass")
                        peg_friction_all = extras.get("eval/peg_friction")
                        socket_friction_all = extras.get("eval/socket_friction")
                        # Compatibility fallback for baseline tasks/checkpoints that
                        # predate the eval/* physical-parameter snapshots.
                        if peg_mass_all is None and held_asset:
                            peg_mass_all = held_asset.data.body_mass.torch.sum(dim=-1)
                        if peg_friction_all is None and held_asset:
                            peg_friction_all = read_friction(held_asset)
                        if socket_friction_all is None and fixed_asset:
                            socket_friction_all = read_friction(fixed_asset)
                    for i in done_ids:
                        if seed_rows >= seed_budget:
                            break
                        row = {
                            "run_id": run_id,
                            "checkpoint": resume_path,
                            "training_seed": training_seed,
                            "evaluation_suite": args_cli.suite,
                            # The seed of this batch — deterministic at batch level, not
                            # per individual episode; see module docstring caveat.
                            "episode_seed": seed,
                            "success": bool(extras["termination_reason"][i] == "success"),
                            "termination_reason": str(extras["termination_reason"][i]),
                            "episode_steps": int(extras["eval/episode_steps"][i].item()),
                            "episode_return": float(episode_return[i].item()),
                            "final_insertion_depth": float(extras["eval/insertion_depth_final"][i].item()),
                            "max_contact_force": float(extras["eval/max_contact_force"][i].item()),
                            "mean_contact_force": float(extras["eval/mean_contact_force"][i].item()),
                            "force_above_threshold_duration": float(
                                extras["eval/force_above_threshold_duration"][i].item()
                            ),
                            "lateral_error_final": float(extras["eval/lateral_error_final"][i].item()),
                            "orientation_error_final": float(extras["eval/orientation_error_final"][i].item()),
                            "peg_mass": float(peg_mass_all[i].item()) if peg_mass_all is not None else None,
                            "peg_friction": (
                                float(peg_friction_all[i].item()) if peg_friction_all is not None else None
                            ),
                            "socket_friction": (
                                float(socket_friction_all[i].item()) if socket_friction_all is not None else None
                            ),
                            "initial_xy_offset": float(extras["eval/initial_xy_offset"][i].item()),
                            "initial_orientation_error": float(extras["eval/initial_orientation_error"][i].item()),
                        }
                        rows.append(row)
                        seed_rows += 1
                        episode_return[i] = 0.0

        # NOTE: everything below MUST stay inside the `with launch_simulation(...)`
        # block. Its __exit__ calls SimulationApp.close() -> app.shutdown(), and with
        # Kit's /app/fastShutdown enabled (the default) that calls
        # quickReleaseFrameworkAndTerminate, which terminates the process immediately.
        # Any code placed after the `with` block silently never runs — the script exits
        # 0 with no traceback and no output file.
        _write_results(rows, suite)

        env.close()


def _write_results(rows: list[dict], suite: dict):
    """Write the per-episode CSV, JSON summary, and rubric decision."""
    if not rows:
        print("[WARN] No episodes completed — nothing to write.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.output)), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(args_cli.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize_rows(rows)
    rubric = load_named_yaml(EVAL_RUBRIC_PATH, args_cli.suite, kind="evaluation rubric")
    grade = grade_summary(summary, rubric)
    summary_document = {
        "evaluation_suite": args_cli.suite,
        "conditions": suite["conditions"],
        "summary": summary,
        "rubric": grade,
        "known_metric_gaps": [
            "jam and drop rates are not graded because their detectors are not implemented",
            "force_limit_label_rate_pct excludes successful episodes that also crossed the force limit because success has label priority",
        ],
    }
    summary_path = os.path.splitext(args_cli.output)[0] + ".summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_document, f, indent=2)
        f.write("\n")

    print(f"Evaluation suite: {args_cli.suite}")
    print(f"Episodes: {summary['episodes']}")
    print(f"Success rate: {summary['success_rate_pct']:.1f}%")
    print(f"Median completion steps: {summary['median_completion_steps']:g}")
    print(f"Peak-force p95: {summary['peak_force_p95_n']:.1f} N")
    print(f"Force-limit label rate: {summary['force_limit_label_rate_pct']:.1f}%")
    print(f"Rubric: {'PASS' if grade['passed'] else 'FAIL'}")
    print(f"Wrote: {args_cli.output}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
