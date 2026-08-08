# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project at a glance

Training a PPO policy for a Franka robot to insert a peg into a socket while minimizing contact force, using NVIDIA Isaac Lab and RL-Games. The project forks the upstream FORGE `Isaac-Forge-PegInsert-Direct-v0` task into a local external IsaacLab project (`force_peg_rl/`) to run ablation studies (Policies A–D) comparing force-aware vs geometry-only policies.

**Authoritative spec:** [`force_aware_peg_insertion_project.md`](force_aware_peg_insertion_project.md) (~1400 lines). Read this first for scope, reward design, experimental plan, and metrics.
**Current status:** Early stage — infrastructure complete (fork proven bit-identical to upstream). Policy A (geometry-only ablation) is next. See [`docs/next_10_steps.md`](docs/next_10_steps.md) and [`docs/experiment_log.md`](docs/experiment_log.md).

## Repository structure (portions most relevant to daily work)

```
force_peg_rl/source/force_peg_rl/force_peg_rl/tasks/direct/
├── force_peg/                     ← REAL RESEARCH CODE. Fork of upstream FORGE.
│   ├── __init__.py                → registers Shaurya-ForcePegInsert-Direct-v0
│   ├── forge_env.py               → core env class (391 lines): obs, actions, rewards, randomization
│   ├── forge_env_cfg.py           → config: gains, noise levels, episode length, obs_order, task variants
│   ├── forge_events.py            → domain randomization events (dead_zone)
│   ├── forge_tasks_cfg.py         → PegInsert / GearMesh / NutThread constants
│   ├── forge_utils.py             → frame conversion, random gain helpers
│   └── agents/rl_games_ppo_cfg.yaml  ← PPO hyperparams (LSTM actor-critic, lr=1e-4)
├── force_peg_rl/                  ← auto-generated template baseline (bit-identical upstream)
reference/isaaclab_source/         ← READ-ONLY upstream IsaacLab FORGE source for reference
```

**Vector specs** (for context when modifying `force_peg/`):
- **Actor obs (24 dims):** `fingertip_pos_rel_fixed`(3) + `fingertip_quat`(4,W=0) + `ee_linvel`(3) + `ee_angvel`(3) + `ft_force`(3) + `force_threshold`(1) + `prev_actions`(6)
- **Critic state (61 dims):** privileged info — joint/hole poses, task gains, thresholds
- **Actions (7 dims):** `[Δx, Δy, Δz, Δroll, Δpitch, Δyaw, success_prediction]`

## Key architecture: the step loop

```
train.py → gym.make() → ForgeEnv(FactoryEnv(DirectRLEnv)) → RL-Games PPO
```

Each `step()` call executes (in order): `_pre_physics_step(action)` → `_apply_action()` × decimation → physics tick → `_get_dones()` → `_get_rewards()` → `_reset_idx()` → `_get_observations()` → return to RL-Games. See `reference/isaaclab_source/README.md` for the full walkthrough.

**Important:** The upstream FORGE baseline is force-aware by default. Policy A (geometry-only) requires *removing* `ft_force` and `force_threshold` from `obs_order` and zeroing the contact penalty — the reverse of what "step 6: add force observations" in §15 implies if taken literally.

## Commands (run on the remote Brev instance via `brev shell isaac-sim`)

Install (editable mode):
```bash
source <isaac_lab_path>/isaaclab.sh -p && pip install -e source/force_peg_rl
python scripts/list_envs.py   # verify task registered
```

Train:
```bash
# Smoke test (fast, for verifying new code doesn't break the pipeline)
python scripts/rl_games/train_rl_games.py --task=Shaurya-ForcePegInsert-Direct-v0 --headless --num_envs=64 --max_iterations=20

# Real training run
python scripts/rl_games/train_rl_games.py --task=Shaurya-ForcePegInsert-Direct-v0 --headless --num_envs=1024  # drop to 512/256 if OOM
```

Playback / evaluation:
```bash
python scripts/rl_games/play_rl_games.py --task=Shaurya-ForcePegInsert-Direct-v0 --checkpoint=<path>

# Sanity-check agents
python scripts/zero_agent.py --task=Shaurya-ForcePegInsert-Direct-v0
python scripts/random_agent.py --task=Shaurya-ForcePegInsert-Direct-v0
```

Code quality:
```bash
pre-commit run --all-files   # ruff + codespell + license headers
```

## Where to look first (by task type)

| When I need to... | Read... |
|---|---|
| Understand the research goal, experiment design, or any policy spec | `force_aware_peg_insertion_project.md` |
| Modify the task environment (obs, actions, rewards, randomization) | `force_peg_rl/source/.../tasks/direct/force_peg/forge_env.py` + `forge_env_cfg.py` |
| Change PPO hyperparameters | `force_peg_rl/source/.../tasks/direct/force_peg/agents/rl_games_ppo_cfg.yaml` |
| Understand IsaacLab internals (the machinery we wrap) | `reference/isaaclab_source/README.md` + source files in that directory |
| Check what's been done / what's next | `docs/experiment_log.md` → `docs/next_10_steps.md` |
| Build the evaluation script or interpret results | `force_aware_peg_insertion_project.md` §11 (evaluation suite) + §22 (results table template) |

## Important constraints & conventions

- **No local tests:** The project has no test suite. Smoke testing uses `--max_iterations 20 --num_envs 64`; regression is proven by bit-identical final reward under the same seed.
- **Remote execution only:** Training runs on a Brev GPU instance, not locally. All commands above assume `brev shell isaac-sim` first. The Isaac Lab path is known on that instance.
- **No test directories or CI/CD.** Pre-commit hooks (ruff + codespell) are the only automated checks.
- **Policy A is the starting point** for all ablation work — geometry-only, removing force observations from the default FORGE config. Do not start from a "blank slate"; the fork already includes everything except what you need to remove.
- **`logs/` is gitignored.** Checkpoints and TensorBoard runs live on the remote instance, not in this repo.
- All code retains original BSD/Apache-2.0 copyright headers — never strip them.
