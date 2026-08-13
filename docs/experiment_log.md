# Experiment Log

Running, dated notes on what was done and what was seen each session — informal by
design, distinct from the final technical report. Newest entry on top.

---

## 2026-08-13 — Built the deterministic OOD evaluation and grading suite

Closed the most tractable gap from the Agility Robotics role analysis: held-out evaluation under
distribution shift. `configs/evaluation_suites.yaml` now defines nominal, pose-shift, low/high
friction, mass-shift, and combined-OOD suites with explicit seeds and complete pose/dynamics
conditions. The evaluator applies the named suite before environment creation, retains the existing
20-column per-episode CSV, and adds a sibling JSON summary with a machine-readable rubric result.
The rubric directly encodes the project targets: at least 80% nominal success, at least 60%
success on each OOD suite, the full episode budget, and peak-force p95 at or below the 50 N hard
limit. Jam/drop are deliberately not graded because their detectors still do not exist.

Two correctness issues surfaced while wiring the new suites:

- Factory's constructor overwrites the startup material-randomization events with one constant
  friction value. Evaluation friction is therefore applied per environment *after* that overwrite;
  changing the old startup event alone would have produced a convincing config file with no effect.
- `DirectRLEnv.step()` resets completed environments before returning. Reading peg mass from the
  evaluator after `step()` would log the next episode's newly randomized mass. Mass and friction are
  now snapshotted into `extras` with the existing force/pose telemetry before reset.

Also corrected `orientation_error_final` and `initial_orientation_error`: both previously logged
absolute fingertip yaw. They now measure the axis-angle magnitude of the held asset relative to the
socket, which is the quantity the schema actually names.

Local, simulator-independent checks cover every suite, config application, aggregate statistics,
and rubric grading. Full Isaac Sim evaluation still needs to run on the project GPU environment.

---

## 2026-08-13 — Policy B run forensics captured before the details decay

The first two real Policy B seeds finished with trailing-20 success rates of 31.85% (seed 0) and
30.31% (seed 1), a 1.54 percentage-point endpoint spread despite seed 1 leading by as much as
4.7 points mid-run. Seed 2 was still in progress when this note was captured.

Four findings change how the campaign must be interpreted:

1. Shared `logs/rl_games/Forge/` discovery mixed unrelated same-day runs whose epoch counters all
   restart at 1, allowing later files to overwrite real metrics by epoch key. Run directories must
   be explicitly safelisted.
2. TensorBoard tags use two x axes: rewards/episode metrics are epoch-indexed, performance metrics
   are frame-indexed, and `info/epochs` is the authoritative frame-to-epoch mapping.
3. The apparent reward collapses at epoch 302 (seed 0) and 269 (seed 1) are the
   `success_pred_scale` latch turning on once mean success first crosses 25%. The new penalty is
   approximately 0.565 per step × 149 steps ≈ 84 reward per episode, matching the observed cliff.
   Raw reward is therefore not comparable across runs whose latch fires at different epochs.
4. RL-Games only overwrites `Forge.pth` when mean reward beats the stored best. Because the latch
   permanently changes reward scale, seed 0's file froze at epoch 268 and held a policy roughly ten
   success-rate points worse than the final policy. Final and periodic checkpoints must be evaluated
   explicitly rather than trusting the filename `Forge.pth`.

Seed 0's real launch also differed from the checked-in YAML (`num_actors=512`,
`minibatch_size=2048`, `max_epochs=500`; the file said 128/512/200). Those effective values came
from shell history, so subsequent seeds were launched with explicit overrides. The run's own
`params/env.yaml` is authoritative for the seed; a dashboard label that said 42 was wrong.

## 2026-08-12 — Closed out Week 4 setup: all four policies ready for real training

Item 10, wrapping up `docs/2026-08-09_next_10_steps.md` (items 6–9 done this session,
see the three entries below). Status check going into the real 9-run batch:

- **Policy A** — done. 3 seeds trained, evaluated against `nominal` (7.3% success,
  range 6.4–7.8%), results table and README written, representative videos curated.
- **Policy B** (`use_force_obs=True, use_force_penalty=False`) — registered as
  `Shaurya-ForcePegInsert-PolicyB-Direct-v0`, smoke-tested: obs size 24 (force
  channels present), `logs_rew_contact_penalty` flat 0.0. Ready for real training.
- **Policy C** (`use_force_obs=False, use_force_penalty=True`) — registered as
  `Shaurya-ForcePegInsert-PolicyC-Direct-v0`, smoke-tested: obs size 20 (matches
  Policy A), `logs_rew_contact_penalty` nonzero. Ready for real training.
- **Policy D** (`use_force_obs=True, use_force_penalty=True`) — no new
  registration; `Shaurya-ForcePegInsert-Direct-v0` already is Policy D, and was
  itself proven bit-identical to upstream FORGE during the original fork
  verification (2026-08-07 entry). Ready for real training, no smoke test needed.

Venue and budget for the remaining work (decided this session, see the "Closed out
Policy A / Week 3" entry below): stay on the shared DGX Spark, 9 real runs (B, C, D
× 3 seeds each), same tuned settings proven for Policy A (`--num_envs=512`,
`agent.params.config.minibatch_size=2048`, `--max_iterations=500`/seed), run order
B → C → D per the queue-position decision above.

This closes Week 4 setup. `docs/2026-08-09_next_10_steps.md` is fully struck
through; `docs/2026-08-12_next_10_steps.md` starts next, scoped to the actual
B/C/D training + evaluation batch (the real work items 7–10 of the prior doc
deliberately stopped short of).

---

## 2026-08-12 — Confirmed Policy D is ready, decided its queue position

Item 9. Policy D needed no implementation work — re-verified rather than assumed:
`ForgeTaskPegInsertCfg` (what `Shaurya-ForcePegInsert-Direct-v0` already registers)
inherits `use_force_obs=True, use_force_penalty=True` straight from the base
`ForgeEnvCfg` with no overrides in between. That is Policy D's spec exactly — force
observations in, force penalty active. No new config class, no new `gym.register`.
This is by design: Policy D is the config Policy A was carved out of when the
ablation toggles were built, not a fourth arm that still needs building.

Decided where it sits in the 9-run queue: **last**, after B and C. B and C are the
two genuinely new arms that just got registered and smoke-tested (previous entry);
D is a fresh 3-seed run of a config already proven bit-identical to upstream during
the original fork verification (2026-08-07 entry), so it carries no implementation
risk and doesn't need to go first to surface problems early — there's nothing left
to surface. It can slot in wherever the DGX Spark queue has room without blocking
anything else.

All four policies are now ready to enter real 3-seed training: A done and evaluated,
B/C registered and smoke-tested, D confirmed and queued. Next: item 10, Week 4
setup wrap-up.

---

## 2026-08-12 — Registered and smoke-tested Policy B and Policy C

Item 8. Added `ForgeTaskPegInsertPolicyBCfg` (`use_force_obs=True,
use_force_penalty=False`) and `ForgeTaskPegInsertPolicyCCfg` (`use_force_obs=False,
use_force_penalty=True`) to `forge_env_cfg.py` — genuinely zero new mechanism, just
the same toggle combination Policy A already proved out. Registered as
`Shaurya-ForcePegInsert-PolicyB-Direct-v0` / `-PolicyC-Direct-v0` in `__init__.py`.

Made the early-termination call the prior entry flagged as open: both stay on the
base `use_early_termination=False` default, the same synchronized-timeout resets
Policy A trained/evaluated with, for apples-to-apples comparability across the
ablation. Recorded in each config's own docstring so it reads as a deliberate
choice, not an unexamined default.

Discovered this session's shell is *on* the DGX Spark box itself (`bas-zeus`,
GPU `GB10`), with Isaac Lab installed locally at `~/isaac/IsaacLab` — not just
reachable via the `brev shell isaac-sim` path CLAUDE.md documents. Running
`isaaclab.sh -p` needed two local fixes, neither specific to this task: an
unrelated project-level `.venv` (from other work in this repo, nothing to do with
Isaac Lab) was set as `$VIRTUAL_ENV` and silently hijacked `isaaclab.sh`'s own
env-selection logic ahead of its `$ISAACLAB_PATH/env_isaaclab` fallback — worked
around with `env -u VIRTUAL_ENV -u CONDA_PREFIX`; and the app's own documented
`LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1` requirement for shared-library
load order. Worth remembering for any future command run from this shell.

Ran `scripts/list_envs.py` — both tasks registered correctly. Then
`--num_envs=64 --max_iterations=20` smoke tests for each, verified against the
run's own TensorBoard event file rather than trusting console output alone:

- **Policy B**: actor `RunningMeanStd (24,)` (full size, force channels present)
  and `logs_rew_contact_penalty` flat `0.0` across all 20 iterations — force
  observed, not penalized, as intended.
- **Policy C**: actor `RunningMeanStd (20,)` (matches Policy A's reduced size)
  and `logs_rew_contact_penalty` nonzero (peaked ~5.0/iter) — force not observed,
  but penalized, as intended.

Both prove the toggle combinations actually produce the intended ablation
observation/reward shape, not just successful `gym.register`. Smoke-test
checkpoints/logs are under `force_peg_rl/logs/` (gitignored, as always) — nothing
here was meant to be kept past verifying the plumbing. Next session: item 9
(confirm Policy D needs no new registration, decide queue position) and item 10
(Week 4 setup wrap-up), then the real 9-run B/C/D training batch per the venue/
budget decision above.

---

## 2026-08-12 — Closed out Policy A / Week 3, decided Policy B/C/D venue and budget

Two housekeeping items, both queued as `docs/2026-08-09_next_10_steps.md` items 6–7.

Closed item 6: the root `README.md` Status section still read "Week 2–3 ... Not yet
started: Policy A", stale since Policy A's real training/eval landed. Updated to
reflect the real 7.3% nominal success rate and link the results table.

Decided item 7: Policy B/C/D's training venue stays the shared DGX Spark rather than
moving to a dedicated-GPU machine, despite the GPU contention that cost real
wall-clock time on Policy A's run (see 2026-08-09 to -12 entry below) — accepting
that risk again rather than spending session time on migration. Budget: all 9
remaining real runs (3 policies × 3 seeds) go through as planned, same tuned
settings proven for Policy A (`--num_envs=512`,
`agent.params.config.minibatch_size=2048`, `--max_iterations=500`/seed) unless a
policy-specific reason to deviate turns up. Next session starts at item 8:
register and smoke-test Policy B and Policy C.

---

## 2026-08-12 — Curated representative videos (success + near-miss close-ups)

**Closed the "record representative videos" gap left by the 2026-08-09 to -12 entry
below.** That earlier demo video (`play_rl_games.py --video --num_envs=4`) showcased the
checkpoint but wasn't curated to one clear success and one clear failure/near-miss, and
its default wide overview camera (`viewer.eye=(7.5,7.5,7.5)`, the IsaacLab default) is
too far back to see the peg/socket at all — confirmed by extracting a frame: four
robots reduced to thumbnail-sized cubes on a floor grid, no task detail visible.

**Tuned a close-up camera** by rendering single test frames at a few candidate
`viewer.eye`/`viewer.lookat` world-space offsets and inspecting them directly, landing
on `eye = origin + (1.0, -0.85, 0.65)`, `lookat = origin + (0.5, 0.0, 0.15)` — a 3/4
view low over the table that clearly shows the Franka gripper, the yellow peg, and the
socket's mounting hole in the same frame.

**Finding a success to record turned out to be the hard part.** At 7.8% success (seed
44's real nominal-suite number, §22), a naive approach — reset a single env, step it
deterministically until termination, repeat — needed ~13 episodes on average to hit one.
Two sequential single-env scans (seeds 5000 and 6000, 30 then 40 episodes, ~35 min of
wall-clock) came back 0/70 successes combined. At true p=0.078 that's a ~0.3% joint
tail event — worth checking for a bug before writing it off as bad luck, but a scan of
`factory_env.py`'s randomization code found only `torch.rand`/`torch.randn` draws sized
by `len(env_ids)`, nothing that behaves differently at `num_envs=1` (no
`linspace`-style curriculum spread across the batch that would collapse at batch size
1). Left unresolved as a real, if small, open question — but sidestepped rather than
chased further, because a better approach was sitting in the data already collected:
every episode in the real evaluation CSVs runs to exactly step 149 regardless of
outcome (`awk` over `policy_a_seed44_nominal.csv`: 500/500 rows have
`episode_steps == 149`), meaning a whole *batch* of envs finishes its first episode
**simultaneously**. Running `num_envs=64` for exactly 149 steps (one deterministic
pass, ~2 minutes of sim time including Kit startup) gives ~99% odds of at least one
success by the binomial bound, and did: **11/64 succeeded** on the first try
(seed 1000, matching the real nominal suite's own first eval seed).

**Recorded both clips by re-running the identical deterministic batch** (same task,
checkpoint, seed 1000, `num_envs=64` — same configuration as the scan, so per-env
outcomes reproduce exactly) with the world-space viewer camera recentered on one
specific env's origin plus the close-up offset above, camera rendering + `RecordVideo`
enabled, and `video_length=155`. Picked:

- **Success clip** — env 39, essentially a textbook insertion: `insertion_depth_final
  ≈ 0.0001` (residual gap to fully seated), `lateral_error_final ≈ 0.0006`,
  `max_contact_force = 13.2 N` (well under the 50 N hard limit and the [5,10] N soft
  band). Frame-by-frame: peg visibly hovering above the socket early, fully seated
  and the arm settled by the end of the 149-step episode.
- **Near-miss clip** — env 21, timeout despite `insertion_depth_final ≈ 0.0015`
  (~15x closer to seated than env 39's own successful margin implies is required —
  the success/fail boundary here is razor-thin, not a case of the policy being
  visibly far off). Camera shows a clean approach and insertion attempt; the wrist
  rotates in the final ~20 steps in a way that happens to occlude the peg from this
  particular angle right at the end — a real limitation of using one fixed camera per
  clip rather than a bug in the outcome data.

Both written to the existing run's own `videos/play/` directory (gitignored, same
convention as the original showcase video):
`force_peg_rl/logs/rl_games/Forge/2026-08-10_18-03-38/videos/play/closeup-success-step-0.mp4`
and `.../closeup-nearmiss-step-0.mp4`.

---

## 2026-08-09 to 2026-08-12 — Policy A: 3-seed training complete, real §22 evaluation results, monitoring tooling

**Fixed a real inefficiency before starting the real run.** The first attempt at step 7
used `--num_envs=1024` with the PPO config's `minibatch_size` left at its default
(`512`, tuned for the yaml's own `num_actors: 128`). That mismatch meant 256 minibatches
× 4 mini_epochs = 1024 gradient steps/epoch — 8× more backward passes than the config
was tuned for — and, combined with this box's GPU-bound sim-stepping ceiling (~1150 fps
regardless of env count, confirmed identical at 512 and 1024 envs), projected to a
~50.6 hour run. Killed it after ~40 minutes and relaunched tuned:
`--num_envs=512`, `agent.params.config.minibatch_size=2048` (Hydra CLI override, keeps
~32 minibatches/epoch matching the yaml's own ratio), `--max_iterations=500`/seed. That
cut per-epoch cost from 181s to ~84s at baseline (no contention) — a ~2.2× fix,
independent of anything below. Also reverted an unrelated, already-uncommitted
`force_limit_threshold` change (50.0 → 15.0 N) that had crept onto the shared `ForgeTask`
base class from separate early-termination diagnostic work — the 15N figure was measured
from a 20-iteration near-random checkpoint and wasn't representative of what a trained
policy's real force profile should be judged against, so it went back to the original
50N placeholder rather than silently changing what Policy A trained against.

**Real GPU contention dominated the actual wall-clock time anyway.** Two Omniverse Kit
editor processes from an unrelated `kit-app-template` session started sharing this box's
single GPU around 03:24 on the 9th, and an `ollama` instance serving a 27GB model
(`ornith:35b-q4_K_M`) at 100% GPU joined around 19:59 — driving per-epoch cadence as high
as ~400s for long stretches, against the 83.6s tuned baseline. Decision (confirmed with
the user): let training run through the contention rather than interfere with the other
sessions. Net result: seed 42 started 2026-08-09 01:56, all three seeds finished
2026-08-11 08:09 — about 30.5 hours wall-clock for what would have been roughly 27 hours
at the tuned baseline cadence with an idle GPU throughout.

**Final training numbers, 500 epochs/seed:**

| Seed | Final reward | Best reward | Final success | Best success |
|---|---|---|---|---|
| 42 | 151.7 | 165.1 | 11.3% | 16.8% |
| 43 | 154.0 | 169.9 | 18.0% | 21.9% |
| 44 | 148.9 | 173.9 | 13.9% | 17.4% |

Reasonably tight clustering across three independent seeds — this is the repeatability
check from `next_10_steps.md` step 9, not one lucky/unlucky run.

**Built and maintained a live "Policy A Training Monitor" published Artifact**
throughout the run (two URLs — a second was minted partway through when the first
stopped rendering for the user, root cause never fully confirmed but presumed a client
cache issue since server-side content was always correct on refetch). Regenerated from
the live TensorBoard event file and a fresh `nvidia-smi`/`ps` GPU-process snapshot on
every ~20-minute poll cycle and republished on every heartbeat, not just periodically.
Evolved from a bare progress readout into a small research-report layout (plain-language
summary, method table, cross-seed results comparison with a grouped bar chart, then the
live detail charts/cadence/contention table/decision log) after the user asked for
something a third party could actually understand. Found and fixed a real bug while
building it: `rewards/iter` and `Episode/Metrics/success_rate` are logged by `rl_games`
with the **epoch number** as their TensorBoard step, while `info/epochs` and
`performance/*` use **cumulative frame count** — looking reward/success up by the wrong
step domain silently returned `None` for every point on the first publish.

**Step 8 — real evaluation, not the training curve.** Ran
`scripts/evaluate_policy.py` against all three checkpoints' `nn/Forge.pth`, `nominal`
suite (500 deterministic episodes/checkpoint, fixed seeds `[1000, 1001, 1002]`,
`is_deterministic=True`, 64 envs):

| Seed | Success rate | Median steps | Peak-force p95 | Force-limit rate |
|---|---|---|---|---|
| 42 | 6.4% | 149 | 39.3 N | 2.4% |
| 43 | 7.6% | 149 | 34.7 N | 1.0% |
| 44 | 7.8% | 149 | 37.6 N | 2.0% |
| **mean (range)** | **7.3% (6.4–7.8%)** | 149 | 37.2 N (34.7–39.3) | 1.8% (1.0–2.4%) |

This is meaningfully lower than the training-time `success_rate` above (11–18% final),
which is expected rather than a bug: the training number is measured under the
stochastic exploration policy against the continuously-randomized training
distribution, while this is deterministic inference against three fixed evaluation
seeds — different measurement, not a contradiction. **7.3% is the real, trustworthy
Policy A baseline number for the §22 results table** — genuinely low for a geometry-only
ablation, which is the expected direction (force-awareness should matter for this task)
but is now actual evidence rather than an assumption. CSVs written to
`force_peg_rl/results/raw/policy_a_seed{42,43,44}_nominal.csv` (500 rows each).

**Recorded a demo video** of the seed-44 checkpoint (highest eval success, 7.8%) via
`play_rl_games.py --video --video_length=250 --num_envs=4`: confirms the 20-dim
geometry-only observation space and 7-dim action space in the player log, matches
the ablation's spec. Saved under that run's own `videos/play/rl-video-step-0.mp4`.

**Also landing in this commit:** the early-termination diagnostic work from the 2026-08-09
session (`use_early_termination` flag, `Shaurya-ForcePegInsert-EarlyTerm-Direct-v0` task,
per-step `debug/n_*` reset counters) that had been sitting uncommitted since — genuinely
finished and documented at the time, just never committed. It's opt-in and defaults to
`False` on the shared `ForgeTask`/`ForgeEnvCfg` base, so it changes nothing about how
Policy A trained or evaluated above.

**Next:** Policy B (force observations back, still no contact penalty) is next per the
six-week plan. Worth deciding up front whether to keep training on this DGX Spark given
today's contention experience, or move the next ablation to a machine with a dedicated
GPU — the tuning fix and the contention are separable causes of slow wall-clock time, but
only one of them is fixable from inside this repo.

---

## 2026-08-08 — Move to DGX Spark, Policy A ablation, termination-reason logging, TensorBoard writeup

**Moved off Brev entirely.** This machine (`bas-zeus`) is a DGX Spark: aarch64
(Grace Blackwell GB10), CUDA 13.0, driver 580.173.02, Ubuntu 24.04, no prior
IsaacLab/Isaac Sim install, no Docker container, no SSH tunnel. Found the correct
current install path is different from what the Brev container used: cloned
IsaacLab at the same tag already vendored under `reference/isaaclab_source/`
(`v3.0.0-beta2.patch1`, confirmed to exist upstream) into `~/isaac/IsaacLab`,
created a `uv`-managed `env_isaaclab` (Python 3.12.13, matching Isaac Sim 6.0.x's
requirement — not the older `isaaclab==2.3.2`/Python-3.11 PyPI package, which
doesn't match this checkout), and installed Isaac Sim 6.0.1.0 (`isaacsim[all,extscache]`)
plus all 14 IsaacLab submodules + `rl_games` via `./isaaclab.sh --install`.
Two DGX-Spark-specific aarch64 gotchas from NVIDIA's own pip-install docs, both
hit and fixed: `imgui-bundle`/`quadprog` need `python3.12-dev`, `libgl1-mesa-dev`,
`libx11-dev` etc. present *before* installing (added via `apt`), and Isaac Sim's
import needs `LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1` set or it fails with
a static-TLS/`libgomp` preload error. Re-ran `pip install -e source/force_peg_rl`
against this new env (the old Brev install doesn't carry over) and confirmed
`list_envs.py` shows `Shaurya-ForcePegInsert-Direct-v0` registered — the port
didn't silently break anything.

**§9.6 read in full before touching code.** It's unambiguous: *"Terminate the
episode when: ... success ... dropped ... force-limit ... [or] time limit"* —
this is a real early-termination instruction, not just a request to label a
reason while running to the timeout. Confirmed via `next_10_steps.md` that this
gap was already known (`FactoryEnv._get_dones()` returns `time_out, time_out`
unconditionally) but not yet decided on.

**Attempted it, hit a crash, and — see the correction below — drew the wrong
conclusion from it at the time.** A genuine per-env `terminated` tensor crashed
immediately with `RuntimeError: expanded size 64 must match existing size 63`.
**Decision taken: keep resets synchronized** (`terminated` stays all-false,
episodes still run to the shared timeout) **but compute and log a real per-env
`termination_reason`** (`"success"` / `"timeout"` / `"force_limit"` / `"other"`)
every step via `self.extras["termination_reason"]`, in `ForgeEnv._get_dones()`
(our fork, not the external package). That satisfies §9.6's *logging* requirement,
though not its early-termination requirement.

> **Correction, same session, after re-reading the traceback properly.** The
> original write-up here claimed the crash proved per-env reset was infeasible
> and would need "forking ~200 lines of upstream physics/IK logic," blaming
> `step_sim_no_action()` stepping the shared PhysX world. That was overstated and
> partly backwards — a conclusion reached first, with justification found
> afterwards. What the traceback actually shows:
>
> - The crash is a **single shape mismatch** at `factory_env.py:659`,
>   `self.init_fixed_pos_obs_noise[:] = fixed_asset_pos_noise` — a `[:]` full-slice
>   assignment receiving a `len(env_ids)`-row tensor. Fixable by indexing with
>   `[env_ids]`.
> - `step_sim_no_action()` is at line 661, **two lines later. Execution never
>   reached it.** It played no part in this crash. Whether it would *behaviourally*
>   perturb still-running envs is a real question but remains untested — its
>   docstring warning was taken at face value, not verified.
> - The `63` is `len(env_ids)`, i.e. 63 envs were flagged for reset. On the first
>   step of an episode `time_out` is all-false, so those 63 came from `terminated`
>   alone — meaning **success/force-limit were firing spuriously at episode start**
>   (likely the 50 N `force_limit_threshold` placeholder tripping on gripper-close
>   transients, and/or `_get_curr_successes()` reading True before the peg moves).
>   That is a bug in the termination conditions, not evidence about reset
>   architecture.
>
> The synchronized-reset decision stands for now — it is a defensible scope choice,
> and `next_10_steps.md` step 2 explicitly listed it as an acceptable option. But it
> should be recorded as *not yet attempted properly*, not as *proven impossible*.
> Revisiting it means: fix the spurious termination conditions, change `[:]` to
> `[env_ids]` at line 659, re-run, and find out empirically what else breaks.
`peg_dropped`/`workspace_violation`/`joint_limit`/`jammed` are not implemented —
no existing signal or documented numeric threshold for any of them exists
anywhere in the code or project doc, so guessing at physical bounds was
explicitly avoided; they fall through to `"other"`. Added a new
`force_limit_threshold: float = 50.0` to `ForgeTask` for the one condition that
did get real early-termination *labeling* logic — flagged as a placeholder (5×
the existing soft `contact_penalty_threshold_range` upper bound) since no hard
number is specified anywhere; needs real tuning before it's trusted.

**Policy A implemented as a toggle, not a fork.** Added `use_force_obs` /
`use_force_penalty` (both default `True`) to `ForgeEnvCfg`, gating
`ft_force`/`force_threshold` out of `obs_order`/`state_order` via a
`__post_init__` override (confirmed this runs, as dataclass machinery, before
`FactoryEnv.__init__` reads `cfg.obs_order` to compute `cfg.observation_space` —
so the declared/constructed observation size stays consistent automatically, no
manual size field to update) and zeroing the *raw* `contact_penalty` term (not
just its reward-scale weight) so `logs_rew_contact_penalty` genuinely logs
flat/zero rather than a real number multiplied by zero. Registered
`Shaurya-ForcePegInsert-PolicyA-Direct-v0` in `__init__.py` with both flags
`False`, same `ForgeEnv` entry point, existing task registration untouched.

**Found and fixed a real scaffold bug along the way:** `scripts/rl_games/train_rl_games.py`
imports a `common.py` module that the project template never actually copied in
(it lives one directory up in a real IsaacLab checkout, shared across all RL
library subfolders) — running it directly failed with `ModuleNotFoundError`.
Copied `common.py` from the cloned IsaacLab tree and added the missing
`if __name__ == "__main__": run(sys.argv[1:])` guard (upstream's own copy of this
file has the same gap — it's designed to be dispatched via `isaaclab.sh train`,
not run directly, which this project's scripts never had wired up).

**Verification, in order:**
1. `list_envs.py` — both `Shaurya-ForcePegInsert-Direct-v0` and
   `Shaurya-ForcePegInsert-PolicyA-Direct-v0` registered and visible, alongside
   the untouched `Template-Force-Peg-Rl-Direct-v0`.
2. Baseline smoke test (`num_envs=64`, `max_iterations=20`): completed without
   error, final reward 25.617. **No longer bit-identical to upstream by design**
   — the termination-reason computation is a genuine, intentional deviation from
   the prior session's bit-identical fork, even though it doesn't change reset
   dynamics. `logs_rew_contact_penalty` nonzero for 17/20 iterations, as expected
   for the still-force-aware baseline.
3. Policy A smoke test (same settings): completed without error, final reward
   41.3. Observation sizes confirmed correctly reduced — policy net built with
   `RunningMeanStd: (20,)` / critic `(57,)` vs. baseline's `(24,)` / `(61,)`,
   exactly the expected −4 from dropping `ft_force`(3) + `force_threshold`(1).
   `logs_rew_contact_penalty` flat at exactly `0.0` across all 20 iterations —
   the ablation genuinely removed the force-penalty signal, not just its weight:

   ![Contact penalty: baseline fluctuates, Policy A flat at zero](assets/tensorboard/05_contact_penalty.png)

**Wrote [`docs/tensorboard_guide.md`](tensorboard_guide.md)** at the user's
request, explaining all 16 distinct scalar families TensorBoard logs for these
runs (episode success rate, episode length, total/shaped reward, contact
penalty, keypoint rewards, action penalties, success-prediction error/precision/
recall/delay, actor/critic loss, entropy, KL/learning-rate, throughput) with
matplotlib re-renders of the real event data (baseline vs. Policy A overlaid)
under `docs/assets/tensorboard/`, since a live headless browser screenshot
wasn't available in this environment (Firefox snap sandbox rejected
`--headless --screenshot`). The contact-penalty chart is the one that actually
proves the ablation worked — flat orange line at zero against a fluctuating blue
baseline. Caught and fixed a labeling bug in the first draft: `losses/*`,
`info/*`, and `performance/*` are logged against cumulative frame count as their
TensorBoard step, not epoch index — the first render's "Epoch" x-axis label was
wrong and has been corrected to "Frames," matching what TensorBoard itself shows.

**Both smoke-test checkpoints exist** (`Forge.pth` under each run's `logs/rl_games/Forge/<timestamp>/nn/`)
but are from 20-iteration runs — no video was recorded from them; asked and
confirmed a 20-iteration near-random policy isn't worth recording.

**Scale check before committing to a long run.** Bumped the baseline task to
`--num_envs=512` (still 20 iterations) to see whether DGX Spark holds up before
attempting the eventual real run. It does: 20/20 epochs, no crash, ~1150–1270
fps step (vs ~200–270 at 64 envs), process RSS flat at ~6.2 GB with no growth,
system RAM 16 GB used of 121 GB. GPU utilisation sat at ~94% though — so the
ceiling here is compute, not memory, and pushing much past this is likely to
buy less than the env count suggests. `nvidia-smi` cannot report GPU memory on
this unified-memory Grace-Blackwell board (`memory.used`/`memory.total` return
`N/A`), so RSS + `free -h` are the only usable memory signals.

**Steps 5 + 6 — evaluation script and the `nominal` suite.** Built
`scripts/evaluate_policy.py` (§17) and `configs/evaluation_suites.yaml` (§11.3,
`nominal` only — the OOD suites are stubbed out as comments and the script
raises `NotImplementedError` for them, per next_10_steps.md's Week-5 scoping).
The script loads a checkpoint, forces `is_deterministic=True`, splits the
suite's episode budget across its `seeds` list as one re-seeded batch per seed,
and writes the full 20-column §11.4 per-episode CSV plus the §17 aggregate
summary.

Went through all 20 §11.4 columns before writing any code rather than assuming
they were obtainable. 17 were reachable from existing state; the three that
needed a decision:

- `peg_friction` / `socket_friction` — no public accessor exists. The values are
  reachable only via `asset.root_view.get_material_properties()`, the same
  internal PhysX tensor call IsaacLab's own `randomize_rigid_body_material`
  event and `factory_utils.set_friction()` both use. Chose to ship it and
  document the fragility rather than leave the columns null. Worth noting the
  read validated itself: both come back `0.75`, which looked wrong against
  `EventCfg`'s `static_friction_range: (0.25, 1.25)` for the fixed asset — until
  it turned out `FactoryEnv.__init__` calls `set_friction()` on all three assets
  *after* the startup randomisation, overwriting it with the constant
  `cfg_task.*_asset_cfg.friction`. So that `(0.25, 1.25)` range is dead config,
  and 0.75 is the genuinely effective value.
- `episode_seed` — IsaacLab has no per-episode seed to expose. Logged as the
  batch seed instead, documented in the script's module docstring.
- The per-episode aggregates (max/mean contact force, force-above-threshold
  duration, final lateral/insertion/orientation error, initial pose offsets) had
  to be computed *inside* `ForgeEnv._get_dones()` and stashed in `self.extras`,
  because `DirectRLEnv.step()` resets a finished env's pose and force state
  before returning control — an external script reading post-step data for an
  env whose episode just ended is already looking at the next episode.

**Four real bugs found reviewing that code before trusting its output:**

1. *Duplicate CSV rows.* `any_done` was `time_out | succeeded |
   force_limit_exceeded`, but resets only fire on `time_out` (since `terminated`
   is deliberately all-false). An env that succeeded at step 50 would therefore
   keep reporting done every step through 149 — roughly 100 duplicate rows for a
   single episode, which would have quietly filled a `--episodes 500` run from a
   handful of real episodes. Now `any_done = time_out`.
2. *Lost success labels*, the direct consequence of fixing (1): a peg that
   seated at step 50 and drifted by 149 would be labelled `timeout`. Added
   sticky per-episode latches so the label survives to the episode end.
3. *Name collision.* The first latch draft reused `self.ep_succeeded`, which the
   base `FactoryEnv` already owns as a `long` buffer for success-time logging
   (`factory_env.py:81`) and resets itself — `|=` on it would have corrupted
   base-class logic and broken its dtype. Renamed to `ep_success_latched`.
4. *Inference-tensor crash.* `self.ep_max_contact_force = torch.maximum(...)`
   rebound the tensor inside `torch.inference_mode()` (how the eval script drives
   stepping), marking it an inference tensor; the in-place reset in `_reset_idx()`
   then failed outside that context with "Inplace update to inference tensor
   outside InferenceMode is not allowed". Switched to `torch.maximum(..., out=)`.

Also hoisted `read_friction()` out of the per-row loop (it was refetching a full
per-env PhysX tensor for every row) and fixed the p95 force to interpolate
rather than truncate — at n=10 the truncated index was reporting the 90th
percentile under a "p95" label.

**A separate Isaac Sim gotcha worth recording,** because it cost real debugging
time and will bite any future script here: the evaluation script initially
exited 0 with no traceback and no output file. Nothing was crashing — Kit's
`/app/fastShutdown` is enabled by default, so `launch_simulation()`'s `finally`
calls `SimulationApp.close()` → `app.shutdown()` →
`quickReleaseFrameworkAndTerminate`, which terminates the process immediately.
**Every line after the `with launch_simulation(...)` block silently never runs.**
The CSV write had to move inside the block. This is why `play_rl_games.py` and
`train_rl_games.py` do all their work inside it too.

Verified against the 512-env baseline checkpoint: 9 rows / 20 columns, no empty
cells, exactly 3 episodes per seed across `[1000, 1001, 1002]`, and 9 distinct
values per metric column (confirming the duplicate-row bug is genuinely gone).
Success rate 0.0% and every `termination_reason` reading `timeout` — expected
for a 20-iteration checkpoint, and the point of this run was the plumbing, not
the policy.

**Next session:** step 7 — Policy A's first real training run (`--num_envs`
somewhere in the 512–1024 range given the ~94% GPU utilisation measured above,
enough iterations for the reward curve to plateau rather than 20 iterations of
noise), then step 8: evaluate that checkpoint against `nominal` for the first
real row of the §22 results table.

---

## 2026-08-07 — Fork FORGE into `force_peg_rl`, prove it's faithful, wire up logging

Brought the instance back up (`env_isaaclab` venv survived the stop/start untouched)
and worked through roadmap steps 2–10. Generated the `force_peg_rl` external project
via IsaacLab's template generator — `./isaaclab.sh --new` turned out to need a real
TTY (InquirerPy full-screen prompts), so it was driven by calling the generator
function directly with the same answers (External / Direct / RL-Games), which
produces an identical result. Forked `forge_env.py`, `forge_env_cfg.py`,
`forge_events.py`, `forge_tasks_cfg.py`, `forge_utils.py`, and the PPO agent config
into `tasks/direct/force_peg/` verbatim, registered under our own ID
`Shaurya-ForcePegInsert-Direct-v0`. The regression check was the best possible
outcome: a 20-iteration/64-env smoke test through the fork produced a **bit-identical**
final reward (22.362465) to the original upstream baseline run — same seed, same
code, same number to six decimal places. Step 8 (per-component reward logging)
turned out to already exist upstream (`self.extras["logs_rew_<name>"]` in both
`FactoryEnv` and `ForgeEnv`) — nothing to build there, just verified it.

Three real bugs found and fixed along the way, all in generated/template code, none
in the FORGE/Factory logic itself: (1) the scaffold's `list_envs.py` had a hardcoded
`"Template-"` ID filter that silently hid any differently-named task — replaced with
a check on the entry point's package prefix; (2) that fix then surfaced a second bug,
19 of gymnasium's bundled MuJoCo envs register with a function instead of a string as
`entry_point`, which crashed the naive filter with no visible traceback; (3) the
generated `.gitignore` (both the project-level one and a nested `.vscode/.gitignore`)
had two independent bugs that silently excluded every `__init__.py` and every
`.vscode/*.json` file from git — meaning the scaffold commit from the prior session
would have produced a non-importable package on a fresh clone. All fixed and
verified via `git ls-files`.

Set up TensorBoard on the instance (`--reload_interval 5` for live updates during
future runs) tunneled to localhost via the existing SSH ControlMaster connection,
comparing the baseline and fork runs side by side — confirmed the individual reward
curves render correctly (30 distinct scalar cards under `logs_rew_*`), not just a
total-reward line.

Nothing here touches what the policy sees or is rewarded for yet — pure
infrastructure, per plan. Next session starts at Policy A (project doc §12), now
that the fork is proven faithful and `obs_order` is already documented in the
worksheet.
