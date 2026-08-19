# Gap analysis: Agility Robotics — Robotics Software Engineer III, Manipulation

**Date:** 2026-08-13
**Assessed against:** this repository only (`Policy-RL-Peg_insertion`). If there is hardware,
teleoperation, or classical-controls experience elsewhere in the CV, discount the gaps below
accordingly — this document does not know about it.

---

## 1. The role, condensed

Agility's Skills team, hybrid (Fremont CA / Salem OR / Pittsburgh PA). Own a manipulation task
vertical from definition through reliable on-robot execution. The listed responsibilities:

| # | Responsibility |
|---|---|
| R1 | Own a task vertical from definition → reliable on-robot execution |
| R2 | Work across **learned and classical** manipulation methods |
| R3 | Run the LfD flywheel: data collection → annotation → training → evaluation → deployment |
| R4 | Work with teleoperators to improve data quality and operator instructions |
| R5 | Build grading rubrics / evaluation workflows; success = reliability on **real** robots |
| R6 | Root-cause failures using logs, replays, state-distribution analysis |
| R7 | Integrate learned policies with higher-level skills to execute complete workflows |

The load-bearing sentence is in R5: *"success gets measured by policies that act more reliably on
real robots in the real world."*

---

## 2. What this project already demonstrates

### 2.1 Failure forensics (R6) — the strongest evidence

This is the responsibility the repo answers best, and it is rarer than it should be. Four
instrumentation or interpretation bugs were caught during the Policy B runs, each of which would
have produced a wrong conclusion if left alone:

**Cross-run metric contamination.** `logs/rl_games/Forge/` stores every run of every policy under
one shared experiment name, and epoch numbers restart at 1 for each. A same-day prefix match pulled
in unrelated smoke tests whose epoch keys silently overwrote real data, fabricating an "epoch 499,
reward 173.9" that did not exist. Caught before publishing. Fixed with an explicit, manually
verified run-directory safelist (`extract_policyb_metrics.py`).

**Mixed x-axis conventions in TensorBoard.** `rewards/iter` and `Episode/Metrics/*` are
epoch-indexed; `performance/*` is frame-indexed. Treating them alike mislabelled the throughput
chart's axis with frame counts; over-correcting then silently zeroed the success-rate series.
`info/epochs` carries the authoritative frame→epoch mapping, and only the frame-indexed tags need it.

**The reward cliff at epoch 302 (seed 0) / 269 (seed 1).** Total reward fell ~87 points in a single
epoch while success rate rose. It was *not* a regression. `forge_env.py:398` latches
`success_pred_scale` from 0.0 → 1.0 the first time mean success crosses
`delay_until_ratio = 0.25`, switching on the `success_pred_error` penalty at ~0.565/step × 149 steps
≈ 84 reward/episode. The arithmetic matches the observed step almost exactly. Two earlier
explanations of mine (blaming `action_penalty_ee`) were wrong because they compared single noisy
epochs; the 20-epoch windowed means show the action penalties are flat.

**Frozen "best" checkpoint.** rl-games only overwrites `Forge.pth` when mean reward beats the stored
best (`a2c_common.py:1405`). Because the reward scale drops permanently at the latch, `Forge.pth`
froze at seed 0's epoch 268 and could never update again — leaving the file labelled "best" holding a
policy ~10 points of success rate *worse* than the final one.

### 2.2 Experimental hygiene (partial R5)

- Controlled ablation design (Policies A–D) isolating force observations from force penalty.
- Fork proven bit-identical to upstream FORGE before modification.
- Three seeds per policy, matched configuration, launched from a checked-in script rather than
  by hand (`run_policyb_seeds.sh`).
- Independent verification that the ablation is wired correctly: `contact_penalty` reads exactly
  0.000 at every epoch of every Policy B run — the signal was removed, not merely down-weighted.

### 2.3 Contact-rich manipulation domain knowledge

Force/torque observations, contact-force penalties, impedance-controlled insertion, success
thresholds on lateral error and insertion depth. The problem domain is the right one.

### 2.4 Reproducibility recovered from a near-miss

Seed 0's actual configuration did **not** match `rl_games_ppo_cfg.yaml` on three counts:

| parameter | YAML says | actually used | consequence if trusted |
|---|---|---|---|
| `num_actors` | 128 | 512 | 4× wrong batch accounting |
| `minibatch_size` | 512 | 2048 | different optimisation |
| `max_epochs` | 200 | 500 | silently no-ops any resume past epoch 200 |

The real values existed only in shell history. Seeds 1 and 2 are launched from a script that passes
all three explicitly, with the reasoning recorded in comments. (A separate bug: the dashboard
reported "seed 42" for many hours; the run's own `params/env.yaml` says `seed: 0`.)

---

## 3. Gaps, ordered by how much they matter

### G1 — No physical robot *(blocks R1, R5)*

Everything here is Isaac Sim. No sim-to-real transfer, no on-robot execution, no deployment data, no
hardware iteration loop. The role's stated success criterion is reliability on real robots. **This is
the gap most likely to filter an application**, and the hardest to close.

### G2 — No learning from demonstration *(blocks R3, R4)*

The role is built around a demonstration flywheel — teleop collection, operator instruction design,
data-quality iteration, annotation. This project is RL-from-scratch against a hand-designed reward,
which is arguably the *opposite* paradigm. There is no behaviour cloning, no imitation learning, and
no experience shaping what human demonstrators produce.

### G3 — No classical manipulation *(blocks R2)*

"Across learned **and** classical" is explicit. There is no motion planning, grasp planning,
admittance/compliance control design, or search primitive in this repo. FORGE supplies a task-space
impedance controller; it was configured, not built.

### G4 — No evaluation suite *(blocks R5)* — **most fixable**

Already specified in `force_aware_peg_insertion_project.md` §11, still unbuilt. Today "success" means
success rate on the *training* distribution. The role wants grading rubrics and held-out evaluation.
This is a missing artifact, not a missing skill.

### G5 — No skill composition *(blocks R7)*

One primitive, no sequencing into a workflow (approach → align → insert → verify → retract), no
behaviour tree or state machine, no failure-recovery branch.

### G6 — No perception *(weakens R1)*

Observations are privileged simulator state: `fingertip_pos_rel_fixed`, `fingertip_quat`, `ee_linvel`,
`ee_angvel`, `ft_force`, plus privileged hole poses in the critic. Real deployment does not hand you
those.

### G7 — Single task, single object *(weakens R1)*

"Own a task vertical" implies breadth across a family of related tasks and object variation. Peg
insertion with one peg is a data point, not a vertical.

---

## 4. Closing plan, ranked by leverage per unit effort

**1. Build the §11 evaluation suite.** *(closes G4, partially addresses G1)*
Deterministic held-out episode seeds; OOD sweeps over friction, mass, initial XY offset, orientation
error; a written pass/fail rubric with thresholds; per-episode logging of max contact force and
insertion depth. Already scoped in the spec. Gives you a robustness-under-distribution-shift story
that is the closest sim-only proxy for deployment reliability.

**2. Add a classical baseline to the A–D table.** *(closes G3)*
Spiral search or compliant insertion with a hand-tuned admittance law. Cheap, and it directly answers
"learned and classical" with a number rather than a claim.

**3. Add a behaviour-cloning baseline from scripted demonstrations in sim.** *(partially closes G2)*
Does not require teleop hardware. Generate demonstrations from a scripted expert, train BC, compare
against PPO. Gets you honest exposure to the demo → policy loop and something concrete to say about
data quality and coverage.

**4. Compose the multi-step workflow.** *(closes G5)*
Approach → align → insert → verify → retract as a state machine over the learned insertion primitive,
with a recovery branch on failed verification. Roughly a day's work.

**5. Get access to any real arm.** *(closes G1)*
Used UR5, xArm, or a low-cost 6-DoF with an F/T sensor. Expensive and slow — and the one change that
moves the application from "simulation researcher" to "manipulation engineer."

Items 1–4 are all reachable with the current setup and no new hardware.

---

## 5. Framing advice

Do not present this project as *"I trained a peg-insertion policy."* Policies are common; the
underlying FORGE task is public.

Present it as: *"I ran a controlled force-ablation study across three seeds and caught four
instrumentation bugs that would each have produced a wrong conclusion — including a reward-scale
discontinuity that made the training curve look like a collapse, and a checkpoint-selection bug that
would have shipped a policy 10 points worse than the one I had."*

The first framing is a tutorial outcome. The second is R6, which is what the role is actually asking
for, and it is backed by specific file-and-line evidence.

Two supporting facts worth having ready:

- **Cross-seed stability.** Seeds 0 and 1 finished at 31.85% and 30.31% trailing-20 success — a
  1.54pp spread — despite seed 1 leading by up to 4.7pp mid-run. Different learning trajectories,
  near-identical endpoints. Seed 2 in progress.
- **Reward is not comparable across runs.** The `success_pred_scale` latch fires at whatever epoch a
  given run first crosses 25% success (seed 0: epoch 302; seed 1: epoch 269). Any A–D comparison must
  therefore key off success rate and contact force, never raw `rewards/iter`. Knowing *why* your
  headline metric is untrustworthy is a better interview answer than the metric itself.

---

## 6. Related documents

- [`force_aware_peg_insertion_project.md`](../force_aware_peg_insertion_project.md) — §11 evaluation
  suite (unbuilt, item 1 above), §22 results table template
- [`experiment_log.md`](experiment_log.md) — **does not yet contain** the 2026-08-12/13 Policy B runs:
  the host reboot at epoch 220, the discarded 128-env resume, the three-seed campaign, the latch
  discovery, or the checkpoint-freeze finding. Worth writing up before the details decay.
- [`next_10_steps.md`](next_10_steps.md) — current planning
