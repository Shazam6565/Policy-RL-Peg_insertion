# Project Glossary — Interview Prep Reference

Purpose: one document you can review end-to-end to be able to answer any question about this
project's concepts, without having to reconstruct them from the code under pressure. Every number
here is pulled directly from the actual config/code (not the generic project-doc template) as of
2026-08-12. For the *derivations* behind the PPO math and the Isaac Lab layer model, this document
points to [`rl-primer.md`](rl-primer.md) and [`ppo_forge_training_notes.md`](ppo_forge_training_notes.md)
rather than repeating them — read those two first if a term below feels too compressed, then use
this as the fast-lookup layer on interview day.

---

## 1. The 30-second pitch

> Train a Franka arm (in NVIDIA Isaac Lab simulation) to insert a peg into a socket, using PPO
> reinforcement learning. The research question: **does giving the policy contact-force
> information, and penalizing excessive force, make it safer and more robust — without hurting its
> success rate?** We test this with a controlled 2×2 ablation (force observation on/off × force
> penalty on/off = Policies A/B/C/D), training each with 3 random seeds and evaluating all of them
> on identical deterministic episode seeds so the comparison is fair.

Everything below unpacks a piece of that sentence.

---

## 2. Reinforcement learning fundamentals (quick reference)

| Term | Meaning here |
|---|---|
| **Agent / policy (π)** | The neural network. Observation in, action out. This *is* the thing being trained and eventually evaluated. |
| **Environment** | Isaac Sim + the `ForgeEnv` wrapper — the simulated Franka, peg, socket, and physics. |
| **Observation / state** | What the policy is given each step (24 numbers for the force-aware config; see §5). |
| **Action** | What the policy outputs each step (7 numbers; see §5). |
| **Reward** | A single number per step scoring how good that step was (§6). |
| **Episode** | One attempt, start (peg above socket, randomized pose) to end (here: always ends at timeout — see §8). ~150 steps / ~10 simulated seconds. |
| **Return** | Total reward summed (discounted by γ) over an episode — not the same as a single step's reward. |
| **On-policy** | PPO only ever learns from data the *current* (or very-recently-current) policy generated — never a replay buffer of old/other data. This is why the network must keep re-collecting fresh rollouts rather than reusing old ones indefinitely. |
| **Domain randomization** | Deliberately varying physical/sensor parameters every episode (mass, friction, pose, gains, noise) so the policy generalizes instead of memorizing one exact simulated setup. Full inventory in §9. |

---

## 3. PPO — the algorithm, in this project's actual numbers

PPO (Proximal Policy Optimization) trains two networks together:

- **Actor** — outputs the action distribution from the observation. This is the deployed policy.
- **Critic** — estimates how good a state is (a value estimate), used only to compute a training
  signal (*advantage*); discarded at deployment. Here it's an **asymmetric actor-critic**: the
  critic gets a privileged 61-number "state" (exact joint/hole poses, task gains — things a real
  sensor can't give you), while the actor only ever sees the noisy 24-number "policy" observation
  it will actually have access to at test/deployment time.

The **"proximal"** part: after each batch of experience, PPO improves the actor toward
higher-reward actions but **clips** how far any single update can move it, so one lucky/unlucky
batch can't wreck an already-working policy.

| Concept | This project's value (from `rl_games_ppo_cfg.yaml`) | Plain meaning |
|---|---|---|
| `horizon_length` (rollout) | 128 steps/env | How much experience is collected per env before pausing to learn. Independent of episode length (~150 steps) — PPO doesn't need full episodes, thanks to GAE bootstrapping. |
| `num_actors` | 128 | Parallel simulated envs collecting experience simultaneously (real training runs override this via `--num_envs`, e.g. 512). |
| Batch size | `num_envs × horizon_length` | Total experience records collected before one learning round. |
| **Mini-batch** | `minibatch_size: 512` (tuned to `2048` for the real 512-env runs) | The buffer is too big for one gradient step, so it's chopped into equal-sized mini-batches; one gradient update happens per mini-batch. `num_minibatches = batch_size / minibatch_size`. |
| `mini_epochs` | 4 | How many times the *entire* collected buffer is reused for gradient updates before being thrown away and re-collected. Total gradient steps per training epoch = `num_minibatches × mini_epochs`. |
| **Epoch** (rl_games sense) | one full collect→learn cycle | *Not* "one pass over a fixed dataset" like supervised ML — here it means one full "gather a rollout, then learn from it" round. `max_epochs: 200` in the yaml (real runs use `--max_iterations=500`). |
| `gamma` (γ, discount factor) | 0.995 | How much future reward matters vs. immediate reward. Close to 1 → the agent cares almost as much about reward 50 steps away as reward right now. |
| `tau` / GAE `lambda` (λ) | 0.95 | How far into the future the advantage estimate trusts (0 = only next step, 1 = whole rest of episode). A stability/accuracy dial. |
| `e_clip` (PPO's ε) | 0.2 | Max allowed change in action-probability ratio per update — the literal "proximal" mechanism, PPO's namesake. |
| `learning_rate` | 1e-4, `adaptive` schedule (adjusts based on KL divergence) | Step size for gradient descent. |
| `kl_threshold` | 0.008 | Target KL divergence the adaptive LR schedule tries to stay near — grows/shrinks LR to keep policy updates from moving too fast or too slow. |
| `entropy_coef` | 0.0 (off) | No explicit exploration bonus; exploration instead comes from the Gaussian action distribution's own learned spread (`sigma`). |
| `critic_coef` | 2 | Relative weight of the critic's value-loss term vs. the actor's clipped policy loss in the combined loss function. |
| `grad_norm` | 1.0 | Gradients are clipped to this norm before the update — caps how big any single gradient step can be. |
| Network | LSTM (1024 units × 2 layers) → MLP `[512, 128, 64]`, ELU activation | *Not* a plain MLP — recurrence gives the policy short-term memory across steps, useful for a contact task where the recent trend in force readings matters, not just the instantaneous value. |

**Advantage (A)** answers "was this action, considering everything that happened afterward, better
or worse than the critic expected?" — computed via GAE (Generalized Advantage Estimation), blending
TD-errors (`δ_t = r_t + γV(s_{t+1}) − V(s_t)`) across future steps weighted by `(γλ)^k`. Positive
advantage → reinforce that action; negative → discourage it. Full derivation with worked numbers:
[`ppo_forge_training_notes.md` §9](ppo_forge_training_notes.md).

---

## 4. Isaac Lab / Isaac Sim architecture

```
Isaac Sim (physics + rendering, PhysX)
  → Isaac Lab (wraps it into an RL environment: obs/action/reward/reset API)
    → RL-Games (the PPO implementation actually training the network)
      → force_peg_rl / ForgeEnv (our task-specific code — the only layer we author)
```

- **Why GPU-parallel matters**: PhysX runs *thousands of cloned physics scenes* as one batched GPU
  operation (`--num_envs`), the same trick a neural network forward pass uses for matrix
  multiplication. This is what makes RL-for-robotics tractable — 3.3M+ environment steps in hours,
  not the months it would take one robot at a time.
- **Direct workflow**: this project uses Isaac Lab's "Direct" RL environment API — you author
  `_setup_scene`, `_pre_physics_step`, `_apply_action`, `_get_observations`, `_get_rewards`,
  `_get_dones`, `_reset_idx` directly on a `DirectRLEnv` subclass, rather than the alternative
  "Manager-based" config-driven API.
- **Factory** (NVIDIA benchmark suite) — contact-rich assembly tasks (peg insertion, gear meshing,
  nut threading) chosen because they demand sub-millimeter precision *and* physical contact
  reasoning, historically hard for RL.
- **FORGE** — extends Factory with force/torque sensing, a randomized "dead zone" (simulating
  unreliable low-force actuation), a contact-force penalty, and a self-predicted success confidence
  output. The stated purpose is closing the **sim-to-real gap**: a pure-geometry policy tends not
  to transfer to real hardware because the final millimeters of insertion are dominated by *feel*
  (force), not position.
- **This project's class hierarchy**: `ForgeEnv(FactoryEnv(DirectRLEnv))` — our `ForgeEnv` is a
  fork of upstream FORGE's env (not a from-scratch build), registered under our own task IDs
  (`Shaurya-ForcePegInsert-*`). Verified bit-identical to upstream on first fork (same seed → same
  reward to 6 decimal places) before any research changes were made.

---

## 5. The MDP for this task — observation, action, control loop

**Control rate**: physics steps at `dt = 1/120 s`; `decimation = 8` (8 physics ticks per policy
decision) → the policy makes a decision **every 8/120 s ≈ 15 times per second**. Episode length
10 s → **~150 policy steps per episode**.

### Observation vector — actor/"policy" observation (force-aware config, 24 numbers)

Order matters — this is the literal `obs_order` in `forge_env_cfg.py`:

| Entry | Dims | What it is |
|---|---|---|
| `fingertip_pos_rel_fixed` | 3 | Fingertip position **relative to the socket** (not raw joint angles / absolute position) — so the same learned skill works regardless of where the socket randomly spawns each episode |
| `fingertip_quat` | 4 | Fingertip orientation (quaternion, W component zeroed by convention here) |
| `ee_linvel` | 3 | End-effector linear velocity (finite-differenced from noisy position) |
| `ee_angvel` | 3 | End-effector angular velocity (finite-differenced from noisy orientation) |
| `ft_force` | 3 | **Force/torque sensor reading** — noisy, smoothed contact force at the end effector (§7). Present only when `use_force_obs=True`. |
| `force_threshold` | 1 | This *episode's* randomly-sampled soft force threshold — see §7. Present only when `use_force_obs=True`. |
| `prev_actions` | 7 (appended, not in `obs_order`) | The action taken last step. Gives the policy short-horizon context beyond what the LSTM alone provides. |

Removing `ft_force` + `force_threshold` (4 dims) is exactly what Policy A/C (geometry-only obs) do
— actor observation drops from 24 → 20.

**Everything is expressed relative to the target**, and real sensor noise is deliberately injected
into fingertip position/rotation and force before the policy ever sees it (`_compute_intermediate_values`)
— the task is not solvable by reading perfect simulator ground truth, on purpose, since a real
robot never gets that.

### Critic / "state" observation (61 numbers, privileged)

Superset of the actor's observation plus simulator ground truth the actor never sees: absolute
`joint_pos`, `held_pos`/`held_quat` (peg pose), `fixed_pos`/`fixed_quat` (socket pose),
`task_prop_gains`, `ema_factor`, `pos_threshold`/`rot_threshold`. This is the **asymmetric
actor-critic** pattern: since we're in simulation and know the true state, let the critic cheat for
a cleaner value estimate, while the actor stays honest about what it could plausibly sense on real
hardware.

### Action vector (7 numbers)

| Dims | Meaning |
|---|---|
| `Δx, Δy, Δz` | Position target delta, relative to a fixed reference frame anchored at the socket ("bolt frame"), not the robot's current pose — scaled by `pos_action_bounds = [0.05, 0.05, 0.05]` m, then clipped per-step to `pos_action_threshold` (randomized per episode, nominal `[0.02, 0.02, 0.02]` m) so the controller target never jumps further than that from the current pose in one tick. |
| `Δroll, Δpitch` | Forced to zero in `_apply_action` — not actually controllable in this task variant. |
| `Δyaw` | The one real rotational DOF, mapped into the valid joint-limit range and similarly clipped per step. |
| 7th number | **Not a physical action at all.** Rescaled from `[-1, 1]` to `[0, 1]` and interpreted as the policy's **self-reported confidence that it has just succeeded** — read only by the reward function's `success_pred_error` term (§6), never sent to the controller. |

The clipping exists to keep controller targets physically reachable in one control tick — prevents
the policy from commanding a jump large enough to destabilize the low-level position controller or
slam the peg into the socket.

Full walkthrough of `_apply_action`'s frame math: `force_peg_rl/.../forge_env.py:169-249`.

---

## 6. Reward function — every term, exactly as computed

Total reward is a **plain weighted sum**, no interactions between terms:

```
R_t = Σ_i  w_i · term_i
```

**7 base terms (from `FactoryEnv._get_factory_rew_dict`)**:

| Term | Formula | Weight (PegInsert) | What it rewards/penalizes |
|---|---|---|---|
| `kp_baseline` | `squashing_fn(keypoint_dist, a=5, b=4)` | +1.0 | Wide, forgiving "closeness" bell — gives gradient signal even far from the socket |
| `kp_coarse` | `squashing_fn(keypoint_dist, a=50, b=2)` | +1.0 | Medium-width bell — rewards coarse alignment |
| `kp_fine` | `squashing_fn(keypoint_dist, a=100, b=0)` | +1.0 | Narrow, picky bell — only rewards near-perfect final alignment |
| `action_penalty_ee` | `‖action‖₂` | `-action_penalty_ee_scale` (0.0 in Forge — see below) | Raw action magnitude — discourages large commands |
| `action_grad_penalty` | `‖action_t − action_{t-1}‖₂` | `-action_grad_penalty_scale` (0.1) | Action-rate penalty — discourages jerky, oscillating control |
| `curr_engaged` | boolean: within `engage_threshold` (0.9, a looser success bound) | +1.0 | "Loose" bonus for being roughly engaged with the socket, before full success |
| `curr_success` | boolean: within `success_threshold` (0.04 for peg insert) | +1.0 | The tight success bonus |

`squashing_fn(x, a, b) = 1 / (e^{a·x} + b + e^{-a·x})` — peaks at `x=0` (perfectly aligned keypoints)
and decays smoothly; `keypoint_dist` is the mean distance between 4 corresponding keypoints
sampled on the held peg and the fixed socket (not a single point-to-point distance — keypoints also
implicitly encode *orientation* alignment, since a rotated peg moves its keypoints apart even at
the same center position).

Why three bells at different widths (`a=5` → `a=100`) instead of one: a single narrow reward gives
zero gradient from far away (nothing to learn from initially); a single wide reward can't drive
final millimeter precision. Stacking three gives a usable signal across the entire approach.

**3 FORGE-specific terms added on top (from `ForgeEnv._get_rewards`)**:

| Term | Formula | Weight (PegInsert) | Notes |
|---|---|---|---|
| `action_penalty_asset` | `‖Δpos‖/pos_action_threshold + \|Δyaw\|/rot_action_threshold` | `-action_penalty_asset_scale` (0.001) | Action penalty in the *asset-relative* action frame (distinct from `action_penalty_ee` above, which is in the raw `[-1,1]` action space) |
| `contact_penalty` | `relu(force − this_episode's_threshold)` | `-contact_penalty_scale` (**0.2** for peg insert) | The actual force/safety term — see §7 |
| `success_pred_error` | `\|true_success − predicted_confidence\|` | `-success_pred_scale` (0 until `true_successes.mean() ≥ delay_until_ratio` (0.25), then 1.0) | Penalizes the 7th action dim's self-reported confidence being wrong; deliberately **not** active from step 1 — delayed until the policy has actually started succeeding sometimes, so it isn't trained to guess "never" from scratch |

**Reward-development discipline** (both project doc §9.4 and what this codebase actually does):
every term is logged separately to TensorBoard as `logs_rew_<name>` — never debug only the total
reward. This is how the Policy A ablation was *proven* to have actually removed the force signal
(not just its weight): `logs_rew_contact_penalty` reads flat exactly `0.0` for Policy A/C, vs.
fluctuating for the force-aware configs.

---

## 7. FT force / force threshold / contact penalty — the core research variable

This is the crux of the whole project's research question, so worth being precise about the
mechanics, not just the name.

- **`ft_force`** — Force/Torque sensor reading. Physically, it's the incoming joint force at a
  virtual `force_sensor` body on the robot (`get_link_incoming_joint_force()`), i.e. what a wrist
  force/torque sensor would read. It is **smoothed** with an exponential moving average
  (`ft_smoothing_factor = 0.25`) before use, and **Gaussian noise** is added on top before the
  policy sees it (`obs_rand.ft_force = 1.0` scale) — simulating a real, imperfect sensor rather than
  ground truth.
- **`force_threshold` (a.k.a. `contact_penalty_threshold`)** — a **soft**, per-episode randomized
  threshold, resampled every reset from `contact_penalty_threshold_range = [5.0, 10.0]` N. Below
  this, contact is free; above it, every Newton of excess force costs reward. Randomizing it per
  episode (rather than a fixed constant) is itself a form of domain randomization — and is also fed
  to the policy as an observation, so a force-aware policy can in principle learn "how much force is
  currently acceptable," not just "force is bad."
- **`contact_penalty`** — the actual reward term: `relu(force_magnitude − force_threshold)`, i.e.
  `max(0, excess)`, scaled by `contact_penalty_scale` (**0.2** for peg insert — Forge's own
  override of the base `ForgeTask`'s default 0.05). Zero penalty for any force under the current
  episode's threshold; grows linearly above it. This asymmetry — no penalty for *necessary* contact,
  only for *excess* — is deliberate: a policy penalized for all contact would refuse to touch the
  socket at all (see "over-cautious behavior" in the failure taxonomy).
- **`force_limit_threshold`** — a separate, **hard** limit (currently `50.0` N, explicitly flagged
  in the code as an unvalidated placeholder — 5× the soft band's upper bound — pending a real
  trained-policy force profile to calibrate against). Intended to *end* an episode outright if
  force stays above it; currently only used to compute the `force_limit` termination **label** (see
  §8) because early termination itself is off by default.
- **`use_force_obs` vs. `use_force_penalty`** — the two independent toggles that define the entire
  ablation (§10). `use_force_obs` controls whether `ft_force`/`force_threshold` are in the
  observation vector at all. `use_force_penalty` controls whether `contact_penalty`'s *raw value* is
  computed at all (zeroed outright when off, not just multiplied by a zero weight — so the log
  itself proves the ablation, per the reward-development discipline in §6).

---

## 8. Termination, success, and a genuine gap worth knowing about

**`_get_curr_successes`** (checked every step, base `FactoryEnv` logic): success requires **both**
(a) lateral (xy) distance between peg and socket centers `< 0.0025` m, **and** (b) vertical
displacement below `height_threshold = fixed_asset.height × success_threshold` (0.04 for peg
insert — i.e., the peg must be seated within 4% of the socket's height of fully inserted).
`engage_threshold` (0.9) is a looser version of the same check used for the `curr_engaged` reward
bonus (§6) — a "getting close" signal distinct from true success.

**The real gap** (documented in `docs/forge_task_worksheet.md` and confirmed by reading
`_get_dones()` directly): **the upstream base implementation only ever checks timeout.**
`_get_dones()` in `FactoryEnv` literally returns `time_out, time_out` — success, dropping the peg,
force limits, and workspace violations are all computed and *logged*, but none of them end an
episode early. Every episode in the *default* config runs the full ~150 steps regardless of
outcome. This fork adds a genuine per-env `termination_reason` label (`success` / `force_limit` /
`timeout`, priority in that order) computed every step, plus an **opt-in, experimental**
`use_early_termination` flag (default `False`) that would actually end episodes on success/force-limit
— left off by default because `FactoryEnv`'s reset routine drives the *shared* PhysX world (a
scene-wide gravity toggle, un-indexed sim writes across all envs) during a scripted ~1-2s IK/grasp
sequence, so a *partial* per-env reset risks perturbing envs that are still mid-episode; every other
`FactoryEnv` subclass in upstream IsaacLab uses this same synchronized-timeout-only pattern, which
is corroborating evidence this isn't a bug being worked around but a real constraint of the base
class.

Because resets are synchronized on timeout only, success and force-limit events are **sticky/latched**
per episode (`ep_success_latched`, `ep_force_limit_hit`) — a peg that seats at step 50 but drifts by
step 149 must still be labeled `success`, not `timeout`, when the label is finally read out.

`peg_dropped`, `workspace_violation`, `joint_limit`, `jammed` (all named in the project doc's
generic template, §9.6) are **not implemented** — no existing signal or documented numeric threshold
for any of them exists in this codebase, so they fall through to `timeout` rather than guessing at
physical bounds. Worth knowing this precisely if asked "does the environment detect jamming?" — the
honest answer is "not directly; it's inferred post-hoc during failure analysis from high sustained
force + low insertion progress," not a first-class termination condition.

---

## 9. Domain randomization — the "synthetic data generation" layer

There's no separate offline dataset generation step in this project — RL is **online**: every
episode reset *is* a fresh synthetic sample of the task's parameter distribution, and training data
(`obs, action, reward, next_obs, done`) is generated live by rolling out the current policy, not
pre-collected. "Domain randomization" is the mechanism that makes each of those rollouts diverse
rather than always the same fixed scenario. Two kinds happen in this codebase:

**At reset** (`_reset_idx`, once per episode):
- Peg mass — `object_scale_mass` event, uniform `±0.005` kg added to the base mass
- Peg/socket initial pose — position and orientation offset (upstream `randomize_initial_state`)
- Controller proportional gains — `task_prop_gains`, resampled per episode via
  `get_random_prop_gains` at `task_prop_gains_noise_level = 0.41` (±41%) around the nominal
  `[565, 565, 565, 28, 28, 28]`
- Action clipping thresholds — `pos_threshold`/`rot_threshold`, similarly resampled (±25%/±29%)
- EMA smoothing factor for the low-level controller — uniform in `[0.025, 0.1]`
- The soft `force_threshold` itself — uniform in `[5, 10]` N (§7)
- Fingertip quaternion sign flip (`flip_quats`) — a 50/50 coin flip per episode, likely covering a
  quaternion double-cover ambiguity (q and −q represent the same rotation) so the policy doesn't
  overfit to one sign convention

**On a repeating in-episode timer** (`EventCfg`, `mode="interval"`, every 2.0s):
- `dead_zone_thresholds` — simulates an unreliable low-force actuation dead zone, re-randomized
  *during* the episode rather than fixed at reset, because a real controller's dead-zone behavior
  can drift over the course of a real interaction, not just differ episode-to-episode.

**At startup only** (`mode="startup"`, sampled once when the simulation first launches, not
per-episode): physics materials — friction/restitution for the held asset, fixed asset (socket),
and robot. Worth knowing as an interview nuance: the fixed asset's `static_friction_range = (0.25,
1.25)` in `EventCfg` is actually **dead config** in the current codebase — `FactoryEnv.__init__`
calls `set_friction()` on all three assets *after* this startup randomization, overwriting it with
a constant `0.75` from `cfg_task.*_asset_cfg.friction`. A real finding from building the evaluation
script, not a design choice — worth mentioning if asked about friction randomization specifically,
since the honest answer is "configured but currently overridden to a constant."

**Sensor-level noise** (every step, not a reset-time event): Gaussian noise on fingertip
position/rotation and force readings (§5, §7) — this is observation-level randomization, distinct
from the physical-parameter randomization above, simulating imperfect sensing rather than a
different physical world.

---

## 10. The ablation study — Policies A–D

The core experiment. Two independent binary toggles (`use_force_obs`, `use_force_penalty` on
`ForgeEnvCfg`) produce a clean 2×2 design — everything else (network architecture, PPO
hyperparameters, task physics, randomization) held identical across all four:

| Policy | Force obs? | Force penalty? | Registered task ID | Research purpose | Status (2026-08-12) |
|---|---|---|---|---|---|
| **A** — Geometry-only | No | No | `Shaurya-ForcePegInsert-PolicyA-Direct-v0` | Baseline: can the policy solve this from geometry alone? | **Done** — 3 seeds trained, evaluated: 7.3% nominal success (range 6.4–7.8%) |
| **B** — Force observation only | Yes | No | `Shaurya-ForcePegInsert-PolicyB-Direct-v0` | Does *seeing* force help even with no penalty for using it? | Registered, smoke-tested (obs size 24 confirmed), ready for real training |
| **C** — Force penalty only | No | Yes | `Shaurya-ForcePegInsert-PolicyC-Direct-v0` | Can safer behavior emerge from being penalized for force *without* being able to see it? | Registered, smoke-tested (obs size 20, `logs_rew_contact_penalty` nonzero confirmed), ready |
| **D** — Force-aware (full FORGE) | Yes | Yes | `Shaurya-ForcePegInsert-Direct-v0` (the original/default task — not a separate config) | Tests the complete hypothesis: obs + penalty together | Config proven bit-identical to upstream FORGE at fork time; ready for real 3-seed training |

**Why Policy D needed no new code**: it *is* `ForgeEnvCfg`'s unmodified defaults (both toggles
default `True`) — Policy A/B/C were carved out of it by flipping toggles off, not the other way
around. Useful framing if asked "which policy did you build first": the fork started at D (full
FORGE) and A/B/C were derived ablations, even though A was the first one *trained*.

**Minimum experimental standard being followed**: 3 random seeds per policy, all evaluated on
identical deterministic episode seeds (`[1000, 1001, 1002]` for the `nominal` suite), same tuned
training settings (`--num_envs=512`, `minibatch_size=2048`, `--max_iterations=500`/seed) — so any
success-rate or force difference between policies reflects the ablation, not incidental
run-to-run variance.

**Policy A's real result so far — 7.3% nominal success** is a genuinely low number for what should
be an easier ablation arm (geometry alone, no penalty getting in the way), which is the direction
the research question predicts (force-awareness should matter), but it's now a measured baseline
rather than an assumption — the comparison against B/C/D is what will actually test the hypothesis.

---

## 11. Evaluation — metrics and how they're produced

Deterministic evaluation is kept **separate from training-time randomization**: the `nominal` suite
(`configs/evaluation_suites.yaml`) fixes `is_deterministic=True` and reuses the same 3 seeds across
every checkpoint, so policies are compared on identical conditions, not lucky/unlucky training
noise.

| Metric (§11.4 schema) | Where it comes from |
|---|---|
| `success` / `termination_reason` | The sticky-latched label from `_get_dones()` (§8) |
| `episode_steps` | Always 149 under the current synchronized-timeout regime (§8) — a real, expected consequence of early termination being off, not a bug |
| `max_contact_force`, `mean_contact_force` | Accumulated per-episode inside `_get_dones()` itself (`torch.maximum`/running sum), **not** read back after `step()` returns — because `DirectRLEnv.step()` already resets a just-finished env's pose/force state by the time external code sees it |
| `force_above_threshold_duration` | Count of steps where `contact_force ≥ this_episode's_soft_threshold`, converted to seconds via `physics_dt` |
| `lateral_error_final`, `insertion_depth_final`, `orientation_error_final` | Geometric error at the moment of episode end |
| `peg_friction`, `socket_friction` | Read via the same internal PhysX tensor call IsaacLab's own randomization event uses (no public accessor exists) — documented as fragile but shipped rather than left null |

**Peak-force p95** (95th percentile) is computed with interpolation, not truncation — a
correctness detail worth knowing, since truncating at small `n` silently reports a lower percentile
than labeled (verified: an n=10 truncated index was actually reporting p90 under a "p95" label
before the fix).

---

## 12. Likely interview questions, answered directly

**"Why PPO and not another RL algorithm?"**
The upstream Factory/FORGE tasks ship a tuned PPO (RL-Games) config, and the project's own
discipline is "reproduce the baseline before changing anything" — starting from a known-working
algorithm and config isolates the actual research variable (force-awareness) instead of confounding
it with an algorithm change.

**"Why is the observation relative to the socket instead of absolute robot state?"**
So the same learned skill (close the remaining gap) works no matter where the socket randomly
spawns each episode — an absolute-position observation would let the policy leak/memorize one fixed
solution instead of learning the general skill, and would fail the moment domain randomization
moved the target.

**"What exactly does 'force-aware' mean here, mechanically?"**
Two independently toggleable things: whether the *policy* observes a noisy, smoothed force reading
and a per-episode force threshold (`use_force_obs`), and whether the *reward* is penalized for force
in excess of that threshold (`use_force_penalty`). They're deliberately separated so the ablation
can distinguish "does seeing force help" from "does being penalized for force help," rather than
only ever testing them bundled together.

**"How do you know the ablation actually removed the signal, not just its weight?"**
The raw `contact_penalty` term itself is zeroed (not multiplied by a zero reward-scale) when
`use_force_penalty=False`, and this is verified by checking `logs_rew_contact_penalty` in
TensorBoard reads flat exactly `0.0` for the whole run — proof at the data level, not just
"the config says so."

**"Why don't episodes end early on success?"**
They can (`use_early_termination` flag exists) but it's off by default, because a partial per-env
reset would perturb other envs still mid-episode in the same shared PhysX simulation during a
scripted multi-second IK/grasp reset sequence. Every episode currently runs the full ~150 steps and
success is tracked as a sticky label rather than an early-exit — a deliberate scope decision, not an
oversight, and it's documented as a live tradeoff (episode efficiency vs. reset-safety) rather than
settled permanently.

**"What would make Policy D fail to beat Policy A, and would that still be a valid result?"**
Yes — the project doc explicitly treats a negative result as legitimate, e.g. "force observations
reduced jamming under pose perturbation but didn't improve success when the policy already has
exact relative geometry." The honest failure mode to watch for is the force penalty making the
policy *overly cautious* (refusing necessary contact) rather than safer — this is exactly why the
2×2 design separates observation from penalty, so that failure mode is attributable to one specific
toggle instead of "force-awareness" as an undifferentiated blob.

---

## Further reading in this repo

- [`rl-primer.md`](rl-primer.md) — the four-layer mental model (Isaac Sim → Isaac Lab → RL-Games →
  our code) and where the actual research contribution sits.
- [`ppo_forge_training_notes.md`](ppo_forge_training_notes.md) — full PPO math derivation with
  plain-English translation next to every formula, plus a worked numeric example of one reward
  computation.
- [`forge_task_worksheet.md`](forge_task_worksheet.md) — the original self-study exercise this
  glossary is a completed/consolidated answer key to.
- [`experiment_log.md`](experiment_log.md) — dated, chronological account of what was actually done,
  including bugs found and corrected explanations (useful for "walk me through a mistake you caught"
  interview questions).
- [`../force_aware_peg_insertion_project.md`](../force_aware_peg_insertion_project.md) — the
  authoritative project spec this whole effort is scoped against.
