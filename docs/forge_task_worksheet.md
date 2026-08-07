# FORGE Task Understanding Worksheet

**Purpose:** the project doc is explicit — *"Read the official FORGE task code and write down the
exact observation vector, action vector, reward terms, and termination conditions... Do not start by
tuning PPO. First make the existing system observable, reproducible, and understandable."* This is
that exercise. Read the pointed-to code yourself and fill in the blanks in **your own words** — not
copy-pasted variable names. The point is to build the mental model, not to produce a transcript.

All file paths below are relative to `reference/isaaclab_source/` in this repo (a local read-only
copy) — or, on the running instance, relative to `~/docker/isaac-sim/data/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/`.

Two classes matter: **`FactoryEnv`** (the base — most of the actual logic) and **`ForgeEnv`**
(extends it, adds force-sensing/penalty/success-prediction). Where a section says "base" and
"Forge addition," read both — Forge usually calls `super()` first, then adds on top.

---

## 1. Observation vector

Where to look:
- `tasks/direct/forge/forge_env_cfg.py`, lines 104–111 (`obs_order` — the literal ordered list)
- `tasks/direct/factory/factory_env_cfg.py`, lines 19–25 (`OBS_DIM_CFG` — how many numbers each entry contributes)
- `tasks/direct/factory/factory_env.py`, `_get_factory_obs_state_dict()` (base terms, ~line 161)
- `tasks/direct/forge/forge_env.py`, `_get_observations()` (Forge additions, ~line 120)

Guiding questions:
- List every entry in `obs_order`. For each one, what physical quantity does it represent, and how
  many numbers does it take up (check `OBS_DIM_CFG` / the update to it in `forge_env_cfg.py` line 16)?
- The project doc's generic template (§9.2) describes raw joint positions/velocities as part of the
  observation. Are they in `obs_order`? If not, what does this task use instead to represent the
  robot's state relative to the target? Why might that be a more useful representation?
- One more entry always gets appended in code, beyond what's in `obs_order` — find where
  (`+ ["...")` appears in `_get_observations`). What is it, and why would a policy need it as input?
- Where does noise get added to these observations, and why would a task deliberately corrupt its
  own sensor readings? (look at `forge_env.py::_compute_intermediate_values`)

Your answers:

| Entry (in order) | Dimensions | What it physically represents |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| *(the appended one)* | | |

---

## 2. Action vector

Where to look:
- `tasks/direct/forge/forge_env_cfg.py`, line 96 (`action_space`)
- `tasks/direct/factory/factory_env_cfg.py`, line 75 (`action_space` — the base value, for comparison)
- `tasks/direct/forge/forge_env.py`, `_apply_action()` (~line 152)

Guiding questions:
- The project doc's template (§9.3) describes a 6-dimensional action:
  `[Δx, Δy, Δz, Δroll, Δpitch, Δyaw]`. Is Forge's action space 6 or 7? If it's different from
  Factory's base value, go find what the extra dimension controls — search for
  `self.actions[:, 6]` across `forge_env.py`. (It's used somewhere you wouldn't expect for a
  "control" action.)
- `_apply_action` clips the requested position and rotation targets before they're sent to the
  controller. Find both clip operations — what real-world failure are they preventing?
- Position actions are computed relative to some reference frame, not the robot's current pose
  directly. What is that frame anchored to? (look at `fixed_pos_action_frame`)

Your answers:

- Total action dimensions: ___
- What each dimension does:
  1.
  2.
  3.
  4.
  5.
  6.
  7. (if present):
- Why the position/rotation targets get clipped:

---

## 3. Reward terms

Where to look:
- `tasks/direct/factory/factory_env.py`, `_get_factory_rew_dict()` (base terms, ~line 427)
- `tasks/direct/forge/forge_env.py`, `_get_rewards()` (Forge additions, ~line 234)

Guiding questions:
- List every key that appears in `rew_dict` across **both** functions (there are 7 base terms + 3
  Forge terms). One sentence each: what behavior does it reward, or what does it penalize?
- The project doc's template (§9.4) describes a single approach reward shaped like
  `exp(-k * distance)`. The real code uses a different shape (`factory_utils.squashing_fn`) and —
  more interestingly — **three separate** distance-based terms (`kp_baseline`, `kp_coarse`,
  `kp_fine`). Look at how `keypoint_coef_baseline/coarse/fine` differ (you may need to peek at
  `factory_tasks_cfg.py`). Why might three reward shapes at different distance scales work better
  than one?
- Which single term is the actual contact-force penalty? What does it compare the current force
  reading against, and what happens when force is *below* that threshold? (look at the `relu`)
- What is `success_pred_error`, and which action dimension feeds into it? (connects back to
  question 2 — this is *not* a physical control action)

Your answers:

| Term | Rewards / penalizes | Base or Forge? |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

---

## 4. Termination conditions

Where to look:
- `tasks/direct/factory/factory_env.py`, `_get_dones()` (very short — ~line 334)
- `tasks/direct/factory/factory_env.py`, `_get_curr_successes()` (~line 344)

Guiding questions:
- Read `_get_dones()` carefully — it's only a few lines. What does it actually check?
- The project doc's template (§9.6) lists eight possible termination reasons: success, timeout,
  peg_dropped, force_limit, workspace_violation, joint_limit, jammed, other. How many of those does
  the *real* code check for ending an episode early? Write down exactly what you find — this is a
  genuine, useful gap between the doc's generic template and the specific upstream implementation.
- If success doesn't end the episode immediately, how does the code know whether an episode
  succeeded at all? Where is that tracked/logged instead of being used to terminate?

Your answers:
- What `_get_dones()` actually returns:
- Termination reasons implemented vs. the doc's template — what's missing, and what does that
  imply about how we'd need to extend this ourselves later:
- How success actually gets tracked if not via early termination:

---

## 5. Reset / domain randomization

Where to look:
- `tasks/direct/factory/factory_env.py`, `randomize_initial_state()` (long — skim it, don't read
  every line, ~line 621)
- `tasks/direct/forge/forge_env_cfg.py`, `EventCfg` (~line 39)

Guiding questions:
- Name three distinct physical quantities that get randomized when an episode resets.
- `EventCfg` randomizes some things at reset (`mode="reset"`) and at least one thing on a repeating
  timer *during* the episode (`mode="interval"`). Find both — what's the difference in what they're
  simulating, and why would one need to change mid-episode instead of once at the start?

Your answers:
- Three things randomized at reset:
- What changes on an interval, and why mid-episode:

---

## Wrap-up

- In 2–3 sentences, describe the full FORGE peg-insert MDP (state → action → reward → done) in
  your own words, as if explaining it to someone who has never seen the code.
- What's one thing about the real implementation that surprised you, or differed from what the
  project doc's generic template led you to expect?
