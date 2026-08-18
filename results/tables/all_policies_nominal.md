# Combined §22 Results Table — All Four Policies

Fills the [§22 README Results Table Template](../../force_aware_peg_insertion_project.md#22-readme-results-table-template)
with real, measured numbers from each policy's 3-seed nominal evaluation
(500 deterministic episodes/seed, fixed evaluation seeds `[1000, 1001, 1002]`,
`evaluate_policy.py --suite nominal`). Per-policy detail, per-seed breakdowns,
and full caveats are in `policy_a_nominal.md` / `policy_b_nominal.md` /
`policy_c_nominal.md` / `policy_d_nominal.md`. OOD Success and Jam Rate are
left `TBD` — out-of-distribution suites and jam detection are Week 5 scope
and have not been implemented.

| Policy | Force Obs. | Force Penalty | Nominal Success | OOD Success | Peak Force p95 | Jam Rate |
|---|---:|---:|---:|---:|---:|---:|
| A: Geometry baseline | No | No | 7.3% (6.4–7.8%) | TBD | 37.2 N (34.7–39.3) | TBD |
| B: Force observation | Yes | No | **14.2% (9.6–17.2%)** | TBD | 34.7 N (31.2–37.2) | TBD |
| C: Force penalty | No | Yes | 3.4% (0.0–10.2%) | TBD | **14.2 N (13.2–16.0)** | TBD |
| D: Force-aware | Yes | Yes | 9.9% (9.2–10.8%) | TBD | 18.8 N (18.3–19.1) | TBD |

## The ablation, closed out

This 9-run batch (B, C, D × 3 seeds each, plus the earlier Policy A batch)
was designed to separate two effects that "force-aware RL" usually bundles
together: **giving the policy the contact-force signal as an observation**,
and **penalizing contact force in the reward**. The four cells above are
what happens when each is toggled independently.

**Force observation is the factor that matters; the penalty alone actively
hurts.** Policy B (observation only) roughly doubles Policy A's baseline
success rate (14.2% vs 7.3%) while running at comparable-to-lower peak
force. Policy C (penalty only, no observation) *drops below the baseline*
(3.4% vs 7.3%) — a policy that is punished for contact but cannot sense it
mostly learns to avoid insertion altogether rather than insert carefully;
its low peak force (14.2 N) is a symptom of that avoidance, not a
capability. **The two effects are not simply additive.** Policy D (both
toggles on) does not stack Policy B's success gain with Policy C's force
reduction — it lands at 9.9% success, above Policy C and the baseline but
*below* Policy B, while achieving the lowest peak force of any force-aware
policy (18.8 N, well below Policy B's 34.7 N). The clearest reading of the
full 2×2 grid: **the observation is what lets the policy learn the task at
all; the penalty then trades some of that success back for lower contact
force**, rather than acting as a pure enhancement.

A second finding, visible only once training-time and evaluation-time
numbers are compared side by side: **training-time success is not a
reliable predictor of deterministic nominal-suite success for this task.**
Policy D's seeds hit the best training-time success trajectories of the
entire batch (up to 30.86% peak) yet evaluate lower than Policy B, whose
training-time numbers were comparatively unremarkable. Policy C showed the
same disconnect even more starkly (one seed at 16.92% training-time
success evaluated at only 10.2%; two seeds near 0% throughout). Any future
work on this task should evaluate on the deterministic suite before drawing
conclusions from training curves alone.

Full per-policy tables, per-seed breakdowns, and caveats (jam rate not
measured, median-steps structural artifact, force-limit rate is an
incidence rate not a termination rate, nominal-suite only): see the four
`policy_{a,b,c,d}_nominal.md` files in this directory.
