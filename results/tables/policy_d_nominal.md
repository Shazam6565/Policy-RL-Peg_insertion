# Policy D — Nominal Evaluation Results

Source: `results/raw/policy_d_seed{0,1,2}_nominal.csv` — 500 deterministic
episodes per seed, fixed evaluation seeds `[1000, 1001, 1002]`,
`evaluate_policy.py --task Shaurya-ForcePegInsert-Direct-v0 --suite nominal`.
Training and evaluation details: `docs/experiment_log.md`'s 2026-08-18 entry.

| Seed | Episodes | Success rate | Median steps | Peak force p95 | Force-limit rate |
|---|---:|---:|---:|---:|---:|
| 0 | 500 | 10.8% | 149 | 19.1 N | 0.2% |
| 1 | 500 | 9.2% | 149 | 18.3 N | 0.0% |
| 2 | 500 | 9.8% | 149 | 18.8 N | 0.0% |
| **Mean (range), n=3 seeds** | 1500 | **9.9% (9.2–10.8%)** | 149 | **18.8 N (18.3–19.1)** | **0.1% (0.0–0.2%)** |

This is the real, measured Policy D row for the
[§22 results table](../../force_aware_peg_insertion_project.md#22-readme-results-table-template)
— not an estimate. Policy D (`use_force_obs=True, use_force_penalty=True`) is
the fully force-aware configuration — both the force observation and the
contact penalty are active. **Mean success (9.9%) sits between Policy A's
geometry-only baseline (7.3%) and Policy B's force-observation-only policy
(14.2%), and well above Policy C's force-penalty-only policy (3.4%)** — but
it does **not** beat Policy B outright, despite having by far the best
training-time success trajectory of any seed in the whole B/C/D batch
(seed 2 peaked at 30.86% training-time success, seed 0 at 27.73%). Peak
contact force (18.8 N mean p95) is the lowest of any force-aware policy
(well below Policy B's 34.7 N), consistent with the contact penalty
successfully suppressing force even where Policy B's reward had no direct
incentive to do so.

## Notes / caveats

- **Training-time success does not translate 1:1 into deterministic nominal
  success for this policy.** All three Policy D seeds finished training with
  22–25% last-20-epoch training success (peaks up to 30.86%), yet the
  deterministic nominal evaluation gives 9.2–10.8% — roughly a 2.5x drop.
  Policy B and Policy C both showed some training-vs-eval gap too, but
  Policy D's is the largest in absolute terms. A plausible explanation is
  that the added contact-force penalty makes the policy more conservative
  under the stochastic exploration noise present during training (where
  occasional risk-taking pays off across many rollouts) than under the
  deterministic, noise-free evaluation policy used here — but this is not
  confirmed and would need an out-of-distribution or noise-injected suite to
  test directly (Week 5 scope).
- **Success is far more consistent across seeds than Policy C's.** Policy C's
  nominal success was almost entirely carried by one outlier seed (0.0%,
  10.2%, 0.0%). Policy D's three seeds land within a tight 9.2–10.8% band —
  having the force observation available lets every seed learn *some* usable
  contact-avoidance behavior, rather than the all-or-nothing outcome seen
  when only the penalty is present without the observation.
- **Jam rate is not measured**, same limitation as Policy A/B/C —
  `termination_reason` only distinguishes `timeout` / `success` / `force_limit`.
  Pooled across all 1500 episodes: 1351 `timeout`, 149 `success`, 0
  `force_limit`, 0 `other`. §22's "Jam Rate" column is left `TBD`.
- **Median completion steps is 149 for every seed**, for the same structural
  reason documented in `policy_a_nominal.md`: `ForgeEnv._get_dones()` never
  terminates an episode early on success, so a peg that seats at step 50 keeps
  being simulated through step 149.
- **"Force-limit rate" is an incidence rate, not a termination rate** — same
  definition and same caveat as Policy A/B/C's tables (episodes never end
  early on force, so this counts "exceeded `force_limit_threshold` (50N
  placeholder) at some point," not "was ended by it"). Policy D's rate
  (0.1% mean, two seeds at exactly 0.0%) is the lowest of the four policies —
  further evidence the contact penalty is doing real work suppressing peak
  force, independent of whether it also improves success.
- Nominal only — no out-of-distribution suite has been run yet (Week 5 scope).
