# Policy B — Nominal Evaluation Results

Source: `results/raw/policy_b_seed{0,1,2}_nominal.csv` — 500 deterministic
episodes per seed, fixed evaluation seeds `[1000, 1001, 1002]`,
`evaluate_policy.py --task Shaurya-ForcePegInsert-PolicyB-Direct-v0 --suite nominal`.
Training and evaluation details: `docs/experiment_log.md`'s 2026-08-14 entry.

| Seed | Episodes | Success rate | Median steps | Peak force p95 | Force-limit rate |
|---|---:|---:|---:|---:|---:|
| 0 | 500 | 17.2% | 149 | 31.2 N | 1.2% |
| 1 | 500 | 15.8% | 149 | 35.6 N | 0.8% |
| 2 | 500 | 9.6% | 149 | 37.2 N | 1.0% |
| **Mean (range), n=3 seeds** | 1500 | **14.2% (9.6–17.2%)** | 149 | **34.7 N (31.2–37.2)** | **1.0% (0.8–1.2%)** |

This is the real, measured Policy B row for the
[§22 results table](../../force_aware_peg_insertion_project.md#22-readme-results-table-template)
— not an estimate. Compared against Policy A's real nominal numbers
(`policy_a_nominal.md`: 7.3% mean success, 37.2 N mean p95 force), Policy B
(force observations restored, contact penalty still off) roughly doubles
success rate while running at comparable-to-slightly-lower peak contact
force — the direction expected for a force-aware ablation, now measured
rather than assumed.

## Notes / caveats

- **Seed 2 is the outlier.** It reached the highest final training reward of
  the three seeds (170.6 vs 132.6 / 141.4, see `docs/experiment_log.md`) but
  the lowest evaluated success rate (9.6%) and highest peak/force-limit
  figures — a reminder that training-time reward and deterministic nominal-suite
  success don't always rank the same way across seeds, same caveat noted for
  Policy A's training-vs-eval gap.
- **Jam rate is not measured**, same limitation as Policy A —
  `termination_reason` only distinguishes `timeout` / `success` / `force_limit`.
  Pooled across all 1500 episodes: 1272 `timeout`, 213 `success`, 15
  `force_limit`, 0 `other`. §22's "Jam Rate" column is left `TBD`.
- **Median completion steps is 149 for every seed**, for the same structural
  reason documented in `policy_a_nominal.md`: `ForgeEnv._get_dones()` never
  terminates an episode early on success, so a peg that seats at step 50 keeps
  being simulated through step 149.
- **"Force-limit rate" is an incidence rate, not a termination rate** — same
  definition and same caveat as Policy A's table (episodes never end early on
  force, so this counts "exceeded `force_limit_threshold` (50N placeholder)
  at some point," not "was ended by it").
- Nominal only — no out-of-distribution suite has been run yet (Week 5 scope).
