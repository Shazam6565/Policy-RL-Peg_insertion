# Policy A — Nominal Evaluation Results

Source: `results/raw/policy_a_seed{42,43,44}_nominal.csv` — 500 deterministic
episodes per seed, fixed evaluation seeds `[1000, 1001, 1002]`,
`evaluate_policy.py --task Shaurya-ForcePegInsert-PolicyA-Direct-v0 --suite nominal`.
Training and evaluation details: `docs/experiment_log.md`'s 2026-08-09–2026-08-12
entry.

| Seed | Episodes | Success rate | Median steps | Peak force p95 | Force-limit rate |
|---|---:|---:|---:|---:|---:|
| 42 | 500 | 6.4% | 149 | 39.3 N | 2.4% |
| 43 | 500 | 7.6% | 149 | 34.7 N | 1.0% |
| 44 | 500 | 7.8% | 149 | 37.6 N | 2.0% |
| **Mean (range), n=3 seeds** | 1500 | **7.3% (6.4–7.8%)** | 149 | **37.2 N (34.7–39.3)** | **1.8% (1.0–2.4%)** |

This is the real, measured Policy A row for the
[§22 results table](../../force_aware_peg_insertion_project.md#22-readme-results-table-template)
— not an estimate.

## Notes / caveats

- **Jam rate is not measured.** The fork's `termination_reason` only distinguishes
  `timeout` / `success` / `force_limit` (see `docs/2026-08-09_next_10_steps.md`'s
  "worth knowing" notes) — no jam-detection signal or threshold has been implemented
  anywhere in the code or project doc. Pooled across all 1500 episodes:
  1364 `timeout`, 109 `success`, 27 `force_limit`, 0 `other`. §22's "Jam Rate" column
  is left `TBD` in the README rather than backfilled with 0%, since "not instrumented"
  and "measured zero" are different claims.
- **Median completion steps is 149 for every seed and every episode**, not just on
  average — because `ForgeEnv._get_dones()` never terminates an episode early on
  success (resets stay synchronized to the shared timeout). This is a real,
  documented property of this fork, not a data artifact: a peg that seats at step 50
  keeps being simulated (and keeps collecting reward) through step 149 regardless.
- **"Force-limit rate" means "contact force exceeded `force_limit_threshold` (50N, an
  untuned placeholder — see the caveats doc) at some point during the episode,"** not
  "the episode was ended by it" — episodes never end early here, so this number is a
  lower-bound *incidence* rate, not a termination rate in the usual sense.
- Nominal only — no out-of-distribution (pose/friction/mass-shift) suite has been run
  yet (Week 5 scope per the six-week plan).
