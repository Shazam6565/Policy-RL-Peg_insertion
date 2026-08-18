# Policy C — Nominal Evaluation Results

Source: `results/raw/policy_c_seed{0,1,2}_nominal.csv` — 500 deterministic
episodes per seed, fixed evaluation seeds `[1000, 1001, 1002]`,
`evaluate_policy.py --task Shaurya-ForcePegInsert-PolicyC-Direct-v0 --suite nominal`.
Training and evaluation details: `docs/experiment_log.md`'s 2026-08-16 entry.

| Seed | Episodes | Success rate | Median steps | Peak force p95 | Force-limit rate |
|---|---:|---:|---:|---:|---:|
| 0 | 500 | 0.0% | 149 | 13.5 N | 0.2% |
| 1 | 500 | 10.2% | 149 | 16.0 N | 0.2% |
| 2 | 500 | 0.0% | 149 | 13.2 N | 0.0% |
| **Mean (range), n=3 seeds** | 1500 | **3.4% (0.0–10.2%)** | 149 | **14.2 N (13.2–16.0)** | **0.1% (0.0–0.2%)** |

This is the real, measured Policy C row for the
[§22 results table](../../force_aware_peg_insertion_project.md#22-readme-results-table-template)
— not an estimate. Policy C (`use_force_obs=False, use_force_penalty=True`) is
the mirror-image ablation of Policy B: the contact penalty is active but the
actor never observes contact force, so it cannot use the penalty signal to
adjust its behavior in real time. **Mean success (3.4%) is lower than both
Policy A's geometry-only baseline (7.3%) and Policy B's force-observation
policy (14.2%)**, and peak contact force (14.2 N mean p95) is dramatically
*lower* than either — the force penalty successfully suppresses contact force,
but without the observation to act on, the policy mostly learns to avoid
insertion altogether rather than insert carefully.

## Notes / caveats

- **Success is almost entirely carried by one seed.** Seed 1 (10.2%) is the
  only seed that learned anything resembling the task; seeds 0 and 2 both
  evaluate at exactly 0.0% success. This mirrors the training-time picture
  (`docs/experiment_log.md`'s 2026-08-16 entry): seed 0 stayed at ~0.0%
  training-time success rate for all 500 epochs, seed 1 climbed to a real
  16.92% (peak 20.9%), and seed 2 also stayed near-zero (~0.02% last-20 mean,
  peak 0.39%) despite training to the same reward plateau (~50) as seed 1.
  **Training reward alone does not predict whether a seed learns the task at
  all under this ablation** — a more extreme version of the same caveat noted
  for Policy A/B's training-vs-eval gap.
- **The low peak-force numbers are a symptom, not a win.** With success this
  low, most episodes never make firm contact in the first place — a policy
  that mostly fails to attempt insertion will trivially show low peak contact
  force. This is not evidence the force penalty is "working well," it is
  evidence the policy under-explores contact-rich states without force
  feedback to guide it.
- **Jam rate is not measured**, same limitation as Policy A/B —
  `termination_reason` only distinguishes `timeout` / `success` / `force_limit`.
  Pooled across all 1500 episodes: 1447 `timeout`, 51 `success`, 2
  `force_limit`, 0 `other`. §22's "Jam Rate" column is left `TBD`.
- **Median completion steps is 149 for every seed**, for the same structural
  reason documented in `policy_a_nominal.md`: `ForgeEnv._get_dones()` never
  terminates an episode early on success, so a peg that seats at step 50 keeps
  being simulated through step 149.
- **"Force-limit rate" is an incidence rate, not a termination rate** — same
  definition and same caveat as Policy A/B's tables (episodes never end early
  on force, so this counts "exceeded `force_limit_threshold` (50N placeholder)
  at some point," not "was ended by it").
- Nominal only — no out-of-distribution suite has been run yet (Week 5 scope).
