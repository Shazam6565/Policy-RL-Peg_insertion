# Results

Curated, portfolio-facing outputs — not raw experiment dumps (those live in
`force_peg_rl/logs/` on the training instance and stay out of git; see the root
README's "Where things actually live" section).

```text
results/
├── videos/     — policy rollout recordings: playback captures, demo clips, and
│                 curated success/near-miss close-ups per policy
├── raw/        — per-episode evaluation CSVs (one row per episode, §11.4 schema),
│                 one file per (policy, seed, suite) — e.g.
│                 policy_a_seed44_nominal.csv. This is `evaluate_policy.py`'s
│                 documented default output location (see its own --help text) —
│                 point it here, not force_peg_rl/results/raw/, so raw data and the
│                 code that produced it stay in separate, purpose-matched trees.
├── tables/     — aggregated results tables (success rate, force metrics per
│                 policy/seed, mean ± range across seeds) — one markdown file per
│                 policy/suite, e.g. policy_a_nominal.md
└── figures/    — plots for the technical report — not yet populated
```

Populated so far: Policy A and Policy B's nominal-suite raw CSVs (3 seeds × 500
episodes each) and their aggregate tables (`tables/policy_a_nominal.md`,
`tables/policy_b_nominal.md`), plus curated success/near-miss videos for both —
see `docs/experiment_log.md` and `docs/2026-08-09_next_10_steps.md` for how these
were produced. `figures/`, Policy C/D's raw/tables entries, and the OOD suites are
still open (Week 5 / remainder-of-batch scope).
