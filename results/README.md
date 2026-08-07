# Results

Curated, portfolio-facing outputs — not raw experiment dumps (those live in
`force_peg_rl/logs/` on the training instance and stay out of git; see the root
README's "Where things actually live" section).

```text
results/
├── videos/     — policy rollout recordings (playback captures, demo clips)
├── raw/        — per-episode evaluation CSVs (one row per episode, see project
│                 doc §11.4 for the schema) — not yet populated, starts once the
│                 evaluation script (next_10_steps.md step 5) exists
├── tables/     — aggregated results tables (success rate, force metrics per
│                 policy/seed) — not yet populated
└── figures/    — plots for the technical report — not yet populated
```

Only `videos/` has content so far (`smoke_test_playback.mp4`, from the very first
baseline PPO smoke test). The other three subfolders get created once Policy A's
evaluation pipeline exists — see `docs/next_10_steps.md`.
