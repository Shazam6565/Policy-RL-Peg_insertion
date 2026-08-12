# Force-Aware Peg Insertion with Reinforcement Learning

A portfolio project training a Franka robot policy (PPO, NVIDIA Isaac Lab) to insert a peg
into a socket while minimizing excessive contact force and staying robust to pose, friction,
mass, and controller variation.

**Research question:** does giving a policy end-effector contact-force information, plus a
penalty for excessive force, actually improve safety and robustness over a geometry-only
baseline — or does exact relative-pose knowledge already make force information redundant?

Full project design — scope, reward formulation, experimental design (Policies A–D), metrics,
six-week plan — lives in [`force_aware_peg_insertion_project.md`](force_aware_peg_insertion_project.md).
This README is the map; that file is the spec.

---

## Status

Early stage — Week 2–3 of the [six-week plan](force_aware_peg_insertion_project.md#20-six-week-execution-plan).
Done so far: environment installed and verified, the official `Isaac-Forge-PegInsert-Direct-v0`
baseline reproduced, forked into our own external project (`force_peg_rl/`) and proven
**bit-identical** to upstream under the same seed, per-component reward logging verified live
in TensorBoard. Not yet started: Policy A (the geometry-only ablation baseline) and everything
downstream of it (B, C, D, evaluation suites, the technical report).

See [`docs/experiment_log.md`](docs/experiment_log.md) for the dated, running account of what
was actually done and observed each session, and [`docs/next_10_steps.md`](docs/next_10_steps.md)
for exactly what's queued next.

---

## Repository map

```text
.
├── force_aware_peg_insertion_project.md   — the spec: scope, reward design, experiment
│                                             design, metrics, six-week plan. Read this first.
├── README.md                              — you are here
│
├── force_peg_rl/                          — THE DELIVERABLE. A standard external IsaacLab
│   │                                         project (generated via the IsaacLab template),
│   │                                         forked from Isaac-Forge-PegInsert-Direct-v0.
│   │                                         This is the only directory meant to be portable/
│   │                                         cloneable on its own — see force_peg_rl/README.md
│   │                                         for the standard IsaacLab install/train/play
│   │                                         instructions.
│   └── source/.../tasks/direct/
│       ├── force_peg_rl/                  — the auto-generated example task from the project
│       │                                     scaffold (`Template-Force-Peg-Rl-Direct-v0`) —
│       │                                     kept as a working sanity-check baseline, not used
│       │                                     for the actual research.
│       └── force_peg/                     — our real forked task
│                                             (`Shaurya-ForcePegInsert-Direct-v0`), copied
│                                             verbatim from upstream FORGE with headers intact.
│                                             Policies A–D get built from here.
│
├── reference/isaaclab_source/             — READ-ONLY reference copy of the upstream IsaacLab
│                                             FORGE/Factory source, for browsing/searching
│                                             locally without SSHing into the training instance
│                                             every time. Not our code — see its own README.
│
├── docs/
│   ├── rl-primer.md                       — RL concepts from scratch, written for someone new
│   │                                         to the field (start here if RL is new to you).
│   ├── forge_task_worksheet.md            — filled-in answers: the exact obs vector, action
│   │                                         vector, reward terms or this project's fork.
│   ├── ppo_forge_training_notes.md        — from-scratch study notes on the PPO + FORGE
│   │                                         training loop, diagram-first.
│   ├── experiment_log.md                  — dated running log: what was done, what was seen,
│   │                                         what was surprising. The project's memory.
│   ├── today_next_steps.md                — completed roadmap for the 2026-08-07 session.
│   ├── next_10_steps.md                   — the current queued roadmap (Policy A next).
│   └── 2026-08-03_early_planning_notes.md — historical: the very first setup notes, superseded
│                                             by everything above.
│
├── results/                               — curated, portfolio-facing outputs (videos, eval
│                                             CSVs, tables, figures). See results/README.md.
│                                             Raw experiment dumps stay out of git — see below.
│
└── logs/                                  — Claude Code tooling artifacts, gitignored, not
                                              project content.
```

### Where things actually live (and don't)

Training runs on a remote GPU instance (Brev), driven over SSH — not locally. That instance
lifecycle is managed by a **separate, reusable tool** (`isaac-sim-workspace`, its own repo, not
project-specific), not anything in this repository. This repo is the source of truth for code
and docs; the instance's disk is scratch compute. Concretely:

- **Checkpoints and raw RL-Games logs/TensorBoard runs** live under `force_peg_rl/logs/` — but
  that path is gitignored (`force_peg_rl/.gitignore`, `**/logs/*`) on purpose: they're large,
  regeneratable from a training command, and not meant to bloat the repo. A curated checkpoint
  worth keeping gets copied into `results/` deliberately, not synced automatically.
- **`force_peg_rl/` itself** is round-tripped by hand between the training instance and this
  repo (edit/generate remotely → copy down → commit here) — it is not a git submodule and has
  no independent git history of its own; this repo is its only source of truth.

---

## Reading order

**New to the project, or picking this up cold:**
1. [`force_aware_peg_insertion_project.md`](force_aware_peg_insertion_project.md) — the spec.
2. [`docs/rl-primer.md`](docs/rl-primer.md) — if RL concepts are new.
3. [`docs/forge_task_worksheet.md`](docs/forge_task_worksheet.md) — what the actual task is:
   the exact observation vector, action vector, reward terms, and termination conditions.
4. [`docs/experiment_log.md`](docs/experiment_log.md) — what's actually been done, in order.
5. [`docs/next_10_steps.md`](docs/next_10_steps.md) — what's next.

**Just want to run something:** skip to
[`force_peg_rl/README.md`](force_peg_rl/README.md) — standard IsaacLab external-project
install/train/play instructions. It's a self-contained IsaacLab extension; the only project-doc
context you need is which task ID to pass: `Shaurya-ForcePegInsert-Direct-v0`.

---

## Results

Policy A has real, measured numbers as of 2026-08-12; B, C, and D are not trained yet. Per the
project doc's own rule: *"Do not fill the table with expected values. Only report measured
results"* — so those rows stay `TBD`, not filled with guesses.

| Policy | Force Obs. | Force Penalty | Nominal Success | OOD Success | Peak Force p95 | Jam Rate |
|---|---:|---:|---:|---:|---:|---:|
| A: Geometry baseline | No | No | 7.3% (6.4–7.8%, n=3 seeds) | TBD | 37.2 N | TBD (not instrumented) |
| B: Force observation | Yes | No | TBD | TBD | TBD | TBD |
| C: Force penalty | No | Yes | TBD | TBD | TBD | TBD |
| D: Force-aware | Yes | Yes | TBD | TBD | TBD | TBD |

Per-seed breakdown, caveats (episodes never terminate early, so "force-limit rate" is an
incidence rate not a true termination rate; jam rate has no detection instrumentation at all
yet), and the raw per-episode CSVs are in
[`results/tables/policy_a_nominal.md`](results/tables/policy_a_nominal.md) /
[`results/raw/`](results/raw/).

Representative videos, Policy A / seed 44 — a clean success and a genuine near-miss (GIFs, so
they play inline; full-quality `.mp4` source for each is linked underneath):

| Success | Near-miss |
|---|---|
| ![Success close-up](results/videos/policy_a_seed44_success_closeup.gif) | ![Near-miss close-up](results/videos/policy_a_seed44_nearmiss_closeup.gif) |
| [.mp4 source](results/videos/policy_a_seed44_success_closeup.mp4) | [.mp4 source](results/videos/policy_a_seed44_nearmiss_closeup.mp4) |

---

## Attribution

The FORGE and Factory task implementations this project forks from are copyright The Isaac Lab
Project Developers, BSD-3-Clause licensed — see
[`reference/isaaclab_source/README.md`](reference/isaaclab_source/README.md) for the exact
upstream commit/tag this was forked from. Copied files in `force_peg_rl/` retain their original
copyright headers unmodified.
