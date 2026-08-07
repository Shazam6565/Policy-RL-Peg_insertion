# Experiment Log

Running, dated notes on what was done and what was seen each session — informal by
design, distinct from the final technical report. Newest entry on top.

---

## 2026-08-07 — Fork FORGE into `force_peg_rl`, prove it's faithful, wire up logging

Brought the instance back up (`env_isaaclab` venv survived the stop/start untouched)
and worked through roadmap steps 2–10. Generated the `force_peg_rl` external project
via IsaacLab's template generator — `./isaaclab.sh --new` turned out to need a real
TTY (InquirerPy full-screen prompts), so it was driven by calling the generator
function directly with the same answers (External / Direct / RL-Games), which
produces an identical result. Forked `forge_env.py`, `forge_env_cfg.py`,
`forge_events.py`, `forge_tasks_cfg.py`, `forge_utils.py`, and the PPO agent config
into `tasks/direct/force_peg/` verbatim, registered under our own ID
`Shaurya-ForcePegInsert-Direct-v0`. The regression check was the best possible
outcome: a 20-iteration/64-env smoke test through the fork produced a **bit-identical**
final reward (22.362465) to the original upstream baseline run — same seed, same
code, same number to six decimal places. Step 8 (per-component reward logging)
turned out to already exist upstream (`self.extras["logs_rew_<name>"]` in both
`FactoryEnv` and `ForgeEnv`) — nothing to build there, just verified it.

Three real bugs found and fixed along the way, all in generated/template code, none
in the FORGE/Factory logic itself: (1) the scaffold's `list_envs.py` had a hardcoded
`"Template-"` ID filter that silently hid any differently-named task — replaced with
a check on the entry point's package prefix; (2) that fix then surfaced a second bug,
19 of gymnasium's bundled MuJoCo envs register with a function instead of a string as
`entry_point`, which crashed the naive filter with no visible traceback; (3) the
generated `.gitignore` (both the project-level one and a nested `.vscode/.gitignore`)
had two independent bugs that silently excluded every `__init__.py` and every
`.vscode/*.json` file from git — meaning the scaffold commit from the prior session
would have produced a non-importable package on a fresh clone. All fixed and
verified via `git ls-files`.

Set up TensorBoard on the instance (`--reload_interval 5` for live updates during
future runs) tunneled to localhost via the existing SSH ControlMaster connection,
comparing the baseline and fork runs side by side — confirmed the individual reward
curves render correctly (30 distinct scalar cards under `logs_rew_*`), not just a
total-reward line.

Nothing here touches what the policy sees or is rewarded for yet — pure
infrastructure, per plan. Next session starts at Policy A (project doc §12), now
that the fork is proven faithful and `obs_order` is already documented in the
worksheet.
