# Next 10 Steps — Today's Session

Continues from roadmap steps 1–10 (all done: instance built, Isaac Sim + Isaac Lab installed, both
smoke tests passed, PPO smoke test + playback done, source reference + primer docs written, repo on
GitHub). Everything below stays in the spirit of the project doc's own sequencing
([§15](../force_aware_peg_insertion_project.md#15-ppo-training-plan)): reproduce baseline unchanged
→ fork → add logging → *then* start changing things. Nothing here modifies reward/observation logic
yet — that starts once Policy A is actually defined (step 8).

1. **Fill in `docs/forge_task_worksheet.md`.** Read the pointed-to code yourself, write the answers
   in your own words. Everything downstream (the fork, Policy A/B/C/D later) depends on actually
   knowing this MDP, not just having it explained.

2. **`/isaac-up`** — bring the instance back online. Confirm the persisted `env_isaaclab` venv
   survived the stop/start (a quick `list_envs.py` re-check is enough — no need to redo the install).

3. **Generate the external project**: `./isaaclab.sh --new` → Project type `External`, Workflow
   `Direct`, RL framework `RL-Games` (project doc [§8](../force_aware_peg_insertion_project.md#8-create-your-own-external-project)).
   This creates the `force_peg_rl` skeleton — a project that lives *outside* the IsaacLab source
   tree and can eventually become its own repo. Don't put anything into it yet.

4. **Sanity-check the empty scaffold** — run the generated template's own example task (whatever
   `--new` registers by default) through a `list_envs.py` check, *before* we put our own code in it.
   Confirms the project-generation machinery itself works, isolating any future bug to "our code"
   vs. "the scaffold."

5. **Fork the FORGE task into it**: copy `forge_env.py`, `forge_env_cfg.py`, `forge_events.py`,
   `forge_tasks_cfg.py`, `forge_utils.py`, and the `agents/` configs into
   `force_peg_rl/.../tasks/direct/force_peg/`, preserving the upstream copyright header (project
   doc explicitly requires this). Register it under our own task ID, e.g.
   `Shaurya-ForcePegInsert-Direct-v0`.

6. **Confirm the fork is registered**: `list_envs.py` should now show both the upstream
   `Isaac-Forge-PegInsert-Direct-v0` *and* our new `Shaurya-ForcePegInsert-Direct-v0`.

7. **Regression check — reproduce the baseline through our fork.** Run the same tiny smoke test
   (`--num_envs 64 --max_iterations 20`) against *our copy*. Reward curve should behave the same as
   the original upstream smoke test. This is the doc's "train the official baseline unchanged"
   step, just now running through our own project instead of the IsaacLab source tree — proves the
   fork is a faithful copy before we touch a single line.

8. **Add reward-component logging.** Right now `rew_dict` terms get summed into one scalar and
   thrown away. Log each term separately (the project's core rule: *"Log every reward component
   separately. Never debug only the total reward."*). This is infrastructure, not a design change —
   still zero modification to what the policy sees or is rewarded for.

9. **Open TensorBoard on a logged run** — port-forward/tunnel it, confirm the individual reward
   curves from step 8 are actually visible, not just total reward. First real use of the metrics
   dashboard mentioned in the primer doc.

10. **Wrap up**: commit today's work (fork + logging) to GitHub, add a short dated entry to
    `docs/experiment_log.md` (create it if it doesn't exist yet — one paragraph: what we did, what
    we saw, anything surprising) — this is the doc's recommended running log, distinct from the
    final technical report — then `/isaac-down`.

---

**Not in scope for today:** anything from Policy A/B/C/D (project doc §12) — that starts once steps
1–10 above are actually done and step 1's worksheet is filled in. Defining Policy A specifically
requires knowing which `obs_order` entries are force-related, which is exactly what the worksheet's
question 1 asks you to find.
