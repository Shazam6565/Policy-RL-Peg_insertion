# Next 10 Steps

Continues from `today_next_steps.md` (steps 1–10, all done as of 2026-08-07: instance
management, external project scaffold, FORGE fork proven bit-identical to upstream,
per-component reward logging verified, TensorBoard live). Per the project doc's own
training sequence ([§15](../force_aware_peg_insertion_project.md#15-ppo-training-plan)),
step 3 ("add your logging") is now done — this is where **Policy A** starts
([§12](../force_aware_peg_insertion_project.md#12-experimental-design)), matching
Week 3 of the [six-week plan](../force_aware_peg_insertion_project.md#20-six-week-execution-plan).

**Two things worth knowing before starting**, found while reading the forked code for
this plan:

- `ForgeEnvCfg.obs_order` is `[fingertip_pos_rel_fixed, fingertip_quat, ee_linvel,
  ee_angvel, ft_force, force_threshold]` — only the last two are force-related
  (`ft_force` = 3-dim, `force_threshold` = 1-dim). Because FORGE ships force-aware
  *by default*, our unmodified fork is already close to **Policy D** (force obs +
  force penalty), not Policy A. Building Policy A means *subtracting* from the
  fork, not adding to a blank slate — the reverse of how §15's step list reads
  ("add force observations" in step 6) if taken too literally.
- `FactoryEnv._get_dones()` currently returns `time_out, time_out` — every episode
  runs to the time limit and terminates only that way. There's no early termination
  on success/drop/force-limit, and nowhere is a `termination_reason` label produced.
  This is a real gap against [§9.6](../force_aware_peg_insertion_project.md#96-failure-and-termination-conditions)
  and the Week 2 deliverable list, not yet closed by anything done so far. Step 4
  below makes a deliberate decision about it rather than discovering it mid-eval.

---

1. **Decide the Policy A–D implementation strategy.** Config-driven (one `ForceEnv`
   class, boolean flags like `use_force_obs`/`use_force_penalty` picked per registered
   task ID) vs. four separate forked directories. Config-driven is less duplication
   and matches [§15](../force_aware_peg_insertion_project.md#15-ppo-training-plan)'s
   "freeze configurations before final comparison" spirit, but means touching shared
   code four times instead of once per policy — worth a real decision, not a default.

2. **Decide how to handle termination reasons** (the `_get_dones()` gap above): extend
   our fork's `_get_dones()`/`_get_rewards()` to compute and stash a per-env
   `termination_reason` in `self.extras` (a real, documented deviation from upstream
   FORGE — timeout-only becomes success/timeout/force_limit/dropped-aware), or leave
   environment termination as timeout-only and classify reasons *post-hoc* in the
   evaluation script from the logged trajectory. Both are defensible; pick one and
   write down why, since it affects every evaluation script downstream.

3. **Implement Policy A**: geometry-only observations (drop `ft_force`,
   `force_threshold` from `obs_order`) and no excessive-force penalty (zero or drop
   `contact_penalty`'s scale). Register under its own task ID (e.g.
   `Shaurya-ForcePegInsert-PolicyA-Direct-v0`) via whatever mechanism step 1 decided.

4. **Smoke-test Policy A** — same tiny `--num_envs 64 --max_iterations 20` check used
   for the baseline fork. Confirm it registers, trains without error, and that the
   `logs_rew_contact_penalty`/force-related TensorBoard curves are now flat/absent
   (proof the ablation actually removed what it was supposed to, not just cosmetically).

5. **Build a minimal evaluation script** (`scripts/evaluate_policy.py`,
   [§17](../force_aware_peg_insertion_project.md#17-evaluation-script-requirements)):
   load a checkpoint, disable stochasticity, run deterministic episode seeds, export
   the per-episode CSV schema from
   [§11.4](../force_aware_peg_insertion_project.md#114-episode-level-logging-schema).
   Doesn't need to be polished yet — Policy A's first real run needs *something* to
   evaluate against immediately, per the "don't wait until the end to build eval
   tooling" spirit of the doc.

6. **Define the first evaluation suite** — just `nominal` from
   [§11.3](../force_aware_peg_insertion_project.md#113-evaluation-dataset)
   (`configs/evaluation_suites.yaml`), fixed seeds, no randomization yet. The
   randomized suites (pose_shift, low/high_friction, mass_shift, combined_ood) can
   wait until Week 5's robustness pass.

7. **Run Policy A's first real training** (not a smoke test) — headless,
   `--num_envs 1024` per [§7](../force_aware_peg_insertion_project.md#7-installation-and-baseline-verification)
   (drop to 512/256 if it exceeds GPU memory on the L40S), enough iterations to see
   the reward curve actually plateau rather than 20 iterations of noise.

8. **Evaluate that checkpoint** against the `nominal` suite with the step 5 script.
   This produces the *first real row* of the results table template in
   [§22](../force_aware_peg_insertion_project.md#22-readme-results-table-template) —
   Policy A's nominal success rate, not yet OOD.

9. **Verify repeatability**: train Policy A with two more seeds (3 total, the
   [§12 minimum standard](../force_aware_peg_insertion_project.md#12-experimental-design)),
   evaluate each on identical seeds, and confirm results are consistent rather than
   one lucky/unlucky run — this is literally step 5 of
   [§15](../force_aware_peg_insertion_project.md#15-ppo-training-plan)'s own sequence.

10. **Wrap up**: dated `experiment_log.md` entry (what Policy A's config actually
    is, what the 3-seed nominal results were, anything surprising), commit + push,
    update this file's checklist or start the next one. This closes out Week 3;
    Policy B (add force observations back, still no penalty) is next after this.

---

**Not in scope for this batch:** Policies B/C/D, any randomized/OOD evaluation
suite, curriculum learning ([§10](../force_aware_peg_insertion_project.md#10-curriculum-learning)),
or report/video packaging ([§21](../force_aware_peg_insertion_project.md#21-technical-report-outline)–[§23](../force_aware_peg_insertion_project.md#23-video-storyboard)).
Those follow the same "prove A works and is repeatable before touching B" discipline
the project doc insists on throughout.
