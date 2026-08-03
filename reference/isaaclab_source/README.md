# Reference Copy: FORGE Task Source Code

**These are NOT our files.** This is a read-only copy of NVIDIA's Isaac Lab source, pulled from the
exact checkout running on our Brev box (`IsaacLab`, tag `v3.0.0-beta2.patch1`,
[github.com/isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab), BSD-3-Clause license — see
`LICENSE` in the real repo) so we can read and search it locally instead of SSH-ing in every time.
Copied here purely for reference while we learn the codebase — see [§8 of the project
doc](../../force_aware_peg_insertion_project.md#8-create-your-own-external-project) for when/how we
eventually fork the real behavior into our own project.

```
tasks/direct/forge/     ForgeEnv — our target task, adds force-sensing + force penalty
tasks/direct/factory/   FactoryEnv — the base class ForgeEnv extends (most of the actual robot logic)
envs/direct_rl_env.py   DirectRLEnv — the generic base class every Isaac Lab task inherits from.
                        This is where the actual step-by-step loop lives.
scripts/train.py        The script we ran for the PPO smoke test — hands control to RL-Games.
```

---

## The chain of command (who calls whom)

```
train.py  (our entry point)
  │
  │  builds the environment: gym.make("Isaac-Forge-PegInsert-Direct-v0", ...)
  │  → this constructs a ForgeEnv, which is a FactoryEnv, which is a DirectRLEnv
  │
  │  hands it to RL-Games: runner.run({"train": True, ...})
  ▼
RL-Games' own PPO trainer (lives inside the `rl_games` pip package, not copied here —
it's third-party, "trusted machinery" we configure but don't modify)
  │
  │  repeatedly calls, for every parallel env at once:
  ▼
DirectRLEnv.step(action)     ← envs/direct_rl_env.py, line 386
  │
  │  this is "one tick" of simulation — see walkthrough below
  ▼
returns (observations, reward, terminated, timed_out, extras) back to RL-Games,
which stores it, and once it has collected enough steps, runs the actual PPO
math (gradient update) and starts the next batch.
```

**The one-sentence version:** `train.py` is just a setup script. The moment `runner.run(...)` is
called, **RL-Games is in the driver's seat** — it decides when to call `step()`, collects the
results, and periodically pauses to update the network. Our code (`ForgeEnv`/`FactoryEnv`) never
calls itself; it only *responds* when RL-Games calls `step()` or `reset()` on it.

---

## Walkthrough: what happens inside one `step()` call

`DirectRLEnv.step()` (`envs/direct_rl_env.py:386`) is the generic recipe. `FactoryEnv` and
`ForgeEnv` fill in the actual task-specific logic by overriding specific methods:

| Order | Method | Defined in | Plain English |
|---|---|---|---|
| 1 | `_pre_physics_step(action)` | `factory_env.py:208` | Takes the raw `[-1, 1]` action from the policy and turns it into a real target pose / gripper command |
| 2 | `_apply_action()` | `factory_env.py:254`, extended in `forge_env.py:152` | Converts that target into low-level controller signals (`generate_ctrl_signals`) — this is the OSC/IK controller box from the architecture diagram. Runs once per physics sub-step (`decimation` times) |
| — | *(physics steps forward here — this is Isaac Sim itself, not our code)* | | |
| 3 | `_get_dones()` | `factory_env.py:334` | Did this env succeed, fail, or time out? |
| 4 | `_get_rewards()` | `factory_env.py:408`, extended in `forge_env.py:234` | Factory computes the base reward dict (approach/align/progress/success); Forge adds the excessive-force penalty on top |
| 5 | `_reset_idx(env_ids)` | `factory_env.py:491`, extended in `forge_env.py:274` | Any env that just finished gets a fresh randomized start pose |
| 6 | `_get_observations()` | `factory_env.py:195`, extended in `forge_env.py:120` | Factory builds the base observation vector; Forge appends force/contact info |

This is exactly the skeleton from [§16 of the project doc](../../force_aware_peg_insertion_project.md#16-implementation-outline)
— good confirmation the real code matches what we were told to expect.

**Where PPO's actual math lives:** nowhere in these files. RL-Games' gradient update (the "L" box
in our [training loop diagram](../../docs/rl-primer.md)) lives inside the installed `rl_games`
package itself — worth knowing it exists, not worth reading line-by-line right now. We configure
it through `agents/rl_games_ppo_cfg.yaml`, we don't rewrite it.

---

## The two config files (the actual knobs)

- **`tasks/direct/forge/forge_env_cfg.py`** — task-specific numbers: reward weights, episode length
  (`episode_length_s = 10.0` for peg insert), controller gains, and — importantly — `obs_order`,
  the literal list defining what's in the observation vector.
- **`tasks/direct/forge/agents/rl_games_ppo_cfg.yaml`** — PPO's own hyperparameters: learning rate,
  network size, rollout horizon, etc. (project doc [§15](../../force_aware_peg_insertion_project.md#15-ppo-training-plan)).

**One detail worth flagging for later:** `ForgeEnvCfg.obs_order` already includes `ft_force` and
`force_threshold` by default — meaning the *upstream baseline* is already force-aware out of the
box. Our ablation's "Policy A: geometry-only" (project doc §12) will mean **removing** those two
entries, not adding force to a force-blind baseline. Worth keeping in mind once we get to forking
this for the ablation study.

---

## How to see this code live on the instance yourself

Everything above lives on the Brev box at `~/docker/isaac-sim/data/IsaacLab/...` (the exact same
path visible inside the container at `/isaac-sim/.local/share/ov/data/IsaacLab/...` — it's one
folder, bind-mounted into two places). Easiest way to browse/search/edit it live, without going
through me:

```bash
brev open isaac-sim code
```

This opens VS Code connected directly to the Brev instance over SSH (Remote-SSH). Open the folder
`~/docker/isaac-sim/data/IsaacLab` once it connects, and you have the real, live source tree —
searchable, with the same file layout as this reference copy. Any edits you make there are
real edits to the running install (unlike this local copy, which is just for reading).

If VS Code isn't installed locally, `brev shell isaac-sim` drops you into a terminal on the box,
where plain `cat`/`less`/`vim` on the same paths works too.
