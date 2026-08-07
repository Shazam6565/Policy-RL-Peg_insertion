1. brev login (locally) — reconnect the Mac to your Brev account (it's currently logged out).
  2. /isaac-status — confirm your existing isaac-sim instance is reachable, note it's the L40S box.
  This is the machine everything else in this project runs on.
  3. SSH in and inspect how Isaac Sim is actually installed (/home/ubuntu/IsaacSim, docker-based). You
  need to know whether Isaac Lab can attach to this install or needs its own container — check the
  Isaac Sim version string and Python version before assuming.
  4. Clone the matched Isaac Lab release onto the Brev box (not the Mac): git clone --branch
  release/3.0.0-beta2 https://github.com/isaac-sim/IsaacLab.git. Term: Isaac Lab ≠ Isaac Sim — Isaac
  Lab is the RL layer (tasks, gym-style envs, training scripts) built on top of the simulator.
  5. Run ./isaaclab.sh --install on the remote box — installs RL-Games, PyTorch, and Isaac Lab's own
  packages against that Isaac Sim.
  6. Run the two smoke tests from the doc §7: create_empty.py (proves the simulator launches) and
  list_envs.py (proves task registry loads) — confirm Isaac-Forge-PegInsert-Direct-v0 appears in the
  list. Don't move on until both pass.
  7. Run the tiny PPO smoke test (--num_envs 64 --max_iterations 20) and find where it writes
  checkpoints/logs. Term: PPO = the RL algorithm; num_envs = parallel simulated copies; iterations ≠
  environment steps.
  8. Play back the trained checkpoint (play.py) and watch/capture it rendering — this closes the loop
  from "training ran" to "I can see what it learned."
  9. Read the FORGE task source Isaac Lab just installed and write down (in your own words): the exact
  observation vector, action vector, reward terms, and termination conditions. No code changes yet —
  this is the doc's explicit "understand before you touch it" rule.
  10. git init this repo and connect it to a new GitHub remote — this becomes home for the
  force_peg_rl external project (generated via ./isaaclab.sh --new in the next step) plus your
  docs/results. This is the repo that stays project-specific; isaac-sim-workspace stays generic infra.



    Is this repo reusable elsewhere? No — it's the deliverable, not infra. Three separate repos will exist by the end:
  1. isaac-sim-workspace — reusable remote-driving tooling, already exists, not peg-insertion-specific.
  2. IsaacLab — upstream framework, lives only on the Brev box as a dependency, you don't commit into it.
  3. force_peg_rl (or however you name it, likely living inside/as this repo) — your actual portfolio work, pushed to its own GitHub repo.
