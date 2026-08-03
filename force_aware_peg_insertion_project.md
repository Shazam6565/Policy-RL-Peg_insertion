# Force-Aware Peg Insertion with Reinforcement Learning

## A portfolio project using NVIDIA Isaac Sim, Isaac Lab, a Franka arm, contact-force observations, domain randomization, and PPO

---

## 1. Project Summary

### Project title

**Force-Aware Peg Insertion: Robust Contact-Rich Manipulation with PPO**

### One-line portfolio description

Train and evaluate a Franka robot policy that inserts a peg into a socket while minimizing excessive contact forces and remaining robust to variations in pose, friction, mass, and controller behavior.

### Research question

> Does adding end-effector contact-force information and an excessive-force penalty improve the safety and robustness of a reinforcement-learning policy for peg insertion?

### Why this project is a good starting point

Peg insertion is a compact but meaningful contact-rich manipulation problem. The robot must:

1. Align the peg with the socket.
2. Correct lateral and angular errors.
3. Make controlled physical contact.
4. Avoid jamming or applying excessive force.
5. Complete the insertion despite uncertainty.

The project is small enough to complete as a first serious robot-learning portfolio project, while still exposing you to:

- Robot observations and action spaces
- Operational-space control
- Contact-force sensing
- Reward shaping
- Reinforcement learning with PPO
- Parallel simulation
- Curriculum learning
- Dynamics and pose randomization
- Policy evaluation and failure analysis
- Reproducible robotics experiments

---

## 2. Final Portfolio Deliverables

Your completed repository should contain:

1. A custom Isaac Lab task derived from the official peg-insertion environment.
2. A reproducible PPO training command.
3. Trained policy checkpoints.
4. An evaluation script that exports episode-level metrics.
5. An ablation comparing policies with and without force information.
6. A robustness evaluation under randomized physical parameters.
7. Training and evaluation plots.
8. A two-minute project video.
9. A four-to-six-page technical report.
10. A clear README with setup, training, evaluation, and results.

The project is complete only when another person can clone the repository, run the environment, train a policy, and reproduce your evaluation.

---

## 3. Project Scope

### Required scope

Build a policy that controls a Franka arm to insert a pre-grasped peg into a socket.

The first version should use simulation state rather than camera pixels. The policy receives robot state, relative peg/socket geometry, previous actions, and optionally force measurements.

### Non-goals for Version 1

Do not begin with:

- A humanoid robot
- A dexterous multi-fingered hand
- Raw RGB observations
- A vision-language-action model
- A custom world foundation model
- Real-hardware deployment
- Training a policy entirely from human demonstrations
- A complicated deformable object

Those can become follow-up projects. Version 1 should establish that you understand the complete RL workflow.

---

## 4. Starting Point

Use Isaac Lab's official contact-rich manipulation environments as the baseline.

Recommended baseline task:

```text
Isaac-Forge-PegInsert-Direct-v0
```

The FORGE peg-insertion environment is particularly useful because it already extends the basic Factory task with:

- End-effector force sensing
- An excessive-force penalty
- Dynamics randomization
- A success-prediction output

Your job is not merely to run the example. Your portfolio contribution is to:

1. Reproduce the baseline.
2. Fork the environment into your own external Isaac Lab project.
3. Document the Markov decision process.
4. Create controlled ablations.
5. Add evaluation tooling.
6. Analyze where and why the learned policy fails.
7. Present a defensible conclusion.

---

## 5. System Architecture

```text
                           +-------------------------+
                           |   Environment Reset     |
                           | pose / friction / mass  |
                           +------------+------------+
                                        |
                                        v
+--------------+       +----------------+----------------+
| Franka Robot | ----> | Isaac Sim / Isaac Lab Physics   |
| + Peg        |       | contact, rigid-body dynamics    |
+------+-------+       +----------------+----------------+
       |                                |
       | robot state                    | contact forces
       | peg/socket pose                | success/failure
       v                                v
+--------------------------------------------------------+
| Observation Builder                                    |
| q, dq, EE pose, relative pose, previous action, force  |
+--------------------------+-----------------------------+
                           |
                           v
                  +--------+---------+
                  | PPO Actor-Critic |
                  +--------+---------+
                           |
                           | delta EE pose action
                           v
                  +--------+---------+
                  | OSC / IK Control |
                  +--------+---------+
                           |
                           v
                     Robot motion

Episode logs -> TensorBoard / Weights & Biases -> evaluation CSV -> report
```

---

## 6. Software and Compute Requirements

### Recommended operating environment

- Ubuntu 24.04
- NVIDIA GPU with RT cores
- At least 16 GB GPU memory
- At least 32 GB system RAM
- At least 50 GB free SSD space
- Current NVIDIA driver supported by the selected Isaac Sim release

A cloud GPU workstation is acceptable. Since you already use Brev, a remote Ubuntu machine with an RTX-class or L-series GPU is a practical option.

Avoid A100 and H100 instances for Isaac Sim rendering because those accelerators do not provide the RT-core support expected by Isaac Sim.

### Required software

- NVIDIA Isaac Sim 6.0.x
- NVIDIA Isaac Lab 3.0-compatible release
- Python 3.12
- Git
- Git LFS
- `uv` or the environment manager recommended by the current Isaac Lab installation guide
- PyTorch, installed through the Isaac Lab setup
- RL-Games
- TensorBoard
- Optional: Weights & Biases
- Optional: FFmpeg for project videos
- Optional: Jupyter for offline result analysis

### Why RL-Games

The official Factory and FORGE contact-rich tasks provide PPO configurations for RL-Games. Begin with the existing configuration before changing algorithms or hyperparameters.

---

## 7. Installation and Baseline Verification

Because Isaac Lab and Isaac Sim change quickly, use a matched Isaac Lab release and Isaac Sim version. Do not combine arbitrary `main`, `develop`, and release branches.

Example setup pattern:

```bash
git clone https://github.com/isaac-sim/IsaacLab.git \
  --branch release/3.0.0-beta2

cd IsaacLab
```

Follow the release-specific Isaac Lab installation instructions for installing Isaac Sim through its supported Python environment.

Install Isaac Lab and its RL dependencies:

```bash
./isaaclab.sh --install
```

Verify that the simulator and Python environment work:

```bash
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --viz kit
```

List registered environments:

```bash
./isaaclab.sh -p scripts/environments/list_envs.py
```

Confirm that this task appears:

```text
Isaac-Forge-PegInsert-Direct-v0
```

### Baseline training smoke test

Start with a small number of environments:

```bash
./isaaclab.sh -p \
  scripts/reinforcement_learning/rl_games/train.py \
  --task Isaac-Forge-PegInsert-Direct-v0 \
  --num_envs 64 \
  --max_iterations 20
```

Once the smoke test succeeds, use headless execution and increase parallelism:

```bash
./isaaclab.sh -p \
  scripts/reinforcement_learning/rl_games/train.py \
  --task Isaac-Forge-PegInsert-Direct-v0 \
  --headless \
  --num_envs 1024 \
  --max_iterations 1000
```

If the run exceeds GPU memory, reduce `--num_envs` to 512 or 256. If GPU utilization is low and memory is available, test 2048 environments.

### Policy playback

```bash
./isaaclab.sh -p \
  scripts/reinforcement_learning/rl_games/play.py \
  --task Isaac-Forge-PegInsert-Direct-v0 \
  --checkpoint /absolute/path/to/checkpoint.pth \
  --num_envs 16
```

Do not continue until you can:

- Launch the environment
- Run a short training job
- Locate the checkpoint and logs
- Play the saved policy
- Explain every major observation, action, reward, and termination signal

---

## 8. Create Your Own External Project

Generate a project from the Isaac Lab template:

```bash
./isaaclab.sh --new
```

Choose:

```text
Project type: External
Workflow: Direct
RL framework: RL-Games
```

Suggested project name:

```text
force_peg_rl
```

Suggested task identifier:

```text
Shaurya-ForcePegInsert-Direct-v0
```

### Recommended repository structure

```text
force_peg_rl/
├── README.md
├── LICENSE
├── pyproject.toml
├── source/
│   └── force_peg_rl/
│       ├── config/
│       │   └── extension.toml
│       └── force_peg_rl/
│           ├── __init__.py
│           └── tasks/
│               └── direct/
│                   └── force_peg/
│                       ├── __init__.py
│                       ├── force_peg_env.py
│                       ├── force_peg_env_cfg.py
│                       └── agents/
│                           └── rl_games_ppo_cfg.yaml
├── scripts/
│   ├── evaluate_policy.py
│   ├── generate_eval_suite.py
│   └── plot_results.py
├── configs/
│   └── evaluation_suites.yaml
├── results/
│   ├── raw/
│   ├── tables/
│   ├── figures/
│   └── videos/
├── checkpoints/
└── docs/
    ├── experiment_log.md
    └── technical_report.md
```

Initially, copy the minimum required behavior from the official FORGE peg-insertion task. Preserve the upstream license and attribution.

---

## 9. Reinforcement-Learning Formulation

## 9.1 Episode definition

At the beginning of each episode:

1. The Franka arm holds the peg.
2. The peg starts above the socket.
3. The initial lateral offset, height, and orientation are randomized.
4. Selected physical and controller parameters may be randomized.
5. The policy acts until success, failure, or timeout.

A useful initial episode length is approximately 5–10 simulated seconds, depending on control frequency.

---

## 9.2 Observation space

Start with low-dimensional state observations.

### Robot proprioception

- Seven Franka joint positions
- Seven Franka joint velocities
- End-effector position
- End-effector orientation
- End-effector linear velocity
- End-effector angular velocity

### Task geometry

- Peg position relative to socket
- Peg orientation relative to socket
- Lateral alignment error
- Angular alignment error
- Current insertion depth
- Distance from peg tip to socket entrance

### Contact information

For the force-aware policy:

- End-effector contact force in the x, y, and z directions
- Force magnitude
- Optional force history or filtered force
- Optional contact-state boolean

### Temporal context

- Previous policy action
- Optional short action history
- Remaining episode time

### Observation normalization

Normalize each observation group to approximately comparable numerical ranges.

Examples:

- Position error: meters divided by a configured maximum error
- Joint velocity: divided by expected maximum velocity
- Force: clipped and divided by a safe force scale
- Orientation error: axis-angle representation scaled to approximately `[-1, 1]`

Do not feed an unbounded raw force value directly into the network.

---

## 9.3 Action space

Use end-effector delta commands rather than raw joint torques for the first version.

Recommended action:

```text
[Δx, Δy, Δz, Δroll, Δpitch, Δyaw]
```

The peg should remain grasped during the episode, so the gripper command can remain fixed.

Map normalized policy outputs from `[-1, 1]` into small task-space changes. Begin conservatively:

- Translation: a few millimeters per policy step
- Rotation: a few degrees per policy step

The delta pose is converted into robot motion through the existing operational-space or inverse-kinematics controller.

### Why this action space

It keeps the research focus on contact-aware insertion rather than forcing the policy to learn low-level torque control and contact strategy simultaneously.

---

## 9.4 Reward function

Use a reward with interpretable components.

A starting formulation is:

```text
reward =
    w_approach    * approach_reward
  + w_align_pos   * position_alignment_reward
  + w_align_rot   * orientation_alignment_reward
  + w_progress    * insertion_progress_reward
  + w_success     * success_bonus
  - w_force       * excessive_force_penalty
  - w_action      * action_magnitude_penalty
  - w_smooth      * action_rate_penalty
  - w_failure     * failure_penalty
```

### Suggested terms

#### Approach reward

Encourages the peg tip to move toward the socket entrance.

```text
r_approach = exp(-k_distance * distance_to_socket)
```

#### Position-alignment reward

Rewards low lateral error between the peg and socket axes.

```text
r_position = exp(-k_xy * lateral_error)
```

#### Orientation-alignment reward

Rewards angular alignment before insertion.

```text
r_orientation = exp(-k_rot * orientation_error)
```

#### Insertion-progress reward

Rewards downward progress after alignment.

```text
r_progress = current_depth - previous_depth
```

This should be a progress reward, not only an absolute-depth reward, so the policy receives feedback for useful movement.

#### Success bonus

Large positive reward when:

- Insertion depth exceeds the success threshold
- Lateral error is inside tolerance
- Orientation error is inside tolerance

#### Excessive-force penalty

Apply a penalty only above a soft safety threshold:

```text
force_excess = max(0, force_magnitude - safe_force_threshold)
r_force = force_excess²
```

A penalty on every contact force can discourage necessary contact. Penalizing force above a threshold distinguishes productive contact from aggressive contact.

#### Action penalty

Discourages large control commands.

#### Action-rate penalty

Discourages rapid changes between consecutive actions and supports smoother behavior.

### Reward-development rule

Log every reward component separately. Never debug only the total reward.

You should be able to answer:

- Which term dominates?
- Which terms increase before success?
- Which term prevents jamming?
- Does the force penalty prevent completion?
- Does the policy exploit any reward term?

---

## 9.5 Success criteria

Define success precisely.

Example:

```text
insertion_depth >= required_depth
and lateral_error <= xy_tolerance
and orientation_error <= rotation_tolerance
and peg_not_dropped
```

Do not define success using only peg height. A badly aligned or physically invalid state may satisfy a naive height threshold.

---

## 9.6 Failure and termination conditions

Terminate the episode when:

- The peg is successfully inserted
- The peg is dropped
- The peg leaves the valid workspace
- The robot exceeds joint or workspace limits
- Contact force remains above a hard threshold
- The socket or fixture becomes invalid
- The time limit is reached

Log a separate termination reason. A generic `done=True` is insufficient for meaningful evaluation.

Suggested labels:

```text
success
timeout
peg_dropped
force_limit
workspace_violation
joint_limit
jammed
other
```

A “jammed” label can initially be inferred when there is high sustained force, negligible insertion progress, and continued contact.

---

## 10. Curriculum Learning

Peg insertion can be difficult when all randomization is enabled from the beginning.

Use a staged curriculum.

### Stage 1: Easy alignment

- Peg starts almost centered
- Minimal yaw and tilt error
- Fixed friction and mass
- No controller randomization
- Large socket clearance if configurable

Goal: policy learns approach and insertion.

### Stage 2: Position uncertainty

- Increase x-y offset
- Increase start-height variation
- Retain small orientation error

Goal: policy learns corrective lateral motion.

### Stage 3: Orientation uncertainty

- Add yaw, pitch, and roll variation
- Reduce tolerances toward the target task

Goal: policy learns pose correction before insertion.

### Stage 4: Dynamics randomization

Randomize:

- Peg mass
- Peg/socket friction
- Controller gains
- Action latency or action dead zone, if supported
- Initial robot joint noise

Goal: policy becomes less dependent on one simulated configuration.

### Stage 5: Force-aware robustness

- Enable broader pose and dynamics randomization
- Penalize excessive force
- Evaluate under held-out parameter ranges

Goal: compare force-aware and non-force-aware policies.

Advance the curriculum based on a rolling success-rate threshold, not merely elapsed iterations.

---

## 11. Required Data

## 11.1 Training data

Pure reinforcement learning does not require a pre-collected labeled dataset.

Training data is generated online as rollouts:

```text
observation_t
action_t
reward_t
observation_t+1
termination_t
episode_metadata
```

PPO uses these on-policy trajectories to update the actor and critic.

### Data volume

Do not define progress only by “number of training iterations.” Record:

- Environment steps
- Number of episodes
- Successful episodes
- Wall-clock training time
- Simulated time
- Frames or control steps per second

---

## 11.2 Simulation assets

You need:

- Franka robot asset
- Peg asset
- Socket/fixture asset
- Table or work surface
- Contact-enabled collision geometry
- Material and physical-property definitions

The official Factory/FORGE task supplies suitable starting assets. For Version 1, do not create custom CAD unless the original task is already working.

---

## 11.3 Evaluation dataset

Create a deterministic evaluation suite separate from training randomization.

Example file:

```yaml
nominal:
  episodes: 500
  seeds: [1000, 1001, 1002]

pose_shift:
  episodes: 500
  xy_offset_range: [-0.012, 0.012]
  yaw_range_deg: [-12, 12]

low_friction:
  episodes: 500
  peg_friction_range: [0.2, 0.4]

high_friction:
  episodes: 500
  peg_friction_range: [0.9, 1.2]

mass_shift:
  episodes: 500
  mass_scale_range: [0.6, 1.5]

combined_ood:
  episodes: 1000
  wider_pose_range: true
  wider_dynamics_range: true
```

The exact numerical ranges should be adjusted to the asset scale and baseline configuration.

Store the episode seed and physical parameters so every failure can be replayed.

---

## 11.4 Episode-level logging schema

Write one row per episode:

```text
run_id
checkpoint
training_seed
evaluation_suite
episode_seed
success
termination_reason
episode_steps
episode_return
final_insertion_depth
max_contact_force
mean_contact_force
force_above_threshold_duration
lateral_error_final
orientation_error_final
peg_mass
peg_friction
socket_friction
initial_xy_offset
initial_orientation_error
```

Save the results as CSV or Parquet.

---

## 12. Experimental Design

Your central comparison should be an ablation study.

### Policy A: Geometry-only baseline

Observations:

- Robot state
- Peg/socket relative pose
- Previous action

Reward:

- Alignment
- Insertion progress
- Success
- Standard control penalties

No force observation and no excessive-force penalty.

### Policy B: Force observation only

Add:

- Contact-force vector
- Force magnitude

Do not add the excessive-force penalty.

Purpose: determine whether force measurements alone help the policy.

### Policy C: Force penalty only

Keep geometry-only observations.

Add the excessive-force penalty.

Purpose: determine whether safer behavior can emerge without explicitly observing force.

### Policy D: Force-aware policy

Add both:

- Force observations
- Excessive-force penalty

Purpose: test the complete hypothesis.

### Minimum experimental standard

- Train each policy with at least three random seeds.
- Evaluate every checkpoint on the same deterministic episode seeds.
- Report mean and standard deviation.
- Report success and force metrics together.
- Include representative failure videos.

One visually impressive episode is not evidence.

---

## 13. Primary Metrics

### Task performance

- Success rate
- Median completion steps
- Mean episode return
- Final insertion depth
- Timeout rate
- Drop rate
- Jam rate

### Contact quality

- Mean contact-force magnitude
- Peak contact force
- 95th-percentile contact force
- Time above the soft-force threshold
- Hard-force-limit termination rate

### Robustness

- Success under nominal conditions
- Success under each randomization suite
- Out-of-distribution success
- Generalization gap:

```text
nominal success - OOD success
```

### Efficiency

- Training wall-clock time
- Environment steps to reach target success
- Simulation throughput
- GPU memory utilization

---

## 14. Portfolio Success Targets

Treat these as project goals, not guaranteed results.

A strong Version 1 result would demonstrate:

- At least 80% success under nominal evaluation
- At least 60% success under the combined randomized evaluation
- A measurable reduction in peak or high-percentile contact force
- No more than a five-percentage-point success decrease relative to the unsafe baseline
- Reproducible results across at least three training seeds
- A clear failure taxonomy rather than only aggregate metrics

A negative result can still be valuable. For example:

> Force observations reduced jamming under pose perturbations but produced no meaningful improvement when the policy was already given exact relative geometry.

That is a legitimate experimental conclusion.

---

## 15. PPO Training Plan

Begin with the official RL-Games PPO configuration supplied with the task.

Do not tune ten hyperparameters before reproducing the baseline.

### Parameters to inspect

- Actor and critic hidden-layer sizes
- Learning rate
- Discount factor
- GAE lambda
- PPO clipping coefficient
- Entropy coefficient
- Rollout horizon
- Number of mini-batches
- Number of learning epochs
- Observation and value normalization
- Gradient clipping
- Desired KL threshold

### Reasonable starting architecture

For low-dimensional observations:

```text
Actor MLP:  256 -> 256 -> 128
Critic MLP: 256 -> 256 -> 128
Activation: ELU or ReLU
```

Use the official configuration as the initial source of truth.

### Training sequence

1. Train the official baseline unchanged.
2. Fork the configuration.
3. Add your logging.
4. Train Policy A.
5. Verify repeatability.
6. Add force observations.
7. Train Policy B.
8. Add force penalty.
9. Train Policy C.
10. Train Policy D.
11. Run all deterministic evaluations.
12. Freeze configurations before final comparison.

---

## 16. Implementation Outline

A direct Isaac Lab RL environment generally needs logic corresponding to:

```python
class ForcePegEnv(DirectRLEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        # Spawn robot, peg, socket, table, sensors, and cloned environments.
        pass

    def _pre_physics_step(self, actions):
        # Clip and store normalized policy actions.
        pass

    def _apply_action(self):
        # Convert delta end-effector commands into controller targets.
        pass

    def _get_observations(self):
        # Build actor and optional critic observations.
        pass

    def _get_rewards(self):
        # Compute and log each reward component.
        pass

    def _get_dones(self):
        # Compute success, failure, timeout, and termination reason.
        pass

    def _reset_idx(self, env_ids):
        # Reset robot, object poses, randomization, and episode buffers.
        pass
```

The exact method names and interfaces must match the Isaac Lab release used by your project.

### Separation of responsibilities

Keep these separate:

- Environment state extraction
- Observation construction
- Reward computation
- Randomization
- Termination
- Evaluation logging

Do not place the entire project inside one large Python file.

---

## 17. Evaluation Script Requirements

Your evaluation script should:

1. Load a checkpoint.
2. Disable training-time stochasticity.
3. Apply a named evaluation suite.
4. Run deterministic episode seeds.
5. Capture success and failure metrics.
6. Save per-episode data.
7. Optionally record selected videos.
8. Print an aggregate summary.

Example interface:

```bash
python scripts/evaluate_policy.py \
  --task Shaurya-ForcePegInsert-Direct-v0 \
  --checkpoint checkpoints/policy_d_seed_1.pth \
  --suite combined_ood \
  --episodes 1000 \
  --output results/raw/policy_d_seed_1_combined_ood.csv
```

Example console output:

```text
Evaluation suite: combined_ood
Episodes: 1000
Success rate: 67.4%
Median completion steps: 92
Peak-force p95: 34.8 N
Force-limit termination rate: 3.1%
Jam rate: 11.7%
```

---

## 18. Failure Taxonomy

Review failed episodes and assign categories.

### Misalignment

The peg reaches the socket but retains excessive lateral or angular error.

### Rim collision

The peg repeatedly collides with the socket entrance.

### Jamming

The policy applies sustained force without making insertion progress.

### Oscillation

The policy alternates actions around the target rather than converging.

### Over-cautious behavior

The force penalty prevents the policy from making necessary contact.

### Aggressive insertion

The policy succeeds but uses unnecessarily large force.

### Distribution-shift failure

The policy works under nominal parameters but fails when mass, friction, or initial pose changes.

### Controller exploitation

The policy exploits a simulator or controller artifact rather than learning physically meaningful behavior.

Your report should contain at least one video or plot for each common failure category.

---

## 19. Common Debugging Problems

### Training return increases but success remains zero

Possible causes:

- Dense shaping rewards dominate the success objective.
- Success criteria are unreachable or incorrectly computed.
- The policy learns to hover close to the socket.
- Insertion progress is measured in the wrong coordinate frame.

Action:

- Plot each reward term.
- Manually place the peg in a successful pose.
- Unit-test success detection.
- Add a large but not overwhelming success bonus.

### Policy slams the peg into the socket

Possible causes:

- No action-rate penalty
- Excessive action scale
- Force penalty too weak
- Control frequency too low
- Force observations not normalized

### Policy refuses to make contact

Possible causes:

- Force penalty activates below normal insertion forces
- Force term dominates all positive rewards
- Hard-force termination is too strict

### Robot or objects become unstable

Possible causes:

- Invalid collision geometry
- Aggressive controller gains
- Large simulation time step
- Excessive action scale
- Spawned objects overlap at reset
- Inconsistent mass or inertia settings

### Policy overfits to one pose

Possible causes:

- Insufficient reset randomization
- Curriculum never advances
- Policy receives an absolute coordinate that leaks a fixed solution
- Evaluation seeds overlap with training conditions

---

## 20. Six-Week Execution Plan

## Week 1: Installation and baseline reproduction

Deliverables:

- Working Isaac Sim and Isaac Lab environment
- Official FORGE peg-insertion task launches
- PPO smoke test completes
- Baseline checkpoint plays
- Notes explaining observations, actions, reward, and termination

Do not modify the environment during the first two days.

## Week 2: External project and instrumentation

Deliverables:

- External repository generated
- Custom task registered
- Upstream baseline behavior reproduced
- Reward components logged separately
- Episode termination reasons logged
- Training curves visible in TensorBoard or W&B

## Week 3: Geometry-only baseline

Deliverables:

- Policy A configuration
- Three training seeds
- Deterministic nominal evaluation
- First result table
- Failure videos

## Week 4: Force-aware variants

Deliverables:

- Force observations normalized and added
- Force penalty implemented
- Policies B, C, and D trained
- Contact-force distributions plotted

## Week 5: Robustness and domain randomization

Deliverables:

- Evaluation suites finalized
- Pose, mass, and friction tests executed
- Generalization gaps calculated
- Representative out-of-distribution failures documented

## Week 6: Portfolio packaging

Deliverables:

- README completed
- Training and evaluation commands verified
- Results table and plots finalized
- Two-minute demo video
- Four-to-six-page technical report
- Portfolio page published
- Repository cleaned and tagged as `v1.0`

---

## 21. Technical Report Outline

### 1. Abstract

State the problem, method, main experiment, and result.

### 2. Motivation

Explain why contact-aware insertion is difficult and why force information may help.

### 3. Task and environment

Describe:

- Franka robot
- Peg/socket geometry
- Simulator
- Controller
- Observation space
- Action space
- Randomization

### 4. Reinforcement-learning method

Describe PPO and the reward function.

### 5. Experimental design

Describe policies A–D, seeds, training steps, and evaluation suites.

### 6. Results

Include:

- Success-rate table
- Contact-force table
- Robustness plot
- Learning curves
- Failure distribution

### 7. Failure analysis

Explain what the policy does incorrectly and why.

### 8. Limitations

Examples:

- State observations rather than vision
- Simulation-only evaluation
- Simplified gripper interaction
- No true tactile sensor
- Limited object geometries

### 9. Next steps

Propose one focused extension.

---

## 22. README Results Table Template

| Policy | Force Obs. | Force Penalty | Nominal Success | OOD Success | Peak Force p95 | Jam Rate |
|---|---:|---:|---:|---:|---:|---:|
| A: Geometry baseline | No | No | TBD | TBD | TBD | TBD |
| B: Force observation | Yes | No | TBD | TBD | TBD | TBD |
| C: Force penalty | No | Yes | TBD | TBD | TBD | TBD |
| D: Force-aware | Yes | Yes | TBD | TBD | TBD | TBD |

Do not fill the table with expected values. Only report measured results.

---

## 23. Video Storyboard

Your two-minute project video should show:

### 0:00–0:15 — Problem

Show a failed peg insertion and briefly explain jamming and excessive force.

### 0:15–0:35 — Environment

Show parallel Isaac Lab environments and identify the Franka, peg, socket, and contact-force signal.

### 0:35–0:55 — RL formulation

Display observations, actions, and major reward terms.

### 0:55–1:20 — Training

Show PPO learning curves and a short montage of policy improvement.

### 1:20–1:45 — Results

Compare the baseline and force-aware policy under the same perturbation.

### 1:45–2:00 — Conclusion

State what improved, what did not, and the next research question.

Include failures. A failure-analysis video is more credible than a success-only montage.

---

## 24. Recommended Follow-Up Projects

After Version 1, choose only one extension.

### Extension A: Vision-based peg insertion

Replace privileged relative-pose observations with RGB or depth features.

Research question:

> How much robustness is lost when the policy must infer alignment from visual input?

### Extension B: Demonstrations plus RL fine-tuning

Collect scripted or teleoperated demonstrations, train behavior cloning, and fine-tune with PPO.

Research question:

> Can demonstrations reduce the environment steps required to learn contact-rich insertion?

### Extension C: Learned contact prediction

Train an auxiliary model to predict:

- Future contact force
- Jam probability
- Success probability

Use the prediction as an observation or safety signal.

Research question:

> Can short-horizon contact prediction improve recovery before a jam occurs?

### Extension D: Multiple peg/socket geometries

Train across a family of assemblies and evaluate unseen geometries.

Research question:

> Which task and geometry representations support generalization across insertion problems?

### Extension E: Sim-to-real deployment

Deploy the policy to a physical Franka or comparable arm after adding:

- Real sensor calibration
- Safety constraints
- Latency randomization
- Observation noise
- Conservative action limits
- Real force-torque sensing

This should be attempted only with supervised laboratory access and an appropriate hardware-safety process.

---

## 25. Final Definition of Done

The project is done when all boxes are checked:

### Environment

- [ ] Custom task runs independently of the Isaac Lab source tree
- [ ] Reset randomization works
- [ ] Success and failures are correctly detected
- [ ] Contact forces are logged and validated
- [ ] Fixed seeds reproduce the same initial conditions

### RL

- [ ] PPO training launches from one documented command
- [ ] Checkpoints and configurations are versioned
- [ ] At least three training seeds are completed
- [ ] Reward components are individually logged
- [ ] Training can be resumed from a checkpoint

### Evaluation

- [ ] Nominal and randomized suites are defined
- [ ] Every policy is tested on identical episode seeds
- [ ] Per-episode results are exported
- [ ] Success and force metrics are reported together
- [ ] Common failures are categorized

### Portfolio

- [ ] Public README
- [ ] Architecture diagram
- [ ] Results table
- [ ] Training curves
- [ ] Failure analysis
- [ ] Two-minute video
- [ ] Technical report
- [ ] Clear limitations and next steps

---

## 26. The First Three Actions

1. Install or verify the matched Isaac Sim and Isaac Lab environment.
2. Run `Isaac-Forge-PegInsert-Direct-v0` with a short PPO smoke test.
3. Read the official FORGE task code and write down the exact observation vector, action vector, reward terms, and reset randomization before changing anything.

Do not start by tuning PPO. First make the existing system observable, reproducible, and understandable.
