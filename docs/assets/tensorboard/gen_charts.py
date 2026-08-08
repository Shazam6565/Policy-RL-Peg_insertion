"""Generate static chart images from real TensorBoard event data for the
baseline (Shaurya-ForcePegInsert-Direct-v0) and Policy A
(Shaurya-ForcePegInsert-PolicyA-Direct-v0) smoke-test runs, for embedding in
docs/tensorboard_guide.md.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

BASELINE_DIR = "/home/as22cq/Projects/Policy-RL-Peg_insertion/force_peg_rl/logs/rl_games/Forge/2026-08-08_01-20-14/summaries"
POLICYA_DIR = "/home/as22cq/Projects/Policy-RL-Peg_insertion/force_peg_rl/logs/rl_games/Forge/2026-08-08_01-34-29/summaries"
import os
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

BLUE = "#2a78d6"    # baseline (force-aware, Policy D-like)
ORANGE = "#eb6834"  # Policy A (geometry-only)
GRID = "#d8d8d4"
TEXT = "#52514e"
BG = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT,
    "text.color": "#0b0b0b",
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "grid.color": GRID,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.dpi": 150,
})


def load(path):
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()
    return ea


baseline = load(BASELINE_DIR)
policya = load(POLICYA_DIR)


def series(ea, tag):
    return [(s.step, s.value) for s in ea.Scalars(tag)]


def style_ax(ax, title, ylabel, xlabel="Iteration"):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linewidth=0.6, alpha=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def single_line(tag, title, ylabel, name, ea=baseline, color=BLUE, label=None):
    data = series(ea, tag)
    xs = [d[0] for d in data]
    ys = [d[1] for d in data]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=4, label=label)
    style_ax(ax, title, ylabel)
    if label:
        ax.legend(frameon=False)
    save(fig, name)


def compare_line(tag, title, ylabel, name):
    bdata = series(baseline, tag)
    pdata = series(policya, tag)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([d[0] for d in bdata], [d[1] for d in bdata], color=BLUE, linewidth=2,
             marker="o", markersize=4, label="Baseline (force-aware)")
    ax.plot([d[0] for d in pdata], [d[1] for d in pdata], color=ORANGE, linewidth=2,
             marker="o", markersize=4, label="Policy A (geometry-only)")
    style_ax(ax, title, ylabel)
    ax.legend(frameon=False)
    save(fig, name)


# 1. Episode/Metrics/success_rate
compare_line("Episode/Metrics/success_rate", "Episode success rate", "Fraction of envs succeeded", "01_success_rate")

# 2. episode_lengths/iter
compare_line("episode_lengths/iter", "Episode length", "Steps per episode", "02_episode_lengths")

# 3. rewards/iter (total scaled reward)
compare_line("rewards/iter", "Total reward (scaled)", "Reward", "03_rewards")

# 4. shaped_rewards/iter
compare_line("shaped_rewards/iter", "Shaped reward (pre-clip)", "Reward", "04_shaped_rewards")

# 5. logs_rew_contact_penalty/iter -- THE key ablation proof
compare_line("logs_rew_contact_penalty/iter", "Contact penalty (raw, unscaled)", "Penalty magnitude", "05_contact_penalty")

# 6. logs_rew_kp_baseline / kp_coarse / kp_fine -- keypoint distance reward family
fig, ax = plt.subplots(figsize=(7, 4))
for tag, color, label in [
    ("logs_rew_kp_baseline/iter", BLUE, "kp_baseline"),
    ("logs_rew_kp_coarse/iter", ORANGE, "kp_coarse"),
    ("logs_rew_kp_fine/iter", "#1baf7a", "kp_fine"),
]:
    data = series(baseline, tag)
    ax.plot([d[0] for d in data], [d[1] for d in data], linewidth=2, marker="o", markersize=4, color=color, label=label)
style_ax(ax, "Keypoint-distance rewards (baseline run)", "Reward")
ax.legend(frameon=False)
save(fig, "06_keypoint_rewards")

# 7. logs_rew_action_penalty_ee / action_penalty_asset / action_grad_penalty
fig, ax = plt.subplots(figsize=(7, 4))
for tag, color, label in [
    ("logs_rew_action_penalty_ee/iter", BLUE, "action_penalty_ee"),
    ("logs_rew_action_penalty_asset/iter", ORANGE, "action_penalty_asset"),
    ("logs_rew_action_grad_penalty/iter", "#1baf7a", "action_grad_penalty"),
]:
    data = series(baseline, tag)
    ax.plot([d[0] for d in data], [d[1] for d in data], linewidth=2, marker="o", markersize=4, color=color, label=label)
style_ax(ax, "Action penalties (baseline run)", "Penalty magnitude")
ax.legend(frameon=False)
save(fig, "07_action_penalties")

# 8. logs_rew_success_pred_error
compare_line("logs_rew_success_pred_error/iter", "Success-prediction error", "|true - predicted|", "08_success_pred_error")

# 9. logs_rew_curr_engaged / curr_success
fig, ax = plt.subplots(figsize=(7, 4))
for tag, color, label in [
    ("logs_rew_curr_engaged/iter", BLUE, "curr_engaged"),
    ("logs_rew_curr_success/iter", ORANGE, "curr_success"),
]:
    data = series(baseline, tag)
    ax.plot([d[0] for d in data], [d[1] for d in data], linewidth=2, marker="o", markersize=4, color=color, label=label)
style_ax(ax, "Engagement / success indicator (baseline run)", "Fraction of envs")
ax.legend(frameon=False)
save(fig, "09_engaged_success")

# 10. success_times/iter
single_line("success_times/iter", "Mean success time (baseline run)", "Steps until first success", "10_success_times", ea=baseline, color=BLUE)

# 11. early_term_precision / recall (per threshold) -- baseline only, multi-line
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for thresh, color in zip(["0.5", "0.6", "0.7", "0.8"], [BLUE, ORANGE, "#1baf7a", "#eda100"]):
    data = series(baseline, f"early_term_precision/{thresh}/iter")
    axes[0].plot([d[0] for d in data], [d[1] for d in data], linewidth=2, marker="o", markersize=4, color=color, label=f"threshold {thresh}")
style_ax(axes[0], "Early-termination precision", "Precision")
axes[0].legend(frameon=False, fontsize=9)
for thresh, color in zip(["0.5", "0.6", "0.7", "0.8", "0.9"], [BLUE, ORANGE, "#1baf7a", "#eda100", "#e87ba4"]):
    data = series(baseline, f"early_term_recall/{thresh}/iter")
    axes[1].plot([d[0] for d in data], [d[1] for d in data], linewidth=2, marker="o", markersize=4, color=color, label=f"threshold {thresh}")
style_ax(axes[1], "Early-termination recall", "Recall")
axes[1].legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/11_early_term_precision_recall.png", bbox_inches="tight")
plt.close(fig)
print("wrote 11_early_term_precision_recall")

# 12. early_term_delay_all / delay_correct
fig, ax = plt.subplots(figsize=(7, 4))
for thresh, color in zip(["0.5", "0.6"], [BLUE, ORANGE]):
    data = series(baseline, f"early_term_delay_all/{thresh}/iter")
    ax.plot([d[0] for d in data], [d[1] for d in data], linewidth=2, marker="o", markersize=4, color=color, label=f"delay_all @ {thresh}")
for thresh, color in zip(["0.5", "0.6"], ["#1baf7a", "#eda100"]):
    data = series(baseline, f"early_term_delay_correct/{thresh}/iter")
    ax.plot([d[0] for d in data], [d[1] for d in data], linewidth=2, marker="o", markersize=4, linestyle="--", color=color, label=f"delay_correct @ {thresh}")
style_ax(ax, "Success-prediction delay (baseline run)", "Steps (+ = late, - = early)")
ax.axhline(0, color=TEXT, linewidth=1, alpha=0.4)
ax.legend(frameon=False, fontsize=9)
save(fig, "12_early_term_delay")

# 13. losses/a_loss, c_loss
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for tag, color, label, ax in [
    ("losses/a_loss", BLUE, "actor (policy) loss", axes[0]),
    ("losses/c_loss", ORANGE, "critic (value) loss", axes[1]),
]:
    data = series(baseline, tag)
    ax.plot([d[0] for d in data], [d[1] for d in data], linewidth=2, marker="o", markersize=4, color=color)
    style_ax(ax, label, "Loss", xlabel="Frames")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/13_losses.png", bbox_inches="tight")
plt.close(fig)
print("wrote 13_losses")

# 14. losses/entropy, bounds_loss
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
data = series(baseline, "losses/entropy")
axes[0].plot([d[0] for d in data], [d[1] for d in data], linewidth=2, marker="o", markersize=4, color=BLUE)
style_ax(axes[0], "Policy entropy", "Entropy (nats)", xlabel="Frames")
data = series(baseline, "losses/bounds_loss")
axes[1].plot([d[0] for d in data], [d[1] for d in data], linewidth=2, marker="o", markersize=4, color=ORANGE)
style_ax(axes[1], "Action-bounds loss", "Loss", xlabel="Frames")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/14_entropy_bounds.png", bbox_inches="tight")
plt.close(fig)
print("wrote 14_entropy_bounds")

# 15. info/kl, info/last_lr
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
data = series(baseline, "info/kl")
axes[0].plot([d[0] for d in data], [d[1] for d in data], linewidth=2, marker="o", markersize=4, color=BLUE)
style_ax(axes[0], "KL divergence (policy update)", "KL", xlabel="Frames")
data = series(baseline, "info/last_lr")
axes[1].plot([d[0] for d in data], [d[1] for d in data], linewidth=2, marker="o", markersize=4, color=ORANGE)
style_ax(axes[1], "Adaptive learning rate", "Learning rate", xlabel="Frames")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/15_kl_lr.png", bbox_inches="tight")
plt.close(fig)
print("wrote 15_kl_lr")

# 16. performance/step_fps, step_time
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
data = series(baseline, "performance/step_fps")
axes[0].plot([d[0] for d in data], [d[1] for d in data], linewidth=2, marker="o", markersize=4, color=BLUE)
style_ax(axes[0], "Simulation throughput", "Steps/sec (FPS)", xlabel="Frames")
data = series(baseline, "performance/step_time")
axes[1].plot([d[0] for d in data], [d[1] for d in data], linewidth=2, marker="o", markersize=4, color=ORANGE)
style_ax(axes[1], "Per-step wall time", "Milliseconds", xlabel="Frames")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/16_performance.png", bbox_inches="tight")
plt.close(fig)
print("wrote 16_performance")

print("DONE")
