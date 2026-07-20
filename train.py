import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configuration import GreenfieldEnergyEnv
from dqn_agent import DQNAgent


N_EPISODES          = 500
TARGET_UPDATE_FREQ  = 10
SAVE_EVERY          = 50
PRINT_EVERY         = 10
LR_DECAY_EVERY      = 100
MODEL_SAVE_DIR      = "models"
RESULTS_SAVE_DIR    = "results"
PLOTS_DIR           = "plots"


BASELINE_FUEL_L     = 149_693.0


def _moving_average(values: np.ndarray, window: int = 20) -> np.ndarray:
    if len(values) < window:
        return values.copy()
    kernel = np.ones(window) / window
    padded = np.concatenate([np.full(window - 1, values[0]), values])
    return np.convolve(padded, kernel, mode="valid")


def _progress_bar(current: int, total: int, width: int = 30) -> str:
    filled = int(width * current / total)
    bar    = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {100*current/total:5.1f}%  ep {current}/{total}"


def _save_training_plots(
    rewards:       np.ndarray,
    fuels:         np.ndarray,
    reliabilities: np.ndarray,
    epsilons:      np.ndarray,
    losses:        np.ndarray,
) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    episodes = np.arange(1, len(rewards) + 1)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))


    ax = axes[0, 0]
    ax.plot(episodes, rewards, color="steelblue", alpha=0.3, linewidth=0.7)
    ax.plot(episodes, _moving_average(rewards), color="steelblue", linewidth=2.0,
            label="Moving avg (20 ep)")
    ax.set_title("Reward Convergence")
    ax.set_ylabel("Total Episode Reward")
    ax.legend(); ax.grid(True, alpha=0.3)


    ax = axes[0, 1]
    ax.plot(episodes, reliabilities, color="seagreen", alpha=0.3, linewidth=0.7)
    ax.plot(episodes, _moving_average(reliabilities), color="seagreen", linewidth=2.0,
            label="Moving avg (20 ep)")
    ax.axhline(69.90, color="tomato",  linestyle="--", linewidth=1.2,
               label="Rule-based baseline (69.90%)")
    ax.axhline(90.0,  color="darkgreen", linestyle="--", linewidth=1.2,
               label="DQN target (90%)")
    ax.set_title("Reliability Index")
    ax.set_ylabel("Energy Reliability (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)


    ax = axes[1, 0]
    ax.plot(episodes, fuels, color="firebrick", alpha=0.3, linewidth=0.7)
    ax.plot(episodes, _moving_average(fuels), color="firebrick", linewidth=2.0,
            label="Moving avg (20 ep)")
    ax.axhline(BASELINE_FUEL_L, color="darkorange", linestyle="--", linewidth=1.2,
               label=f"Rule-based baseline ({BASELINE_FUEL_L:,.0f} L)")
    ax.set_title("Fuel Consumption (Obj 6: must stay ≤ baseline)")
    ax.set_ylabel("Total Fuel (L)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)


    ax = axes[1, 1]
    ax.plot(episodes, epsilons, color="darkorchid", linewidth=1.5)
    ax.set_title("Exploration Rate (ε) Decay")
    ax.set_ylabel("Epsilon")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)


    ax = axes[2, 0]
    valid_losses = losses[losses > 0]
    valid_eps    = episodes[losses > 0]
    if len(valid_losses) > 0:
        ax.plot(valid_eps, valid_losses, color="gray", alpha=0.4, linewidth=0.7)
        ax.plot(valid_eps, _moving_average(valid_losses), color="gray", linewidth=2.0,
                label="Moving avg")
    ax.set_title("Mean TD Loss")
    ax.set_ylabel("Smooth L1 Loss")
    ax.legend(); ax.grid(True, alpha=0.3)


    ax = axes[2, 1]
    sc = ax.scatter(fuels, reliabilities, c=episodes, cmap="viridis",
                    alpha=0.5, s=8)
    ax.axvline(BASELINE_FUEL_L, color="darkorange", linestyle="--", linewidth=1.2,
               label="Fuel baseline")
    ax.axhline(90.0, color="darkgreen", linestyle="--", linewidth=1.2,
               label="90% target")
    ax.axhline(69.90, color="tomato", linestyle="--", linewidth=1.2,
               label="Baseline reliability")
    plt.colorbar(sc, ax=ax, label="Episode")
    ax.set_title("Reliability vs Fuel (Objective 6 frontier)")
    ax.set_xlabel("Fuel consumed (L)")
    ax.set_ylabel("Reliability (%)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("Episode" if ax.get_xlabel() == "" else ax.get_xlabel())

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "training_dashboard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def train() -> None:
    os.makedirs(MODEL_SAVE_DIR,   exist_ok=True)
    os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,        exist_ok=True)

    print("=" * 68)
    print("  Greenfield University — DQN Energy Management Training")
    print("=" * 68)
    print(f"  Episodes         : {N_EPISODES}")
    print(f"  Target update    : every {TARGET_UPDATE_FREQ} episodes")
    print(f"  Checkpoint save  : every {SAVE_EVERY} episodes")
    print(f"  Baseline fuel    : {BASELINE_FUEL_L:,.0f} L  (Objective 6 constraint)")
    print("=" * 68)

    env   = GreenfieldEnergyEnv()
    agent = DQNAgent(
        state_size  = env.state_size,
        action_size = env.action_size,
    )

    all_rewards        = np.zeros(N_EPISODES, dtype=np.float64)
    all_fuels          = np.zeros(N_EPISODES, dtype=np.float64)
    all_reliabilities  = np.zeros(N_EPISODES, dtype=np.float64)
    all_shedding       = np.zeros(N_EPISODES, dtype=np.int64)
    all_losses         = np.zeros(N_EPISODES, dtype=np.float64)
    all_epsilons       = np.zeros(N_EPISODES, dtype=np.float64)

    best_reliability   = 0.0
    best_model_path    = os.path.join(MODEL_SAVE_DIR, "dqn_best.pth")

    train_start = time.time()

    for ep in range(1, N_EPISODES + 1):
        state            = env.reset()
        total_reward     = 0.0
        total_fuel       = 0.0
        shedding_count   = 0
        total_demand_kw  = 0.0
        total_served_kw  = 0.0
        episode_losses   = []

        for _ in range(env.n_timesteps):
            action                         = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.store_experience(state, action, reward, next_state, float(done))
            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)

            total_reward    += reward
            total_fuel      += sum(info["fuel_consumed_per_source"].values())
            total_demand_kw += info["total_demand_kw"]
            total_served_kw += info["load_served_kw"]
            if info["load_shedding_occurred"]:
                shedding_count += 1

            state = next_state
            if done:
                break

        agent.decay_epsilon()
        agent._episode_count += 1

        if agent._episode_count % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        if agent._episode_count % LR_DECAY_EVERY == 0:
            agent.step_scheduler()

        mean_loss   = float(np.mean(episode_losses)) if episode_losses else 0.0
        reliability = (100.0 * total_served_kw / total_demand_kw
                       if total_demand_kw > 0 else 0.0)

        idx = ep - 1
        all_rewards[idx]       = total_reward
        all_fuels[idx]         = total_fuel
        all_reliabilities[idx] = reliability
        all_shedding[idx]      = shedding_count
        all_losses[idx]        = mean_loss
        all_epsilons[idx]      = agent.epsilon


        if reliability > best_reliability and total_fuel <= BASELINE_FUEL_L:
            best_reliability = reliability
            agent.save(best_model_path)


        if ep % SAVE_EVERY == 0:
            ckpt = os.path.join(MODEL_SAVE_DIR, f"checkpoint_ep{ep}.pth")
            agent.save(ckpt)
            print(f"  [checkpoint] {ckpt}")

        if ep % PRINT_EVERY == 0:
            fuel_flag = "✓" if total_fuel <= BASELINE_FUEL_L else "✗ OVER"
            print(
                f"{_progress_bar(ep, N_EPISODES)} | "
                f"Reward {total_reward:>10.1f} | "
                f"Fuel {total_fuel:>8.1f} L [{fuel_flag}] | "
                f"Rel {reliability:>6.2f}% | "
                f"ε {agent.epsilon:.4f} | "
                f"Loss {mean_loss:.5f}"
            )


    final_path = os.path.join(MODEL_SAVE_DIR, "dqn_final.pth")
    agent.save(final_path)
    print(f"\n  Final model  → {final_path}")
    print(f"  Best model   → {best_model_path}  (reliability={best_reliability:.2f}%)")

    np.savez(
        os.path.join(RESULTS_SAVE_DIR, "training_metrics.npz"),
        rewards       = all_rewards,
        fuels         = all_fuels,
        reliabilities = all_reliabilities,
        shedding      = all_shedding,
        losses        = all_losses,
        epsilons      = all_epsilons,
    )
    print(f"  Metrics      → {RESULTS_SAVE_DIR}/training_metrics.npz")

    print("\nGenerating training dashboard …")
    _save_training_plots(all_rewards, all_fuels, all_reliabilities,
                         all_epsilons, all_losses)

    elapsed   = time.time() - train_start
    hrs, rem  = divmod(int(elapsed), 3600)
    mins, sec = divmod(rem, 60)
    print(f"\n  Total training time : {hrs:02d}h {mins:02d}m {sec:02d}s")


    print("\n" + "=" * 68)
    print("  Training Summary:  Episode 1  vs  Final Episode")
    print("=" * 68)

    def _row(label, v1, v2, fmt=".2f"):
        delta = v2 - v1
        sign  = "+" if delta >= 0 else ""
        print(f"  {label:<30} {v1:>12{fmt}}   {v2:>12{fmt}}   {sign}{delta:>10{fmt}}")

    print(f"  {'Metric':<30} {'Ep 1':>12}   {'Ep ' + str(N_EPISODES):>12}   {'Change':>10}")
    print("  " + "-" * 66)
    _row("Total Reward",             all_rewards[0],       all_rewards[-1],       ".1f")
    _row("Fuel Consumed (L)",        all_fuels[0],         all_fuels[-1],         ".1f")
    _row("Reliability % (energy)",   all_reliabilities[0], all_reliabilities[-1], ".2f")
    _row("Load Shedding Events",     float(all_shedding[0]), float(all_shedding[-1]), ".0f")
    _row("Mean TD Loss",             all_losses[0],        all_losses[-1],        ".5f")
    _row("Epsilon",                  all_epsilons[0],      all_epsilons[-1],      ".4f")
    print("=" * 68)

    meets_obj6 = all_fuels[-1] <= BASELINE_FUEL_L
    print(f"\n  Objective 6 (fuel ≤ baseline): "
          f"{'✓ MET' if meets_obj6 else '✗ NOT MET'}  "
          f"({all_fuels[-1]:,.0f} L vs {BASELINE_FUEL_L:,.0f} L baseline)")
    print(f"  Best reliability achieved     : {best_reliability:.2f}%  "
          f"({'≥ 90%' if best_reliability >= 90.0 else '< 90% — continue training'})\n")


if __name__ == "__main__":
    train()
