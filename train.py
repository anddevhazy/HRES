import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from formulas import GreenfieldEnergyEnv
from dqn_agent import DQNAgent


N_EPISODES        = 500
TARGET_UPDATE_FREQ = 10
SAVE_EVERY         = 50
PRINT_EVERY        = 10
MODEL_SAVE_DIR     = "models"
RESULTS_SAVE_DIR   = "results"


def _moving_average(values: np.ndarray, window: int = 20) -> np.ndarray:
    if len(values) < window:
        return values.copy()
    kernel = np.ones(window) / window
    padded = np.concatenate([np.full(window - 1, values[0]), values])
    return np.convolve(padded, kernel, mode="valid")


def _progress_bar(current: int, total: int, width: int = 30) -> str:
    filled = int(width * current / total)
    bar    = "█" * filled + "░" * (width - filled)
    pct    = 100.0 * current / total
    return f"[{bar}] {pct:5.1f}%  ep {current}/{total}"


def _save_plots(
    rewards:      np.ndarray,
    fuels:        np.ndarray,
    reliabilities: np.ndarray,
    epsilons:     np.ndarray,
    plots_dir:    str = "plots",
) -> None:
    os.makedirs(plots_dir, exist_ok=True)
    episodes = np.arange(1, len(rewards) + 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(episodes, rewards, color="steelblue", alpha=0.4, linewidth=0.8,
            label="Episode reward")
    ax.plot(episodes, _moving_average(rewards), color="steelblue", linewidth=2.0,
            label="Moving avg (20 ep)")
    ax.set_title("Reward Convergence over Training")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "training_reward.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {plots_dir}/training_reward.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(episodes, fuels, color="firebrick", alpha=0.4, linewidth=0.8,
            label="Fuel (L)")
    ax.plot(episodes, _moving_average(fuels), color="firebrick", linewidth=2.0,
            label="Moving avg (20 ep)")
    ax.set_title("Fuel Consumption over Training")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Fuel Consumed (L)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "training_fuel.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {plots_dir}/training_fuel.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(episodes, reliabilities, color="seagreen", alpha=0.4, linewidth=0.8,
            label="Reliability (%)")
    ax.plot(episodes, _moving_average(reliabilities), color="seagreen", linewidth=2.0,
            label="Moving avg (20 ep)")
    ax.set_title("Reliability Index over Training")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reliability (%)")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "training_reliability.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {plots_dir}/training_reliability.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(episodes, epsilons, color="darkorchid", linewidth=1.5)
    ax.set_title("Epsilon Decay over Training")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "training_epsilon.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {plots_dir}/training_epsilon.png")


def train() -> None:
    os.makedirs(MODEL_SAVE_DIR,   exist_ok=True)
    os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)
    os.makedirs("plots",          exist_ok=True)

    print("=" * 65)
    print("  Greenfield University — DQN Training")
    print("=" * 65)
    print(f"  Episodes         : {N_EPISODES}")
    print(f"  Target update    : every {TARGET_UPDATE_FREQ} episodes")
    print(f"  Checkpoint save  : every {SAVE_EVERY} episodes")
    print("=" * 65)

    env   = GreenfieldEnergyEnv()
    agent = DQNAgent(
        state_size  = env.state_size,
        action_size = env.action_size,
    )

    all_rewards       = np.zeros(N_EPISODES, dtype=np.float64)
    all_fuels         = np.zeros(N_EPISODES, dtype=np.float64)
    all_reliabilities = np.zeros(N_EPISODES, dtype=np.float64)
    all_shedding      = np.zeros(N_EPISODES, dtype=np.int64)
    all_losses        = np.zeros(N_EPISODES, dtype=np.float64)
    all_epsilons      = np.zeros(N_EPISODES, dtype=np.float64)

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
            action                      = agent.select_action(state)
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

        if ep % SAVE_EVERY == 0:
            ckpt_path = os.path.join(MODEL_SAVE_DIR, f"checkpoint_ep{ep}.pth")
            agent.save(ckpt_path)
            print(f"  [checkpoint] Saved → {ckpt_path}")

        if ep % PRINT_EVERY == 0:
            bar = _progress_bar(ep, N_EPISODES)
            print(
                f"{bar} | "
                f"Reward {total_reward:>10.1f} | "
                f"Fuel {total_fuel:>8.1f} L | "
                f"Rel {reliability:>6.2f}% | "
                f"ε {agent.epsilon:.4f} | "
                f"Loss {mean_loss:.5f}"
            )

    final_path = os.path.join(MODEL_SAVE_DIR, "dqn_final.pth")
    agent.save(final_path)
    print(f"\n  Final model saved → {final_path}")

    metrics_path = os.path.join(RESULTS_SAVE_DIR, "training_metrics.npz")
    np.savez(
        metrics_path,
        rewards       = all_rewards,
        fuels         = all_fuels,
        reliabilities = all_reliabilities,
        shedding      = all_shedding,
        losses        = all_losses,
        epsilons      = all_epsilons,
    )
    print(f"  Metrics saved    → {metrics_path}")

    print("\nGenerating training plots …")
    _save_plots(all_rewards, all_fuels, all_reliabilities, all_epsilons)

    elapsed   = time.time() - train_start
    mins, sec = divmod(int(elapsed), 60)
    hrs, mins = divmod(mins, 60)
    print(f"\n  Total training time: {hrs:02d}h {mins:02d}m {sec:02d}s")

    print("\n" + "=" * 65)
    print("  Training Summary: Episode 1  vs  Final Episode")
    print("=" * 65)
    header = f"  {'Metric':<28} {'Ep 1':>12} {'Ep ' + str(N_EPISODES):>12}  {'Change':>10}"
    print(header)
    print("  " + "-" * 63)

    def _row(label, v1, v2, fmt=".2f", unit=""):
        delta = v2 - v1
        sign  = "+" if delta >= 0 else ""
        print(f"  {label:<28} {v1:>12{fmt}} {v2:>12{fmt}}  {sign}{delta:>9{fmt}}{unit}")

    _row("Total Reward",          all_rewards[0],       all_rewards[-1],       fmt=".1f")
    _row("Fuel Consumed (L)",     all_fuels[0],         all_fuels[-1],         fmt=".1f")
    _row("Reliability % (energy)", all_reliabilities[0], all_reliabilities[-1], fmt=".2f")
    _row("Load Shedding Events",  all_shedding[0],      all_shedding[-1],      fmt=".0f")
    _row("Mean Loss",             all_losses[0],        all_losses[-1],        fmt=".5f")
    _row("Epsilon",               all_epsilons[0],      all_epsilons[-1],      fmt=".4f")
    print("=" * 65)
    print("  Training complete.\n")


if __name__ == "__main__":
    train()

