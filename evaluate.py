import os
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configuration import GreenfieldEnergyEnv
from dqn_agent import DQNAgent
from rule_based import GreenfieldRuleBasedController


BASELINE_FUEL_L    = 149_693.0
BASELINE_RELIAB    = 69.90


def run_dqn_episode(env: GreenfieldEnergyEnv, agent: DQNAgent) -> dict:
    state = env.reset(demand_jitter=0.0)
    done  = False
    return _collect_episode(env, state, done,
                            policy_fn=lambda s: agent.select_action(s))


def run_rule_episode(env: GreenfieldEnergyEnv,
                     controller: GreenfieldRuleBasedController) -> dict:
    state = env.reset(demand_jitter=0.0)
    done  = False
    return _collect_episode(env, state, done,
                            policy_fn=lambda s: controller.select_action(s, env)[0])


def run_ceiling_episode(env: GreenfieldEnergyEnv) -> dict:
    state = env.reset(demand_jitter=0.0)
    done  = False
    return _collect_episode(env, state, done, policy_fn=lambda s: 5)


def _collect_episode(env, state, done, policy_fn) -> dict:
    total_demand_kw  = 0.0
    total_served_kw  = 0.0
    total_fuel_L     = 0.0
    total_fuel_NGN   = 0.0
    shedding_steps   = 0

    solar_hist   = []; diesel_hist  = []; batt_net_hist = []
    demand_hist  = []; served_hist  = []; unmet_hist    = []
    soc_hist     = []; fuel_hist    = []; shed_mask     = []
    reward_hist  = []

    while not done:
        action                         = policy_fn(state)
        state, reward, done, info      = env.step(action)
        fuel = float(sum(info["fuel_consumed_per_source"].values()))

        total_demand_kw += info["total_demand_kw"]
        total_served_kw += info["load_served_kw"]
        total_fuel_L    += fuel
        total_fuel_NGN  += float(info["fuel_cost_NGN"])
        shedding_steps  += int(info["load_shedding_occurred"])

        solar_hist.append(info["solar_output_kw"])
        diesel_hist.append(info["diesel_output_kw"])
        batt_net_hist.append(info["battery_discharge_kw"] - info["battery_charge_kw"])
        demand_hist.append(info["total_demand_kw"])
        served_hist.append(info["load_served_kw"])
        unmet_hist.append(info["unmet_load_kw"])
        soc_hist.append(info["battery_soc_kwh"])
        fuel_hist.append(fuel)
        shed_mask.append(info["load_shedding_occurred"])
        reward_hist.append(reward)

    n = len(served_hist)
    energy_reliability = (100.0 * total_served_kw / total_demand_kw
                          if total_demand_kw > 0 else 0.0)
    step_reliability   = 100.0 * (n - shedding_steps) / n

    return {
        "energy_reliability":  energy_reliability,
        "step_reliability":    step_reliability,
        "total_fuel_L":        total_fuel_L,
        "total_fuel_NGN":      total_fuel_NGN,
        "total_demand_kwh":    total_demand_kw,
        "total_served_kwh":    total_served_kw,
        "total_unmet_kwh":     total_demand_kw - total_served_kw,
        "shedding_steps":      shedding_steps,
        "n_timesteps":         n,
        "solar_hist":          np.array(solar_hist),
        "diesel_hist":         np.array(diesel_hist),
        "batt_net_hist":       np.array(batt_net_hist),
        "demand_hist":         np.array(demand_hist),
        "served_hist":         np.array(served_hist),
        "unmet_hist":          np.array(unmet_hist),
        "soc_hist":            np.array(soc_hist),
        "fuel_hist":           np.array(fuel_hist),
        "shed_mask":           np.array(shed_mask, dtype=bool),
        "reward_hist":         np.array(reward_hist),
    }


def print_comparison(dqn: dict, rule: dict, ceiling: dict) -> None:
    print("\n" + "=" * 76)
    print("  Evaluation Results — DQN vs Rule-Based vs Hardware Ceiling")
    print("=" * 76)

    col = 18
    metrics = [
        ("Reliability (energy %)",  "energy_reliability", ".2f"),
        ("Reliability (step %)",    "step_reliability",   ".2f"),
        ("Fuel consumed (L)",       "total_fuel_L",       ",.0f"),
        ("Fuel cost (NGN)",         "total_fuel_NGN",     ",.0f"),
        ("Total demand (kWh)",      "total_demand_kwh",   ",.0f"),
        ("Total served (kWh)",      "total_served_kwh",   ",.0f"),
        ("Total unmet (kWh)",       "total_unmet_kwh",    ",.0f"),
        ("Shedding hours",          "shedding_steps",     ",d"),
    ]

    print(f"  {'Metric':<30} {'DQN':>{col}} {'Rule-Based':>{col}} {'Ceiling':>{col}}")
    print("  " + "-" * 74)
    for label, key, fmt in metrics:
        dv = dqn[key]; rv = rule[key]; cv = ceiling[key]
        print(f"  {label:<30} {dv:{col}{fmt}} {rv:{col}{fmt}} {cv:{col}{fmt}}")
    print("=" * 76)


    rel_gain   = dqn["energy_reliability"] - rule["energy_reliability"]
    fuel_delta = dqn["total_fuel_L"]       - rule["total_fuel_L"]
    unmet_red  = rule["total_unmet_kwh"]   - dqn["total_unmet_kwh"]

    print(f"\n  DQN vs Rule-Based Baseline:")
    print(f"    Reliability gain   : {rel_gain:+.2f} pp  "
          f"({dqn['energy_reliability']:.2f}% vs {rule['energy_reliability']:.2f}%)")
    pct_unmet = (100*unmet_red/rule["total_unmet_kwh"]
                 if rule["total_unmet_kwh"] > 0 else 0)
    print(f"    Unmet load reduced : {unmet_red:,.0f} kWh  ({pct_unmet:.1f}% less)")
    print(f"    Fuel delta         : {fuel_delta:+,.0f} L  "
          f"({'more' if fuel_delta > 0 else 'less'} than rule-based)")


    obj6_met = dqn["total_fuel_L"] <= BASELINE_FUEL_L
    print(f"\n  Objective 6 (fuel ≤ {BASELINE_FUEL_L:,.0f} L baseline): "
          f"{'✓ MET' if obj6_met else '✗ NOT MET'}")

    gap = ceiling["energy_reliability"] - dqn["energy_reliability"]
    print(f"\n  Gap to hardware ceiling: {gap:.2f} pp  "
          f"({dqn['energy_reliability']:.2f}% vs {ceiling['energy_reliability']:.2f}%)\n")


def save_plots(dqn: dict, rule: dict, ceiling: dict,
               env: GreenfieldEnergyEnv, plots_dir: str = "plots") -> None:
    os.makedirs(plots_dir, exist_ok=True)
    hours = np.arange(dqn["n_timesteps"])


    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Rule-Based\n(baseline)", "DQN Agent\n(this work)"]
    values = [rule["energy_reliability"], dqn["energy_reliability"]]
    colors = ["#e07b54", "#4c9be8"]
    bars   = ax.bar(labels, values, color=colors, width=0.4,
                    edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                f"{val:.2f}%", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    ax.axhline(90,  color="green",  linestyle="--", linewidth=1.2,
               label="90% DQN target", alpha=0.8)
    ax.axhline(BASELINE_RELIAB, color="tomato", linestyle=":",
               linewidth=1.2, label=f"Dataset baseline ({BASELINE_RELIAB}%)", alpha=0.8)
    ax.set_ylabel("Energy Reliability (%)")
    ax.set_title("Energy Reliability: DQN vs Rule-Based Baseline")
    ax.set_ylim(60, 100)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "eval_reliability_bar.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved → {plots_dir}/eval_reliability_bar.png")


    fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    axes[0].fill_between(hours, rule["unmet_hist"], alpha=0.6,
                         color="#e07b54", label="Rule-Based unmet (kW)")
    axes[0].fill_between(hours, dqn["unmet_hist"],  alpha=0.6,
                         color="#4c9be8", label="DQN unmet (kW)")
    axes[0].set_ylabel("Unmet load (kW)")
    axes[0].set_title("Hourly Unmet Load: DQN vs Rule-Based")
    axes[0].legend(fontsize=8); axes[0].grid(True, linewidth=0.3, alpha=0.4)
    axes[1].plot(hours, np.cumsum(rule["unmet_hist"]),
                 color="#e07b54", linewidth=1.0, label="Rule-Based cumulative")
    axes[1].plot(hours, np.cumsum(dqn["unmet_hist"]),
                 color="#4c9be8", linewidth=1.0, label="DQN cumulative")
    axes[1].set_ylabel("Cumulative unmet (kWh)")
    axes[1].set_xlabel("Hour of year")
    axes[1].legend(fontsize=8); axes[1].grid(True, linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "eval_unmet_load.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved → {plots_dir}/eval_unmet_load.png")


    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(hours, dqn["soc_hist"],  color="#4c9be8", linewidth=0.6,
            alpha=0.85, label="DQN SOC")
    ax.plot(hours, rule["soc_hist"], color="#e07b54", linewidth=0.6,
            alpha=0.85, label="Rule-Based SOC")
    ax.axhline(env.BATTERY_MIN_SOC_KWH, color="darkred", linestyle=":",
               linewidth=0.9, label=f"Min SOC ({env.BATTERY_MIN_SOC_KWH:.0f} kWh)")
    ax.axhline(env.BATTERY_MAX_SOC_KWH, color="green", linestyle=":",
               linewidth=0.9, label=f"Max SOC ({env.BATTERY_MAX_SOC_KWH:.0f} kWh)")
    ax.axhline(960, color="darkorange", linestyle="--", linewidth=0.9,
               label="Rule-based discharge threshold (960 kWh)", alpha=0.7)
    ax.set_xlabel("Hour of year"); ax.set_ylabel("Battery SOC (kWh)")
    ax.set_title("Battery SOC: DQN vs Rule-Based (note rule-based barely uses capacity below 960 kWh)")
    ax.set_ylim(0, env.BATTERY_CAPACITY_KWH + 100)
    ax.legend(loc="upper right", fontsize=7, ncol=3)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "eval_soc.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved → {plots_dir}/eval_soc.png")


    window = 336
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    hrs = np.arange(window)
    for ax, res, label, color in [
        (axes[0], dqn,  "DQN Agent",  "#4c9be8"),
        (axes[1], rule, "Rule-Based", "#e07b54"),
    ]:
        ax.fill_between(hrs, res["demand_hist"][:window],
                        alpha=0.2, color="tomato", label="Demand")
        ax.fill_between(hrs, res["solar_hist"][:window],
                        alpha=0.5, color="gold", label="Solar")
        ax.fill_between(hrs,
                        res["solar_hist"][:window] + res["diesel_hist"][:window],
                        res["solar_hist"][:window],
                        alpha=0.4, color="steelblue", label="Diesel")
        for t in range(window):
            if res["shed_mask"][t]:
                ax.axvspan(t, t+1, color="red", alpha=0.15, linewidth=0)
        ax.set_ylabel("Power (kW)")
        ax.set_title(f"{label} — Power Balance (first 2 weeks)  [red = shedding]")
        ax.legend(fontsize=7, ncol=3, loc="upper right")
        ax.grid(True, linewidth=0.3, alpha=0.4)
    axes[1].set_xlabel("Hour")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "eval_power_detail.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved → {plots_dir}/eval_power_detail.png")


    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(hours, np.cumsum(rule["fuel_hist"]), color="#e07b54", linewidth=1.0,
            label=f"Rule-Based  ({rule['total_fuel_L']:,.0f} L)")
    ax.plot(hours, np.cumsum(dqn["fuel_hist"]),  color="#4c9be8", linewidth=1.0,
            label=f"DQN Agent   ({dqn['total_fuel_L']:,.0f} L)")
    ax.axhline(BASELINE_FUEL_L, color="darkorange", linestyle="--", linewidth=1.2,
               label=f"Baseline fuel limit ({BASELINE_FUEL_L:,.0f} L)")
    ax.set_xlabel("Hour of year"); ax.set_ylabel("Cumulative fuel (L)")
    ax.set_title("Cumulative Fuel Consumption — Objective 6: DQN must stay below orange line")
    ax.legend(fontsize=9); ax.grid(True, linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "eval_fuel.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved → {plots_dir}/eval_fuel.png")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained DQN agent.")
    parser.add_argument("--model",     default="models/dqn_best.pth",
                        help="Path to DQN model weights (default: models/dqn_best.pth)")
    parser.add_argument("--plots-dir", default="plots",
                        help="Output directory for plots (default: plots)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    env   = GreenfieldEnergyEnv()
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)
    agent.load(args.model)
    agent.epsilon = 0.0

    print(f"\n  Model loaded : {args.model}")
    print(f"  Evaluating over {env.n_timesteps} timesteps (full year) …\n")

    print("[1/3] DQN agent …")
    dqn_results = run_dqn_episode(env, agent)

    rule_env    = GreenfieldEnergyEnv()
    controller  = GreenfieldRuleBasedController(rule_env)
    print("[2/3] Rule-based controller …")
    rule_results = run_rule_episode(rule_env, controller)

    ceil_env = GreenfieldEnergyEnv()
    print("[3/3] Hardware ceiling (always ON + DISCHARGE) …")
    ceiling_results = run_ceiling_episode(ceil_env)

    print_comparison(dqn_results, rule_results, ceiling_results)

    print("Saving evaluation plots …")
    save_plots(dqn_results, rule_results, ceiling_results, rule_env, args.plots_dir)
    print(f"\n  Done. All plots saved to {args.plots_dir}/\n")


if __name__ == "__main__":
    main()
