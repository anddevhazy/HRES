"""
capacity_diagnostic.py
======================
Answers one question before retraining:

  "Is 90%+ reliability physically achievable with this hardware?"

Three policies are simulated back-to-back over the same 8,760-hour year:

  1. MAX SUPPLY  — action 5 every step (diesel ON + battery DISCHARGE).
                   Represents the absolute ceiling: every dispatchable source
                   running flat-out at every hour.  If this cannot reach 90%,
                   no controller ever will with the current hardware.

  2. DIESEL ONLY — action 4 every step (diesel ON + battery IDLE).
                   Shows how much the battery actually contributes.

  3. RULE-BASED  — the existing deterministic baseline (70.51%).

Outputs
-------
  - Console table comparing all three on reliability, fuel, and shedding events.
  - plots/capacity_diagnostic.png  — hourly power balance under MAX SUPPLY,
    with shedding hours highlighted so you can see *when* and *why* shedding
    still occurs even at maximum output.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from formulas import GreenfieldEnergyEnv
from rule_based import GreenfieldRuleBasedController


# ─────────────────────────────────────────────────────────────────────────────
# Generic episode runner — takes a callable policy(state, env) → action
# ─────────────────────────────────────────────────────────────────────────────

def run_policy(env, policy_fn, label: str) -> dict:
    """Run one full episode with policy_fn and return a metrics dict."""
    state = env.reset()
    done  = False

    total_fuel        = 0.0
    total_demand_kw   = 0.0
    total_served_kw   = 0.0
    shedding_steps    = 0
    served_steps      = 0          # timesteps with zero shedding

    solar_history    = []
    diesel_history   = []
    batt_history     = []          # net battery contribution (discharge − charge)
    demand_history   = []
    served_history   = []
    soc_history      = []
    shedding_mask    = []          # True at timesteps where load shedding occurred

    while not done:
        action = policy_fn(state, env)
        state, reward, done, info = env.step(action)

        fuel = float(sum(info["fuel_consumed_per_source"].values()))
        total_fuel      += fuel
        total_demand_kw += info["total_demand_kw"]
        total_served_kw += info["load_served_kw"]

        shed = info["load_shedding_occurred"]
        shedding_steps += int(shed)
        served_steps   += int(not shed)

        solar_history.append(info["solar_output_kw"])
        diesel_history.append(info["diesel_output_kw"])
        batt_history.append(info["battery_discharge_kw"] - info["battery_charge_kw"])
        demand_history.append(info["total_demand_kw"])
        served_history.append(info["load_served_kw"])
        soc_history.append(info["battery_soc_kwh"])
        shedding_mask.append(shed)

    n = len(served_history)
    reliability_by_steps = 100.0 * served_steps / n
    reliability_by_energy = 100.0 * total_served_kw / total_demand_kw if total_demand_kw > 0 else 0.0

    return {
        "label":                   label,
        "reliability_pct_steps":   reliability_by_steps,
        "reliability_pct_energy":  reliability_by_energy,
        "total_fuel_litres":       total_fuel,
        "shedding_steps":          shedding_steps,
        "served_steps":            served_steps,
        "n_timesteps":             n,
        "solar_history":           np.array(solar_history),
        "diesel_history":          np.array(diesel_history),
        "batt_net_history":        np.array(batt_history),
        "demand_history":          np.array(demand_history),
        "served_history":          np.array(served_history),
        "soc_history":             np.array(soc_history),
        "shedding_mask":           np.array(shedding_mask, dtype=bool),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Shedding hour analyser
# ─────────────────────────────────────────────────────────────────────────────

def analyse_shedding_hours(result: dict, env: GreenfieldEnergyEnv) -> None:
    """
    Print a breakdown of *why* shedding still occurs under MAX SUPPLY.
    Buckets shedding hours by gap size (demand − max_possible_supply).
    """
    mask    = result["shedding_mask"]
    demand  = result["demand_history"]
    served  = result["served_history"]
    solar   = result["solar_history"]
    diesel  = result["diesel_history"]
    batt    = result["batt_net_history"]

    gaps = demand[mask] - served[mask]          # kW gap at each shedding hour
    max_supply = solar[mask] + diesel[mask] + np.maximum(0, batt[mask])

    print(f"\n── Shedding analysis under '{result['label']}' ──────────────────────")
    print(f"   Total shedding hours : {mask.sum()}")
    if mask.sum() == 0:
        print("   No shedding — 100% reliability achieved!")
        return

    print(f"   Avg gap (demand−served): {gaps.mean():.1f} kW")
    print(f"   Max gap                : {gaps.max():.1f} kW")
    print(f"   Avg max supply at shed : {max_supply.mean():.1f} kW")
    print(f"   Avg demand at shed     : {demand[mask].mean():.1f} kW")

    # Hour-of-day distribution of shedding
    hours_of_day = env.hour_of_day[:len(mask)]
    shed_hours   = hours_of_day[mask]
    print(f"\n   Shedding by hour-of-day (how many shedding events per hour):")
    for h in range(0, 24, 4):
        count = int(np.sum((shed_hours >= h) & (shed_hours < h + 4)))
        bar   = "█" * (count // 5)
        print(f"     {h:02d}:00–{h+3:02d}:59  {count:4d}  {bar}")

    # Gap size buckets
    buckets = [(0, 10), (10, 50), (50, 100), (100, 200), (200, 1e9)]
    print(f"\n   Shedding by shortfall size:")
    for lo, hi in buckets:
        n = int(np.sum((gaps >= lo) & (gaps < hi)))
        label = f"  <{hi:.0f} kW" if hi < 1e9 else f"≥{lo:.0f} kW"
        print(f"     {lo:>5.0f}–{hi:>5.0f} kW : {n:4d} hours")


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def save_diagnostic_plot(max_result: dict, rule_result: dict,
                         env: GreenfieldEnergyEnv, plots_dir: str = "plots") -> None:
    os.makedirs(plots_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    hours = np.arange(len(max_result["demand_history"]))

    # ── Panel 1: Power balance under MAX SUPPLY ───────────────────────────────
    ax = axes[0]
    ax.fill_between(hours, max_result["demand_history"],
                    alpha=0.25, color="tomato", label="Eligible demand (kW)")
    ax.fill_between(hours, max_result["solar_history"],
                    alpha=0.5, color="gold", label="Solar (kW)")
    ax.fill_between(hours,
                    max_result["solar_history"] + max_result["diesel_history"],
                    max_result["solar_history"],
                    alpha=0.4, color="steelblue", label="Diesel (kW)")
    # Shade shedding hours
    for t, shed in enumerate(max_result["shedding_mask"]):
        if shed:
            ax.axvspan(t, t + 1, color="red", alpha=0.15, linewidth=0)
    ax.set_ylabel("Power (kW)")
    ax.set_title("MAX SUPPLY (action 5 every step): Power Balance  [red = shedding]")
    ax.legend(loc="upper right", fontsize=7, ncol=3)
    ax.grid(True, linewidth=0.3, alpha=0.4)

    # ── Panel 2: Battery SOC comparison ──────────────────────────────────────
    ax = axes[1]
    ax.plot(hours, max_result["soc_history"],
            color="steelblue", linewidth=0.7, alpha=0.8, label="SOC — MAX SUPPLY")
    ax.plot(hours, rule_result["soc_history"],
            color="darkorange", linewidth=0.7, alpha=0.8, label="SOC — Rule-based")
    ax.axhline(env.BATTERY_MIN_SOC_KWH, color="red", linestyle=":",
               linewidth=0.8, label=f"Min SOC ({env.BATTERY_MIN_SOC_KWH:.0f} kWh)")
    ax.axhline(env.BATTERY_MAX_SOC_KWH, color="green", linestyle=":",
               linewidth=0.8, label=f"Max SOC ({env.BATTERY_MAX_SOC_KWH:.0f} kWh)")
    ax.set_ylabel("Battery SOC (kWh)")
    ax.set_title("Battery SOC: MAX SUPPLY vs Rule-Based")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.set_ylim(0, env.BATTERY_CAPACITY_KWH + 100)
    ax.grid(True, linewidth=0.3, alpha=0.4)

    # ── Panel 3: Hourly shedding comparison ───────────────────────────────────
    ax = axes[2]
    max_shed_kw  = np.maximum(0, max_result["demand_history"] - max_result["served_history"])
    rule_shed_kw = np.maximum(0, rule_result["demand_history"] - rule_result["served_history"])
    ax.plot(hours, max_shed_kw,  color="steelblue", linewidth=0.5,
            alpha=0.7, label="Unmet load — MAX SUPPLY (kW)")
    ax.plot(hours, rule_shed_kw, color="darkorange", linewidth=0.5,
            alpha=0.7, label="Unmet load — Rule-Based (kW)")
    ax.set_xlabel("Hour of year")
    ax.set_ylabel("Unmet load (kW)")
    ax.set_title("Unmet Load: MAX SUPPLY vs Rule-Based")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, linewidth=0.3, alpha=0.4)

    fig.tight_layout()
    path = os.path.join(plots_dir, "capacity_diagnostic.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    env = GreenfieldEnergyEnv()

    print("\n" + "=" * 65)
    print("  Capacity Diagnostic — Can 90% reliability be achieved?")
    print("=" * 65)

    # ── Policy 1: MAX SUPPLY (action 5 every step) ────────────────────────────
    print("\n[1/3] Simulating MAX SUPPLY (diesel ON + battery DISCHARGE every step)…")
    max_result = run_policy(env, lambda state, env: 5, "MAX SUPPLY")

    # ── Policy 2: DIESEL ONLY (action 4 every step) ───────────────────────────
    print("[2/3] Simulating DIESEL ONLY (diesel ON + battery IDLE every step)…")
    diesel_only_result = run_policy(env, lambda state, env: 4, "DIESEL ONLY")

    # ── Policy 3: RULE-BASED ──────────────────────────────────────────────────
    print("[3/3] Simulating RULE-BASED controller…")
    controller = GreenfieldRuleBasedController(env)
    rule_result = run_policy(
        env,
        lambda state, env: controller.select_action(state, env),
        "RULE-BASED"
    )

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Results")
    print("=" * 65)
    header = f"  {'Policy':<18} {'Rel (steps)':>12} {'Rel (energy)':>13} {'Fuel (L)':>10} {'Shed hrs':>10}"
    print(header)
    print("  " + "-" * 63)
    for r in (max_result, diesel_only_result, rule_result):
        print(
            f"  {r['label']:<18} "
            f"{r['reliability_pct_steps']:>11.2f}% "
            f"{r['reliability_pct_energy']:>12.2f}% "
            f"{r['total_fuel_litres']:>10.1f} "
            f"{r['shedding_steps']:>10d}"
        )
    print("=" * 65)

    # ── Verdict ───────────────────────────────────────────────────────────────
    ceiling = max_result["reliability_pct_steps"]
    print(f"\n  Physical ceiling (MAX SUPPLY): {ceiling:.2f}%")
    if ceiling >= 90.0:
        gap = ceiling - rule_result["reliability_pct_steps"]
        print(f"  ✓ 90% IS physically achievable — the hardware can do it.")
        print(f"    Gap between ceiling and rule-based: {gap:.2f} percentage points.")
        print(f"    The DQN must close this gap through smarter dispatch.")
    else:
        shortfall = 90.0 - ceiling
        print(f"  ✗ 90% is NOT achievable with current hardware.")
        print(f"    Even with diesel + battery at full output every hour,")
        print(f"    reliability only reaches {ceiling:.2f}% — {shortfall:.2f} pp short of 90%.")
        print(f"    Recommendation: revisit the capacity assumptions in your")
        print(f"    data or discuss with your supervisor whether the target")
        print(f"    needs to be adjusted.")

    # ── Battery contribution ──────────────────────────────────────────────────
    batt_gain = max_result["reliability_pct_steps"] - diesel_only_result["reliability_pct_steps"]
    print(f"\n  Battery contribution: {batt_gain:+.2f} pp over diesel-only")
    if abs(batt_gain) < 0.5:
        print("  → Battery is making almost no difference to reliability.")
        print("    This means shedding occurs during hours when the battery")
        print("    is already depleted. The DQN's battery dispatch strategy")
        print("    matters less than diesel scheduling in this system.")

    # ── Detailed shedding analysis for MAX SUPPLY ─────────────────────────────
    analyse_shedding_hours(max_result, env)

    # ── Plot ──────────────────────────────────────────────────────────────────
    save_diagnostic_plot(max_result, rule_result, env)

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
