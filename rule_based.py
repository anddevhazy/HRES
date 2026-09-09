"""
rule_based.py
=============
Deterministic rule-based energy management controller for the Greenfield
University off-grid hybrid energy system (GreenfieldEnergyEnv in formulas.py).

Serves as the baseline benchmark against which the DQN agent is compared.

This controller directly implements the 11 rules from university_metadata.json.
Rules R2–R4 (priority dispatch), R8–R11 (diesel sizing, fuel model, time windows,
emergency protection) are enforced by the environment itself.  This file
implements the two decisions the controller makes per timestep:

  Decision 1 — Diesel ON/OFF:
    R7a: Start diesel if solar + max-battery-discharge < LP1–LP4 demand.
    R7b: Start diesel if SOC < 240 kWh AND solar output < 30 kW.
    Diesel is not started for non-priority loads (LP5–LP8) alone.

  Decision 2 — Battery mode (CHARGE / IDLE / DISCHARGE):
    R5: Charge battery when solar output exceeds eligible load demand (surplus).
    R6: Discharge battery when solar output falls short of eligible demand (deficit).
    Idle when battery is at a bound or conditions are neutral.

Rule reference:
  R1  Solar-first dispatch            (R1 is always satisfied — solar is a fixed input)
  R2  Daytime priority order          (enforced in env._dispatch_loads)
  R3  Night priority order            (enforced in env._dispatch_loads)
  R4  Load shedding threshold         (enforced in env._dispatch_loads)
  R5  Battery charges on surplus      → implemented here
  R6  Battery discharges on deficit   → implemented here
  R7  Diesel start condition          → implemented here
  R8  Diesel output sizing            (enforced in env.step)
  R9  Diesel fuel cost accounting     (enforced in env._compute_diesel_fuel)
  R10 Time-window restrictions        (enforced in env._get_eligible_lps)
  R11 Emergency battery protection    (enforced in env._get_eligible_lps)
"""

import os
import numpy as np


class GreenfieldRuleBasedController:
    """
    Fixed-rule controller for GreenfieldEnergyEnv.

    Parameters
    ----------
    env : GreenfieldEnergyEnv
        The environment instance — stored only for initial reference checks.
        The live env passed to select_action / run_episode is used at runtime.
    """

    def __init__(self, env):
        self.env = env

    # ─────────────────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray, env) -> int:
        """
        Choose a discrete action using rules R5, R6, and R7.

        Parameters
        ----------
        state : np.ndarray  (accepted for API compatibility; not used directly)
        env   : GreenfieldEnergyEnv  live environment at the current timestep

        Returns
        -------
        action_idx : int  index into env.ACTION_MAP
        """
        t            = env.t
        hour         = int(env.hour_of_day[t])
        solar_kw     = float(env.solar_output_kw[t])
        soc_kwh      = env.battery_soc_kwh
        diesel_avail = bool(env.diesel_available[t])
        lp_demands   = {lp: float(env.lp_demand_kw[lp][t]) for lp in env.LP_IDS}

        # Eligible LPs at this hour (R10, R11 applied by env)
        eligible_lps   = env._get_eligible_lps(hour, soc_kwh, solar_kw)
        eligible_demand = sum(lp_demands.get(lp, 0.0) for lp in eligible_lps)

        # Priority loads (LP1–LP4) that are currently eligible
        priority_lps    = [lp for lp in ("LP1", "LP2", "LP3", "LP4")
                           if lp in eligible_lps]
        priority_demand = sum(lp_demands.get(lp, 0.0) for lp in priority_lps)

        # Maximum battery discharge deliverable from current SOC
        depth_kwh          = max(0.0, soc_kwh - env.BATTERY_MIN_SOC_KWH)
        max_batt_discharge = min(env.BATTERY_MAX_DISCHARGE_KW,
                                 depth_kwh * env.BATTERY_ETA)

        # ── R7: Diesel start condition ────────────────────────────────────────
        # R7a: solar + max battery discharge cannot cover priority loads LP1–LP4
        # R7b: low SOC emergency (SOC < 240 kWh AND solar < 30 kW)
        cond_a    = ((solar_kw + max_batt_discharge) < priority_demand
                     and priority_demand > 0.0)
        cond_b    = (soc_kwh < env.DIESEL_EMERGENCY_SOC_KWH
                     and solar_kw < env.DIESEL_EMERGENCY_SOLAR_KW)
        diesel_on = diesel_avail and (cond_a or cond_b)

        # ── R5 / R6: Battery mode ─────────────────────────────────────────────
        if solar_kw >= eligible_demand:
            # Surplus solar available → charge battery (R5)
            if soc_kwh < env.BATTERY_MAX_SOC_KWH - 1.0:
                batt_mode = 0   # CHARGE
            else:
                batt_mode = 1   # IDLE (battery full)
        else:
            # Solar deficit → discharge battery to help cover loads (R6)
            if soc_kwh > env.BATTERY_MIN_SOC_KWH + 1.0:
                batt_mode = 2   # DISCHARGE
            else:
                batt_mode = 1   # IDLE (battery depleted, protect remaining energy)

        return env._action_from_decisions(int(diesel_on), batt_mode)

    def run_episode(self, env) -> dict:
        """
        Execute one full episode (all 8,760 timesteps) and collect metrics.

        Parameters
        ----------
        env : GreenfieldEnergyEnv  reset() is called internally.

        Returns
        -------
        results : dict
            total_reward          – sum of per-step rewards
            total_fuel_litres     – total diesel fuel consumed (litres)
            total_fuel_cost_NGN   – total fuel expenditure (Nigerian Naira)
            total_unmet_load_kwh  – total unmet eligible load energy (kWh)
            reliability_index     – Σ load_served / Σ eligible_demand  [0, 1]
            soc_history           – battery SOC at each timestep (kWh)
            reward_history        – per-step reward
            load_served_history   – per-step load served (kW)
            fuel_history          – per-step fuel consumed (litres)
            fuel_cost_history     – per-step fuel cost (NGN)
            lp_served_history     – per-step dict of served kW per LP
            load_shedding_events  – number of timesteps with load shedding
        """
        state = env.reset()
        done  = False

        total_reward     = 0.0
        total_fuel       = 0.0
        total_fuel_cost  = 0.0
        total_unmet      = 0.0
        total_demand_kwh = 0.0
        total_served_kwh = 0.0
        shedding_events  = 0

        soc_history         = []
        reward_history      = []
        load_served_history = []
        fuel_history        = []
        fuel_cost_history   = []
        lp_served_history   = []

        while not done:
            action = self.select_action(state, env)
            state, reward, done, info = env.step(action)

            step_fuel = float(sum(info["fuel_consumed_per_source"].values()))

            total_reward     += reward
            total_fuel       += step_fuel
            total_fuel_cost  += float(info["fuel_cost_NGN"])
            total_unmet      += info["unmet_load_kw"]
            total_demand_kwh += info["total_demand_kw"]
            total_served_kwh += info["load_served_kw"]

            if info["load_shedding_occurred"]:
                shedding_events += 1

            soc_history.append(info["battery_soc_kwh"])
            reward_history.append(reward)
            load_served_history.append(info["load_served_kw"])
            fuel_history.append(step_fuel)
            fuel_cost_history.append(float(info["fuel_cost_NGN"]))
            lp_served_history.append(info["lp_served_kw"])

        reliability_index = (total_served_kwh / total_demand_kwh
                             if total_demand_kwh > 0.0 else 0.0)

        return {
            "total_reward":         total_reward,
            "total_fuel_litres":    total_fuel,
            "total_fuel_cost_NGN":  total_fuel_cost,
            "total_unmet_load_kwh": total_unmet,
            "reliability_index":    reliability_index,
            "soc_history":          soc_history,
            "reward_history":       reward_history,
            "load_served_history":  load_served_history,
            "fuel_history":         fuel_history,
            "fuel_cost_history":    fuel_cost_history,
            "lp_served_history":    lp_served_history,
            "load_shedding_events": shedding_events,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point — Greenfield benchmark run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from formulas import GreenfieldEnergyEnv

    env        = GreenfieldEnergyEnv()
    controller = GreenfieldRuleBasedController(env)

    print("\nRunning rule-based controller for one full episode (8,760 timesteps)…")
    results = controller.run_episode(env)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────────────────────")
    print(f"  Total reward          : {results['total_reward']:>14.2f}")
    print(f"  Total fuel consumed   : {results['total_fuel_litres']:>12.2f} L")
    print(f"  Total fuel cost       : {results['total_fuel_cost_NGN']:>12.0f} NGN")
    print(f"  Reliability index     : {results['reliability_index'] * 100:>12.2f} %")
    print(f"  Total unmet load      : {results['total_unmet_load_kwh']:>12.2f} kWh")
    print(f"  Load shedding events  : {results['load_shedding_events']:>12d} timesteps")
    print("─────────────────────────────────────────────────────────────────────")

    # ── Plots ─────────────────────────────────────────────────────────────────
    os.makedirs("plots", exist_ok=True)
    timesteps = np.arange(1, len(results["reward_history"]) + 1)

    # Plot 1: per-step and cumulative reward
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(timesteps, results["reward_history"],
                 color="steelblue", linewidth=0.5, alpha=0.7)
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Rule-Based Controller — Per-Step Reward")
    axes[0].grid(True, linewidth=0.4, alpha=0.5)

    cumulative = np.cumsum(results["reward_history"])
    axes[1].plot(timesteps, cumulative, color="darkorange", linewidth=1.2)
    axes[1].set_ylabel("Cumulative Reward")
    axes[1].set_xlabel("Timestep (hour of year)")
    axes[1].set_title("Rule-Based Controller — Cumulative Reward")
    axes[1].grid(True, linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    reward_path = os.path.join("plots", "rule_based_reward_history.png")
    fig.savefig(reward_path, dpi=150)
    plt.close(fig)
    print(f"\n  Reward plot saved  → {reward_path}")

    # Plot 2: battery SOC with rule thresholds
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(timesteps, results["soc_history"], color="steelblue",
            linewidth=0.6, alpha=0.85, label="Battery SOC (kWh)")
    ax.axhline(GreenfieldEnergyEnv.DIESEL_EMERGENCY_SOC_KWH,
               color="red", linestyle="--", linewidth=0.9,
               label=f"Emergency threshold "
                     f"({GreenfieldEnergyEnv.DIESEL_EMERGENCY_SOC_KWH:.0f} kWh — R7b/R11)")
    ax.axhline(GreenfieldEnergyEnv.BATTERY_MIN_SOC_KWH,
               color="darkred", linestyle=":", linewidth=0.9,
               label=f"Min SOC ({GreenfieldEnergyEnv.BATTERY_MIN_SOC_KWH:.0f} kWh)")
    ax.axhline(GreenfieldEnergyEnv.BATTERY_MAX_SOC_KWH,
               color="green", linestyle=":", linewidth=0.9,
               label=f"Max SOC ({GreenfieldEnergyEnv.BATTERY_MAX_SOC_KWH:.0f} kWh)")
    ax.set_xlabel("Timestep (hour of year)")
    ax.set_ylabel("Battery SOC (kWh)")
    ax.set_title("Rule-Based Controller — Battery SOC History")
    ax.set_ylim(0, GreenfieldEnergyEnv.BATTERY_CAPACITY_KWH + 50)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    soc_path = os.path.join("plots", "rule_based_soc_history.png")
    fig.savefig(soc_path, dpi=150)
    plt.close(fig)
    print(f"  SOC plot saved     → {soc_path}")

    # Plot 3: load served vs demand (first 336 hours = 2 weeks)
    window = 336
    fig, ax = plt.subplots(figsize=(14, 4))
    total_demand_ts = [
        sum(env.lp_demand_kw[lp][t] for lp in env.LP_IDS)
        for t in range(window)
    ]
    ax.fill_between(range(window), total_demand_ts,
                    alpha=0.25, color="tomato", label="Total demand (kW)")
    ax.plot(range(window), results["load_served_history"][:window],
            color="steelblue", linewidth=0.8, label="Load served (kW)")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Power (kW)")
    ax.set_title("Rule-Based Controller — Load Served vs Demand (first 2 weeks)")
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    load_path = os.path.join("plots", "rule_based_load_served.png")
    fig.savefig(load_path, dpi=150)
    plt.close(fig)
    print(f"  Load plot saved    → {load_path}")
