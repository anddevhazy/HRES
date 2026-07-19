import os
import numpy as np


class GreenfieldRuleBasedController:

    def __init__(self, env):
        self.env = env


    def select_action(self, state: np.ndarray, env) -> int:
        t            = env.t
        hour         = int(env.hour_of_day[t])
        solar_kw     = float(env.solar_output_kw[t])
        soc_kwh      = env.battery_soc_kwh
        diesel_avail = bool(env.diesel_available[t])

        pre_charge_kw   = solar_kw * 0.10
        avail_for_loads = solar_kw - pre_charge_kw

        if hour >= 18 or hour <= 6:
            order = ["LP1", "LP8", "LP4", "LP2", "LP3", "LP5", "LP6", "LP7"]
        else:
            order = ["LP1", "LP2", "LP3", "LP4", "LP5", "LP6", "LP7", "LP8"]
        eligible = [lp for lp in order if env._lp_in_time_window(lp, hour)]
        eligible_demand = sum(float(env.lp_demand_kw[lp][t]) for lp in eligible)

  
        if avail_for_loads >= eligible_demand * 0.80:
            batt_mode = 0  
        else:
            batt_mode = 2   
        WEAK_DIESEL_SOC   = 125.0   
        WEAK_DIESEL_SOLAR = 10.0   
        diesel_on = (
            diesel_avail
            and soc_kwh <= WEAK_DIESEL_SOC
            and solar_kw < WEAK_DIESEL_SOLAR
        )

        return env._action_from_decisions(int(diesel_on), batt_mode)

    def run_episode(self, env) -> dict:
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


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from configuration import GreenfieldEnergyEnv

    env        = GreenfieldEnergyEnv()
    controller = GreenfieldRuleBasedController(env)

    print("\nRunning rule-based controller for one full episode (8,760 timesteps)…")
    results = controller.run_episode(env)

    print("\n── Summary ──────────────────────────────────────────────────────────")
    print(f"  Total reward          : {results['total_reward']:>14.2f}")
    print(f"  Total fuel consumed   : {results['total_fuel_litres']:>12.2f} L")
    print(f"  Total fuel cost       : {results['total_fuel_cost_NGN']:>12.0f} NGN")
    print(f"  Reliability index     : {results['reliability_index'] * 100:>12.2f} %")
    print(f"  Total unmet load      : {results['total_unmet_load_kwh']:>12.2f} kWh")
    print(f"  Load shedding events  : {results['load_shedding_events']:>12d} timesteps")
    print("─────────────────────────────────────────────────────────────────────")

    os.makedirs("plots", exist_ok=True)
    timesteps = np.arange(1, len(results["reward_history"]) + 1)

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

