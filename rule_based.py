import os
import numpy as np


class GreenfieldRuleBasedController:


    DISCHARGE_THRESHOLD_KWH = 960.0


    DIESEL_SHORTFALL_FRAC = 0.60


    DIESEL_MIN_LOAD_FRAC  = 0.50

    def __init__(self, env):
        self.env = env

    def select_action(self, state: np.ndarray, env) -> tuple:
        t            = env.t
        hour         = int(env.hour_of_day[t])
        solar_kw     = float(env.solar_output_kw[t])
        soc_kwh      = env.battery_soc_kwh
        diesel_avail = bool(env.diesel_available[t])


        if hour >= 18 or hour <= 6:
            order = ["LP1", "LP8", "LP4", "LP2", "LP3", "LP5", "LP6", "LP7"]
        else:
            order = ["LP1", "LP2", "LP3", "LP4", "LP5", "LP6", "LP7", "LP8"]

        eligible        = [lp for lp in order if env._lp_in_time_window(lp, hour)]
        eligible_demand = sum(float(env.lp_demand_kw[lp][t]) for lp in eligible)
        priority_demand = sum(
            float(env.lp_demand_kw[lp][t])
            for lp in eligible if lp in env.PRIORITY_LPS
        )


        if soc_kwh > self.DISCHARGE_THRESHOLD_KWH:
            batt_mode = 2
        elif solar_kw > eligible_demand:
            batt_mode = 0
        else:
            batt_mode = 1


        supply_available = solar_kw + (
            (soc_kwh - self.DISCHARGE_THRESHOLD_KWH) * env.BATTERY_ETA
            if soc_kwh > self.DISCHARGE_THRESHOLD_KWH else 0.0
        )
        shortfall  = max(0.0, priority_demand - supply_available)
        diesel_on  = int(
            diesel_avail
            and shortfall > priority_demand * self.DIESEL_SHORTFALL_FRAC
        )

        if diesel_on:
            batt_mode = 1


        action          = env._action_from_decisions(diesel_on, batt_mode)
        force_diesel_kw = (
            max(shortfall, env.DIESEL_MIN_LOAD_KW)
            if diesel_on else 0.0
        )

        return action, 0.0, force_diesel_kw

    def run_episode(self, env) -> dict:
        state = env.reset(demand_jitter=0.0)
        done  = False

        total_reward     = 0.0
        total_fuel       = 0.0
        total_fuel_cost  = 0.0
        total_unmet      = 0.0
        total_demand_kwh = 0.0
        total_served_kwh = 0.0
        shedding_events  = 0

        soc_history          = []
        reward_history       = []
        load_served_history  = []
        fuel_history         = []
        fuel_cost_history    = []
        lp_served_history    = []
        unmet_history        = []

        while not done:
            action, pre_charge_kw, force_kw = self.select_action(state, env)
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
            unmet_history.append(info["unmet_load_kw"])

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
            "unmet_history":        unmet_history,
            "load_shedding_events": shedding_events,
        }


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from configuration import GreenfieldEnergyEnv

    env        = GreenfieldEnergyEnv()
    controller = GreenfieldRuleBasedController(env)

    print("\nRunning rule-based controller (8,760 timesteps) …")
    results = controller.run_episode(env)

    print("\n── Baseline Summary ──────────────────────────────────────────────────")
    print(f"  Total reward          : {results['total_reward']:>14.2f}")
    print(f"  Total fuel consumed   : {results['total_fuel_litres']:>12.2f} L")
    print(f"  Total fuel cost       : {results['total_fuel_cost_NGN']:>14,.0f} NGN")
    print(f"  Reliability index     : {results['reliability_index'] * 100:>12.2f} %")
    print(f"  Total unmet load      : {results['total_unmet_load_kwh']:>12.2f} kWh")
    print(f"  Load shedding events  : {results['load_shedding_events']:>12d} timesteps")
    print("─────────────────────────────────────────────────────────────────────")
    print(f"\n  Dataset baseline: 69.90% — runtime result above should be close.")

    os.makedirs("plots", exist_ok=True)
    timesteps = np.arange(1, len(results["reward_history"]) + 1)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(timesteps, results["reward_history"],
                 color="steelblue", linewidth=0.5, alpha=0.7)
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Rule-Based Controller — Per-Step Reward")
    axes[0].grid(True, linewidth=0.4, alpha=0.5)

    axes[1].plot(timesteps, np.cumsum(results["reward_history"]),
                 color="darkorange", linewidth=1.2)
    axes[1].set_ylabel("Cumulative Reward")
    axes[1].set_title("Rule-Based Controller — Cumulative Reward")
    axes[1].grid(True, linewidth=0.4, alpha=0.5)

    axes[2].plot(timesteps, results["soc_history"],
                 color="seagreen", linewidth=0.6, alpha=0.85)
    axes[2].axhline(960, color="darkred", linestyle=":",
                    linewidth=0.9, label="Discharge threshold (960 kWh)")
    axes[2].axhline(120, color="red", linestyle="--",
                    linewidth=0.9, label="Min SOC (120 kWh)")
    axes[2].set_ylabel("Battery SOC (kWh)")
    axes[2].set_xlabel("Timestep (hour of year)")
    axes[2].set_title("Rule-Based Controller — Battery SOC (note: barely used below 960 kWh)")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    fig.savefig("plots/rule_based_baseline.png", dpi=150)
    plt.close(fig)
    print("  Plot saved → plots/rule_based_baseline.png")
