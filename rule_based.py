"""
rule_based.py
=============
Deterministic rule-based energy management controller for the modular hybrid
energy system environment defined in energy_env.py.

This controller serves as the **baseline benchmark** against which the DQN
agent's performance is compared in Chapter 4.  It applies a fixed priority
hierarchy — no learning, no exploration — and must be evaluated under
identical environment conditions to the DQN agent for a fair comparison.

Rule priority hierarchy (applied in order at every timestep):
  1. Renewables cover full load  → all controllable OFF, all batteries CHARGE
  2. Renewables short but SoC > 0.30 → all controllable OFF, batteries DISCHARGE
  3. Need controllable + SoC > 0.20  → all controllable ON,  batteries DISCHARGE
  4. Battery depleted (SoC ≤ 0.20)  → all controllable ON,  batteries IDLE
     (load shedding handled automatically by the environment)

Action encoding (mirrors env._build_action_space):
  controllable per source  :  0 = OFF,      1 = ON
  battery per unit         :  0 = CHARGE,   1 = IDLE,   2 = DISCHARGE
"""

import os
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# RuleBasedController
# ─────────────────────────────────────────────────────────────────────────────

class RuleBasedController:
    """
    Fixed-priority rule-based controller for HybridEnergyEnv.

    Parameters
    ----------
    env : HybridEnergyEnv
        The environment instance this controller will operate on.  Stored
        only as a reference for initialisation checks; the live env passed to
        select_action / run_episode is used for all runtime queries.
    """

    # SoC thresholds that define the boundary between operating rules
    SOC_THRESHOLD_HIGH = 0.30   # above this: use battery alone before diesel
    SOC_THRESHOLD_LOW  = 0.20   # above this (but ≤ HIGH): use diesel + battery

    def __init__(self, env):
        self.env = env

    # ─────────────────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray, env) -> int:
        """
        Choose a discrete action using the fixed priority rule hierarchy.

        The method reads current operating conditions directly from `env`
        (availability profiles, SoC, load profile) rather than decoding the
        normalised state vector, which avoids any rounding from normalisation.

        Parameters
        ----------
        state : np.ndarray
            Normalised observation vector — accepted for API compatibility with
            the DQN agent interface but not used internally.
        env : HybridEnergyEnv
            Live environment instance at the current timestep.

        Returns
        -------
        action_idx : int
            Index into env.action_map for the chosen operating mode.
        """
        t = env.t

        # ── Compute available renewable power at this timestep ────────────────
        renewable_power_kw = sum(
            float(env.sources[i]["availability_profile"][t])
            * float(env.sources[i]["rated_capacity_kw"])
            for i in env.renewable_indices
        )

        # ── Current load demand (kW) ──────────────────────────────────────────
        load_kw = float(env.load_profile[t])

        # ── Mean battery SoC across all units ────────────────────────────────
        # Using the mean provides a system-level indicator of stored energy;
        # a single depleted battery should not block discharge from healthy ones.
        mean_soc = float(np.mean(env.soc)) if env.n_batteries > 0 else 0.0

        # ── Pre-build decision patterns ───────────────────────────────────────
        ctrl_all_off       = [0] * env.n_controllable   # all generators OFF
        ctrl_all_on        = [1] * env.n_controllable   # all generators ON
        batt_all_charge    = [0] * env.n_batteries      # all batteries: CHARGE
        batt_all_idle      = [1] * env.n_batteries      # all batteries: IDLE
        batt_all_discharge = [2] * env.n_batteries      # all batteries: DISCHARGE

        # ── Rule 1 ─────────────────────────────────────────────────────────────
        # Available renewable generation alone can fully serve the load.
        # No fossil fuel is needed; store any surplus in the batteries.
        if renewable_power_kw >= load_kw:
            return self._find_action(env, ctrl_all_off, batt_all_charge)

        # ── Rule 2 ─────────────────────────────────────────────────────────────
        # Renewables fall short, but batteries are sufficiently charged (SoC > 0.30).
        # Supplement renewable output with battery discharge; keep diesel off
        # to avoid fuel burn and carbon emissions.
        if mean_soc > self.SOC_THRESHOLD_HIGH:
            return self._find_action(env, ctrl_all_off, batt_all_discharge)

        # ── Rule 3 ─────────────────────────────────────────────────────────────
        # Renewables and battery together cannot reliably cover the load
        # (battery SoC is in the 0.20–0.30 band).  Start all controllable sources
        # and also discharge the battery to maximise supply before shedding.
        if mean_soc > self.SOC_THRESHOLD_LOW:
            return self._find_action(env, ctrl_all_on, batt_all_discharge)

        # ── Rule 4 ─────────────────────────────────────────────────────────────
        # Battery is effectively depleted (SoC ≤ 0.20).  Run all controllable
        # sources at full capacity; leave batteries idle to protect the remaining
        # stored energy.  Any residual deficit is handled by the environment's
        # priority-based load shedding engine (Eq 3.9).
        return self._find_action(env, ctrl_all_on, batt_all_idle)

    def run_episode(self, env) -> dict:
        """
        Execute one full episode (all timesteps in the load profile) and collect
        per-step metrics.

        Parameters
        ----------
        env : HybridEnergyEnv
            The environment to run.  reset() is called internally before the
            first timestep.

        Returns
        -------
        results : dict
            total_reward          – sum of all per-step rewards
            total_fuel_litres     – total fuel consumed across all controllable sources
            total_unmet_load_kwh  – total unmet load energy (1 kW unmet × 1 h = 1 kWh)
            reliability_index     – fraction of total load demand successfully served
                                    (Eq 3.14: Σ load_served / Σ load_demand)
            soc_history           – list of n_batteries lists; each inner list holds
                                    the SoC value recorded at every timestep
            reward_history        – per-step reward values
            load_served_history   – per-step load served (kW)
            fuel_history          – per-step total fuel consumed (litres)
            load_shedding_events  – number of timesteps where shedding occurred
        """
        state = env.reset()
        done  = False

        # ── Accumulators ──────────────────────────────────────────────────────
        total_reward     = 0.0
        total_fuel       = 0.0
        total_unmet      = 0.0
        total_demand_kwh = 0.0
        total_served_kwh = 0.0
        shedding_events  = 0

        # Per-step history lists
        soc_history         = [[] for _ in range(env.n_batteries)]
        reward_history      = []
        load_served_history = []
        fuel_history        = []

        # ── Main simulation loop ──────────────────────────────────────────────
        while not done:
            action = self.select_action(state, env)
            next_state, reward, done, info = env.step(action)

            # Fuel consumed this step across all controllable sources (litres)
            step_fuel = float(sum(info["fuel_consumed_per_source"].values()))

            # Update accumulators
            total_reward     += reward
            total_fuel       += step_fuel
            total_unmet      += info["unmet_load_kw"]
            total_demand_kwh += info["total_demand_kw"]    # Δt = 1 h → kWh
            total_served_kwh += info["load_served_kw"]

            if info["load_shedding_occurred"]:
                shedding_events += 1

            # Record SoC for each battery (post-step values from info dict)
            for j, bat in enumerate(env.batteries):
                soc_history[j].append(info["soc_per_battery"][bat["name"]])

            reward_history.append(reward)
            load_served_history.append(info["load_served_kw"])
            fuel_history.append(step_fuel)

            state = next_state

        # ── Reliability index (Eq 3.14) ───────────────────────────────────────
        # RI = total energy successfully served / total energy demanded
        reliability_index = (
            total_served_kwh / total_demand_kwh if total_demand_kwh > 0.0 else 0.0
        )

        return {
            "total_reward":         total_reward,
            "total_fuel_litres":    total_fuel,
            "total_unmet_load_kwh": total_unmet,
            "reliability_index":    reliability_index,
            "soc_history":          soc_history,
            "reward_history":       reward_history,
            "load_served_history":  load_served_history,
            "fuel_history":         fuel_history,
            "load_shedding_events": shedding_events,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _find_action(self, env, ctrl_pattern: list, batt_pattern: list) -> int:
        """
        Linear scan through env.action_map to find the integer index whose
        controllable and battery sub-vectors match the requested patterns.

        This avoids hardcoding action indices, which vary with system size.

        Parameters
        ----------
        env          : HybridEnergyEnv
        ctrl_pattern : list of int   Desired on/off values per controllable source.
        batt_pattern : list of int   Desired charge/idle/discharge per battery.

        Returns
        -------
        action_idx : int
            Index of the first matching action.  Falls back to 0 if no exact
            match is found (should not happen with a valid config).
        """
        for idx, action_spec in enumerate(env.action_map):
            if (action_spec["controllable"] == ctrl_pattern
                    and action_spec["batteries"] == batt_pattern):
                return idx
        # Fallback: return index 0 and warn — indicates a misconfiguration
        print(
            f"[RuleBasedController] WARNING: no action matched "
            f"ctrl={ctrl_pattern}, batt={batt_pattern}. Returning action 0."
        )
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Entry point — FUNAAB benchmark run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from data_generator import generate_data
    from energy_env import HybridEnergyEnv

    # ── Build FUNAAB configuration (mirrors energy_env.py test) ──────────────
    solar_irradiance, load_demand = generate_data()

    # Convert irradiance (W/m²) to availability fraction; peak ≈ 1000 W/m²
    solar_availability  = np.clip(solar_irradiance / 1000.0, 0.0, 1.0)
    diesel_availability = np.ones(8760, dtype=np.float64)

    config_funaab = {
        "sources": [
            {
                "name":                 "solar_pv",
                "type":                 "renewable",
                "rated_capacity_kw":    800.0,
                "availability_profile": solar_availability,
            },
            {
                "name":                 "diesel_generator",
                "type":                 "controllable",
                "rated_capacity_kw":    1000.0,
                "fuel_coefficient_a":   0.084,   # L/kWh — variable term  (Eq 3.7)
                "fuel_coefficient_b":   0.246,   # L/kWh — no-load term   (Eq 3.7)
                "availability_profile": diesel_availability,
            },
        ],
        "batteries": [
            {
                "name":                  "bess_1",
                "capacity_kwh":          3000.0,
                "max_charge_rate_kw":    600.0,
                "max_discharge_rate_kw": 600.0,
                "charge_efficiency":     0.95,
                "discharge_efficiency":  0.95,
                "soc_min":               0.20,
                "soc_max":               0.95,
                "initial_soc":           0.50,
            },
        ],
        "load_priorities": [
            {"name": "critical",      "fraction": 0.20, "sheddable": False},
            {"name": "essential",     "fraction": 0.50, "sheddable": True},
            {"name": "non_essential", "fraction": 0.30, "sheddable": True},
        ],
        "load_profile": load_demand,
    }

    # ── Instantiate environment and controller ────────────────────────────────
    env        = HybridEnergyEnv(config_funaab)
    controller = RuleBasedController(env)

    print("\nRunning rule-based controller for one full episode (8,760 timesteps)…")
    results = controller.run_episode(env)

    # ── Print full results dictionary ─────────────────────────────────────────
    import pprint
    print("\n── Full results dictionary ──────────────────────────────────────────")
    display = {
        k: (f"[list of {len(v)} values]" if isinstance(v, list) and len(v) > 10 else v)
        for k, v in results.items()
        if k != "soc_history"
    }
    display["soc_history"] = [
        f"[list of {len(h)} SoC values for battery {i}]"
        for i, h in enumerate(results["soc_history"])
    ]
    pprint.pprint(display, sort_dicts=False, width=72)

    # ── Clean summary ─────────────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────────────────────")
    print(f"  Total reward          : {results['total_reward']:>12.2f}")
    print(f"  Total fuel consumed   : {results['total_fuel_litres']:>12.2f} litres")
    print(f"  Reliability index     : {results['reliability_index'] * 100:>11.2f} %")
    print(f"  Total unmet load      : {results['total_unmet_load_kwh']:>12.2f} kWh")
    print(f"  Load shedding events  : {results['load_shedding_events']:>12d} timesteps")
    print("─────────────────────────────────────────────────────────────────────")

    # ── Save plots ────────────────────────────────────────────────────────────
    os.makedirs("plots", exist_ok=True)
    timesteps = np.arange(1, len(results["reward_history"]) + 1)

    # Plot 1: per-step and cumulative reward history
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(timesteps, results["reward_history"],
                 color="steelblue", linewidth=0.6, alpha=0.8)
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Rule-Based Controller — Per-Step Reward")
    axes[0].grid(True, linewidth=0.4, alpha=0.5)

    cumulative_reward = np.cumsum(results["reward_history"])
    axes[1].plot(timesteps, cumulative_reward, color="darkorange", linewidth=1.2)
    axes[1].set_ylabel("Cumulative Reward")
    axes[1].set_xlabel("Timestep (hour of year)")
    axes[1].set_title("Rule-Based Controller — Cumulative Reward")
    axes[1].grid(True, linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    reward_plot_path = os.path.join("plots", "rule_based_reward_history.png")
    fig.savefig(reward_plot_path, dpi=150)
    plt.close(fig)
    print(f"\n  Reward history plot saved → {reward_plot_path}")

    # Plot 2: battery SoC history with rule threshold annotations
    fig, ax = plt.subplots(figsize=(14, 5))

    for j, soc_vals in enumerate(results["soc_history"]):
        bat_name = config_funaab["batteries"][j]["name"]
        ax.plot(timesteps, soc_vals, linewidth=0.8,
                label=f"{bat_name} SoC", alpha=0.85)

    ax.axhline(RuleBasedController.SOC_THRESHOLD_HIGH, color="orange",
               linestyle="--", linewidth=0.9,
               label=f"SoC = {RuleBasedController.SOC_THRESHOLD_HIGH:.2f} (Rule 2/3 boundary)")
    ax.axhline(RuleBasedController.SOC_THRESHOLD_LOW, color="red",
               linestyle="--", linewidth=0.9,
               label=f"SoC = {RuleBasedController.SOC_THRESHOLD_LOW:.2f} (Rule 3/4 boundary)")

    ax.set_xlabel("Timestep (hour of year)")
    ax.set_ylabel("State of Charge (fraction)")
    ax.set_title("Rule-Based Controller — Battery SoC History")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    soc_plot_path = os.path.join("plots", "rule_based_soc_history.png")
    fig.savefig(soc_plot_path, dpi=150)
    plt.close(fig)
    print(f"  SoC history plot saved     → {soc_plot_path}")
