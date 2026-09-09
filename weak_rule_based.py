"""
weak_rule_based.py
==================
Conventional rule-based controller and its simulation environment,
implementing the five operational characteristics documented in
university_metadata.json that define the 70.51% energy reliability baseline.

This module provides:
  WeakGreenfieldEnv       — environment subclass that enforces the dispatch
                            constraints of the conventional controller
  WeakRuleBasedController — the controller's decision logic

Both classes are used only in evaluate.py for baseline comparison.
"""

import numpy as np
from formulas import GreenfieldEnergyEnv


# ─────────────────────────────────────────────────────────────────────────────
# Environment with conventional controller constraints
# ─────────────────────────────────────────────────────────────────────────────

class WeakGreenfieldEnv(GreenfieldEnergyEnv):
    """
    Simulation environment that reproduces the operational behaviour of the
    conventional rule-based controller deployed at Greenfield University.

    Overrides three aspects of the base environment:
      - LP viability threshold raised to 80% (vs 50% in the base env)
      - Load priority order fixed at daytime ordering for all hours
      - Step logic modified to implement 10% solar pre-charge and
        full-capacity diesel dispatch
    """

    # Conventional controller sheds a load point unless 80% of its demand can
    # be met — more aggressive than the 50% threshold used in the DQN environment
    LP_MIN_FRACTION = {
        "LP1": 0.0,
        "LP2": 0.8,
        "LP3": 0.8,
        "LP4": 0.8,
        "LP5": 0.8,
        "LP6": 0.8,
        "LP7": 0.8,
        "LP8": 0.0,
    }

    # ── Load dispatch (fixed day-order priority at all hours) ─────────────────

    def _dispatch_loads(self, available_power, hour, lp_demands, eligible_lps):
        """
        Priority dispatch with two conventional constraints applied:
          - Always uses the daytime priority order regardless of hour
            (LP8 street lighting is therefore always last priority)
          - 80% viability threshold via the overridden LP_MIN_FRACTION
        """
        # Night priority reordering is absent — always use the daytime order
        priority_order = self.DAY_PRIORITY
        ordered   = [lp for lp in priority_order if lp in eligible_lps]
        remaining = available_power
        lp_served = {lp: 0.0 for lp in self.LP_IDS}

        for lp in ordered:
            demand = lp_demands.get(lp, 0.0)
            if demand <= 1e-6:
                continue
            min_frac = self.LP_MIN_FRACTION[lp]
            if remaining >= demand:
                lp_served[lp] = demand
                remaining -= demand
            elif remaining >= demand * min_frac:
                lp_served[lp] = remaining
                remaining = 0.0

            if remaining <= 0.0:
                break

        eligible_demand = sum(lp_demands.get(lp, 0.0) for lp in ordered)
        total_served    = sum(lp_served.values())
        unmet_load      = max(0.0, eligible_demand - total_served)
        return lp_served, unmet_load, unmet_load > 1e-3

    # ── Timestep (modified for pre-charge bias and full-capacity diesel) ──────

    def step(self, action: int):
        """
        Executes one simulation timestep with two additional constraints:
          - 10% of solar output is diverted to charge the battery before
            any load is served (pre-charge bias)
          - When the diesel generator is activated, it always runs at its
            full rated capacity of 150 kW regardless of the actual shortfall
        """
        if not (0 <= action < self.action_size):
            raise ValueError(f"Action {action} out of range.")

        spec          = self.ACTION_MAP[action]
        diesel_on_req = spec["diesel"]
        batt_mode     = spec["battery"]

        t            = self.t
        hour         = int(self.hour_of_day[t])
        solar_kw     = float(self.solar_output_kw[t])
        diesel_avail = bool(self.diesel_available[t])
        fuel_price   = float(self.fuel_price[t])
        soc_kwh      = self.battery_soc_kwh
        lp_demands   = {lp: float(self.lp_demand_kw[lp][t]) for lp in self.LP_IDS}

        # ── 10% solar pre-charge (diverted before loads are considered) ───────
        pre_charge_kw    = solar_kw * 0.10
        solar_for_loads  = solar_kw * 0.90

        eligible_lps       = self._get_eligible_lps(hour, soc_kwh, solar_kw)
        eligible_demand_kw = sum(lp_demands.get(lp, 0.0) for lp in eligible_lps)

        # ── Battery discharge (against reduced solar available for loads) ──────
        if batt_mode == 2:
            solar_deficit      = max(0.0, eligible_demand_kw - solar_for_loads)
            depth_kwh          = max(0.0, soc_kwh - self.BATTERY_MIN_SOC_KWH)
            max_from_depth     = depth_kwh * self.BATTERY_ETA
            battery_discharge_kw = min(self.BATTERY_MAX_DISCHARGE_KW,
                                       max_from_depth,
                                       solar_deficit)
        else:
            battery_discharge_kw = 0.0

        # ── Diesel dispatch: always full 150 kW when activated ────────────────
        diesel_kw = 0.0
        if diesel_on_req and diesel_avail:
            supply_excl_diesel = solar_for_loads + battery_discharge_kw
            shortfall          = max(0.0, eligible_demand_kw - supply_excl_diesel)
            low_soc_low_solar  = (soc_kwh < self.DIESEL_EMERGENCY_SOC_KWH
                                  and solar_kw < self.DIESEL_EMERGENCY_SOLAR_KW)
            if shortfall > 0.0 or low_soc_low_solar:
                diesel_kw = self.DIESEL_CAPACITY_KW   # always full capacity

        total_supply = solar_for_loads + battery_discharge_kw + diesel_kw

        # ── Battery charging from surplus ─────────────────────────────────────
        if batt_mode == 0:
            surplus_kw        = max(0.0, total_supply - eligible_demand_kw)
            headroom_kwh      = max(0.0, self.BATTERY_MAX_SOC_KWH - soc_kwh)
            max_from_headroom = headroom_kwh / self.BATTERY_ETA
            battery_charge_kw = min(self.BATTERY_MAX_CHARGE_KW,
                                    surplus_kw,
                                    max_from_headroom)
        else:
            battery_charge_kw = 0.0

        net_available = total_supply - battery_charge_kw

        # ── Priority dispatch (overridden _dispatch_loads handles F3 and F4) ──
        lp_served, unmet_load, shedding_occurred = self._dispatch_loads(
            net_available, hour, lp_demands, eligible_lps
        )
        load_served_kw = sum(lp_served.values())

        # ── Update battery SOC ────────────────────────────────────────────────
        new_soc = soc_kwh
        if battery_discharge_kw > 1e-9:
            new_soc = max(self.BATTERY_MIN_SOC_KWH,
                          new_soc - battery_discharge_kw / self.BATTERY_ETA)
        if battery_charge_kw > 1e-9:
            new_soc = min(self.BATTERY_MAX_SOC_KWH,
                          new_soc + battery_charge_kw * self.BATTERY_ETA)

        # Pre-charge contribution to SOC
        if pre_charge_kw > 1e-9:
            headroom = max(0.0, self.BATTERY_MAX_SOC_KWH - new_soc)
            actual_pre_charge = min(pre_charge_kw, headroom / self.BATTERY_ETA)
            new_soc = min(self.BATTERY_MAX_SOC_KWH,
                          new_soc + actual_pre_charge * self.BATTERY_ETA)
        else:
            actual_pre_charge = 0.0

        self.battery_soc_kwh = new_soc
        soc_violated = (self.battery_soc_kwh <= self.BATTERY_MIN_SOC_KWH + 1e-3
                        or self.battery_soc_kwh >= self.BATTERY_MAX_SOC_KWH - 1e-3)

        # ── Fuel and reward ───────────────────────────────────────────────────
        fuel_consumed_litres = self._compute_diesel_fuel(diesel_kw)
        renewable_bonus = (1.0 if unmet_load <= 1e-6 and fuel_consumed_litres <= 1e-9
                           else 0.0)
        fuel_penalty = self.FUEL_WEIGHT      * fuel_consumed_litres
        batt_penalty = self.BATTERY_WEIGHT   * float(soc_violated)
        load_penalty = self.LOAD_WEIGHT      * unmet_load
        renew_reward = self.RENEWABLE_WEIGHT * renewable_bonus
        reward       = -fuel_penalty - batt_penalty - load_penalty + renew_reward

        self.t += 1
        done = self.t >= self.n_timesteps

        info = {
            "solar_output_kw":          solar_kw,
            "diesel_output_kw":         diesel_kw,
            "battery_soc_kwh":          self.battery_soc_kwh,
            "battery_charge_kw":        battery_charge_kw + actual_pre_charge,
            "battery_discharge_kw":     battery_discharge_kw,
            "lp_served_kw":             lp_served,
            "total_demand_kw":          eligible_demand_kw,
            "load_served_kw":           load_served_kw,
            "unmet_load_kw":            unmet_load,
            "load_shedding_occurred":   shedding_occurred,
            "fuel_consumed_per_source": {"diesel_generator": fuel_consumed_litres},
            "fuel_cost_NGN":            fuel_consumed_litres * fuel_price,
            "soc_violated":             soc_violated,
            "reward_components": {
                "fuel_penalty":    float(-fuel_penalty),
                "battery_penalty": float(-batt_penalty),
                "load_penalty":    float(-load_penalty),
                "renewable_bonus": float(renew_reward),
                "total_reward":    float(reward),
            },
        }

        next_state = (self._get_state() if not done
                      else np.zeros(self.state_size, dtype=np.float64))
        return next_state, reward, done, info


# ─────────────────────────────────────────────────────────────────────────────
# Conventional rule-based controller
# ─────────────────────────────────────────────────────────────────────────────

class WeakRuleBasedController:
    """
    Fixed-rule controller implementing the conventional dispatch heuristics,
    including the late diesel trigger and pre-charge-aware battery mode logic.
    """

    def __init__(self, env):
        self.env = env

    def select_action(self, state: np.ndarray, env) -> int:
        t            = env.t
        hour         = int(env.hour_of_day[t])
        solar_kw     = float(env.solar_output_kw[t])
        soc_kwh      = env.battery_soc_kwh
        diesel_avail = bool(env.diesel_available[t])
        lp_demands   = {lp: float(env.lp_demand_kw[lp][t]) for lp in env.LP_IDS}

        eligible_lps    = env._get_eligible_lps(hour, soc_kwh, solar_kw)
        eligible_demand = sum(lp_demands.get(lp, 0.0) for lp in eligible_lps)

        # Controller perceives only 90% of solar as available for loads
        solar_for_loads = solar_kw * 0.90

        # Late diesel trigger: only start when SOC is almost at the floor
        # AND solar is essentially zero — far too conservative
        diesel_on = diesel_avail and (soc_kwh <= 125.0 and solar_kw < 10.0)

        # Battery mode: based on perceived (90%) solar vs demand
        if solar_for_loads >= eligible_demand:
            batt_mode = 0 if soc_kwh < env.BATTERY_MAX_SOC_KWH - 1.0 else 1
        else:
            batt_mode = 2 if soc_kwh > env.BATTERY_MIN_SOC_KWH + 1.0 else 1

        return env._action_from_decisions(int(diesel_on), batt_mode)
