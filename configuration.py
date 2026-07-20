import os
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_DIR = os.path.join(_HERE, "data")


class GreenfieldEnergyEnv:


    SOLAR_CAPACITY_KW        = 500.0
    DIESEL_CAPACITY_KW       = 110.0
    DIESEL_MIN_LOAD_KW       = 55.0
    DIESEL_FULL_LOAD_L_HR    = 40.0
    DIESEL_IDLE_L_HR         = 10.0

    BATTERY_CAPACITY_KWH     = 1200.0
    BATTERY_MIN_SOC_KWH      = 120.0
    BATTERY_MAX_SOC_KWH      = 1080.0
    BATTERY_MAX_CHARGE_KW    = 200.0
    BATTERY_MAX_DISCHARGE_KW = 200.0
    BATTERY_ETA              = 0.90


    DIESEL_EMERGENCY_SOC_KWH  = 240.0
    DIESEL_EMERGENCY_SOLAR_KW = 30.0


    LP_IDS = ["LP1", "LP2", "LP3", "LP4", "LP5", "LP6", "LP7", "LP8"]


    DAY_PRIORITY   = ["LP1", "LP2", "LP3", "LP4", "LP5", "LP6", "LP7"]
    NIGHT_PRIORITY = ["LP1", "LP8", "LP4", "LP2", "LP3", "LP5", "LP6", "LP7"]


    LP_MIN_FRACTION = {
        "LP1": 0.0,
        "LP2": 0.6,
        "LP3": 0.7,
        "LP4": 0.6,
        "LP5": 0.7,
        "LP6": 0.7,
        "LP7": 0.8,
        "LP8": 0.0,
    }


    PRIORITY_LPS = {"LP1", "LP2", "LP3", "LP4"}


    BASELINE_ANNUAL_FUEL_L = 149_693.0


    FUEL_WEIGHT      = 0.005
    BATTERY_WEIGHT   = 0.05
    LOAD_WEIGHT      = 1.0
    RENEWABLE_WEIGHT = 0.5
    FUEL_BUDGET_WEIGHT = 2.0


    ACTION_MAP = [
        {"diesel": 0, "battery": 0},
        {"diesel": 0, "battery": 1},
        {"diesel": 0, "battery": 2},
        {"diesel": 1, "battery": 0},
        {"diesel": 1, "battery": 1},
        {"diesel": 1, "battery": 2},
        {"diesel": 0, "battery": 3},
    ]


    _PRE_CHARGE_FRAC = 0.40

    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = _DEFAULT_DATA_DIR
        self._load_data(data_dir)
        self.n_timesteps = len(self.solar_output_kw)


        self._max_lp_demand = {
            lp: float(np.max(self.lp_demand_kw[lp])) for lp in self.LP_IDS
        }
        self._min_fuel_price = float(np.min(self.fuel_price))
        self._max_fuel_price = float(np.max(self.fuel_price))


        self._hourly_fuel_budget = self.BASELINE_ANNUAL_FUEL_L / 8760.0

        self.battery_soc_kwh = self._initial_soc_kwh
        self.t = 0

        print(
            f"[GreenfieldEnergyEnv] Loaded {self.n_timesteps} timesteps | "
            f"state_size={self.state_size} | action_size={self.action_size} | "
            f"initial SOC={self._initial_soc_kwh:.1f} kWh"
        )


    @property
    def state_size(self) -> int:

        return 22

    @property
    def action_size(self) -> int:
        return len(self.ACTION_MAP)


    def reset(self, demand_jitter: float = 0.05) -> np.ndarray:
        self.t = 0
        self.battery_soc_kwh = self._initial_soc_kwh

        if demand_jitter > 0.0:
            for lp in self.LP_IDS:
                noise = np.random.normal(1.0, demand_jitter, self.n_timesteps)
                noise = np.clip(noise, 1.0 - 3*demand_jitter, 1.0 + 3*demand_jitter)
                self.lp_demand_kw[lp] = np.clip(
                    self._base_lp_demand_kw[lp] * noise,
                    0.0,
                    self._max_lp_demand[lp] * 1.2,
                )
        else:
            self.lp_demand_kw = {
                lp: arr.copy() for lp, arr in self._base_lp_demand_kw.items()
            }
        return self._get_state()

    def step(self, action: int):
        if not (0 <= action < self.action_size):
            raise ValueError(f"Action {action} out of range [0, {self.action_size-1}].")

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
        cum_fuel_today = float(self.cum_daily_fuel[t])


        pre_charge_kw = 0.0
        if batt_mode == 3:
            headroom  = (self.BATTERY_MAX_SOC_KWH - soc_kwh) / self.BATTERY_ETA
            pre_charge_kw = min(
                solar_kw * self._PRE_CHARGE_FRAC,
                self.BATTERY_MAX_CHARGE_KW,
                max(0.0, headroom),
            )
            stored = pre_charge_kw * self.BATTERY_ETA
            self.battery_soc_kwh = min(self.BATTERY_MAX_SOC_KWH,
                                       self.battery_soc_kwh + stored)
            soc_kwh = self.battery_soc_kwh

        effective_solar_kw = solar_kw - pre_charge_kw


        eligible_lps      = self._get_eligible_lps(hour, soc_kwh, solar_kw)
        eligible_demand_kw = sum(lp_demands.get(lp, 0.0) for lp in eligible_lps)
        priority_demand_kw = sum(
            lp_demands.get(lp, 0.0)
            for lp in eligible_lps if lp in self.PRIORITY_LPS
        )


        battery_discharge_kw = 0.0
        if batt_mode == 2:
            solar_deficit        = max(0.0, eligible_demand_kw - effective_solar_kw)
            available_from_soc   = (soc_kwh - self.BATTERY_MIN_SOC_KWH) * self.BATTERY_ETA
            battery_discharge_kw = min(
                self.BATTERY_MAX_DISCHARGE_KW,
                solar_deficit,
                max(0.0, available_from_soc),
            )


        diesel_kw = 0.0
        if diesel_on_req and diesel_avail:
            supply_without_diesel = effective_solar_kw + battery_discharge_kw
            shortfall             = max(0.0, priority_demand_kw - supply_without_diesel)
            low_soc_low_solar     = (
                soc_kwh < self.DIESEL_EMERGENCY_SOC_KWH
                and solar_kw < self.DIESEL_EMERGENCY_SOLAR_KW
            )
            if shortfall > 0.0 or low_soc_low_solar:
                target    = shortfall if shortfall > 0.0 else self.DIESEL_MIN_LOAD_KW
                diesel_kw = max(self.DIESEL_MIN_LOAD_KW,
                                min(target, self.DIESEL_CAPACITY_KW))

        total_supply = effective_solar_kw + battery_discharge_kw + diesel_kw


        battery_charge_kw = 0.0
        if batt_mode == 0:
            surplus   = max(0.0, total_supply - eligible_demand_kw)
            headroom  = (self.BATTERY_MAX_SOC_KWH - soc_kwh) / self.BATTERY_ETA
            battery_charge_kw = min(
                self.BATTERY_MAX_CHARGE_KW,
                surplus,
                max(0.0, headroom),
            )

        net_available = total_supply - battery_charge_kw


        lp_served, unmet_load, shedding_occurred = self._dispatch_loads(
            net_available, hour, lp_demands, eligible_lps
        )
        load_served_kw = sum(lp_served.values())


        soc_violated = False
        if battery_discharge_kw > 1e-9:
            draw = battery_discharge_kw / self.BATTERY_ETA
            self.battery_soc_kwh = max(self.BATTERY_MIN_SOC_KWH,
                                       self.battery_soc_kwh - draw)
        if battery_charge_kw > 1e-9:
            self.battery_soc_kwh = min(self.BATTERY_MAX_SOC_KWH,
                                       self.battery_soc_kwh + battery_charge_kw * self.BATTERY_ETA)
        if (self.battery_soc_kwh <= self.BATTERY_MIN_SOC_KWH + 1e-3
                or self.battery_soc_kwh >= self.BATTERY_MAX_SOC_KWH - 1e-3):
            soc_violated = True


        fuel_consumed_litres = self._compute_diesel_fuel(diesel_kw)
        fuel_cost_ngn        = fuel_consumed_litres * fuel_price


        hours_into_day      = hour + 1
        daily_budget_so_far = self._hourly_fuel_budget * hours_into_day
        fuel_budget_penalty = max(0.0, cum_fuel_today + fuel_consumed_litres
                                  - daily_budget_so_far)


        renewable_bonus    = (1.0 if unmet_load <= 1e-6
                              and fuel_consumed_litres <= 1e-9 else 0.0)
        fuel_penalty       = self.FUEL_WEIGHT        * fuel_consumed_litres
        batt_penalty       = self.BATTERY_WEIGHT     * float(soc_violated)
        load_penalty       = self.LOAD_WEIGHT        * unmet_load
        renew_reward       = self.RENEWABLE_WEIGHT   * renewable_bonus
        budget_penalty     = self.FUEL_BUDGET_WEIGHT * fuel_budget_penalty
        reward = (-fuel_penalty - batt_penalty - load_penalty
                  + renew_reward - budget_penalty)


        self.t += 1
        done = self.t >= self.n_timesteps

        info = {
            "hour":                     hour,
            "solar_output_kw":          solar_kw,
            "effective_solar_kw":       effective_solar_kw,
            "pre_charge_kw":            pre_charge_kw,
            "diesel_output_kw":         diesel_kw,
            "battery_soc_kwh":          self.battery_soc_kwh,
            "battery_charge_kw":        battery_charge_kw,
            "battery_discharge_kw":     battery_discharge_kw,
            "lp_served_kw":             lp_served,
            "total_demand_kw":          eligible_demand_kw,
            "load_served_kw":           load_served_kw,
            "unmet_load_kw":            unmet_load,
            "load_shedding_occurred":   shedding_occurred,
            "fuel_consumed_per_source": {"diesel_generator": fuel_consumed_litres},
            "fuel_cost_NGN":            fuel_cost_ngn,
            "soc_violated":             soc_violated,
            "fuel_budget_penalty":      fuel_budget_penalty,
            "reward_components": {
                "fuel_penalty":    float(-fuel_penalty),
                "battery_penalty": float(-batt_penalty),
                "load_penalty":    float(-load_penalty),
                "renewable_bonus": float(renew_reward),
                "budget_penalty":  float(-budget_penalty),
                "total_reward":    float(reward),
            },
        }

        next_state = (self._get_state() if not done
                      else np.zeros(self.state_size, dtype=np.float64))
        return next_state, reward, done, info


    def _get_state(self) -> np.ndarray:
        t = self.t


        solar_frac   = float(self.solar_output_kw[t]) / self.SOLAR_CAPACITY_KW
        soc_norm     = ((self.battery_soc_kwh - self.BATTERY_MIN_SOC_KWH)
                        / (self.BATTERY_MAX_SOC_KWH - self.BATTERY_MIN_SOC_KWH))
        hour_rad     = 2.0 * np.pi * float(self.hour_of_day[t]) / 24.0
        hour_sin     = np.sin(hour_rad)
        hour_cos     = np.cos(hour_rad)
        weekday      = float(self.is_weekday[t])
        diesel_avail = float(self.diesel_available[t])


        lp_norms = np.array(
            [float(self.lp_demand_kw[lp][t]) / self._max_lp_demand[lp]
             for lp in self.LP_IDS],
            dtype=np.float64,
        )


        price_range = self._max_fuel_price - self._min_fuel_price
        price_norm  = ((float(self.fuel_price[t]) - self._min_fuel_price) / price_range
                       if price_range > 0.0 else 0.5)


        sol_fc_t1  = float(self.solar_fc_t1[t])  / self.SOLAR_CAPACITY_KW
        sol_fc_t3  = float(self.solar_fc_t3[t])  / self.SOLAR_CAPACITY_KW
        sol_fc_t6  = float(self.solar_fc_t6[t])  / self.SOLAR_CAPACITY_KW


        total_lp_cap = 360.0
        dem_fc_t1  = float(self.demand_fc_t1[t]) / total_lp_cap
        dem_fc_t3  = float(self.demand_fc_t3[t]) / total_lp_cap
        dem_fc_t6  = float(self.demand_fc_t6[t]) / total_lp_cap


        daily_budget = self._hourly_fuel_budget * 24.0
        cum_fuel_norm = float(self.cum_daily_fuel[t]) / daily_budget

        state = np.concatenate([
            [solar_frac, np.clip(soc_norm, 0.0, 1.0),
             hour_sin, hour_cos, weekday, diesel_avail],
            np.clip(lp_norms, 0.0, 1.0),
            [np.clip(price_norm, 0.0, 1.0)],
            [np.clip(sol_fc_t1, 0.0, 1.0),
             np.clip(sol_fc_t3, 0.0, 1.0),
             np.clip(sol_fc_t6, 0.0, 1.0)],
            [np.clip(dem_fc_t1, 0.0, 1.5),
             np.clip(dem_fc_t3, 0.0, 1.5),
             np.clip(dem_fc_t6, 0.0, 1.5)],
            [np.clip(cum_fuel_norm, 0.0, 3.0)],
        ])
        return state.astype(np.float64)


    def _load_data(self, data_dir: str) -> None:
        src  = pd.read_csv(os.path.join(data_dir, "source_availability.csv"))
        load = pd.read_csv(os.path.join(data_dir, "load_demand_and_dispatch.csv"))

        self.solar_output_kw  = src["solar_pv_output_kw"].to_numpy(dtype=np.float64)
        self.diesel_available = src["diesel_available"].to_numpy(dtype=np.int32)
        self.fuel_price       = src["diesel_fuel_price_NGN_per_litre"].to_numpy(dtype=np.float64)
        self.hour_of_day      = src["hour_of_day"].to_numpy(dtype=np.int32)
        self.is_weekday       = src["is_weekday"].to_numpy(dtype=np.float32)
        self._initial_soc_kwh = float(src["battery_soc_kwh"].iloc[0])


        self.solar_fc_t1   = src["solar_forecast_t1h_kw"].to_numpy(dtype=np.float64)
        self.solar_fc_t3   = src["solar_forecast_t3h_kw"].to_numpy(dtype=np.float64)
        self.solar_fc_t6   = src["solar_forecast_t6h_kw"].to_numpy(dtype=np.float64)
        self.demand_fc_t1  = src["demand_forecast_t1h_kw"].to_numpy(dtype=np.float64)
        self.demand_fc_t3  = src["demand_forecast_t3h_kw"].to_numpy(dtype=np.float64)
        self.demand_fc_t6  = src["demand_forecast_t6h_kw"].to_numpy(dtype=np.float64)
        self.cum_daily_fuel = src["cumulative_daily_fuel_litres"].to_numpy(dtype=np.float64)

        self._base_lp_demand_kw = {
            lp: load[f"{lp}_demand_kw"].to_numpy(dtype=np.float64)
            for lp in self.LP_IDS
        }
        self.lp_demand_kw = {
            lp: arr.copy() for lp, arr in self._base_lp_demand_kw.items()
        }


    def _get_eligible_lps(self, hour: int, soc_kwh: float, solar_kw: float) -> set:
        eligible = {lp for lp in self.LP_IDS if self._lp_in_time_window(lp, hour)}

        if (soc_kwh < self.DIESEL_EMERGENCY_SOC_KWH
                and solar_kw < self.DIESEL_EMERGENCY_SOLAR_KW):
            eligible.discard("LP7")
        return eligible

    def _lp_in_time_window(self, lp: str, hour: int) -> bool:
        if lp in ("LP1", "LP2", "LP3", "LP4"):
            return True
        if lp == "LP5":
            return 8 <= hour <= 22
        if lp == "LP6":
            return 6 <= hour <= 21
        if lp == "LP7":
            return 6 <= hour <= 20
        if lp == "LP8":
            return hour >= 18 or hour <= 6
        return False

    def _dispatch_loads(
        self,
        available_power: float,
        hour: int,
        lp_demands: dict,
        eligible_lps: set,
    ):
        priority_order = (self.DAY_PRIORITY if 7 <= hour <= 17
                          else self.NIGHT_PRIORITY)
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
                remaining    -= demand
            elif remaining >= demand * min_frac:

                lp_served[lp] = remaining
                remaining      = 0.0

            if remaining <= 0.0:
                break

        eligible_demand = sum(lp_demands.get(lp, 0.0) for lp in ordered)
        total_served    = sum(lp_served.values())
        unmet_load      = max(0.0, eligible_demand - total_served)
        return lp_served, unmet_load, unmet_load > 1e-3

    def _compute_diesel_fuel(self, diesel_kw: float) -> float:
        if diesel_kw <= 1e-9:
            return 0.0
        return (self.DIESEL_IDLE_L_HR
                + (self.DIESEL_FULL_LOAD_L_HR - self.DIESEL_IDLE_L_HR)
                * (diesel_kw / self.DIESEL_CAPACITY_KW))

    def _action_from_decisions(self, diesel_on: int, batt_mode: int) -> int:
        if batt_mode == 3:
            return 6
        return int(diesel_on) * 3 + int(batt_mode)

    def __repr__(self) -> str:
        return (
            f"GreenfieldEnergyEnv("
            f"t={self.t}/{self.n_timesteps}, "
            f"soc={self.battery_soc_kwh:.1f} kWh, "
            f"state_size={self.state_size}, "
            f"action_size={self.action_size})"
        )


if __name__ == "__main__":
    import pprint
    pp = pprint.PrettyPrinter(indent=2, width=88, sort_dicts=False)

    env   = GreenfieldEnergyEnv()
    state = env.reset()

    print(f"\nenv repr      : {env}")
    print(f"state_size    : {env.state_size}")
    print(f"action_size   : {env.action_size}")
    print(f"initial SOC   : {env.battery_soc_kwh:.1f} kWh")
    print(f"\nInitial state ({env.state_size} features):")
    labels = (
        ["solar_frac", "soc_norm", "hour_sin", "hour_cos", "weekday", "diesel_avail"]
        + [f"{lp}_norm" for lp in env.LP_IDS]
        + ["price_norm",
           "solar_fc_t1h", "solar_fc_t3h", "solar_fc_t6h",
           "demand_fc_t1h", "demand_fc_t3h", "demand_fc_t6h",
           "cum_fuel_norm"]
    )
    for label, val in zip(labels, state):
        print(f"  {label:<20}: {val:.4f}")

    print("\n─── Running 7 manual timesteps (one per action) ───────────────────")
    action_labels = [
        "OFF+CHARGE", "OFF+IDLE", "OFF+DISCHARGE",
        "ON+CHARGE",  "ON+IDLE",  "ON+DISCHARGE",
        "OFF+PRE-CHARGE",
    ]
    for action, label in enumerate(action_labels):
        next_state, reward, done, info = env.step(action)
        print(f"\nStep {env.t}  action={action} ({label})")
        print(f"  solar={info['solar_output_kw']:.1f} kW  "
              f"diesel={info['diesel_output_kw']:.1f} kW  "
              f"SOC={info['battery_soc_kwh']:.1f} kWh")
        print(f"  demand={info['total_demand_kw']:.1f} kW  "
              f"served={info['load_served_kw']:.1f} kW  "
              f"unmet={info['unmet_load_kw']:.1f} kW")
        print(f"  fuel={sum(info['fuel_consumed_per_source'].values()):.2f} L  "
              f"reward={reward:.4f}")
        if done:
            print("  [Episode finished]")
            break
