"""
formulas.py
===========
Data-driven simulation environment for the Greenfield University off-grid
hybrid energy system, built for Deep Q-Network (DQN) reinforcement learning.

Class: GreenfieldEnergyEnv

System (from university_metadata.json):
  Sources : 500 kW solar PV (renewable) + 150 kW diesel generator (dispatchable)
  Storage : 1,200 kWh battery bank  (120–1,080 kWh operational range, 200 kW rate)
  Loads   : 8 load points (LP1–LP8) with hourly demand profiles (8,760 steps)

Data files (data/Greenfield/data/):
  source_availability.csv      – hourly solar output, diesel availability/price,
                                  initial battery SOC
  load_demand_and_dispatch.csv – per-LP hourly unconstrained demand profiles

Action space (6 discrete actions):
  diesel ∈ {0=OFF, 1=ON}  ×  battery ∈ {0=CHARGE, 1=IDLE, 2=DISCHARGE}
  index = diesel_on × 3 + batt_mode

State vector (13 features, all normalised to [0, 1]):
  [solar_frac, soc_norm, hour_norm, is_weekday,
   lp1_norm, …, lp8_norm, fuel_price_norm]

Reward (per timestep):
  r(t) = −w_f·F(t) − w_b·B(t) − w_l·L(t) + w_r·R(t)
  F(t) = diesel fuel consumed (litres)       w_f = 1.0
  B(t) = battery SOC bound violation (0/1)   w_b = 3.0
  L(t) = unmet eligible load (kW)            w_l = 20.0
  R(t) = 1 if load fully met and no fuel     w_r = 2.0

Controller rules reference (university_metadata.json):
  R1  Solar-first dispatch
  R2  Daytime priority order  (07:00–17:59)
  R3  Night priority order    (18:00–06:59)
  R4  Load shedding threshold (50% for LP2–LP7; 0% for LP1/LP8)
  R5  Battery charges on surplus
  R6  Battery discharges on deficit
  R7  Diesel start condition
  R8  Diesel output sizing    (25%–100% of 150 kW)
  R9  Diesel fuel cost        (linear derating model)
  R10 Time-window restrictions
  R11 Emergency battery protection
"""

import os
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_DIR = os.path.join(_HERE, "data", "Greenfield", "data")


class GreenfieldEnergyEnv:
    """
    Simulation environment for Greenfield University's off-grid hybrid energy
    system.  Loads one full year of real hourly data (8,760 timesteps) from
    CSV files and exposes the standard RL interface: reset / step / state.

    The agent controls two decisions per timestep:
      - Diesel generator: ON (1) or OFF (0)
      - Battery:          CHARGE (0), IDLE (1), or DISCHARGE (2)

    All priority-based load dispatch, time-window restrictions, emergency
    battery protection, and fuel physics are handled deterministically by the
    environment (rules R2–R4, R8–R11).  The agent learns when to start the
    diesel (R7) and when to store vs. release battery energy (R5–R6).
    """

    # ── System constants (university_metadata.json) ───────────────────────────
    SOLAR_CAPACITY_KW      = 500.0
    DIESEL_CAPACITY_KW     = 150.0
    DIESEL_MIN_LOAD_KW     = 37.5        # 25% × 150 kW  (R8)
    DIESEL_FULL_LOAD_L_HR  = 40.0        # at 150 kW     (R9)
    DIESEL_IDLE_L_HR       = 10.0        # no-load rate  (R9)

    BATTERY_CAPACITY_KWH     = 1200.0
    BATTERY_MIN_SOC_KWH      = 120.0     # 10%  (R6, R7, R11)
    BATTERY_MAX_SOC_KWH      = 1080.0    # 90%  (R5)
    BATTERY_MAX_CHARGE_KW    = 200.0
    BATTERY_MAX_DISCHARGE_KW = 200.0
    BATTERY_ETA              = 0.9 ** 0.5   # per-direction efficiency ≈ 0.9487

    # Emergency thresholds (R7b, R11)
    DIESEL_EMERGENCY_SOC_KWH  = 240.0    # 20% of 1,200 kWh
    DIESEL_EMERGENCY_SOLAR_KW = 30.0

    # ── Load point identifiers ────────────────────────────────────────────────
    LP_IDS = ["LP1", "LP2", "LP3", "LP4", "LP5", "LP6", "LP7", "LP8"]

    # Priority dispatch order (R2: day 07:00–17:59, R3: night 18:00–06:59)
    DAY_PRIORITY   = ["LP1", "LP2", "LP3", "LP4", "LP5", "LP6", "LP7"]
    NIGHT_PRIORITY = ["LP1", "LP8", "LP4", "LP2", "LP3", "LP5", "LP6", "LP7"]

    # Minimum served fraction before an LP is considered viable (R4)
    # LP1 and LP8 are always served if any power is available (threshold = 0)
    LP_MIN_FRACTION = {
        "LP1": 0.0, "LP2": 0.5, "LP3": 0.5, "LP4": 0.5,
        "LP5": 0.5, "LP6": 0.5, "LP7": 0.5, "LP8": 0.0,
    }

    # ── Reward weights ────────────────────────────────────────────────────────
    # Scaled so per-step rewards stay in roughly [-5, +1] rather than
    # [-thousands], which prevents gradient explosion in the Q-network.
    # LOAD_WEIGHT dominates to make reliability the primary objective.
    FUEL_WEIGHT      = 0.005   # per litre — marginal cost signal
    BATTERY_WEIGHT   = 0.05    # SOC bound violation discouraged
    LOAD_WEIGHT      = 1.0     # per kW unmet — reliability is the priority
    RENEWABLE_WEIGHT = 0.5     # bonus for zero-fuel zero-shed steps

    # ── Action map (6 actions) ────────────────────────────────────────────────
    # action index = diesel_on * 3 + batt_mode
    ACTION_MAP = [
        {"diesel": 0, "battery": 0},   # 0: OFF  + CHARGE
        {"diesel": 0, "battery": 1},   # 1: OFF  + IDLE
        {"diesel": 0, "battery": 2},   # 2: OFF  + DISCHARGE
        {"diesel": 1, "battery": 0},   # 3: ON   + CHARGE
        {"diesel": 1, "battery": 1},   # 4: ON   + IDLE
        {"diesel": 1, "battery": 2},   # 5: ON   + DISCHARGE
    ]

    # ─────────────────────────────────────────────────────────────────────────
    # Initialisation
    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self, data_dir=None):
        """
        Load CSV data from data_dir and prepare the environment.

        Parameters
        ----------
        data_dir : str, optional
            Path to the directory containing source_availability.csv and
            load_demand_and_dispatch.csv.  Defaults to
            <project_root>/data/Greenfield/data/.
        """
        if data_dir is None:
            data_dir = _DEFAULT_DATA_DIR

        self._load_data(data_dir)
        self.n_timesteps = len(self.solar_output_kw)

        # Per-LP normalisation denominators (observed max demand in the dataset)
        self._max_lp_demand = {
            lp: float(np.max(self.lp_demand_kw[lp])) for lp in self.LP_IDS
        }
        self._min_fuel_price = float(np.min(self.fuel_price))
        self._max_fuel_price = float(np.max(self.fuel_price))

        # Mutable episode state
        self.battery_soc_kwh = self._initial_soc_kwh
        self.t = 0

        print(
            f"[GreenfieldEnergyEnv] Loaded {self.n_timesteps} timesteps  |  "
            f"state_size={self.state_size}  |  action_size={self.action_size}  |  "
            f"initial SOC={self._initial_soc_kwh:.1f} kWh"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def state_size(self) -> int:
        """Length of the normalised observation vector (13 features)."""
        return 13

    @property
    def action_size(self) -> int:
        """Total number of discrete actions (6)."""
        return len(self.ACTION_MAP)

    # ─────────────────────────────────────────────────────────────────────────
    # Core RL interface
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """
        Reset to the start of the year (t=0) and restore the initial battery SOC.

        Returns
        -------
        state : np.ndarray, shape (13,)
        """
        self.t = 0
        self.battery_soc_kwh = self._initial_soc_kwh
        return self._get_state()

    def step(self, action: int):
        """
        Execute one simulation timestep.

        Sequence inside each step:
          1. Decode action into diesel_on and batt_mode.
          2. Battery discharge: compute deliverable kW from available depth.
          3. Determine eligible LPs for this hour (R10, R11).
          4. Diesel sizing: if agent requests ON, size to shortfall per R8.
          5. Battery charging: allocate surplus after loads to charging (R5).
          6. Priority-based load dispatch (R2, R3, R4).
          7. Update battery SOC for discharge then charge.
          8. Compute fuel consumption (R9) and reward.

        Parameters
        ----------
        action : int  Index in [0, 5].

        Returns
        -------
        next_state : np.ndarray, shape (13,)
        reward     : float
        done       : bool
        info       : dict
        """
        if not (0 <= action < self.action_size):
            raise ValueError(f"Action {action} out of range [0, {self.action_size - 1}].")

        spec         = self.ACTION_MAP[action]
        diesel_on_req = spec["diesel"]    # agent's diesel request
        batt_mode     = spec["battery"]   # 0=CHARGE, 1=IDLE, 2=DISCHARGE

        t            = self.t
        hour         = int(self.hour_of_day[t])
        solar_kw     = float(self.solar_output_kw[t])
        diesel_avail = bool(self.diesel_available[t])
        fuel_price   = float(self.fuel_price[t])
        soc_kwh      = self.battery_soc_kwh
        lp_demands   = {lp: float(self.lp_demand_kw[lp][t]) for lp in self.LP_IDS}

        # ── 1. Eligible LP demand ─────────────────────────────────────────────
        eligible_lps       = self._get_eligible_lps(hour, soc_kwh, solar_kw)
        eligible_demand_kw = sum(lp_demands.get(lp, 0.0) for lp in eligible_lps)

        # ── 2. Battery discharge (covers solar deficit only — R6) ─────────────
        # The battery only delivers what solar cannot, up to the rate/depth limit.
        # This prevents over-discharging when solar already covers most of the load.
        if batt_mode == 2:
            solar_deficit = max(0.0, eligible_demand_kw - solar_kw)
            depth_kwh     = max(0.0, soc_kwh - self.BATTERY_MIN_SOC_KWH)
            max_from_depth = depth_kwh * self.BATTERY_ETA
            battery_discharge_kw = min(self.BATTERY_MAX_DISCHARGE_KW,
                                       max_from_depth,
                                       solar_deficit)
        else:
            battery_discharge_kw = 0.0

        # ── 3. Diesel sizing (R8) ─────────────────────────────────────────────
        diesel_kw = 0.0
        if diesel_on_req and diesel_avail:
            supply_excl_diesel = solar_kw + battery_discharge_kw
            shortfall = max(0.0, eligible_demand_kw - supply_excl_diesel)
            low_soc_and_low_solar = (soc_kwh < self.DIESEL_EMERGENCY_SOC_KWH
                                     and solar_kw < self.DIESEL_EMERGENCY_SOLAR_KW)
            if shortfall > 0.0 or low_soc_and_low_solar:
                target = shortfall if shortfall > 0.0 else self.DIESEL_MIN_LOAD_KW
                diesel_kw = max(self.DIESEL_MIN_LOAD_KW,
                                min(target, self.DIESEL_CAPACITY_KW))

        total_supply = solar_kw + battery_discharge_kw + diesel_kw

        # ── 4. Battery charging from surplus (R5) ─────────────────────────────
        # headroom_kwh is energy still absorbable; dividing by η gives the
        # input power limit (since stored_energy = P_in × η × Δt).
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

        # ── 5. Priority-based load dispatch (R2, R3, R4) ─────────────────────
        lp_served, unmet_load, shedding_occurred = self._dispatch_loads(
            net_available, hour, lp_demands, eligible_lps
        )
        load_served_kw = sum(lp_served.values())

        # ── 6. Update battery SOC ─────────────────────────────────────────────
        soc_violated = False
        if battery_discharge_kw > 1e-9:
            draw = battery_discharge_kw / self.BATTERY_ETA
            self.battery_soc_kwh = max(self.BATTERY_MIN_SOC_KWH, soc_kwh - draw)
        if battery_charge_kw > 1e-9:
            stored = battery_charge_kw * self.BATTERY_ETA
            self.battery_soc_kwh = min(self.BATTERY_MAX_SOC_KWH,
                                       self.battery_soc_kwh + stored)
        if (self.battery_soc_kwh <= self.BATTERY_MIN_SOC_KWH + 1e-3
                or self.battery_soc_kwh >= self.BATTERY_MAX_SOC_KWH - 1e-3):
            soc_violated = True

        # ── 7. Fuel consumption (R9) ──────────────────────────────────────────
        fuel_consumed_litres = self._compute_diesel_fuel(diesel_kw)

        # ── 8. Reward ─────────────────────────────────────────────────────────
        renewable_bonus = (1.0 if unmet_load <= 1e-6 and fuel_consumed_litres <= 1e-9
                           else 0.0)
        fuel_penalty    = self.FUEL_WEIGHT      * fuel_consumed_litres
        batt_penalty    = self.BATTERY_WEIGHT   * float(soc_violated)
        load_penalty    = self.LOAD_WEIGHT      * unmet_load
        renew_reward    = self.RENEWABLE_WEIGHT * renewable_bonus
        reward          = -fuel_penalty - batt_penalty - load_penalty + renew_reward

        # ── Advance timestep ──────────────────────────────────────────────────
        self.t += 1
        done = self.t >= self.n_timesteps

        info = {
            "solar_output_kw":          solar_kw,
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

    # ─────────────────────────────────────────────────────────────────────────
    # State construction
    # ─────────────────────────────────────────────────────────────────────────

    def _get_state(self) -> np.ndarray:
        """
        Build the normalised 13-feature observation vector.

        Layout:
          [solar_frac, soc_norm, hour_norm, is_weekday,
           lp1_norm, lp2_norm, lp3_norm, lp4_norm,
           lp5_norm, lp6_norm, lp7_norm, lp8_norm,
           fuel_price_norm]
        """
        t = self.t

        solar_frac = float(self.solar_output_kw[t]) / self.SOLAR_CAPACITY_KW
        soc_norm   = ((self.battery_soc_kwh - self.BATTERY_MIN_SOC_KWH)
                      / (self.BATTERY_MAX_SOC_KWH - self.BATTERY_MIN_SOC_KWH))
        hour_norm  = float(self.hour_of_day[t]) / 23.0
        weekday    = float(self.is_weekday[t])

        lp_norms = np.array(
            [float(self.lp_demand_kw[lp][t]) / self._max_lp_demand[lp]
             for lp in self.LP_IDS],
            dtype=np.float64,
        )

        price_range = self._max_fuel_price - self._min_fuel_price
        price_norm  = ((float(self.fuel_price[t]) - self._min_fuel_price) / price_range
                       if price_range > 0.0 else 0.5)

        state = np.concatenate([
            [solar_frac, np.clip(soc_norm, 0.0, 1.0), hour_norm, weekday],
            np.clip(lp_norms, 0.0, 1.0),
            [np.clip(price_norm, 0.0, 1.0)],
        ])
        return state.astype(np.float64)

    # ─────────────────────────────────────────────────────────────────────────
    # Data loading
    # ─────────────────────────────────────────────────────────────────────────

    def _load_data(self, data_dir: str) -> None:
        src  = pd.read_csv(os.path.join(data_dir, "source_availability.csv"))
        load = pd.read_csv(os.path.join(data_dir, "load_demand_and_dispatch.csv"))

        self.solar_output_kw  = src["solar_pv_output_kw"].to_numpy(dtype=np.float64)
        self.diesel_available = src["diesel_available"].to_numpy(dtype=np.int32)
        self.fuel_price       = src["diesel_fuel_price_NGN_per_litre"].to_numpy(dtype=np.float64)
        self.hour_of_day      = src["hour_of_day"].to_numpy(dtype=np.int32)
        self.is_weekday       = src["is_weekday"].to_numpy(dtype=np.float32)
        self._initial_soc_kwh = float(src["battery_soc_kwh"].iloc[0])

        self.lp_demand_kw = {
            lp: load[f"{lp}_demand_kw"].to_numpy(dtype=np.float64)
            for lp in self.LP_IDS
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Eligible LP determination (R10, R11)
    # ─────────────────────────────────────────────────────────────────────────

    def _get_eligible_lps(self, hour: int, soc_kwh: float, solar_kw: float) -> set:
        """
        Return the set of LP IDs that may be served at this hour.

        Applies R10 (time-window restrictions) and R11 (emergency LP7 shedding).
        """
        eligible = {lp for lp in self.LP_IDS if self._lp_in_time_window(lp, hour)}

        # R11: shed LP7 (Sports Complex) under emergency low-energy conditions
        if soc_kwh < self.DIESEL_EMERGENCY_SOC_KWH and solar_kw < self.DIESEL_EMERGENCY_SOLAR_KW:
            eligible.discard("LP7")

        return eligible

    def _lp_in_time_window(self, lp: str, hour: int) -> bool:
        """Check R10 time-window constraint for a single LP."""
        if lp in ("LP1", "LP2", "LP3", "LP4"):
            return True                        # no restriction
        if lp == "LP5":                        # Library: 08:00–22:00
            return 8 <= hour <= 21
        if lp == "LP6":                        # Cafeteria: 06:00–21:00
            return 6 <= hour <= 20
        if lp == "LP7":                        # Sports: 06:00–20:00
            return 6 <= hour <= 19
        if lp == "LP8":                        # Street Lighting: 18:00–06:00
            return hour >= 18 or hour <= 5
        return False

    # ─────────────────────────────────────────────────────────────────────────
    # Priority-based load dispatch (R2, R3, R4)
    # ─────────────────────────────────────────────────────────────────────────

    def _dispatch_loads(
        self,
        available_power: float,
        hour: int,
        lp_demands: dict,
        eligible_lps: set,
    ):
        """
        Serve eligible LPs in priority order until power is exhausted.

        Day priority  (07:00–17:59): LP1 > LP2 > LP3 > LP4 > LP5 > LP6 > LP7
        Night priority (18:00–06:59): LP1 > LP8 > LP4 > LP2 > LP3 > LP5 > LP6 > LP7

        R4 threshold: LP2–LP7 are only served if ≥ 50% of their demand can be
        met from remaining power; otherwise they are skipped (load shed).
        LP1 and LP8 have a 0% threshold — always served if any power remains.

        Returns (lp_served dict, unmet_load_kw, shedding_occurred).
        """
        if 7 <= hour <= 17:
            priority_order = self.DAY_PRIORITY
        else:
            priority_order = self.NIGHT_PRIORITY

        ordered    = [lp for lp in priority_order if lp in eligible_lps]
        remaining  = available_power
        lp_served  = {lp: 0.0 for lp in self.LP_IDS}

        for lp in ordered:
            demand = lp_demands.get(lp, 0.0)
            if demand <= 1e-6:
                continue

            min_frac = self.LP_MIN_FRACTION[lp]
            if remaining >= demand:
                lp_served[lp] = demand
                remaining -= demand
            elif remaining >= demand * min_frac:
                # Partial service — still above the viability threshold (R4)
                lp_served[lp] = remaining
                remaining = 0.0
            # else: remaining < threshold → skip this LP entirely

            if remaining <= 0.0:
                break

        eligible_demand = sum(lp_demands.get(lp, 0.0) for lp in ordered)
        total_served    = sum(lp_served.values())
        unmet_load      = max(0.0, eligible_demand - total_served)

        return lp_served, unmet_load, unmet_load > 1e-3

    # ─────────────────────────────────────────────────────────────────────────
    # Diesel fuel consumption model (R9)
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_diesel_fuel(self, diesel_kw: float) -> float:
        """
        Linear derating fuel model (R9):
          L = L_idle + (L_full − L_idle) × (P / P_max)

        Returns litres consumed in one hour (Δt = 1 h).
        """
        if diesel_kw <= 1e-9:
            return 0.0
        return (self.DIESEL_IDLE_L_HR
                + (self.DIESEL_FULL_LOAD_L_HR - self.DIESEL_IDLE_L_HR)
                * (diesel_kw / self.DIESEL_CAPACITY_KW))

    # ─────────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────────

    def _action_from_decisions(self, diesel_on: int, batt_mode: int) -> int:
        """Return the ACTION_MAP index for the given (diesel_on, batt_mode) pair."""
        return int(diesel_on) * 3 + int(batt_mode)

    def __repr__(self) -> str:
        return (
            f"GreenfieldEnergyEnv("
            f"t={self.t}/{self.n_timesteps}, "
            f"soc={self.battery_soc_kwh:.1f} kWh, "
            f"state_size={self.state_size}, "
            f"action_size={self.action_size})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Quick demo / smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pprint
    pp = pprint.PrettyPrinter(indent=2, width=88, sort_dicts=False)

    env   = GreenfieldEnergyEnv()
    state = env.reset()

    print(f"\nenv repr      : {env}")
    print(f"state_size    : {env.state_size}")
    print(f"action_size   : {env.action_size}")
    print(f"n_timesteps   : {env.n_timesteps}")
    print(f"initial SOC   : {env.battery_soc_kwh:.1f} kWh")
    print(f"\nInitial state : {np.array2string(state, precision=4, separator=', ')}")

    print("\n─── Running 6 manual timesteps (one per action) ───────────────────")
    action_labels = [
        "OFF+CHARGE", "OFF+IDLE", "OFF+DISCHARGE",
        "ON+CHARGE",  "ON+IDLE",  "ON+DISCHARGE",
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
