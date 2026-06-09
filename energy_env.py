"""
energy_env.py
=============
Fully modular, source-agnostic off-grid hybrid renewable energy system
simulation environment for Deep Q-Network (DQN) reinforcement learning.

Class:  HybridEnergyEnv

Design principle
----------------
Every structural aspect of the system — number of sources, their types,
number of batteries, load priority tiers — is driven entirely by the config
dictionary passed to __init__.  The same class works identically for a
simple 1-source / 1-battery system and a complex 5-source / 3-battery system.

Equation references (from thesis methodology chapter)
------------------------------------------------------
Eq 3.3  – state-vector feature construction (normalised inputs)
Eq 3.5  – battery state-of-charge update (energy balance)
Eq 3.7  – diesel generator fuel consumption model (linear regression)
Eq 3.9  – priority-based load shedding rule
Eq 3.12 – power balance and reward formulation
"""

import numpy as np
import itertools


# ─────────────────────────────────────────────────────────────────────────────
# HybridEnergyEnv
# ─────────────────────────────────────────────────────────────────────────────

class HybridEnergyEnv:
    """
    Modular off-grid hybrid energy system simulation environment.

    The environment ingests a single `config` dictionary at construction time
    and builds the full action/observation spaces automatically.  It is
    intentionally agnostic about the number or type of energy sources,
    batteries, and load priority categories.

    Observation (state) vector layout  (Eq 3.3)
    -------------------------------------------
    [ avail_s0, avail_s1, …, avail_sN,     <- availability fraction per source
      soc_norm_b0, …, soc_norm_bM,          <- normalised SoC per battery
      load_norm, time_of_day_norm, day_norm ]  <- contextual scalars

    Total length = n_sources + n_batteries + 3  (scales automatically)

    Action space
    ------------
    Each integer action maps to a unique combination of:
      - On/off decision for every *controllable* source  (binary per source)
      - Charge / Idle / Discharge mode for every battery (ternary per battery)
    Total actions = 2^(n_controllable) × 3^(n_batteries)

    Reward (Eq 3.12)
    ----------------
    r(t) = −w_f×F(t) − w_b×D(t) − w_l×L(t) + w_r×R_bonus(t)
      F(t)       = total fuel consumed [litres]          weight w_f = 2.0
      D(t)       = battery stress count                  weight w_b = 5.0
      L(t)       = unmet load [kW]                       weight w_l = 10.0
      R_bonus(t) = 1 if load fully met with zero fuel    weight w_r = 3.0
    """

    # ── Reward weights (class-level constants) ────────────────────────────────
    FUEL_WEIGHT     = 2.0   # penalise diesel fuel consumption
    BATTERY_WEIGHT  = 5.0   # penalise SoC bound violations (battery stress)
    LOAD_WEIGHT     = 10.0  # penalise unmet load (reliability)
    RENEWABLE_WEIGHT = 3.0   # bonus for clean, fully-served operation

    # ─────────────────────────────────────────────────────────────────────────
    # Initialisation
    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self, config: dict):
        """
        Parse the configuration dictionary, build the action space, and
        initialise all state variables.

        Parameters
        ----------
        config : dict
            Must contain:
              "sources"         – list of source config dicts
              "batteries"       – list of battery config dicts
              "load_priorities" – list of load tier dicts (ordered highest→lowest)
              "load_profile"    – np.ndarray of shape (T,), hourly load in kW
        """
        self.config = config

        # ── Parse sources ─────────────────────────────────────────────────────
        self.sources = config["sources"]
        self.n_sources = len(self.sources)

        # Split sources by type for internal bookkeeping
        # Index mapping: controllable_source_idx[i] = position of i-th controllable
        # source in self.sources — needed to correlate actions back to sources.
        self.controllable_indices = [
            i for i, s in enumerate(self.sources) if s["type"] == "controllable"
        ]
        self.renewable_indices = [
            i for i, s in enumerate(self.sources) if s["type"] == "renewable"
        ]
        self.n_controllable = len(self.controllable_indices)

        # ── Parse batteries ───────────────────────────────────────────────────
        self.batteries = config["batteries"]
        self.n_batteries = len(self.batteries)

        # ── Parse load priorities ─────────────────────────────────────────────
        # Stored in the original order supplied by the user (index 0 = highest
        # priority).  The shedding engine works from the end of this list back.
        self.load_priorities = config["load_priorities"]
        self._validate_load_fractions()

        # ── Parse load profile ────────────────────────────────────────────────
        self.load_profile = np.asarray(config["load_profile"], dtype=np.float64)
        self.n_timesteps   = len(self.load_profile)
        self.max_load      = float(np.max(self.load_profile))  # for normalisation

        # ── Build discrete action space (Eq 3.12 — decision variables) ───────
        # Returns a list; index i → dict describing the full decision at step i.
        self.action_map = self._build_action_space()
        self.n_actions  = len(self.action_map)
        print(
            f"[HybridEnergyEnv] Initialised  |  "
            f"sources={self.n_sources} "
            f"(renewable={len(self.renewable_indices)}, "
            f"controllable={self.n_controllable})  |  "
            f"batteries={self.n_batteries}  |  "
            f"load_tiers={len(self.load_priorities)}  |  "
            f"state_size={self.state_size}  |  "
            f"action_size={self.n_actions}"
        )

        # ── Mutable simulation state ──────────────────────────────────────────
        self.t   = 0                          # current timestep index
        self.soc = np.array(                  # battery state-of-charge [fraction]
            [b["initial_soc"] for b in self.batteries], dtype=np.float64
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def state_size(self) -> int:
        """
        Length of the observation vector (Eq 3.3).

        = n_sources        (one availability value per source)
        + n_batteries      (one normalised SoC per battery)
        + 3                (normalised load, time-of-day, day-of-year)
        """
        return self.n_sources + self.n_batteries + 3

    @property
    def action_size(self) -> int:
        """Total number of discrete actions in the action space."""
        return self.n_actions

    # ─────────────────────────────────────────────────────────────────────────
    # Core RL interface
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """
        Reset the environment to its initial conditions.

        Resets the timestep counter to 0 and restores every battery to its
        configured initial SoC.

        Returns
        -------
        state : np.ndarray, shape (state_size,)
            The normalised initial observation vector.
        """
        self.t   = 0
        self.soc = np.array(
            [b["initial_soc"] for b in self.batteries], dtype=np.float64
        )
        return self._get_state()

    def step(self, action: int):
        """
        Execute one simulation timestep given a discrete action.

        Power balance (Eq 3.12):
          P_available(t) = Σ P_source(t) + Σ P_discharge(t) − Σ P_charge(t)

        The method proceeds through five sub-steps:
          1. Decode the action into per-source and per-battery decisions.
          2. Compute power output from every source.
          3. Execute battery discharge (adds to available supply).
          4. Execute battery charge (draws from surplus only — never at the
             expense of unmet demand).
          5. Apply load shedding if the net balance is still negative.

        Parameters
        ----------
        action : int
            Index into self.action_map; must be in [0, action_size − 1].

        Returns
        -------
        next_state : np.ndarray, shape (state_size,)
            Normalised observation at t+1 (zero vector if episode just ended).
        reward : float
            Scalar reward for this timestep.
        done : bool
            True when the episode has consumed all T timesteps.
        info : dict
            Diagnostic breakdown — see docstring body for keys.
        """
        if not (0 <= action < self.n_actions):
            raise ValueError(
                f"Action {action} is outside valid range [0, {self.n_actions - 1}]."
            )

        # ── Step 1: Decode action ─────────────────────────────────────────────
        action_spec   = self.action_map[action]
        ctrl_decisions = action_spec["controllable"]  # list[int], 0=off / 1=on
        batt_modes     = action_spec["batteries"]     # list[int], 0=charge/1=idle/2=discharge

        # ── Step 2: Source outputs ────────────────────────────────────────────
        source_outputs: dict[str, float] = {}
        fuel_consumed:  dict[str, float] = {}
        total_source_power = 0.0

        ctrl_cursor = 0  # index into ctrl_decisions (only incremented for controllable)
        for source in self.sources:
            if source["type"] == "renewable":
                decision = None              # renewables always generate
            else:
                decision = ctrl_decisions[ctrl_cursor]
                ctrl_cursor += 1

            p_out = self._compute_source_output(source, decision)
            source_outputs[source["name"]] = p_out
            total_source_power += p_out

            if source["type"] == "controllable":
                fuel_consumed[source["name"]] = self._compute_fuel_consumption(
                    source, p_out
                )

        # ── Step 3: Battery discharge (boosts available supply) ───────────────
        # Discharge is processed first so we know the full supply before deciding
        # how much surplus is available for charging.
        total_discharge       = 0.0
        actual_discharge_kws  = []
        actual_charge_kws     = [0.0] * self.n_batteries  # filled in step 4
        batt_soc_violated     = []

        for j, bat in enumerate(self.batteries):
            if batt_modes[j] == 2:   # discharge mode
                requested_d = bat["max_discharge_rate_kw"]
            else:
                requested_d = 0.0

            act_d, violated = self._execute_battery_discharge(j, requested_d)
            actual_discharge_kws.append(act_d)
            batt_soc_violated.append(violated)   # placeholder; updated in step 4
            total_discharge += act_d

        # ── Step 4: Battery charging (uses surplus only) ─────────────────────
        # Surplus = sources + discharge − current demand.
        # Batteries may only charge from genuine surplus; they must not worsen
        # a supply deficit.
        total_demand  = float(self.load_profile[self.t])
        surplus       = total_source_power + total_discharge - total_demand
        total_charge  = 0.0

        for j, bat in enumerate(self.batteries):
            if batt_modes[j] == 0:   # charge mode
                # Never commit more power to charging than is genuinely spare
                available_for_charge = max(0.0, surplus - total_charge)
                requested_c = min(bat["max_charge_rate_kw"], available_for_charge)
            else:
                requested_c = 0.0

            act_c, charge_violated = self._execute_battery_charge(j, requested_c)
            actual_charge_kws[j]  = act_c
            # OR the discharge violation flag with the charge violation flag
            batt_soc_violated[j] = batt_soc_violated[j] or charge_violated
            total_charge += act_c

        # ── Step 5: Net power balance and load shedding (Eq 3.9) ─────────────
        # P_available = sources + discharge − charge
        available_power = total_source_power + total_discharge - total_charge

        load_served, unmet_load, shedding_occurred = self._apply_load_shedding(
            available_power, total_demand
        )

        # ── Reward computation (Eq 3.12) ──────────────────────────────────────
        total_fuel   = sum(fuel_consumed.values())
        batt_stress  = sum(1 for v in batt_soc_violated if v)

        # Renewable bonus: full load served AND no fuel burned this step
        renewable_bonus = 1.0 if (unmet_load <= 1e-6 and total_fuel <= 1e-9) else 0.0

        fuel_penalty     = self.FUEL_WEIGHT     * total_fuel
        battery_penalty  = self.BATTERY_WEIGHT  * batt_stress
        load_penalty     = self.LOAD_WEIGHT     * unmet_load
        renewable_reward = self.RENEWABLE_WEIGHT * renewable_bonus


        reward = -fuel_penalty - battery_penalty - load_penalty + renewable_reward

        # ── Advance timestep ──────────────────────────────────────────────────
        self.t += 1
        done = self.t >= self.n_timesteps

        # ── Build info dictionary ─────────────────────────────────────────────
        info = {
            # Per-source and per-battery details
            "power_output_per_source":  {k: float(v) for k, v in source_outputs.items()},
            "fuel_consumed_per_source": {k: float(v) for k, v in fuel_consumed.items()},
            "soc_per_battery": {
                bat["name"]: float(self.soc[j])
                for j, bat in enumerate(self.batteries)
            },
            # Load balance
            "total_demand_kw":     float(total_demand),
            "available_power_kw":  float(available_power),
            "load_served_kw":      float(load_served),
            "unmet_load_kw":       float(unmet_load),
            "load_shedding_occurred": bool(shedding_occurred),
            # Power flow summary
            "total_source_power_kw":   float(total_source_power),
            "total_discharge_kw":      float(total_discharge),
            "total_charge_kw":         float(total_charge),
            # Reward breakdown
            "reward_components": {
                "fuel_penalty":     float(-fuel_penalty),
                "battery_penalty":  float(-battery_penalty),
                "load_penalty":     float(-load_penalty),
                "renewable_bonus":  float(renewable_reward),
                "total_reward":     float(reward),
            },
        }

        # Return zero-vector observation when the episode has ended (terminal state)
        next_state = (
            self._get_state() if not done else np.zeros(self.state_size, dtype=np.float64)
        )
        return next_state, reward, done, info

    # ─────────────────────────────────────────────────────────────────────────
    # State construction
    # ─────────────────────────────────────────────────────────────────────────

    def _get_state(self) -> np.ndarray:
        """
        Construct and return the normalised observation vector (Eq 3.3).

        Layout:
          [availability_s0, …, availability_sN,    ← raw availability fractions
           soc_norm_b0,     …, soc_norm_bM,         ← SoC mapped to [0, 1]
           load_norm, time_of_day_norm, day_norm]

        SoC normalisation maps the operational window [soc_min, soc_max] onto
        [0, 1]:
            soc_norm = (SoC − soc_min) / (soc_max − soc_min)   (Eq 3.3)

        Returns
        -------
        state : np.ndarray, shape (state_size,), dtype float64
        """
        # Source availability fractions — already in [0, 1] by construction
        availabilities = np.array(
            [s["availability_profile"][self.t] for s in self.sources],
            dtype=np.float64,
        )

        # Normalised battery SoC (Eq 3.3)
        soc_normalised = np.array(
            [
                (self.soc[j] - self.batteries[j]["soc_min"])
                / (self.batteries[j]["soc_max"] - self.batteries[j]["soc_min"])
                for j in range(self.n_batteries)
            ],
            dtype=np.float64,
        )
        soc_normalised = np.clip(soc_normalised, 0.0, 1.0)

        # Contextual scalars
        load_norm     = self.load_profile[self.t] / self.max_load   # [0, 1]
        time_norm     = (self.t % 24)  / 23.0                       # hour-of-day → [0, 1]
        day_norm      = (self.t // 24) / 364.0                      # day-of-year → [0, 1]

        state = np.concatenate(
            [availabilities, soc_normalised, [load_norm, time_norm, day_norm]]
        )
        return state.astype(np.float64)

    # ─────────────────────────────────────────────────────────────────────────
    # Action space builder
    # ─────────────────────────────────────────────────────────────────────────

    def _build_action_space(self) -> list:
        """
        Dynamically construct the discrete action space.

        Controllable sources:  binary options  {0 = off,       1 = on}
        Batteries:             ternary options  {0 = charge, 1 = idle, 2 = discharge}

        Total actions = 2^(n_controllable) × 3^(n_batteries)

        The Cartesian product of all per-source and per-battery option sets is
        enumerated with itertools.product to guarantee every unique combination
        appears exactly once.

        Returns
        -------
        action_map : list of dicts
            action_map[i] = {
                "controllable": [int, …],   # 0/1 per controllable source
                "batteries":    [int, …],   # 0/1/2 per battery
            }
        """
        # Generate all on/off combinations for controllable sources
        if self.n_controllable > 0:
            ctrl_combos = list(itertools.product([0, 1], repeat=self.n_controllable))
        else:
            ctrl_combos = [()]   # one "empty" combo when there are no controllable sources

        # Generate all charge/idle/discharge combinations for batteries
        if self.n_batteries > 0:
            batt_combos = list(itertools.product([0, 1, 2], repeat=self.n_batteries))
        else:
            batt_combos = [()]   # one "empty" combo when there are no batteries

        action_map = []
        for ctrl_combo in ctrl_combos:
            for batt_combo in batt_combos:
                action_map.append(
                    {
                        "controllable": list(ctrl_combo),
                        "batteries":    list(batt_combo),
                    }
                )

        return action_map

    # ─────────────────────────────────────────────────────────────────────────
    # Source physics
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_source_output(self, source: dict, action_decision) -> float:
        """
        Compute the power output (kW) of a single source at the current timestep.

        Renewable sources (Eq 3.3 — solar/wind generation model):
            P(t) = availability(t) × P_rated
            Output is uncontrolled; it is always equal to the available capacity.

        Controllable sources (e.g. diesel generators):
            P(t) = action_decision × availability(t) × P_rated
            When action_decision = 0 (OFF), output is zero regardless of
            availability; when 1 (ON), the generator runs at its available
            fraction of rated capacity.

        Parameters
        ----------
        source : dict
            Source configuration entry from config["sources"].
        action_decision : int or None
            For controllable sources: 1 = on, 0 = off.
            For renewable sources: ignored (pass None for clarity).

        Returns
        -------
        output_kw : float
        """
        availability = float(source["availability_profile"][self.t])
        rated        = float(source["rated_capacity_kw"])

        if source["type"] == "renewable":
            # Renewable: cannot be curtailed by the agent in this model
            return availability * rated
        else:
            # Controllable: agent decides whether the unit runs
            return float(action_decision) * availability * rated

    def _compute_fuel_consumption(self, source: dict, output_kw: float) -> float:
        """
        Compute fuel consumed by a controllable source over one hour (Δt = 1 h).

        Diesel generator linear fuel consumption model (Eq 3.7):
            F(t) = [a × P_gen(t)  +  b × P_rated]  ×  Δt

        where
            a  = fuel_coefficient_a  [L/kWh]  – variable (load-dependent) term
            b  = fuel_coefficient_b  [L/kWh]  – fixed no-load consumption term
            Δt = 1 hour

        If the generator is off (output_kw ≤ 0), fuel consumption is zero.

        Parameters
        ----------
        source : dict
            Must contain "fuel_coefficient_a", "fuel_coefficient_b",
            and "rated_capacity_kw".
        output_kw : float
            Actual power output this timestep (kW).

        Returns
        -------
        fuel_litres : float
            Fuel consumed in litres during this 1-hour interval.
        """
        if output_kw <= 1e-9:
            return 0.0

        a     = float(source["fuel_coefficient_a"])
        b     = float(source["fuel_coefficient_b"])
        rated = float(source["rated_capacity_kw"])

        # Eq 3.7: F = (a × P_output + b × P_rated) × Δt,   Δt = 1 h
        return (a * output_kw + b * rated) * 1.0

    # ─────────────────────────────────────────────────────────────────────────
    # Battery physics  (split into charge / discharge for clarity)
    # ─────────────────────────────────────────────────────────────────────────

    def _update_battery_soc(
        self,
        battery_idx: int,
        charge_power: float,
        discharge_power: float,
    ):
        """
        Update the SoC of one battery using the energy balance equation (Eq 3.5).

        Eq 3.5:
            SoC(t+1) = SoC(t) + [P_charge(t)×η_c  −  P_discharge(t)/η_d] × Δt
                                  ─────────────────────────────────────────────
                                                  E_capacity

        Physical constraints applied automatically:
          • Charging  capped at max_charge_rate_kw  AND  SoC headroom to soc_max
          • Discharge capped at max_discharge_rate_kw AND  SoC depth to soc_min

        A stress flag is set to True when the SoC touches or crosses either bound
        after the update — used to compute D(t) in the reward function.

        Parameters
        ----------
        battery_idx    : int    Index into self.batteries and self.soc.
        charge_power   : float  Requested charge power (kW, ≥ 0).
        discharge_power: float  Requested discharge power (kW, ≥ 0).

        Returns
        -------
        actual_charge_kw    : float
        actual_discharge_kw : float
        soc_violated        : bool   True → SoC hit a bound (stress event).
        """
        bat         = self.batteries[battery_idx]
        soc_cur     = self.soc[battery_idx]
        capacity    = float(bat["capacity_kwh"])
        eta_c       = float(bat["charge_efficiency"])
        eta_d       = float(bat["discharge_efficiency"])
        soc_min     = float(bat["soc_min"])
        soc_max     = float(bat["soc_max"])

        actual_charge_kw    = 0.0
        actual_discharge_kw = 0.0
        soc_violated        = False

        if charge_power > 1e-9:
            # Maximum energy the battery can still absorb before hitting soc_max
            max_energy_in   = (soc_max - soc_cur) * capacity          # kWh
            # Convert energy headroom to a power limit (accounting for efficiency)
            max_charge_from_headroom = max_energy_in / eta_c          # kW for 1 h
            # Actual charge power: minimum of requested, rate limit, headroom
            actual_charge_kw = min(
                charge_power,
                float(bat["max_charge_rate_kw"]),
                max_charge_from_headroom,
            )
            actual_charge_kw = max(actual_charge_kw, 0.0)

            # Update SoC (Eq 3.5 — charge term only, Δt = 1 h)
            delta_soc = (actual_charge_kw * eta_c) / capacity
            self.soc[battery_idx] = min(soc_cur + delta_soc, soc_max)

        elif discharge_power > 1e-9:
            # Maximum energy the battery can deliver before hitting soc_min
            max_energy_out  = (soc_cur - soc_min) * capacity          # kWh
            # Convert energy depth to a power limit (accounting for efficiency)
            max_discharge_from_depth = max_energy_out * eta_d         # kW for 1 h
            # Actual discharge power: minimum of requested, rate limit, depth
            actual_discharge_kw = min(
                discharge_power,
                float(bat["max_discharge_rate_kw"]),
                max_discharge_from_depth,
            )
            actual_discharge_kw = max(actual_discharge_kw, 0.0)

            # Update SoC (Eq 3.5 — discharge term only, Δt = 1 h)
            delta_soc = (actual_discharge_kw / eta_d) / capacity
            self.soc[battery_idx] = max(soc_cur - delta_soc, soc_min)

        # Stress check: touching either bound signals operational stress
        new_soc = self.soc[battery_idx]
        if new_soc <= soc_min + 1e-6 or new_soc >= soc_max - 1e-6:
            soc_violated = True

        return actual_charge_kw, actual_discharge_kw, soc_violated

    def _execute_battery_discharge(self, battery_idx: int, requested_kw: float):
        """
        Convenience wrapper: discharge only.  Delegates to _update_battery_soc.

        Returns (actual_discharge_kw, soc_violated).
        """
        _, act_d, violated = self._update_battery_soc(battery_idx, 0.0, requested_kw)
        return act_d, violated

    def _execute_battery_charge(self, battery_idx: int, requested_kw: float):
        """
        Convenience wrapper: charge only.  Delegates to _update_battery_soc.

        Returns (actual_charge_kw, soc_violated).
        """
        act_c, _, violated = self._update_battery_soc(battery_idx, requested_kw, 0.0)
        return act_c, violated

    # ─────────────────────────────────────────────────────────────────────────
    # Load shedding engine
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_load_shedding(
        self,
        available_power: float,
        total_demand: float,
    ):
        """
        Priority-based load shedding (Eq 3.9).

        When available_power < total_demand, sheddable load categories are
        curtailed in reverse-priority order (lowest-priority tier first) until
        the supply deficit is eliminated or all sheddable load has been shed.

        Non-sheddable tiers are NEVER curtailed.

        Eq 3.9:
            L_shed(t) = max(0, P_demand(t) − P_available(t))
            applied iteratively, starting from the lowest-priority sheddable tier.

        Parameters
        ----------
        available_power : float   Total power deliverable this timestep (kW).
        total_demand    : float   Total requested load this timestep (kW).

        Returns
        -------
        load_served    : float   Actual load served after shedding (kW).
        unmet_load     : float   Remaining unmet demand after all shedding (kW).
        shedding_occurred : bool
        """
        if available_power >= total_demand - 1e-6:
            # Supply meets or exceeds demand — no shedding needed
            return total_demand, 0.0, False

        # Compute the absolute demand (kW) assigned to each priority tier
        tier_demands = [
            float(lp["fraction"]) * total_demand
            for lp in self.load_priorities
        ]

        deficit = total_demand - available_power

        # Iterate in reverse (lowest priority last in the list → shed first)
        for i in reversed(range(len(self.load_priorities))):
            if deficit <= 1e-6:
                break
            if self.load_priorities[i]["sheddable"]:
                shed_amount   = min(tier_demands[i], deficit)
                tier_demands[i] -= shed_amount
                deficit       -= shed_amount

        load_served = float(sum(tier_demands))
        unmet_load  = max(0.0, total_demand - load_served)

        return load_served, unmet_load, True

    # ─────────────────────────────────────────────────────────────────────────
    # Validation helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _validate_load_fractions(self):
        """
        Verify that the load priority fractions sum to 1.0 (within tolerance).
        Raises ValueError if the configuration is inconsistent.
        """
        total = sum(lp["fraction"] for lp in self.load_priorities)
        if abs(total - 1.0) > 1e-4:
            raise ValueError(
                f"Load priority fractions must sum to 1.0; got {total:.4f}. "
                f"Check config['load_priorities']."
            )

    def __repr__(self) -> str:
        return (
            f"HybridEnergyEnv("
            f"sources={self.n_sources}, "
            f"batteries={self.n_batteries}, "
            f"load_tiers={len(self.load_priorities)}, "
            f"state_size={self.state_size}, "
            f"action_size={self.n_actions}, "
            f"t={self.t}/{self.n_timesteps})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Demo / Modularity test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    import pprint
    pp = pprint.PrettyPrinter(indent=2, width=88, sort_dicts=False)

    SEPARATOR = "=" * 70

    # =========================================================================
    # TEST 1 — FUNAAB configuration
    #   Sources  : Solar PV (renewable) + Diesel generator (controllable)
    #   Batteries: 1 × BESS
    #   Load tiers: 3 (critical / essential / non-essential)
    #   Data      : generated by data_generator.generate_data()
    # =========================================================================

    print(f"\n{SEPARATOR}")
    print("TEST 1 — FUNAAB configuration (solar + diesel + 1 battery + 3 tiers)")
    print(SEPARATOR)

    from data_generator import generate_data

    # Generate one year of synthetic FUNAAB data
    solar_irradiance, load_demand = generate_data()   # shapes: (8760,) each

    # Convert irradiance (W/m²) to availability fraction [0, 1]
    # Peak irradiance at FUNAAB ≈ 1000 W/m²; clip ensures the fraction
    # stays in [0, 1] even if the synthetic data briefly exceeds peak.
    solar_availability = np.clip(solar_irradiance / 1000.0, 0.0, 1.0)

    # Diesel is mechanically always available; agent decides whether to run it.
    diesel_availability = np.ones(8760, dtype=np.float64)

    config_funaab = {
        "sources": [
            {
                "name":                  "solar_pv",
                "type":                  "renewable",
                "rated_capacity_kw":     800.0,
                "availability_profile":  solar_availability,
            },
            {
                "name":                  "diesel_generator",
                "type":                  "controllable",
                "rated_capacity_kw":     1000.0,
                "fuel_coefficient_a":    0.084,   # L/kWh  (variable component, Eq 3.7)
                "fuel_coefficient_b":    0.246,   # L/kWh  (fixed no-load component, Eq 3.7)
                "availability_profile":  diesel_availability,
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

    env1 = HybridEnergyEnv(config_funaab)
    print(f"\nenv1.state_size  = {env1.state_size}")
    print(f"env1.action_size = {env1.action_size}")

    state = env1.reset()
    print(f"\nInitial state vector (length={len(state)}):")
    print("  " + np.array2string(state, precision=4, separator=", "))

    # Run 5 manual timesteps with explicit, varied action choices to exercise
    # different operating modes: charge-only, diesel-on, etc.
    test_actions = [0, 3, 5, 2, 4]
    print("\n--- Running 5 manual timesteps ---")
    for step_num, action in enumerate(test_actions, start=1):
        next_state, reward, done, info = env1.step(action)
        print(f"\nStep {step_num}  (action={action})")
        print(f"  State  : {np.array2string(next_state, precision=4, separator=', ')}")
        print(f"  Reward : {reward:.4f}")
        print("  Info:")
        pp.pprint(info)
        if done:
            print("  [Episode finished]")
            break

    # =========================================================================
    # TEST 2 — Extended configuration
    #   Sources  : Solar PV + Wind + Diesel generator
    #   Batteries: 2 × BESS
    #   Load tiers: 4
    #   Data      : Fully synthetic (numpy random) — no external data dependency
    #
    # This test proves the class scales to a completely different system without
    # any code changes.  The different state_size and action_size printed here
    # confirm the modularity is working correctly.
    # =========================================================================

    print(f"\n\n{SEPARATOR}")
    print("TEST 2 — Extended configuration (solar+wind+diesel + 2 batteries + 4 tiers)")
    print(SEPARATOR)

    rng = np.random.default_rng(seed=2024)

    # Synthetic availability profiles for a second hypothetical site
    solar_avail_2   = np.clip(rng.beta(a=2, b=1, size=8760), 0, 1)
    wind_avail_2    = np.clip(rng.beta(a=3, b=2, size=8760), 0, 1)
    diesel_avail_2  = np.ones(8760, dtype=np.float64)
    load_profile_2  = rng.uniform(500, 3000, size=8760).astype(np.float64)

    config_extended = {
        "sources": [
            {
                "name":                  "solar_pv_2",
                "type":                  "renewable",
                "rated_capacity_kw":     500.0,
                "availability_profile":  solar_avail_2,
            },
            {
                "name":                  "wind_turbine",
                "type":                  "renewable",
                "rated_capacity_kw":     400.0,
                "availability_profile":  wind_avail_2,
            },
            {
                "name":                  "diesel_gen_2",
                "type":                  "controllable",
                "rated_capacity_kw":     800.0,
                "fuel_coefficient_a":    0.084,
                "fuel_coefficient_b":    0.246,
                "availability_profile":  diesel_avail_2,
            },
        ],
        "batteries": [
            {
                "name":                  "bess_alpha",
                "capacity_kwh":          2000.0,
                "max_charge_rate_kw":    400.0,
                "max_discharge_rate_kw": 400.0,
                "charge_efficiency":     0.95,
                "discharge_efficiency":  0.95,
                "soc_min":               0.15,
                "soc_max":               0.95,
                "initial_soc":           0.60,
            },
            {
                "name":                  "bess_beta",
                "capacity_kwh":          1500.0,
                "max_charge_rate_kw":    300.0,
                "max_discharge_rate_kw": 300.0,
                "charge_efficiency":     0.92,
                "discharge_efficiency":  0.92,
                "soc_min":               0.20,
                "soc_max":               0.90,
                "initial_soc":           0.45,
            },
        ],
        "load_priorities": [
            {"name": "mission_critical", "fraction": 0.15, "sheddable": False},
            {"name": "critical",         "fraction": 0.25, "sheddable": False},
            {"name": "essential",        "fraction": 0.35, "sheddable": True},
            {"name": "deferrable",       "fraction": 0.25, "sheddable": True},
        ],
        "load_profile": load_profile_2,
    }

    env2 = HybridEnergyEnv(config_extended)
    print(f"\nenv2.state_size  = {env2.state_size}   ← different from env1 ({env1.state_size}) ✓")
    print(f"env2.action_size = {env2.action_size}  ← different from env1 ({env1.action_size}) ✓")

    state2 = env2.reset()
    print(f"\nInitial state vector (length={len(state2)}):")
    print("  " + np.array2string(state2, precision=4, separator=", "))

    print("\n--- Running 3 manual timesteps ---")
    for step_num in range(1, 4):
        action = rng.integers(0, env2.action_size)
        next_state2, reward2, done2, info2 = env2.step(int(action))
        print(f"\nStep {step_num}  (action={action})")
        print(f"  State  : {np.array2string(next_state2, precision=4, separator=', ')}")
        print(f"  Reward : {reward2:.4f}")
        print("  Info:")
        pp.pprint(info2)
        if done2:
            print("  [Episode finished]")
            break

    print(f"\n{SEPARATOR}")
    print("Modularity confirmed:")
    print(f"  env1 → state_size={env1.state_size}, action_size={env1.action_size}")
    print(f"  env2 → state_size={env2.state_size}, action_size={env2.action_size}")
    print(f"  Both created from HybridEnergyEnv with different config dicts.")
    print(SEPARATOR)
