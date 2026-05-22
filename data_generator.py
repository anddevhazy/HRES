import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Random seed for reproducibility while still producing realistic-looking noise
SEED = 42
rng = np.random.default_rng(SEED)

# 
# Constants
# 
HOURS_PER_YEAR = 8_760          # non-leap year
PEAK_IRRADIANCE = 1000          # W/m²  (mid-point of 900–1100 range)
PEAK_IRRADIANCE_TIME = 12.5               # hour of peak irradiance
HOURS_OF_STRONG_SUN_ON_EACH_SIDE_OF_PEAK = 4.0         # how many hours either side of peak the sun stays strong

# Load profile anchor points (kW)
LOAD_NIGHT_BASE = 1000          # 00–05 h  (security, servers, essential loads)
LOAD_PEAK = 2000                # 08–17 h  working hours
LOAD_EVENING = 1250             # 18–22 h  hostels, lighting
LOAD_NOISE_FRAC = 0.10          # ±10 % random noise on load
WEEKEND_REDUCTION = 0.25        # 25 % lower on weekends


def _hour_of_day_array():
    return np.arange(HOURS_PER_YEAR) % 24


def _day_of_year_array():
    return np.arange(HOURS_PER_YEAR) // 24


def _month_of_year_array():
# Using pandas to create a proper calendar, then extract the month number (1–12)
#  and subtract 1 to make it 0-based (0=January, 11=December).
    idx = pd.date_range("2024-01-01", periods=HOURS_PER_YEAR, freq="h")
    return idx.month.to_numpy() - 1         


def _day_of_week_array():
    # Similarly using pandas to get day-of-week. 0=Monday, 6=Sunday. Used to detect weekends.
    idx = pd.date_range("2024-01-01", periods=HOURS_PER_YEAR, freq="h")
    return idx.dayofweek.to_numpy()


# 
# Solar irradiance generator
# 

def _generate_solar_irradiance(hour_of_day, month):
    """
    This to Synthesise hourly global horizontal irradiance for FUNAAB, Abeokuta.

    Strategy:
      - Gaussian bell curve centred on solar noon during daylight (06–18 h).
      - Night-time (19–23 h and 00–05 h) forced to zero.
      - Multiplicative seasonal factor: slightly higher irradiance in the
        dry season (Nov–Mar, months 10–2) and lower in the wet season
        (Apr–Oct, months 3–9) due to increased cloud cover.
      - Multiplicative noise layer to simulate transient cloud events.
    """
    # ── Bell-curve base irradiance ────────────────────────────────────────────
    # Gaussian: G(h) = G_peak * exp(-0.5 * ((h - noon) / half_width)²)
    irradiance = PEAK_IRRADIANCE * np.exp(
        -0.5 * ((hour_of_day - PEAK_IRRADIANCE_TIME ) / HOURS_OF_STRONG_SUN_ON_EACH_SIDE_OF_PEAK) ** 2
    )

    # ── Night mask: zero outside daylight window (06 h–18 h inclusive) ────────
    night_mask = (hour_of_day < 6) | (hour_of_day >= 19)
    irradiance[night_mask] = 0.0

    # ── Seasonal scaling factor (sinusoidal over 12 months) ───────────────────
    # Dry season peak factor ≈ +8 %, wet season trough ≈ -8 %
    # Month 0 (Jan) is mid-dry season → cosine starts at +1
    season_factor = 1.0 + 0.08 * np.cos(2 * np.pi * month / 12)
    irradiance *= season_factor

    # ── Multiplicative cloud-cover noise ─────────────────────────────────────
    # Uses beta distribution to model right-skewed cloud attenuation:
    # clear-sky is common, heavy overcast is less frequent.
    noise = rng.beta(a=5, b=1.5, size=HOURS_PER_YEAR)   # skewed toward 1
    irradiance *= noise

    # Ensure physically valid range: no negatives, cap at 1200 W/m²
    irradiance = np.clip(irradiance, 0, 1200)

    return irradiance


# 
# Campus load demand generator
# 

def _base_load_profile(hour_of_day):
    """
    Return a weekday load profile (kW) for each timestep based on hour-of-day.

    Profile is built with linear ramps between anchor points to avoid
    unrealistic step changes while keeping the shape physically plausible.

    Segment anchors (weekday):
      00–05 h  → LOAD_NIGHT_BASE (~1000 kW)
      06 h     → ramp starts
      08 h     → LOAD_PEAK (~2000 kW)
      17 h     → LOAD_PEAK sustained
      18 h     → ramp down begins
      20 h     → LOAD_EVENING (~1250 kW)
      22 h     → ramp down continues
      23 h     → LOAD_NIGHT_BASE
    """
    load = np.zeros(HOURS_PER_YEAR)

    for i, h in enumerate(hour_of_day):
        if h <= 5:
            # Deep night — only essential services active
            load[i] = LOAD_NIGHT_BASE
        elif h == 6:
            # Start of morning ramp-up
            load[i] = LOAD_NIGHT_BASE + (LOAD_PEAK - LOAD_NIGHT_BASE) * 0.25
        elif h == 7:
            load[i] = LOAD_NIGHT_BASE + (LOAD_PEAK - LOAD_NIGHT_BASE) * 0.70
        elif 8 <= h <= 17:
            # Core working hours — peak demand
            load[i] = LOAD_PEAK
        elif h == 18:
            # Start of evening ramp-down
            load[i] = LOAD_PEAK - (LOAD_PEAK - LOAD_EVENING) * 0.50
        elif 19 <= h <= 21:
            # Evening residential / hostel peak
            load[i] = LOAD_EVENING
        elif h == 22:
            load[i] = LOAD_EVENING - (LOAD_EVENING - LOAD_NIGHT_BASE) * 0.50
        else:                    # h == 23
            load[i] = LOAD_NIGHT_BASE

    return load


def _generate_load_demand(hour_of_day, day_of_week):
    """
    Synthesise hourly campus electrical load for FUNAAB.

    Layers applied on top of the deterministic diurnal profile:
      1. Weekend reduction: academic facilities largely unoccupied Sat–Sun.
      2. Gaussian additive noise (±10 %) for equipment switching variability.
      3. Occasional demand spikes (large lab sessions, events) sampled from a
         Poisson process — adds realism for RL exploration.
    """
    # ── Deterministic diurnal profile ─────────────────────────────────────────
    load = _base_load_profile(hour_of_day)

    # ── Weekend scaling ───────────────────────────────────────────────────────
    # dayofweek: 5=Sat, 6=Sun
    is_weekend = (day_of_week >= 5).astype(float)
    load *= (1.0 - WEEKEND_REDUCTION * is_weekend)

    # ── Gaussian multiplicative noise (±10 %) ─────────────────────────────────
    noise_factor = rng.normal(loc=1.0, scale=LOAD_NOISE_FRAC, size=HOURS_PER_YEAR)
    load *= noise_factor

    # ── Occasional demand spikes (special events, lab sessions) ───────────────
    # Poisson: average ~2 spikes per week → λ ≈ 2/168 per hour
    spike_mask = rng.poisson(lam=2 / 168, size=HOURS_PER_YEAR).astype(bool)
    spike_magnitude = rng.uniform(0.05, 0.15, size=HOURS_PER_YEAR)  # 5–15 % above normal
    load[spike_mask] *= (1.0 + spike_magnitude[spike_mask])

    # Clamp to physically reasonable bounds: never below 600 kW or above 3000 kW
    load = np.clip(load, 600, 3000)

    return load


# 
# Public API
# 

def generate_data():
    """
    Generate one year of synthetic hourly data for an off-grid hybrid
    renewable energy system at FUNAAB, Abeokuta, Nigeria.

    Returns
    -------
    solar_irradiance : np.ndarray, shape (8760,)
        Global horizontal irradiance in W/m².
    load_demand : np.ndarray, shape (8760,)
        Campus electrical load demand in kW.
    """
    hour_of_day = _hour_of_day_array()
    month       = _month_of_year_array()
    day_of_week = _day_of_week_array()

    solar_irradiance = _generate_solar_irradiance(hour_of_day, month)
    load_demand      = _generate_load_demand(hour_of_day, day_of_week)

    return solar_irradiance, load_demand


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_time_index():
    return pd.date_range("2024-01-01", periods=HOURS_PER_YEAR, freq="h")


def _sample_week_slice(time_index, start="2024-01-08"):
    """Return the integer slice for the 168-hour week beginning `start`."""
    pos = time_index.searchsorted(pd.Timestamp(start))
    return slice(pos, pos + 168)


def _plot_series(time_index, data, week_slice, title, ylabel, colour, filename):
    """
    Produce a figure with two subplots:
      - Top: full-year time series
      - Bottom: zoomed-in sample week (Jan 8–14)
    Save to plots/<filename>.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Full-year view
    axes[0].plot(time_index, data, colour, linewidth=0.4, alpha=0.8)
    axes[0].set_title("Full Year (2024)")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel(ylabel)
    axes[0].grid(True, alpha=0.3)

    # Sample-week view
    week_time = time_index[week_slice]
    week_data = data[week_slice]
    axes[1].plot(week_time, week_data, colour, linewidth=1.2)
    axes[1].set_title("Sample Week: 8–14 January 2024")
    axes[1].set_xlabel("Date / Hour")
    axes[1].set_ylabel(ylabel)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    fig.savefig(os.path.join("plots", filename), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → plots/{filename}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating synthetic FUNAAB energy system data …\n")

    solar_irradiance, load_demand = generate_data()

    # ── Summary statistics ────────────────────────────────────────────────────
    print("Solar Irradiance (W/m²):")
    print(f"  Min  : {solar_irradiance.min():.2f}")
    print(f"  Max  : {solar_irradiance.max():.2f}")
    print(f"  Mean : {solar_irradiance.mean():.2f}\n")

    print("Campus Load Demand (kW):")
    print(f"  Min  : {load_demand.min():.2f}")
    print(f"  Max  : {load_demand.max():.2f}")
    print(f"  Mean : {load_demand.mean():.2f}\n")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("Generating plots …")
    time_index = _make_time_index()
    week_slice = _sample_week_slice(time_index, start="2024-01-08")

    _plot_series(
        time_index, solar_irradiance, week_slice,
        title    = "FUNAAB Campus — Solar Irradiance",
        ylabel   = "Irradiance (W/m²)",
        colour   = "darkorange",
        filename = "solar_irradiance.png",
    )

    _plot_series(
        time_index, load_demand, week_slice,
        title    = "FUNAAB Campus — Electrical Load Demand",
        ylabel   = "Load (kW)",
        colour   = "steelblue",
        filename = "load_demand.png",
    )

    print("\nDone.")
