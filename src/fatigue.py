"""
Stress-life (S-N) fatigue assessment with Goodman mean-stress correction.

S-N data for PA12 SLS compiled from:
- Van Hooreweder et al. (2010), Polymer Testing 29(8): 1011–1019
- Munguia et al. & Crupi et al. — independent characterisations

Curve at R = -1 (fully reversed) gives alternating stress vs. cycles to failure.
For non-zero mean stress, Goodman correction maps the actual cycle to an
equivalent R=-1 cycle:
    sigma_a_eq = sigma_a / (1 - sigma_m / sigma_uts)
which is then evaluated against the S-N curve by log-log interpolation.
"""
from __future__ import annotations

import numpy as np


# 7-point S-N curve for PA12 SLS at R = -1 (alternating stress, MPa)
SN_CYCLES = np.array([1e2, 1e3, 1e4, 1e5, 1e6, 5e6, 1e7], dtype=float)
SN_ALT_MPA = np.array([45.0, 38.0, 28.0, 20.0, 13.0, 12.0, 12.0], dtype=float)


def sn_life(sigma_alt_eq_Pa: np.ndarray) -> np.ndarray:
    """Predict cycles to failure for an equivalent fully-reversed alternating
    stress (Pa), by log-log interpolation of the PA12 SLS S-N curve.

    Beyond the table: life clipped to endurance ( >= 1e7 cycles ) when the
    stress is below the endurance limit, and to 1 cycle when above the highest
    tabulated stress (static failure on first cycle).
    """
    s_mpa = np.asarray(sigma_alt_eq_Pa, dtype=float) / 1e6
    log_n_arr = np.log10(SN_CYCLES)            # increasing
    log_s_arr = np.log10(SN_ALT_MPA)
    # SN curve is *decreasing* in s vs n.  We invert to (s -> n) by
    # interpolating in the reverse direction.
    # Build a strictly-monotone (decreasing) sigma -> increasing N table.
    # Use np.interp by reversing both arrays.
    log_s_rev = log_s_arr[::-1]
    log_n_rev = log_n_arr[::-1]
    s_safe = np.where(s_mpa > 0, s_mpa, 1e-10)
    log_s = np.log10(s_safe)
    # np.interp needs xp strictly increasing
    # log_s_rev is sorted? sn_alt_mpa = [45,38,28,20,13,12,12]  reversed = [12,12,12,13,20,28,38,45]
    # The repeated 12s break strict monotonicity. Make strictly increasing by adding
    # tiny perturbations to ties.
    eps = 1e-12
    xp = log_s_rev.copy()
    for i in range(1, len(xp)):
        if xp[i] <= xp[i - 1]:
            xp[i] = xp[i - 1] + eps
    log_n = np.interp(log_s, xp, log_n_rev,
                      left=log_n_rev[0],     # clip low-stress side
                      right=log_n_rev[-1])   # clip high-stress side
    # For points below endurance limit, return infinity-ish (1e7 cycles is the plateau)
    # For points above highest tabulated stress (45 MPa), set life to 1 cycle (static-like)
    life = 10.0 ** log_n
    life = np.where(s_mpa < SN_ALT_MPA[-1] * 0.99, 1e9, life)   # below endurance => "infinite"
    life = np.where(s_mpa > SN_ALT_MPA[0] * 1.5, 1.0, life)     # extreme overload
    return life


def equivalent_alt_stress_goodman(sigma_alt: np.ndarray,
                                   sigma_mean: np.ndarray,
                                   sigma_uts_Pa: float) -> np.ndarray:
    """Apply Goodman mean-stress correction.
       sigma_a_eq = sigma_a / (1 - sigma_m / sigma_uts)
    Compressive mean stresses (sigma_m < 0) reduce damage; we conservatively
    use max(0, sigma_m) so compressive means do not increase predicted life.
    """
    sm_pos = np.maximum(sigma_mean, 0.0)
    denom = 1.0 - sm_pos / sigma_uts_Pa
    # Numerical safety: avoid division by ~0 (mean stress -> uts)
    denom = np.maximum(denom, 1e-6)
    return sigma_alt / denom


def fatigue_life_field(stress_lc1: np.ndarray, stress_lc2: np.ndarray,
                       sigma_uts_Pa: float) -> dict:
    """Compute the fatigue life field for a cycle that oscillates between
    LC1 (hover) and LC2 (maneuver), evaluated element-wise on the von Mises
    equivalent stress.

    stress_lc1, stress_lc2 : (n_active,) von Mises stress (Pa) per element
                              for the two load cases.
    """
    # Use von Mises magnitude as the cyclic scalar driver.
    s1 = np.asarray(stress_lc1, dtype=float)
    s2 = np.asarray(stress_lc2, dtype=float)
    # The max von Mises is treated as 'peak' and min as 'valley'
    s_peak = np.maximum(s1, s2)
    s_valley = np.minimum(s1, s2)
    sigma_mean = 0.5 * (s_peak + s_valley)
    sigma_alt = 0.5 * (s_peak - s_valley)
    # Goodman correction
    sigma_alt_eq = equivalent_alt_stress_goodman(sigma_alt, sigma_mean,
                                                  sigma_uts_Pa)
    life = sn_life(sigma_alt_eq)
    # Damage per cycle relative to a 1e6-cycle design life
    damage = 1.0 / np.maximum(life, 1.0)
    # Fatigue safety factor:  ratio of the alternating stress at design life
    # to the actual equivalent alternating stress.
    s_at_design_life = sn_alt_stress_at(1e6) * 1e6     # Pa
    fs = s_at_design_life / np.maximum(sigma_alt_eq, 1e-6)
    return {
        "sigma_mean": sigma_mean,
        "sigma_alt": sigma_alt,
        "sigma_alt_eq": sigma_alt_eq,
        "life": life,
        "damage": damage,
        "factor_of_safety": fs,
    }


def sn_alt_stress_at(N: float) -> float:
    """Return alternating stress (MPa) at N cycles by log-log interpolation."""
    log_n_arr = np.log10(SN_CYCLES)
    log_s_arr = np.log10(SN_ALT_MPA)
    s = 10.0 ** np.interp(np.log10(N), log_n_arr, log_s_arr,
                            left=log_s_arr[0], right=log_s_arr[-1])
    return float(s)


if __name__ == "__main__":
    # Quick check: pure cyclic load between 0 and 30 MPa at R=0
    # mean = 15, alt = 15, sigma_uts = 48
    # sigma_a_eq = 15 / (1 - 15/48) = 15 / 0.6875 = 21.8 MPa
    # On the S-N curve, 22 MPa interpolates between (1e4, 28) and (1e5, 20):
    # log_s = (log22-log28)/(log20-log28)*log(1e5/1e4)+log(1e4)
    s_mean = np.array([15e6])
    s_alt = np.array([15e6])
    s_eq = equivalent_alt_stress_goodman(s_alt, s_mean, 48e6)
    N = sn_life(s_eq)
    print(f"  mean={s_mean[0]/1e6} MPa, alt={s_alt[0]/1e6} MPa => "
          f"eq={s_eq[0]/1e6:.2f} MPa => N={N[0]:.2e} cycles")
    # And a low-stress cycle near endurance
    s_eq2 = equivalent_alt_stress_goodman(np.array([6e6]), np.array([6e6]), 48e6)
    print(f"  6 MPa cycle => N={sn_life(s_eq2)[0]:.2e}")
    # High overload
    s_eq3 = equivalent_alt_stress_goodman(np.array([40e6]), np.array([5e6]), 48e6)
    print(f"  alt 40 MPa, mean 5 MPa => N={sn_life(s_eq3)[0]:.2e}")
