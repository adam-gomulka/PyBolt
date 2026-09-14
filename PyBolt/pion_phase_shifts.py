"""pi-pi phase shifts in the elastic region, sqrt(s) <= 2 M_K.

Parametrizations and CFD parameters are transcribed from Garcia-Martin, Kaminski,
Pelaez, Ruiz de Elvira, Yndurain, arXiv:1102.2183, Appendix A. 2211.03799 (App. E)
uses exactly these three waves (S0, P, S2) below the KKbar threshold.

The masses here are the ones that fit was made with. They are fit inputs and are
kept apart from the kinematic pion mass in pion_amplitudes.
"""

import numpy as np

M_PI_GM = 0.13957  # GeV
M_K_GM = 0.496  # GeV
SQRT_S_MAX = 2.0 * M_K_GM  # GeV
S_THRESHOLD = 4.0 * M_PI_GM**2  # GeV^2
S_M = 0.85**2  # S0 and S2 matching point, GeV^2

# --- CFD parameters, 1102.2183 App. A tables -------------------------------------
S0_B = (7.14, -25.3, -33.2, -26.2)
S0_Z0 = M_PI_GM
S0_D0 = 226.5  # deg
S0_C = -81.0  # deg

S2_B0, S2_B1 = -79.4, -63.0
S2_Z2 = 0.1435  # GeV
S2_BH2 = 32.0
S2_SL = 1.05**2  # GeV^2
S2_SH = 1.42**2  # GeV^2

P_B0, P_B1 = 1.043, 0.19
P_MRHO = 0.7736  # GeV
P_S0 = 1.05**2  # GeV^2


def _k(s):
    """Centre-of-mass pion momentum. [GeV]"""
    return np.sqrt(s / 4.0 - M_PI_GM**2)


def _w(s, s0):
    """Conformal variable of 1102.2183 eq. (ws)."""
    return (np.sqrt(s) - np.sqrt(s0 - s)) / (np.sqrt(s) + np.sqrt(s0 - s))


def _elastic(s):
    s = np.asarray(s, dtype=float)
    return s, (s > S_THRESHOLD) & (s <= SQRT_S_MAX**2)


# --- S0 -----------------------------------------------------------------------------

def _cot_delta00_low(s):
    """1102.2183 eq. (AppendixS0lowparam), s <= s_M, s0 = 4 M_K^2."""
    w = _w(s, 4.0 * M_K_GM**2)
    b0, b1, b2, b3 = S0_B
    brace = S0_Z0**2 / (M_PI_GM * np.sqrt(s)) + b0 + b1 * w + b2 * w**2 + b3 * w**3
    return np.sqrt(s) / (2.0 * _k(s)) * M_PI_GM**2 / (s - 0.5 * S0_Z0**2) * brace


def _delta00_low_deg(s):
    return np.degrees(np.arctan2(1.0, _cot_delta00_low(s)))


# delta_M and delta_M' are "obtained from Eq. (AppendixS0lowparam)" at s_M.
_H = 1e-7
_DELTA_M = float(_delta00_low_deg(S_M))  # deg
_DELTA_M_PRIME = float(
    (_delta00_low_deg(S_M + _H) - _delta00_low_deg(S_M - _H)) / (2.0 * _H)
)  # deg / GeV^2


def _delta00_mid_deg(s):
    """1102.2183 eq. (Appendixnewparam), first branch: s_M < s < 4 M_K^2."""
    k2 = np.sqrt(np.maximum(M_K_GM**2 - s / 4.0, 0.0))  # |k_2|
    k_m = np.sqrt(M_K_GM**2 - S_M / 4.0)
    r = k2 / k_m
    return (
        S0_D0 * (1.0 - r) ** 2
        + _DELTA_M * r * (2.0 - r)
        + k2 * (k_m - k2) * (8.0 * _DELTA_M_PRIME + S0_C * (k_m - k2) / M_K_GM**3)
    )


def delta00(s):
    """I=0 S-wave phase shift. [rad]"""
    s, elastic = _elastic(s)
    out = np.zeros_like(s)
    low = elastic & (s <= S_M)
    mid = elastic & (s > S_M)
    out[low] = np.radians(_delta00_low_deg(s[low]))
    out[mid] = np.radians(_delta00_mid_deg(s[mid]))
    return out


# --- S2 -----------------------------------------------------------------------------

def _cot_delta20(s):
    """1102.2183 eqs. (S2lowparam) and (S2highparam)."""
    prefactor = np.sqrt(s) / (2.0 * _k(s)) * M_PI_GM**2 / (s - 2.0 * S2_Z2**2)
    brace_low = S2_B0 + S2_B1 * _w(s, S2_SL)

    b_h0 = S2_B0 + S2_B1 * _w(S_M, S2_SL)
    b_h1 = (
        S2_B1 * (S2_SL / S2_SH) * np.sqrt(S2_SH - S_M) / np.sqrt(S2_SL - S_M)
        * ((np.sqrt(S_M) + np.sqrt(S2_SH - S_M)) / (np.sqrt(S_M) + np.sqrt(S2_SL - S_M))) ** 2
    )
    d = _w(s, S2_SH) - _w(S_M, S2_SH)
    brace_high = b_h0 + b_h1 * d + S2_BH2 * d**2

    return prefactor * np.where(s <= S_M, brace_low, brace_high)


def delta20(s):
    """I=2 S-wave phase shift, negative in this range. [rad]"""
    s, elastic = _elastic(s)
    out = np.zeros_like(s)
    out[elastic] = np.arctan(1.0 / _cot_delta20(s[elastic]))
    return out


# --- P --------------------------------------------------------------------------------

def delta11(s):
    """I=1 P-wave phase shift, 1102.2183 eq. (Plowparam), valid up to 2 M_K. [rad]"""
    s, elastic = _elastic(s)
    out = np.zeros_like(s)
    se = s[elastic]
    brace = 2.0 * M_PI_GM**3 / (P_MRHO**2 * np.sqrt(se)) + P_B0 + P_B1 * _w(se, P_S0)
    cot = np.sqrt(se) / (2.0 * _k(se) ** 3) * (P_MRHO**2 - se) * brace
    out[elastic] = np.arctan2(1.0, cot)
    return out
