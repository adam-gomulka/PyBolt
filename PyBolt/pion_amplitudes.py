"""Reduced a-pi scattering amplitudes.

2211.03799 eq. (6): M_{a pi -> pi pi} = (eps f_pi / 2 f_a) M_{pi0 pi -> pi pi}. The
factor (eps f_pi / 2 f_a)^2 is kept out of everything here, so each function returns
the "reduced" |M|^2 that multiplies it:

  pheno: |M|^2_pipi of 2211.03799 eq. (E.7), from measured phase shifts
  LO:    (s^2 + t^2 + u^2 - 3 m^4) / f_pi^4, 2211.03799 App. B, s + t + u = 3 m^2
"""

import numpy as np

from .pion_phase_shifts import SQRT_S_MAX, delta00, delta11, delta20

M_PI = 0.138  # GeV, average pion mass used in 2211.03799
F_PI = 0.0923  # GeV
Z_UD = 0.47  # m_u / m_d
EPSILON = (1.0 - Z_UD) / (1.0 + Z_UD)


def isospin_sum_msq(T0, T1, T2):
    """(|T0|^2 + 3|T1|^2 + 5|T2|^2) / 6.

    Equal to 2211.03799 eq. (E.7) with T^I = 32 pi sum (2l+1) P_l t^I_l; the printed
    32 pi^2 / 3 has to be read as (32 pi)^2 / 3.
    """
    return (np.abs(T0) ** 2 + 3.0 * np.abs(T1) ** 2 + 5.0 * np.abs(T2) ** 2) / 6.0


def weinberg_A(s, t, u):
    """LO chiPT pi-pi amplitude A(s,t,u) (Gasser & Leutwyler)."""
    return (s - M_PI**2) / F_PI**2


def isospin_amplitudes(A, s, t, u):
    """s-channel isospin amplitudes, 2211.03799 App. E."""
    return (
        3.0 * A(s, t, u) + A(t, u, s) + A(u, s, t),
        A(t, u, s) - A(u, s, t),
        A(t, u, s) + A(u, s, t),
    )


def msq_hat_lo(s, t):
    """Reduced LO |M|^2, 2211.03799 App. B."""
    u = 3.0 * M_PI**2 - s - t
    return (s**2 + t**2 + u**2 - 3.0 * M_PI**4) / F_PI**4


def cos_xi(s, t):
    """t = (s - 4 m^2)(cos xi - 1)/2 inverted, with the a-pi t. Clipped (O(m^2/s))."""
    return np.clip(1.0 + 2.0 * t / (s - 4.0 * M_PI**2), -1.0, 1.0)


def _partial_wave(s, delta):
    """t^I_l = sqrt(s/(s-4m^2)) e^{i delta} sin delta, 2211.03799 App. E."""
    return np.sqrt(s / (s - 4.0 * M_PI**2)) * np.sin(delta) * np.exp(1j * delta)


def msq_hat_pheno(s, t):
    """Reduced phenomenological |M|^2 (S0, P, S2 only; zero above 2 M_K)."""
    s, t = np.broadcast_arrays(np.asarray(s, dtype=float), np.asarray(t, dtype=float))
    out = np.zeros(s.shape)

    window = (s > 4.0 * M_PI**2) & (np.sqrt(s) <= SQRT_S_MAX)
    sw, tw = s[window], t[window]
    t00 = _partial_wave(sw, delta00(sw))
    t11 = _partial_wave(sw, delta11(sw))
    t20 = _partial_wave(sw, delta20(sw))

    T0 = 32.0 * np.pi * t00
    T1 = 32.0 * np.pi * 3.0 * t11 * cos_xi(sw, tw)
    T2 = 32.0 * np.pi * t20
    out[window] = isospin_sum_msq(T0, T1, T2)
    return out
