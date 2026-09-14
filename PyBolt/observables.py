"""Observables built from a final axion spectrum."""

import numpy as np

from .cosmology import h_s

T_NU = 3e-3  # GeV, where neutrinos start decoupling (2211.03799 App. A)


def delta_neff(q, F, T_final, T_nu=T_NU):
    """2211.03799 App. A: (8/7) rho_a/rho_gamma at T_final, times
    [g_{*S}(T_final)/g_{*S}(T_nu)]^{-4/3}. F = q^2 f on a q = k/T grid, massless axion.

    rho_a/rho_gamma = [T^4/(2 pi^2) int q F dq] / [pi^2 T^4 / 15].
    """
    ratio = 15.0 / (2.0 * np.pi**4) * np.trapezoid(np.asarray(q) * np.asarray(F), q)
    dilution = (h_s(T_final) / h_s(T_nu)) ** (-4.0 / 3.0)
    return float(8.0 / 7.0 * ratio * dilution)
