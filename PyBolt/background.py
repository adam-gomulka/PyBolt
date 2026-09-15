"""Pluggable expansion histories.

A background supplies ``H(T)`` and ``w(T) = -dlnT/dlna``. Everything the solvers
need (the dt -> dx Jacobian, the momentum-redshift coefficient and the growth of
comoving entropy) follows from these two.

``w = 1/(1 + gtilda)`` conserves entropy and gives ``StandardCosmology``.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

from .constants import MPL
from .cosmology import (
    G_PLATEAU,
    H as H_radiation,
    LOG10_T_TABLE_MAX,
    g_rho,
    gtilda,
    gy_spline,
    h_s,
    s_ent,
)

_dg_rho_dlog10T = gy_spline.derivative()


def dln_g_rho_dln_T(T):
    """``dln g_rho / dln T``. Zero above the table, where g_rho is clamped flat."""

    log10T = np.log10(T)
    inside = _dg_rho_dlog10T(log10T) / (g_rho(T) * np.log(10.0))

    return np.where(log10T > LOG10_T_TABLE_MAX, 0.0, inside)


class Background:
    """An expansion history.

    Subclasses supply H and w; the derived quantities below should not be
    overridden, so that they stay consistent with each other.
    """

    def H(self, T):
        """Hubble rate at photon temperature ``T``. [GeV]"""

        raise NotImplementedError

    def w(self, T):
        """``-dlnT/dlna``. Equals ``1/(1+gtilda)`` when entropy is conserved. [1]"""

        raise NotImplementedError

    # -- derived --------------------------------------------------------------

    def H_t(self, T):
        """``-dlnT/dt``, the Jacobian for using ``x = m/T`` as the time variable. [GeV]"""

        return self.w(T) * self.H(T)

    def redshift_coeff(self, T):
        """Coefficient of ``(q df/dq - 2f)`` in the fBE right-hand side. [1]

        This is ``H/H_t - 1``, which reduces to ``gtilda`` in the adiabatic case.
        """

        w = self.w(T)

        return (1.0 - w) / w

    def dln_sa3_dlnx(self, T):
        """Growth rate of the comoving entropy ``s a^3`` per unit ``ln x``. [1]

        Zero whenever entropy is conserved.
        """

        return 3.0 / self.w(T) - 3.0 - 3.0 * gtilda(T)


class StandardCosmology(Background):
    """Radiation domination with a conserved comoving entropy. The default."""

    def H(self, T):
        return H_radiation(T)

    def w(self, T):
        return 1.0 / (1.0 + gtilda(T))


def w_reheating(T):
    """``-dlnT/dlna`` while the decaying field replenishes the bath.

    On the reheating attractor rho_R ~ Gamma rho_phi / H ~ a^-3/2. With
    rho_R = (pi^2/30) g_rho(T) T^4 this gives

        (4 + dln g_rho/dln T) dlnT/dlna = -3/2

    i.e. w = 3/8 for constant g_rho, dropping to about 0.25 at the QCD transition.
    """

    return 1.5 / (4.0 + dln_g_rho_dln_T(T))


class SuddenDecayReheating(Background):
    """Early matter domination ending in a hard switch to radiation at T_rh.

    Above T_rh the decaying field dominates and the bath sits on the reheating
    attractor rho_R = (2/5) Gamma rho_phi / H, which with
    H^2 = 8 pi rho / (3 MPL^2) gives

        H(T) = 2 pi^3 g_rho(T) T^4 / (9 Gamma MPL^2)

    independent of the initial inflaton density. Below T_rh this is
    ``StandardCosmology``.

    With Gamma = gamma_factor * H_rad(T_rh) and the default gamma_factor = 3, H
    jumps by a factor 5/6 at T_rh. ``PerturbativeReheating`` resolves the
    transition smoothly.

    Parameters
    ----------
    T_rh: float
        Reheat temperature [GeV].
    gamma_factor: float
        Sets Gamma in units of the radiation-domination Hubble rate at T_rh.
    """

    def __init__(self, T_rh: float, gamma_factor: float = 3.0):
        if T_rh <= 0.0:
            raise ValueError("T_rh must be positive, got {}".format(T_rh))

        self._T_rh = T_rh
        self._gamma_factor = gamma_factor
        self._Gamma = gamma_factor * H_radiation(T_rh)
        self._standard = StandardCosmology()

    @property
    def T_rh(self):
        return self._T_rh

    @property
    def Gamma(self):
        """Decay width of the field driving reheating. [GeV]"""

        return self._Gamma

    def H(self, T):
        reheating = 2.0 * np.pi**3 * g_rho(T) * T**4 / (9.0 * self._Gamma * MPL**2)

        return np.where(T > self._T_rh, reheating, self._standard.H(T))

    def w(self, T):
        return np.where(T > self._T_rh, w_reheating(T), self._standard.w(T))


class PerturbativeReheating(Background):
    """Reheating integrated as a two-fluid system.

    Integrates, in ln a,

        d rho_phi / dlna = -3 rho_phi - Gamma rho_phi / H
        d s       / dlna = -3 s       + Gamma rho_phi / (H T)
        H = sqrt(8 pi (rho_phi + rho_R(T)) / 3) / MPL

    and splines H and w against T recovered from s. The bath is sourced through
    its entropy, which stays valid when g_rho varies.

    Integration starts on the reheating attractor at T_max, skipping the phase in
    which T rises (x = m/T must be monotonic). It stops once the source is
    negligible; below that the background is ``StandardCosmology``.

    Parameters
    ----------
    T_rh: float
        Reheat temperature [GeV], defining ``Gamma = gamma_factor * H_rad(T_rh)``.
    T_start: float
        Temperature at which the Boltzmann run begins [GeV].
    T_max: float, optional
        Highest temperature the bath reached [GeV]. Defaults to
        start_factor * T_start.
    gamma_factor: float
        Gamma in units of the radiation-domination Hubble rate at T_rh.
    start_factor: float
        Sets the default T_max.
    """

    # Stop integrating once the entropy source is this fraction of the 3s dilution.
    SOURCE_FLOOR = 1e-6

    _LNA_MAX = 200.0
    _N_SAMPLES = 4000

    def __init__(
        self,
        T_rh: float, #GeV
        T_start: float, #GeV
        T_max: float = None, #GeV
        gamma_factor: float = 3.0,
        start_factor: float = 10.0,
    ):
        if T_rh <= 0.0:
            raise ValueError("T_rh must be positive, got {}".format(T_rh))
        if T_start <= T_rh:
            raise ValueError(
                "T_start ({}) must be above T_rh ({}), otherwise the run never sees "
                "the reheating era".format(T_start, T_rh)
            )

        if T_max is None:
            # H(T) on the attractor depends only on Gamma, so this choice is harmless.
            T_max = start_factor * T_start
        elif T_max < T_start:
            raise ValueError(
                "T_start ({:.4e} GeV) is above T_max ({:.4e} GeV): the universe never "
                "reached that temperature in this scenario, so there is nothing to "
                "start from. Lower the starting temperature or raise T_max.".format(
                    T_start, T_max)
            )

        self._T_rh = T_rh
        self._T_start = T_start
        self._T_max = T_max
        self._gamma_factor = gamma_factor
        self._Gamma = gamma_factor * H_radiation(T_rh)
        self._standard = StandardCosmology()

        self._integrate(T_max)

    # -- construction ---------------------------------------------------------

    def _attractor_H(self, T):
        return 2.0 * np.pi**3 * g_rho(T) * T**4 / (9.0 * self._Gamma * MPL**2)

    @staticmethod
    def _rho_R_of_T(T):
        return np.pi**2 * g_rho(T) * T**4 / 30.0

    @staticmethod
    def _T_of_entropy(s):
        """Invert s = h_s(T) 4 pi^2 T^3 / 90 by fixed-point iteration."""
        T = (90.0 * s / (4.0 * np.pi**2 * G_PLATEAU)) ** (1.0 / 3.0)
        for _ in range(12):
            T = (90.0 * s / (4.0 * np.pi**2 * h_s(T))) ** (1.0 / 3.0)

        return T

    def _hubble(self, rho_phi, rho_R):
        return np.sqrt(8.0 * np.pi * (rho_phi + rho_R) / 3.0) / MPL

    def _integrate(self, T_hi):
        # Initial state on the attractor.
        H_hi = self._attractor_H(T_hi)
        rho_R_hi = self._rho_R_of_T(T_hi)
        rho_tot_hi = 3.0 * MPL**2 * H_hi**2 / (8.0 * np.pi)
        rho_phi_hi = rho_tot_hi - rho_R_hi
        s_hi = s_ent(T_hi)

        if rho_phi_hi <= 0.0:
            raise ValueError(
                "the attractor puts no inflaton energy at T={}; T_rh and T_start are "
                "probably too close together".format(T_hi)
            )

        def state(y):
            """(T, rho_R, H, entropy source) for one point of the integration."""
            rho_phi, s = y
            T = self._T_of_entropy(s)
            rho_R = self._rho_R_of_T(T)
            H = self._hubble(rho_phi, rho_R)

            return T, rho_R, H, self._Gamma * rho_phi / (H * T)

        def rhs(lna, y):
            rho_phi, s = y
            T, _, H, entropy_source = state(y)

            return [
                -3.0 * rho_phi - self._Gamma * rho_phi / H,
                -3.0 * s + entropy_source,
            ]

        def source_exhausted(lna, y):
            """Injection has become negligible against the -3s dilution term."""
            _, _, _, entropy_source = state(y)

            return entropy_source / (3.0 * y[1]) - self.SOURCE_FLOOR

        source_exhausted.terminal = True
        source_exhausted.direction = -1

        lna = np.linspace(0.0, self._LNA_MAX, self._N_SAMPLES)
        sol = solve_ivp(
            rhs,
            [0.0, self._LNA_MAX],
            [rho_phi_hi, s_hi],
            t_eval=lna,
            events=source_exhausted,
            method="LSODA",
            rtol=1e-10,
            atol=1e-300,
        )

        if not sol.success:
            raise RuntimeError("reheating background failed to integrate: " + sol.message)

        rho_phi, s = sol.y
        T = self._T_of_entropy(s)
        rho_R = self._rho_R_of_T(T)
        H = self._hubble(rho_phi, rho_R)

        # s ~ h_s(T) T^3 gives dln s/dlna = -3 (1 + gtilda) w.
        entropy_source = self._Gamma * rho_phi / (H * T)
        dln_s_dlna = -3.0 + entropy_source / s
        w = -dln_s_dlna / (3.0 * (1.0 + gtilda(T)))

        # T decreases with a; splines need an increasing abscissa.
        order = np.argsort(T)
        logT = np.log(T[order])
        self._log_H_spline = CubicSpline(logT, np.log(H[order]))
        self._w_spline = CubicSpline(logT, w[order])
        self._T_lo = float(T.min())
        self._T_hi = float(T.max())

    # -- interface ------------------------------------------------------------

    @property
    def Gamma(self):
        """Decay width of the field driving reheating. [GeV]"""

        return self._Gamma

    @property
    def T_max(self):
        """Highest temperature the bath reached; a run cannot start above it. [GeV]"""

        return self._T_max

    @property
    def T_range(self):
        """Temperatures the integration covers; below this it is StandardCosmology."""

        return (self._T_lo, self._T_hi)

    # Absorbs round-off in T = m/(m/T) for a run starting exactly at T_max.
    _RANGE_TOLERANCE = 1e-9

    def _check_range(self, T):
        if np.any(np.asarray(T) > self._T_hi * (1.0 + self._RANGE_TOLERANCE)):
            raise ValueError(
                "temperature above the integrated range (max {:.4e} GeV); raise "
                "T_max to cover it".format(self._T_hi)
            )

    def H(self, T):
        self._check_range(T)
        spline = np.exp(self._log_H_spline(np.log(T)))

        return np.where(T < self._T_lo, self._standard.H(T), spline)

    def w(self, T):
        self._check_range(T)

        return np.where(T < self._T_lo, self._standard.w(T), self._w_spline(np.log(T)))
