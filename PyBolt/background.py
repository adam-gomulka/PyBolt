"""Pluggable expansion histories.

The Boltzmann solvers need three things from the background, and they are not
independent: the dt -> dx Jacobian, the coefficient of the momentum-redshift term in
the fBE, and (once entropy is injected) the rate at which comoving entropy grows.
All three follow from a single quantity,

    w(T) = -dlnT / dlna

so a background is defined by supplying ``H`` and ``w`` and nothing else.

Setting the entropy-conserving value ``w = 1/(1 + gtilda)`` recovers the standard
radiation-dominated cosmology exactly, which is what ``StandardCosmology`` does and
what ``tests/test_background.py`` pins down.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

from .constants import MPL
from .cosmology import (
    G_PLATEAU,
    H as H_radiation,
    g_rho,
    gtilda,
    h_s,
    s_ent,
)


class Background:
    """An expansion history.

    Subclasses supply ``H`` and ``w``. Everything the solvers actually call is
    derived from those two here, so a new cosmology cannot make the Jacobian and the
    redshift term disagree with each other.

    The derived quantities are deliberately *not* overridden by subclasses, even
    where a closed form is available: running the generic path on a case with a known
    answer is what tests that the generic path is right.
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

        Zero whenever entropy is conserved, so it costs nothing on a standard
        cosmology and only bites once a decaying field is sourcing the bath.
        """

        return 3.0 / self.w(T) - 3.0 - 3.0 * gtilda(T)


class StandardCosmology(Background):
    """Radiation domination with a conserved comoving entropy. The default."""

    def H(self, T):
        return H_radiation(T)

    def w(self, T):
        return 1.0 / (1.0 + gtilda(T))


# w during perturbative reheating: the bath is replenished as fast as it redshifts,
# so T ~ a^(-3/8) instead of a^(-1).
W_REHEATING = 3.0 / 8.0


class SuddenDecayReheating(Background):
    """Early matter domination ending in a hard switch to radiation at ``T_rh``.

    Above ``T_rh`` the decaying field dominates and the bath sits on the reheating
    attractor ``rho_R = (2/5) Gamma rho_phi / H``, which with
    ``H^2 = 8 pi rho / (3 MPL^2)`` gives

        H(T) = 2 pi^3 g_rho(T) T^4 / (9 Gamma MPL^2)

    depending on ``Gamma`` alone -- the initial inflaton density sets only ``T_max``,
    not the curve. Below ``T_rh`` this is exactly ``StandardCosmology``.

    ``Gamma = gamma_factor * H_rad(T_rh)`` is a convention rather than a matching
    condition, so ``H`` steps by exactly 5/6 at ``T_rh`` (independent of ``T_rh`` and
    of ``g_rho``). That step is deliberate: comparing it against
    ``PerturbativeReheating``, which resolves the transition properly, is what tells
    you whether the cheap background is good enough for a given question.

    Parameters
    ----------
    T_rh: float
        Reheat temperature [GeV].
    gamma_factor: float
        Sets ``Gamma`` in units of the radiation-domination Hubble rate at ``T_rh``.
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
        return np.where(T > self._T_rh, W_REHEATING, self._standard.w(T))


class PerturbativeReheating(Background):
    """Reheating solved as a two-fluid system rather than matched by hand.

    Integrates, in ``ln a``,

        d rho_phi / dlna = -3 rho_phi - Gamma rho_phi / H
        d s       / dlna = -3 s       + Gamma rho_phi / (H T)
        H = sqrt(8 pi (rho_phi + rho_R(T)) / 3) / MPL

    and splines ``H`` and ``w`` against ``T`` recovered from ``s``. Unlike
    ``SuddenDecayReheating`` the transition at ``T_rh`` is resolved, so the two
    together measure the error of the cheap background.

    The bath is sourced through its *entropy*, not its energy. Decay deposits
    ``Gamma rho_phi`` into a bath at temperature ``T`` and so produces entropy at
    ``Gamma rho_phi / T``. Evolving ``rho_R`` as ``-4 rho_R + source`` instead would
    assume ``g_rho`` is constant, which fails wherever a species is going
    non-relativistic -- most importantly across the QCD transition, which is exactly
    where a low ``T_rh`` puts the interesting physics.

    Both limits then come out exact rather than approximate. With the source off,
    ``s a^3`` is constant and ``w -> 1/(1+gtilda)``, which is ``StandardCosmology``
    including every dof feature. Deep in reheating ``s ~ a^(-9/8)`` and ``w -> 3/8``.

    Integration starts *on* the reheating attractor at ``start_factor * T_start``
    rather than from ``rho_R = 0``, which skips the rising-temperature branch
    entirely -- that branch would make ``x = m/T`` non-monotonic and is out of scope.
    Because the attractor is fixed by ``Gamma`` alone, ``start_factor`` moves where
    the integration comes in without moving the resulting ``H(T)``.

    It stops once the source term has died away, and below that hands over to
    ``StandardCosmology``. The handover is continuous in both ``H`` and ``w`` by
    construction: with the inflaton gone ``sqrt(8 pi rho_R/3)/MPL`` is exactly
    ``H_rad``, and ``s a^3 = const`` is exactly what ``StandardCosmology`` assumes.
    It is a shortcut past integrating a dead source, not a change of model.

    Parameters
    ----------
    T_rh: float
        Reheat temperature [GeV], defining ``Gamma = gamma_factor * H_rad(T_rh)``.
    T_start: float
        Temperature at which the Boltzmann run begins [GeV].
    gamma_factor: float
        ``Gamma`` in units of the radiation-domination Hubble rate at ``T_rh``.
    start_factor: float
        How far above ``T_start`` to begin integrating. Affects only coverage.
    """

    # Hand over to the standard cosmology once entropy injection is this small
    # compared with the -3s dilution it is competing against.
    SOURCE_FLOOR = 1e-6

    _LNA_MAX = 200.0
    _N_SAMPLES = 4000

    def __init__(
        self,
        T_rh: float,
        T_start: float,
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

        self._T_rh = T_rh
        self._T_start = T_start
        self._gamma_factor = gamma_factor
        self._Gamma = gamma_factor * H_radiation(T_rh)
        self._standard = StandardCosmology()

        self._integrate(start_factor * T_start)

    # -- construction ---------------------------------------------------------

    def _attractor_H(self, T):
        return 2.0 * np.pi**3 * g_rho(T) * T**4 / (9.0 * self._Gamma * MPL**2)

    @staticmethod
    def _rho_R_of_T(T):
        return np.pi**2 * g_rho(T) * T**4 / 30.0

    @staticmethod
    def _T_of_entropy(s):
        """Invert s = h_s(T) 4 pi^2 T^3 / 90 by fixed-point iteration.

        h_s varies slowly with T and is flat above the table, so a handful of passes
        is plenty. Seeded with the plateau value.
        """
        T = (90.0 * s / (4.0 * np.pi**2 * G_PLATEAU)) ** (1.0 / 3.0)
        for _ in range(12):
            T = (90.0 * s / (4.0 * np.pi**2 * h_s(T))) ** (1.0 / 3.0)

        return T

    def _hubble(self, rho_phi, rho_R):
        return np.sqrt(8.0 * np.pi * (rho_phi + rho_R) / 3.0) / MPL

    def _integrate(self, T_hi):
        # Start on the attractor: H is set by the (dominant) inflaton, and the bath
        # carries the rho_R that the attractor implies at this temperature.
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

        # w = -dlnT/dlna, from dln(s)/dlna and s ~ h_s(T) T^3, so that
        # dln s/dlna = 3 (1 + gtilda) dlnT/dlna. With the source off this is exactly
        # 1/(1+gtilda); deep in reheating s ~ a^(-9/8) and it is exactly 3/8.
        entropy_source = self._Gamma * rho_phi / (H * T)
        dln_s_dlna = -3.0 + entropy_source / s
        w = -dln_s_dlna / (3.0 * (1.0 + gtilda(T)))

        # T falls monotonically on this branch; splines need an increasing abscissa.
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
    def T_range(self):
        """Temperatures the integration covers; below this it is StandardCosmology."""

        return (self._T_lo, self._T_hi)

    def _check_range(self, T):
        if np.any(np.asarray(T) > self._T_hi):
            raise ValueError(
                "temperature above the integrated range (max {:.4e} GeV); raise "
                "start_factor to cover it".format(self._T_hi)
            )

    def H(self, T):
        self._check_range(T)
        spline = np.exp(self._log_H_spline(np.log(T)))

        return np.where(T < self._T_lo, self._standard.H(T), spline)

    def w(self, T):
        self._check_range(T)

        return np.where(T < self._T_lo, self._standard.w(T), self._w_spline(np.log(T)))
