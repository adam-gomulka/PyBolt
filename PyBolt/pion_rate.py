"""Axion destruction rate a pi -> pi pi.

Gamma^>(k, T) = (eps f_pi / 2 f_a)^2 T G(q, x), q = k/T, x = m_pi/T.

G is the integral of 2211.03799 eq. (E.2), evaluated in units of T with the reduced
|M|^2 of pion_amplitudes. The beta integral is done with the delta function as in
Hannestad & Madsen, astro-ph/9506015, eqs. (A8)-(A11): the on-shell condition for the
second outgoing pion, h(beta) = E3^2 - |k3|^2 - m^2, is linear in cos(beta), and
|dh/dbeta| = 2 |k1||k2| sin(alpha) sin(theta) |sin(beta0)|.
Labels follow 2211.03799: k axion, k1 incoming pion, k2 outgoing pion at angle theta
to k, k3 = k + k1 - k2 fixed by momentum conservation.
"""

import json

import numpy as np
from scipy.interpolate import RectBivariateSpline

from .constants import Zeta3
from .pion_amplitudes import M_PI

_EDGE = 1e-12


def _bose(E):
    # expm1 overflows to inf for E > ~709; 1/inf = 0 is the right occupation.
    with np.errstate(over="ignore"):
        return 1.0 / np.expm1(E)


def destruction_integrand(u, q, x, msq_hat):
    """Integrand of G over the unit 4-cube, Jacobian included."""
    u = np.clip(np.atleast_2d(u), _EDGE, 1.0 - _EDGE)
    m = x  # pion mass in units of T
    T = M_PI / x  # GeV, to hand GeV^2 Mandelstams to msq_hat

    k1 = u[:, 0] / (1.0 - u[:, 0])
    k2 = u[:, 1] / (1.0 - u[:, 1])
    jacobian = 4.0 / ((1.0 - u[:, 0]) ** 2 * (1.0 - u[:, 1]) ** 2)
    ca = 2.0 * u[:, 2] - 1.0
    ct = 2.0 * u[:, 3] - 1.0
    sa = np.sqrt(1.0 - ca**2)
    st = np.sqrt(1.0 - ct**2)

    E1 = np.sqrt(k1**2 + m**2)
    E2 = np.sqrt(k2**2 + m**2)
    E3 = q + E1 - E2

    # h(beta) = h0 + h1 cos(beta)
    h0 = (E3**2 - m**2 - q**2 - k1**2 - k2**2
          - 2.0 * q * k1 * ca + 2.0 * q * k2 * ct + 2.0 * k1 * k2 * ca * ct)
    h1 = 2.0 * k1 * k2 * sa * st

    out = np.zeros(u.shape[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        cb = -h0 / h1
        ok = (E3 > 0.0) & (h1 > 0.0) & (cb**2 < 1.0)

    k1, k2, E1, E2, E3 = k1[ok], k2[ok], E1[ok], E2[ok], E3[ok]
    ca, ct, sa, st, cb = ca[ok], ct[ok], sa[ok], st[ok], cb[ok]
    h_prime = h1[ok] * np.sqrt(1.0 - cb**2)

    s = m**2 + 2.0 * q * (E1 - k1 * ca)  # (k + k1)^2, massless axion
    t = 2.0 * m**2 - 2.0 * (E1 * E2 - k1 * k2 * (ca * ct + sa * st * cb))  # (k1 - k2)^2 = (k - k3)^2
    msq = msq_hat(s * T**2, t * T**2)

    stat = _bose(E1) * (1.0 + _bose(E2)) * (1.0 + _bose(E3))
    out[ok] = (
        1.0 / (2.0 * q) / (2.0 * np.pi) ** 4
        / (2.0 * E1) / (2.0 * E2)
        * stat * 2.0 / h_prime * k1**2 * k2**2 * msq
        * jacobian[ok]
    )
    return out


def gamma_reduced(q, x, msq_hat, neval=200_000, nitn=10):
    """G(q, x) by vegas. Returns (mean, sdev)."""
    import vegas  # tabulation-only dependency

    @vegas.lbatchintegrand
    def integrand(u):
        return destruction_integrand(u, q, x, msq_hat)

    integrator = vegas.Integrator(4 * [[0.0, 1.0]])
    integrator(integrand, nitn=max(2, nitn // 2), neval=neval // 5)  # adapt the grid
    result = integrator(integrand, nitn=nitn, neval=neval)
    return float(result.mean), float(result.sdev)


def thermal_average(q, G, weight):
    """(1/n_hat) int d^3q/(2pi)^3 w(q) G(q), n_hat = zeta3/pi^2.

    weight "absorption": w = f_BE (hep-ph/0504059 convention).
    weight "production": w = e^{-q} = f_BE/(1+f_BE), 2211.03799 eqs. (4), (E.8).
    """
    q = np.asarray(q, dtype=float)
    if weight == "absorption":
        w = 1.0 / np.expm1(q)
    elif weight == "production":
        w = np.exp(-q)
    else:
        raise ValueError("weight must be 'absorption' or 'production'")
    integral = np.trapezoid(q**3 * w * np.asarray(G), np.log(q)) / (2.0 * np.pi**2)
    return integral / (Zeta3 / np.pi**2)


class PionRateTable:
    """G(q, x) on a (log10 q, log10 x) grid, interpolated cubically in log G.

    Values are f_a- and eps-independent, so one table serves a whole f_a scan.
    """

    def __init__(self, log10_q, log10_x, G, sdev=None, meta=None):
        self.log10_q = np.asarray(log10_q, dtype=float)
        self.log10_x = np.asarray(log10_x, dtype=float)
        self.G = np.asarray(G, dtype=float)
        self.sdev = None if sdev is None else np.asarray(sdev, dtype=float)
        self.meta = dict(meta or {})

        if self.G.shape != (self.log10_q.size, self.log10_x.size):
            raise ValueError("G must have shape (len(log10_q), len(log10_x))")
        if not np.all(self.G > 0.0):
            raise ValueError("G must be strictly positive; floor it when tabulating")

        # s=0: an interpolating bicubic spline, exact at the nodes.
        self._spline = RectBivariateSpline(self.log10_q, self.log10_x, np.log(self.G),
                                           kx=3, ky=3, s=0)

    @property
    def q_nodes(self):
        return 10.0**self.log10_q

    # log10(10**a) lands a few ulps off a; snap such points onto the edge rather than
    # reject them. Anything further out is a genuine request outside the table.
    _EDGE_TOLERANCE = 1e-9

    def _snap(self, values, axis):
        lo, hi = axis[0], axis[-1]
        near = (values > lo - self._EDGE_TOLERANCE) & (values < hi + self._EDGE_TOLERANCE)
        return np.where(near, np.clip(values, lo, hi), values)

    def __call__(self, q, x):
        q = np.asarray(q, dtype=float)
        log_q = self._snap(np.log10(q).ravel(), self.log10_q)
        log_x = self._snap(np.full(q.size, np.log10(x)), self.log10_x)
        outside = ((log_q < self.log10_q[0]) | (log_q > self.log10_q[-1])
                   | (log_x < self.log10_x[0]) | (log_x > self.log10_x[-1]))
        if np.any(outside):
            raise ValueError("(q, x) outside the rate table: q in [{:.3g}, {:.3g}], "
                             "x in [{:.3g}, {:.3g}]".format(*10.0**self.log10_q[[0, -1]],
                                                          *10.0**self.log10_x[[0, -1]]))
        return np.exp(self._spline.ev(log_q, log_x)).reshape(q.shape)

    def save(self, path):
        np.savez(
            path,
            log10_q=self.log10_q,
            log10_x=self.log10_x,
            G=self.G,
            sdev=np.array([]) if self.sdev is None else self.sdev,
            meta=json.dumps(self.meta),
        )

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            sdev = data["sdev"] if data["sdev"].size else None
            return cls(data["log10_q"], data["log10_x"], data["G"], sdev=sdev,
                       meta=json.loads(str(data["meta"])))
