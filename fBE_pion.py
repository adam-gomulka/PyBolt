#!/usr/bin/env python3
"""Hot axions from a pi <-> pi pi below T_c = 150 MeV (arXiv:2211.03799).

Runs end at 15 MeV, not the paper's 30 MeV. For f_a <~ 1e6 GeV the axions are still
coupled at 30 MeV and stopping there treats them as decoupled before muon
annihilation; by 15 MeV Gamma/H < 1e-6 even at f_a = 1e5 GeV. For f_a >~ 1e7 GeV the
two agree to 1e-4.

Zero axion abundance at T_c ("Pions only"). One rate table serves the whole f_a scan.

Under a reheating background (--bg sudden|reheating) the run starts at
min(T_c, T_max) and ends only once entropy injection is over, so that the
g_{*S} dilution in delta_neff is the whole story from there on.
"""

import argparse
import os
import sys

import numpy as np

from PyBolt import axion_grid
from PyBolt import boltzmann_solver as bz
from PyBolt.background import PerturbativeReheating, StandardCosmology, SuddenDecayReheating
from PyBolt.observables import T_NU, delta_neff
from PyBolt.pion_amplitudes import M_PI
from PyBolt.pion_rate import PionRateTable
from PyBolt.processes import PionScatteringToAxion

T_START = 0.150  # GeV, T_c
T_END = 0.015  # GeV, just above the rate table's lowest temperature (13.8 MeV)
# Energy-weighted Gamma/H above which the axions are not decoupled where the pion rate
# stops. The rate falls by ~10 per e-fold there, so 0.1 bounds the missed change at ~1%.
COUPLED_WARNING = 0.1
N_X = 500
M_DM = 1.0e-10  # GeV, massless for production purposes
G_X = 1.0
PARTICLE_TYPE = "b"
DEFAULT_TABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                             "pion_rate_pheno.npz")
BACKGROUNDS = ("standard", "sudden", "reheating")
SOLVER_OPTIONS = {"method": "LSODA", "rtol": 1e-6, "atol": 1e-24, "lband": 2, "uband": 2}


def build_background(bg="standard", T_rh=None, T_max=None):
    """The expansion history, temperatures in GeV.

    PerturbativeReheating is integrated from T_START (or T_max, if lower) upwards.
    """
    if bg == "standard":
        return StandardCosmology()
    if bg not in BACKGROUNDS:
        raise ValueError("bg must be one of {}, got {!r}".format(BACKGROUNDS, bg))
    if T_rh is None:
        raise ValueError("--bg {} needs --t-rh".format(bg))
    if bg == "sudden":
        return SuddenDecayReheating(T_rh=T_rh)
    T_start = T_START if T_max is None else min(T_START, T_max)
    return PerturbativeReheating(T_rh=T_rh, T_start=T_start, T_max=T_max)


def run_window(background):
    """(T_start, T_end) in GeV for a run against ``background``.

    Starts at T_c, or at T_max when the bath never got that hot. Ends at T_END or,
    under reheating, where entropy injection has stopped (T_rh for the sudden model,
    the handover to StandardCosmology for the integrated one, about T_rh/3).
    """
    if isinstance(background, StandardCosmology):
        return T_START, T_END

    if isinstance(background, SuddenDecayReheating):
        T_rh, T_start, injection_end = background.T_rh, T_START, background.T_rh
    else:
        T_rh = None
        T_start = min(T_START, background.T_max)
        injection_end = background.T_range[0]

    if T_rh is not None and T_rh >= T_START:
        raise ValueError("T_rh = {:.3g} GeV is not below T_c; use --bg standard".format(T_rh))

    T_end = min(T_END, injection_end)
    if T_end <= T_NU:
        raise ValueError(
            "entropy injection lasts until T = {:.3g} MeV, past neutrino decoupling "
            "({:.3g} MeV); delta_neff does not cover that. Raise T_rh.".format(
                injection_end * 1e3, T_NU * 1e3))
    return T_start, T_end


def coupling_where_rate_stops(f_a, table, background, T_end, q):
    """Energy-weighted Gamma/H where production ends: at T_end, or at the table's lowest
    temperature if the run goes below it (the rate is zero there).

    Gamma_E = int q^3 f_BE Gamma dq / int q^3 f_BE dq with Gamma = Gamma^> (1 - e^{-q})
    the relaxation rate towards equilibrium (2211.03799 App. A), so f_BE Gamma = Gamma^<.
    It is how fast the energy density, and so Delta N_eff, still changes. The high-q
    tail stays coupled longer but carries no energy, so a plain max over q misleads.
    """
    T_table_min = M_PI / 10.0 ** table.log10_x[-1]
    T_stop = max(T_end, T_table_min)
    gamma_less = PionScatteringToAxion(table, f_a).gamma_destruction(M_PI / T_stop, q) * np.exp(-q)
    gamma_energy = np.trapezoid(q**3 * gamma_less, q) / np.trapezoid(q**3 / np.expm1(q), q)
    return float(gamma_energy / float(background.H(T_stop))), T_stop


def x_grid(n_x, T_start=T_START, T_end=T_END):
    return np.linspace(M_PI / T_start, M_PI / T_end, n_x)


def _model(x, q, background=None):
    model = bz.Model(M_PI, M_DM, G_X, PARTICLE_TYPE, background=background)
    model.changeGrid(x, q)
    return model


def solve_distribution(f_a, table, x, q, background=None):
    """Final F = q^2 f at the end of the x grid."""
    model = _model(x, q, background)
    model.addCollisionTerm(PionScatteringToAxion(table, f_a).collisionTerm)
    model.solve_fBE(np.zeros(len(q)), dict(SOLVER_OPTIONS))  # raises on failure
    return model.getSolution()[-1, :]


def solve_number_density(f_a, table, x, background=None):
    """Final Y = n/s at the end of the x grid."""
    process = PionScatteringToAxion(table, f_a)
    Y = _model(x, np.linspace(0.01, 30.0, 10), background).solve_nBE(x, process.rate, 0.0)
    if Y is None:
        raise RuntimeError("solve_nBE did not converge for f_a={:.5e}".format(f_a))
    return Y[-1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--mode", choices=("fbe", "nbe"), default="fbe")
    parser.add_argument("--f_min", type=float, default=1e6)
    parser.add_argument("--f_max", type=float, default=1e9)
    parser.add_argument("--f_num", type=int, default=30)
    parser.add_argument("--q_min", type=float, default=0.01)
    parser.add_argument("--q_max", type=float, default=30.0)
    parser.add_argument("--q_num", type=int, default=250)
    parser.add_argument("--n_x", type=int, default=N_X)
    parser.add_argument("--bg", choices=BACKGROUNDS, default="standard",
                        help="expansion history (default: standard)")
    parser.add_argument("--t-rh", dest="t_rh", type=float, default=None,
                        help="reheat temperature in GeV, for --bg sudden|reheating")
    parser.add_argument("--t-max", dest="t_max", type=float, default=None,
                        help="highest bath temperature in GeV, for --bg reheating")
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    table = PionRateTable.load(args.table)
    background = build_background(args.bg, args.t_rh, args.t_max)
    T_start, T_end = run_window(background)
    print("background {}: T = {:.4g} -> {:.4g} MeV".format(args.bg, T_start * 1e3, T_end * 1e3))
    x = x_grid(args.n_x, T_start, T_end)
    q = np.linspace(args.q_min, args.q_max, args.q_num)
    f_vals = axion_grid.f_grid(args.f_min, args.f_max, args.f_num)

    # The smallest f_a is the most strongly coupled one.
    ratio, T_stop = coupling_where_rate_stops(f_vals.min(), table, background, T_end, q)
    if ratio > COUPLED_WARNING:
        print("WARNING: at f_a = {:.3g} GeV the axions are still coupled (Gamma/H = {:.2g}) "
              "where pion production stops, T = {:.3g} MeV. Delta N_eff for such f_a is "
              "a lower bound.".format(f_vals.min(), ratio, T_stop * 1e3), file=sys.stderr)

    stem, _ = os.path.splitext(args.output)
    dneff_rows = []
    with open(args.output, "w") as handle:
        if args.mode == "fbe":
            handle.write(axion_grid.HEADER)
            handle.write(axion_grid.format_q_header(q))
        else:
            handle.write(axion_grid.HEADER_NUMBER_DENSITY)

        for f_a in f_vals:
            if args.mode == "fbe":
                F = solve_distribution(f_a, table, x, q, background)
                handle.write(axion_grid.format_row(f_a, F))
                dneff_rows.append((f_a, delta_neff(q, F, T_end)))
            else:
                Y = solve_number_density(f_a, table, x, background)
                handle.write(axion_grid.format_row(f_a, [Y]))
            handle.flush()

    if args.mode == "fbe":
        np.savetxt(stem + "_dneff.dat", np.array(dneff_rows), delimiter=",",
                   header="f_a [GeV], dNeff", fmt="%.5e")
    return 0


if __name__ == "__main__":
    sys.exit(main())
