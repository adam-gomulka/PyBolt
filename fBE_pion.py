#!/usr/bin/env python3
"""Hot axions from a pi <-> pi pi between T_c = 150 MeV and 30 MeV (arXiv:2211.03799).

Zero axion abundance at T_c ("Pions only"). One rate table serves the whole f_a scan.
"""

import argparse
import os
import sys

import numpy as np

from PyBolt import axion_grid
from PyBolt import boltzmann_solver as bz
from PyBolt.observables import delta_neff, m_a_eV
from PyBolt.pion_amplitudes import M_PI
from PyBolt.pion_rate import PionRateTable
from PyBolt.processes import PionScatteringToAxion

T_START = 0.150  # GeV
T_END = 0.030  # GeV
N_X = 500
M_DM = 1.0e-10  # GeV, massless for production purposes
G_X = 1.0
PARTICLE_TYPE = "b"
DEFAULT_TABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                             "pion_rate_pheno.npz")
SOLVER_OPTIONS = {"method": "LSODA", "rtol": 1e-6, "atol": 1e-24, "lband": 2, "uband": 2}


def x_grid(n_x):
    return np.linspace(M_PI / T_START, M_PI / T_END, n_x)


def _model(x, q):
    model = bz.Model(M_PI, M_DM, G_X, PARTICLE_TYPE)
    model.changeGrid(x, q)
    return model


def solve_distribution(f_a, table, x, q):
    """Final F = q^2 f at T_END."""
    model = _model(x, q)
    model.addCollisionTerm(PionScatteringToAxion(table, f_a).collisionTerm)
    model.solve_fBE(np.zeros(len(q)), dict(SOLVER_OPTIONS))
    F = model.getSolution()[-1, :]
    # solve_fBE logs failures and leaves the solution at zeros instead of raising.
    if not np.any(F):
        raise RuntimeError("solve_fBE did not converge for f_a={:.5e}".format(f_a))
    return F


def solve_number_density(f_a, table, x):
    """Final Y = n/s at T_END."""
    process = PionScatteringToAxion(table, f_a)
    Y = _model(x, np.linspace(0.01, 30.0, 10)).solve_nBE(x, process.rate, 0.0)
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
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    table = PionRateTable.load(args.table)
    x = x_grid(args.n_x)
    q = np.linspace(args.q_min, args.q_max, args.q_num)
    f_vals = axion_grid.f_grid(args.f_min, args.f_max, args.f_num)

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
                F = solve_distribution(f_a, table, x, q)
                handle.write(axion_grid.format_row(f_a, F))
                dneff_rows.append((f_a, m_a_eV(f_a), delta_neff(q, F, T_END)))
            else:
                handle.write(axion_grid.format_row(f_a, [solve_number_density(f_a, table, x)]))
            handle.flush()

    if args.mode == "fbe":
        np.savetxt(stem + "_dneff.dat", np.array(dneff_rows), delimiter=",",
                   header="f_a, m_a[eV], dNeff", fmt="%.5e")
    return 0


if __name__ == "__main__":
    sys.exit(main())
