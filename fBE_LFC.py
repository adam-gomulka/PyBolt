#!/usr/bin/env python3
import argparse
import sys
from dataclasses import dataclass, field

import numpy as np

from PyBolt import axion_grid
from PyBolt import boltzmann_solver as bz
from PyBolt.processes import LeptonAnnihilationToAxionMB, PrimakoffScatteringMB

try:
    from tqdm import tqdm
except ImportError:  # the cluster environment does not ship tqdm
    def tqdm(iterable, **kwargs):
        return iterable

# Grid and model parameters, held fixed so output stays comparable across runs.
N_X = 500
N_Q = 100
X_FIN = 30.0
Q_START = 0.01
Q_END = 15.0
M_DM = 1.0e-10   # GeV, not really relevant
G_X = 1.0        # we default to g_x = 1 as for alps
PARTICLE_TYPE = "b"

SOLVER_OPTIONS = {
    "method": "LSODA",
    "rtol": 1e-6,
    "atol": 1e-24,
    "lband": 2,
    "uband": 2,
}

LEPTONS = {
    "taon": (1.77686, 2.0),
    "tau": (1.77686, 2.0),
    "muon": (0.10565, 2.0),
    "mu": (0.10565, 2.0),
    "electron": (0.000511, 2.0),
    "e": (0.000511, 2.0),
}


def lepton_properties(name):
    """(mass in GeV, internal dof) for a lepton name."""
    try:
        return LEPTONS[name.lower()]
    except KeyError:
        raise ValueError(
            "Unsupported lepton type '{}'. Choose 'tau', 'muon' or 'electron'.".format(
                name
            )
        )


@dataclass
class RunConfig:
    """Everything one solve needs, apart from the value of f_a itself."""

    m_lepton: float
    g_lepton: float
    mDM: float
    g_x: float
    T_reh: float
    x_lin: np.ndarray
    q_lin: np.ndarray
    simplify: bool
    solver_options: dict = field(default_factory=lambda: dict(SOLVER_OPTIONS))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run Boltzmann solver for specific Reheating Ratio."
    )
    parser.add_argument("--lepton", type=str, required=True,
                        help="Lepton type (e.g., muon, electron)")
    parser.add_argument("--ratio", type=float, required=True,
                        help="T_reh_to_mass_ratio: Ratio of Reheating T to Lepton mass")
    parser.add_argument("--output", type=str, required=True,
                        help="Name for the output file")
    parser.add_argument("--f_min", type=float, default=1e7,
                        help="Minimum axion decay constant f_a in GeV (default: 1e7)")
    parser.add_argument("--f_max", type=float, default=1e9,
                        help="Maximum axion decay constant f_a in GeV (default: 1e9)")
    parser.add_argument("--f_num", type=int, default=200,
                        help="Length of axion decay const. grid (default: 200)")
    parser.add_argument("--simplify", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Simplify the collision terms by setting f/f_eq=1 and "
                             "neglecting quantum corrections? (default=False)")
    return parser.parse_args(argv)


def build_config(args):
    """Assemble the run configuration from parsed command-line arguments."""
    m_lepton, g_lepton = lepton_properties(args.lepton)
    T_reh = args.ratio * m_lepton

    return RunConfig(
        m_lepton=m_lepton,
        g_lepton=g_lepton,
        mDM=M_DM,
        g_x=G_X,
        T_reh=T_reh,
        x_lin=np.linspace(m_lepton / T_reh, X_FIN, N_X),
        q_lin=np.linspace(Q_START, Q_END, N_Q),
        simplify=args.simplify,
    )


def solve_distribution(f_a, config):
    """Solve the full Boltzmann equation for one f_a; returns the final f(q)."""
    inv_f = 1.0 / f_a

    model = bz.Model(config.m_lepton, config.mDM, config.g_x, PARTICLE_TYPE)
    model.changeGrid(config.x_lin, config.q_lin)

    annihilation = LeptonAnnihilationToAxionMB(
        config.m_lepton, config.g_lepton, inv_f, config.simplify
    )
    primakoff = PrimakoffScatteringMB(
        config.m_lepton, config.g_lepton, inv_f, config.simplify
    )
    model.addCollisionTerm(annihilation.collisionTerm)
    model.addCollisionTerm(primakoff.collisionTerm)

    model.solve_fBE(np.zeros(len(config.q_lin)), config.solver_options)
    return model.getSolution()[-1, :]


def run_serial(config, f_vals, output):
    """Solve the whole grid into one file, appending as each f value finishes."""
    print("Writing results incrementally to {}...".format(output))

    with open(output, "w") as handle:
        handle.write(axion_grid.HEADER)
        handle.flush()

        for f_a in tqdm(f_vals, desc="Solving fBE"):
            try:
                distribution = solve_distribution(f_a, config)
            except Exception as error:
                print("Error solving for f={:.2e}: {}".format(f_a, error),
                      file=sys.stderr)
                sys.stderr.flush()
                continue
            handle.write(axion_grid.format_row(f_a, distribution))
            handle.flush()


def main(argv=None):
    args = parse_args(argv)
    config = build_config(args)

    print("Starting run for T_reh/m_{} = {} (T_reh = {:.4e} GeV)".format(
        args.lepton.lower(), args.ratio, config.T_reh))
    if config.T_reh < 1e-3:
        print("Warning: reheating temperature below 1 MeV is unreliable in this setup.")

    f_vals = axion_grid.f_grid(args.f_min, args.f_max, args.f_num)
    run_serial(config, f_vals, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
