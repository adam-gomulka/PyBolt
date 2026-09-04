#!/usr/bin/env python3
import argparse
import multiprocessing
import os
import sys
import traceback
from dataclasses import dataclass, field

import numpy as np

from PyBolt import axion_grid
from PyBolt import boltzmann_solver as bz
from PyBolt.background import (
    PerturbativeReheating,
    StandardCosmology,
    SuddenDecayReheating,
)
from PyBolt.processes import LeptonAnnihilationToAxionMB, PrimakoffScatteringMB

try:
    from tqdm import tqdm
except ImportError:  # the cluster environment does not ship tqdm
    def tqdm(iterable, **kwargs):
        return iterable

# Grid and model parameters.  The q grid is set from the command line; the x grid
# stays fixed, since x_lin is only solve_ivp's t_eval and its size does not affect
# the result.
N_X = 500
X_FIN = 30.0
DEFAULT_Q_MIN = 1e-2
DEFAULT_Q_MAX = 20.0
DEFAULT_Q_NUM = 250
M_DM = 1.0e-10   # GeV, not really relevant
G_X = 1.0        # we default to g_x = 1 as for alps
PARTICLE_TYPE = "b"
Y0 = 0.0         # the axion population starts empty, as f0 = 0 does for the fBE

MODES = ("fbe", "nbe")
BACKGROUNDS = ("standard", "sudden", "reheating")

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
    T_start: float
    x_lin: np.ndarray
    q_lin: np.ndarray
    simplify: bool
    background: object = field(default_factory=StandardCosmology)
    mode: str = "fbe"
    solver_options: dict = field(default_factory=lambda: dict(SOLVER_OPTIONS))
    tabulate: bool = False
    # Filled in once per task by build_kernel_tables when tabulate is on, then
    # carried to the pool workers with the rest of the config.
    kernel_tables: object = None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run Boltzmann solver for specific Reheating Ratio."
    )
    parser.add_argument("--lepton", type=str, required=True,
                        help="Lepton type (e.g., muon, electron)")
    parser.add_argument("--ratio", type=float, required=True,
                        help="Ratio of the run's starting temperature to the lepton "
                             "mass. Named --ratio for backwards compatibility; it "
                             "sets T_start, not the reheat temperature (--t-rh).")
    parser.add_argument("--bg", type=str, choices=BACKGROUNDS, default="standard",
                        help="Expansion history. standard: radiation domination "
                             "with conserved entropy. sudden: piecewise-analytic "
                             "reheating, hard switch at --t-rh. reheating: the same "
                             "scenario integrated as a two-fluid system. "
                             "(default: standard)")
    parser.add_argument("--t-rh", type=float, default=None,
                        help="Reheat temperature in GeV, required by --bg sudden "
                             "and --bg reheating. Must be below T_start.")
    parser.add_argument("--t-max", type=float, default=None,
                        help="Highest temperature the bath ever reached, in GeV. Set "
                             "by the initial inflaton density, so it is an input in "
                             "its own right rather than something --t-rh fixes. The "
                             "run cannot start above it. Omit to assume the reheating "
                             "attractor extends as high as the run begins.")
    parser.add_argument("--output", type=str, default=None,
                        help="Name for the combined output file (serial mode)")
    parser.add_argument("--f_min", type=float, default=1e7,
                        help="Minimum axion decay constant f_a in GeV (default: 1e7)")
    parser.add_argument("--f_max", type=float, default=1e9,
                        help="Maximum axion decay constant f_a in GeV (default: 1e9)")
    parser.add_argument("--f_num", type=int, default=100,
                        help="Length of axion decay const. grid (default: 100)")
    parser.add_argument("--q_min", type=float, default=DEFAULT_Q_MIN,
                        help="Lowest q = p/T on the momentum grid (default: 1e-2)")
    parser.add_argument("--q_max", type=float, default=DEFAULT_Q_MAX,
                        help="Highest q = p/T on the momentum grid (default: 20)")
    parser.add_argument("--q_num", type=int, default=DEFAULT_Q_NUM,
                        help="Number of momentum grid points (default: 250). The "
                             "grid is linear: the derivative stencil in "
                             "boltzmann_solver assumes uniform spacing.")
    parser.add_argument("--simplify", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Simplify the collision terms by setting f/f_eq=1 and "
                             "neglecting quantum corrections? (default=False)")
    parser.add_argument("--mode", type=str, choices=MODES, default="fbe",
                        help="fbe: solve the phase-space equation and store the "
                             "final f(q). nbe: solve the number-density equation "
                             "and store the final Y. (default: fbe)")
    parser.add_argument("--parts-dir", type=str, default=None,
                        help="Write one part file per f index into this directory "
                             "instead of one combined output file")
    parser.add_argument("--f-index-start", type=int, default=0,
                        help="First f grid index this task solves (default: 0)")
    parser.add_argument("--f-count", type=int, default=None,
                        help="Number of f grid indices this task solves "
                             "(default: all remaining)")
    parser.add_argument("--nproc", type=int, default=1,
                        help="Worker processes within this task (default: 1)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute indices whose part file already exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip the solver and emit a zero row of the correct "
                             "width, to exercise the pipeline cheaply")
    parser.add_argument("--tabulate", action="store_true",
                        help="Precompute the collision kernels on an x grid once "
                             "and interpolate, instead of integrating adaptively "
                             "at every solver step. One table serves the whole f "
                             "scan. Changes results at the 1e-5 level, so it is "
                             "recorded in meta.json and tabulated parts cannot be "
                             "mixed with adaptive ones. (fbe mode only)")
    args = parser.parse_args(argv)

    # The nbe path goes through rate(), which never touches the kernel, so a table
    # there would cost minutes to build and then go unread.
    if args.tabulate and args.mode != "fbe":
        parser.error("--tabulate applies to --mode fbe only; the number-density "
                     "solver does not evaluate the collision kernel")

    return args


def build_background(args, T_start):
    """The expansion history this run solves against.

    Both reheating tiers take the same T_rh and therefore the same Gamma, so the
    cheap and the numerical background can be compared directly.
    """
    if args.bg == "standard":
        return StandardCosmology()

    if args.t_rh is None:
        raise ValueError("--bg {} requires --t-rh".format(args.bg))
    if args.t_rh >= T_start:
        raise ValueError(
            "--t-rh ({:.4e} GeV) must be below the starting temperature "
            "({:.4e} GeV), otherwise the run begins after reheating has already "
            "finished and the background is just the standard one.".format(
                args.t_rh, T_start)
        )

    # T_max bounds where any run may begin, whichever tier models the era, so the
    # check belongs here rather than inside one background.
    if args.t_max is not None and args.t_max < T_start:
        raise ValueError(
            "the run starts at {:.4e} GeV, above --t-max ({:.4e} GeV): the universe "
            "never reached that temperature in this scenario. Lower --ratio or raise "
            "--t-max.".format(T_start, args.t_max)
        )

    if args.bg == "sudden":
        return SuddenDecayReheating(T_rh=args.t_rh)
    return PerturbativeReheating(T_rh=args.t_rh, T_start=T_start, T_max=args.t_max)


def build_config(args):
    """Assemble the run configuration from parsed command-line arguments."""
    m_lepton, g_lepton = lepton_properties(args.lepton)
    T_start = args.ratio * m_lepton

    return RunConfig(
        m_lepton=m_lepton,
        g_lepton=g_lepton,
        mDM=M_DM,
        g_x=G_X,
        T_start=T_start,
        x_lin=np.linspace(m_lepton / T_start, X_FIN, N_X),
        q_lin=np.linspace(args.q_min, args.q_max, args.q_num),
        simplify=args.simplify,
        background=build_background(args, T_start),
        mode=args.mode,
        tabulate=args.tabulate,
    )


def make_processes(config, inv_f):
    """The two collision processes for one f_a, sharing the run's kernel tables.

    Nothing under the kernel integral depends on the coupling, so the tables built
    once by build_kernel_tables are valid for every f_a in the scan.
    """
    annihilation = LeptonAnnihilationToAxionMB(
        config.m_lepton, config.g_lepton, inv_f, config.simplify
    )
    primakoff = PrimakoffScatteringMB(
        config.m_lepton, config.g_lepton, inv_f, config.simplify
    )

    if config.kernel_tables is not None:
        annihilation.adopt_table(config.kernel_tables[0])
        primakoff.adopt_table(config.kernel_tables[1])

    return annihilation, primakoff


def build_kernel_tables(config):
    """Tabulate both collision kernels once for the whole f_a scan.

    This is the expensive step -- one x node costs one full adaptive evaluation --
    and it is the reason tabulation pays: doing it per f_a would cost more than
    solving adaptively, while doing it once amortises over the entire grid.

    The donor coupling is arbitrary, since it sits outside the integral.
    """
    donors = (
        LeptonAnnihilationToAxionMB(config.m_lepton, config.g_lepton, 1.0, config.simplify),
        PrimakoffScatteringMB(config.m_lepton, config.g_lepton, 1.0, config.simplify),
    )

    for donor in donors:
        donor.tabulate(config.x_lin, config.q_lin)

    return donors


def solve_distribution(f_a, config):
    """Solve the full Boltzmann equation for one f_a; returns the final f(q)."""
    inv_f = 1.0 / f_a

    model = bz.Model(
        config.m_lepton, config.mDM, config.g_x, PARTICLE_TYPE,
        background=config.background,
    )
    model.changeGrid(config.x_lin, config.q_lin)

    annihilation, primakoff = make_processes(config, inv_f)
    model.addCollisionTerm(annihilation.collisionTerm)
    model.addCollisionTerm(primakoff.collisionTerm)

    model.solve_fBE(np.zeros(len(config.q_lin)), config.solver_options)
    return model.getSolution()[-1, :]


def solve_number_density(f_a, config):
    """Solve the number-density Boltzmann equation for one f_a.

    Returns a one-element array holding the final comoving abundance Y, so that a
    row is written exactly like an fBE row -- "f_a,Y" instead of "f_a,f(q)...".
    """
    inv_f = 1.0 / f_a

    model = bz.Model(
        config.m_lepton, config.mDM, config.g_x, PARTICLE_TYPE,
        background=config.background,
    )
    model.changeGrid(config.x_lin, config.q_lin)

    annihilation, primakoff = make_processes(config, inv_f)

    def total_rate(x, Y):
        return annihilation.rate(x, Y) + primakoff.rate(x, Y)

    abundance = model.solve_nBE(config.x_lin, total_rate, Y0)

    # solve_nBE returns None instead of raising when the integration fails; turn
    # that into an exception so the caller writes a .fail marker like any other
    # failure rather than silently storing a bad row.
    if abundance is None:
        raise RuntimeError("solve_nBE did not converge for f_a={:.5e}".format(f_a))

    return np.array([abundance[-1]])


def solve_for_mode(f_a, config):
    """Dispatch to the solver this run's mode asks for."""
    if config.mode == "nbe":
        return solve_number_density(f_a, config)
    return solve_distribution(f_a, config)


def expected_columns(config):
    """Number of comma-separated fields in one data row, for this run's mode.

    nbe stores a single scalar per f value, fbe the whole final f(q).
    """
    if config.mode == "nbe":
        return 2
    return 1 + len(config.q_lin)


def header_for(config):
    """The header line the merged file of this mode should carry."""
    if config.mode == "nbe":
        return axion_grid.HEADER_NUMBER_DENSITY
    return axion_grid.HEADER


def run_serial(config, f_vals, output):
    """Solve the whole grid into one file, appending as each f value finishes."""
    print("Writing results incrementally to {}...".format(output))


    with open(output, "w") as handle:
        handle.write(header_for(config))
        if config.mode == "fbe":
            handle.write(axion_grid.format_q_header(config.q_lin))
        handle.flush()

        for f_a in tqdm(f_vals, desc="Solving " + config.mode):
            try:
                distribution = solve_for_mode(f_a, config)
            except Exception as error:
                print("Error solving for f={:.2e}: {}".format(f_a, error),
                      file=sys.stderr)
                sys.stderr.flush()
                continue
            handle.write(axion_grid.format_row(f_a, distribution))
            handle.flush()


def part_is_valid(path, expected_width):
    """True if ``path`` already holds one usable row of the expected width."""
    if not os.path.exists(path):
        return False
    try:
        line = axion_grid.read_row(path)
    except (OSError, ValueError):
        return False
    return axion_grid.row_width(line) == expected_width


def process_index(index, f_a, parts_dir, config, overwrite=False, dry_run=False,
                  solver=None):
    """Solve one grid index into its part file.

    Returns "skipped", "solved" or "failed".  A solver failure writes a .fail
    marker and is reported, never raised -- the array task must survive it and
    let the merge be the gate.
    """
    part_path = os.path.join(parts_dir, axion_grid.part_filename(index))
    fail_path = os.path.join(parts_dir, axion_grid.fail_filename(index))
    expected_width = expected_columns(config)

    if not overwrite and part_is_valid(part_path, expected_width):
        return "skipped"

    try:
        if dry_run:
            distribution = np.zeros(expected_width - 1)
        else:
            distribution = (solver or solve_for_mode)(f_a, config)
        axion_grid.atomic_write_text(
            part_path, axion_grid.format_row(f_a, distribution)
        )
    except Exception:
        axion_grid.atomic_write_text(
            fail_path,
            "index = {}\nf_a = {:.5e}\n\n{}".format(
                index, f_a, traceback.format_exc()
            ),
        )
        print("index {} (f_a={:.5e}) failed, see {}".format(index, f_a, fail_path),
              file=sys.stderr)
        sys.stderr.flush()
        return "failed"

    # A retry that succeeds must clear the old marker, or the merge report lies.
    if os.path.exists(fail_path):
        os.remove(fail_path)
    return "solved"


def pool_context():
    """A multiprocessing context that is safe when the parent has threads.

    The Linux default is fork, which can deadlock a child that inherits a lock
    held by another thread of the parent -- and the parent here has imported
    numpy and scipy.  forkserver is cheap and safe; spawn is the portable
    fallback.  Either costs one process start per worker, which is nothing
    beside a solve.
    """
    available = multiprocessing.get_all_start_methods()
    for method in ("forkserver", "spawn"):
        if method in available:
            return multiprocessing.get_context(method)
    return multiprocessing.get_context()


def _pool_worker(payload):
    """Pickled entry point for the multiprocessing pool (must be module level)."""
    index, f_a, parts_dir, config, overwrite, dry_run = payload
    return process_index(
        index, f_a, parts_dir, config, overwrite=overwrite, dry_run=dry_run
    )


def select_indices(f_vals, start, count):
    """The grid indices this task is responsible for, clipped to the grid.

    Clipping here is what lets the last array task need no special case when
    f_num is not a multiple of the chunk size.
    """
    begin = min(start, len(f_vals))
    stop = len(f_vals) if count is None else min(begin + count, len(f_vals))
    return range(begin, stop)


def build_meta(args, config):
    """The run metadata written next to the part files.

    The background keys are added only when the background is not the standard one.
    check_meta_matches compares every key, so writing them unconditionally would make
    every parts directory created before this feature look like a parameter mismatch
    and refuse to resume.

    The "T_reh" key keeps its name for the same reason, even though it holds the
    run's starting temperature rather than a reheat temperature. The reheat
    temperature is "background_T_rh"; the two are deliberately not spelled alike.
    """
    meta = {
        "mode": config.mode,
        "n_columns": expected_columns(config),
        "header": header_for(config),
        "lepton": args.lepton.lower(),
        "m_lepton": config.m_lepton,
        "g_lepton": config.g_lepton,
        "ratio": args.ratio,
        "T_reh": config.T_start,
        "mDM": config.mDM,
        "g_x": config.g_x,
        "particle_type": PARTICLE_TYPE,
        "simplify": bool(config.simplify),
        "f_min": args.f_min,
        "f_max": args.f_max,
        "f_num": args.f_num,
        "N_x": len(config.x_lin),
        "x_start": float(config.x_lin[0]),
        "x_fin": float(config.x_lin[-1]),
        "x_spacing": "linear",
        "N_q": len(config.q_lin),
        "q_start": float(config.q_lin[0]),
        "q_end": float(config.q_lin[-1]),
        "q_spacing": "linear",
        "solver_options": dict(config.solver_options),
    }

    if args.bg != "standard":
        meta["background"] = args.bg
        meta["background_T_rh"] = args.t_rh
        if args.t_max is not None:
            meta["background_T_max"] = args.t_max

    # Written only when on, for the same reason the background keys are: every
    # parts directory created before this feature existed has no such key, and
    # check_meta_matches compares the full key set. Recording it at all matters
    # because tabulation moves the numbers at the 1e-5 level, so tabulated and
    # adaptive parts must never merge into one grid.
    if config.tabulate:
        meta["tabulate"] = True

    return meta


class ParameterMismatch(Exception):
    """The parts directory was built by a run with different parameters."""


def check_meta_matches(parts_dir, meta):
    """Refuse to add parts to a directory built with other parameters.

    ``meta.json`` is written once and never updated, so without this a rerun with
    a changed f range or q grid would leave stale metadata describing the old run.
    The merge would still refuse to write -- the gate holds -- but it would
    complain about row widths or f_a mismatches, which points at the symptom
    rather than the cause.
    """
    try:
        stored = axion_grid.read_meta(parts_dir)
    except FileNotFoundError:
        return

    differences = [
        key for key in sorted(set(stored) | set(meta))
        if stored.get(key) != meta.get(key)
    ]
    if not differences:
        return

    lines = [
        "parts directory {} was built with different parameters:".format(parts_dir)
    ]
    for key in differences:
        lines.append("  {}: stored {!r}, requested {!r}".format(
            key, stored.get(key), meta.get(key)))
    lines.append("")
    lines.append("Use a different --output, or delete the parts directory to start")
    lines.append("over. Reusing it would mix results from two different runs.")
    raise ParameterMismatch("\n".join(lines))


def run_parts(args, config, f_vals):
    """Solve this task's slice into ``args.parts_dir``. Returns outcome counts."""
    os.makedirs(args.parts_dir, exist_ok=True)
    meta = build_meta(args, config)
    check_meta_matches(args.parts_dir, meta)
    axion_grid.write_meta_if_absent(args.parts_dir, meta)

    # Only fbe output has a q axis; an nbe row is a single Y.
    if config.mode == "fbe":
        axion_grid.write_q_grid_if_absent(args.parts_dir, config.q_lin)

    index_list = list(select_indices(f_vals, args.f_index_start, args.f_count))

    # Build the kernel tables once for this task's whole slice, before the pool
    # forks, so every worker inherits them instead of rebuilding them per f_a.
    # Skipped when there is nothing to solve, so an already-finished array task
    # still exits promptly.
    if config.tabulate and index_list and not args.dry_run:
        print("tabulating collision kernels on {} x nodes...".format(
            len(config.x_lin)))
        config.kernel_tables = build_kernel_tables(config)

    if index_list:
        print("task solving indices {}..{} of {}".format(
            index_list[0], index_list[-1], args.f_num))
    else:
        print("task has no indices to solve (start {} is past the grid of {})".format(
            args.f_index_start, args.f_num))

    counts = {"solved": 0, "skipped": 0, "failed": 0}

    if args.nproc > 1:
        payloads = [
            (index, f_vals[index], args.parts_dir, config, args.overwrite,
             args.dry_run)
            for index in index_list
        ]
        with pool_context().Pool(args.nproc) as pool:
            for outcome in pool.imap_unordered(_pool_worker, payloads):
                counts[outcome] += 1
    else:
        for index in tqdm(index_list, desc="Solving fBE"):
            outcome = process_index(
                index, f_vals[index], args.parts_dir, config,
                overwrite=args.overwrite, dry_run=args.dry_run,
            )
            counts[outcome] += 1

    print("solved={solved} skipped={skipped} failed={failed}".format(**counts))
    return counts


def main(argv=None):
    args = parse_args(argv)
    config = build_config(args)

    print("Starting run for T_start/m_{} = {} (T_start = {:.4e} GeV)".format(
        args.lepton.lower(), args.ratio, config.T_start))
    if config.T_start < 1e-3:
        print("Warning: starting temperature below 1 MeV is unreliable in this setup.")
    if args.bg != "standard":
        print("Background: {} (T_rh = {:.4e} GeV, Gamma = {:.4e} GeV)".format(
            args.bg, args.t_rh, config.background.Gamma))

    if (args.parts_dir is None) == (args.output is None):
        print("ERROR: give exactly one of --output (serial) or --parts-dir (parallel)",
              file=sys.stderr)
        return 2

    f_vals = axion_grid.f_grid(args.f_min, args.f_max, args.f_num)

    if args.parts_dir is not None:
        try:
            run_parts(args, config, f_vals)
        except ParameterMismatch as error:
            # A task-level error, not a per-index one: exit non-zero.
            print("ERROR: " + str(error), file=sys.stderr)
            return 2
    else:
        run_serial(config, f_vals, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
