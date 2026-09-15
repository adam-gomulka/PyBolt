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

# The q-grid defaults live in run_layout because it has to leave a default grid
# out of a run's name. It is stdlib-only, so importing it here costs nothing.
from run_layout import DEFAULT_Q_MAX, DEFAULT_Q_MIN, DEFAULT_Q_NUM

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
M_DM = 1.0e-10   # GeV, not really relevant
G_X = 1.0        # we default to g_x = 1 as for alps
PARTICLE_TYPE = "b"
Y0 = 0.0         # the axion population starts empty, as f0 = 0 does for the fBE

MODES = ("fbe", "nbe")
BACKGROUNDS = ("standard", "sudden", "reheating")

# Which collision processes each channel switches on. The two single-process
# channels used to be their own scripts; they are the same solver with one term
# left out, so they live here and inherit parts mode, the backgrounds,
# tabulation and nbe rather than each needing its own copy.
CHANNEL_PROCESSES = {
    "combined": (LeptonAnnihilationToAxionMB, PrimakoffScatteringMB),
    "annihilation": (LeptonAnnihilationToAxionMB,),
    "primakoff": (PrimakoffScatteringMB,),
}

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
    channel: str = "combined"
    background: object = field(default_factory=StandardCosmology)
    mode: str = "fbe"
    solver_options: dict = field(default_factory=lambda: dict(SOLVER_OPTIONS))
    tabulate: bool = False
    # Set once per task by build_kernel_tables when tabulate is on.
    kernel_tables: object = None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run Boltzmann solver for specific Reheating Ratio."
    )
    parser.add_argument("--lepton", type=str, required=True,
                        help="Lepton type (e.g., muon, electron)")
    parser.add_argument("--t-start-ratio", "--ratio", dest="ratio", type=float,
                        required=True,
                        help="The run's starting temperature in units of the lepton "
                             "mass: T_start = ratio * m_lepton. This is where the "
                             "integration begins, not the reheat temperature, which "
                             "is --t-rh-ratio. Under --bg standard the two coincide, since "
                             "the universe starts at reheating by assumption. "
                             "--ratio is kept as an alias.")
    parser.add_argument("--channel", type=str, choices=sorted(CHANNEL_PROCESSES),
                        default="combined",
                        help="Which production processes to include. combined "
                             "(default) is lepton annihilation plus Primakoff "
                             "scattering; annihilation and primakoff switch one "
                             "off. The channel picks the top-level output "
                             "directory, so the three never share a run.")
    parser.add_argument("--bg", type=str, choices=BACKGROUNDS, default="standard",
                        help="Expansion history. standard: radiation domination "
                             "with conserved entropy. sudden: piecewise-analytic "
                             "reheating, hard switch at --t-rh-ratio. reheating: the same "
                             "scenario integrated as a two-fluid system. "
                             "(default: standard)")
    parser.add_argument("--t-rh-ratio", dest="t_rh", type=float, default=None,
                        help="Reheat temperature in units of the lepton mass, "
                             "required by --bg sudden and --bg reheating. Same "
                             "units as --t-start-ratio, so it simply has to be "
                             "the smaller of the two.")
    parser.add_argument("--t-max-ratio", dest="t_max", type=float, default=None,
                        help="Highest temperature the bath ever reached, in units "
                             "of the lepton mass. Set by the initial inflaton "
                             "density, so it is an input in its own right rather "
                             "than something --t-rh-ratio fixes. The run cannot "
                             "start above it. Omit to assume the reheating "
                             "attractor extends as high as the run begins.")
    # These took GeV. Renaming rather than aliasing is deliberate: an alias would
    # reinterpret every existing command as a temperature ~10x smaller for the
    # muon, with no error and quietly different physics.
    parser.add_argument("--t-rh", dest="_t_rh_gev", type=float, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--t-max", dest="_t_max_gev", type=float, default=None,
                        help=argparse.SUPPRESS)
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
                        help="Drop back-reaction (f/f_eq -> 0, pure production) and "
                             "treat the integrated-over bath particle as "
                             "Maxwell-Boltzmann instead of Bose-Einstein (photon, "
                             "annihilation) or Fermi-Dirac (lepton, Primakoff). "
                             "fbe mode only; nbe ignores it. (default: False)")
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

    for old, new, value in (("--t-rh", "--t-rh-ratio", args._t_rh_gev),
                            ("--t-max", "--t-max-ratio", args._t_max_gev)):
        if value is not None:
            m_lepton = lepton_properties(args.lepton)[0]
            parser.error(
                "{0} took GeV and no longer exists; use {1}, which is in units of "
                "the lepton mass like --t-start-ratio. For the {2}, {0} {3:g} is "
                "{1} {4:g}.".format(old, new, args.lepton.lower(), value,
                                    value / m_lepton)
            )

    # The nbe path goes through rate(), which never touches the kernel.
    if args.tabulate and args.mode != "fbe":
        parser.error("--tabulate applies to --mode fbe only; the number-density "
                     "solver does not evaluate the collision kernel")

    return args


def build_background(args, m_lepton, T_start):
    """The expansion history this run solves against.

    Both reheating tiers take the same T_rh, and therefore the same Gamma.

    The command line states every temperature in units of the lepton mass, so the
    comparisons below are between plain ratios and the conversion to GeV happens
    once, here, where the physics classes need it.
    """
    if args.bg == "standard":
        return StandardCosmology()

    if args.t_rh is None:
        raise ValueError("--bg {} requires --t-rh-ratio".format(args.bg))
    if args.t_rh >= args.ratio:
        raise ValueError(
            "--t-rh-ratio ({:g}) must be below --t-start-ratio ({:g}), otherwise "
            "the run begins after reheating has already finished and the "
            "background is just the standard one.".format(args.t_rh, args.ratio)
        )

    # T_max bounds where a run may begin under either reheating tier.
    if args.t_max is not None and args.t_max < args.ratio:
        raise ValueError(
            "the run starts at --t-start-ratio {:g}, above --t-max-ratio ({:g}): "
            "the universe never reached that temperature in this scenario. Lower "
            "--t-start-ratio or raise --t-max-ratio.".format(
                args.ratio, args.t_max)
        )

    T_rh = args.t_rh * m_lepton
    T_max = None if args.t_max is None else args.t_max * m_lepton

    if args.bg == "sudden":
        return SuddenDecayReheating(T_rh=T_rh)
    return PerturbativeReheating(T_rh=T_rh, T_start=T_start, T_max=T_max)


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
        channel=args.channel,
        background=build_background(args, m_lepton, T_start),
        mode=args.mode,
        tabulate=args.tabulate,
    )


def make_processes(config, inv_f):
    """This channel's collision processes for one f_a, sharing the kernel tables.

    Returns a tuple parallel to CHANNEL_PROCESSES[config.channel], so it lines up
    with the tables build_kernel_tables produced for the same channel.
    """
    processes = tuple(
        cls(config.m_lepton, config.g_lepton, inv_f, config.simplify)
        for cls in CHANNEL_PROCESSES[config.channel]
    )

    if config.kernel_tables is not None:
        for process, table in zip(processes, config.kernel_tables):
            process.adopt_table(table)

    return processes


def build_kernel_tables(config):
    """Tabulate this channel's collision kernels once for the whole f_a scan.

    One x node costs one full adaptive evaluation, so this is the expensive step.
    The donor coupling is arbitrary, since it sits outside the integral. A
    single-process channel tabulates only its own kernel, so it pays half.
    """
    donors = tuple(
        cls(config.m_lepton, config.g_lepton, 1.0, config.simplify)
        for cls in CHANNEL_PROCESSES[config.channel]
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

    for process in make_processes(config, inv_f):
        model.addCollisionTerm(process.collisionTerm)

    model.solve_fBE(np.zeros(len(config.q_lin)), config.solver_options)
    return model.getSolution()[-1, :]


def solve_number_density(f_a, config):
    """Solve the number-density Boltzmann equation for one f_a.

    Returns a one-element array holding the final comoving abundance Y, so a row is
    written like an fBE row: "f_a,Y" instead of "f_a,f(q)...".
    """
    inv_f = 1.0 / f_a

    model = bz.Model(
        config.m_lepton, config.mDM, config.g_x, PARTICLE_TYPE,
        background=config.background,
    )
    model.changeGrid(config.x_lin, config.q_lin)

    processes = make_processes(config, inv_f)

    def total_rate(x, Y):
        return sum(process.rate(x, Y) for process in processes)

    abundance = model.solve_nBE(config.x_lin, total_rate, Y0)

    # solve_nBE returns None instead of raising when the integration fails; the
    # caller only writes a .fail marker for an exception.
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
    marker and is reported, never raised: the array task must survive it, and the
    merge is the gate.
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
    numpy and scipy.  forkserver is safe; spawn is the portable fallback.  Either
    costs one process start per worker, which is nothing beside a solve.
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

    Clipping is what makes the last array task need no special case when f_num is
    not a multiple of the chunk size.
    """
    begin = min(start, len(f_vals))
    stop = len(f_vals) if count is None else min(begin + count, len(f_vals))
    return range(begin, stop)


def build_meta(args, config):
    """The run metadata written next to the part files.

    Optional keys (the background ones, and "tabulate") are written only when they
    are in use. check_meta_matches compares the full key set, so writing them
    unconditionally would make every parts directory predating them look like a
    parameter mismatch and refuse to resume.

    "T_start" is where the integration begins, in GeV. Runs written before the
    backgrounds existed spell it "T_reh"; check_meta_matches translates that so
    they still resume.

    The reheat temperature is recorded twice, as the ratio that was typed and as
    the GeV it works out to. Runs predating the unit change stored GeV under
    "background_T_rh", which no longer appears, so they refuse to resume instead
    of being reread in the wrong units.
    """
    meta = {
        "mode": config.mode,
        "n_columns": expected_columns(config),
        "header": header_for(config),
        "lepton": args.lepton.lower(),
        "m_lepton": config.m_lepton,
        "g_lepton": config.g_lepton,
        "ratio": args.ratio,
        "T_start": config.T_start,
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

    # Written only when it is not the default, like the background keys: every
    # parts directory predating the channel selector is a combined run, and an
    # unconditional key would make all of them read as a mismatch.
    if args.channel != "combined":
        meta["channel"] = args.channel

    if args.bg != "standard":
        meta["background"] = args.bg
        # Both the input and the temperature it works out to. The "_ratio" names
        # are new: runs predating the unit change stored GeV under
        # "background_T_rh", so the key set differs and check_meta_matches refuses
        # to resume into one rather than reading 0.1 GeV as 0.1 lepton masses.
        meta["background_T_rh_ratio"] = args.t_rh
        meta["background_T_rh_GeV"] = args.t_rh * config.m_lepton
        if args.t_max is not None:
            meta["background_T_max_ratio"] = args.t_max
            meta["background_T_max_GeV"] = args.t_max * config.m_lepton

    # Recorded because tabulation moves the numbers at the 1e-5 level: tabulated and
    # adaptive parts must never merge into one grid.
    if config.tabulate:
        meta["tabulate"] = True

    return meta


class ParameterMismatch(Exception):
    """The parts directory was built by a run with different parameters."""


def check_meta_matches(parts_dir, meta):
    """Refuse to add parts to a directory built with other parameters.

    ``meta.json`` is written once and never updated, so without this a rerun with a
    changed f range or q grid would leave stale metadata describing the old run.
    """
    try:
        stored = axion_grid.read_meta(parts_dir)
    except FileNotFoundError:
        return

    # "T_reh" was this key's name before the background models made a reheat
    # temperature a separate thing. Translating it here means a parts directory
    # written under the old name still resumes instead of reading as a mismatch.
    if "T_reh" in stored and "T_start" not in stored:
        stored = dict(stored)
        stored["T_start"] = stored.pop("T_reh")

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

    # Built before the pool forks, so every worker inherits the tables instead of
    # rebuilding them per f_a.
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
        print("Background: {} (T_rh/m_{} = {:g}, T_rh = {:.4e} GeV, "
              "Gamma = {:.4e} GeV)".format(
                  args.bg, args.lepton.lower(), args.t_rh,
                  args.t_rh * config.m_lepton, config.background.Gamma))

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
