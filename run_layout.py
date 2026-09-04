#!/usr/bin/env python3
"""The naming grammar for run directories.

A run's directory name is a canonical function of its knobs, computed here and
nowhere else. axion_paths.sh used to rebuild the same name by string
concatenation in bash, which is why --bg reached fBE_LFC.py but no shell script,
and why the tree holds both fa_1000.dat and fa_1000.0.dat.

STDLIB ONLY. submit_axion.sh calls this on the login node before conda activate,
so it must run under whatever python3 is on PATH. Importing PyBolt would pull in
numpy and scipy through PyBolt/__init__.py.
"""


def format_number(value):
    """A canonical, round-trippable, filename-safe rendering of a number.

    Python's repr is shortest-round-trip, so no precision is invented or lost.
    The trailing '.0' is dropped so that 1000 and 1000.0 cannot name two
    different directories for one run.
    """

    number = float(value)
    text = repr(number)

    if text.endswith(".0"):
        text = text[:-2]

    return text


BACKGROUNDS = ("standard", "sudden", "reheating")
MODES = ("fbe", "nbe")

# The tag each non-standard background contributes to a name.
BACKGROUND_TAGS = {"sudden": "sud", "reheating": "reh"}


def run_name(lepton, ratio, bg="standard", t_rh=None, t_max=None,
             simplify=False, tabulate=False, mode="fbe"):
    """The canonical directory name for one run.

    Ordered physics, then method, then mode. Defaults contribute nothing, so a
    plain run of the muon is 'muon_r1000' and stays that way as knobs are added
    to the pipeline.
    """

    if bg not in BACKGROUNDS:
        raise ValueError(
            "bg must be one of {}, got {!r}".format(BACKGROUNDS, bg))
    if mode not in MODES:
        raise ValueError("mode must be one of {}, got {!r}".format(MODES, mode))

    if bg == "standard":
        if t_rh is not None or t_max is not None:
            raise ValueError(
                "t_rh/t_max need bg 'sudden' or 'reheating'; the standard "
                "cosmology ignores them, so a name carrying them would "
                "describe a scenario that was not run"
            )
    elif t_rh is None:
        raise ValueError("bg {!r} requires t_rh".format(bg))

    parts = ["{}_r{}".format(lepton.lower(), format_number(ratio))]

    if bg != "standard":
        parts.append("{}{}".format(BACKGROUND_TAGS[bg], format_number(t_rh)))
        if t_max is not None:
            parts.append("Tmax{}".format(format_number(t_max)))

    if simplify:
        parts.append("simp")
    if tabulate:
        parts.append("tab")
    if mode != "fbe":
        parts.append(mode)

    return "_".join(parts)


# The existing top-level split under distributions/.
CHANNEL_SUBDIRS = {
    "combined": "Combined",
    "primakoff": "Primakoff",
    "annihilation": "Annihilation",
}

# fBE_LFC.py names its output by mode; the directory already carries the mode.
MODE_FILENAMES = {"fbe": "fa.dat", "nbe": "Y.dat"}


def run_args(bg="standard", t_rh=None, t_max=None, simplify=False,
             tabulate=False, mode="fbe", **_ignored):
    """The fBE_LFC.py flags these knobs imply.

    Emitted from the same call as the paths, so the flags handed to the solver
    and the directory it writes into are generated from one argument set and
    cannot describe different runs.

    simplify is always stated explicitly because fBE_LFC.py uses
    BooleanOptionalAction, and the shell has always passed one or the other.
    mode, like bg, is only emitted when it differs from fBE_LFC.py's own
    default ("fbe"), so the common case stays short.
    """

    args = ["--simplify" if simplify else "--no-simplify"]

    if tabulate:
        args.append("--tabulate")

    if bg != "standard":
        args.extend(["--bg", bg, "--t-rh", format_number(t_rh)])
        if t_max is not None:
            args.extend(["--t-max", format_number(t_max)])

    if mode != "fbe":
        args.extend(["--mode", mode])

    return " ".join(args)


def run_paths(channel, study, base="distributions", **knobs):
    """Every path and flag string one run needs, from one argument set."""

    if channel not in CHANNEL_SUBDIRS:
        raise ValueError(
            "channel must be one of {}, got {!r}".format(
                tuple(CHANNEL_SUBDIRS), channel)
        )

    name = run_name(**knobs)
    directory = "/".join([base, CHANNEL_SUBDIRS[channel], study, name])
    filename = MODE_FILENAMES[knobs.get("mode", "fbe")]

    return {
        "RUN_NAME": name,
        "RUN_DIR": directory,
        "PARTS_DIR": directory + "/parts",
        "OUTPUT_PATH": directory + "/" + filename,
        "RUN_ARGS": run_args(**knobs),
    }
