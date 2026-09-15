#!/usr/bin/env python3
"""Canonical run-directory names and paths, derived from a run's parameters.

Standard library only: it must run before the conda environment is activated.
"""


def format_number(value):
    """A canonical, filename-safe rendering of a number (1000.0 -> '1000')."""

    number = float(value)
    text = repr(number)

    if text.endswith(".0"):
        text = text[:-2]

    return text


BACKGROUNDS = ("standard", "sudden", "reheating")
MODES = ("fbe", "nbe")

# Default momentum grid, shared with fBE_LFC.py.
DEFAULT_Q_MIN = 1e-2
DEFAULT_Q_MAX = 20.0
DEFAULT_Q_NUM = 250

# The tag each non-standard background contributes to a name.
BACKGROUND_TAGS = {"sudden": "sud", "reheating": "reh"}


def _q_overrides(q_min=None, q_max=None, q_num=None):
    """The momentum-grid settings that differ from the defaults.

    Returns (name, rendered value) pairs, e.g. [("min", "0.05"), ("num", "400")].
    """

    overrides = []

    for name, value, default in (
        ("min", q_min, DEFAULT_Q_MIN),
        ("max", q_max, DEFAULT_Q_MAX),
        ("num", q_num, DEFAULT_Q_NUM),
    ):
        if value is None or float(value) == float(default):
            continue
        overrides.append((name, format_number(value)))

    return overrides


def run_name(lepton, ratio, bg="standard", t_rh=None, t_max=None,
             q_min=None, q_max=None, q_num=None,
             simplify=False, tabulate=False, mode="fbe"):
    """The canonical directory name for one run.

    Ordered physics, then method, then mode. Defaults contribute nothing, so a
    plain run of the muon is 'muon_r1000'.
    """

    # Errors name command-line flags because they are shown to shell users.
    if bg not in BACKGROUNDS:
        raise ValueError(
            "--bg must be one of {}, got {!r}".format(
                ", ".join(BACKGROUNDS), bg))
    if mode not in MODES:
        raise ValueError(
            "--mode must be one of {}, got {!r}".format(", ".join(MODES), mode))

    if tabulate and mode != "fbe":
        raise ValueError(
            "--tabulate applies to --mode fbe only, got '--mode {}'; it caches "
            "the collision kernel, and the number-density path evaluates rate() "
            "instead, so the table would be built and never read".format(mode)
        )

    if bg == "standard":
        if t_rh is not None or t_max is not None:
            raise ValueError(
                "--t-rh-ratio/--t-max-ratio need --bg sudden or --bg reheating; "
                "the standard cosmology ignores them, so a name carrying them "
                "would describe a scenario that was not run"
            )
    elif t_rh is None:
        raise ValueError(
            "--bg {} requires --t-rh-ratio (reheat temperature in units of "
            "the lepton mass)".format(bg))

    parts = ["{}_r{}".format(lepton.lower(), format_number(ratio))]

    if bg != "standard":
        parts.append("{}{}".format(BACKGROUND_TAGS[bg], format_number(t_rh)))
        if t_max is not None:
            parts.append("Tmax{}".format(format_number(t_max)))

    parts.extend(
        "q{}{}".format(name, value)
        for name, value in _q_overrides(q_min, q_max, q_num)
    )

    if simplify:
        parts.append("simp")
    if tabulate:
        parts.append("tab")
    if mode != "fbe":
        parts.append(mode)

    return "_".join(parts)


# Top-level directory under distributions/ for each channel.
CHANNEL_SUBDIRS = {
    "combined": "Combined",
    "primakoff": "Primakoff",
    "annihilation": "Annihilation",
}

MODE_FILENAMES = {"fbe": "fa.dat", "nbe": "Y.dat"}


def run_args(channel="combined", bg="standard", t_rh=None, t_max=None,
             q_min=None, q_max=None, q_num=None, simplify=False,
             tabulate=False, mode="fbe", **_ignored):
    """The fBE_LFC.py flags these knobs imply.

    --simplify/--no-simplify is always given; other flags only when not default.
    """

    args = ["--simplify" if simplify else "--no-simplify"]

    if channel != "combined":
        args.extend(["--channel", channel])

    if tabulate:
        args.append("--tabulate")

    if bg != "standard":
        args.extend(["--bg", bg, "--t-rh-ratio", format_number(t_rh)])
        if t_max is not None:
            args.extend(["--t-max-ratio", format_number(t_max)])

    for name, value in _q_overrides(q_min, q_max, q_num):
        args.extend(["--q_{}".format(name), value])

    if mode != "fbe":
        args.extend(["--mode", mode])

    return " ".join(args)


def run_paths(channel, study, base="distributions", **knobs):
    """Every path and flag string one run needs, from one argument set."""

    if channel not in CHANNEL_SUBDIRS:
        raise ValueError(
            "--channel must be one of {}, got {!r}".format(
                ", ".join(sorted(CHANNEL_SUBDIRS)), channel)
        )

    name = run_name(**knobs)
    directory = "/".join([base, CHANNEL_SUBDIRS[channel], study, name])
    filename = MODE_FILENAMES[knobs.get("mode", "fbe")]

    return {
        "RUN_NAME": name,
        "RUN_DIR": directory,
        "PARTS_DIR": directory + "/parts",
        "OUTPUT_PATH": directory + "/" + filename,
        "RUN_ARGS": run_args(channel=channel, **knobs),
    }


def _shell_quote(value):
    """Single-quote a value for safe eval by bash."""

    return "'" + str(value).replace("'", "'\\''") + "'"


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Print the canonical paths and solver flags for one run."
    )
    parser.add_argument("--shell", action="store_true",
                        help="Print assignments for the shell to eval")
    parser.add_argument("--channel", default="combined",
                        choices=sorted(CHANNEL_SUBDIRS))
    parser.add_argument("--study", required=True)
    parser.add_argument("--base", default="distributions")
    parser.add_argument("--lepton", required=True)
    parser.add_argument("--ratio", required=True)
    parser.add_argument("--bg", default="standard")
    parser.add_argument("--t-rh-ratio", dest="t_rh", default=None)
    parser.add_argument("--t-max-ratio", dest="t_max", default=None)
    parser.add_argument("--q-min", default=None)
    parser.add_argument("--q-max", default=None)
    parser.add_argument("--q-num", default=None)
    parser.add_argument("--simplify", action="store_true")
    parser.add_argument("--tabulate", action="store_true")
    parser.add_argument("--mode", default="fbe")
    args = parser.parse_args(argv)

    try:
        paths = run_paths(
            args.channel, args.study, base=args.base,
            lepton=args.lepton, ratio=args.ratio, bg=args.bg,
            t_rh=args.t_rh, t_max=args.t_max,
            q_min=args.q_min, q_max=args.q_max, q_num=args.q_num,
            simplify=args.simplify, tabulate=args.tabulate, mode=args.mode,
        )
    except ValueError as error:
        parser.error(str(error))

    for key in ("RUN_NAME", "RUN_DIR", "PARTS_DIR", "OUTPUT_PATH", "RUN_ARGS"):
        print("{}={}".format(key, _shell_quote(paths[key])))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
