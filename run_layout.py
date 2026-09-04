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
