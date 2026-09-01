"""Shared primitives for the parallel generation of axion distribution files.

Both the worker (``fBE_LFC.py``) and the merger (``merge_distributions.py``)
import from here, so the f grid, the part filenames and the on-disk row format
are each defined exactly once.
"""

import json
import os

import numpy as np

HEADER = "# f_a, f(q)\n"          # phase-space mode (fbe)
HEADER_NUMBER_DENSITY = "# f_a, Y\n"  # number-density mode (nbe)
VALUE_FORMAT = "{:.5e}"
PART_PREFIX = "f_"
PART_SUFFIX = ".dat"
FAIL_SUFFIX = ".fail"
INDEX_DIGITS = 5
META_FILENAME = "meta.json"
Q_GRID_FILENAME = "q_grid.dat"

# The q grid is written into the .dat as a second comment line, column-aligned
# with the data rows: field 0 is the label, fields 1..N are the q values sitting
# above the f(q) values they belong to.
Q_HEADER_PREFIX = "# q,"


def f_grid(f_min, f_max, f_num):
    """The full grid of axion decay constants for a run.

    Every process recomputes this identically from the same three numbers, which
    is what makes a grid index a globally meaningful name for one f value.
    """
    return np.logspace(np.log10(f_min), np.log10(f_max), f_num)


def part_filename(index):
    """Name of the part file holding the solved row for grid index ``index``."""
    return "{}{:0{}d}{}".format(PART_PREFIX, index, INDEX_DIGITS, PART_SUFFIX)


def fail_filename(index):
    """Name of the failure marker for grid index ``index``."""
    return "{}{:0{}d}{}".format(PART_PREFIX, index, INDEX_DIGITS, FAIL_SUFFIX)


def parse_part_index(filename):
    """Grid index encoded in a part filename, or None if it is not a part file."""
    name = os.path.basename(filename)
    if not name.startswith(PART_PREFIX) or not name.endswith(PART_SUFFIX):
        return None
    digits = name[len(PART_PREFIX):-len(PART_SUFFIX)]
    if not digits.isdigit():
        return None
    return int(digits)


def format_row(f_a, distribution):
    """One data row, in exactly the format the final .dat file uses."""
    fields = [VALUE_FORMAT.format(float(f_a))]
    fields.extend(VALUE_FORMAT.format(float(value)) for value in distribution)
    return ",".join(fields) + "\n"


def read_row(path):
    """The single data line of a part file, without its trailing newline.

    Raises ValueError if the file does not hold exactly one non-blank line,
    which is how a truncated or doubly-written part is caught.
    """
    with open(path) as handle:
        lines = [line for line in handle.read().splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(
            "{}: expected exactly 1 data row, found {}".format(path, len(lines))
        )
    return lines[0]


def row_f_a(line):
    """The f_a value a data row starts with."""
    return float(line.split(",", 1)[0])


def row_width(line):
    """Number of comma-separated fields in a data row (1 + N_q when valid)."""
    return len(line.split(","))


def atomic_write_text(path, text):
    """Write ``text`` to ``path`` so no reader ever observes a partial file.

    The temp file is created in the destination directory so the final rename
    stays within one filesystem, where os.replace is atomic.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary = os.path.join(directory, "." + os.path.basename(path) + ".tmp")
    with open(temporary, "w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def format_grid(values):
    """One line of comma-separated values, no leading f_a.

    This is the layout ``fit_params.load_data`` expects from its ``q_file``
    argument: a single row it reads with ``pd.read_csv(..., header=None)``.
    """
    return ",".join(VALUE_FORMAT.format(float(value)) for value in values) + "\n"


def format_q_header(q_values):
    """The q-grid comment line for a merged .dat."""
    return Q_HEADER_PREFIX + format_grid(q_values)


def read_q_grid_from_file(path):
    """The q grid recorded in a merged .dat, or None if it carries none.

    nbe files have no q axis, and files written before the grid was recorded have
    no q line either.

    Downstream readers that only want the numbers do not need this: every header
    line starts with '#', so ``np.loadtxt(path, delimiter=',', comments='#')`` and
    ``pd.read_csv(path, comment='#', header=None)`` both skip them regardless of
    how many there are.
    """
    with open(path) as handle:
        for line in handle:
            if not line.startswith("#"):
                return None
            if line.startswith(Q_HEADER_PREFIX):
                fields = line[len(Q_HEADER_PREFIX):].strip().split(",")
                return np.array([float(field) for field in fields])
    return None


def q_grid_path(parts_dir):
    """Path of the recorded q grid inside a parts directory."""
    return os.path.join(parts_dir, Q_GRID_FILENAME)


def write_q_grid_if_absent(parts_dir, q_values):
    """Record the q grid the solver used. Returns True if it wrote.

    Written rather than re-derived from the metadata bounds so the values that
    ship next to the data are exactly the ones the solver saw, with no chance of
    the two drifting apart.
    """
    path = q_grid_path(parts_dir)
    if os.path.exists(path):
        return False
    atomic_write_text(path, format_grid(q_values))
    return True


def read_q_grid_text(parts_dir):
    """The recorded q grid line, or None when the run did not record one.

    nbe runs have no q axis in their output, so they record no grid.
    """
    path = q_grid_path(parts_dir)
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return handle.read()


def meta_path(parts_dir):
    """Path of the run-metadata file inside a parts directory."""
    return os.path.join(parts_dir, META_FILENAME)


def write_meta_if_absent(parts_dir, meta):
    """Write the run metadata unless it is already there. Returns True if written.

    Array tasks all try this; they race harmlessly, because they write identical
    content and the write itself is atomic.
    """
    path = meta_path(parts_dir)
    if os.path.exists(path):
        return False
    atomic_write_text(path, json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return True


def read_meta(parts_dir):
    """The run metadata for a parts directory.

    Raises FileNotFoundError if absent -- without it there is no way to know how
    many rows to expect or what q grid the data sits on.
    """
    path = meta_path(parts_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "no {} in {} -- cannot merge without the run metadata".format(
                META_FILENAME, parts_dir
            )
        )
    with open(path) as handle:
        return json.load(handle)
