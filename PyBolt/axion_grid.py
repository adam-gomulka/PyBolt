"""File formats and helpers for axion distribution runs: the f grid, part files,
run metadata and the on-disk row format."""

import json
import os
import uuid

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

# Second header line of a .dat: the q values, column-aligned with f(q).
Q_HEADER_PREFIX = "# q,"


def f_grid(f_min, f_max, f_num):
    """The log-spaced grid of axion decay constants for a run."""
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

    Raises ValueError unless the file holds exactly one non-blank line.
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

    Writes to a uniquely named temp file in the same directory, then renames it.
    The unique name lets concurrent writers target the same path safely.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary = os.path.join(
        directory,
        ".{}.{}.tmp".format(os.path.basename(path), uuid.uuid4().hex),
    )
    try:
        with open(temporary, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.remove(temporary)
        except OSError:
            pass
        raise


def format_grid(values):
    """One line of comma-separated values, no leading f_a."""
    return ",".join(VALUE_FORMAT.format(float(value)) for value in values) + "\n"


def format_q_header(q_values):
    """The q-grid comment line for a merged .dat."""
    return Q_HEADER_PREFIX + format_grid(q_values)


def read_q_grid_from_file(path):
    """The q grid recorded in a .dat header, or None (e.g. for nbe files)."""
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
    """Record the q grid the solver used. Returns True if it wrote."""
    path = q_grid_path(parts_dir)
    if os.path.exists(path):
        return False
    atomic_write_text(path, format_grid(q_values))
    return True


def read_q_grid_text(parts_dir):
    """The recorded q grid line, or None (nbe runs record none)."""
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

    Concurrent tasks may both write; they write identical content.
    """
    path = meta_path(parts_dir)
    if os.path.exists(path):
        return False
    atomic_write_text(path, json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return True


def read_meta(parts_dir):
    """The run metadata for a parts directory.

    Raises FileNotFoundError if absent.
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


def read_run(run_dir):
    """Load one run directory: (meta, f_a, f, q).

    ``q`` is None for nbe runs, which have no momentum axis; ``f`` is then the
    single Y column.
    """

    meta_file = os.path.join(run_dir, META_FILENAME)
    with open(meta_file) as handle:
        meta = json.load(handle)

    filename = "Y.dat" if meta.get("mode") == "nbe" else "fa.dat"
    data_path = os.path.join(run_dir, filename)

    table = np.loadtxt(data_path, delimiter=",", ndmin=2)
    f_a = table[:, 0]
    values = table[:, 1:]

    return meta, f_a, values, read_q_grid_from_file(data_path)
