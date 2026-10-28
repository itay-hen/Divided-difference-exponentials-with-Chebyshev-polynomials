#!/usr/bin/env python3
"""
mpmath_reference.py

Generate a high-precision ground-truth reference for exp[x_0, ..., x_q]
using mpmath's arbitrary-precision matrix exponential applied to the
Opitz bidiagonal matrix.  Both Chebyshev and MNP can be checked
against this reference; since mpmath operates at user-chosen
precision (default 100 decimal digits here), neither method gets to
"check itself" as in the earlier long-double MNP self-reference.

Algorithm
---------
For nodes x_0, ..., x_q the (q+1)x(q+1) Opitz matrix
    J = diag(x_i) + superdiag(1)
satisfies (Opitz' theorem)
    f(J)_{i,j} = f[x_i, ..., x_j]      for any analytic f.
With f = exp, the divided differences are the first row of exp(J).
mpmath's matrix exponential expm() handles confluent / nearly-
confluent nodes automatically and is accurate to the working
precision.

Inputs
------
A CSV with rows describing test cases:
    case_id, b, q, x_0, x_1, ..., x_q
Empty trailing node slots (when q varies across rows) are tolerated.

Outputs
-------
Two CSVs:
    cases.csv     : the input cases, unchanged.
    reference.csv : case_id, b, q, exp_dd_string
                    where exp_dd_string is the mpmath value as a
                    decimal string at full precision, suitable for
                    parsing back into long double or just plain
                    double on the C++ side.

Usage
-----
    # generate the default grid into cases.csv and reference.csv
    python3 mpmath_reference.py --generate

    # use a custom input CSV
    python3 mpmath_reference.py --input my_cases.csv

    # choose a different working precision
    python3 mpmath_reference.py --dps 200

Dependencies: mpmath.
"""

import argparse
import csv
import os
import random
import sys

try:
    from mpmath import mp, mpf, matrix, expm
except ImportError:
    sys.stderr.write("mpmath_reference.py: requires mpmath\n"
                     "Install with: pip install mpmath\n")
    sys.exit(1)


def divided_diffs_exp(nodes):
    """
    Compute all divided differences exp[x_0, ..., x_k] for k = 0..q
    via the Opitz matrix exponential.  Returns a list of mpf values.
    """
    n = len(nodes)
    if n == 0:
        return []
    J = matrix(n, n)
    for i in range(n):
        J[i, i] = mpf(nodes[i])
        if i + 1 < n:
            J[i, i + 1] = mpf(1)
    E = expm(J)
    return [E[0, k] for k in range(n)]


def divided_diff_exp_full(nodes):
    """
    Return only exp[x_0, ..., x_q], i.e. the last element of the
    first row of exp(Opitz).
    """
    return divided_diffs_exp(nodes)[-1]


def default_grid(quick=False):
    """Default (b, q) grid covering the regimes the paper discusses:
      - small to moderate width:    b in {0.5, 1, 2, 5, 10, 20}
      - large width:                b in {50, 100}
      - q from low to fairly high:  q in {1, 3, 5, 10, 15, 25, 50}
    Plus a few confluent-node stress cases.

    Pass quick=True for a much smaller grid (~20 cases) suitable for
    automated CI; the full grid (~180 cases) is for offline
    validation.

    Returns: list of (case_id, b, q, nodes) tuples.
    """
    random.seed(2026)
    cases = []
    cid = 0

    if quick:
        bs = [1.0, 10.0]
        qs = [1, 5, 15]
        trials_per_cell = 2
    else:
        bs = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
        qs = [1, 3, 5, 10, 15, 25, 50]
        trials_per_cell = 3

    for b in bs:
        for q in qs:
            for _ in range(trials_per_cell):
                nodes = [random.uniform(-b, b) for _ in range(q + 1)]
                cases.append((cid, b, q, nodes))
                cid += 1

    # Confluent / tightly-clustered stress cases.
    stress_bs = [1.0] if quick else [1.0, 10.0]
    stress_qs = [3] if quick else [3, 10]
    for b in stress_bs:
        for q in stress_qs:
            x = random.uniform(-b / 2, b / 2)
            cases.append((cid, b, q, [x] * (q + 1)))
            cid += 1
        for q in stress_qs:
            x0 = random.uniform(-b / 2, b / 2)
            cases.append((cid, b, q, [x0 + i * 1e-6 for i in range(q + 1)]))
            cid += 1

    return cases


def write_cases_csv(cases, path):
    """Write cases to CSV atomically, with a trailing manifest line.

    Writes to path.tmp first, then renames to path on success.  The
    final line is a manifest:  #manifest,n_cases=<N>  which the C++
    loader cross-checks against the number of data lines read.  An
    interrupted write leaves only the .tmp behind; the manifest line
    is missing so the C++ side refuses to use it.
    """
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "b", "q", "nodes_semicolon"])
        for cid, b, q, nodes in cases:
            w.writerow([cid, b, q, ";".join(repr(x) for x in nodes)])
        w.writerow(["#manifest", "n_cases=" + str(len(cases)), "", ""])
    os.replace(tmp, path)


def read_cases_csv(path):
    """Read cases CSV; verify the trailing manifest matches row count.

    Nodes are parsed as ``mpf`` directly from the CSV string, so
    a custom cases file with high-precision decimal nodes is read
    without truncation through Python ``float``.  For default-grid
    cases generated by this script the values originate as Python
    floats and the mpf parse is exact.
    """
    cases = []
    manifest_n = None
    with open(path, "r", newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        for row in r:
            if not row:
                continue
            if row[0].startswith("#manifest"):
                m = row[1].split("=", 1)
                if len(m) == 2:
                    manifest_n = int(m[1])
                continue
            cid = int(row[0])
            b = float(row[1])
            q = int(row[2])
            # Parse as mpf strings to preserve precision.
            nodes = [mpf(s) for s in row[3].split(";")]
            cases.append((cid, b, q, nodes))
    if manifest_n is None:
        raise RuntimeError(
            "cases.csv has no #manifest line; file may be incomplete. "
            "Regenerate with mpmath_reference.py --generate.")
    if manifest_n != len(cases):
        raise RuntimeError(
            "cases.csv manifest says %d cases but %d were read; "
            "file is inconsistent." % (manifest_n, len(cases)))
    return cases


def write_reference_csv(cases, path, dps):
    """
    Compute the mpmath reference for each case and write to CSV
    atomically.  Like write_cases_csv this writes path.tmp first and
    appends a #manifest footer with the case count before renaming
    into place.  An interrupted run leaves only path.tmp (without the
    manifest), and the C++ loader refuses to use such a file.
    """
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "b", "q", "exp_dd"])
        for cid, b, q, nodes in cases:
            v = divided_diff_exp_full(nodes)
            # Emit the full-precision decimal string.  The C++ side
            # parses to long double (~19 digits); the mpmath value
            # carries dps+5 digits so the parse is the limiting
            # precision in C++, not the reference string.
            w.writerow([cid, b, q, mp.nstr(v, dps + 5,
                                          strip_zeros=False)])
        w.writerow(["#manifest", "n_cases=" + str(len(cases)), "", ""])
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generate", action="store_true",
                    help="generate the default grid into cases.csv")
    ap.add_argument("--quick", action="store_true",
                    help="when used with --generate, produce a much "
                         "smaller (~20-case) grid suitable for CI; "
                         "the full grid is ~180 cases and is for "
                         "offline validation only")
    ap.add_argument("--input",    default=None,
                    help="input cases CSV (default: cases.csv)")
    ap.add_argument("--output",   default="reference.csv",
                    help="output reference CSV (default: reference.csv)")
    ap.add_argument("--dps",      type=int, default=100,
                    help="mpmath working precision in decimal digits "
                         "(default 100)")
    args = ap.parse_args()

    mp.dps = args.dps

    if args.generate:
        cases = default_grid(quick=args.quick)
        write_cases_csv(cases, "cases.csv")
        print(f"Wrote {len(cases)} cases to cases.csv"
              f"{' (quick mode)' if args.quick else ''}")
        if args.input is None:
            args.input = "cases.csv"
    if args.input is None:
        ap.error("--input or --generate is required")

    cases = read_cases_csv(args.input)
    print(f"Read {len(cases)} cases from {args.input}")
    print(f"Computing reference at mp.dps = {args.dps} ...")
    write_reference_csv(cases, args.output, args.dps)
    print(f"Wrote reference to {args.output}")


if __name__ == "__main__":
    main()
