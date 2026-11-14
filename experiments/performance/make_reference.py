#!/usr/bin/env python3
"""
make_reference.py -- high-precision references for the E5 performance runs.

Reads the node sets written by e5_runtime (e5_runtime_nodes.csv) and
e5_interval_sensitivity (e5_interval_nodes.csv) and computes
exp[x_0, ..., x_q] with mpmath, using the Newton divided-difference
recurrence at adaptive precision: the value is computed at D and 1.6 D
decimal digits and D is increased until the two agree to 1e-30 relative.
Nodes are converted from their 17-digit decimal form to the exact binary
double first (float(s)), so the reference is for exactly the nodes the C++
code saw.  (The recurrence is exact in exact arithmetic for distinct nodes;
uniform random doubles are distinct with probability one.)

Outputs (same folder):
  e5_runtime_ref.csv   c,q,set,log10_ref,rel_err_cheb,rel_err_mnp
  e5_interval_ref.csv  q,set,log10_ref,ref_mantissa,ref_exp10

Usage: python3 -I make_reference.py <data_dir>
"""
import csv
import math
import sys

from mpmath import mp, mpf, exp, log10, fabs


def divdiff_exp(xs, dps):
    mp.dps = dps
    x = [mpf(v) for v in xs]
    f = [exp(v) for v in x]
    n = len(x)
    for k in range(1, n):
        for i in range(n - k):
            f[i] = (f[i + 1] - f[i]) / (x[i + k] - x[i])
    return f[0]


def reference(xs):
    q = len(xs) - 1
    dps = 40 + int(1.2 * q)
    while True:
        v1 = divdiff_exp(xs, dps)
        v2 = divdiff_exp(xs, int(1.6 * dps))
        mp.dps = int(1.6 * dps)
        if v2 != 0 and fabs(v1 / v2 - 1) < mpf(10) ** -30:
            return v2
        dps = int(dps * 1.6)
        if dps > 20000:
            raise RuntimeError("reference did not converge")


def rel_err(approx, ref):
    if approx is None or not math.isfinite(approx) or approx == 0.0:
        return float("nan")
    mp.dps = 50
    return float(fabs(mpf(approx) / ref - 1))


def main(d):
    # ---- E5 runtime grid -------------------------------------------------
    rows_out = []
    with open(f"{d}/e5_runtime_nodes.csv") as fh:
        for row in csv.reader(fh):
            c, q, s = float(row[0]), int(row[1]), int(row[2])
            cm, ce, mv = float(row[3]), int(row[4]), float(row[5])
            xs = [float(v) for v in row[6:]]
            ref = reference(xs)
            mp.dps = 60
            # Chebyshev value = mantissa * 2^exponent (may be outside double range)
            if math.isfinite(cm) and cm != 0.0:
                cheb = mpf(cm) * mpf(2) ** ce
                ec = float(fabs(cheb / ref - 1))
            else:
                ec = float("nan")
            em = rel_err(mv, ref) if math.isfinite(mv) else float("nan")
            rows_out.append((c, q, s, float(log10(ref)), ec, em))
            print(f"c={c:g} q={q} log10(ref)={float(log10(ref)):.2f} "
                  f"err_cheb={ec:.2e} err_mnp={em:.2e}", flush=True)
    with open(f"{d}/e5_runtime_ref.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["c", "q", "set", "log10_ref", "rel_err_cheb", "rel_err_mnp"])
        w.writerows(rows_out)

    # ---- E5 interval sensitivity --------------------------------------------
    rows_out = []
    with open(f"{d}/e5_interval_nodes.csv") as fh:
        for row in csv.reader(fh):
            q, s = int(row[0]), int(row[1])
            xs = [float(v) for v in row[2:]]
            ref = reference(xs)
            mp.dps = 40
            rows_out.append((q, s, float(log10(ref)), mp.nstr(ref, 25)))
    with open(f"{d}/e5_interval_ref.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["q", "set", "log10_ref", "ref"])
        w.writerows(rows_out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
