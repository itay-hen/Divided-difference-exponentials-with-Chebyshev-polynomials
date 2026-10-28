#!/usr/bin/env python3
"""
crosscheck_reference.py -- validate mp_reference.py (sorted divided-difference
table, adaptive precision) against an algorithmically different
high-precision method: the (0, q) entry of the matrix exponential of the
Opitz matrix J = diag(x) + superdiag(1), computed by mpmath.expm at
generous precision.

Usage:
    python3 -I crosscheck_reference.py --nodes accuracy_e2_nodes.csv \
        --ref accuracy_e2_reference.csv [--qmax 20]
Checks rep 0 of every (q <= qmax, c, config) instance.
"""
import argparse
import csv

import mpmath
from mpmath import mp, mpf


def opitz_expm(x, dps):
    with mp.workdps(dps):
        n = len(x)
        J = mpmath.zeros(n, n)
        for i in range(n):
            J[i, i] = mpf(x[i])
            if i + 1 < n:
                J[i, i + 1] = 1
        return +mpmath.expm(J)[0, n - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--qmax", type=int, default=20)
    a = ap.parse_args()
    ref = {r["instance"]: r for r in csv.DictReader(open(a.ref))}
    worst = mpf(0)
    n = 0
    for r in csv.DictReader(open(a.nodes)):
        q, c = int(r["q"]), float(r["c"])
        if q > a.qmax or r["rep"] != "0":
            continue
        x = [float(t) for t in r["nodes"].split()]
        # expm loses ~ log10(e^{2c}) + q*log10(range/gap) digits; be generous
        dps = int(80 + 2 * c / 2.302585 + 10 * q)
        v = opitz_expm(x, dps)
        with mp.workdps(60):
            R = mpf(ref[r["instance"]]["value"])
            rel = abs(v - R) / abs(R)
        worst = max(worst, rel)
        n += 1
        print(f"q={q:3d} c={c:5g} {r['config']:20s} rel.diff = {mpmath.nstr(rel, 3)}")
    print(f"{n} instances; max relative difference table-vs-expm = {mpmath.nstr(worst, 3)}")


if __name__ == "__main__":
    main()
