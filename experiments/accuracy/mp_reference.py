#!/usr/bin/env python3
"""
mp_reference.py -- independent high-precision reference values for the
exponential divided difference exp[x_0, ..., x_q].

Method
------
Divided differences are symmetric in the nodes, so the nodes are sorted and
the classical divided-difference table is run in mpmath:

    f[x_i..x_{i+k}] = (f[x_{i+1}..x_{i+k}] - f[x_i..x_{i+k-1}]) / (x_{i+k} - x_i)

After sorting, x_{i+k} == x_i only if the whole window is constant, and then
the confluent value f[x,..,x] = e^x / k! is used.  This covers repeated nodes
of any multiplicity exactly.

Nodes are taken as the exact binary doubles (mpf(float)), so the reference is
the exact divided difference of the very doubles the C++ code receives.

Precision is chosen adaptively.  The table amplifies relative errors at level
k by roughly k / w_k (w_k = smallest nonzero width of a level-k window, with
an O(1) value scale), so an a-priori digit loss is estimated as

    loss = sum_k max(0, log10(k / w_k)) + log10(max |exp| / min |exp|) term,

and the table is run at P = 40 + 1.2 * loss decimal digits and at 2P,
doubling until two consecutive results agree to REL_TOL = 1e-25 relative.
The result at the higher precision is returned.  This agreement test, not the
heuristic, is what certifies the digits.

Usage
-----
    python3 -I mp_reference.py --selftest
    python3 -I mp_reference.py --nodes nodes.csv --out ref.csv [--jobs 8]

The node CSV must have columns  instance,...,nodes  where `nodes` is a
space-separated list of %.17g doubles.  The output CSV has columns
    instance, q, sign, log10_abs, value (30 significant digits), dps
"""

import argparse
import csv
import math
import sys
from multiprocessing import Pool

import mpmath
from mpmath import mp, mpf

REL_TOL = mpf("1e-25")


def _dd_table(xs, dps):
    """exp[xs] for SORTED mpf nodes, at `dps` decimal digits."""
    with mp.workdps(dps):
        n = len(xs)
        col = [mpmath.exp(x) for x in xs]          # level 0
        fact = mpf(1)
        for k in range(1, n):
            fact *= k
            new = []
            for i in range(n - k):
                lo, hi = xs[i], xs[i + k]
                if hi == lo:                        # constant window
                    new.append(mpmath.exp(lo) / fact)
                else:
                    new.append((col[i + 1] - col[i]) / (hi - lo))
            col = new
        return +col[0]


def _loss_estimate(xs_f):
    """A-priori estimate of decimal digits lost in the table (sorted floats)."""
    n = len(xs_f)
    loss = 0.0
    for k in range(1, n):
        w = min((xs_f[i + k] - xs_f[i] for i in range(n - k)
                 if xs_f[i + k] > xs_f[i]), default=None)
        if w is not None and w > 0:
            loss += max(0.0, math.log10(k / w))
    # magnitude spread of the level-0 values e^{x_i}
    loss += (xs_f[-1] - xs_f[0]) / math.log(10)
    return loss


def exp_divdiff(nodes, rel_tol=REL_TOL, max_dps=200000):
    """Return (value, dps) with value = exp[nodes] to rel_tol relative accuracy.

    `nodes` is any iterable of Python floats (exact binary doubles)."""
    xs_f = sorted(float(x) for x in nodes)
    P = int(40 + 1.2 * _loss_estimate(xs_f))
    xs = [mpf(x) for x in xs_f]                      # exact conversion
    v1 = _dd_table(xs, P)
    while True:
        P2 = 2 * P
        if P2 > max_dps:
            raise RuntimeError("mp_reference: precision limit reached")
        v2 = _dd_table(xs, P2)
        with mp.workdps(P2):
            if v2 == 0:
                ok = (v1 == 0)
            else:
                ok = abs(v1 - v2) <= rel_tol * abs(v2)
        if ok:
            return v2, P2
        v1, P = v2, P2


# ---------------------------------------------------------------------------
def selftest():
    mp.dps = 50
    checks = []
    # exp[0,...,0] = 1/q!
    for q in (0, 1, 5, 20, 100):
        v, _ = exp_divdiff([0.0] * (q + 1))
        checks.append((f"exp[0 x{q + 1}] = 1/{q}!", v, 1 / mpmath.factorial(q)))
    # exp[a,b] = (e^b - e^a)/(b - a)
    for a, b in ((0.3, 1.7), (-10.0, 10.0), (1.0, 1.0 + 2**-40)):
        v, _ = exp_divdiff([a, b])
        A, B = mpf(a), mpf(b)
        with mp.workdps(200):
            exact = (mpmath.exp(B) - mpmath.exp(A)) / (B - A)
        checks.append((f"exp[{a!r},{b!r}]", v, exact))
    # exp[-1,1] = sinh(1)
    v, _ = exp_divdiff([-1.0, 1.0])
    checks.append(("exp[-1,1] = sinh 1", v, mpmath.sinh(1)))
    # exp[x,x,y] = (e^y - e^x - (y-x)e^x)/(y-x)^2  (repeated node)
    x, y = 0.25, 2.0
    v, _ = exp_divdiff([y, x, x])
    X, Y = mpf(x), mpf(y)
    checks.append(("exp[x,x,y] (confluent)", v,
                   (mpmath.exp(Y) - mpmath.exp(X) - (Y - X) * mpmath.exp(X)) / (Y - X) ** 2))
    # exp[x0..xq] for q+1 nodes all within 1e-12 of 0.5: ~ e^0.5/q! (Taylor),
    # compared with the mean-value bracket [e^min, e^max]/q!
    q = 30
    nodes = [0.5 + 1e-12 * (i - q / 2) for i in range(q + 1)]
    v, _ = exp_divdiff(nodes)
    lo = mpmath.exp(mpf(min(nodes))) / mpmath.factorial(q)
    hi = mpmath.exp(mpf(max(nodes))) / mpmath.factorial(q)
    print(f"  near-coincident q=30: value in mean-value bracket: {lo <= v <= hi}")
    # equispaced nodes x_j = j h: exp[x_0..x_q] = (e^h - 1)^q / (q! h^q)
    q, h = 40, 0.125
    v, _ = exp_divdiff([j * h for j in range(q + 1)])
    H = mpf(h)
    checks.append(("equispaced q=40 closed form", v,
                   (mpmath.exp(H) - 1) ** q / (mpmath.factorial(q) * H ** q)))
    bad = 0
    for name, v, ex in checks:
        rel = abs(v - ex) / abs(ex)
        flag = "OK " if rel < mpf("1e-25") else "BAD"
        bad += flag == "BAD"
        print(f"  [{flag}] {name:32s} rel.err = {mpmath.nstr(rel, 3)}")
    print("selftest:", "PASSED" if bad == 0 else f"{bad} FAILURES")
    return bad == 0


def _work(row):
    inst, q, nodes = row
    v, dps = exp_divdiff(nodes)
    with mp.workdps(dps):
        sign = 0 if v == 0 else (1 if v > 0 else -1)
        l10 = mpmath.log10(abs(v)) if v != 0 else mpf("-inf")
        return (inst, q, sign, mpmath.nstr(l10, 20), mpmath.nstr(v, 30), dps)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--nodes")
    ap.add_argument("--out")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    rows = []
    with open(a.nodes) as f:
        for r in csv.DictReader(f):
            nodes = [float(t) for t in r["nodes"].split()]
            rows.append((int(r["instance"]), len(nodes) - 1, nodes))
    # biggest jobs first for better load balance
    rows.sort(key=lambda r: -r[1])
    # results are collected in memory and written once at the end (the output
    # may live on a cloud-synced folder where long-open handles can time out)
    out = []
    with Pool(a.jobs) as pool:
        for k, res in enumerate(pool.imap_unordered(_work, rows, chunksize=1)):
            out.append(res)
            if k % 100 == 0:
                print(f"  {k}/{len(rows)}", file=sys.stderr, flush=True)
    out.sort(key=lambda r: r[0])
    with open(a.out, "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["instance", "q", "sign", "log10_abs", "value", "dps"])
        w.writerows(out)

if __name__ == "__main__":
    main()
