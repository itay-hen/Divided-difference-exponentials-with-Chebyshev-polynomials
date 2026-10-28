#!/usr/bin/env python3
"""
e3_truncation.py -- Experiment E3: observed truncation error of the
Chebyshev--Bessel series versus the certified a-priori bound.

For each (q, c) uniform random nodes on [-c, c] (fixed seed) are drawn and,
on the exact interval [a, b] = [-c, c] (d = 0, y = x / c), we compute in
mpmath at high precision

    D_q^(n) = T_n[y_0..y_q]   via   T_{n+1}[y_0..y_k] = 2(y_k T_n[y_0..y_k]
                                    + T_n[y_0..y_{k-1}]) - T_{n-1}[y_0..y_k],
    R_n     = I_n(c) / I_0(c),
    S_N     = D_q^(0) + 2 sum_{n=1}^N R_n D_q^(n),
    S       = exp[x_0..x_q] c^q / I_0(c)      (exact; exp[...] from
                                               mp_reference.exp_divdiff,
                                               i.e. independent of the series),
    e_N     = |S - S_N| / |S|                 (true relative truncation error),
    B_N     = 2 sum_{n>N} R_n B(n,q) / ( c^q e^{-c} / (I_0(c) q!) ),
              B(n,q) = 2^{q-1} (n/q) (2q)_{n-q} / (n-q)!   (n >= q),
    t_N     = |2 R_N D_q^(N)| / |S|           (what the stopping rule watches),

and compare with N_code, the order at which the double-precision C++ code
(e3_code_N, tol 1e-14) stops for the same doubles.

Usage:
    python3 -I e3_truncation.py --cheb-bin ./e3_code_N --csv out.csv \
        --summary summary.csv --fig fig_truncation_bounds.pdf
"""

import argparse
import csv
import math
import os
import subprocess
import sys

import numpy as np
import mpmath
from mpmath import mp, mpf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mp_reference import exp_divdiff          # noqa: E402
import plot_style                             # noqa: E402

CASES = [(1, 10), (5, 50), (20, 20), (50, 10), (10, 100)]
SEED = 20261027
TOL = 1e-14


def nodes_for(q, c):
    rng = np.random.default_rng(SEED + 1000 * q + c)
    return [float(v) for v in rng.uniform(-c, c, q + 1)]


def series(q, c, x, dps, Nmax):
    """Return lists D[n] (n=0..Nmax) and R[n] at `dps` digits."""
    with mp.workdps(dps):
        C = mpf(c)
        y = [mpf(v) / C for v in x]
        # T_n[y_0..y_k] for k = 0..q : prev = layer n-1, cur = layer n
        prev = [mpf(1)] + [mpf(0)] * q                 # n = 0
        cur = [y[0]] + ([mpf(1)] if q >= 1 else []) + [mpf(0)] * max(0, q - 1)
        D = [prev[q], cur[q]]
        for n in range(1, Nmax):
            nxt = [mpf(0)] * (q + 1)
            for k in range(q + 1):
                v = y[k] * cur[k] + (cur[k - 1] if k > 0 else 0)
                nxt[k] = 2 * v - prev[k]
            prev, cur = cur, nxt
            D.append(cur[q])
        # Bessel ratios by Miller's backward recurrence for rho_n = I_n/I_{n-1}
        # (stable), started far above Nmax; checked against besseli below.
        top = Nmax + int(2 * c) + 60
        rho = [mpf(0)] * (top + 2)
        r = mpf(0)
        for n in range(top, 0, -1):
            r = 1 / (2 * n / C + r)          # rho_n = 1/(2n/c + rho_{n+1})
            rho[n] = r
        R = [mpf(1)]
        for n in range(1, Nmax + 1):
            R.append(R[-1] * rho[n])
        chk = mpmath.besseli(min(Nmax, 40), C) / mpmath.besseli(0, C)
        assert abs(R[min(Nmax, 40)] / chk - 1) < mpf(10) ** (-(dps - 15)), "Bessel check"
        return D, R


def bound_B(n, q):
    if n < q:
        return mpf(0)
    if q == 0:
        return mpf(2)
    return mpf(2) ** (q - 1) * (mpf(n) / q) * mpmath.rf(2 * q, n - q) / mpmath.factorial(n - q)


def run_case(q, c, code_row):
    x = nodes_for(q, c)
    ref, _ = exp_divdiff(x)
    # Nmax: well beyond the point where the per-term size and the bound tail
    # are below 1e-30 (checked after the fact).
    Nmax = int(q + 2 * c + 25 * math.sqrt(c + 1) + 60)
    # working precision: cancellation in S_N is up to ~ e^{2c} and the
    # D_q^(n) recurrence loses ~ log10 B(Nmax, q); add generous guard digits.
    dps0 = int(60 + 2 * c / math.log(10) + 0.5 * float(mpmath.log10(bound_B(Nmax, q) + 1)) + 2 * q)
    out = {}
    for dps in (dps0, dps0 + 60):                       # precision check
        D, R = series(q, c, x, dps, Nmax)
        with mp.workdps(dps):
            C = mpf(c)
            S = ref * C ** q / mpmath.besseli(0, C)
            lower = C ** q * mpmath.exp(-C) / (mpmath.besseli(0, C) * mpmath.factorial(q))
            terms = [D[0]] + [2 * R[n] * D[n] for n in range(1, Nmax + 1)]
            partial, s = [], mpf(0)
            for t in terms:
                s += t
                partial.append(s)
            e = [abs(S - p) / abs(S) for p in partial]
            tN = [abs(t) / abs(S) for t in terms]
            Bt = [2 * R[n] * bound_B(n, q) for n in range(Nmax + 1)]
            tail, acc = [mpf(0)] * (Nmax + 1), mpf(0)
            for n in range(Nmax, -1, -1):
                tail[n] = acc                            # sum_{m > n}
                acc += Bt[n]
            BN = [t / lower for t in tail]
        out[dps] = (e, tN, BN, partial[-1], S)
    e, tN, BN, SNmax, S = out[dps0 + 60]
    e_lo = out[dps0][0]
    # precision check on the converged partial sum and on e_N where e_N > 1e-40
    prec_ok = all(abs(a - b) <= mpf("1e-3") * b for a, b in zip(e_lo, e) if b > mpf("1e-40"))
    assert e[-1] < mpf("1e-30") and BN[-1] < mpf("1e-30"), f"Nmax too small for {(q, c)}"

    def first_below(arr, thr):
        for n, v in enumerate(arr):
            if v < thr and all(w < thr for w in arr[n:]):
                return n
        return None

    Ncode = int(code_row["N"])
    S_code = float(code_row["S_mant"]) * 2.0 ** int(code_row["S_exp"])
    with mp.workdps(40):
        code_err = abs(mpf(S_code) - S) / abs(S)
    return dict(q=q, c=c, x=x, e=e, tN=tN, BN=BN, Nmax=Nmax, dps=dps0 + 60,
                prec_ok=prec_ok, N_true=first_below(e, mpf(TOL)),
                N_bound=first_below(BN, mpf(TOL)), N_code=Ncode,
                eN_at_code=e[Ncode], BN_at_code=BN[Ncode],
                code_total_err=code_err, code_kuN=float(code_row["kuN"]),
                prod_status=code_row["prod_status"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cheb-bin", required=True)
    ap.add_argument("--csv", required=True, help="per-N curves")
    ap.add_argument("--summary", required=True)
    ap.add_argument("--fig", required=True)
    a = ap.parse_args()

    inp = "".join(f"{q} {c} " + " ".join(f"{v:.17g}" for v in nodes_for(q, c)) + "\n"
                  for q, c in CASES)
    res = subprocess.run([a.cheb_bin], input=inp, capture_output=True, text=True, check=True)
    code_rows = list(csv.DictReader(res.stdout.splitlines()))

    results = [run_case(q, c, row) for (q, c), row in zip(CASES, code_rows)]

    with open(a.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["q", "c", "N", "true_rel_trunc_err", "per_term_rel", "bound_rel"])
        for r in results:
            for n in range(r["Nmax"] + 1):
                w.writerow([r["q"], r["c"], n, mpmath.nstr(r["e"][n], 6),
                            mpmath.nstr(r["tN"][n], 6), mpmath.nstr(r["BN"][n], 6)])
    with open(a.summary, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["q", "c", "N_true_lt_tol", "N_bound_lt_tol", "N_code",
                    "eN_at_N_code", "BN_at_N_code", "code_total_rel_err", "code_kuN",
                    "prod_status", "dps", "precision_check_ok", "nodes"])
        for r in results:
            w.writerow([r["q"], r["c"], r["N_true"], r["N_bound"], r["N_code"],
                        mpmath.nstr(r["eN_at_code"], 4), mpmath.nstr(r["BN_at_code"], 4),
                        mpmath.nstr(r["code_total_err"], 4), f"{r['code_kuN']:.3e}",
                        r["prod_status"], r["dps"], r["prec_ok"],
                        " ".join(f"{v:.17g}" for v in r["x"])])
            print(r["q"], r["c"], "N_true", r["N_true"], "N_bound", r["N_bound"],
                  "N_code", r["N_code"], "e_N(code)", mpmath.nstr(r["eN_at_code"], 3),
                  "code err", mpmath.nstr(r["code_total_err"], 3), "kuN", r["code_kuN"],
                  r["prod_status"], "prec_ok", r["prec_ok"])

    # ---------------------------------------------------------------- figure
    import matplotlib.pyplot as plt
    plot_style.apply()
    P = plot_style.PALETTE
    fig, axs = plt.subplots(2, 3, figsize=(6.5, 4.0), sharey=True)
    axs = axs.ravel()
    floor = 1e-30
    for ax, r in zip(axs, results):
        n = np.arange(r["Nmax"] + 1)
        f = lambda arr: np.array([float(max(v, mpf(floor) / 10)) for v in arr])
        e, tN, BN = f(r["e"]), f(r["tN"]), f(r["BN"])
        tmask = tN > floor
        ax.semilogy(n, BN, color=P[1], lw=1.2, label=r"certified bound $B_N$")
        ax.semilogy(n, e, color=P[0], lw=1.4, label=r"true error $e_N$")
        ax.semilogy(n[tmask], tN[tmask], ".", color=P[2], ms=2.2,
                    label=r"term size $|2R_ND_q^{(N)}|/|S|$")
        ax.axhline(TOL, color="0.5", lw=0.6, ls=":")
        ax.axvline(r["N_code"], color=P[3], lw=0.9, ls="--", label=r"$N$ chosen by code")
        ax.set_ylim(1e-20, 1e12)
        xmax = max(r["N_bound"] or 0, r["N_code"]) * 1.25 + 5
        ax.set_xlim(0, min(r["Nmax"], xmax))
        ax.text(0.97, 0.95, f"$q={r['q']},\\ c={r['c']}$", transform=ax.transAxes,
                ha="right", va="top", fontsize=8.5,
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
        ax.set_xlabel(r"truncation order $N$")
    for ax in axs[::3]:
        ax.set_ylabel("relative error / bound")
    axs[1].set_yticks([1e-20, 1e-14, 1e-8, 1e-2, 1e4, 1e10])
    h, l = axs[0].get_legend_handles_labels()
    axs[-1].axis("off")
    axs[-1].legend(h, l + [], loc="center", frameon=False, fontsize=8)
    axs[-1].text(0.5, 0.12, r"dotted line: tol $=10^{-14}$", transform=axs[-1].transAxes,
                 ha="center", fontsize=8, color="0.3")
    fig.tight_layout(pad=0.4, w_pad=0.6, h_pad=0.8)
    fig.savefig(a.fig)
    print("wrote", a.fig)


if __name__ == "__main__":
    main()
