"""e8_analyze.py -- accuracy map of single values exp[x_0..x_q] over (c, q).

Inputs : E8.csv (e8_values on the e7 node sets), REF.csv (mp_reference on
         the same nodes), RESULTS7.csv (e7 results, for the (q, c) of each
         pair; instance 2k and 2k+1 belong to pair k)
Outputs: FIG.pdf (median relative error per cell, Chebyshev--Bessel with
         default settings and MNP in double precision), SUMMARY.csv;
         a summary on stdout.

Chebyshev cells: dots mark cells where some sets were evaluated by
scaling and squaring.  MNP cells: crosses mark cells where some sets
failed (non-finite, zero, or below the smallest normal double), grey
cells where all failed.

Usage: python3 e8_analyze.py E8.csv REF.csv RESULTS7.csv FIG.pdf SUMMARY.csv
"""
import csv, math, statistics as st, sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import mpmath as mp

e8_f, ref_f, res7_f, fig_f, sum_f = sys.argv[1:6]
mp.mp.dps = 50
ref = {r["instance"]: mp.mpf(r["value"]) for r in csv.DictReader(open(ref_f))}
qc = {}
for r in csv.DictReader(open(res7_f)):
    k = int(r["pair"])
    qc[str(2 * k)] = qc[str(2 * k + 1)] = (int(r["q"]), float(r["c"]))
TINY = 2.2250738585072014e-308
FLOOR = 1e-17
cells = defaultdict(lambda: {"cheb": [], "mnp": [], "n": 0, "refused": 0, "sq": 0, "mnp_fail": 0})
worst = {"series": 0.0, "squaring": 0.0}
ratio = {"series": 0.0, "squaring": 0.0}
mnp_worst = 0.0
for r in csv.DictReader(open(e8_f)):
    i = r["instance"]; ex = ref[i]; d = cells[qc[i]]; d["n"] += 1
    if r["cheb_ok"] != "1":
        d["refused"] += 1
    else:
        v = mp.mpf(r["cheb_mant"]) * mp.mpf(2) ** int(r["cheb_exp"])
        e = float(abs((v - ex) / ex)); d["cheb"].append(e)
        key = "squaring" if int(r["cheb_sq"]) > 0 else "series"
        d["sq"] += key == "squaring"
        worst[key] = max(worst[key], e)
        ratio[key] = max(ratio[key], e / float(r["cheb_est"]))
    m = float(r["mnp"])
    if math.isnan(m) or m < TINY:
        d["mnp_fail"] += 1
    else:
        e = float(abs((mp.mpf(m) - ex) / ex)); d["mnp"].append(e); mnp_worst = max(mnp_worst, e)

qs = sorted({k[0] for k in cells}); cs = sorted({k[1] for k in cells})
n_all = sum(d["n"] for d in cells.values())
print(f"e8: {n_all} values; refused {sum(d['refused'] for d in cells.values())}; "
      f"by squaring {sum(d['sq'] for d in cells.values())}")
print(f"e8: max rel. error: series {worst['series']:.2e}, squaring {worst['squaring']:.2e}; "
      f"max error/indicator: series {ratio['series']:.3f}, squaring {ratio['squaring']:.3f}")
print(f"e8: MNP failed on {sum(d['mnp_fail'] for d in cells.values())}; "
      f"max error where it returned a normal value {mnp_worst:.2e}")
bad = sorted((k, st.median(d['mnp'])) for k, d in cells.items() if d["mnp"] and st.median(d["mnp"]) > 1e-10)
print(f"e8: cells with MNP median error > 1e-10: {bad}")
with open(sum_f, "w", newline="") as fo:
    w = csv.writer(fo)
    w.writerow(["q", "c", "sets", "cheb_refused", "cheb_squaring", "cheb_median", "cheb_max",
                "mnp_failed", "mnp_median", "mnp_max"])
    f = lambda v, g: f"{g(v):.3e}" if v else ""
    for q in qs:
        for c in cs:
            d = cells[(q, c)]
            w.writerow([q, c, d["n"], d["refused"], d["sq"], f(d["cheb"], st.median), f(d["cheb"], max),
                        d["mnp_fail"], f(d["mnp"], st.median), f(d["mnp"], max)])

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                     "font.size": 9, "axes.labelsize": 9})
fig, ax = plt.subplots(1, 2, figsize=(6.5, 3.0), constrained_layout=True)
cmap = plt.get_cmap("RdYlGn_r").copy(); cmap.set_bad("0.75")
for a, key, title in ((ax[0], "cheb", "Chebyshev--Bessel"), (ax[1], "mnp", "MNP (double)")):
    Z = np.full((len(qs), len(cs)), np.nan)
    for i, q in enumerate(qs):
        for j, c in enumerate(cs):
            v = cells[(q, c)][key]
            if v:
                Z[i, j] = math.log10(max(st.median(v), FLOOR))
    im = a.pcolormesh(np.arange(len(cs) + 1), np.arange(len(qs) + 1),
                      np.ma.masked_invalid(Z), cmap=cmap, vmin=-17, vmax=0, shading="flat")
    for i, q in enumerate(qs):
        for j, c in enumerate(cs):
            d = cells[(q, c)]
            if key == "cheb" and d["sq"] > 0:
                a.plot(j + 0.5, i + 0.5, "k.", ms=4)
            if key == "mnp" and 0 < d["mnp_fail"] < d["n"]:
                a.plot(j + 0.5, i + 0.5, "kx", ms=5)
    a.set_xticks(np.arange(len(cs)) + 0.5); a.set_xticklabels([f"{c:g}" for c in cs], rotation=90, fontsize=7)
    a.set_yticks(np.arange(len(qs)) + 0.5); a.set_yticklabels([str(q) for q in qs], fontsize=7)
    a.set_xlabel("half-width $c$"); a.set_ylabel("order $q$"); a.set_title(title, fontsize=9)
cb = fig.colorbar(im, ax=ax, shrink=0.9)
cb.set_label(r"$\log_{10}$ median relative error")
for a, lab in zip(ax, "ab"):
    a.text(-0.02, 1.02, f"({lab})", transform=a.transAxes, ha="right", va="bottom", fontsize=9)
fig.savefig(fig_f)
