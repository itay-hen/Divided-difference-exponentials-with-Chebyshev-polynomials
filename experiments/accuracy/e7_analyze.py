"""e7_analyze.py -- heatmap of ratio errors, Chebyshev--Bessel vs MNP.

Inputs : PREFIX_results.csv (e7_ratio), REF.csv (mp_reference.py on
         PREFIX_nodes.csv; instance 2k is x, 2k+1 is x' of pair k)
Outputs: FIG.pdf (two heatmaps of the median relative error of the ratio
         against the exact ratio, per (c, q) cell), SUMMARY.csv
         (per-cell medians and counts); a short summary on stdout.

Chebyshev cells: ratios from evaluate_ratio_scaled with the default
settings (no pair is refused); dots mark cells where some pairs were
evaluated by scaling and squaring.
MNP cells use pairs where both MNP values were finite and positive;
crosses mark cells with at least one failure, and a cell where all
failed is grey.

Usage: python3 e7_analyze.py RESULTS.csv REF.csv FIG.pdf SUMMARY.csv
"""
import csv, math, statistics as st, sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import mpmath as mp

res_f, ref_f, fig_f, sum_f = sys.argv[1:5]
mp.mp.dps = 40
ref = {int(r["instance"]): mp.mpf(r["value"]) for r in csv.DictReader(open(ref_f))}

FLOOR = 1e-17
cells = defaultdict(lambda: {"cheb": [], "mnp": [], "n": 0, "refused": 0, "sq": 0,
                             "mnp_fail": 0, "ratio_est": []})
worst_cheb, worst_ratio_est, worst_mnp = 0.0, 0.0, 0.0
for r in csv.DictReader(open(res_f)):
    k = int(r["pair"]); q = int(r["q"]); c = float(r["c"])
    exact = ref[2 * k] / ref[2 * k + 1]
    cell = cells[(q, c)]; cell["n"] += 1
    est = float(r["cheb_est"])
    if r["cheb_refused"] == "1":
        cell["refused"] += 1
    else:
        cell["sq"] += int(r["cheb_sq"]) > 0
        e = float(abs(mp.expm1(mp.mpf(r["cheb_log"]) - mp.log(exact))))
        cell["cheb"].append(e); worst_cheb = max(worst_cheb, e)
        if est > 0:
            cell["ratio_est"].append(e / est); worst_ratio_est = max(worst_ratio_est, e / est)
    mnp = float(r["mnp"])
    if not math.isfinite(mnp):
        cell["mnp_fail"] += 1
    else:
        e = float(abs((mp.mpf(mnp) - exact) / exact))
        cell["mnp"].append(e); worst_mnp = max(worst_mnp, e)

qs = sorted({k[0] for k in cells}); cs = sorted({k[1] for k in cells})
with open(sum_f, "w", newline="") as fo:
    w = csv.writer(fo)
    w.writerow(["q", "c", "pairs", "cheb_refused", "cheb_squaring", "cheb_median", "cheb_max",
                "mnp_failed", "mnp_median", "mnp_max"])
    for q in qs:
        for c in cs:
            d = cells[(q, c)]
            f = lambda v, g: f"{g(v):.3e}" if v else ""
            w.writerow([q, c, d["n"], d["refused"], d["sq"], f(d["cheb"], st.median), f(d["cheb"], max),
                        d["mnp_fail"], f(d["mnp"], st.median), f(d["mnp"], max)])

n_all = sum(d["n"] for d in cells.values())
print(f"e7: {n_all} pairs; Chebyshev refused {sum(d['refused'] for d in cells.values())}, "
      f"by squaring {sum(d['sq'] for d in cells.values())}, "
      f"max rel. error of accepted ratios {worst_cheb:.2e}, max error/indicator {worst_ratio_est:.2f}")
print(f"e7: MNP failed on {sum(d['mnp_fail'] for d in cells.values())} pairs; "
      f"max rel. error where finite {worst_mnp:.2e}")
bad = [(q, c) for (q, c), d in cells.items() if d["mnp"] and st.median(d["mnp"]) > 1e-8]
print(f"e7: cells with MNP median error > 1e-8: {sorted(bad)}")

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                     "font.size": 9, "axes.labelsize": 9})
fig, ax = plt.subplots(1, 2, figsize=(6.5, 3.0), constrained_layout=True)
cmap = plt.get_cmap("RdYlGn_r").copy(); cmap.set_bad("0.75")
for a, key, fail, title in ((ax[0], "cheb", "refused", "Chebyshev--Bessel"),
                            (ax[1], "mnp", "mnp_fail", "MNP (double)")):
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
            if key == "mnp" and 0 < d[fail] < d["n"]:
                a.plot(j + 0.5, i + 0.5, "kx", ms=5)
    a.set_xticks(np.arange(len(cs)) + 0.5); a.set_xticklabels([f"{c:g}" for c in cs], rotation=90, fontsize=7)
    a.set_yticks(np.arange(len(qs)) + 0.5); a.set_yticklabels([str(q) for q in qs], fontsize=7)
    a.set_xlabel("half-width $c$"); a.set_ylabel("order $q$"); a.set_title(title, fontsize=9)
cb = fig.colorbar(im, ax=ax, shrink=0.9)
cb.set_label(r"$\log_{10}$ median relative error of the ratio")
for a, lab in zip(ax, "ab"):
    a.text(-0.02, 1.02, f"({lab})", transform=a.transAxes, ha="right", va="bottom", fontsize=9)
fig.savefig(fig_f)
