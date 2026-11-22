"""Figures for the truncation length N(q, c) and the error indicator map.

Input : n_kappa_sweep_pinned.csv (n_kappa_sweep, mode "pinned": nodes
        uniform on [-c, c] with x_0 = -c, x_q = c; 30 node sets per cell)
Output: OUTDIR/fig_N_vs_c.pdf     N against c for fixed q (median, IQR band),
                                  with sqrt(2 c ln(1/eps))
        OUTDIR/fig_N_vs_q.pdf     N - q against q for fixed c (median, IQR
                                  band), with L / W(L/(e c)), L = ln(4/eps),
                                  drawn for q >= 3c
        OUTDIR/fig_kappa_map.pdf  median kappa*u*N of the series alone on
                                  the tight interval (supplement)

Usage: python3 plot_N.py SWEEP.csv OUTDIR
"""
import csv, math, os, sys, statistics as st
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import lambertw

src, outdir = sys.argv[1], sys.argv[2]
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                     "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5})
g = defaultdict(list)
for r in csv.DictReader(open(src)):
    g[(int(r["q"]), float(r["c"]))].append(r)
qs = sorted({k[0] for k in g}); cs = sorted({k[1] for k in g})
EPS = 1e-14


def stats(q, c, f, shift=0.0):
    v = sorted(float(r[f]) - shift for r in g[(q, c)])
    return st.median(v), np.percentile(v, 25), np.percentile(v, 75)


# ---- N against c
fig, ax = plt.subplots(figsize=(3.4, 2.7), constrained_layout=True)
show_q = [1, 10, 50, 100, 200, 500]
cmap = plt.get_cmap("viridis")
for i, q in enumerate(show_q):
    col = cmap(i / (len(show_q) - 1))
    m, lo, hi = zip(*(stats(q, c, "N") for c in cs))
    ax.fill_between(cs, lo, hi, color=col, alpha=0.25, lw=0)
    ax.plot(cs, m, "o-", ms=2.5, lw=1, color=col, label=f"$q={q}$")
cc = np.logspace(-1, 3, 100)
ax.plot(cc, np.sqrt(2 * cc * math.log(1 / EPS)), "k--", lw=0.8,
        label=r"$\sqrt{2c\ln(1/\varepsilon)}$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("half-width $c$"); ax.set_ylabel("truncation length $N$")
ax.legend(frameon=False, ncol=2, loc="lower right", handlelength=1.5, columnspacing=0.8)
fig.savefig(os.path.join(outdir, "fig_N_vs_c.pdf"))
plt.close(fig)

# ---- N - q against q
fig, ax = plt.subplots(figsize=(3.4, 2.7), constrained_layout=True)
show_c = [0.1, 1, 10, 100, 1000]
L4 = math.log(4 / EPS)
for i, c in enumerate(show_c):
    col = plt.get_cmap("plasma")(i / len(show_c))
    m, lo, hi = zip(*(stats(q, c, "N", shift=q) for q in qs))
    ax.fill_between(qs, lo, hi, color=col, alpha=0.25, lw=0)
    ax.plot(qs, m, "s-", ms=2.5, lw=1, color=col, label=f"$c={c:g}$")
    est = L4 / lambertw(L4 / (math.e * c)).real
    qq = [q for q in qs if q >= 3 * c]
    if qq:
        ax.plot([qq[0], qq[-1]], [est, est], "--", lw=0.9, color=col)
ax.plot([], [], "k--", lw=0.9, label=r"$L/W(L/(ec))$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("order $q$"); ax.set_ylabel("$N-q$")
ax.legend(frameon=False, ncol=2, loc="upper left", handlelength=1.5, columnspacing=0.8)
ax.set_ylim(top=ax.get_ylim()[1] * 3)
fig.savefig(os.path.join(outdir, "fig_N_vs_q.pdf"))
plt.close(fig)

# ---- kappa*u*N map (supplement)
fig, ax = plt.subplots(figsize=(3.4, 2.8), constrained_layout=True)
Z = np.array([[math.log10(stats(q, c, "est")[0]) for c in cs] for q in qs])
im = ax.pcolormesh(np.arange(len(cs) + 1), np.arange(len(qs) + 1), Z,
                   cmap="RdYlGn_r", vmin=-15, vmax=0, shading="flat")
ax.contour(np.arange(len(cs)) + 0.5, np.arange(len(qs)) + 0.5, Z, levels=[-10],
           colors="k", linewidths=1.0)
ax.set_xticks(np.arange(len(cs)) + 0.5); ax.set_xticklabels([f"{c:g}" for c in cs], rotation=90, fontsize=7)
ax.set_yticks(np.arange(len(qs)) + 0.5); ax.set_yticklabels([str(q) for q in qs], fontsize=7)
ax.set_xlabel("half-width $c$"); ax.set_ylabel("order $q$")
cb = fig.colorbar(im, ax=ax, shrink=0.9); cb.set_label(r"$\log_{10}(\kappa u N)$")
fig.savefig(os.path.join(outdir, "fig_kappa_map.pdf"))
plt.close(fig)
