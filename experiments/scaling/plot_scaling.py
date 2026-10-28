"""Figure: truncation length N(q,c) and error indicator kappa*u*N.

Input : Data and figures/v18/n_kappa_sweep_pinned.csv (n_kappa_sweep pinned)
Output: tex/figs_v18/fig_scaling.pdf
"""
import csv, math, sys, statistics as st
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                     "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5})
g = defaultdict(list)
for r in csv.DictReader(open(src)):
    g[(int(r["q"]), float(r["c"]))].append(r)
qs = sorted({k[0] for k in g}); cs = sorted({k[1] for k in g})
med = lambda q, c, f: st.median([float(r[f]) for r in g[(q, c)]])
L = math.log(1e14)

fig, ax = plt.subplots(1, 3, figsize=(6.5, 2.9), constrained_layout=True)
cmap = plt.get_cmap("viridis")
show_q = [1, 10, 50, 100, 200, 500]
for i, q in enumerate(show_q):
    col = cmap(i / (len(show_q) - 1))
    ax[0].plot(cs, [med(q, c, "N") for c in cs], "o-", ms=3, lw=1, color=col, label=f"$q={q}$")
cc = np.logspace(-1, 3, 100)
ax[0].plot(cc, np.sqrt(2 * cc * L), "k--", lw=0.8, label=r"$\sqrt{2c\ln(1/\varepsilon)}$")
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("half-width $c$"); ax[0].set_ylabel("truncation length $N$")
ax[0].legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.28), handlelength=1.5, columnspacing=0.8)

show_c = [0.1, 1, 10, 100, 1000]
for i, c in enumerate(show_c):
    col = plt.get_cmap("plasma")(i / len(show_c))
    ax[1].plot(qs, [med(q, c, "N") - q for q in qs], "s-", ms=3, lw=1, color=col, label=f"$c={c:g}$")
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel("order $q$"); ax[1].set_ylabel("$N-q$")
ax[1].legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.28), handlelength=1.5, columnspacing=0.8)

Z = np.array([[math.log10(med(q, c, "est")) for c in cs] for q in qs])
im = ax[2].pcolormesh(np.arange(len(cs) + 1), np.arange(len(qs) + 1), Z,
                      cmap="RdYlGn_r", vmin=-15, vmax=0, shading="flat")
ax[2].contour(np.arange(len(cs)) + 0.5, np.arange(len(qs)) + 0.5, Z, levels=[-10],
              colors="k", linewidths=1.0)
ax[2].set_xticks(np.arange(len(cs)) + 0.5); ax[2].set_xticklabels([f"{c:g}" for c in cs], rotation=90, fontsize=7)
ax[2].set_yticks(np.arange(len(qs)) + 0.5); ax[2].set_yticklabels([str(q) for q in qs], fontsize=7)
ax[2].set_xlabel("half-width $c$"); ax[2].set_ylabel("order $q$")
cb = fig.colorbar(im, ax=ax[2], shrink=0.9); cb.set_label(r"$\log_{10}(\kappa u N)$")
for a, lab in zip(ax, "abc"):
    a.text(-0.02, 1.02, f"({lab})", transform=a.transAxes, ha="right", va="bottom", fontsize=9)
fig.savefig(dst)
