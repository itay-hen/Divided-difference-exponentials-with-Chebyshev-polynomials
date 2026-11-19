"""Figure: modified-Bessel ratios R_n(c) = I_n(c)/I_0(c).

Left : R_n against c for several n.
Right: R_n against n for several c, with the bound (c/2)^n/n! (dashed)
       and the Debye approximation exp(-(c + n asinh(n/c) - sqrt(n^2+c^2)))
       (dotted). The bound is drawn for c <= 10 only: it is below 1
       only for n > ec/2 (about 136 for c = 100).
Values come from scipy's exponentially scaled ive(n, c), so they do not
overflow; spot checks against mpmath are printed.

Usage: python3 plot_Rn.py OUT.pdf
"""
import math, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ive, gammaln

dst = sys.argv[1]
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                     "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5})


def R(n, c):
    return ive(n, c) / ive(0, c)


# spot checks against mpmath
try:
    import mpmath as mp
    mp.mp.dps = 40
    worst = 0.0
    for n, c in [(1, 0.1), (5, 1.0), (50, 10.0), (200, 100.0), (900, 1000.0), (30, 1000.0)]:
        ref = mp.besseli(n, c) / mp.besseli(0, c)
        worst = max(worst, abs(float((R(n, c) - ref) / ref)))
    print(f"plot_Rn: max rel. deviation from mpmath at spot checks: {worst:.1e}")
except ImportError:
    pass

fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.7), constrained_layout=True)
cc = np.logspace(-1, 3, 300)
cmap = plt.get_cmap("viridis")
ns = [1, 2, 5, 10, 20, 50]
for i, n in enumerate(ns):
    ax[0].plot(cc, R(n, cc), lw=1.1, color=cmap(i / (len(ns) - 1)), label=f"$n={n}$")
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_ylim(1e-16, 2)
ax[0].set_xlabel("half-width $c$"); ax[0].set_ylabel(r"$R_n(c)$")
ax[0].legend(frameon=False, ncol=2, loc="center right")

show_c = [1, 10, 100, 1000]
for i, c in enumerate(show_c):
    col = plt.get_cmap("plasma")(i / len(show_c))
    n = np.arange(1, 5000)
    r = R(n, c)
    keep = r > 1e-17
    ax[1].plot(n[keep], r[keep], lw=1.1, color=col, label=f"$c={c:g}$")
    bound = np.exp(n * math.log(c / 2) - gammaln(n + 1))
    kb = (bound > 1e-17) & (bound < 1.0) & (c <= 10)
    ax[1].plot(n[kb], bound[kb], "--", lw=0.8, color=col)
    deb = np.exp(-(c + n * np.arcsinh(n / c) - np.sqrt(n ** 2 + c ** 2)))
    kd = deb > 1e-17
    ax[1].plot(n[kd], deb[kd], ":", lw=1.0, color="k")
ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_ylim(1e-16, 2)
ax[1].set_xlim(1, 5000)
ax[1].set_xlabel("$n$"); ax[1].set_ylabel(r"$R_n(c)$")
ax[1].plot([], [], "k--", lw=0.8, label=r"$(c/2)^n/n!$")
ax[1].plot([], [], "k:", lw=1.0, label="Debye")
ax[1].legend(frameon=False, ncol=1, loc="center right")
for a, lab in zip(ax, "ab"):
    a.text(-0.02, 1.02, f"({lab})", transform=a.transAxes, ha="right", va="bottom", fontsize=9)
fig.savefig(dst)
