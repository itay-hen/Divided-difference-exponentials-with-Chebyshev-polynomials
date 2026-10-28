#!/usr/bin/env python3
"""
make_figures.py -- figures and the LaTeX table for experiments E5 and E6.

Usage: python3 -I make_figures.py <data_dir> <fig_dir>
Reads : e5_runtime.csv, e5_runtime_ref.csv (optional), e5_interval.csv,
        e6_incremental.csv, e6_qmc_timing.csv, e6_qmc_stats.csv, e6_qmc_hist.csv
Writes: fig_runtime.pdf, fig_interval_sensitivity.pdf, fig_incremental.pdf,
        fig_qmc_benchmark.pdf, table_qmc.tex
Also prints the derived numbers quoted in the report (crossovers etc.).
"""
import csv
import math
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

# Okabe-Ito (colorblind-safe); identity is always also encoded by marker/line style.
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "verm": "#D55E00",
      "purple": "#CC79A7", "sky": "#56B4E9", "black": "#000000", "yellow": "#F0E442"}

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
    "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.6, "lines.linewidth": 1.2, "lines.markersize": 4,
    "legend.frameon": False, "pdf.fonttype": 42, "axes.grid": True,
    "grid.color": "#dddddd", "grid.linewidth": 0.4,
})
FULL, SINGLE = 6.5, 3.2


def read(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def crossover(q, t_cheb, t_mnp):
    """Smallest q (log-log interpolation) at which t_cheb <= t_mnp."""
    r = np.log(np.asarray(t_cheb) / np.asarray(t_mnp))
    q = np.asarray(q, float)
    if r[0] <= 0:
        return f"<= {int(q[0])}"
    for i in range(1, len(r)):
        if r[i] <= 0 < r[i - 1]:
            lq = np.log(q[i - 1]) + (np.log(q[i]) - np.log(q[i - 1])) * r[i - 1] / (r[i - 1] - r[i])
            return f"{math.exp(lq):.1f}"
    return "none"


# =============================================================================
def fig_runtime(d, out):
    rows = read(f"{d}/e5_runtime.csv")
    ref = {}
    if os.path.exists(f"{d}/e5_runtime_ref.csv"):
        for r in read(f"{d}/e5_runtime_ref.csv"):
            ref[(float(r["c"]), int(r["q"]))] = (fnum(r["rel_err_cheb"]), fnum(r["rel_err_mnp"]))
    cs = sorted({float(r["c"]) for r in rows})
    fig, axes = plt.subplots(2, len(cs), figsize=(FULL, 4.4), sharex=True,
                             gridspec_kw={"height_ratios": [1.6, 1]})
    print("\n=== E5 runtime ===")
    for j, c in enumerate(cs):
        R = sorted([r for r in rows if float(r["c"]) == c], key=lambda r: int(r["q"]))
        q = np.array([int(r["q"]) for r in R])
        ta = np.array([fnum(r["t_amort_us"]) for r in R])
        tf = np.array([fnum(r["t_fresh_us"]) for r in R])
        tm = np.array([fnum(r["t_mnp_us"]) for r in R])
        fb = np.array([int(r["cheb_fallbacks"]) for r in R])
        rf = np.array([int(r["cheb_refusals"]) for r in R])
        ax = axes[0, j]
        ax.loglog(q, ta, "-o", color=OI["blue"], label="Chebyshev, amortized")
        ax.loglog(q, tf, "--s", color=OI["green"], mfc="none", label=r"Chebyshev, incl. $R_n$ precomp.")
        ax.loglog(q, tm, "-.^", color=OI["verm"], label="MNP")
        # Mark grid points where evaluate() fell back to [min x, max x] (ring)
        # or refused some node sets (cross).
        sel = fb > 0
        ax.plot(q[sel], ta[sel], "o", ms=8, mfc="none", mec=OI["black"], mew=0.7,
                label=r"fallback to $[\min_i x_i,\,\max_i x_i]$" if j == 0 else None)
        sel = rf > 0
        ax.plot(q[sel], ta[sel], "x", ms=7, color=OI["black"], mew=0.9,
                label="some node sets refused" if j == 0 else None)
        # MNP underflow (value < DBL_MIN): open triangles.
        uf = np.array([int(r["mnp_underflow"]) for r in R]) > 0
        ax.plot(q[uf], tm[uf], "^", ms=7, mfc="white", mec=OI["verm"], mew=0.8,
                label="MNP result underflows" if j == 0 else None)
        ax.text(0.04, 0.95, f"$c={c:g}$", transform=ax.transAxes, va="top")
        if j == 0:
            ax.set_ylabel(r"time per evaluation [$\mu$s]")
        ax2 = axes[1, j]
        ax2.loglog(q, tm / ta, "-o", color=OI["blue"], label="MNP / Cheb. (amortized)")
        ax2.loglog(q, tm / tf, "--s", color=OI["green"], mfc="none", label="MNP / Cheb. (incl. precomp.)")
        ax2.axhline(1.0, color="0.3", lw=0.8)
        ax2.set_xlabel("order $q$")
        if j == 0:
            ax2.set_ylabel(r"$t_{\rm MNP}/t_{\rm Cheb}$")
        print(f"c={c:g}: crossover q* amortized {crossover(q, ta, tm)}, incl. precomp "
              f"{crossover(q, tf, tm)}; ratio MNP/Cheb at q={q[-1]}: amortized {tm[-1]/ta[-1]:.0f}, "
              f"fresh {tm[-1]/tf[-1]:.0f}")
        for r in R:
            k = (c, int(r["q"]))
            e = ref.get(k, (float("nan"),) * 2)
            print(f"   q={int(r['q']):4d} N={r['N_max']:>4} kuN={fnum(r['kuN_max']):.1e} fb={r['cheb_fallbacks']} "
                  f"ref={r['cheb_refusals']} uf={r['mnp_underflow']} maxreldiff={fnum(r['max_rel_diff']):.1e} "
                  f"err_cheb(set0)={e[0]:.1e} err_mnp(set0)={e[1]:.1e} t/(qN)={fnum(r['t_per_qN_ns']):.2f}ns "
                  f"t/cell={fnum(r['t_per_cell_ns']):.2f}ns precomp_share={fnum(r['precomp_share']):.2f}")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=6.8, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.4, h_pad=0.3, w_pad=0.6, rect=(0, 0, 1, 0.9))
    fig.savefig(f"{out}/fig_runtime.pdf")
    plt.close(fig)


# =============================================================================
def fig_interval(d, out):
    rows = read(f"{d}/e5_interval.csv")
    modes = [("symmetric", r"symmetric $[-1-s,\,1+s]$", "-", "o"),
             ("left", r"below only $[-1-2s,\,1]$", "--", "s"),
             ("right", r"above only $[-1,\,1+2s]$", "-.", "^")]
    qcol = {20: OI["blue"], 100: OI["verm"]}
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.9))
    print("\n=== E5 interval sensitivity (median over node sets) ===")
    for q in (20, 100):
        for m, lab, ls, mk in modes:
            R = [r for r in rows if int(r["q"]) == q and r["mode"] == m]
            ss = sorted({float(r["s"]) for r in R})
            N = [np.median([int(r["N_raw"]) for r in R if float(r["s"]) == s]) for s in ss]
            K = [np.median([fnum(r["kuN_raw"]) for r in R if float(r["s"]) == s]) for s in ss]
            T = [np.median([fnum(r["t_eval_us"]) for r in R if float(r["s"]) == s]) for s in ss]
            FB = [sum(int(r["eval_fallback"]) for r in R if float(r["s"]) == s) for s in ss]
            RF = [sum(int(r["eval_refused"]) for r in R if float(r["s"]) == s) for s in ss]
            nset = len({r["set"] for r in R})
            axes[0].plot(ss, N, ls=ls, marker=mk, color=qcol[q], mfc="none" if q == 100 else qcol[q],
                         label=f"$q={q}$, {lab}")
            axes[1].semilogy(ss, K, ls=ls, marker=mk, color=qcol[q], mfc="none" if q == 100 else qcol[q])
            print(f"q={q} {m:9s} " + "  ".join(
                f"s={s:g}:N={n:.0f},kuN={k:.1e},t={t:.2f}us,fb={f}/{nset},ref={rf}"
                for s, n, k, t, f, rf in zip(ss, N, K, T, FB, RF)))
    axes[1].axhline(1e-10, color="0.3", lw=0.8, ls=":")
    axes[1].text(0.3, 2e-10, "fallback threshold $10^{-10}$", fontsize=7, color="0.3")
    ss = np.linspace(0, 20, 50)
    axes[1].semilogy(ss, 1e-14 * np.exp(2 * ss), color="0.6", lw=0.7, ls="-")
    axes[1].annotate(r"$\propto e^{2s}=e^{\,b-\max_i x_i}$", xy=(8, 1e-14 * math.exp(16)),
                     xytext=(1.0, 1e0), fontsize=7, color="0.35",
                     arrowprops=dict(arrowstyle="->", color="0.5", lw=0.6))
    axes[1].set_ylim(1e-15, 1e5)
    axes[0].set_xlabel("slack parameter $s$")
    axes[1].set_xlabel("slack parameter $s$")
    axes[0].set_ylabel("Chebyshev terms $N$ on $[a,b]$")
    axes[1].set_ylabel(r"$\kappa u N$ on $[a,b]$")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=6.8, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.4, w_pad=1.0, rect=(0, 0, 1, 0.83))
    fig.savefig(f"{out}/fig_interval_sensitivity.pdf")
    plt.close(fig)


# =============================================================================
def fig_incremental(d, out):
    """IncrementalStack microbenchmark (e6_stack.csv); legacy append dashed grey."""
    rows = read(f"{d}/e6_stack.csv")
    legacy = read(f"{d}/e6_incremental_legacy.csv") if os.path.exists(f"{d}/e6_incremental_legacy.csv") else []
    cs = sorted({float(r["c"]) for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 3.1))
    print("\n=== E6a IncrementalStack microbenchmark (q_max = q+10) ===")
    for j, c in enumerate(cs):
        ax = axes[j]
        R = sorted([r for r in rows if float(r["c"]) == c], key=lambda r: int(r["q"]))
        q = [int(r["q"]) for r in R]
        ax.loglog(q, [fnum(r["t_push_us"]) for r in R], "-o", color=OI["blue"], label="push (append)")
        ax.loglog(q, [fnum(r["t_pop_us"]) for r in R], "-v", color=OI["sky"], label="pop (LIFO removal)")
        ax.loglog(q, [fnum(r["t_batch_amort_us"]) for r in R], "--s", color=OI["green"],
                  label="batch recompute (amortized)")
        ax.loglog(q, [fnum(r["t_mnp_us"]) for r in R], "-.^", color=OI["verm"], label="MNP recompute")
        LR = sorted([r for r in legacy if float(r["c"]) == c and r["maxq_rule"] == "q+10"
                     and r["status"] == "ok"], key=lambda r: int(r["q"]))
        if LR:
            ax.loglog([int(r["q"]) for r in LR], [fnum(r["t_add_us"]) for r in LR], ":", marker="x",
                      color="0.5", label="legacy IncrementalExpDivDiff append")
        ax.text(0.04, 0.95, f"$c={c:g}$", transform=ax.transAxes, va="top")
        ax.set_xlabel("order $q$ before the update")
        if j == 0:
            ax.set_ylabel(r"time per update [$\mu$s]")
        for r in R:
            print(f"c={c:g} q={int(r['q']):4d} Nmax={r['nmax']:>4} N={r['N_top']:>4} redraws={r['init_redraws']} "
                  f"push_refused={fnum(r['push_refused_frac']):.3f} ctor={fnum(r['t_ctor_us']):.3g} "
                  f"init={fnum(r['t_init_us']):.3g} push={fnum(r['t_push_us']):.3g} pop={fnum(r['t_pop_us']):.3g} "
                  f"batch={fnum(r['t_batch_amort_us']):.3g} fresh={fnum(r['t_batch_fresh_us']):.3g} "
                  f"mnp={fnum(r['t_mnp_us']):.3g} batch/push={fnum(r['t_batch_amort_us'])/fnum(r['t_push_us']):.1f} "
                  f"mnp/push={fnum(r['t_mnp_us'])/fnum(r['t_push_us']):.0f} rel={r['rel_diff_stack_batch']}")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=6.8, bbox_to_anchor=(0.5, 1.0))
    ax = axes[2]
    for c, col, mk in ((cs[0], OI["blue"], "o"), (cs[1], OI["verm"], "s")):
        R = sorted([r for r in rows if float(r["c"]) == c], key=lambda r: int(r["q"]))
        q = [int(r["q"]) for r in R]
        ax.loglog(q, [fnum(r["t_ctor_us"]) for r in R], ls="--", marker=mk, color=col, mfc="none",
                  label=f"constructor, $c={c:g}$")
        ax.loglog(q, [fnum(r["t_init_us"]) for r in R], ls="-", marker=mk, color=col,
                  label=f"initialize(), $c={c:g}$")
    ax.set_xlabel("order $q$")
    ax.set_ylabel(r"setup time [$\mu$s]")
    ax.legend(fontsize=6.3, loc="upper left")
    fig.tight_layout(pad=0.4, w_pad=0.6, rect=(0, 0, 1, 0.82))
    fig.savefig(f"{out}/fig_incremental.pdf")
    plt.close(fig)


# =============================================================================
IMPL_LABEL = {"stack_global": "stack, global interval", "stack_anchored": "stack, anchored interval",
              "stack_local": "stack, local interval",
              "batch": "batch (fresh obj.)", "batch_c": "batch (cached obj.)", "mnp": "MNP"}
IMPL_COLOR = {"stack_global": OI["blue"], "stack_anchored": OI["sky"], "stack_local": OI["purple"],
              "batch": OI["green"], "batch_c": OI["yellow"], "mnp": OI["verm"]}
IMPL_HATCH = {"stack_global": "", "stack_anchored": "////", "stack_local": "\\\\",
              "batch": "", "batch_c": "....", "mnp": ""}


def fig_qmc(d, out, figdir_tex):
    T = read(f"{d}/e6_qmc_timing.csv")
    S = read(f"{d}/e6_qmc_stats.csv")
    H = read(f"{d}/e6_qmc_hist.csv")
    A = read(f"{d}/e6_qmc_agreement.csv") if os.path.exists(f"{d}/e6_qmc_agreement.csv") else []
    keys = [(int(r["L"]), float(r["beta"])) for r in S]
    t = {(int(r["L"]), float(r["beta"]), r["impl"], r["mode"]): r for r in T}

    fig = plt.figure(figsize=(FULL, 4.3))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15])
    # (a) q histograms for L=32 at several beta
    ax = fig.add_subplot(gs[0, 0])
    for beta, col, ls in ((1.0, OI["blue"], "-"), (4.0, OI["green"], "--"), (16.0, OI["verm"], "-.")):
        hv = [(int(float(r["value"])), int(r["count"])) for r in H
              if int(r["L"]) == 32 and float(r["beta"]) == beta and r["quantity"] == "q"]
        if not hv:
            continue
        hv.sort()
        tot = sum(c for _, c in hv)
        ax.step([v for v, _ in hv], [c / tot for _, c in hv], where="mid", color=col, ls=ls,
                label=fr"$\beta={beta:g}$")
    ax.set_xlabel("order $q$")
    ax.set_ylabel("fraction of moves")
    ax.legend(title="$L=32$", fontsize=6.5, title_fontsize=6.5)
    # (b) spread histogram, L=32 beta=4
    ax = fig.add_subplot(gs[0, 1])
    for k, (beta, col, hatch) in enumerate(((1.0, OI["blue"], ""), (4.0, OI["green"], "////"),
                                            (16.0, OI["verm"], ".."))):
        hv = sorted([(float(r["value"]), int(r["count"])) for r in H
                     if int(r["L"]) == 32 and float(r["beta"]) == beta and r["quantity"] == "spread"])
        if not hv:
            continue
        tot = sum(c for _, c in hv)
        ax.bar([v / beta + (k - 1) * 0.3 for v, _ in hv], [c / tot for _, c in hv], width=0.28,
               color=col, hatch=hatch, label=fr"$\beta={beta:g}$", edgecolor="0.25", linewidth=0.4)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xlabel(r"node spread $(\max_i x_i-\min_i x_i)/(2\beta)$")
    ax.set_ylabel("fraction of moves")
    ax.legend(title="$L=32$", fontsize=6.5, title_fontsize=6.5)
    # (c) <q> vs beta
    ax = fig.add_subplot(gs[0, 2])
    for L, col, mk in ((8, OI["blue"], "o"), (16, OI["green"], "s"), (32, OI["verm"], "^")):
        R = sorted([r for r in S if int(r["L"]) == L], key=lambda r: float(r["beta"]))
        b = [float(r["beta"]) for r in R]
        ax.errorbar(b, [fnum(r["q_mean"]) for r in R],
                    yerr=[[fnum(r["q_mean"]) - fnum(r["q_p05"]) for r in R],
                          [fnum(r["q_p95"]) - fnum(r["q_mean"]) for r in R]],
                    color=col, marker=mk, capsize=2, lw=1, label=f"$L={L}$")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\langle q\rangle$ (bars: 5-95% range)")
    ax.legend(fontsize=6.5)
    # (d) time per move, naive mode, grouped bars
    ax = fig.add_subplot(gs[1, :])
    impls = ["stack_global", "stack_anchored", "stack_local", "batch", "batch_c", "mnp"]
    w = 0.135
    x = np.arange(len(keys))
    for k, im in enumerate(impls):
        vals = [fnum(t[(L, b, im, "naive")]["t_move_us"]) if (L, b, im, "naive") in t else np.nan
                for (L, b) in keys]
        ax.bar(x + (k - 2.5) * w, vals, w * 0.92, color=IMPL_COLOR[im], hatch=IMPL_HATCH[im],
               edgecolor="0.25" if IMPL_HATCH[im] else "white", linewidth=0.4, label=IMPL_LABEL[im])
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"$L={L}$\n$\\beta={b:g}$" for (L, b) in keys], fontsize=7)
    ax.set_ylabel(r"wall time per move [$\mu$s]")
    ax.legend(ncol=3, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="x", visible=False)
    fig.tight_layout(pad=0.4, h_pad=0.6, w_pad=0.8)
    fig.savefig(f"{out}/fig_qmc_benchmark.pdf")
    plt.close(fig)

    # ---- print summary --------------------------------------------------
    print("\n=== E6b QMC-like chain ===")
    for r in S:
        L, b = int(r["L"]), float(r["beta"])
        line = (f"L={L:2d} beta={b:4g} c={r['c']:>4} <q>={fnum(r['q_mean']):6.2f} [{r['q_p05']},{r['q_p95']}] "
                f"qmax={r['q_max']} spread={fnum(r['spread_mean']):.3f} [{r['spread_p05']},{r['spread_p95']}] "
                f"near_b={fnum(r['frac_within1_of_b']):.3f} at_b={fnum(r['frac_at_b']):.3f} "
                f"acc={r['acc_add']}/{r['acc_remove']}/{r['acc_classical']} "
                f"batchN={r['batch_N_mean']}({r['batch_N_max']}) kuN med={r['batch_kuN_median']} "
                f"p95={r['batch_kuN_p95']} max={r['batch_kuN_max']} batch_ref={r['batch_refusals']} "
                f"mnp_nonnormal={r['mnp_nonnormal']}")
        print(line)
        for mode in ("naive", "stack"):
            s = "   " + mode + ": " + "  ".join(
                f"{im}={fnum(t[(L, b, im, mode)]['t_move_us']):.3g}us"
                f"(thr={t[(L, b, im, mode)]['throws']},fb={t[(L, b, im, mode)]['fallback_moves']},"
                f"clMNP={t[(L, b, im, mode)].get('classical_mnp', '')},ok={t[(L, b, im, mode)]['checksum_ok']},"
                f"Nmax={t[(L, b, im, mode)].get('nmax', t[(L, b, im, mode)].get('max_order'))},"
                f"rebuilds={t[(L, b, im, mode)].get('interval_rebuilds', '')},"
                f"a={t[(L, b, im, mode)].get('a_final', '')})"
                for im in impls if (L, b, im, mode) in t)
            print(s)
    for r in A:
        if int(r["decision_disagreements"]) > 0 or fnum(r["max_rel_diff_ratio"]) > 1e-10:
            print("   AGREEMENT NOTE:", r)
    print("max rel diff of ratios per impl (over all params):")
    agg = defaultdict(float)
    dis = defaultdict(int)
    for r in A:
        agg[(r["impl"], r["mode"])] = max(agg[(r["impl"], r["mode"])], fnum(r["max_rel_diff_ratio"]))
        dis[(r["impl"], r["mode"])] += int(r["decision_disagreements"])
    for k in sorted(agg):
        print(f"   {k}: max rel diff {agg[k]:.2e}, disagreements {dis[k]}")

    # ---- LaTeX table ------------------------------------------------------
    def tm(L, b, im, mode):
        return fnum(t[(L, b, im, mode)]["t_move_us"])

    lines = [
        r"% Generated by experiments/performance/make_figures.py from e6_qmc_*.csv",
        r"\begin{tabular}{rrrr rrrrrr rr}",
        r"\toprule",
        r" & & & & \multicolumn{6}{c}{time/move [$\mu$s]} & \multicolumn{2}{c}{speedup of best stack} \\",
        r"\cmidrule(lr){5-10}\cmidrule(lr){11-12}",
        r"$L$ & $\beta$ & $\langle q\rangle$ & $\langle\text{spread}\rangle$ & stack glob. & stack anch. & stack loc. & batch & batch cached & MNP & vs.\ best batch & vs.\ MNP \\",
        r"\midrule",
    ]
    for mode, title in (("naive", "recompute proposed set"), ("stack", "with prefix-value stack")):
        lines.append(r"\multicolumn{12}{l}{\emph{" + title + r"}} \\")
        for r in S:
            L, b = int(r["L"]), float(r["beta"])
            sg, sa = tm(L, b, "stack_global", mode), tm(L, b, "stack_anchored", mode)
            sl = tm(L, b, "stack_local", mode)
            tb, tc, tmn = tm(L, b, "batch", mode), tm(L, b, "batch_c", mode), tm(L, b, "mnp", mode)
            best = min(sg, sa, sl)
            lines.append(f"{L} & {b:g} & {fnum(r['q_mean']):.1f} & {fnum(r['spread_mean']):.2f} & "
                         f"{sg:.2f} & {sa:.2f} & {sl:.2f} & {tb:.2f} & {tc:.2f} & {tmn:.2f} & "
                         f"{min(tb, tc)/best:.2f} & {tmn/best:.1f} \\\\")
        if mode == "naive":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    with open(f"{figdir_tex}/table_qmc.tex", "w") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    d = sys.argv[1]
    out = sys.argv[2]
    os.makedirs(out, exist_ok=True)
    fig_runtime(d, out)
    fig_interval(d, out)
    if os.path.exists(f"{d}/e6_stack.csv"):
        fig_incremental(d, out)
    if os.path.exists(f"{d}/e6_qmc_stats.csv"):
        fig_qmc(d, out, out)
