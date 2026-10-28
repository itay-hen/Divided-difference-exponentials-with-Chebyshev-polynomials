#!/usr/bin/env python3
"""
e2_analyze.py -- Experiment E2 post-processing.

Inputs (written by e2_accuracy and mp_reference.py):
    accuracy_e2_nodes.csv, accuracy_e2_results.csv, accuracy_e2_reference.csv
Outputs:
    accuracy_e2_cases.csv          one row per (instance, ordering) with all
                                   relative errors against the mpmath reference
    table_accuracy.tex             booktabs tabular (no table wrapper), library
                                   default (nodes processed in nonincreasing order)
    fig_error_vs_indicator.pdf     observed error vs kappa*u*N (default)
    fig_ordering_spread.pdf        caller-order ("unsorted") vs default results
and prints the summary numbers quoted in the report.

Run variants (see e2_accuracy.cpp):
    prod      default evaluate_scaled(): returned value or refusal
    ab        default processing on [-c,c], threshold disabled (forced value)
    tight     default processing on [min x, max x], threshold disabled
    unsorted  as ab, but set_sort_nodes(false): recurrence in the caller's order

Usage:
    python3 -I e2_analyze.py --data <dir with the three CSVs> --figs <fig dir>
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np
import mpmath
from mpmath import mp, mpf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_style  # noqa: E402

mp.dps = 40
U = 2.0 ** -53                 # unit roundoff of double
DBL_MIN = 2.2250738585072014e-308
THRESH = 1e-10                 # default threshold on kappa*u*N
ERR_FLOOR = 1e-18              # plotting floor for errors that are exactly 0
CLIP = 1e24                    # plotting ceiling for (forced) errors
VARIANTS = ("prod", "ab", "tight", "unsorted")
ORDERINGS = ["generated", "ascending", "descending", "random1", "random2", "random3",
             "alternating"]

CONFIG_LABEL = {
    "uniform": "uniform",
    "cluster_left": "clustered at $a$",
    "cluster_right": "clustered at $b$",
    "chebyshev": "Chebyshev (jittered)",
    "near_coincident": r"nearly coincident ($10^{-8}c$)",
    "near_coincident_far": "nearly coinc.\\ + far node",
    "repeated": "repeated (mult.\\ $\\approx 5$)",
    "two_clusters": "two end clusters",
    "equispaced": "equispaced",
}
CONFIGS = list(CONFIG_LABEL)


def relerr(mant, expo, ref):
    v = mpmath.ldexp(mpf(mant), int(expo))
    return float(abs(v - ref) / abs(ref))


def fmt(v):
    if v is None or np.isnan(v):
        return "--"
    m, e = f"{v:.1e}".split("e")
    return f"{m}e{int(e)}"


def load(data):
    ref = {}
    with open(os.path.join(data, "accuracy_e2_reference.csv")) as f:
        for r in csv.DictReader(f):
            ref[int(r["instance"])] = (mpf(r["value"]), float(r["log10_abs"]))
    spread = {}
    with open(os.path.join(data, "accuracy_e2_nodes.csv")) as f:
        for r in csv.DictReader(f):
            x = [float(t) for t in r["nodes"].split()]
            spread[int(r["instance"])] = max(x) - min(x)
    rows = []
    with open(os.path.join(data, "accuracy_e2_results.csv")) as f:
        for r in csv.DictReader(f):
            i = int(r["instance"])
            R, l10 = ref[i]
            q, c = int(r["q"]), float(r["c"])
            o = dict(instance=i, q=q, c=c, config=r["config"], rep=int(r["rep"]),
                     ordering=r["ordering"], log10_ref=l10)
            for p in VARIANTS:
                st = r[p + "_status"]
                o[p + "_status"] = st
                o[p + "_N"] = int(r[p + "_N"])
                o[p + "_ku"] = float(r[p + "_ku"])
                o[p + "_kuN"] = float(r[p + "_kuN"])
                o[p + "_err"] = (relerr(r[p + "_mant"], r[p + "_exp"], R)
                                 if st == "ok" else float("nan"))
                o[p + "_key"] = (r[p + "_mant"], r[p + "_exp"], r[p + "_N"])
            # which interval produced the returned (default) value
            if o["prod_status"] != "ok":
                o["interval"] = "-"
            elif o["prod_key"] == o["ab_key"]:
                o["interval"] = "tight(prescreen)" if c > 100 * spread[i] else "[a,b]"
            else:
                o["interval"] = "tight(retry)"
            m = float(r["mnp"])
            if not np.isfinite(m) or m == 0.0:
                o["mnp_status"], o["mnp_err"] = "underflow", float("nan")
            else:
                o["mnp_status"] = "subnormal" if abs(m) < DBL_MIN else "ok"
                o["mnp_err"] = float(abs(mpf(m) - R) / abs(R))
            rows.append(o)
    return rows


def write_cases(rows, path):
    hdr = ["q", "c", "config", "rep", "ordering", "log10_ref",
           "cheb_status", "cheb_rel_err", "cheb_N", "cheb_kappa_u", "cheb_kappa_u_N",
           "cheb_interval",
           "refused_forced_rel_err", "refused_forced_N", "refused_forced_kappa_u_N",
           "ab_forced_rel_err", "ab_forced_N", "ab_forced_kappa_u_N",
           "tight_forced_rel_err", "tight_forced_N", "tight_forced_kappa_u_N",
           "unsorted_rel_err", "unsorted_N", "unsorted_kappa_u_N",
           "mnp_status", "mnp_rel_err"]
    cols = ["q", "c", "config", "rep", "ordering", "log10_ref",
            "prod_status", "prod_err", "prod_N", "prod_ku", "prod_kuN", "interval",
            "forced_err", "forced_N", "forced_kuN",
            "ab_err", "ab_N", "ab_kuN", "tight_err", "tight_N", "tight_kuN",
            "unsorted_err", "unsorted_N", "unsorted_kuN", "mnp_status", "mnp_err"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        for o in rows:
            refused = o["prod_status"] == "refused"
            # forced value of a refused case = the rejected tight-interval value
            o["forced_err"] = o["tight_err"] if refused else ""
            o["forced_N"] = o["tight_N"] if refused else ""
            o["forced_kuN"] = o["tight_kuN"] if refused else ""
            vals = []
            for k in cols:
                v = o[k]
                if isinstance(v, float):
                    v = "" if np.isnan(v) else f"{v:.4g}"
                vals.append(v)
            w.writerow(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--figs", required=True)
    a = ap.parse_args()

    rows = load(a.data)
    write_cases(rows, os.path.join(a.data, "accuracy_e2_cases.csv"))
    cs = sorted({o["c"] for o in rows})
    qs = sorted({o["q"] for o in rows})

    # ================================================= default (sorted) results
    print("== DEFAULT (nodes processed in nonincreasing order) ==")
    print("status:", {s: sum(o["prod_status"] == s for o in rows)
                      for s in ("ok", "refused", "noconv")})
    print("intervals:", {k: sum(o["interval"] == k for o in rows)
                         for k in ("[a,b]", "tight(prescreen)", "tight(retry)", "-")})
    ok = [o for o in rows if o["prod_status"] == "ok"]
    print("largest accepted errors:")
    for o in sorted(ok, key=lambda o: -o["prod_err"])[:5]:
        print(f"  q={o['q']:3d} c={o['c']:5g} {o['config']:20s} {o['ordering']:11s} "
              f"err={o['prod_err']:.3e} kuN={o['prod_kuN']:.2e} N={o['prod_N']} {o['interval']}")
    print("accepted with err > 1e-10:", sum(o["prod_err"] > 1e-10 for o in ok))
    rat = [(o["prod_err"] / o["prod_kuN"], o) for o in ok if o["prod_kuN"] > 0]
    r, o = max(rat, key=lambda t: t[0])
    print(f"max err/(kuN), accepted: {r:.3g} (q={o['q']}, c={o['c']:g}, {o['config']}, "
          f"{o['ordering']}); #>1: {sum(t[0] > 1 for t in rat)} of {len(rat)}")
    fab = [(o["ab_err"] / o["ab_kuN"], o) for o in rows
           if o["ab_status"] == "ok" and 0 < o["ab_kuN"] <= 1e-2]
    r, o = max(fab, key=lambda t: t[0])
    print(f"max err/(kuN), forced ab with kuN<=1e-2: {r:.3g} (q={o['q']}, c={o['c']:g}, "
          f"{o['config']}, {o['ordering']}, kuN={o['ab_kuN']:.2e}); n={len(fab)}")
    refd = [o for o in rows if o["prod_status"] == "refused"]
    fe = [o["tight_err"] for o in refd if not np.isnan(o["tight_err"])]
    if fe:
        print(f"refused: {len(refd)}; rejected (tight) value error: median "
              f"{np.median(fe):.2e}, max {max(fe):.2e}; #<1e-10: {sum(e < 1e-10 for e in fe)}")
    # ordering-dependence that remains with sorting (refusal and error)
    by_inst = defaultdict(list)
    for o in rows:
        by_inst[o["instance"]].append(o)
    mixed = sum(1 for lst in by_inst.values()
                if len({o["prod_status"] for o in lst}) > 1)
    spr = [max(max(o["prod_err"], U) for o in lst) / min(max(o["prod_err"], U) for o in lst)
           for lst in by_inst.values() if all(o["prod_status"] == "ok" for o in lst)]
    print(f"instances with ordering-dependent refusal: {mixed}; "
          f"max/min err over orderings (all accepted): max {max(spr):.3g}")

    print("\n== MNP ==")
    print({s: sum(o["mnp_status"] == s for o in rows) for s in ("ok", "subnormal", "underflow")})
    mok = [o for o in rows if o["mnp_status"] == "ok"]
    print(f"  max MNP error (normal results): {max(o['mnp_err'] for o in mok):.3e}")
    both = [o for o in ok if o["mnp_status"] == "ok"]
    print(f"  Chebyshev error <= MNP error in {sum(o['prod_err'] <= o['mnp_err'] for o in both)}"
          f"/{len(both)} cases")

    # ------------------------------------------------------------ LaTeX table
    lines = [r"\begin{tabular}{l" + "rrr" * len(cs) + "}", r"\toprule"]
    lines.append(" & " + " & ".join(
        rf"\multicolumn{{3}}{{c}}{{$c={c:g}$}}" for c in cs) + r" \\")
    lines.append("".join(rf"\cmidrule(lr){{{2 + 3 * k}-{4 + 3 * k}}}" for k in range(len(cs))))
    lines.append("configuration & " + " & ".join(["Cheb.", "MNP", "ref."] * len(cs)) + r" \\")
    lines.append(r"\midrule")
    for cfg in CONFIGS:
        cells = [CONFIG_LABEL[cfg]]
        for c in cs:
            sub = [o for o in rows if o["config"] == cfg and o["c"] == c]
            ce = [o["prod_err"] for o in sub if o["prod_status"] == "ok"]
            me = [o["mnp_err"] for o in sub if o["mnp_status"] == "ok"]
            nref = sum(o["prod_status"] != "ok" for o in sub)
            cells += [fmt(max(ce) if ce else None), fmt(max(me) if me else None),
                      f"{nref}/{len(sub)}"]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    hdr = ("% Generated by experiments/accuracy/e2_analyze.py.\n"
           f"% Entries: max relative error vs. the mpmath reference over q in {qs},\n"
           "% 10 random instances (1 for equispaced) and 7 node orderings.\n"
           "% Cheb. = values returned (accepted) by ExpDivDiff::evaluate_scaled with the\n"
           "% library defaults (tol 1e-14, nodes processed in nonincreasing order,\n"
           "% threshold kappa*u*N <= 1e-10); MNP = double-precision McCurdy-Ng-Parlett,\n"
           "% values >= DBL_MIN only (MNP underflows to 0 for every q = 300 case);\n"
           "% ref. = refused (exception) / total cases.\n")
    with open(os.path.join(a.figs, "table_accuracy.tex"), "w") as f:
        f.write(hdr + "\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))

    # ================================================= unsorted (caller order)
    print("\n== UNSORTED (caller order, [-c,c], threshold off) ==")
    uok = [o for o in rows if o["unsorted_status"] == "ok"]
    uacc = [o for o in uok if o["unsorted_kuN"] <= THRESH]      # would pass the threshold
    print(f"runs: {len(uok)}; with kuN <= 1e-10 (would be accepted): {len(uacc)}")
    print("max error per ordering (kuN<=1e-10):")
    for od in ORDERINGS:
        l = [o for o in uacc if o["ordering"] == od]
        w = max(l, key=lambda o: o["unsorted_err"])
        print(f"  {od:11s} {w['unsorted_err']:.3e}  (q={w['q']}, c={w['c']:g}, {w['config']}, "
              f"kuN={w['unsorted_kuN']:.2e})")
    ur = [o["unsorted_err"] / o["unsorted_kuN"] for o in uacc if o["unsorted_kuN"] > 0]
    print(f"#err/(kuN) > 10: {sum(x > 10 for x in ur)} of {len(ur)}; max {max(ur):.3g}")
    print(f"#err > 1e-10: {sum(o['unsorted_err'] > 1e-10 for o in uacc)}")
    for od in ORDERINGS:
        n10 = sum(1 for o in uacc if o["ordering"] == od and o["unsorted_kuN"] > 0
                  and o["unsorted_err"] / o["unsorted_kuN"] > 10)
        print(f"  {od:11s} #err/(kuN)>10: {n10}")

    # ================================================= figures
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator
    plot_style.apply()
    P = plot_style.PALETTE

    # ---- error vs indicator (default)
    fig, ax = plt.subplots(figsize=(plot_style.SINGLE_WIDTH + 0.4, 3.1))
    rej = np.array([(o["ab_kuN"], min(max(o["ab_err"], ERR_FLOOR), CLIP)) for o in rows
                    if o["ab_status"] == "ok" and o["ab_kuN"] > THRESH])
    acc = np.array([(o["prod_kuN"], max(o["prod_err"], ERR_FLOOR)) for o in ok
                    if o["prod_kuN"] > 0])
    nclip = int(np.sum(rej[:, 1] >= CLIP))
    print(f"\nindicator figure: {len(acc)} accepted, {len(rej)} forced; "
          f"{nclip} forced errors clipped at {CLIP:g}; "
          f"{int(np.sum(acc[:, 1] <= ERR_FLOOR))} accepted errors at floor")
    ax.loglog(rej[:, 0], rej[:, 1], "x", color="0.55", ms=2.5, mew=0.5, alpha=0.7,
              label=r"rejected ($\kappa uN>10^{-10}$), value forced")
    ax.loglog(acc[:, 0], acc[:, 1], "o", color=P[0], ms=2.2, mew=0, alpha=0.6,
              label="accepted (returned) value")
    lo, hi = 1e-17, 1e24
    ax.plot([lo, hi], [lo, hi], "k-", lw=0.8, label=r"error $=\kappa uN$")
    ax.axvline(THRESH, color="0.4", lw=0.6, ls=":")
    ax.set_xlim(lo, hi)
    ax.set_ylim(ERR_FLOOR / 3, CLIP * 5)
    yt = [1e-16, 1e-12, 1e-8, 1e-4, 1, 1e4, 1e8, 1e12, 1e16, 1e20, CLIP]
    ax.yaxis.set_major_locator(FixedLocator(yt))
    ax.set_yticklabels([rf"$10^{{{int(np.log10(t))}}}$" for t in yt[:-1]]
                       + [r"$\geq\!10^{24}$"])
    ax.minorticks_off()
    ax.set_xlabel(r"error indicator $\kappa\,u\,N$")
    ax.set_ylabel("observed relative error (vs. mpmath)")
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor="none",
              fontsize=6.5, handletextpad=0.2, borderaxespad=0.2, markerscale=1.6)
    fig.savefig(os.path.join(a.figs, "fig_error_vs_indicator.pdf"))
    plt.close(fig)

    # ---- ordering effect: unsorted (caller order) vs default
    fig, axs = plt.subplots(1, 2, figsize=(plot_style.FULL_WIDTH, 2.7))
    oc = {"generated": ("v", P[4]), "ascending": ("s", P[3]), "descending": ("o", P[0]),
          "random1": ("^", P[2]), "random2": ("<", P[5]), "random3": (">", "#999933"),
          "alternating": ("D", P[1])}
    c0 = 100.0
    print(f"\nmax error vs q at c={c0:g}:")
    for od in ORDERINGS:
        ys = [max([o["unsorted_err"] for o in uacc if o["ordering"] == od and o["q"] == q
                   and o["c"] == c0] or [np.nan]) for q in qs]
        print(f"  unsorted {od:11s}", " ".join(f"{v:9.2e}" for v in ys))
        m, col = oc[od]
        axs[0].semilogy(qs, np.maximum(ys, ERR_FLOOR), "-", marker=m, color=col, ms=3.2,
                        lw=0.8, label=f"caller order: {od}")
    ys = [max([o["prod_err"] for o in ok if o["q"] == q and o["c"] == c0] or [np.nan])
          for q in qs]
    print(f"  default (sorted)     ", " ".join(f"{v:9.2e}" for v in ys))
    axs[0].semilogy(qs, ys, "k-", lw=1.6, marker="*", ms=5,
                    label="default (sorted), all orderings")
    axs[0].axhline(1e-10, color="0.5", lw=0.6, ls=":")
    axs[0].set_xscale("log")
    axs[0].set_xticks(qs)
    axs[0].set_xticklabels([str(q) for q in qs])
    axs[0].minorticks_off()
    axs[0].set_xlabel("order $q$")
    axs[0].set_ylabel(f"max rel. error, $c={c0:g}$\n" r"(cases with $\kappa uN\leq10^{-10}$)")
    axs[0].set_ylim(1e-16, 1e28)
    axs[0].legend(fontsize=5.8, frameon=False, loc="upper left", ncol=2,
                  columnspacing=0.8, handlelength=1.6)

    # right: complementary CDF of err/(kappa u N)
    def ccdf(v, **kw):
        v = np.sort(np.maximum(np.asarray(v), 1e-6))
        y = 1.0 - np.arange(len(v)) / len(v)
        axs[1].loglog(v, y, drawstyle="steps-post", **kw)
    ccdf([o["prod_err"] / o["prod_kuN"] for o in ok if o["prod_kuN"] > 0],
         color="k", lw=1.5, label="default (sorted), accepted")
    ccdf(ur, color=P[3], lw=1.2, label=r"caller order, $\kappa uN\leq10^{-10}$")
    axs[1].axvline(1, color="0.5", lw=0.6, ls=":")
    axs[1].set_xlabel(r"observed error $/\ \kappa uN$")
    axs[1].set_ylabel("fraction of cases with ratio $\\geq x$")
    axs[1].set_xlim(1e-4, 1e36)
    axs[1].set_ylim(5e-5, 1.5)
    axs[1].legend(fontsize=6.5, frameon=False, loc="lower left")
    fig.tight_layout(pad=0.4, w_pad=1.5)
    fig.savefig(os.path.join(a.figs, "fig_ordering_spread.pdf"))
    plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main()
