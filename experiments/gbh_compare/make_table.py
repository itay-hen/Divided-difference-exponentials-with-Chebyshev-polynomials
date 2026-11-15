"""LaTeX table from gbh_vs_stack.csv (Table 6.2 of the manuscript).
Usage: python3 -I make_table.py <gbh_vs_stack.csv> <table_gbh.tex>"""
import csv, sys
src, dst = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(src)))
L = [f"% Generated from {src.split('/')[-1]} (experiments/gbh_compare/make_table.py)",
     r"\begin{tabular}{rr r rr r r}", r"\toprule",
     r" & & & \multicolumn{2}{c}{GBH~\cite{GuptaBarashHen2020CPC} [$\mu$s]} & & \\",
     r"\cmidrule(lr){4-5}",
     r"$c$ & $q$ & $s$ & tuned & released & stack [$\mu$s] & speedup \\", r"\midrule"]
for r in rows:
    ts, tt, td = float(r["t_stack_ns"]) / 1e3, float(r["t_gbh_tuned_ns"]) / 1e3, float(r["t_gbh_default_ns"]) / 1e3
    L.append(f"{float(r['c']):g} & {r['q']} & {r['s_gbh']} & {tt:.1f} & {td:.0f} & {ts:.2f} & {tt/ts:.0f} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(dst, "w").write("\n".join(L) + "\n")
