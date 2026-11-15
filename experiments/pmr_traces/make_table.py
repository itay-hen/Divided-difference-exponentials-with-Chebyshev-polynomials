"""LaTeX table from replay CSV (PMR-QMC traces).
Usage: python3 -I make_table.py <replay.csv> <table_pmr.tex>"""
import csv, sys
src, dst = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(src)))
name = {"heis": "Heisenberg", "tfim": "TFIM"}
def label(t):
    m, b = t.replace("trace_", "").replace(".txt", "").split("_b")
    model = name[m[:4]]; size = m[4:].replace("x", r"\times")
    return f"{model} ${size}$", b
L = [f"% Generated from {src.split('/')[-1]} (experiments/pmr_traces/make_table.py)",
     r"\begin{tabular}{lr rrr rrr rr}", r"\toprule",
     r" & & & & & \multicolumn{3}{c}{time per query [$\mu$s]} & \multicolumn{2}{c}{fallback [\%]} \\",
     r"\cmidrule(lr){6-8}\cmidrule(lr){9-10}",
     r"model & $\beta$ & $\langle q\rangle$ & $\langle c\rangle$ & ops/query & GBH & batch & stack & batch & stack \\",
     r"\midrule"]
for r in rows:
    m, b = label(r["trace"]); nq = float(r["queries"])
    us = lambda k: float(r[k]) * 1e3 / nq
    L.append(f"{m} & {b} & {float(r['mean_q']):.0f} & {float(r['mean_c']):.0f} & "
             f"{float(r['ops'])/nq:.1f} & {us('t_gbh_ms'):.1f} & {us('t_batch_ms'):.1f} & "
             f"{us('t_stack_ms'):.1f} & {100*float(r['batch_fallback']):.2g} & "
             f"{100*float(r['stack_fallback']):.2g} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(dst, "w").write("\n".join(L) + "\n")
