"""e9_analyze.py -- calibration of the scaling-and-squaring indicator.

Inputs : e9_out.csv (e9_squaring), any number of reference CSVs
         (instance,value[,...]) and optionally families CSV (instance,family)
Prints, per family and overall:
  - max relative error of the squaring path in working precision (prec 1)
    and in double-double (prec 2), and of the default evaluate_scaled;
  - max of err1 / (2^s u)  -> the constant C of the indicator 2^s u C;
  - max of err / indicator for each path;
  - how the default call chose (series, squaring in double, squaring in dd).

Usage: python3 e9_analyze.py OUT.csv REF1.csv [REF2.csv ...] [--families FAM.csv]
         [--csv SUMMARY.csv]
"""
import argparse, csv, math, sys
from collections import defaultdict
import mpmath as mp

sys.set_int_max_str_digits(0)
ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("refs", nargs="+")
ap.add_argument("--families"); ap.add_argument("--csv")
a = ap.parse_args()
mp.mp.dps = 60
U = 2.0 ** -52
ref = {}
for f in a.refs:
    for r in csv.DictReader(open(f)):
        ref[r["instance"]] = mp.mpf(r["value"])
fam = defaultdict(lambda: "e7")
if a.families:
    for r in csv.DictReader(open(a.families)):
        fam[r["instance"]] = r["family"]


def err(m, x, R):
    if m in ("nan", ""):
        return float("nan")            # path not run (e.g. double-double skipped)
    if not math.isfinite(float(m)) or float(m) == 0.0:
        return float("inf")
    v = mp.mpf(m) * mp.mpf(2) ** int(x)
    return float(abs((v - R) / R))


stats = defaultdict(lambda: defaultdict(float))
choice = defaultdict(lambda: defaultdict(int))
rows_out = []
for r in csv.DictReader(open(a.out)):
    i = r["instance"]
    if i not in ref:
        continue
    R = ref[i]; F = fam[i]; s = int(r["s"])
    e1, e2, ed = err(r["m1"], r["x1"], R), err(r["m2"], r["x2"], R), err(r["md"], r["xd"], R)
    est1, est2, estd = float(r["est1"]), float(r["est2"]), float(r["estdef"])
    for key, val in (("err1", e1), ("err2", e2), ("errdef", ed),
                     ("C1", e1 / (2 ** s * U)), ("err1/est1", e1 / est1),
                     ("err2/est2", e2 / est2 if math.isfinite(est2) else float("nan")),
                     ("errdef/estdef", ed / estd)):
        for g in (F, "ALL"):
            if not math.isnan(val):
                stats[g][key] = max(stats[g][key], val)
    sq, pr = int(r["sqdef"]), int(r["precdef"])
    path = "failed" if not math.isfinite(ed) else ("series" if sq == 0 else ("squaring-dd" if pr == 2 else "squaring"))
    for g in (F, "ALL"):
        choice[g][path] += 1
    rows_out.append([i, F, r["q"], r["c"], s, f"{e1:.3e}", f"{e2:.3e}", f"{ed:.3e}", path])

for g in sorted(stats):
    st = stats[g]
    print(f"[{g}] err1 {st['err1']:.2e}  err2 {st['err2']:.2e}  errdef {st['errdef']:.2e}  "
          f"max err1/(2^s u) {st['C1']:.2f}  err1/est1 {st['err1/est1']:.3f}  "
          f"err2/est2 {st['err2/est2']:.3f}  errdef/estdef {st['errdef/estdef']:.3f}  "
          f"paths {dict(choice[g])}")
if a.csv:
    with open(a.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance", "family", "q", "c", "s", "err1", "err2", "errdef", "default_path"])
        w.writerows(rows_out)
