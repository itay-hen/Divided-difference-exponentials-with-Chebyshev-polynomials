"""e9_make.py -- node sets and exact references for the squaring calibration.

Families (instance ranges):
  A  equispaced on [-c, c], c = 2^k, q a power of two (nodes exact doubles);
     exp[x_0..x_q] = e^{x_0} (e^h - 1)^q / (q! h^q), h = 2c/q
  B  cluster + far node: q nodes at -c, one at +c;
     exp[a x q, b] = (e^b - sum_{k<q} e^a (b-a)^k / k!) / (b-a)^q
  C  off-centre random: x uniform on [o - c, o + c] with both ends
     included, o = 0.37 c + 0.1234 (so that x - d rounds); references by
     mp_reference.py (written as a separate node file).
Closed forms are evaluated at 300 and 600 digits and must agree to 1e-30.

Usage: python3 e9_make.py OUTDIR
  writes OUTDIR/e9_nodes.csv (A, B, C), OUTDIR/e9_ref_closed.csv (A, B),
  OUTDIR/e9_nodes_C.csv (C only, for mp_reference.py)
"""
import csv, os, random, sys
import mpmath as mp
sys.set_int_max_str_digits(0)

out = sys.argv[1]
rows, refs, rowsC = [], [], []
inst = 0


def closed_eq(c, q, dps):
    with mp.workdps(dps):
        h = mp.mpf(2 * c) / q
        return mp.exp(-mp.mpf(c)) * mp.expm1(h) ** q / (mp.factorial(q) * h ** q)


def closed_cf(c, q, dps):
    with mp.workdps(dps):
        a, b = mp.mpf(-c), mp.mpf(c)
        L = b - a
        t = mp.mpf(0); term = mp.mpf(1)
        for k in range(q):
            t += term
            term = term * L / (k + 1)
        # e^b - e^a T_{q-1}(L) = e^a (e^L - T_{q-1}(L)); use the tail series
        # when it is small, to avoid cancellation.
        tail = mp.exp(L) - t
        return mp.exp(a) * tail / L ** q


for k in (4, 7, 10, 13, 16, 20, 24, 28):
    c = float(2 ** k)
    for q in (2, 8, 32, 128):
        h = 2 * c / q
        x = [-c + j * h for j in range(q + 1)]
        v1, v2 = closed_eq(c, q, 300), closed_eq(c, q, 600)
        assert abs(v1 - v2) <= mp.mpf("1e-30") * abs(v2)
        rows.append((inst, "A", x)); refs.append((inst, mp.nstr(v2, 35))); inst += 1
        x = [-c] * q + [c]
        v1, v2 = closed_cf(c, q, 300), closed_cf(c, q, 600)
        assert abs(v1 - v2) <= mp.mpf("1e-30") * abs(v2), (c, q)
        rows.append((inst, "B", x)); refs.append((inst, mp.nstr(v2, 35))); inst += 1

rng = random.Random(20261122)
for c in (3000.0, 10000.0, 30000.0):
    o = 0.37 * c + 0.1234
    for q in (5, 20, 50):
        for t in range(4):
            x = [o - c] + [o + rng.uniform(-c, c) for _ in range(q - 1)] + [o + c]
            rows.append((inst, "C", x)); rowsC.append((inst, x)); inst += 1

with open(os.path.join(out, "e9_nodes.csv"), "w") as f:
    f.write("instance,nodes\n")
    for i, fam, x in rows:
        f.write(f"{i}," + " ".join(f"{v:.17g}" for v in x) + "\n")
with open(os.path.join(out, "e9_nodes_C.csv"), "w") as f:
    f.write("instance,nodes\n")
    for i, x in rowsC:
        f.write(f"{i}," + " ".join(f"{v:.17g}" for v in x) + "\n")
with open(os.path.join(out, "e9_families.csv"), "w") as f:
    f.write("instance,family\n")
    for i, fam, x in rows:
        f.write(f"{i},{fam}\n")
with open(os.path.join(out, "e9_ref_closed.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["instance", "value"]); w.writerows(refs)
print(f"e9_make: {len(rows)} sets ({len(refs)} closed-form, {len(rowsC)} random off-centre)")
