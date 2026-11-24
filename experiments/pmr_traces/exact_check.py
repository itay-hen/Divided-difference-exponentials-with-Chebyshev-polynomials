"""Exact-value check of sampled PMR-QMC queries.

For each trace, samples 200 queries (seed 1), computes exp[z_0..z_q]
exactly (sum of residues at the distinct node values, mpmath at 300 and
600 digits, which must agree), and compares the default Chebyshev
evaluation, the tight-interval-only evaluation, and a fresh GBH
evaluation (exact_eval).
Usage: python3 -I exact_check.py <exact_eval binary> <out.csv> trace.txt ...
"""
import random, subprocess, sys
from collections import Counter
from mpmath import mp, mpf, exp, log, binomial, factorial

def ddexp(nodes, dps):
    mp.dps = dps
    C = Counter(mpf(v) for v in nodes); vals = list(C); tot = mpf(0)
    for vj in vals:
        K = C[vj] - 1
        ser = [exp(vj) / factorial(k) for k in range(K + 1)]
        for vi in vals:
            if vi == vj: continue
            mi, dl = C[vi], vj - vi
            fac = [binomial(-mi, k) * dl ** (-mi - k) for k in range(K + 1)]
            ser = [sum(ser[a] * fac[k - a] for a in range(k + 1)) for k in range(K + 1)]
        tot += ser[K]
    return tot

def sets_of(fn, k):
    sh, out = [], []
    for L in open(fn):
        t = L[0]
        if t == 'A': sh.append(float(L[2:]))
        elif t == 'R': sh.pop()
        elif t == 'V':
            x = float(L[2:])
            for i in range(len(sh) - 1, -1, -1):
                if sh[i] == x: del sh[i]; break
        elif t == 'C': sh = []
        elif t == 'Q': out.append(list(sh))
    random.seed(1)
    return random.sample(out, min(k, len(out)))

binary, dst = sys.argv[1], sys.argv[2]
with open(dst, 'w') as f:
    f.write("trace,n,mean_q,default_accepted,default_max_err,default_max_err_over_est,tight_accepted,tight_max_err,gbh_max_err\n")
    for fn in sys.argv[3:]:
        S = sets_of(fn, 200)
        inp = "\n".join(" ".join("%.17g" % v for v in s) for s in S) + "\n"
        res = subprocess.run([binary], input=inp, capture_output=True, text=True, check=True).stdout.split("\n")
        da = ta = 0; de = te = ge = rr = 0.0
        for s, r in zip(S, res):
            q, m0, x0, e0, m1, x1, e1, gm, gx = r.split()
            a, b = ddexp(s, 300), ddexp(s, 600)
            assert abs(a / b - 1) < mpf(10) ** -100
            rel = lambda v: float(abs(v / b - 1))
            ge = max(ge, rel(mpf(gm) * mpf(2) ** int(gx) / factorial(int(q))))
            if m0 != 'nan':
                da += 1; err = rel(mpf(m0) * mpf(2) ** int(x0)); de = max(de, err)
                if float(e0) > 0: rr = max(rr, err / float(e0))
            if m1 != 'nan': ta += 1; te = max(te, rel(mpf(m1) * mpf(2) ** int(x1)))
        name = fn.split('/')[-1]
        f.write(f"{name},{len(S)},{sum(len(s)-1 for s in S)/len(S):.1f},{da},{de:.2e},{rr:.3f},{ta},{te:.2e},{ge:.2e}\n")
        f.flush()
        print(name, da, de, rr, ta, te, ge)
