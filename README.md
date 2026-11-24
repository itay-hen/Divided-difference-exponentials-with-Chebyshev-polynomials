# Exponential divided differences via Chebyshev polynomials

Reference C++17 implementation accompanying the paper
*Exponential divided differences via Chebyshev polynomials* (I. Hen).

It computes exp[x₀, …, x_q] by expanding e^{cy} = I₀(c)(1 + 2 Σ Rₙ(c) Tₙ(y))
on the mapped interval and obtaining the Chebyshev divided differences
Tₙ[y₀, …, y_k] from a three-term recurrence. Header-only. Requires the
C++17 standard library and MPFR/GMP (link with `-lmpfr -lgmp`), which the
batch evaluator uses for its extended-precision last resort; Boost.Math is
used for `cyl_bessel_i` on platforms whose standard library lacks it, e.g.
Apple libc++.

## What it is good for — and what not

* **Batch evaluation.** O(qN) per evaluation. On operation traces recorded
  from a permutation-matrix-representation QMC code (PMR-QMC), batch
  evaluation was 1.6–34× faster than the incremental routine that code uses
  (`experiments/pmr_traces`).
* **Incremental updates.** With a fixed interval, appending a node costs
  Θ(N_max − q) and removing the most recently appended node costs O(1)
  (`chebyshev_stack.hpp`). For last-in-first-out updates an append takes
  0.13 µs at c = 5 (independent of q) and 0.36–0.44 µs at c = 50, up to
  1800× less than a full recomputation (q = 800). Interior
  removals cost a rebuild of the columns above them; on the PMR-QMC traces,
  which remove interior nodes and need large nodes first, the stack was
  slower than batch evaluation.
* **Accuracy.** Near working precision when q ≫ c, when q ≲ 10, or when
  c ≲ 30 (c = half-width of the interval), including clustered, repeated and
  nearly coincident nodes. The series is **ill-conditioned** when q and c are
  both large with q ≲ c, when the upper end b lies above the nodes, and,
  for q ≳ c, when the nodes lie mostly below the midpoint of the interval.
  `evaluate()` retries on [min x, max x] and then on [min x − s·w, max x],
  s = 1, 2 (w = max x − min x), which puts the midpoint at the smallest node
  and removes the last mechanism. The code computes an error indicator κ·u·N
  and never returns a value whose indicator exceeds the threshold (1e-10).
  Where no interval helps (q ≲ c, large c), the **same series is evaluated
  in extended precision** (MPFR, p = 128, 256, … bits) until its indicator
  κ·2^{1−p}·N is below the threshold. The condition number satisfies
  κ ≤ e^{b−ξ} (exp[x] = e^ξ/q!), so at most about 53 + 2.9c bits are ever
  needed; in all tests (c ≤ 10³) 128 or 256 bits sufficed and every call
  returned an accurate value. An extended evaluation is considerably more
  expensive than one in double precision and is used only where needed.
* **Range.** All intermediate quantities are scaled; any order works (tested
  to q = 3000). Values outside double range are returned in mantissa/exponent
  form; ratios of same-order values on the same interval are formed directly.
* **Truncation length.** N ≥ q exactly; N ≈ √(2c ln(1/ε)) when the width
  dominates.

## Quick start

```cpp
#include "chebyshev_exp_divdiff.hpp"   // batch evaluation
#include "chebyshev_stack.hpp"         // incremental evaluation

// Batch: one divided difference.
chebyshev::ExpDivDiff<double> e(-1.0, 1.0);          // interval [a, b]
double v = e.evaluate({-0.5, 0.0, 0.3, 0.7});
auto s   = e.evaluate_scaled(nodes);                 // mantissa * 2^exponent
double r = e.evaluate_ratio(nodes1, nodes2);         // same order, same interval
auto rs  = e.evaluate_ratio_scaled(nodes1, nodes2);  // ratios outside double range
long double est = e.last_error_estimate();           // error indicator
int bits = e.last_precision_bits();                  // 53: double; >53: extended precision

// Incremental: append / LIFO remove on a fixed interval.
chebyshev::IncrementalStack<double> st(a, b, /*q_max=*/1000);
st.initialize(initial_nodes);       // inserted in nonincreasing order
st.push(x);                         // Theta(N_max - q); throws if ill-conditioned
auto val = st.value_scaled();       // exp[x_0..x_q] as mantissa/exponent
st.pop();                           // O(1)
```

Choose b close to the largest node (for a single evaluation, b = max xᵢ;
`evaluate()` retries on [min xᵢ, max xᵢ] automatically). Lowering a only
costs extra terms; raising b above the nodes costs accuracy ~ e^{b − max xᵢ}.

## Files

| File | Contents |
|---|---|
| `chebyshev_bessel.hpp` | Bessel ratios (backward ratio recurrence), I₀, portable `cyl_bessel_i` |
| `chebyshev_exp_divdiff.hpp` | Batch evaluator (Algorithm 1): scaling, error indicator, re-anchoring, nonincreasing processing order |
| `chebyshev_stack.hpp` | Incremental evaluator (weighted contiguous columns, per-prefix gating) |
| `mccurdy.hpp` | McCurdy–Ng–Parlett baseline (entry-wise Taylor convergence, divided-difference squaring) |
| `incremental_chebyshev_exp_divdiff.hpp` | Legacy incremental class, superseded by `chebyshev_stack.hpp`; kept for older tests |
| `main.cpp` + `test_*.hpp` | Test driver: 11 required groups + 1 informational (mpmath) group |
| `mpmath_reference.py`, `cases.csv`, `reference.csv` | High-precision reference values for the informational group |
| `experiments/` | Drivers and plotting scripts for every figure and table of the paper; `experiments/run_v20.sh` regenerates all of them |
| `experiments/pmr_traces/` | PMR-QMC operation traces: recorder, replay, exact-value check |
| `CODE_CHANGES.md` | Detailed change log |

## Build and test

```sh
g++ -std=c++17 -O2 -I. -I/opt/homebrew/include main.cpp -L/opt/homebrew/lib -lmpfr -lgmp -o ci_runner
./ci_runner          # exits non-zero if a required group fails
```

Sanitizers: add `-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer`.

Experiments (Python 3 with numpy, matplotlib, mpmath):
`experiments/accuracy/run_all.sh`, `experiments/performance/build_and_run.sh`,
and `experiments/scaling/n_kappa_sweep.cpp` + `plot_scaling.py`. The shell
drivers write CSV data to `$DATA` and figures to `$FIGS`; set both, e.g.
`DATA=out FIGS=out experiments/accuracy/run_all.sh` (the defaults point to the
author's paper folders).

## Citation

```bibtex
@misc{Hen2025DividedDifferenceChebyshev,
  author       = {Hen, Itay},
  title        = {Divided-difference exponentials with {C}hebyshev polynomials},
  howpublished = {GitHub repository},
  url          = {https://github.com/itay-hen/Divided-difference-exponentials-with-Chebyshev-polynomials},
  year         = {2025}
}
```

## License

MIT License.

## Author

Itay Hen — Information Sciences Institute, University of Southern California — itayhen@isi.edu

AI-based tools assisted with programming; all code was reviewed and tested by the author.
