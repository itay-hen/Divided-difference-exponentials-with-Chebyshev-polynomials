# Exponential divided differences via Chebyshev polynomials

Reference C++17 implementation accompanying the paper
*Exponential divided differences via Chebyshev polynomials* (I. Hen).

It computes exp[x₀, …, x_q] by expanding e^{cy} = I₀(c)(1 + 2 Σ Rₙ(c) Tₙ(y))
on the mapped interval and obtaining the Chebyshev divided differences
Tₙ[y₀, …, y_k] from a three-term recurrence. Header-only; no dependencies
beyond the C++17 standard library (Boost.Math is used for `cyl_bessel_i` on
platforms whose standard library lacks it, e.g. Apple libc++).

## What it is good for — and what not

* **Incremental updates.** With a fixed interval, appending a node costs
  Θ(N_max − q) and removing the most recently appended node costs O(1)
  (`chebyshev_stack.hpp`). In our measurements an append takes 0.15–0.35 µs,
  up to ~10³× less than a full recomputation (q = 800). This is the update
  pattern of permutation-matrix-representation quantum Monte Carlo.
* **Accuracy.** Near working precision when q ≫ c, when q ≲ 10, or when
  c ≲ 30 (c = half-width of the interval), including clustered, repeated and
  nearly coincident nodes. The series is **ill-conditioned** when q and c are
  both large with q ≲ c, and when the nodes lie far below the upper end b of
  the interval. The code computes an error indicator κ·u·N and **raises an
  exception instead of returning an inaccurate value**; use MNP
  (`mccurdy.hpp`) or higher precision there.
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
long double est = e.last_error_estimate();           // kappa * u * N

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
| `main.cpp` + `test_*.hpp` | Test driver: 8 required groups + 1 informational (mpmath) group |
| `mpmath_reference.py`, `cases.csv`, `reference.csv` | High-precision reference values for the informational group |
| `experiments/` | Drivers and plotting scripts for every figure and table of the paper |
| `CODE_CHANGES.md` | Detailed change log |

## Build and test

```sh
g++ -std=c++17 -O2 -I. -I/opt/homebrew/include main.cpp -o ci_runner   # -I for Boost on macOS
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
