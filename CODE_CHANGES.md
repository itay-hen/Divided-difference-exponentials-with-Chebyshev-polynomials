# Code patches summary

Five files modified or added. All five CI groups pass on `g++ -std=c++17 -O2`.

## Files

| File | Status |
|------|--------|
| `chebyshev_exp_divdiff.hpp` | modified |
| `incremental_chebyshev_exp_divdiff.hpp` | modified |
| `mccurdy.hpp` | header rewritten (file kept for include-compatibility) |
| `test_incremental_path.hpp` | new file |
| `main.cpp` | rewritten as CI driver |

The following files are unchanged and should be kept as-is:

- `comparison_test.hpp`
- `chebyshev_convergence_stats.hpp`
- `test_chebyshev_bessel_Rn.hpp`
- `test_chebyshev_bessel_I0.hpp`
- `test_comparison_chebyshev_mccurdy.hpp`
- `test_comparison_ratios_chebyshev_mccurdy.hpp`

## What changed

### Patch 1 — Bessel `I_0(c)` accuracy floor lifted from ~10⁻⁷ to ~10⁻¹⁵

Both `chebyshev_exp_divdiff.hpp::bessel_I0` and
`incremental_chebyshev_exp_divdiff.hpp::modified_bessel_I0` now wrap
`std::cyl_bessel_i(0, c)` (C++17). The previous Abramowitz–Stegun
polynomial approximation (eq. 9.8.1/9.8.2) is documented in A&S itself
as accurate to |ε| < 1.6×10⁻⁷ at small c and 1.9×10⁻⁷ at large c, and
empirically agrees with `std::cyl_bessel_i` only to 1.3e-8 at c=10,
3.0e-7 at c=50, 4.7e-7 at c=100. After the patch, agreement is at the
~10⁻¹⁵ level across c ∈ [0.5, 500] (Group 1 of the CI suite).

**This is the most consequential patch.** It removes a ~10⁻⁷ relative-
error floor in every full unnormalized `evaluate()` value at large c.

### Patch 2 — `mccurdy.hpp` now implements the actual MNP algorithm

The file previously contained only Sections 2.2–2.3 of McCurdy 1984
(Taylor for clustered nodes, classical recurrence otherwise), despite
being labeled as McCurdy–Ng–Parlett. It now implements the full
Opitz/scaling/squaring algorithm from Sections 3–4:

1. Build the (q+1) × (q+1) Opitz bidiagonal matrix `J` with diagonal
   `x_i` and superdiagonal `1`. Opitz's theorem gives
   `(f(J))_{i,j} = f[x_i,…,x_j]`, so `exp[x_0,…,x_q]` is in the first
   row of `exp(J)`.
2. Spectral shift: `d = (min + max) / 2`, form `M = J - dI` so the
   shifted eigenvalues lie in `[-c, c]`.
3. Choose `s` such that `||M||_inf / 2^s ≤ 1/2`; halve `M` in place
   `s` times (exact in binary float).
4. Compute `exp(M_scaled)` by truncated Taylor series. With
   `||M_scaled||_inf ≤ 1/2` the kth term is bounded by `(1/2)^k/k!`,
   so the series converges in O(log(1/ε)) terms.
5. Square s times to recover `exp(J - dI)`.
6. Multiply by `e^d` to get `exp(J)`; read the divided differences
   from the first row.

The implementation uses `std::vector<std::vector<Real>>` matrices with
an upper-triangular-aware multiply (`mul_upper`) that exploits the
fact that Opitz J is upper bidiagonal and the entire flow preserves
upper-triangular structure. Per-multiply cost is `O(q^3 / 6)`, total
`O(q^3 log(1+c) / 6)`. The 1984 paper quotes `O(q^2 log(1+c))` via a
more aggressive structured multiply that we do not implement; for
the regimes used in the accompanying paper's experiments this is
adequate.

**Verified behavior**:

- *Closed-form correctness*: bit-exact for `n=1, 2` against the
  analytic divided-difference formula.
- *Coincident nodes* (`exp[x, x, …, x]`): bit-exact match with
  `exp(x)/k!` for k = 0..10, no special-case branch (the Opitz matrix
  becomes a shifted nilpotent, the Taylor series terminates exactly
  after q+1 nonzero terms, and the squaring step gives the right
  answer).
- *Tightly clustered nodes* (spread 10⁻⁶, q=10): relative error
  4×10⁻¹⁵ — machine precision. Classical recurrence on `exp(x_i)`
  would lose roughly 10 digits at this clustering.
- *Where the old reference failed* (b=0.5, q=15 with random nodes,
  where the Taylor+classical reference returned **negative** values
  for what must mathematically be positive divided differences): the
  new MNP returns positive values and agrees with Chebyshev to
  ~10⁻⁹.

**Known limit**: at very high `q` (e.g. q=50) and very small interval
half-width (e.g. b=0.5), divided differences fall to ~10⁻⁶⁵ and the
Taylor + squaring chain in double/long-double precision starts to lose
digits. This is the regime the original 1984 paper notes as requiring
extended precision. The CI runner's Group 3 skips comparison trials
where `|ref_val| < 10⁻²⁰` for this reason, and counts them in a
"skipped" tally rather than failing the test.

**Manuscript implication**: the v15/v16 description of MNP as
"matrix exponential of the Opitz bidiagonal matrix... spectral
shifting, Taylor approximation of exp(J - dI), and repeated squaring"
is now exactly what the code does. The earlier mismatch (paper
described full MNP, code did only Taylor + classical recurrence) is
resolved by upgrading the code rather than retracting the paper.

The public API (`mccurdy_templated::ExpDivDiff<Real>::compute` and
`compute_all`) is unchanged, so the existing test files that include
the header continue to work without modification.

### Patch 3 — Single-node initialize + add_node no longer segfaults (A3)

The degenerate branch in
`IncrementalExpDivDiff::initialize()` returned with `is_initialized_
= true` but without populating `cached_D_`. A subsequent `add_node`
then dereferenced an empty cache. The fix calls
`initialize_base_layers(current_q_)` even in the degenerate path, so
the Chebyshev layer cache is valid for downstream column updates.

Verified: single-node init followed by `add_node(0.2)` now returns the
expected value `exp[0, 0.2] ≈ 1.10701`, and a three-coincident-node
init followed by `add_node` produces a result that agrees with fresh
batch recomputation.

### Patch 4 — `remove_last_node` from `q=0` throws cleanly (A4)

Previously, removing the only remaining node decremented `current_q_`
to -1 and then dispatched to `evaluate_from_cache(-1, false)`, which
threw or undefined-behaved depending on cache state. The fix
explicitly refuses to remove from a single-node state, with a clear
error message directing callers to `initialize()`. The fix also
shrinks each cached Chebyshev column on every successful removal to
preserve the cache-invariant that all layers in `cached_D_` have the
same width.

### Patch 5 — Silent non-convergence replaced with throws (B9)

Four loops are affected:
- `chebyshev::ExpDivDiff::evaluate_normalized` (batch);
- `chebyshev_incremental::IncrementalExpDivDiff::evaluate_from_cache`;
- `chebyshev_incremental::IncrementalExpDivDiff::initialize` build loop;
- `chebyshev_incremental::IncrementalExpDivDiff::add_node` build loop.

Each now tracks a `converged` flag set inside the five-consecutive-
terms stopping check, and throws `std::runtime_error` with a
descriptive message if the loop bound (`max_iterations_` or
`max_terms_`) is hit without convergence. Callers no longer silently
receive a partial sum.

### Patch 6 — Zero-padded Bessel ratios replaced with lazy expansion (B10)

`precompute_bessel_ratios()` used to allocate `bessel_ratios_` to
`max_terms_+1` and fill only entries up to a heuristic
`effective_max`, leaving zeros beyond. If a high-q evaluation needed a
ratio past `effective_max`, the silent zero would corrupt the result.

The new behavior mirrors the batch class: an initial cache of size
`max(100, 3c + 50)` is computed eagerly, and a new private accessor
`get_bessel_ratio(int n)` expands the cache on demand. Requesting an
order above `max_terms_` throws rather than rounding to zero.

All internal accesses to `bessel_ratios_[n]` (five sites) were
mechanically rewritten to use `get_bessel_ratio(n)`.

### Patch 7 — Stale `O(N) (independent of q)` comments fixed (B6)

Three comments in `incremental_chebyshev_exp_divdiff.hpp` claimed the
incremental cost was `O(N)` and "independent of q!". This contradicted
the v16 paper, which states the cost as `O(N(q, c; ε))`, with N
scaling with q in the order-dominated regime. The class-level header
and the docstrings for `add_node` and `remove_last_node` were
rewritten to match the paper's regime-aware language: the speedup
versus full recomputation is the removal of the explicit O(q) sweep,
not q-independence of the cost.

### New file — `test_incremental_path.hpp` (addresses A5)

The existing `comparison_test.hpp` only tested
`batch.evaluate(full_set)` against
`IncrementalExpDivDiff::initialize(full_set)` — two ways of building
the same state from scratch. It never exercised the advertised
incremental path of `add_node` and `remove_last_node`.

The new file adds five scenarios:

1. Progressive append at b=2 (small c), q up to 10, comparing the
   incremental running value against a fresh batch recomputation
   after every append.
2. Same at b=20 (large c), q up to 15.
3. A deterministic LIFO push/pop pattern.
4. Twenty randomized LIFO walks of mixed `add_node`/`remove_last_node`
   operations.
5. Coincident-start initialize (the previously-crashing case) followed
   by progressive append.

All five pass at relative tolerance 10⁻¹⁰ (the project's existing
"good" threshold). See Group 5 of the CI driver.

### Rewritten `main.cpp` — real CI driver (C13)

Old `main.cpp` was mostly commented-out exploration. New `main.cpp`
runs five test groups, reports per-group pass/fail, and exits non-zero
on any failure:

```
=== Group 1: Bessel I_0 wrapper accuracy ===           PASS (8/8)
=== Group 2: Bessel ratios R_n(c) vs scipy ===         PASS (50/50)
=== Group 3: Batch Chebyshev vs Taylor+classical ref ===PASS (80/80)
=== Group 4: Batch vs Incremental (full-set init) ===  PASS
=== Group 5: Incremental update path ===               PASS (all scenarios)
 5/5 groups passed
```

Group 3 is now restricted to (b, q) combinations where the reference
itself is in its stable regime — at b=0.5, q=15 the unmodified
reference returns negative values, which is mathematically impossible
for real-node exp DDs (cancellation collapse in the classical
recurrence). Trials where the reference is non-positive or non-finite
are skipped with a counter, not silently passed.

## Items NOT addressed in this pass

These were lower priority in the review and are noted for follow-up:

- **B7 (append cost framing)** — partially addressed via patch 7
  (docstrings now state O(N(q+1, c; ε)) and reference the standing
  assumptions of Sec. 5 of the paper). Not made into a runtime cost
  meter.
- **B8 (stale cache entries past current N)** — the new lazy Bessel
  cache and the column-width shrink in `remove_last_node` reduce the
  surface area of this concern. No explicit per-column "valid N"
  bookkeeping has been added; if stale-tail failures ever surface in
  practice, that would be the next step.
- **C11 (independent high-precision reference)** — still open in the
  sense that no MPFR/Arb benchmark has been added. However, with the
  full MNP now in `mccurdy.hpp`, Group 3 of the CI compares two
  genuinely different algorithms (a Chebyshev–Bessel expansion vs an
  Opitz matrix exponential). They agree to ~10⁻⁷ across the regime
  where double-precision MNP can hold several digits. This is not a
  high-precision certification but it is a meaningful cross-check.
- **D14 (hardcoded `1e-15` threshold for confluent-node detection)** —
  not changed.
- **D15 (overstated "arbitrary precision" templating)** — not changed
  in code; the rewritten `incremental_chebyshev_exp_divdiff.hpp` class
  header now describes the precision support more conservatively.
- **D16 (duplicated Bessel code)** — not merged. Both files now use
  `std::cyl_bessel_i` so the duplication is far smaller than before,
  but they still each carry their own Miller routine.

## How to build and run

```
g++ -std=c++17 -O2 -I. main.cpp -o ci_runner
./ci_runner
```

Exit code is 0 if all five groups pass, 1 otherwise.

## Paper implications

The most important paper-side consequence of Patch 1: the v16 §4.3
claim that "Using extended precision (long double, 80-bit), both
methods can be pushed to c ∼ 10³, agreeing to relative errors of order
10⁻¹¹" was inconsistent with the previous I0 accuracy floor of ~10⁻⁷
at c ∼ 10². With the patched I0, this measurement should be re-run
and the figure updated to whatever it actually is.

The Fig. 5 heatmap is unaffected by Patch 1 in the sense that it
compares ratios, in which the I0 factor cancels exactly.  But with
the full MNP now in place (Patch 2), the "red region" of Fig. 5
needs reinterpretation: the heatmap was generated with
double-precision Chebyshev compared against (the old, broken) MNP.
With the real MNP running in long double, what the red region
actually shows is **double-precision Chebyshev failing while
long-double MNP is correct** — the opposite of the v16 §4.3 caveat.

With Patch 2 (full MNP) and Patches 3–7 + Round 2 (A1, A2), the
manuscript's reference-implementation description of Sec. 5 is now
consistent with the code: a global N_max is chosen at construction
and the column stack is pre-filled to that depth, so add_node and
remove_last_node really do cost O(N_max) without layer-creation
overhead.  The certified stopping mode from §3.2.2 is also present
in code as StoppingMode::CertifiedTailBound, matching the v16 text
"the certified criterion is available as an optional safety mode."

## Round 2 additions (A1 global N_max, A2 certified mode)

After the third round of code review the following further changes
were made; all are confined to the incremental class and its public
API:

**A1 — Global N_max caching.**  The constructor now accepts an
explicit `max_q` parameter and either an explicit `max_terms` or
auto-chooses one from the regime estimates of Sec. 5:

    N_max = max(ceil(c + sqrt(2 c log(1/eps))),
                max_q + sqrt(2 max_q log(1/eps))) + 50.

Bessel ratios are pre-built to `max_terms_` at construction, and
`initialize()` pre-fills `cached_D_` to `max_terms_` layers eagerly.
Subsequent `add_node()` calls only widen each cached layer by one
column entry — there is no layer-creation path.  This makes the
"global N_max" invariant of v16 §5 a code-level guarantee.

`add_node()` now throws `std::invalid_argument` if `current_q_+1`
would exceed `max_q_`, signalling a configuration error rather than
silently growing memory.

**A2 — Certified stopping mode (Eq. 33).**  Added
`enum class StoppingMode { HeuristicFiveTerm, CertifiedTailBound }`,
defaulting to the heuristic for backward compatibility.  The
certified mode maintains a running upper bound on the still-to-add
tail using the precomputed structural bounds

    B(n,q) = 2^{q-1} * (n/q) * (2q)_{n-q} / (n-q)!     for n >= q

and the cached Bessel ratios; convergence is declared when

    2 e^d I_0(c) q! / (c^q e^a) * sum_{n'>n} R_{n'}(c) B(n',q)  <=  eps,

which is exactly the conservative criterion of Eq. (33).  Slower per
step than the heuristic but provides a provable relative-error
guarantee at exit.  Selectable both at construction and at runtime
via `set_stopping_mode()`.

**A3 — Normalized routines no longer form full exponentials.**
The batch class's `evaluate_normalized()` had a coincident-node
shortcut that computed `exp(x_0)/q! * c^q / (exp(d) * I_0(c))`,
which can overflow at large nodes even though the normalized result
is well-behaved.  Replaced with the safer closed form

    S = exp(c * y_0) * c^q / (q! * I_0(c)),

bounded by exp(c)*c^q/(q!*I_0(c)) (and exp(c)/I_0(c) is itself
bounded by sqrt(2 pi c) for large c).  In the incremental class,
the same closed form is used for `q == 0` (the single-node case),
which also fixes a separate cancellation issue at large c when y_0
is near the interval endpoints (the explicit Chebyshev sum loses
about log_10(I_0(c)) digits to cancellation, e.g. ~7 digits at c=20).

`add_node_normalized()` no longer recomputes via
`evaluate_from_cache(...,true)`: the running normalized sum
`current_S_` is now tracked through every state-changing call, so
the normalized read is O(1) and never goes through the full-value
prefactor.

**A4 — Exception safety in `add_node` / `remove_last_node`.**
Both methods now have strong exception-safety semantics:

- `add_node()` uses an RAII rollback guard.  All mutations
  (current_nodes_, normalized_nodes_, current_q_, the new column in
  every cached layer, current_value_, current_S_) are made
  optimistically; the guard's destructor restores the pre-call state
  if `committed` is not set before the method returns.  Any throw
  from convergence, allocation, or `get_bessel_ratio` therefore
  leaves the object usable.

- `remove_last_node()` uses a compute-then-commit pattern.  The
  post-removal value is computed via `evaluate_from_cache` at
  `current_q_-1` (a read), and only after that succeeds does the
  routine pop the node, decrement `current_q_`, and shrink each
  cached layer.  An exception in evaluate leaves the object
  unchanged.

This matters in QMC samplers that catch convergence exceptions
around a rejected move: the sampler can now safely discard the
exception and continue from a consistent state.

**A5 — `bessel_ratios_` is now `mutable`.**  The const-cast in
`get_bessel_ratio` is gone; the field carries the `mutable`
qualifier so the (defensive) lazy expansion path can run from
logically-const methods without UB.

**B1 — Exception-message wording.**  All convergence-failure
messages now advise increasing `max_terms_` or loosening
`tolerance_`, dropping the earlier (incorrect) suggestion to raise
the interval half-width, which makes the Chebyshev expansion harder
rather than easier.

**B4 — Dead code in batch constructor.**  The
`std::min(nmax, std::max(1000, ...))` chain (which always returned
99) is replaced with an adaptive `initial_max = max(100, 3c + 50)`
seed and a single Miller call.  Lazy expansion still extends the
cache on demand when q drives the truncation length past the seed.

**Verification.**  A new Group 6 in the CI runner covers A2 and A4
directly:
- A4.1–A4.3: throws on out-of-range node, on q exceeding max_q, and
  on removing from q=0, in each case verifying that current_order()
  and current_value() are unchanged after the throw.
- A2.1: certified mode matches heuristic to <= 1e-10 on a generic
  10-node input.
- A2.2: certified mode at requested tolerance 1e-8 agrees with a
  tight heuristic reference to <= 1e-7, confirming the bound is
  conservative but not vacuous.
- A2.3: `set_stopping_mode` switches the rule at runtime; subsequent
  `add_node` uses the new rule and matches a fresh batch
  recomputation.

Full CI: 6/6 groups PASS.

## Round 3 additions (independent mpmath reference + matched-precision tests)

After the Round-2 implementations landed, two further additions were
made to address concerns about how Chebyshev and MNP were being
compared.

**Independent ground-truth reference.**  Added two files:

- `mpmath_reference.py` — Python script that generates a list of
  test cases (a default `(b, q)` grid plus confluent/clustered stress
  cases) and computes `exp[x_0, ..., x_q]` for each at ~100 decimal
  digits using `mpmath.expm` applied to the Opitz bidiagonal matrix.
  Outputs `cases.csv` (the inputs) and `reference.csv` (full-precision
  decimal string values).  Run once with:

      python3 mpmath_reference.py --generate --dps 100

- `test_against_reference.hpp` — C++ test that reads the two CSVs and
  evaluates both Chebyshev and MNP **at matched working precision**
  (a `double` pass and a `long double` pass) against the mpmath
  reference.  Reports per-method success rate, median, and max
  relative error at each precision.  Used as Group 7 of the CI; runs
  if the CSVs are present, prints a hint and is skipped otherwise.

This closes the loop the reviewer asked for: neither method gets to
"check itself" in higher precision, and both are checked against a
value computed by a third, much-higher-precision algorithm.

**Matched-precision in the existing comparison tests.**  The
`test_comparison_chebyshev_mccurdy.hpp` and
`test_comparison_ratios_chebyshev_mccurdy.hpp` files previously used
MNP at `long double` as a "reference" against which both methods'
`double` results were checked.  That was a precision-asymmetric setup
that credited MNP for matching itself in higher precision.  Both
files are now templated on `Real`:

    template<typename Real = double>
    inline TrialResult run_single_trial(...);

    template<typename Real = double>
    inline void run_all_tests(...);   // both methods evaluated at Real

    // and similarly run_ratio_trial, test_ratio_case, generate_ratio_heatmap.

Default usage (no template argument) preserves the historical `double`
behavior.  Explicit `<long double>` runs the matched-precision long
double comparison.  The long-double MNP "reference" is dropped; the
reported error is now the pairwise difference between MNP and
Chebyshev at the same precision.  For true accuracy assessment, the
mpmath workflow above is the correct tool.

**Group 7 results on the default 176-case grid:**

| Run | Method | Median rel_err | Max rel_err |
|---|---|---|---|
| double | Chebyshev | 7e-16 | 1.5e+16 |
| double | MNP | 9e-15 | 4.8e-6 |
| long double | Chebyshev | 1e-18 | 9.1e+13 |
| long double | MNP | 3e-18 | 2.7e-7 |

This is honest information that the v16 manuscript does not yet
reflect.  The Chebyshev catastrophic max at large `c` and low `q` is
real and appears in both precisions, confirming the diagnostic from
earlier in the patch series.  MNP loses several digits in the tiny-
value regime at small `c` and high `q` but never goes catastrophically
wrong.  Both findings are consistent with §3 of the McCurdy-Ng-Parlett
1984 paper (extended precision needed at high `q`) and with the
diagnosis in this codebase's earlier discussion of cancellation in
the normalized Chebyshev sum.

Full CI: **7/7 groups PASS**.

## Round 4 fixes (Round-3 review response)

The Round-3 review flagged five "must fix" items.  All are addressed.

**A1 — Certified prefactor: wrong sign corrected, computation moved
to log space.**  The certified mode previously used a prefactor
proportional to `e^{-c} * I_0(c) q! / c^q`.  The correct prefactor
is `e^{+c} * I_0(c) q! / c^q` (because `d - a = +c`, not `-c`); the
old sign made the bound off by `e^{-2c}`, declaring convergence
absurdly early at large `c`.  At c=100, q=3 the reviewer measured an
actual rel error of 2.4e-4 against a certified tolerance of 1e-8 —
six orders of magnitude over budget.

The fix is computed in log space because `e^{+c} * I_0(c)` overflows
double for `c >~ 700`:

    log(prefactor) = log(2) + c + log(I_0(c)) + log(q!) - q log(c).

`log(I_0(c))` itself uses `std::log(modified_bessel_I0(c))` for c<50
and a three-term asymptotic `c - (1/2)log(2πc) + log(1+1/(8c)+...)`
for larger c.  Convergence is now checked as

    log(prefactor) + log(tail_sum) <= log(tolerance).

Verified: the same c=100, q=3 case at long double precision now
achieves rel error 2e-9 against a tight reference (requested 1e-8).
At double precision the limiting factor is round-off in the
Chebyshev sum itself (~1e-5 at c=100), not the certified bound.

**A2 — Infinite-tail residual added.**  The certified tail sum was
truncated at `max_terms_`; the bound was therefore only over n ≤
max_terms_.  A new helper `certified_tail_residual(q)` adds an
analytic upper bound on `Σ_{n > max_terms_} R_n(c) B(n,q)`:

  * R_{N+1} computed from the Bessel three-term recurrence
    `R_{n+1} = R_{n-1} - (2n/c) R_n` using the cached `R_{N-1}`, `R_N`.
  * B(N+1, q) from the B-recurrence.
  * Worst-case term ratio in the tail bounded at n = N+1.  If that
    ratio r < 1, geometric sum gives `residual <= term_{N+1}/(1-r)`.
  * If R_{N+1} comes out non-positive (Miller instability) or r ≥ 1,
    the function returns +infinity, which the caller turns into a
    clear "increase max_terms" exception rather than pretending to
    certify.

`certified` is now a real certificate over the full infinite sum, not
just the finite cache.

**A3 — Normalized API truly avoids the full-value scaling.**
`add_node` and `initialize` previously always formed
`scale_full(S, q) = e^d * I_0(c) * c^{-q} * S`, even when
`add_node_normalized` / `initialize_normalized` were called: those
wrappers just discarded the full value after computing it.  At
extreme c the full-value step overflows even though S is fine; the
"normalized" path was therefore not actually normalized-safe.

Refactored as private helpers `add_node_compute_S_(new_node, form_full)`
and `initialize_compute_S_(nodes, form_full)` plus a parallel
`remove_last_node_compute_S_(form_full)`.  Public wrappers:

    T add_node(T new_node);                   // form_full=true
    T add_node_normalized(T new_node);        // form_full=false
    T initialize(...);                        // form_full=true
    T initialize_normalized(...);             // form_full=false
    T remove_last_node();                     // form_full=true
    T remove_last_node_normalized();          // form_full=false

A new member `full_value_valid_` tracks whether `current_value_`
holds a current full divided difference.  Any `*_normalized` call
sets it to false; `current_value()` then throws rather than returning
a NaN or stale value.  A new public `compute_full_value()` opts in
to forming the full value from `current_S_` on demand.

**A4 — Batch `evaluate_normalized` validates nodes before the
coincident branch.**  The coincident-node shortcut previously fired
before node-range checking, so out-of-interval coincident nodes
returned a number instead of throwing.  Reordered: now every node is
validated against `[a, b]` first; the coincident branch runs only on
valid input.

**A5 — Required vs informational CI groups separated.**  The CI
driver previously counted Group 7 (mpmath ground truth) as a passed
group even though it always returned `true`.  Now reported as

    6/6 required groups passed
    1 informational group(s) completed (not counted toward CI pass/fail)

`main` returns nonzero only on required-group failures; Group 7
output is unchanged but no longer inflates the headline pass count.

**B1 — Atomic mpmath file writes + C++ manifest cross-check.**
`mpmath_reference.py` now writes to `cases.csv.tmp` /
`reference.csv.tmp` and renames atomically on completion.  A
trailing manifest line `#manifest,n_cases=N,,` is appended to each
file.  The C++ loader requires both manifest lines and refuses to
use a file whose loaded row count disagrees with its manifest.  An
interrupted Python run therefore cannot leave a silently-truncated
file that the C++ test would treat as complete.

**B2 — Wording fixed in `test_against_reference.hpp`.**  The header
no longer claims the C++ side performs 100-digit comparison; it now
says explicitly that the mpmath reference is parsed into long double
via `std::stold` for the C++ comparison, with a ~10^-18 floor on the
long-double pass.

**B3 — Strict Bessel-ratio cross-check added.**  Group 2 now reports
both the old scipy-rounded smoke test (~1e-6 tolerance) and a new
strict check against `std::cyl_bessel_i(n,c)/std::cyl_bessel_i(0,c)`
at 1e-12 tolerance for a moderate (n, c) grid.  Result: 25/25 pass,
max rel_err = 1.47e-14.  The Miller-cached ratios are now
independently validated, not just spot-checked against scipy.

**B4 — Less tautological I_0 test.**  Group 1 now has three
sub-tests: (a) wrapper plumbing against `std::cyl_bessel_i` (smoke),
(b) `bessel_I0(c)` against the large-c asymptotic
`I_0(c) ~ e^c/sqrt(2πc) * (1 + 1/(8c) + ...)` at c ∈ {30..500}, and
(c) `bessel_In(n, c)` against `std::cyl_bessel_i(n, c)` for moderate
(n, c) at 1e-12.  (a) is a plumbing check; (b) and (c) are
genuine independent validation.

**B5 — Portability documented.**  Header docstring at the top of
`chebyshev_exp_divdiff.hpp` now states the `std::cyl_bessel_i`
requirement and the known macOS libc++ limitation, with a pointer
to `boost::math::cyl_bessel_i` as a drop-in replacement if needed.

**C1 — Duplicate `@brief Append a node` comment block fixed.**
The doc block had a stray nested comment that caused
`-Wcomment` to fire.  Removed.

**C2 — Unused state removed from batch class.**  Members
`nodes_, y_, D_curr_, D_prev_, D_prev2_, last_n_, last_sum_,
cache_valid_` were dead code from an abandoned in-class incremental
attempt; the batch evaluator is stateless except for the interval
parameters and the lazy Bessel-ratio cache.  All eight members and
their constructor initializers are now gone.

**C4 — `bessel_In()` validated.**  The Group 1 cross-check (B4
above) directly tests it against `std::cyl_bessel_i`.  No need to
hide the API; it is now demonstrably correct.

**C5 — Stale `main.cpp` top comment refreshed.**  The file header
correctly describes the 6+1 group structure and notes that Group 3
uses the full Opitz/scaling/squaring MNP reference.

## Not done

**C3 — Factor out `chebyshev_bessel.hpp`.**  Cosmetic; the Bessel
duplication between `chebyshev_exp_divdiff.hpp` and
`incremental_chebyshev_exp_divdiff.hpp` is now small (both delegate
to `std::cyl_bessel_i` and an independent Miller routine).  Left as
a follow-up.

Full CI: **6/6 required groups PASS, 1 informational group reported
honestly.**  Group 6 now has 9 sub-tests, including A2.4 (the c=100
adversarial case from the review that motivated the A1 fix), A3.1
(normalized invalidates full value), and A3.2 (compute_full_value
restores it).

## Round 5 fixes (Round-4 review response)

The Round-4 review confirmed Round-3 progress, then flagged one new
serious bug (state corruption in `initialize()`), two rigor gaps in
the certified mode, an off-by-one in `max_terms_` semantics, and a
collection of polish items.  All addressed below.

**A1 — `initialize()` is now transactional.**  Previously
`initialize_compute_S_` mutated `current_nodes_`, `current_q_`, and
`normalized_nodes_` before validation, leaving the object in a
poisoned hybrid state when validation later threw.  Rewritten to
build all new state into local variables (`new_nodes`,
`new_normalized`, `new_cached`, `new_S`) and commit via `std::move`
only after every validation and the evaluation succeed.  Two helpers
support this: `compute_next_layer_local` takes the normalized-nodes
vector as a parameter (no member dependency), and
`evaluate_from_cache_local` is a read-only sibling of
`evaluate_from_cache` operating on a supplied cache.

Verified against the reviewer's exact scenario:

    inc.initialize({0.0, 0.2});
    auto old_nodes = inc.current_nodes();
    auto old_val   = inc.current_value();
    try { inc.initialize({0.1, 5.0}); } catch (...) {}
    // current_nodes() == old_nodes
    // current_value() == old_val
    // current_order() unchanged

This is now Group 6 subtest A1.1 and passes.

**A2 — Certified mode is now rigorously one-sided.**  Two
under-estimation hazards in the Round-3 implementation:

  * `log_I0_safe(c)` for `c >= 50` used a truncated asymptotic series
    that *under*-estimated `log(I_0(c))` by ~6e-7 at c=50.  Fixed by
    adding the next-omitted-term upper bound `+75/(1024 c^3)` (loose
    but strict) plus a `+16 ulps` pad in the direct-evaluation regime.
    The function now returns a rigorous upper bound on `log(I_0(c))`.
  * `certified_tail_residual` computed `R_{N+1}` via the subtractive
    Bessel recurrence `R_{n+1} = R_{n-1} - (2n/c) R_n`, which suffers
    cancellation in the tail and could underestimate without going
    negative.  Replaced with `std::cyl_bessel_i(N+1, c) /
    std::cyl_bessel_i(0, c)`, which is well-conditioned and
    independent of any cached recurrence drift.

Both changes preserve the upper-bound invariant the certified mode
relies on.  Verified at c=100, q=3, tol=1e-8 in long double: cert
rel_err = 2.06e-9 (well within the bound).

**A3 — `max_terms_` renamed to `max_order_`, semantics made
consistent.**  The old name was a number-of-layers count whose
indices ran 0..max_terms_-1, while `certified_B_` indexed
0..max_terms_.  Now `max_order_` is unambiguously the maximum
Chebyshev order N retained.  Invariants after `initialize()`:

    cached_D_.size()     == max_order_ + 1   (indices 0..max_order_)
    certified_B_.size()  == max_order_ + 1   (indices 0..max_order_)
    bessel_ratios_.size() == max_order_ + 1   (indices 0..max_order_)

The certified-mode residual covers the tail beyond `n = max_order_`.
Public accessor `max_order()` returns this value.  Documented at the
member declaration.

**A4 — Group 7 reports `SKIPPED`, not `PASS`, when no reference
data is found.**  Reference test now returns
`enum class InfoResult { Passed, FailedInformational, Skipped }`.
The CI driver prints `Group 7 SKIPPED` when cases.csv / reference.csv
are absent, `Group 7 PASS` when both precision passes meet tolerance,
and `Group 7 INFO` when some cases drift over the per-precision floor.

**B1 — CI test for failed `initialize()`.**  Added as A1.1; see A1
above.

**B2 — Bessel infrastructure factored into `chebyshev_bessel.hpp`.**
The duplicated `numeric_traits` and `BesselRatiosMiller` template
classes (one copy in each evaluator header) now live in one place,
inside a neutral `chebyshev_bessel::` namespace.  Both
`chebyshev_exp_divdiff.hpp` and
`incremental_chebyshev_exp_divdiff.hpp` `#include` it and re-export
the names into their own namespace via `using` declarations:

    namespace chebyshev {
        using chebyshev_bessel::numeric_traits;
        using chebyshev_bessel::BesselRatiosMiller;
        ...
    }

Call sites in both evaluators continue to use the unqualified names.
The portability note about `std::cyl_bessel_i` now lives in the
shared header where it belongs.

**B3 — Miller-recurrence comments rewritten** to match the actual
sliding-window indexing in `bessel_In`.  The previous comments
implied a different windowing convention than the code actually uses;
the new comments walk the reader through how `I_curr`, `I_next`, and
`I_prev` shift on each iteration, and why `I_n_unnorm` is captured at
the iteration where the recurrence produces order n.

**B4 — `read_cases_csv` parses nodes as `mpf(s)`.**  Previously the
script went through Python `float(s)` before converting to mpf,
silently truncating any custom CSV that contained high-precision
decimal nodes.  Default-grid cases generated by this same script are
unaffected (they originate as Python floats); the change matters for
user-supplied input CSVs.

**B5 — `--quick` option for a small CI-sized grid.**  `default_grid()`
now accepts `quick=True`, returning ~20 cases (vs ~180 for the full
grid).  Wired through `--quick` on the CLI.  The full grid is
preserved for offline validation but no longer the only choice.

**C1 — Batch coincident-node threshold removed.**  The fuzzy
`1e-15`-spread test in `evaluate_normalized` silently substituted
the closed-form `exp(c y_0) c^q / (q! I_0(c))` for
near-but-not-equal nodes — which is the wrong mathematical object
(a divided difference, not a value of e^x).  Now triggers only on
exact equality across all nodes (`nodes[i] == nodes[0]` for all i).
Tightly clustered nodes go through the normal Chebyshev path.
Documented inline with the rationale.

**C2 — `last_used_order()` exposed; reported by Group 4.**  New
member `last_used_order_` records the Chebyshev order `n` at which
the stopping rule fired on the most recent evaluation.  Tracked in
both `evaluate_from_cache` and `evaluate_from_cache_local`.  Group 4
output now has two columns: `Incr Nmax` (cache depth, ~231) and
`Incr used` (actual stopping order, typically 20-40 even at q=15).
Makes the incremental method's cost transparent.

**C4 — "Opitz/MNP reference" wording.**  Renamed throughout
user-facing output and comments.  The MNP method as
implemented in `mccurdy.hpp` is the Opitz/scaling-squaring version
with effective O(q^3 log(1+c)) cost; the paper's optimized
structured MNP variant is O(q^2 log(1+c)).  Calling our
implementation "Opitz/MNP reference" is the honest description.

## Not done

**C3 — alternate Bessel backends.**  Boost.Math fallback for
non-glibc platforms.  Left as future work; the portability note
in the header tells users what to do meanwhile.

Full CI after Round 5: **6/6 required groups PASS, Group 7 reports
honestly as PASS/INFO/SKIPPED.**  Group 6 has 10 sub-tests: A1.1
(failed-initialize rollback), A2.1–A2.4 (certified mode), A3.1–A3.2
(normalized API invariants), A4.1–A4.3 (add/remove rollback).

## Round 6 fixes (Round-5 review response)

The Round-5 review acknowledged the structural progress and narrowed
the remaining concerns to eight items.  All addressed.

**A1 — Mode renamed and weakened wording.**  `CertifiedTailBound`
renamed to `ConservativeTailEstimate`, with the old name kept as
a deprecated `enum class` alias for source compatibility.  Docstrings
on the mode and on `log_I0_safe` now say explicitly that this is a
*conservative floating-point implementation, not a proof-level
interval-arithmetic certificate*, and point users who need actual
certification to Arb / Boost.Interval / MPFI.

**A2 — Log-space confluent / q=0 branches.**  All "normalized" paths
that used to compute closed-form expressions like
`exp(c * y_0) * c^q / (q! * I_0(c))` or `exp(x_0 - d) / I_0(c)` in
linear scale -- which overflow when c is large enough that
`exp(c y_0)` exceeds the type's range, even though the normalized
result is fine -- are now computed in log space:

    log S = c y_0 + q log c - log(q!) - log_I0_safe(c)

with an explicit overflow check that returns +inf when
`log S > log(max(T))`.  Verified end-to-end at c=1000 with both the
batch confluent branch (q=2: result `2.82e-210`, was NaN) and the
incremental q=0 path (result `2.95e-42`, was NaN).  `log_I0_safe`
moved into `chebyshev_bessel.hpp` for use by both evaluators.

**A3 — `last_used_order_` is now transactional.**  Added a
`DiagGuard` RAII helper at the top of `initialize_compute_S_`
that snapshots `last_used_order_` and `certified_B_q_` and
restores them on any throw.  Extended the existing `Rollback`
inside `add_node_compute_S_` similarly (it now also restores
`full_value_valid_`).  Failed initialization (whether validation
*or* convergence failure) leaves both the mathematical state and
the diagnostic-only fields exactly as they were.

**A4 — `certified_B_` moved to log space; certified path uses
log-sum-exp throughout.**  The previous linear-scale `certified_B_`
could overflow at large q (B(n,q) grows polynomially in n with
power 2q-1, and the cached `R_n * B(n,q)` product was formed in
linear scale even when each factor was within range).  Now stored
as `log_certified_B_[n]`, computed by the same B-recurrence in
log space.  `certified_log_tail_residual` returns a log-space
residual built from `std::cyl_bessel_i(N+1, c)` and the
log-space B-recurrence.  Both `evaluate_from_cache` and
`evaluate_from_cache_local` precompute
`log_term[n] = log R_n + log_B[n]`, then build
`log_tail_partial[k] = log( sum_{j>=k} R_j B(j,q) + residual )`
by a reverse log-sum-exp pass; at step n the convergence check
becomes a single lookup `log_pref + log_tail_partial[n+1] <= log_tol`.
Linear-scale `R_n * B(n,q)` is never formed.  The
log_sum_exp helper is defined once on the class and handles
the -infinity case correctly.

Side effect: the more honest residual exposed that the auto-chosen
`max_order` was sometimes too small for `ConservativeTailEstimate`
mode at high c (the old residual was unsafely under-estimated by the
subtractive R-recurrence; the new one is not).  The auto formula now
adds `2 * c` extra terms when `ConservativeTailEstimate` is selected
at construction.  An up-front check in the eval loop additionally
detects "max_order is too small for this c, q, tol" before the loop
even starts and throws a clear message rather than running through
the cache and reporting the misleading "cached layers exhausted."

**B1 — `q=0` certified-bound comment corrected.**  The comment in
`precompute_certified_bounds` previously claimed `B(n,0) = 0` for
`n >= 1`; the code actually used `B(n,0) = 1` (the safe envelope
from `|T_n(y_0)| <= 1`).  The code is correct; the comment now says
so explicitly, with the analytic-continuation rationale spelled out.
In log space this is `log_certified_B_[n] = 0` for all n.

**B2 — `bessel_In()` documented as moderate-range only.**  The
docstring now warns that the routine returns `I_n(x)` in linear
unscaled form and overflows for `x >~ 700` (double) / `~11000`
(long double); the main evaluator path uses `BesselRatiosMiller`,
which stays bounded.  Users who need many ratios at large c are
directed to `BesselRatiosMiller::compute_R` instead.

**B3 — Small committed reference data.**  `cases.csv` and
`reference.csv` (14 cases, generated by `mpmath_reference.py
--generate --quick --dps 100`) are now shipped in the patch set.
A fresh checkout therefore runs Group 7 against real data
instead of silently SKIPPING.  Replace with `--generate` (full
~180-case grid) for offline validation.

**C2 — Dead `total_scale` variable removed** from
`BesselRatiosMiller::compute_R`.  The accumulated rescaling
factor was correct in intent (it would have been needed if the
function returned `I_n` values) but unused (the function returns
ratios `R_n = I_n / I_0`, which are invariant under uniform
rescaling).  Removed along with the misleading comment about
"tracking ALL rescaling to preserve ratios"; the actual reason
ratios survive rescaling -- that they are rescaling-invariant by
construction -- is now documented inline.

## Not done

**C1 — Interval-arithmetic backend.**  The `ConservativeTailEstimate`
mode is now an honest conservative-floating-point bound, but it is
not a proof-level interval-arithmetic certificate; for that the
Bessel calls and the asymptotic remainder estimate would need to
go through Arb / Boost.Interval / MPFI.  The renamed mode and
weakened docstring make this explicit; an actual interval-arithmetic
backend is left as future work.

Full CI after Round 6: **6/6 required groups PASS, Group 7 reports
honestly** (PASS / INFO / SKIPPED).  Group 6 has 10 sub-tests all
passing, including A2.4 (the c=100 adversarial case which now meets
its 1e-8 cert tolerance at rel=6.81e-10 using the more generous
auto-sized `max_order`).

## Round 7 fixes (Round-6 review response)

Round-6 review found three real bugs and several smaller items.  All
addressed.

**A1 — Separate `log_I0_value` (accurate) from `log_I0_upper`
(conservative bound).**  The Round-5 fix used `log_I0_safe` (padded
upward for one-sided-bound use) inside the *value-computing* paths of
the batch confluent branch and the incremental q=0 normalized path.
That over-padding flowed straight into the computed normalized sum
and biased the full-value result by ~1.7e-8 at c=50 (and ~1e-9 at
c=100) — far above the advertised 1e-14 accuracy.

Fix: in `chebyshev_bessel.hpp`,

  - `log_I0_value(c)` — accurate.  Uses `std::log(bessel_I0(c))` while
    that fits in Real (c < log(max(Real)) - 10), otherwise switches
    to a four-term asymptotic
    `c - (1/2) log(2 pi c) + log(1 + 1/(8c) + 9/(128 c^2) + 75/(1024 c^3))`
    whose truncation error is ~3675/(98304 c^4), well below 1e-11
    for c >= 5.

  - `log_I0_upper(c)` — `log_I0_value(c)` plus a small positive pad
    (16 ulps in the direct regime, next-asymptotic-term `3675/(98304 c^4)`
    in the large-c regime).  Used by `ConservativeTailEstimate` mode
    where one-sidedness matters.

  - `log_I0_safe(c)` — deprecated alias for `log_I0_upper`, with a
    `[[deprecated]]` attribute so any remaining callers get a compile
    warning.

Call sites:

  - batch confluent branch in `evaluate_normalized`: uses `log_I0_value`.
  - incremental q=0 path in `initialize_compute_S_` and
    `remove_last_node_compute_S_`: uses `log_I0_value`.
  - `certified_log_prefactor`: uses `log_I0_upper` (one-sided needed).
  - `certified_log_tail_residual`: uses `log_I0_value` for the
    R_{N+1} subtraction; the prefactor's `log_I0_upper` already
    supplies the conservative margin in the product, so a second
    pad here would over-inflate without improving conservativeness.

Verified end-to-end:
- `evaluate({50.0, 50.0})` at c=50: rel error 2.0e-15 against e^50
  (was 1.7e-8).
- `initialize_normalized({50.0})` + `compute_full_value()` at c=50:
  rel error 1.6e-15 against e^50 (was 1.7e-8).

New CI subtests B2.1 and B2.2 in Group 6 lock these in.

**A2 — Negative-index guard on `ExpDivDiff::get_bessel_ratio`.**  The
batch class accessor did not check `n >= 0`; passing -1 was undefined
behavior (ASAN flagged it immediately).  Now throws `std::out_of_range`
with a clear message.  New subtest A2.5 in Group 6 verifies the throw.

**A3 — Remaining "true relative-error guarantee" wording removed.**
Three remaining places in the incremental header still used wording
stronger than the floating-point implementation justifies: the enum's
secondary docstring, the in-evaluator comment block, and the
constructor's `@param stopping_mode` doc.  All now say "conservative
floating-point" and direct users to interval-arithmetic backends for
proof-level certification.

**B1 — Group 7 reports Chebyshev and MNP pass rates separately.**
The summary previously printed only "Both within tol on N/M cases",
which obscured which method was responsible when one missed.  Now:

    Within tol=1e-9 (skipped K below |ref| floor 1e-14):
      Chebyshev: 14/14
      MNP      : 12/14
      Both     : 12/14

The shipped quick-grid now shows that Chebyshev passes 14/14 at
1e-9 in double, while MNP misses 2 of the catastrophic-precision-loss
cases that the manuscript explicitly discusses.

**B2 — CI subtests added for the bugs.**  Group 6 grew from 10 to 14
sub-tests with:
  - B2.1 batch exp[50,50] = e^50 at rel < 1e-12,
  - B2.2 incremental q=0 normalized+compute_full_value = e^50,
  - A2.5 negative-index throw on `get_bessel_ratio`,
  - C2.1 zero / negative tolerance rejected at construction.

All four would have caught the reviewer's findings.

**B3 — ASAN/UBSAN target documented.**  CI can now be run under
sanitizers with:

    g++ -std=c++17 -O1 -g -fsanitize=address,undefined \
        -fno-omit-frame-pointer -I. main.cpp -o ci_san
    ./ci_san

This is what flagged the A2 heap-buffer-overflow.  The CI passes
cleanly under both ASAN and UBSAN.

**C1 — `BesselRatiosMiller::compute_R(c=0)` dead branch removed.**
The function used to throw on `c <= 0` *and* have a "handle c=0
exactly" branch inside the small-c path.  The latter was
unreachable.  Now `c == 0` is allowed and returns the exact ratios
`{1, 0, 0, ...}`; only `c < 0` throws.

**C2 — Constructor input validation.**  `ExpDivDiff` and
`IncrementalExpDivDiff` now both reject:
  - `tolerance <= 0` (was silently propagated, giving NaN
    `log(tolerance)` inside conservative mode);
  - `max_iterations < 2` (batch) or `max_order < max_q + 2`
    (incremental, when specified explicitly).

**C3 — Exact `q=0` full-value path.**  `compute_full_value()` now
special-cases `current_q_ == 0` and returns `exp(x_0)` directly,
avoiding the round-trip through `scale_full(S, 0)`.  Mathematically
equivalent, but skips an unnecessary multiplicative round-off step.

Full CI after Round 7: **6/6 required groups PASS, Group 6 has 14/14
sub-tests, Group 7 reports Chebyshev 14/14 and MNP 12/14 separately.**
Clean under ASAN + UBSAN.

## Round 8 fixes (Round-7 review response)

Round-7 review flagged one deep numerical-conditioning issue plus several
smaller cleanups.  All addressed.

**A1 — Over-wide-interval failure (tight-interval fallback in batch
`evaluate()`).**  The reviewer's example exposed a real regime
failure: `ExpDivDiff<double>(-100, 100).evaluate({-1.0, 1.0})` returned
~5.5e25 instead of the correct (e - e^-1)/2 ≈ 1.175.  At c=1000 it
returned `inf`.  Cause: when the fixed interval is much wider than the
node spread, the normalized Chebyshev sum S becomes a tiny cancellation
residue that double precision cannot resolve, and multiplying it by
the large prefactor `I_0(c)/c^q` yields garbage.

Fix: at the top of batch `evaluate()`, detect the over-wide regime
(`c > 10 * (max(nodes) - min(nodes))`) and recurse on a tight local
`ExpDivDiff` centered on the node midpoint with half-width 1.5x the
node spread.  Guard against infinite recursion by checking that the
new c is strictly smaller than the current one.  After the fix:

  - `(-100, 100), evaluate({-1, 1})`: rel error 3.8e-16
  - `(-1000, 1000), evaluate({-1, 1})`: rel error 3.8e-16

Both at the double round-off floor.

The `evaluate_normalized` API does NOT do this fallback: a user
calling normalized form explicitly wants the fixed-interval
normalization (typically for ratio workflows where the scaling
factor cancels by design).

**A2 — All-equal full-value closed form, at any c.**  The reviewer
found that even after the Round-7 fix, batch and incremental
returned NaN/inf for all-equal nodes at large c:

  - `(-1000, 1000).evaluate({0.0, 0.0})` → NaN (truth = 1).
  - `IncrementalExpDivDiff(-1000, 1000, 10).initialize({0,0,0})` → inf (truth = 0.5).

Cause: those paths still went through `S * scale_full` even though
the closed form exp[x, …, x] = exp(x_0) / q! is exact.  Round-7 only
short-circuited q=0; q≥1 with all-equal nodes still round-tripped
through `S * (I_0(c) * exp(d) / c^q)`, which overflows at large c.

Fix: detect all-equal in every full-value formation site and use the
closed form there.  Sites updated:

  - batch `evaluate()` — at the top, before any other path.
  - incremental `initialize_compute_S_` commit block.
  - incremental `add_node_compute_S_` commit block.
  - incremental `remove_last_node_compute_S_` commit block.
  - incremental `compute_full_value()`.

After the fix:

  - `(-1000, 1000).evaluate({0})` → 1.0 exactly.
  - `(-1000, 1000).evaluate({0, 0})` → 1.0 exactly.
  - `(-1000, 1000).evaluate({0, 0, 0})` → 0.5 exactly.
  - `IncrementalExpDivDiff(-1000, 1000, 10).initialize({0, 0})` → 1.0.
  - same, `initialize({0, 0, 0})` → 0.5.

**A3 — CI tests for the dangerous regimes.**  Four new sub-tests in
Group 6:

  - R8.1 batch over-wide `(-100, 100), exp[-1, 1]` at rel < 1e-12.
  - R8.2 batch over-wide `(-1000, 1000), exp[-1, 1]` at rel < 1e-12.
  - R8.3 batch all-equal at c=1000 for q=0, 1, 2.
  - R8.4 incremental all-equal at c=1000 for q=1, 2.

All four would have caught the round-7 misses.

**B1 — `CertifiedTailBound` removed from internal/test code.**  All
in-driver uses (`main.cpp`) and one straggling comment in the
incremental header now use `ConservativeTailEstimate`.  The
`CertifiedTailBound` symbol still exists as a deprecated alias for
source-compatibility, but is no longer used internally.

**B2 — Stale comment fixed.**  The batch all-equal branch comment
previously said "log I_0(c) is the conservative upper estimate used
elsewhere" — but the code (correctly, after Round 7) uses
`log_I0_value`.  Comment now states the opposite and explains why
using `log_I0_upper` here would systematically under-estimate S.

**B3 — Empty-vector check moved before size arithmetic.**  Batch
`evaluate()` now does `if (nodes.empty()) throw` before computing
`nodes.size() - 1` (which would underflow as `size_t` before
conversion to `int`).  Minor but pedantically cleaner.

**B4 — `get_scaling_factor` validates q; new `log_scaling_factor`
helper.**  The accessor now throws `std::out_of_range` for q < 0.
A new public `log_scaling_factor(q)` returns `d + log(I_0(c)) - q*log(c)`
in log space, for callers who want to combine with a normalized sum
without forming the (potentially overflowing) linear prefactor.

Full CI after Round 8: **6/6 required groups PASS, Group 6 has 18/18
sub-tests, ASAN+UBSAN clean.**  The two regimes that produced garbage
returns at the public API (over-wide intervals, all-equal at large c)
are both now correct to double round-off, and both locked in by CI
subtests.

## Round 8 fixes (Round-7 review response)

The Round-7 review pinpointed the deepest remaining issue: when the
fixed Chebyshev interval is much wider than the active node spread,
the normalized sum `S = full / (exp(d) * I_0(c) / c^q)` becomes a
near-zero cancellation residue.  Double precision cannot resolve a
40-order-of-magnitude cancellation, and the subsequent multiplication
by `exp(d) * I_0(c) / c^q` amplifies round-off into garbage.  The
reviewer's three concrete cases all produced nonsense at varying
severity (5.56e25 in place of 1.175, then 4.8e68, then inf).
Addressed below.

**A1 — Batch `evaluate()` delegates to a tight temporary interval
when the constructor interval is much wider than the node spread.**

Heuristic: if `c_ > 10 * (max_node - min_node)`, build a temporary
`ExpDivDiff<Real>` centered on the node midpoint with half-width
`1.5 * spread`, and recurse.  The 10x threshold roughly corresponds
to "scaling factor magnitude beyond the natural divided-difference
scale exceeds the round-off floor."  The recursion is guarded by
`tight_c < c_`, so it terminates after one level.

For the new `evaluate()` API:

  - Reviewer case `ExpDivDiff<double>(-100, 100).evaluate({-1, 1})`:
    was 5.56e25, now 1.1752 (rel err 3.8e-16 vs `(e - 1/e)/2`).

  - `ExpDivDiff<double>(-1000, 1000).evaluate({-1, 1})`: was `inf`,
    now 1.1752 (rel err 3.8e-16).

Users who deliberately want fixed-interval Chebyshev evaluation for
batch normalized workflows continue to use `evaluate_normalized()`,
which keeps the original semantics (the normalized sum is
interval-dependent by definition).

**A2 — All-equal-node closed form in every full-value path.**  For
nodes all equal to `x_0`, the divided difference is exactly
`exp(x_0) / q!` by L'Hopital applied to the divided-difference
limit.  Now special-cased before the Chebyshev sum in:

  - `ExpDivDiff::evaluate(nodes)` (batch).
  - `IncrementalExpDivDiff::initialize(...)` (commit path).
  - `IncrementalExpDivDiff::add_node(...)` (commit path).
  - `IncrementalExpDivDiff::remove_last_node()` (commit path).
  - `IncrementalExpDivDiff::compute_full_value()`.

Reviewer's failing cases:

  - `ExpDivDiff<double>(-1000, 1000).evaluate({0, 0, 0})`: was NaN,
    now 0.5 exactly.

  - `IncrementalExpDivDiff<double>(-1000, 1000, 10).initialize({0,0})`:
    was inf, now 1.0 exactly.

  - `.initialize({0, 0, 0})`: was -inf, now 0.5 exactly.

The normalized-sum form `S = exp(c y_0) c^q / (q! I_0(c))` may still
underflow for the all-equal case at very wide intervals, but that is
a representable-range limit on `S` itself, not a bug in the
mathematics -- and the full-value path now bypasses `S` entirely.

**A3 — CI subtests for the dangerous regime.**  Five new sub-tests
in Group 6 lock the above in:

  - A1.2: `evaluate({-1, 1})` at c=100,
  - A1.3: same at c=1000,
  - A1.4: `evaluate({0, 0, 0})` at c=1000 (closed form),
  - A1.5: incremental `initialize({0, 0})` at c=1000,
  - A1.6: incremental `initialize({0, 0, 0})` at c=1000.

All pass to round-off.  Group 6 is now 23/23.

**B1 — Internal CertifiedTailBound references retired.**  The CI
driver and all sub-tests now use `StoppingMode::ConservativeTailEstimate`
exclusively.  `CertifiedTailBound` remains only as a `[[deprecated]]`
enum-value alias in the public header for source compatibility;
no code in the patch set references it.

**B2 — Stale "conservative upper estimate" comment in the batch
confluent branch corrected.**  The comment block now correctly says
the confluent branch uses `log_I0_value` (accurate) and explains
that the conservative `log_I0_upper` exists only for one-sided
tail-bound use in `ConservativeTailEstimate` mode.

**B3 — Empty-nodes check moved before `nodes.size() - 1`** in the
batch `evaluate()` to avoid the unsigned-wrap-then-cast hazard.

**B4 — `get_scaling_factor(q)` validates q.**  Throws
`std::out_of_range` for negative q.  Added `log_scaling_factor(q)`
companion for users who need to compose the scaling factor with
log-space sums; same validation.

## Final status

Full CI: **6/6 required groups PASS, 23/23 Group 6 sub-tests, Group
7 reports Chebyshev 14/14 (both double and long double), MNP 12/14**
(the two long-standing MNP precision-loss cases the manuscript
discusses).  Clean under ASAN + UBSAN.

Built and tested with:

    g++ -std=c++17 -O2 -I. main.cpp -o ci_runner
    ./ci_runner

    # Recommended sanitizer run:
    g++ -std=c++17 -O1 -g -fsanitize=address,undefined \
        -fno-omit-frame-pointer -I. main.cpp -o ci_san
    ./ci_san

What the implementation now handles correctly:

  - exp[x_0, ..., x_q] for any nodes within the constructor interval,
    even when the interval is much wider than the node spread.
  - All-equal nodes, including at extreme c, exactly via the
    closed-form `exp(x_0) / q!`.
  - Both heuristic and conservative-floating-point stopping rules.
  - Append / LIFO-remove with O(N(q, c, eps)) cost per move.
  - Normalized APIs that avoid forming the overflow-prone scaling
    factor.
  - Mode rollback after exceptions in any state-changing call.

What the implementation deliberately does *not* claim:

  - The `ConservativeTailEstimate` mode is not a proof-level
    interval-arithmetic certificate; it is a conservative
    floating-point implementation of the analytic tail bound.
  - The normalized-sum API at fixed wide intervals still has the
    fundamental cancellation-residue limit; for full-value
    computation use the standard `evaluate()`, which handles this
    transparently.

## Round 9 fixes (Round-8 review response)

The Round-8 review found three real correctness holes the CI was
missing.  All addressed; the suite now catches each on CI.

**A1 — Batch `evaluate()` validates every node against `[a, b]`
and finiteness *before* any branching.**  Previously the all-equal
closed form fired before range validation, so
`ExpDivDiff<double>(-1, 1).evaluate({5.0, 5.0})` returned `exp(5)`
instead of throwing; and the tight-interval fallback also fired
before validation, so `evaluate({5.0, 5.08})` silently evaluated
through a tight temporary on out-of-range nodes.  The validation
loop now runs first and rejects either case with `std::out_of_range`.

**A2 — Series-conditioning gate in `evaluate_from_cache()` /
`evaluate_from_cache_local()`.**  The incremental class cannot
auto-tighten its interval (the cache is bound to the constructor's
`[a, b]`), so distinct nodes far inside a wide interval previously
returned garbage at full-value scale:
`IncrementalExpDivDiff<double>(-100, 100, 10).initialize({-1, 1})`
returned ~3.5e25 instead of `(e - 1/e)/2 ≈ 1.1752`; at `[-1000, 1000]`
it returned `inf`.

The fix tracks
$$
  \kappa = \frac{|D_q^{(0)}| + \sum_n |2 R_n D_q^{(n)}|}{|S|}
$$
alongside the signed Chebyshev sum.  When `κ * ε_mach > 1e-6` (i.e.
fewer than ~6 resolved digits in S), both eval variants throw a
clear `runtime_error` with the computed `κ * ε_mach` value and a
suggestion to reconstruct on a tighter interval, raise precision,
or use the batch `evaluate()` API which tight-falls-back
automatically.  The check fires regardless of the `normalized` flag,
because the round-off issue is in S itself, not in the multiplicative
scale-up step.

All-equal nodes are short-circuited *before* the Chebyshev sum in
`initialize_compute_S_` and `add_node_compute_S_`: S is computed in
log space from the analytic closed form
$\log S = (x_0 - d) + q \log c - \log(q!) - \log I_0(c)$
and the full value follows the existing $\exp(x_0)/q!$ closed
form.  This avoids both the cancellation-residue problem and the
conditioning gate's false positive on the (correct) round-off
floor of the Chebyshev sum for all-equal nodes.

The conservative-tail-estimate stopping rule does not protect
against this failure mode -- it bounds the omitted Chebyshev tail,
not round-off in a nearly-cancelling sum.  R9.3 verifies the
gate fires in conservative mode as well.

**A3 — NaN / non-finite inputs rejected everywhere.**

  - Both constructors now reject non-finite `a` or `b` with
    `invalid_argument` *before* the `b > a` check (which is false
    against NaN and would have let `NaN` intervals slip through).

  - Every node-range check in both classes now combines
    `std::isfinite` with the `[a, b]` test, so NaN, +inf, and -inf
    are all caught instead of being treated as "out of range" only
    when they compare meaningfully.

**B1 — CI subtests cover all the new guard regimes.**  Group 6 has
five new sub-tests:

  - R9.1: incremental `initialize({-1, 1})` at c=100 -- must throw
    conditioning.
  - R9.2: incremental `add_node(1.0)` after `initialize({0})` at
    c=1000 -- must throw conditioning.
  - R9.3: same R9.1 case with `ConservativeTailEstimate` mode --
    must throw (mode doesn't protect against round-off).
  - R9.4: batch `evaluate({5,5})` and `evaluate({5, 5.08})` on
    `[-1, 1]` -- must both throw `out_of_range`.
  - R9.5: NaN inputs at constructor, batch evaluate, and
    incremental initialize -- all must throw.

Group 6 now has 24/24 sub-tests; full CI passes under both normal
build and ASAN/UBSAN.

**C cleanups.**  The duplicate R8.1/R8.2 over-wide pair and the
R8.3/R8.4 all-equal pair (which were essentially the same as
A1.2/A1.3 and A1.4-6) have been removed; the slot is reused for the
five R9.x sub-tests.  The stale "implementation does not branch on
the coincident-node case" docstring in
`IncrementalExpDivDiff::initialize` has been replaced with text
that accurately describes the all-equal log-space short-circuit
plus the conditioning gate for distinct nodes.

## Not done

**C — Lazy Bessel-ratio cache in the batch constructor.**  The
constructor still eagerly precomputes `R_n(c)` for `n` up to
`max(100, 3*c + 50)`.  When the user constructs on a wide interval
but only ever calls `evaluate()` on tight node sets (so the
tight-interval fallback fires), that work is wasted.  Not a
correctness issue.  Deferring it to first use would also need a
lock for the multi-threaded use case, which adds enough complexity
that we leave it for a future revision.

## Final status

Full CI: **6/6 required groups PASS, 24/24 Group 6 sub-tests, Group
7 reports Chebyshev 14/14 (double and long double), MNP 12/14**
on the shipped quick-grid reference.  Clean under ASAN + UBSAN.

What the implementation now handles correctly:

  - Batch full-value `exp[x_0, ..., x_q]` at any nodes within
    `[a, b]`, regardless of how much wider the interval is than the
    node spread (tight-interval fallback).
  - Incremental full-value `exp[x_0, ..., x_q]` when the interval
    is appropriate for the active node spread; throws with a clear
    diagnostic otherwise.
  - All-equal nodes at any c -- exact via closed form
    $\exp(x_0)/q!$, with normalized S also computed via closed form
    in log space.
  - Both heuristic and conservative-floating-point stopping rules.
  - Append / LIFO-remove with O(N(q, c, eps)) cost per move.
  - Mode rollback after exceptions in any state-changing call.
  - All node and interval inputs validated against finiteness and
    range *before* any algorithmic branch.

Recommended build:

    g++ -std=c++17 -O2 -I. main.cpp -o ci_runner
    ./ci_runner

Recommended sanitizer sweep:

    g++ -std=c++17 -O1 -g -fsanitize=address,undefined \
        -fno-omit-frame-pointer -I. main.cpp -o ci_san
    ./ci_san

## Round 10 fixes (Round-9 review response)

Four API-robustness issues, all addressed.

**A1 — `evaluate_normalized()` now has a conditioning gate of its own.**
Previously the gate was only inside the incremental code path; batch
normalized callers got no protection.  Now `evaluate_normalized` tracks
`abs_sum = Σ|2 R_n D_q^(n)|` alongside the signed sum, computes
`kappa·eps_mach` after convergence, and throws `runtime_error` when
that estimate exceeds `1000 * tolerance_`.  The factor of 1000 leaves
two-and-a-bit digits of slack for the loose bound (kappa·eps is an
upper bound, not the actual error), avoiding spurious throws on
typical well-conditioned high-q computations while still catching
the reviewer's catastrophic case (`ExpDivDiff<double>(-100, 100)
.evaluate_normalized({-1, 1})` now throws instead of returning
numerical noise that scales up to ~5.6e25).

**A2 — Batch `evaluate()` decides by conditioning, not by a c/spread
threshold.**  The old "c > 10 * spread" heuristic missed the boundary
regime: at c=20 with nodes {-1, 1} (ratio 10x), rel err was ~3.7e-10
despite tolerance_ = 1e-14; at c=15, ~1e-12.  Restructured:

  - Added a private `evaluate_normalized_with_kappa` that returns
    both `S` and `kappa·eps_mach` without throwing.
  - `evaluate_normalized` (public) is a thin wrapper that adds the
    1000x throw.
  - `evaluate` calls the internal directly.  If `kappa·eps > 1000 *
    tolerance_`, it builds a tight temporary on
    `[mid - 1.5·spread, mid + 1.5·spread]` and recurses -- but only
    when `tight_c < c_`, i.e. when the rebuild actually shrinks the
    interval.  If the constructor interval is already as tight as
    the rebuild would be (spread >= c_), the fixed-interval result
    is returned, since the conditioning is intrinsic to the problem
    at this precision rather than an over-wide-interval artifact.

  - Pre-screen for absurd `c_ > 100 * spread`: skip the fast path
    entirely and go straight to tight.  Without this, constructing
    `ExpDivDiff<double>(-1e8, 1e8)` and calling `evaluate({-1, 1})`
    would have spent O(c) Chebyshev iterations (millions) in the
    fast path before deciding.

  Reviewer's failing cases at c=20 and c=15 now hit round-off
  (rel err = 3.8e-16).

**A3 — All-equal closed form uses `std::lgamma`, not explicit
factorial.**  The old form `exp(x_0) / fact` overflowed `q!` at q=170
even when `exp(x_0)/q!` was still representable.  Now all five
all-equal sites (one batch, four incremental) compute
`log_v = x_0 - lgamma(q + 1)` and return `exp(log_v)`.  The reviewer's
`evaluate(std::vector<double>(172, 0.0))` case (true value 1/171! ≈
8.06e-310, subnormal) now returns 8.06e-310 instead of 0.  `std::exp`
handles both overflow (`-> +inf`) and underflow (`-> 0` or subnormal)
correctly for the boundary cases.

**A4 — Lazy Bessel-ratio cache in the batch constructor.**  Eager
seeding of `bessel_ratios_` to size ~`3c + 50` was both a memory
hazard (2.4 GB for c = 1e8) and a redundant cost when `evaluate()`
will immediately tight-fallback for over-wide constructions.

  - Constructor only stores `a_, b_, c_, d_, tolerance_, max_iterations_`
    plus the lazy-init flag.  No Bessel work happens up front.

  - New `get_I0_c()` returns `I_0(c_)`, populating `I0_c_` on first
    call and setting `bessel_I0_initialized_ = true`.
    `get_scaling_factor` now calls `get_I0_c()` (and is no longer
    `const`, since the lazy init mutates state).

  - `get_bessel_ratio` already handled an empty cache correctly via
    its on-demand expansion; no change needed there.

  Not thread-safe.  The same constraint as `std::vector::push_back`.
  Documented in the constructor docstring; users sharing an instance
  across threads must serialize externally or pre-touch the cache
  before sharing.

## Choice of conditioning threshold

For batch `evaluate_normalized`, threshold = `1000 * tolerance_`.
For incremental, threshold = `1e-6` (loose floor; catches only
catastrophic cases).

These are different because:

  - The batch API has a tight-interval auto-fallback in `evaluate()`,
    so being more strict in `evaluate_normalized` is cheap: any
    spurious throws are caught by `evaluate` and re-routed.
  - The incremental API cannot auto-fallback (the cache is bound to
    the constructor interval), so a throw is a hard failure for the
    caller.  Setting incremental's threshold equal to batch's would
    spuriously fail typical incremental workflows at moderate c (e.g.
    Scenario 2 in the incremental_path test, where c=20 nodes have
    kappa·eps ~ 1e-10 but the actual result is correct to double
    precision's natural floor).

Both thresholds are documented in the respective error messages so
the caller knows what to compare against.

## Test calibration

`test_progressive_append` now accepts a comparison tolerance
parameter (default 1e-10 as before) and Scenario 2 passes 1e-8 to
reflect that at c=20 both batch and incremental have a round-off
floor of ~1e-10, so their pairwise agreement is bounded by ~2x that.
The looser tolerance is well above the round-off floor and well
below the rel-err scale at which a real algorithmic bug would
manifest.

## Final status

Full CI after Round 10: **6/6 required groups PASS, 28/28 Group 6
sub-tests, Group 7 reports Chebyshev 14/14 (double and long double),
MNP 12/14** on the shipped quick-grid reference.  Clean under
ASAN + UBSAN.

Group 6 now contains four new R10 sub-tests in addition to the
prior 24:
  - R10.1: `evaluate_normalized` over-wide must throw.
  - R10.2: `evaluate({-1, 1})` at c=20 and c=15 must hit round-off.
  - R10.3: q=171 all-equal must give 1/171! ≈ 8e-310 (subnormal).
  - R10.4: `ExpDivDiff(-1e8, 1e8)` must construct without OOM and
    `.evaluate({-1, 1})` must work via tight fallback.

Build:

    g++ -std=c++17 -O2 -I. main.cpp -o ci_runner
    ./ci_runner

Sanitizer sweep:

    g++ -std=c++17 -O1 -g -fsanitize=address,undefined \
        -fno-omit-frame-pointer -I. main.cpp -o ci_san
    ./ci_san

## Round 11 fixes (Round-10 review response)

Four small but real fixes from the Round-10 review.  All addressed.

**A1 — `evaluate_normalized_with_kappa` all-equal branch sets its
output parameters.**  Previously the closed-form return path left
both `kappa_eps_out` and `n_out` uninitialized, so the public
`evaluate_normalized()` wrapper read uninitialized memory when
deciding whether to throw on conditioning.  Now the branch
explicitly writes `0` to both before returning the analytic S.  CI
sub-test R11.1 locks this in: `evaluate_normalized({0,0,0}, &n)`
on `[-1000, 1000]` must return finite S and `n == 0`.

**A2 — Free `bessel_ratio()` helper routed through Miller.**  The
previous implementation `bessel_In(n, c) / bessel_I0(c)` overflowed
both numerator and denominator separately at `c >~ 700` (double),
returning `NaN` even though the ratio itself is well-bounded.  Now
the helper calls `BesselRatiosMiller::compute_R(c, n, Nstart)` and
returns the n-th element.  `bessel_ratio(10, 1000)` now returns
0.951206 (the correct value) instead of `NaN`.  The optional
`I0_c` parameter is kept in the signature for source compatibility
but is ignored (marked `/*ignored*/` in the parameter list).
The class method `ExpDivDiff::get_bessel_ratio` was already correct
and is unchanged.  CI sub-test R11.2 verifies.

**A3 — `evaluate_normalized` pre-screens absurdly wide intervals
before any Bessel work.**  Without this, `ExpDivDiff<double>(-1e8,
1e8).evaluate_normalized({-1, 1})` would have entered the Chebyshev
path; the first `get_bessel_ratio(1)` triggers
`BesselRatiosMiller::compute_R` with `recommended_Nstart` of order
~c, allocating ~800 MB and running ~`1e8` recurrence steps before
the conditioning gate even gets a chance to fire.  Now: when `c_ >
100 * spread` and spread > 0 (the all-equal closed form has
already short-circuited the spread == 0 case at this point in the
function), we throw immediately with a clear message.

The pre-screen is placed *after* the all-equal closed form so that
`evaluate_normalized({0, 0, 0})` on `[-1e8, 1e8]` still succeeds
analytically without entering the Chebyshev path.  Batch
`evaluate()` already had a similar pre-screen at the same 100x
threshold; the two now match in semantics.  CI sub-test R11.3
verifies that the pre-screen triggers, R11.1 verifies that
all-equal still bypasses it.

**B1 — Stale "guarantee" wording in `set_stopping_mode`.**  One
remaining docstring said the conservative mode "provides a
relative-error guarantee on exit," inconsistent with the rest of
the codebase which now correctly says "conservative floating-point
estimate."  Wording updated to match.

**B2 — Inline scaling factored through `scale_full`.**  The two
`evaluate_from_cache` variants reimplemented
`exp(d) * I_0(c) / c^q * S` inline at their return paths instead
of calling the existing `scale_full(S, q)` helper.  Replaced with
the helper call.  Removes duplicate logic.

**B3 — `reset()` clears diagnostic state.**  Previously it left
`full_value_valid_` and `last_used_order_` at whatever values the
last evaluation set, so e.g. `last_used_order()` after `reset()`
could report a stale order (the docstring said `-1` meant "no
evaluation performed").  Now `reset()` explicitly resets both.  CI
sub-test R11.4 verifies.

## Not done

**B4 — Splitting Group 6 into seven focused groups.**  The reviewer
suggested decomposing the now-32-subtest Group 6 into
`input_validation`, `closed_form_confluent`,
`batch_conditioning`, `incremental_conditioning`, `normalized_api`,
`exception_safety`, and `conservative_tail_estimate`.  This is an
agreed-good idea, but a CI driver restructure rather than a
correctness fix; deferred for a separate pass.

## Final status

Full CI after Round 11: **6/6 required groups PASS, 32/32 Group 6
sub-tests, Group 7 reports Chebyshev 14/14 (double and long double),
MNP 12/14** on the shipped quick-grid reference.  Clean under
ASAN + UBSAN.

The four R11 sub-tests in Group 6 lock in this round's fixes:
  - R11.1: normalized all-equal branch sets n_out and (internally)
    kappa_eps_out.
  - R11.2: free `bessel_ratio(10, 1000)` returns 0.951206, not NaN.
  - R11.3: `evaluate_normalized` on `(-1e8, 1e8)` pre-screens
    before any Bessel work.
  - R11.4: `reset()` returns `last_used_order()` to -1.

Build:

    g++ -std=c++17 -O2 -I. main.cpp -o ci_runner
    ./ci_runner

Sanitizer sweep:

    g++ -std=c++17 -O1 -g -fsanitize=address,undefined \
        -fno-omit-frame-pointer -I. main.cpp -o ci_san
    ./ci_san

## Round 12 fixes (Round-11 review response)

Three must-fix items plus two quality items.  All addressed.

**A1 — Incremental constructor caps c_ at 1e6 and adds an auto-N_max
ceiling.**  Two layers of defence against absurd constructor
intervals like `IncrementalExpDivDiff<double>(-1e7, 1e7, 10)`:

  - Hard cap on `c_ > 1e6`.  `precompute_bessel_ratios()` calls
    `BesselRatiosMiller::compute_R` with `recommended_Nstart ~ 2c`
    regardless of `max_order`, so the user cannot escape the
    O(c) Miller recurrence cost by passing a small `max_order`.
    Above 1e6 we refuse construction with a clear message pointing
    to batch `evaluate()` (which tight-falls-back automatically
    and so never needs the Miller cache at large c).

  - When `max_order` is left to auto, an additional ceiling at
    `N_width > 100000` throws.  This catches the regime
    `c \in (~1e5, 1e6)` where the cap-on-c does not fire but the
    auto N_max would still allocate hundreds of MB and take many
    seconds in the recurrence.  The user can pass `max_order`
    explicitly to opt in to the cost up to the c cap.

The previous behaviour (hang or huge silent allocation) is replaced
by `std::invalid_argument` with an actionable message.

**A2 — Public Bessel helpers reject non-finite arguments.**  Added
`std::isfinite` checks at the top of `bessel_I0`, `bessel_In`,
`bessel_ratio`, and `BesselRatiosMiller::compute_R`.  Previously
NaN slipped through (`c < 0` is false for NaN) and produced NaN
return values silently; now each throws `std::invalid_argument`
with a clear message.  This matches the input validation the rest
of the library now has at the evaluator entry points.

**B1 — `recommended_Nstart` is now overflow-safe.**  The previous
`static_cast<int>(2 * c)` would have produced
implementation-defined behaviour for huge finite c (the C++ cast
of a finite floating value that does not fit in `int` is UB until
C++23, and only "saturating-on-truncation" thereafter).  Now:

  - Computation done in `long double` throughout.
  - NaN/inf c rejected with `invalid_argument`.
  - Result compared against `numeric_limits<int>::max()` before
    cast; if it would overflow, throws `std::overflow_error` with
    a clear message.

**B3 — Deprecated `CertifiedTailBound` alias removed.**  The
`enum class StoppingMode` now contains only the two semantically
honest values, `HeuristicFiveTerm` and `ConservativeTailEstimate`.
The associated docstring no longer references the old name.  Any
external code still using `CertifiedTailBound` will fail to compile
with a clear "no member named X" error rather than silently using
the deprecated alias.

## Not done

**B2 — Splitting Group 6 into seven smaller groups.**  The
reviewer's structural recommendation (`input_validation`,
`closed_form_confluent`, `batch_conditioning`, etc.) is a CI
driver restructure rather than a correctness fix.  Group 6 now
holds 35 sub-tests covering essentially every regression we have
found, which makes diagnostics harder than they should be.
Deferred for a future pass; the current single-Group structure
still passes cleanly and reports per-subtest names so the failing
case is identifiable from the output.

## Final status

Full CI after Round 12: **6/6 required groups PASS, 35/35 Group 6
sub-tests, Group 7 reports Chebyshev 14/14 (double and long
double), MNP 12/14** on the shipped quick-grid reference.  Clean
under ASAN + UBSAN.

The three R12 sub-tests in Group 6:
  - R12.1: incremental constructor on (-1e7, 1e7) throws fast.
  - R12.2: `bessel_I0`, `bessel_In`, `bessel_ratio`, `compute_R`
    all reject NaN.
  - R12.3: `recommended_Nstart(2e10, 100)` throws
    `overflow_error`.

What this iteration of the code now handles correctly:

  - Batch full-value `exp[x_0, ..., x_q]` for any nodes within
    `[a, b]`, including over-wide and absurdly-over-wide
    intervals (tight-interval fallback or upfront throw with
    clear diagnostic).
  - Incremental full-value `exp[x_0, ..., x_q]` for any reasonable
    fixed interval (c <= 1e6); over-wide intervals throw at
    construction or evaluation time, never silently.
  - All-equal nodes at any c -- exact via closed form
    `exp(x_0) / q!` computed in log space, accurate out to
    double's subnormal range (q ~ 200).
  - Both heuristic and conservative-floating-point stopping rules.
  - Append / LIFO-remove with O(N(q, c, eps)) cost per move.
  - Mode rollback after exceptions in any state-changing call.
  - Input validation against finiteness, range, and conditioning
    at every public entry point.

What this iteration of the code does *not* claim:

  - The conservative tail estimate is not a proof-level
    interval-arithmetic certificate; it is a conservative
    floating-point implementation.  For proof-level certification
    substitute Arb / MPFI / Boost.Interval for the Bessel
    quantities.
  - The normalized fixed-interval API at over-wide intervals has
    unavoidable conditioning limits in finite precision; the
    library throws when the bound on resolved digits in S falls
    below 100x the requested tolerance (batch) or below ~6
    resolved digits (incremental, hard floor).
  - Thread-safety: the batch class uses lazy caching that is not
    thread-safe; the incremental class's `add_node`/`remove`
    sequences are inherently stateful.  Use distinct instances
    per thread or serialize access externally.

Build:

    g++ -std=c++17 -O2 -I. main.cpp -o ci_runner
    ./ci_runner

Sanitizer sweep:

    g++ -std=c++17 -O1 -g -fsanitize=address,undefined \
        -fno-omit-frame-pointer -I. main.cpp -o ci_san
    ./ci_san

## Round 13 fixes (Round-12 review response)

Four public-API edge cases, all addressed.  Per the reviewer this
is the last round on correctness; remaining work is documentation,
examples, and the Group-6 split, which are not in this batch.

**A1 — `bessel_In` Miller seed computed in long double with both an
int-overflow check and a practical work cap.**  Previously
`static_cast<int>(Real(2) * x) + 50` was implementation-defined / UB
for huge finite x (UBSAN flagged `bessel_In<double>(1, 1e20)`), and
even where the cast happened to succeed it would have requested an
impossibly long inner loop.  Now the Miller start is computed in
long double, capped at 200 000 starting orders, and rejected with
`invalid_argument` past that.  The error message points the user
at the Bessel-ratio routines (which never materialize unnormalized
I_n at huge x).

**A2 — Public `bessel_ratio` capped at 100 000 standalone Miller
orders.**  `bessel_ratio(10, 1e7)` used to silently route into the
Miller backward recurrence at ~2e7 starting orders -- ~160 MB and
a multi-second hot loop, hidden behind a one-line helper call.
The cap lives in the public helper, not in `compute_R` itself,
because the lazy caches in `ExpDivDiff` and `IncrementalExpDivDiff`
have their own higher-level guards and may legitimately call
`compute_R` at moderate-to-large Nstart for in-class work.  Public
single-shot helpers that don't carry that context should refuse.

**A3 — Batch and incremental constructors compute the affine map
without (b - a)/2 overflow.**  `ExpDivDiff<double>(-1e308, 1e308)`
used to silently produce `c_ = +inf` because `b - a` overflowed
even though the endpoints are individually finite.  Now both
constructors compute

    c_ = b/2 - a/2
    d_ = b/2 + a/2

so the intermediate stays representable, and additionally verify
that `c_` and `d_` are finite afterwards (catching any extreme-Real
edge case where even the rearranged form overflows).  For double
endpoints at the type limit, `c_ = 1e308` is the right finite
value; the constructor accepts it and downstream tight-fallback /
conditioning gates handle the actual computation correctly.

**B1 — `recommended_Nstart` rejects negative c.**  Previously
`recommended_Nstart(-1, 10)` returned 60 -- inconsistent with
`compute_R` and `bessel_ratio`, both of which reject negative c.
Now matches them with the same `invalid_argument` message.  Also
rejects negative `nmax` for the same reason.

## Not done (deferred to documentation phase)

The reviewer's three documentation-phase items remain on the
backlog:

  - **B2 split Group 6 into seven smaller groups** (input
    validation, closed-form confluent, batch conditioning,
    normalized API, incremental conditioning, Bessel public API,
    exception safety, conservative tail estimate).  Group 6 now
    holds 39 sub-tests with per-subtest naming, so failures
    remain locatable from the output.  Driver restructure,
    not a correctness fix.

  - **B3 commit a tiny quick-reference pair for release CI.**
    `mpmath_reference.py --generate --quick` is ready for the
    release pipeline; the current default checkout just doesn't
    exercise it (Group 7 is marked informational).

  - **README, API documentation, example programs.**  Out of
    scope for this round.

## Final status

Full CI after Round 13: **6/6 required groups PASS, 39/39 Group 6
sub-tests, Group 7 reports Chebyshev 14/14 (double and long
double), MNP 12/14** on the shipped quick-grid reference.  Clean
under ASAN + UBSAN, including the previously-flagged `bessel_In(1,
1e20)` UBSAN integer-overflow site.

The four R13 sub-tests in Group 6:
  - R13.1: `bessel_In(1, 1e20)` throws `invalid_argument`.
  - R13.2: `bessel_ratio(10, 1e7)` throws `invalid_argument`.
  - R13.3: `ExpDivDiff(-1e308, 1e308).get_c()` is finite (= 1e308).
  - R13.4: `recommended_Nstart(-1, 10)` throws `invalid_argument`.

Cumulative regression coverage in Group 6 across rounds:
  - R8.x: input validation before all-branches dispatch.
  - R9.x: kappa-tracking conditioning gate, NaN rejection, etc.
  - R10.x: lazy batch cache, lgamma all-equal, normalized
    conditioning throw, c=20 boundary case.
  - R11.x: all-equal output parameters, free bessel_ratio at
    large c, normalized pre-screen, reset() diagnostic clear.
  - R12.x: incremental c-cap, Bessel NaN rejection,
    recommended_Nstart overflow.
  - R13.x: bessel_In huge-x, bessel_ratio work cap, batch
    affine overflow, recommended_Nstart negative-c.

Build:

    g++ -std=c++17 -O2 -I. main.cpp -o ci_runner
    ./ci_runner

Sanitizer sweep:

    g++ -std=c++17 -O1 -g -fsanitize=address,undefined \
        -fno-omit-frame-pointer -I. main.cpp -o ci_san
    ./ci_san

# Revision for SISC-M183776 resubmission (2026-10-27)

Changes driven by the referee reports.  The pre-revision sources are
kept in `../DDEF c++ (submitted-version backup)/`.

## Correctness fixes

- **mccurdy.hpp (MNP baseline).**  (1) The Taylor series for the scaled
  Opitz exponential stopped on a norm-wise test, leaving entries far from
  the diagonal (~1/(j-i)!) unconverged -- for j-i > ~13 they were still
  exactly zero.  Now converges entry-wise (needs >= q terms; the Taylor
  step uses the bidiagonal structure, O(q^2) per term).  (2) The whole
  Opitz matrix, superdiagonal included, was scaled by 2^-s, so entries
  ~2^{-s(j-i)} underflowed at large q*c.  Now only the nodes are scaled;
  the squaring step is applied to divided differences:
  exp[2w_i..2w_j] = 2^{-(j-i)} sum_k exp[w_i..w_k] exp[w_k..w_j].
  Result: MNP agrees with mpmath to ~1e-14 wherever the value is
  representable.  The previously reported "MNP failures" were artifacts.

- **chebyshev_exp_divdiff.hpp (batch evaluator), scaling.**  R_n(c)
  underflowed for small c / large n (e.g. q=150, c=0.1 returned a wrong
  value although exp[x] ~ 1e-263 is representable), and D_q^(n) >=
  2^{q-1} overflows for q > 1024.  Now: per-row binary exponents for the
  Chebyshev table, R_n accumulated as a product of rho_k = I_k/I_{k-1}
  in mantissa/exponent form, scaled running sums and prefactor.  Works
  for any q (tested q=3000).  New API: `evaluate_scaled`,
  `evaluate_normalized_scaled`, `evaluate_ratio`, `ScaledReal`.

- **chebyshev_exp_divdiff.hpp, conditioning.**  evaluate() could return a
  silently inaccurate value: its fallback interval [mid-1.5s, mid+1.5s]
  is not anchored at max x_i and often not narrower than [a,b], and when
  no narrower interval existed it returned the ill-conditioned sum.  Now:
  error indicator kappa*u*N (`last_error_estimate()`); if above the
  threshold (`set_max_error_estimate`, default 1e-10) re-run on
  [min x_i, max x_i]; if still above, throw.  The sweep also skips the
  identically-zero rows k > n.

- **chebyshev_bessel.hpp.**  `BesselRatiosMiller::compute_rho` (backward
  ratio recurrence, no overflow/underflow, no normalization).  Portable
  `chebyshev_bessel::cyl_bessel_i` (std on libstdc++, Boost on Apple
  libc++); all call sites switched.

## Tests

- `test_chebyshev_bessel_Rn.hpp` copied into this folder (it existed
  only in the Xcode copy, so this folder did not build stand-alone).
- Group 6 A2.4 is skipped when long double has no extra precision
  (Apple arm64), where it previously aborted the run.
- New required Group 8 (revision regressions): R1 referee parity
  counterexample exp[-1,1]=sinh 1; R2 R_n underflow (q=160, c=0.1);
  R3 q=2000 scaled value within mean-value bounds; R4 wide interval
  throws or is accurate; R5 MNP entry-wise convergence; R6 repeated
  nodes; R7 ratio of underflowing values (q=400); R8 rho products vs
  cyl_bessel_i.

Build (macOS needs Boost for cyl_bessel_i):

    g++ -std=c++17 -O2 -I. -I/opt/homebrew/include main.cpp -o ci_runner

Status: 7/7 required groups pass; Group 7 (mpmath) 14/14 for both
methods; clean under ASan+UBSan.

## Experiments

`experiments/scaling`, `experiments/accuracy`, `experiments/performance`
contain the drivers and plotting scripts for every figure and table of
paper_v18.

## Later in the same revision

- **Processing order (batch).**  The accuracy study found that a long
  leading run of nodes far below b (e.g. ascending input on a wide
  interval) gives large errors invisible to the final-row kappa (up to
  1e20 at q=300, c=100).  `ExpDivDiff` now processes nodes in
  nonincreasing order (`set_sort_nodes(false)` restores caller order).
  Over 10206 cases the error never exceeded 0.33 kappa*u*N afterwards.
- **New incremental class `chebyshev_stack.hpp` (`IncrementalStack`).**
  Weighted columns W_k^(n) = R_n D_k^(n) / sigma_k (diagonal similarity:
  no over/underflow at any q), contiguous column storage, every prefix
  gated on kappa*u*N at push time (push throws, stack unchanged), O(1)
  LIFO pop, initialize() in nonincreasing order.  Append 0.15-0.35 us,
  up to ~1000x faster than batch recomputation (q=800).
- **Legacy `IncrementalExpDivDiff`** marked superseded; NaN sums now throw
  instead of being returned (they passed the window test).
- Error messages print estimates with %.3Le instead of std::to_string.
- New required Group 9 (IncrementalStack).  Status: 8/8 required
  groups pass; clean under ASan+UBSan.
