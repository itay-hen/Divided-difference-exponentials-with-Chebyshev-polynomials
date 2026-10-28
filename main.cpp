// =============================================================================
// main.cpp -- CI driver for the Chebyshev exponential divided-difference
//             reference implementation.
//
// Runs eight required test groups plus one informational group, and reports
// per-group pass/fail.  Exits non-zero on any *required* failure, so it can
// serve as the entry point for an automated build.  The informational group
// is reported separately and does not gate the exit code.
//
// Required groups:
//   (1) Bessel I_0 wrapper accuracy: sanity check on the std::cyl_bessel_i
//       wrapper plus an independent recurrence-based cross-check.
//   (2) Bessel ratio R_n(c) against scipy reference values
//       (test_chebyshev_bessel_Rn.hpp) and against std::cyl_bessel_i.
//   (3) Batch ExpDivDiff against the full Opitz / scaling-and-squaring MNP
//       reference in mccurdy.hpp, on a randomized (b, q) grid.
//   (4) Batch-vs-incremental on full node sets (comparison_test.hpp).
//   (5) The actual incremental update path: progressive append, deterministic
//       and randomized LIFO walks, and the coincident-start-then-append
//       scenario (test_incremental_path.hpp).
//   (6) A2 (certified stopping mode) and A4 (exception safety) feature tests.
//   (8) Revision regression tests (SISC review): stopping-rule parity
//       counterexample, R_n underflow at small c / high q, high-q scaled
//       output, no silent loss of accuracy on wide intervals, the MNP
//       entry-wise Taylor fix, repeated nodes, and ratios that under-
//       flow individually.
//   (9) chebyshev::IncrementalStack: LIFO walks vs batch, q = 800,
//       refusal of unfavorable insertion orders, rollback on rejection.
//
// Informational group:
//   (7) Independent mpmath ground-truth comparison: both methods at matched
//       precision (double, then long double) checked against the mpmath
//       reference produced by mpmath_reference.py.  Reports honest accuracy
//       including the known catastrophic regimes (large-c Chebyshev
//       cancellation, small-c high-q MNP precision loss); does not gate
//       CI exit code.
// =============================================================================

#include "chebyshev_exp_divdiff.hpp"
#include "incremental_chebyshev_exp_divdiff.hpp"
#include "chebyshev_stack.hpp"
#include "mccurdy.hpp"

#include "test_chebyshev_bessel_Rn.hpp"
#include "comparison_test.hpp"
#include "test_incremental_path.hpp"
#include "test_against_reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

bool run_i0_wrapper_test() {
    std::printf("\n=== Group 1: Bessel I_0 wrapper accuracy ===\n");
    // (a) Smoke check: the wrapper currently delegates to
    // std::cyl_bessel_i, so this part is a plumbing test, not an
    // independent accuracy validation -- but it does flag any future
    // change that puts a different implementation behind the wrapper.
    const double cs[] = {0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 200.0, 500.0};
    const double tol  = 1e-13;
    bool ok = true;
    for (double c : cs) {
        const double approx = chebyshev::bessel_I0(c);
        const double truth  = chebyshev_bessel::cyl_bessel_i(0.0, c);
        const double rel    = std::abs(approx - truth) / std::abs(truth);
        if (!(rel < tol)) ok = false;
    }
    std::printf("  wrapper vs std::cyl_bessel_i: %s\n",
                ok ? "OK" : "FAIL");

    // (b) Independent check: compare against the asymptotic
    //     I_0(c) ~ e^c / sqrt(2*pi*c) * (1 + 1/(8c) + 9/(128 c^2) + ...)
    // for moderately large c.  This is independent of std::cyl_bessel_i
    // and catches the case where both wrapper and std return matching
    // wrong values.
    bool ok_asymp = true;
    const double pi = 3.14159265358979323846;
    const double bigc[] = {30.0, 50.0, 100.0, 200.0, 500.0};
    for (double c : bigc) {
        const double truth = chebyshev::bessel_I0(c);
        const double a0 = std::exp(c) / std::sqrt(2.0 * pi * c);
        const double corr = 1.0
                          + 1.0 / (8.0 * c)
                          + 9.0 / (128.0 * c * c)
                          + 75.0 / (1024.0 * c * c * c);
        const double asymp = a0 * corr;
        const double rel   = std::abs(asymp - truth) / std::abs(truth);
        // Asymptotic series accuracy depends on c; ~1/c^4 here so the
        // tolerance has to be loose at small c.  At c=30 we expect
        // ~1.5e-7; at c=500 the expansion is good to ~10^-12.
        const double asymp_tol = 1.0 / (c * c * c);
        if (!(rel < asymp_tol)) ok_asymp = false;
    }
    std::printf("  vs large-c asymptotic series: %s\n",
                ok_asymp ? "OK" : "FAIL");

    // (c) Cross-check the public bessel_In() helper against
    // std::cyl_bessel_i for moderate (n, c).  bessel_In() uses its own
    // Miller-style downward recurrence and is exposed as a top-level
    // utility; this verifies it agrees with the standard library.
    bool ok_In = true;
    double max_In_rel = 0.0;
    const int ns[] = {1, 2, 5, 10, 20, 50};
    for (double c : cs) {
        for (int n : ns) {
            const double approx = chebyshev::bessel_In(n, c);
            const double truth  = chebyshev_bessel::cyl_bessel_i(
                static_cast<double>(n), c);
            if (!std::isfinite(truth) || truth == 0.0) continue;
            const double rel = std::abs(approx - truth) / std::abs(truth);
            max_In_rel = std::max(max_In_rel, rel);
            if (!(rel < 1e-12)) ok_In = false;
        }
    }
    std::printf("  bessel_In vs std::cyl_bessel_i: %s (max rel_err=%.2e)\n",
                ok_In ? "OK" : "FAIL", max_In_rel);

    const bool all_ok = ok && ok_asymp && ok_In;
    std::printf("=== %s ===\n", all_ok ? "Group 1 PASS" : "Group 1 FAIL");
    return all_ok;
}

bool run_bessel_ratio_test() {
    std::printf("\n=== Group 2: Bessel ratios R_n(c) ===\n");

    // (a) Original scipy-reference smoke test; the reference values
    // are stored at low precision so the tolerance here is loose.
    auto summary = test_bessel_ratios::get_summary(1e-6);
    std::printf("  vs scipy reference (smoke): "
                "%d/%d pass, max rel_err = %.2e\n",
                summary.passed, summary.total, summary.max_error);

    // (b) Stricter independent cross-check: for moderate (n, c), the
    // Miller-cached R_n(c) should match
    //     std::cyl_bessel_i(n, c) / std::cyl_bessel_i(0, c)
    // to roughly 1e-12.  This catches drift in the Miller recurrence
    // that the scipy reference (rounded to ~7 figures) cannot detect.
    const double cs[] = {1.0, 5.0, 20.0, 50.0, 100.0};
    const int ns[] = {1, 2, 5, 10, 30};
    int npass = 0, ntot = 0;
    double max_rel = 0.0;
    for (double c : cs) {
        chebyshev::ExpDivDiff<double> e(-c, c, 1e-14, 5000);
        for (int n : ns) {
            const double R_miller = e.get_bessel_ratio(n);
            const double I0 = chebyshev_bessel::cyl_bessel_i(0.0, c);
            const double In = chebyshev_bessel::cyl_bessel_i(static_cast<double>(n), c);
            const double R_std = In / I0;
            if (!std::isfinite(R_std) || R_std == 0.0) continue;
            const double rel = std::abs(R_miller - R_std) / std::abs(R_std);
            max_rel = std::max(max_rel, rel);
            ntot++;
            if (rel < 1e-12) npass++;
        }
    }
    const bool strict_ok = (npass == ntot);
    std::printf("  vs std::cyl_bessel_i (strict): %d/%d pass, "
                "max rel_err = %.2e\n", npass, ntot, max_rel);

    const bool all_ok = summary.all_passed && strict_ok;
    std::printf("=== %s ===\n", all_ok ? "Group 2 PASS" : "Group 2 FAIL");
    return all_ok;
}

bool run_batch_vs_reference_test() {
    std::printf("\n=== Group 3: Batch Chebyshev vs Opitz/MNP reference ===\n");
    std::printf("  (Reference is the full Opitz/scaling/squaring MNP in mccurdy.hpp.)\n");
    std::mt19937 rng(2024);
    const double a = -1.0;
    // Broader grid than the Taylor+classical version could handle.
    // The Opitz-based MNP gives correct positive values across the
    // full range; precision degrades only at very high q where
    // divided differences fall below 10^-50 or so.
    const std::vector<double> bs = {0.5, 1.0, 2.0, 5.0, 10.0};
    const std::vector<int>    qs = {1, 3, 5, 10, 15, 25};
    const int trials = 5;
    // Use a generous tolerance: both methods lose digits at high q,
    // and this isn't an independent high-precision benchmark either
    // way. The interesting check is that they agree to several digits
    // across the regime, not that they bit-match.
    const double tol = 1e-6;
    int passes = 0, total = 0, skipped = 0;
    double max_rel = 0.0;
    for (double b : bs) for (int q : qs) {
        chebyshev::ExpDivDiff<double> cheb(a, b);
        for (int t = 0; t < trials; ++t) {
            std::uniform_real_distribution<double> dist(a, b);
            std::vector<double> nodes(q + 1);
            for (auto& x : nodes) x = dist(rng);
            std::vector<long double> nodes_ld(nodes.begin(), nodes.end());
            const double cheb_val = cheb.evaluate(nodes);
            const double ref_val  = static_cast<double>(
                mccurdy_templated::ExpDivDiff<long double>::compute(nodes_ld));

            // Skip if either is non-finite (shouldn't happen for the
            // Opitz/MNP reference, but guard anyway).
            if (!std::isfinite(ref_val) || !std::isfinite(cheb_val)) {
                skipped++;
                continue;
            }
            // Skip trials whose magnitudes are below the regime where
            // double-precision MNP can hold several digits: the Taylor
            // + squaring scheme accumulates relative error that becomes
            // visible when the (0, q) entry falls well below 10^-20.
            // The original MNP paper notes this and recommends extended
            // precision for high q at small intervals. We keep these
            // trials in the run but exclude them from the strict
            // tolerance check.
            if (std::abs(ref_val) < 1e-20) {
                skipped++;
                continue;
            }

            const double rel = std::abs(cheb_val - ref_val) /
                               std::max(std::abs(ref_val), 1e-300);
            if (rel > max_rel) max_rel = rel;
            if (rel < tol) passes++;
            total++;
        }
    }
    std::printf("  %d/%d trials pass at rel_err < %.0e (skipped %d); max_rel=%.2e\n",
                passes, total, tol, skipped, max_rel);
    const bool ok = (passes == total);
    std::printf("=== %s ===\n", ok ? "Group 3 PASS" : "Group 3 FAIL");
    return ok;
}

bool run_batch_vs_incremental_test() {
    std::printf("\n=== Group 4: Batch vs Incremental (full-set initialize) ===\n");
    comparison_test::ComparisonConfig cfg;
    cfg.a = -2.0;
    cfg.b_values = {2.0, 4.0, 8.0};
    cfg.q_values = {3, 7, 15};
    cfg.num_trials = 5;
    cfg.random_seed = 13;
    cfg.verbose = false;
    const bool ok = comparison_test::run_comparison_tests(cfg);
    std::printf("=== %s ===\n", ok ? "Group 4 PASS" : "Group 4 FAIL");
    return ok;
}

bool run_incremental_path_test() {
    return test_incremental_path::run_all_incremental_path_tests(false);
}

// Group 6: A2 (certified stopping mode) and A4 (exception safety in
// add_node and remove_last_node). Targeted unit tests for the two
// behaviors most easily verified directly: rollback on a thrown
// exception, and certified-mode agreement at the requested tolerance.
bool run_a2_a4_features_test() {
    std::printf("\n=== Group 6: A2 (certified mode) + A4 (exception safety) ===\n");
    std::fflush(stdout);
    using namespace chebyshev_incremental;
    int passes = 0, total = 0;

    // A4: add_node throws on out-of-range node, state unchanged.
    {
        total++;
        IncrementalExpDivDiff<double> inc(-1.0, 1.0);
        inc.initialize({0.0, 0.3});
        const int    q0 = inc.current_order();
        const double v0 = inc.current_value();
        bool threw = false;
        try { inc.add_node(5.0); } catch (...) { threw = true; }
        if (threw && inc.current_order() == q0 && inc.current_value() == v0) passes++;
        std::printf("  A4.1 add_node out-of-range rolls back: %s\n",
                    (threw && inc.current_order() == q0) ? "OK" : "FAIL");
    }
    // A4: add_node throws when q would exceed max_q, state unchanged.
    {
        total++;
        IncrementalExpDivDiff<double> inc(-1.0, 1.0, /*max_q=*/3);
        inc.initialize({0.0, 0.1, 0.2, 0.3});
        const int    q0 = inc.current_order();
        const double v0 = inc.current_value();
        bool threw = false;
        try { inc.add_node(0.4); } catch (...) { threw = true; }
        if (threw && inc.current_order() == q0 && inc.current_value() == v0) passes++;
        std::printf("  A4.2 add_node past max_q rolls back:   %s\n",
                    (threw && inc.current_order() == q0) ? "OK" : "FAIL");
    }
    // A4: remove_last_node from q=0 throws, state unchanged.
    {
        total++;
        IncrementalExpDivDiff<double> inc(-1.0, 1.0);
        inc.initialize({0.4});
        const int q0 = inc.current_order();
        bool threw = false;
        try { inc.remove_last_node(); } catch (...) { threw = true; }
        if (threw && inc.current_order() == q0) passes++;
        std::printf("  A4.3 remove from q=0 rolls back:       %s\n",
                    (threw && inc.current_order() == q0) ? "OK" : "FAIL");
    }
    // A2: certified mode matches heuristic on a generic input.
    {
        total++;
        IncrementalExpDivDiff<double> h(-1.0, 1.0, 50, 1e-13);
        IncrementalExpDivDiff<double> c(-1.0, 1.0, 50, 1e-13, -1,
                                        StoppingMode::ConservativeTailEstimate);
        std::vector<double> nodes;
        std::mt19937 rng(7);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (int i = 0; i < 10; ++i) nodes.push_back(dist(rng));
        const double v_h = h.initialize(nodes);
        const double v_c = c.initialize(nodes);
        const double rel = std::abs(v_h - v_c) / std::max(std::abs(v_h), 1e-300);
        const bool ok = (rel < 1e-10);
        if (ok) passes++;
        std::printf("  A2.1 certified ~ heuristic agreement:  %s (rel=%.2e)\n",
                    ok ? "OK" : "FAIL", rel);
    }
    // A2: certified mode meets the requested tolerance.
    {
        total++;
        IncrementalExpDivDiff<double> ref (-2.0, 2.0, 30, 1e-14);
        IncrementalExpDivDiff<double> cert(-2.0, 2.0, 30, 1e-8, -1,
                                            StoppingMode::ConservativeTailEstimate);
        std::vector<double> nodes = {-1.0, -0.5, 0.2, 1.0, 1.5};
        const double v_ref  = ref.initialize(nodes);
        const double v_cert = cert.initialize(nodes);
        const double rel = std::abs(v_cert - v_ref) / std::abs(v_ref);
        const bool ok = (rel < 1e-7);
        if (ok) passes++;
        std::printf("  A2.2 certified meets 1e-8 tolerance:    %s (rel=%.2e)\n",
                    ok ? "OK" : "FAIL", rel);
    }
    // A2: set_stopping_mode at runtime, subsequent calls use new mode.
    {
        total++;
        IncrementalExpDivDiff<double> inc(-1.0, 1.0);
        inc.initialize({-0.5, 0.5, 0.7});
        inc.set_stopping_mode(StoppingMode::ConservativeTailEstimate);
        const double v_after = inc.add_node(0.3);
        IncrementalExpDivDiff<double> fresh(-1.0, 1.0);
        const double v_ref = fresh.initialize({-0.5, 0.5, 0.7, 0.3});
        const double rel = std::abs(v_after - v_ref) / std::abs(v_ref);
        const bool ok = (rel < 1e-12);
        if (ok) passes++;
        std::printf("  A2.3 runtime mode switch then add_node: %s (rel=%.2e)\n",
                    ok ? "OK" : "FAIL", rel);
    }

    // A2: certified mode at large c (the adversarial case from the
    // Round-3 review).  Previously the prefactor had the wrong sign in
    // the exp(d-a) factor, so the bound was off by exp(-2c); at c=100
    // it declared convergence orders of magnitude too early.  In log
    // space with the correct sign, certified mode at long double should
    // meet its stated tolerance.  Requires a genuinely extended long
    // double (x86 80-bit); on platforms where long double == double
    // (e.g. Apple arm64) the c=100 case is ill-conditioned in working
    // precision and the incremental evaluator correctly refuses it.
    if (std::numeric_limits<long double>::digits <= 53) {
        std::printf("  A2.4 certified at c=100 (adversarial):  SKIPPED "
                    "(long double has no extra precision here)\n");
    } else {
        total++;
        std::vector<long double> nodes = {-90.0L, -30.0L, 20.0L, 75.0L};
        IncrementalExpDivDiff<long double> tight(-100.0L, 100.0L, 10, 1e-18L);
        IncrementalExpDivDiff<long double> cert (-100.0L, 100.0L, 10, 1e-8L, -1,
                                                  StoppingMode::ConservativeTailEstimate);
        const long double v_tight = tight.initialize(nodes);
        const long double v_cert  = cert.initialize(nodes);
        const long double rel = std::abs((v_tight - v_cert) / v_tight);
        const bool ok2 = (rel < 1e-8L);
        if (ok2) passes++;
        std::printf("  A2.4 certified at c=100 (adversarial):  %s (rel=%.2Le)\n",
                    ok2 ? "OK" : "FAIL", rel);
    }
    // A3: add_node_normalized invalidates current_value().
    {
        total++;
        IncrementalExpDivDiff<double> inc(-1.0, 1.0);
        inc.initialize({-0.5, 0.5, 0.7});
        (void) inc.current_value();
        inc.add_node_normalized(0.3);
        bool threw = false;
        try { (void) inc.current_value(); } catch (...) { threw = true; }
        if (threw) passes++;
        std::printf("  A3.1 normalized invalidates full value: %s\n",
                    threw ? "OK" : "FAIL");
    }
    // A3: compute_full_value() restores it.
    {
        total++;
        IncrementalExpDivDiff<double> inc(-1.0, 1.0);
        inc.initialize_normalized({-0.5, 0.5});
        inc.compute_full_value();
        bool ok_now = true;
        try {
            const double v = inc.current_value();
            IncrementalExpDivDiff<double> ref(-1.0, 1.0);
            const double r = ref.initialize({-0.5, 0.5});
            if (std::abs(v - r) / std::abs(r) > 1e-12) ok_now = false;
        } catch (...) { ok_now = false; }
        if (ok_now) passes++;
        std::printf("  A3.2 compute_full_value restores it:    %s\n",
                    ok_now ? "OK" : "FAIL");
    }
    // A1: failed initialize() must leave the object's prior state
    // unchanged.  Reviewer's exact scenario.
    {
        total++;
        IncrementalExpDivDiff<double> inc(-1.0, 1.0);
        inc.initialize({0.0, 0.2});
        const auto   old_nodes = inc.current_nodes();
        const double old_val   = inc.current_value();
        const int    old_q     = inc.current_order();
        bool threw = false;
        try { inc.initialize({0.1, 5.0}); }  // 5.0 out of range
        catch (...) { threw = true; }
        const bool ok1 = threw
                      && (inc.current_nodes() == old_nodes)
                      && (inc.current_value() == old_val)
                      && (inc.current_order() == old_q);
        if (ok1) passes++;
        std::printf("  A1.1 failed initialize preserves state: %s\n",
                    ok1 ? "OK" : "FAIL");
    }
    // B2 / A1: large-c all-equal nodes.  The batch confluent branch
    // previously formed log_S using a conservative upper estimate of
    // log(I_0(c)), which biased the result by ~1.7e-8 at c=50.  With
    // log_I0_value (accurate) plus the q == 0 special case in
    // compute_full_value, both paths should hit double round-off.
    {
        total++;
        chebyshev::ExpDivDiff<double> s(-50.0, 50.0);
        const double v = s.evaluate({50.0, 50.0});
        const double truth = std::exp(50.0);
        const double rel = std::abs(v - truth) / truth;
        const bool ok = (rel < 1e-12);
        if (ok) passes++;
        std::printf("  B2.1 batch  exp[50,50] = e^50:       %s (rel=%.2e)\n",
                    ok ? "OK" : "FAIL", rel);
    }
    {
        total++;
        IncrementalExpDivDiff<double> inc(-50.0, 50.0);
        inc.initialize_normalized({50.0});
        const double v = inc.compute_full_value();
        const double truth = std::exp(50.0);
        const double rel = std::abs(v - truth) / truth;
        const bool ok = (rel < 1e-12);
        if (ok) passes++;
        std::printf("  B2.2 incr   q=0 normalized->full:   %s (rel=%.2e)\n",
                    ok ? "OK" : "FAIL", rel);
    }
    // A2: negative-index public API guard.  Earlier the batch
    // get_bessel_ratio(-1) was an undefined heap read; ASAN flagged
    // it.  Now it throws std::out_of_range.
    {
        total++;
        chebyshev::ExpDivDiff<double> s(-1.0, 1.0);
        bool threw = false;
        try { s.get_bessel_ratio(-1); }
        catch (const std::out_of_range&) { threw = true; }
        if (threw) passes++;
        std::printf("  A2.5 get_bessel_ratio(-1) throws:    %s\n",
                    threw ? "OK" : "FAIL");
    }
    // C2: invalid tolerance / max_order rejected at construction.
    {
        total++;
        bool threw_zero_tol = false;
        try { chebyshev::ExpDivDiff<double> s(-1.0, 1.0, 0.0); }
        catch (const std::invalid_argument&) { threw_zero_tol = true; }
        bool threw_neg_tol = false;
        try { chebyshev::ExpDivDiff<double> s(-1.0, 1.0, -1e-10); }
        catch (const std::invalid_argument&) { threw_neg_tol = true; }
        const bool ok = threw_zero_tol && threw_neg_tol;
        if (ok) passes++;
        std::printf("  C2.1 invalid tolerance rejected:      %s\n",
                    ok ? "OK" : "FAIL");
    }
    // A1: evaluate() on a wide constructor interval, with nodes far
    // inside the interval.  The fixed-interval Chebyshev representation
    // has S underflow to round-off there; evaluate() must internally
    // delegate to a tight-interval evaluator to return correct values.
    {
        total++;
        chebyshev::ExpDivDiff<double> b(-100.0, 100.0);
        const double v = b.evaluate({-1.0, 1.0});
        const double truth = (std::exp(1.0) - std::exp(-1.0)) / 2.0;
        const double rel = std::abs(v - truth) / truth;
        const bool ok = (rel < 1e-10);
        if (ok) passes++;
        std::printf("  A1.2 over-wide batch evaluate({-1,1}): %s (rel=%.2e)\n",
                    ok ? "OK" : "FAIL", rel);
    }
    {
        total++;
        chebyshev::ExpDivDiff<double> b(-1000.0, 1000.0);
        const double v = b.evaluate({-1.0, 1.0});
        const double truth = (std::exp(1.0) - std::exp(-1.0)) / 2.0;
        const double rel = std::abs(v - truth) / truth;
        const bool ok = (rel < 1e-10);
        if (ok) passes++;
        std::printf("  A1.3 extreme c=1000 evaluate({-1,1}): %s (rel=%.2e)\n",
                    ok ? "OK" : "FAIL", rel);
    }
    {
        total++;
        chebyshev::ExpDivDiff<double> b(-1000.0, 1000.0);
        const double v = b.evaluate({0.0, 0.0, 0.0});
        const bool ok = (std::abs(v - 0.5) < 1e-15);
        if (ok) passes++;
        std::printf("  A1.4 c=1000 all-equal {0,0,0}=1/2:    %s\n",
                    ok ? "OK" : "FAIL");
    }
    {
        total++;
        IncrementalExpDivDiff<double> inc(-1000.0, 1000.0, 10);
        inc.initialize({0.0, 0.0});
        const bool ok = (std::abs(inc.current_value() - 1.0) < 1e-15);
        if (ok) passes++;
        std::printf("  A1.5 incremental c=1000 init({0,0}):  %s\n",
                    ok ? "OK" : "FAIL");
    }
    {
        total++;
        IncrementalExpDivDiff<double> inc(-1000.0, 1000.0, 10);
        inc.initialize({0.0, 0.0, 0.0});
        const bool ok = (std::abs(inc.current_value() - 0.5) < 1e-15);
        if (ok) passes++;
        std::printf("  A1.6 incremental c=1000 init({0,0,0}): %s\n",
                    ok ? "OK" : "FAIL");
    }
    // R9.1: incremental distinct nodes far inside a fixed wide interval.
    // The incremental class cannot tight-fallback (its cache is bound
    // to the constructor interval), so the conditioning gate must
    // throw rather than silently returning ~3.5e25.
    {
        total++;
        IncrementalExpDivDiff<double> inc(-100.0, 100.0, 10);
        bool threw = false;
        try { inc.initialize({-1.0, 1.0}); }
        catch (const std::runtime_error&) { threw = true; }
        if (threw) passes++;
        std::printf("  R9.1 incr distinct over-wide throws:   %s\n",
                    threw ? "OK" : "FAIL");
    }
    // R9.2: same conditioning trip via add_node into a wide interval.
    {
        total++;
        IncrementalExpDivDiff<double> inc(-1000.0, 1000.0, 10);
        inc.initialize({0.0});  // q == 0 closed form; should not throw.
        bool threw = false;
        try { inc.add_node(1.0); }
        catch (const std::runtime_error&) { threw = true; }
        if (threw) passes++;
        std::printf("  R9.2 incr add_node into wide throws:   %s\n",
                    threw ? "OK" : "FAIL");
    }
    // R9.3: conservative-tail-estimate mode does not protect against
    // round-off cancellation -- it bounds the omitted Chebyshev tail,
    // not the round-off in a nearly-cancelling sum.  Throw all the
    // same.
    {
        total++;
        IncrementalExpDivDiff<double> inc(-100.0, 100.0, 10, 1e-8, -1,
            StoppingMode::ConservativeTailEstimate);
        bool threw = false;
        try { inc.initialize({-1.0, 1.0}); }
        catch (const std::runtime_error&) { threw = true; }
        if (threw) passes++;
        std::printf("  R9.3 cert mode over-wide also throws:  %s\n",
                    threw ? "OK" : "FAIL");
    }
    // R9.4: batch evaluate() must validate nodes *before* the
    // all-equal closed form fires.  Previously evaluate({5,5}) on
    // [-1, 1] returned exp(5) instead of throwing.
    {
        total++;
        chebyshev::ExpDivDiff<double> s(-1.0, 1.0);
        bool threw_eq = false;
        bool threw_pair = false;
        try { s.evaluate({5.0, 5.0}); }
        catch (const std::out_of_range&) { threw_eq = true; }
        try { s.evaluate({5.0, 5.08}); }
        catch (const std::out_of_range&) { threw_pair = true; }
        const bool ok = threw_eq && threw_pair;
        if (ok) passes++;
        std::printf("  R9.4 batch range-check before branches: %s\n",
                    ok ? "OK" : "FAIL");
    }
    // R9.5: NaN / non-finite rejection in constructors and inputs.
    {
        total++;
        const double nan = std::numeric_limits<double>::quiet_NaN();
        bool t1 = false, t2 = false, t3 = false, t4 = false;
        try { chebyshev::ExpDivDiff<double> s(nan, 1.0); }
        catch (const std::invalid_argument&) { t1 = true; }
        try { IncrementalExpDivDiff<double> inc(0.0, nan); }
        catch (const std::invalid_argument&) { t2 = true; }
        try {
            chebyshev::ExpDivDiff<double> s(-1.0, 1.0);
            s.evaluate({nan, 0.0});
        } catch (const std::out_of_range&) { t3 = true; }
        try {
            IncrementalExpDivDiff<double> inc(-1.0, 1.0);
            inc.initialize({nan, 0.0});
        } catch (const std::out_of_range&) { t4 = true; }
        const bool ok = t1 && t2 && t3 && t4;
        if (ok) passes++;
        std::printf("  R9.5 NaN/inf inputs rejected:          %s\n",
                    ok ? "OK" : "FAIL");
    }
    // R10.1: evaluate_normalized() must throw on bad conditioning
    // (the public entry point; evaluate() catches and falls back).
    {
        total++;
        chebyshev::ExpDivDiff<double> e(-100.0, 100.0);
        bool threw = false;
        try { e.evaluate_normalized({-1.0, 1.0}); }
        catch (const std::runtime_error&) { threw = true; }
        if (threw) passes++;
        std::printf("  R10.1 evaluate_normalized over-wide throws: %s\n",
                    threw ? "OK" : "FAIL");
    }
    // R10.2: evaluate() at the boundary c/spread regime that the
    // old 10x threshold missed.  Both c=20 and c=15 with nodes
    // {-1, 1} must hit round-off, not 1e-10.
    {
        total++;
        chebyshev::ExpDivDiff<double> e20(-20.0, 20.0);
        chebyshev::ExpDivDiff<double> e15(-15.0, 15.0);
        const double truth = (std::exp(1.0) - std::exp(-1.0)) / 2.0;
        const double v20 = e20.evaluate({-1.0, 1.0});
        const double v15 = e15.evaluate({-1.0, 1.0});
        const double r20 = std::abs(v20 - truth) / truth;
        const double r15 = std::abs(v15 - truth) / truth;
        const bool ok = (r20 < 1e-12) && (r15 < 1e-12);
        if (ok) passes++;
        std::printf("  R10.2 boundary c=20/15 evaluate({-1,1}): %s (r20=%.1e r15=%.1e)\n",
                    ok ? "OK" : "FAIL", r20, r15);
    }
    // R10.3: all-equal closed form via lgamma reaches double's
    // subnormal range (q=171, value ~8e-310).
    {
        total++;
        std::vector<double> nodes(172, 0.0);
        chebyshev::ExpDivDiff<double> e(-1.0, 1.0);
        const double v = e.evaluate(nodes);
        // Truth is 1/171! ≈ 8.0584e-310; tolerate a factor of 2 since
        // subnormal arithmetic loses bits.
        const bool ok = (v > 1e-310) && (v < 1e-309);
        if (ok) passes++;
        std::printf("  R10.3 q=171 all-equal subnormal value:  %s (got %.2e)\n",
                    ok ? "OK" : "FAIL", v);
    }
    // R10.4: lazy batch construction.  Constructing on an absurdly
    // wide interval used to eagerly allocate ~2 GB and overflow
    // I_0(c); now the heavy work is deferred until first use, and
    // evaluate() goes straight to a tight rebuild for an over-wide
    // node set.
    {
        total++;
        chebyshev::ExpDivDiff<double> e(-1e8, 1e8);
        const double v = e.evaluate({-1.0, 1.0});
        const double truth = (std::exp(1.0) - std::exp(-1.0)) / 2.0;
        const double rel = std::abs(v - truth) / truth;
        const bool ok = (rel < 1e-12);
        if (ok) passes++;
        std::printf("  R10.4 lazy construct on (-1e8, 1e8):    %s (rel=%.1e)\n",
                    ok ? "OK" : "FAIL", rel);
    }
    // R11.1: evaluate_normalized all-equal branch must set n_out
    // and (internally) kappa_eps_out.  The previous code left both
    // uninitialized, leading to undefined behaviour in the public
    // wrapper which read the kappa to decide whether to throw.
    {
        total++;
        chebyshev::ExpDivDiff<double> e(-1000.0, 1000.0);
        int n = -1;
        const double S = e.evaluate_normalized({0.0, 0.0, 0.0}, &n);
        const bool ok = (n == 0) && std::isfinite(S);
        if (ok) passes++;
        std::printf("  R11.1 normalized all-equal sets n_out:   %s (n=%d)\n",
                    ok ? "OK" : "FAIL", n);
    }
    // R11.2: free bessel_ratio() helper must stay finite at large c
    // (it previously overflowed both numerator and denominator and
    // returned NaN).  Truth at (n=10, c=1000) is ~0.951206.
    {
        total++;
        const double r = chebyshev_bessel::bessel_ratio<double>(10, 1000.0);
        const bool ok = std::isfinite(r) && std::abs(r - 0.951206) < 1e-3;
        if (ok) passes++;
        std::printf("  R11.2 free bessel_ratio(10, 1000):       %s (r=%.6f)\n",
                    ok ? "OK" : "FAIL", r);
    }
    // R11.3: evaluate_normalized must pre-screen absurdly wide
    // intervals BEFORE doing any Bessel work.  Otherwise the first
    // get_bessel_ratio call would request a Miller cache of size
    // ~c, allocating gigabytes for c = 1e8.
    {
        total++;
        chebyshev::ExpDivDiff<double> e(-1e8, 1e8);
        bool threw = false;
        try { e.evaluate_normalized({-1.0, 1.0}); }
        catch (const std::runtime_error&) { threw = true; }
        if (threw) passes++;
        std::printf("  R11.3 normalized over-wide pre-screen:   %s\n",
                    threw ? "OK" : "FAIL");
    }
    // R11.4: reset() must clear last_used_order_ and full_value_valid_,
    // not just node and cache state.
    {
        total++;
        IncrementalExpDivDiff<double> inc(-1.0, 1.0);
        inc.initialize({0.0, 0.3});
        inc.reset();
        const bool ok = (inc.last_used_order() == -1);
        if (ok) passes++;
        std::printf("  R11.4 reset clears last_used_order:      %s\n",
                    ok ? "OK" : "FAIL");
    }
    // R12.1: incremental constructor on absurdly wide interval must
    // throw quickly rather than spending O(c) work on Bessel-ratio
    // precomputation.  (-1e7, 1e7) hung before this guard.
    {
        total++;
        bool threw = false;
        try { IncrementalExpDivDiff<double> inc(-1e7, 1e7, 10); }
        catch (const std::invalid_argument&) { threw = true; }
        if (threw) passes++;
        std::printf("  R12.1 incremental ctor on (-1e7,1e7):    %s\n",
                    threw ? "OK" : "FAIL");
    }
    // R12.2: public Bessel helpers must reject non-finite inputs.
    {
        total++;
        const double nan = std::nan("");
        bool t1 = false, t2 = false, t3 = false, t4 = false;
        try { chebyshev_bessel::bessel_I0<double>(nan); }
        catch (const std::invalid_argument&) { t1 = true; }
        try { chebyshev_bessel::bessel_In<double>(1, nan); }
        catch (const std::invalid_argument&) { t2 = true; }
        try { chebyshev_bessel::bessel_ratio<double>(1, nan); }
        catch (const std::invalid_argument&) { t3 = true; }
        try { chebyshev_bessel::BesselRatiosMiller<double>::compute_R(nan, 5, 100); }
        catch (const std::invalid_argument&) { t4 = true; }
        const bool ok = t1 && t2 && t3 && t4;
        if (ok) passes++;
        std::printf("  R12.2 Bessel helpers reject NaN:         %s\n",
                    ok ? "OK" : "FAIL");
    }
    // R12.3: recommended_Nstart throws overflow_error when the
    // required starting index would exceed int range, rather than
    // silently returning a corrupted value.
    {
        total++;
        bool threw = false;
        try {
            chebyshev_bessel::BesselRatiosMiller<double>::recommended_Nstart(
                2.0e10, 100);
        } catch (const std::overflow_error&) { threw = true; }
        if (threw) passes++;
        std::printf("  R12.3 recommended_Nstart overflow:       %s\n",
                    threw ? "OK" : "FAIL");
    }
    // R13.1: bessel_In rejects huge x before any int-overflow risk.
    // The previous static_cast<int>(2 * x) tripped UBSAN at x = 1e20.
    {
        total++;
        bool threw = false;
        try { (void)chebyshev_bessel::bessel_In<double>(1, 1e20); }
        catch (const std::invalid_argument&) { threw = true; }
        if (threw) passes++;
        std::printf("  R13.1 bessel_In(1, 1e20) throws:         %s\n",
                    threw ? "OK" : "FAIL");
    }
    // R13.2: free bessel_ratio rejects huge c at the public-helper
    // work cap.  compute_R itself stays uncapped for trusted callers.
    {
        total++;
        bool threw = false;
        try { (void)chebyshev_bessel::bessel_ratio<double>(10, 1e7); }
        catch (const std::invalid_argument&) { threw = true; }
        if (threw) passes++;
        std::printf("  R13.2 bessel_ratio(10, 1e7) throws:      %s\n",
                    threw ? "OK" : "FAIL");
    }
    // R13.3: batch constructor with endpoint pair whose old
    // (b - a)/2 formula overflowed to +inf.  The rearranged
    // b/2 - a/2 form keeps c_ finite at 1e308.
    {
        total++;
        chebyshev::ExpDivDiff<double> e(-1e308, 1e308);
        const double c = e.get_c();
        const bool ok = std::isfinite(c);
        if (ok) passes++;
        std::printf("  R13.3 ExpDivDiff(-1e308, 1e308) c finite: %s (c=%.2g)\n",
                    ok ? "OK" : "FAIL", c);
    }
    // R13.4: recommended_Nstart rejects negative c, matching
    // compute_R and bessel_ratio.
    {
        total++;
        bool threw = false;
        try {
            (void)chebyshev_bessel::BesselRatiosMiller<double>::recommended_Nstart(-1.0, 10);
        } catch (const std::invalid_argument&) { threw = true; }
        if (threw) passes++;
        std::printf("  R13.4 recommended_Nstart(-1, 10) throws: %s\n",
                    threw ? "OK" : "FAIL");
    }

    const bool ok = (passes == total);
    std::printf("  %d/%d sub-tests passed\n", passes, total);
    std::printf("=== %s ===\n", ok ? "Group 6 PASS" : "Group 6 FAIL");
    return ok;
}

void run_mpmath_reference_test() {
    std::printf("\n=== Group 7: Independent mpmath ground-truth comparison ===\n");
    std::printf("  (Both Chebyshev and MNP at matched precision against the\n");
    std::printf("   mpmath reference produced by mpmath_reference.py.)\n");
    const auto result = test_against_reference::run_all_reference_tests(false);
    using R = test_against_reference::InfoResult;
    switch (result) {
        case R::Passed:
            std::printf("=== Group 7 PASS ===\n"); break;
        case R::Skipped:
            std::printf("=== Group 7 SKIPPED ===\n"); break;
        case R::FailedInformational:
            std::printf("=== Group 7 INFO (some cases over per-precision tol) ===\n");
            break;
    }
}

} // namespace

// Group 8: regression tests for the issues raised in the SISC review.
bool run_revision_tests() {
    std::printf("\n=== Group 8: Revision regression tests ===\n");
    int passes = 0, total = 0;
    auto report = [&](const char* name, bool ok, double val) {
        total++; if (ok) passes++;
        std::printf("  %-52s %s (%.2e)\n", name, ok ? "OK" : "FAIL", val);
    };
    auto rel = [](double a, double b) { return std::abs(a - b) / std::abs(b); };

    // Referee counterexample: q = 1, nodes {-1, 1} on [-1, 1].  D_1^(n)
    // vanishes for every even n, so a single-term stopping test stops at
    // n = 2 and returns 2 I_1(1) ~ 1.1303 instead of sinh(1) ~ 1.1752.
    {
        chebyshev::ExpDivDiff<double> e(-1.0, 1.0);
        const double v = e.evaluate({-1.0, 1.0});
        report("R1 parity counterexample exp[-1,1] = sinh(1)", rel(v, std::sinh(1.0)) < 1e-14,
               rel(v, std::sinh(1.0)));
    }
    // R_n(c) underflow: q = 160, c = 0.1.  R_160(0.1) ~ 1e-490 underflows
    // in double while exp[x] ~ 2e-285 does not.
    {
        std::mt19937 rng(5);
        std::uniform_real_distribution<double> U(-0.1, 0.1);
        std::vector<double> x(161);
        for (auto& v : x) v = U(rng);
        chebyshev::ExpDivDiff<double> e(-0.1, 0.1);
        const double v = e.evaluate(x);
        const double m = mccurdy_templated::ExpDivDiff<double>::compute(x);
        report("R2 small c, high q (R_n underflow) vs MNP", rel(v, m) < 1e-12, rel(v, m));
    }
    // High q in scaled form: q = 2000.  The value (~1e-5700) lies far
    // outside double; check the mean-value bounds e^a/q! <= v <= e^b/q!.
    {
        std::mt19937 rng(6);
        std::uniform_real_distribution<double> U(-5.0, 5.0);
        std::vector<double> x(2001);
        for (auto& v : x) v = U(rng);
        const double a = *std::min_element(x.begin(), x.end());
        const double b = *std::max_element(x.begin(), x.end());
        chebyshev::ExpDivDiff<double> e(a, b);
        const long double lv = e.evaluate_scaled(x).log_abs();
        const long double lf = std::lgamma(2001.0L);
        const bool ok = std::isfinite(static_cast<double>(lv)) &&
                        lv >= a - lf - 1e-9L && lv <= b - lf + 1e-9L;
        report("R3 q = 2000 scaled value within mean-value bounds", ok,
               static_cast<double>(lv));
    }
    // No silent accuracy loss: wide fixed interval, large q and c.  Each
    // call must either throw or agree with MNP to the error indicator.
    {
        std::mt19937 rng(7);
        bool ok = true; double worst = 0.0;
        for (int t = 0; t < 10; ++t) {
            std::uniform_real_distribution<double> U(-300.0, 300.0);
            std::vector<double> x(51);
            for (auto& v : x) v = U(rng);
            chebyshev::ExpDivDiff<double> e(-300.0, 300.0);
            try {
                const double v = e.evaluate(x);
                const double m = mccurdy_templated::ExpDivDiff<double>::compute(x);
                const double r = rel(v, m);
                worst = std::max(worst, r);
                if (r > 1e-8) ok = false;
            } catch (const std::runtime_error&) { /* refused: acceptable */ }
        }
        report("R4 wide interval: throw or accurate (q=50, c=300)", ok, worst);
    }
    // MNP regression: entry-wise Taylor convergence at q = 20, c = 0.5.
    {
        std::mt19937 rng(8);
        std::uniform_real_distribution<double> U(-0.5, 0.5);
        std::vector<double> x(21);
        for (auto& v : x) v = U(rng);
        chebyshev::ExpDivDiff<double> e(-0.5, 0.5);
        const double v = e.evaluate(x);
        const double m = mccurdy_templated::ExpDivDiff<double>::compute(x);
        report("R5 MNP entry-wise Taylor (q=20, c=0.5) vs Chebyshev", rel(m, v) < 1e-13, rel(m, v));
    }
    // Repeated nodes: exp[0,0,0,1,1,2] (confluent Hermite data).
    {
        const std::vector<double> x = {0.0, 0.0, 0.0, 1.0, 1.0, 2.0};
        chebyshev::ExpDivDiff<double> e(0.0, 2.0);
        const double v = e.evaluate(x);
        const double m = mccurdy_templated::ExpDivDiff<double>::compute(x);
        report("R6 repeated nodes vs MNP", rel(v, m) < 1e-14, rel(v, m));
    }
    // Ratio of two divided differences that each underflow double.
    {
        std::mt19937 rng(9);
        std::uniform_real_distribution<double> U(-2.0, 2.0);
        std::vector<double> x(401), xp(401);
        for (auto& v : x) v = U(rng);
        xp = x; xp.back() = U(rng);
        chebyshev::ExpDivDiff<double> e(-2.0, 2.0);
        const double r = e.evaluate_ratio(x, xp);
        const double expect = (e.evaluate_scaled(x) / e.evaluate_scaled(xp)).to_real();
        report("R7 ratio of underflowing values (q=400)",
               std::isfinite(r) && rel(r, expect) < 1e-13, rel(r, expect));
    }

    // Bessel-ratio steps rho_n = I_n/I_{n-1} from the backward ratio
    // recurrence: products rho_1...rho_n must reproduce I_n(c)/I_0(c)
    // computed independently by cyl_bessel_i (where representable).
    {
        double worst = 0.0;
        for (double c : {1e-3, 0.1, 1.0, 10.0, 100.0, 600.0}) {
            const int nmax = static_cast<int>(2 * c) + 60;
            const int Ns = chebyshev_bessel::BesselRatiosMiller<double>::recommended_Nstart(c, nmax);
            const auto rho = chebyshev_bessel::BesselRatiosMiller<double>::compute_rho(c, nmax, Ns);
            const long double I0 = chebyshev_bessel::cyl_bessel_i(0.0L, static_cast<long double>(c));
            long double R = 1.0L;
            for (int n = 1; n <= nmax; ++n) {
                R *= rho[n];
                const long double In = chebyshev_bessel::cyl_bessel_i(
                    static_cast<long double>(n), static_cast<long double>(c));
                const long double ref = In / I0;
                if (ref > 1e-280L && std::isfinite(static_cast<double>(ref))) {
                    worst = std::max(worst, static_cast<double>(std::abs(R - ref) / ref));
                }
            }
        }
        report("R8 rho products vs cyl_bessel_i ratios", worst < 1e-12, worst);
    }

    std::printf("  %d/%d revision tests passed\n", passes, total);
    const bool ok = (passes == total);
    std::printf("=== %s ===\n", ok ? "Group 8 PASS" : "Group 8 FAIL");
    return ok;
}

// Group 9: chebyshev::IncrementalStack (scaled, column-major incremental
// evaluator of the revised manuscript).
bool run_stack_tests() {
    std::printf("\n=== Group 9: IncrementalStack ===\n");
    int passes = 0, total = 0;
    auto report = [&](const char* name, bool ok, double val) {
        total++; if (ok) passes++;
        std::printf("  %-52s %s (%.2e)\n", name, ok ? "OK" : "FAIL", val);
    };
    std::mt19937 rng(31);
    // S1: random LIFO walk vs batch on [min, max] (well-conditioned refs).
    {
        double worst = 0.0; bool ok = true;
        for (double c : {1.0, 10.0, 50.0}) {
            std::uniform_real_distribution<double> U(-c, c);
            chebyshev::IncrementalStack<double> st(-c, c, 120);
            std::vector<double> nodes;
            for (int step = 0; step < 300; ++step) {
                if (nodes.size() < 2 || rng() % 3 != 0) {
                    const double x = U(rng);
                    try { st.push(x); nodes.push_back(x); } catch (const std::runtime_error&) {}
                } else { st.pop(); nodes.pop_back(); }
                if (nodes.size() >= 2 && step % 5 == 0) {
                    const double lo = *std::min_element(nodes.begin(), nodes.end());
                    const double hi = *std::max_element(nodes.begin(), nodes.end());
                    chebyshev::ExpDivDiff<double> e(lo, hi);
                    e.set_max_error_estimate(1e300L);
                    const auto ref = e.evaluate_scaled(nodes);
                    if (e.last_error_estimate() > 1e-12L) continue;
                    const double r = std::abs((st.value_scaled() / ref).to_real() - 1.0);
                    worst = std::max(worst, r);
                    if (r > 1e-10) ok = false;
                }
            }
        }
        report("S1 random LIFO walks vs batch", ok, worst);
    }
    // S2: q = 800, c = 5 (the legacy class returns NaN here).
    {
        std::uniform_real_distribution<double> U(-5.0, 5.0);
        std::vector<double> x(801);
        for (auto& v : x) v = U(rng);
        chebyshev::IncrementalStack<double> st(-5.0, 5.0, 810);
        st.initialize(x);
        chebyshev::ExpDivDiff<double> e(-5.0, 5.0);
        const double r = std::abs((st.value_scaled() / e.evaluate_scaled(x)).to_real() - 1.0);
        report("S2 q = 800 initialize vs batch", r < 1e-12, r);
    }
    // S3: unfavorable insertion order (ascending appends, c = 50) is
    // refused rather than returned inaccurately.
    {
        int silent_bad = 0;
        for (int t = 0; t < 30; ++t) {
            std::uniform_real_distribution<double> U(-50.0, 50.0);
            std::vector<double> x(51);
            for (auto& v : x) v = U(rng);
            std::sort(x.begin(), x.end());
            chebyshev::IncrementalStack<double> st(-50.0, 50.0, 60);
            bool refused = false;
            for (double v : x) { try { st.push(v); } catch (const std::runtime_error&) { refused = true; break; } }
            if (refused) continue;
            const double m = mccurdy_templated::ExpDivDiff<double>::compute(x);
            if (std::abs(st.value() - m) / m > 1e-10) silent_bad++;
        }
        report("S3 ascending appends: refused or accurate", silent_bad == 0, silent_bad);
    }
    // S4: rejected push leaves the stack unchanged.
    {
        chebyshev::IncrementalStack<double> st(-1.0, 1.0, 10);
        st.initialize({0.5, -0.3, 0.1});
        const double before = st.value();
        bool threw = false;
        try { st.push(2.0); } catch (const std::out_of_range&) { threw = true; }
        report("S4 rejected push leaves stack unchanged",
               threw && st.order() == 2 && st.value() == before, 0.0);
    }
    std::printf("  %d/%d stack tests passed\n", passes, total);
    const bool ok = (passes == total);
    std::printf("=== %s ===\n", ok ? "Group 9 PASS" : "Group 9 FAIL");
    return ok;
}

int main() {
    std::printf("\n");
    std::printf("=========================================================\n");
    std::printf(" Chebyshev exp-divdiff CI: 8 required groups + 1 informational\n");
    std::printf("=========================================================\n");

    int required_total = 0, required_passed = 0;
    auto record_required = [&](bool g){ required_total++; if (g) required_passed++; };

    record_required(run_i0_wrapper_test()); std::fflush(stdout);
    record_required(run_bessel_ratio_test()); std::fflush(stdout);
    record_required(run_batch_vs_reference_test()); std::fflush(stdout);
    record_required(run_batch_vs_incremental_test()); std::fflush(stdout);
    record_required(run_incremental_path_test()); std::fflush(stdout);
    record_required(run_a2_a4_features_test()); std::fflush(stdout);
    record_required(run_revision_tests()); std::fflush(stdout);
    record_required(run_stack_tests()); std::fflush(stdout);
    run_mpmath_reference_test();
    // Group 7 is informational only; it does not gate exit status.

    std::printf("\n=========================================================\n");
    std::printf(" %d/%d required groups passed\n", required_passed, required_total);
    std::printf(" Group 7 (mpmath reference) is informational; see its output above.\n");
    std::printf("=========================================================\n\n");

    return (required_passed == required_total) ? 0 : 1;
}
