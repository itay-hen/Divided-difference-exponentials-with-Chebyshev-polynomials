#ifndef TEST_INCREMENTAL_PATH_HPP
#define TEST_INCREMENTAL_PATH_HPP

// =============================================================================
// test_incremental_path.hpp
//
// Tests that actually exercise the advertised incremental path:
//
//   initialize(prefix);
//   add_node(...);
//   remove_last_node();
//   add_node(...);
//
// and compare each intermediate state to a fresh batch recomputation on the
// equivalent full node set. The pre-existing comparison_test.hpp does NOT
// test this path; it only compares batch.evaluate(full_set) against
// IncrementalExpDivDiff::initialize(full_set), which is just two batch
// constructors.
// =============================================================================

#include "chebyshev_exp_divdiff.hpp"
#include "incremental_chebyshev_exp_divdiff.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

namespace test_incremental_path {

struct CheckResult {
    bool ok;
    double rel_err;
};

// Compare an incremental value against a fresh batch recomputation on the
// same nodes. Returns relative error.
inline CheckResult check_against_batch(double a, double b,
                                       const std::vector<double>& nodes,
                                       double incr_value,
                                       double tol = 1e-10) {
    chebyshev::ExpDivDiff<double> batch(a, b);
    double batch_value = batch.evaluate(nodes);
    double scale = std::max(std::abs(batch_value), 1e-100);
    double rel_err = std::abs(batch_value - incr_value) / scale;
    return {rel_err < tol, rel_err};
}

// Test 1: build a node sequence one append at a time, compare after each.
inline bool test_progressive_append(double a, double b,
                                    const std::vector<double>& full_seq,
                                    bool verbose = false,
                                    double cmp_tol = 1e-10) {
    chebyshev_incremental::IncrementalExpDivDiff<double> inc(a, b);

    if (full_seq.empty()) {
        std::fprintf(stderr, "test_progressive_append: empty sequence\n");
        return false;
    }

    double v = inc.initialize({full_seq[0]});
    std::vector<double> running = {full_seq[0]};
    auto chk = check_against_batch(a, b, running, v, cmp_tol);
    if (!chk.ok) {
        std::printf("  FAIL at q=0: rel_err=%g\n", chk.rel_err);
        return false;
    }
    if (verbose) std::printf("  q=0  v=%.15e  rel_err=%g\n", v, chk.rel_err);

    for (size_t i = 1; i < full_seq.size(); ++i) {
        v = inc.add_node(full_seq[i]);
        running.push_back(full_seq[i]);
        chk = check_against_batch(a, b, running, v, cmp_tol);
        if (!chk.ok) {
            std::printf("  FAIL at q=%zu: rel_err=%g\n", i, chk.rel_err);
            return false;
        }
        if (verbose) std::printf("  q=%zu  v=%.15e  rel_err=%g\n",
                                 i, v, chk.rel_err);
    }
    return true;
}

// Test 2: a sequence of mixed add / LIFO remove operations.
//   ops[i] >= 0  : append node value ops_values[i] (next available)
//   ops[i] <  0  : remove last node
inline bool test_lifo_sequence(double a, double b,
                               const std::vector<double>& initial,
                               const std::vector<int>& ops,
                               const std::vector<double>& append_values,
                               bool verbose = false) {
    chebyshev_incremental::IncrementalExpDivDiff<double> inc(a, b);

    if (initial.empty()) return false;
    double v = inc.initialize(initial);
    std::vector<double> running = initial;
    auto chk = check_against_batch(a, b, running, v);
    if (!chk.ok) {
        std::printf("  FAIL after initialize: rel_err=%g\n", chk.rel_err);
        return false;
    }

    size_t append_idx = 0;
    for (size_t step = 0; step < ops.size(); ++step) {
        if (ops[step] >= 0) {
            if (append_idx >= append_values.size()) {
                std::fprintf(stderr, "  ran out of append values\n");
                return false;
            }
            double x = append_values[append_idx++];
            v = inc.add_node(x);
            running.push_back(x);
        } else {
            // Skip remove if it would empty the state.
            if (running.size() <= 1) continue;
            v = inc.remove_last_node();
            running.pop_back();
        }
        chk = check_against_batch(a, b, running, v);
        if (!chk.ok) {
            std::printf("  FAIL at step %zu (op=%d, q=%zu): rel_err=%g\n",
                        step, ops[step], running.size() - 1, chk.rel_err);
            return false;
        }
        if (verbose) std::printf("  step=%zu  op=%d  q=%zu  rel_err=%g\n",
                                 step, ops[step], running.size() - 1,
                                 chk.rel_err);
    }
    return true;
}

// Driver: a few canned scenarios plus randomized LIFO walks.
inline bool run_all_incremental_path_tests(bool verbose = false) {
    std::printf("\n=== Incremental-path tests ===\n");

    bool ok = true;

    // Scenario 1: progressive append on a small interval.
    {
        std::printf("Scenario 1: progressive append, b=2, q up to 10\n");
        std::vector<double> seq;
        std::mt19937 rng(7);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (int i = 0; i < 11; ++i) seq.push_back(dist(rng));
        bool s1 = test_progressive_append(-1.0, 1.0, seq, verbose);
        std::printf("  %s\n", s1 ? "OK" : "FAIL");
        ok = ok && s1;
    }

    // Scenario 2: progressive append on a larger interval.
    // At c=20, both batch and incremental have round-off floors of
    // ~kappa*eps ~ 1e-10 (each computed by its own ordering of the
    // Chebyshev sum), so their pairwise agreement is bounded by
    // about 2x that.  Use 1e-8 as the comparison tolerance -- well
    // above the round-off floor at this c, well below the rel-err
    // scale at which a real algorithmic bug would manifest.
    {
        std::printf("Scenario 2: progressive append, b=20, q up to 15\n");
        std::vector<double> seq;
        std::mt19937 rng(11);
        std::uniform_real_distribution<double> dist(-20.0, 20.0);
        for (int i = 0; i < 16; ++i) seq.push_back(dist(rng));
        bool s2 = test_progressive_append(-20.0, 20.0, seq, verbose, 1e-8);
        std::printf("  %s\n", s2 ? "OK" : "FAIL");
        ok = ok && s2;
    }

    // Scenario 3: deterministic LIFO push/pop pattern.
    {
        std::printf("Scenario 3: deterministic LIFO push/pop\n");
        // start with two nodes, then push, pop, push, push, pop, push.
        std::vector<int> ops = { +1, -1, +1, +1, -1, +1 };
        std::vector<double> appends = { 0.1, 0.5, 0.7, -0.3, 0.4, 0.9 };
        bool s3 = test_lifo_sequence(-2.0, 2.0,
                                     {-1.0, 0.0}, ops, appends, verbose);
        std::printf("  %s\n", s3 ? "OK" : "FAIL");
        ok = ok && s3;
    }

    // Scenario 4: randomized LIFO walks.
    {
        std::printf("Scenario 4: 20 randomized LIFO walks\n");
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> dist(-5.0, 5.0);
        std::uniform_int_distribution<int> op_dist(0, 3); // 0->remove, 1..3->add
        int passes = 0;
        for (int trial = 0; trial < 20; ++trial) {
            std::vector<int> ops;
            std::vector<double> appends;
            for (int s = 0; s < 12; ++s) {
                int o = op_dist(rng);
                if (o == 0) ops.push_back(-1);
                else { ops.push_back(+1); appends.push_back(dist(rng)); }
            }
            std::vector<double> start = { dist(rng), dist(rng) };
            bool ok_i = test_lifo_sequence(-5.0, 5.0, start, ops, appends, false);
            if (ok_i) passes++;
            else std::printf("  trial %d FAILED\n", trial);
        }
        std::printf("  %d/20 trials passed\n", passes);
        ok = ok && (passes == 20);
    }

    // Scenario 5: start from coincident nodes, then append distinct ones.
    {
        std::printf("Scenario 5: coincident-start then progressive append\n");
        chebyshev_incremental::IncrementalExpDivDiff<double> inc(-1.0, 1.0);
        double v = inc.initialize({0.3, 0.3, 0.3});
        std::vector<double> running = {0.3, 0.3, 0.3};
        auto chk = check_against_batch(-1.0, 1.0, running, v, 1e-10);
        bool s5 = chk.ok;
        if (!s5) std::printf("    initial coincident: rel_err=%g\n", chk.rel_err);
        for (double x : {0.1, -0.4, 0.7}) {
            v = inc.add_node(x);
            running.push_back(x);
            chk = check_against_batch(-1.0, 1.0, running, v, 1e-10);
            if (!chk.ok) {
                std::printf("    after append %g: rel_err=%g\n", x, chk.rel_err);
                s5 = false;
            }
        }
        std::printf("  %s\n", s5 ? "OK" : "FAIL");
        ok = ok && s5;
    }

    std::printf("=== %s ===\n\n", ok ? "ALL INCREMENTAL TESTS PASSED" : "SOME INCREMENTAL TESTS FAILED");
    return ok;
}

} // namespace test_incremental_path

#endif // TEST_INCREMENTAL_PATH_HPP
