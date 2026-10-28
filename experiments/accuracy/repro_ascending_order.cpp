// =============================================================================
// repro_ascending_order.cpp -- minimal reproducer for the ordering-dependent
// silent inaccuracy found in experiment E2.
//
// 21 nodes in two tight clusters at the ends of [-100, 100]:
//   x = -100, -99.9, ..., -99.1 (10 nodes) and 100, 99.9, ..., 99.0 (11 nodes)
// The exact value (mpmath, mp_reference.py) is 27926248594688.342104...
// With set_sort_nodes(false) (recurrence in the caller's order), ASCENDING
// input returns a value with relative error ~6e-6 while kappa*u*N ~ 2.5e-14
// (accepted, no exception); DESCENDING input gives ~1e-15.  The library
// default (set_sort_nodes(true): nodes processed in nonincreasing order)
// gives ~1e-15 for any input order.  MNP agrees with the reference.
//
// Build (from the "DDEF c++" folder):
//   g++ -std=c++17 -O2 -I. -I/opt/homebrew/include \
//       experiments/accuracy/repro_ascending_order.cpp -o repro && ./repro
// =============================================================================
#include "chebyshev_exp_divdiff.hpp"
#include "mccurdy.hpp"

#include <algorithm>
#include <cstdio>
#include <vector>

int main() {
    const double c = 100.0, ref = 27926248594688.342104;
    std::vector<double> x;
    for (int j = 0; j < 10; ++j) x.push_back(-c + 1e-3 * c * j);
    for (int j = 0; j < 11; ++j) x.push_back(c - 1e-3 * c * j);
    std::sort(x.begin(), x.end());                      // ascending
    const std::vector<double> xd(x.rbegin(), x.rend());  // descending

    int N;
    auto run = [&](const char* name, const std::vector<double>& nodes, bool sort) {
        chebyshev::ExpDivDiff<double> E(-c, c);
        E.set_sort_nodes(sort);
        const double v = E.evaluate(nodes, &N);
        std::printf("%-28s: %.16e  rel.err %.2e  kappa*u*N %.2e  N %d\n", name, v,
                    std::abs(v / ref - 1), (double)E.last_error_estimate(), N);
    };
    run("ascending, caller order", x, false);
    run("descending, caller order", xd, false);
    run("ascending, default (sorted)", x, true);
    const double vm = mccurdy_templated::ExpDivDiff<double>::compute(x);
    std::printf("%-28s: %.16e  rel.err %.2e\n", "MNP", vm, std::abs(vm / ref - 1));
    return 0;
}
