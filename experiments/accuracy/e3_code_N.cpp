// =============================================================================
// e3_code_N.cpp -- Experiment E3 helper: the number of Chebyshev orders N
// actually chosen by the double-precision code (heuristic stopping rule,
// tol = 1e-14) for given node sets on the EXACT interval [-c, c].
//
// Build (from the "DDEF c++" folder):
//   g++ -std=c++17 -O2 -I. -I/opt/homebrew/include \
//       experiments/accuracy/e3_code_N.cpp -o e3_code_N
// Input (stdin), one case per line:   q c x_0 x_1 ... x_q   (%.17g doubles)
// Output (stdout), CSV:
//   q,c,N,S_mant,S_exp,ku,kuN,prod_status,prod_N,prod_kuN
// where (N, S) come from evaluate_normalized_scaled() on [-c, c] with the
// conditioning threshold disabled (so the interval is never changed), and
// prod_* from the default evaluate_scaled() (which may retry on
// [min x, max x] or refuse).
// =============================================================================

#include "chebyshev_exp_divdiff.hpp"

#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main() {
    std::string line;
    std::printf("q,c,N,S_mant,S_exp,ku,kuN,prod_status,prod_N,prod_kuN\n");
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::istringstream is(line);
        int q;
        double c;
        is >> q >> c;
        std::vector<double> x(q + 1);
        for (auto& v : x) is >> v;

        chebyshev::ExpDivDiff<double> E(-c, c, 1e-14, 10000);
        E.set_max_error_estimate(1e300);
        int N = -1;
        const auto S = E.evaluate_normalized_scaled(x, &N);
        const double ku = static_cast<double>(E.last_condition_estimate());
        const double kuN = static_cast<double>(E.last_error_estimate());

        chebyshev::ExpDivDiff<double> P(-c, c, 1e-14, 10000);
        int Np = -1;
        std::string st = "ok";
        try { (void)P.evaluate_scaled(x, &Np); }
        catch (const std::runtime_error&) { st = "refused"; }

        std::printf("%d,%g,%d,%.17g,%ld,%.6e,%.6e,%s,%d,%.6e\n", q, c, N, S.mantissa,
                    S.exponent, ku, kuN, st.c_str(), Np,
                    static_cast<double>(P.last_error_estimate()));
    }
    return 0;
}
