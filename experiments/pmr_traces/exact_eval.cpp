// exact_eval.cpp -- evaluates node sets (one per line on stdin) with
//   default  ExpDivDiff::evaluate_scaled on [min x, max x] (with retries),
//   tight    the same with set_lowered_retry(false),
//   gbh      a fresh evaluation with the PMR-QMC divdiff class,
// printing log values ("nan" if refused) and indicators.  Used by
// exact_check.py, which compares them with exact values.
// Build: g++ -std=c++17 -O2 -I../.. -I/opt/homebrew/include exact_eval.cpp -o exact_eval
#include "chebyshev_exp_divdiff.hpp"
#include "divdiff_pmrqmc.hpp"
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
int main() {
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream is(line); std::vector<double> x; double v;
        while (is >> v) x.push_back(v);
        const auto mm = std::minmax_element(x.begin(), x.end());
        const double lo = *mm.first, hi = *mm.second;
        long double out[2] = {NAN, NAN}, est[2] = {NAN, NAN};
        for (int k = 0; k < 2; ++k) {
            chebyshev::ExpDivDiff<double> E(lo < hi ? lo : lo - 1, lo < hi ? hi : hi + 1);
            E.set_lowered_retry(k == 0);
            try { out[k] = E.evaluate_scaled(x).log_abs(); } catch (const std::runtime_error&) {}
            est[k] = E.last_error_estimate();
        }
        divdiff d(static_cast<int>(x.size()) + 4, 500);
        for (double t : x) d.AddElement(t);
        const int q = static_cast<int>(x.size()) - 1;
        ExExFloat g = d.divdiffs[q];
        const long double gl = std::log(std::fabs(static_cast<long double>(g.get_mantissa()))) +
            g.get_exponent() * 0.693147180559945309417232121458176568L - std::lgamma(q + 1.0L);
        std::printf("%d %.18Le %.3Le %.18Le %.3Le %.18Le\n", q, out[0], est[0], out[1], est[1], gl);
    }
}
