// exact_eval.cpp -- evaluates node sets (one per line on stdin) with
//   default  ExpDivDiff::evaluate_scaled on [min x, max x] (with retries),
//   tight    the same with set_lowered_retry(false),
//   gbh      a fresh evaluation with the PMR-QMC divdiff class,
// printing each value as mantissa and binary exponent ("nan" if refused;
// GBH: mantissa, exponent of q! times the value) and the indicators.  The
// values are not printed as logarithms: long double is double on arm64, and
// log|v| would then carry an absolute error of about u |log v|.  Used by
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
        double mant[2] = {NAN, NAN}; long ex[2] = {0, 0}; long double est[2] = {NAN, NAN};
        for (int k = 0; k < 2; ++k) {
            chebyshev::ExpDivDiff<double> E(lo < hi ? lo : lo - 1, lo < hi ? hi : hi + 1);
            E.set_lowered_retry(k == 0);
            try { const auto r = E.evaluate_scaled(x); mant[k] = r.mantissa; ex[k] = r.exponent; } catch (const std::runtime_error&) {}
            est[k] = E.last_error_estimate();
        }
        divdiff d(static_cast<int>(x.size()) + 4, 500);
        for (double t : x) d.AddElement(t);
        const int q = static_cast<int>(x.size()) - 1;
        ExExFloat g = d.divdiffs[q];
        std::printf("%d %.17g %ld %.3Le %.17g %ld %.3Le %.17g %ld\n", q, mant[0], ex[0], est[0],
                    mant[1], ex[1], est[1], static_cast<double>(g.get_mantissa()),
                    static_cast<long>(g.get_exponent()));
    }
}
