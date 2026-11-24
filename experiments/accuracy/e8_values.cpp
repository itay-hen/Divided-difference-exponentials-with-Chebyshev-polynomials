// e8_values.cpp -- accuracy of single values exp[x_0..x_q] over a (q, c)
// grid: Chebyshev--Bessel (evaluate_scaled, default settings: threshold
// 1e-10 and the retry sequence) and MNP in double precision.
//
// Input : the node file of e7_ratio (instance,nodes; nodes uniform on
//         [-c, c] with x_0 = -c and x_q = c, 40 sets per (q, c) cell).
// Output: CSV on stdout
//   instance,cheb_ok,cheb_mant,cheb_exp,cheb_N,cheb_est,mnp
//   (cheb_ok = 0: refused; mnp = nan if not finite and positive; values
//    below the smallest normal double count as underflow downstream)
//
// Build: g++ -std=c++17 -O3 -I../.. -I/opt/homebrew/include e8_values.cpp
#include "chebyshev_exp_divdiff.hpp"
#include "mccurdy.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: e8_values NODES.csv\n"); return 1; }
    std::ifstream f(argv[1]);
    std::string line;
    std::getline(f, line);
    std::printf("instance,cheb_ok,cheb_mant,cheb_exp,cheb_N,cheb_est,cheb_ext,mnp\n");
    while (std::getline(f, line)) {
        const auto p = line.find(',');
        const int inst = std::stoi(line.substr(0, p));
        std::istringstream ss(line.substr(p + 1));
        std::vector<double> x;
        double v;
        while (ss >> v) x.push_back(v);
        const double lo = *std::min_element(x.begin(), x.end());
        const double hi = *std::max_element(x.begin(), x.end());
        int ok = 0, N = -1, sq = 0;
        double mant = NAN;
        long ex = 0;
        long double est = NAN;
        try {
            chebyshev::ExpDivDiff<double> e(lo, hi, 1e-14, 100000);
            const auto r = e.evaluate_scaled(x, &N);
            ok = 1; mant = r.mantissa; ex = r.exponent; est = e.last_error_estimate(); sq = (e.last_precision_bits() > 53 ? e.last_precision_bits() : 0);
        } catch (...) {}
        double mnp = NAN;
        try {
            const double m = mccurdy_templated::ExpDivDiff<double>::compute(x);
            if (std::isfinite(m) && m > 0) mnp = m;
        } catch (...) {}
        std::printf("%d,%d,%.17g,%ld,%d,%.3Le,%d,%.17g\n", inst, ok, mant, ex, N, est, sq, mnp);
    }
    return 0;
}
