// e9_squaring.cpp -- calibration of the scaling-and-squaring path.
//
// Reads a node file (instance,nodes) and evaluates every set with the
// squaring path forced (ExpDivDiff::evaluate_squaring_scaled), in working
// precision (prec 1) and in double-double (prec 2), and with the default
// evaluate_scaled (which uses the squaring path only as a last resort).
// Output CSV on stdout:
//   instance,q,c,s,m1,x1,est1,ms1,m2,x2,est2,ms2,md,xd,estdef,sqdef,precdef
// value = m * 2^x (any magnitude).
//
// Build: g++ -std=c++17 -O3 -I../.. -I/opt/homebrew/include e9_squaring.cpp
#include "chebyshev_exp_divdiff.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: e9_squaring NODES.csv [skip_dd_above_q]\n"); return 1; }
    const int dd_qmax = argc > 2 ? std::atoi(argv[2]) : 1 << 30;
    std::ifstream f(argv[1]);
    std::string line;
    std::getline(f, line);
    std::printf("instance,q,c,s,m1,x1,est1,ms1,m2,x2,est2,ms2,md,xd,estdef,sqdef,precdef\n");
    using clk = std::chrono::steady_clock;
    while (std::getline(f, line)) {
        const auto p = line.find(',');
        const int inst = std::stoi(line.substr(0, p));
        std::istringstream ss(line.substr(p + 1));
        std::vector<double> x;
        double v;
        while (ss >> v) x.push_back(v);
        const int q = static_cast<int>(x.size()) - 1;
        const double lo = *std::min_element(x.begin(), x.end());
        const double hi = *std::max_element(x.begin(), x.end());
        chebyshev::ExpDivDiff<double> e(lo, hi, 1e-14, 1000000);
        long double e1 = NAN, e2 = NAN, ed = NAN;
        double m1 = NAN, m2 = NAN, md = NAN; long x1 = 0, x2 = 0, xd = 0;
        double ms1 = NAN, ms2 = NAN;
        int s = 0, sqd = -1, prd = -1;
        try {
            auto t0 = clk::now();
            const auto r = e.evaluate_squaring_scaled(x, 1);
            ms1 = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
            m1 = r.mantissa; x1 = r.exponent; e1 = e.last_error_estimate(); s = e.last_squarings();
        } catch (const std::exception& ex) { std::fprintf(stderr, "%d prec1: %s\n", inst, ex.what()); }
        if (q <= dd_qmax) {
            try {
                auto t0 = clk::now();
                const auto r = e.evaluate_squaring_scaled(x, 2);
                ms2 = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
                m2 = r.mantissa; x2 = r.exponent; e2 = e.last_error_estimate();
            } catch (const std::exception& ex) { std::fprintf(stderr, "%d prec2: %s\n", inst, ex.what()); }
        }
        try {
            const auto r = e.evaluate_scaled(x);
            md = r.mantissa; xd = r.exponent; ed = e.last_error_estimate();
            sqd = e.last_squarings(); prd = e.last_squaring_precision();
        } catch (const std::exception& ex) { std::fprintf(stderr, "%d default: %s\n", inst, ex.what()); }
        std::printf("%d,%d,%.17g,%d,%.17g,%ld,%.3Le,%.3f,%.17g,%ld,%.3Le,%.3f,%.17g,%ld,%.3Le,%d,%d\n",
                    inst, q, hi / 2 - lo / 2, s, m1, x1, e1, ms1, m2, x2, e2, ms2, md, xd, ed, sqd, prd);
        std::fflush(stdout);
    }
    return 0;
}
