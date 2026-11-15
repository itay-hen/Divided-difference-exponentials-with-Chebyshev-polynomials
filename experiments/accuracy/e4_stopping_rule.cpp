// e4_stopping_rule.cpp -- silent failures of the window stopping rule
// (W = 5 consecutive small terms) versus the certified rule.
//
// Node sets on [-c, c] (q + 1 nodes):
//   lobatto : x_j = c cos(pi j / q)            (Chebyshev--Lobatto points)
//   lobatto_shift : Lobatto points of [-c, c'] with c' = c(1 - 2/q),
//                   i.e. not symmetric about the mapping midpoint
//   uniform : uniform random, with x_0 = c, x_1 = -c
// Reference: the corrected MNP method (mccurdy.hpp), accurate to ~1e-14
// in this range (c <= 50); errors are reported only as a classification
// above 1e-10, far above the reference error.
// Output CSV (stdout): set,c,q,err_window,N_window,est_window,
//                      err_cert,N_cert,est_cert
// Build: g++ -std=c++17 -O2 -I../.. -I/opt/homebrew/include e4_stopping_rule.cpp
#include "chebyshev_exp_divdiff.hpp"
#include "mccurdy.hpp"
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

int main() {
    std::printf("set,c,q,err_window,N_window,est_window,err_cert,N_cert,est_cert\n");
    std::mt19937 rng(4);
    for (const std::string set : {"lobatto", "lobatto_shift", "uniform"}) {
        for (double c : {1.0, 5.0, 20.0, 50.0}) {
            for (int q = 2; q <= 80; ++q) {
                std::vector<double> x(q + 1);
                if (set == "lobatto") {
                    for (int j = 0; j <= q; ++j) x[j] = c * std::cos(M_PI * j / q);
                } else if (set == "lobatto_shift") {
                    const double hi = c * (1.0 - 2.0 / q), mid = (hi - c) / 2, hw = (hi + c) / 2;
                    for (int j = 0; j <= q; ++j) x[j] = mid + hw * std::cos(M_PI * j / q);
                } else {
                    std::uniform_real_distribution<double> U(-c, c);
                    for (auto& v : x) v = U(rng);
                    x[0] = c; x[1] = -c;
                }
                const double ref = mccurdy_templated::ExpDivDiff<double>::compute(x);
                double err[2] = {NAN, NAN}, est[2] = {NAN, NAN}; int N[2] = {-1, -1};
                for (int m = 0; m < 2; ++m) {
                    chebyshev::ExpDivDiff<double> e(-c, c);
                    e.set_stopping_rule(m == 0 ? chebyshev::StoppingRule::Window
                                               : chebyshev::StoppingRule::Certified);
                    try {
                        const double v = e.evaluate(x, &N[m]);
                        err[m] = std::abs(v - ref) / std::abs(ref);
                        est[m] = static_cast<double>(e.last_error_estimate());
                    } catch (const std::runtime_error&) {
                        est[m] = static_cast<double>(e.last_error_estimate());
                    }
                }
                std::printf("%s,%g,%d,%.3e,%d,%.3e,%.3e,%d,%.3e\n", set.c_str(), c, q,
                            err[0], N[0], est[0], err[1], N[1], est[1]);
            }
        }
    }
}
