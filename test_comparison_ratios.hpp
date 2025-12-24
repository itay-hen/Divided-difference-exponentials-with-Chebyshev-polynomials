#ifndef TEST_COMPARISON_RATIOS_HPP
#define TEST_COMPARISON_RATIOS_HPP

#include "mccurdy.hpp"
#include "chebyshev_exp_divdiff.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>

/**
 * @file test_comparison_ratios.hpp
 * @brief Test ratios of exponential divided differences
 * 
 * This tests the key advantage of the Chebyshev normalized method:
 * When computing ratios of divided differences with same interval,
 * the scaling factors cancel out, allowing higher accuracy.
 * 
 * For Chebyshev:
 *   exp[nodes1] / exp[nodes2] = (exp(d)*I0(c)/c^q * S1) / (exp(d)*I0(c)/c^q * S2)
 *                              = S1 / S2
 * 
 * This eliminates the large scaling factors and works purely with O(1) normalized sums!
 */

namespace test_comparison_ratios {

/**
 * @brief Result for a single ratio trial
 */
struct RatioTrialResult {
    double b;              // Interval half-width
    int q;                 // Divided difference order
    int trial;             // Trial number
    
    // Individual divided differences
    double mccurdy_num;      // Numerator (McCurdy)
    double mccurdy_den;      // Denominator (McCurdy)
    double cheby_norm_num;   // Numerator (Chebyshev normalized)
    double cheby_norm_den;   // Denominator (Chebyshev normalized)
    double cheby_full_num;   // Numerator (Chebyshev full)
    double cheby_full_den;   // Denominator (Chebyshev full)
    
    // Computed ratios
    double mccurdy_ratio;
    double cheby_normalized_ratio;
    double cheby_full_ratio;
    double reference_ratio;  // Long double McCurdy reference
    
    // Errors
    double mccurdy_error;
    double cheby_normalized_error;
    double cheby_full_error;
    
    // Performance
    double mccurdy_time_us;
    double cheby_normalized_time_us;
    double cheby_full_time_us;
    
    int cheby_n_terms_num;
    int cheby_n_terms_den;
    
    bool mccurdy_valid;
    bool cheby_normalized_valid;
    bool cheby_full_valid;
    
    RatioTrialResult() : b(0), q(0), trial(0),
                         mccurdy_num(0), mccurdy_den(0),
                         cheby_norm_num(0), cheby_norm_den(0),
                         cheby_full_num(0), cheby_full_den(0),
                         mccurdy_ratio(0), cheby_normalized_ratio(0), cheby_full_ratio(0),
                         reference_ratio(0),
                         mccurdy_error(0), cheby_normalized_error(0), cheby_full_error(0),
                         mccurdy_time_us(0), cheby_normalized_time_us(0), cheby_full_time_us(0),
                         cheby_n_terms_num(0), cheby_n_terms_den(0),
                         mccurdy_valid(false), cheby_normalized_valid(false), cheby_full_valid(false) {}
};

/**
 * @brief Run a single ratio trial
 */
inline RatioTrialResult run_ratio_trial(double b, int q, int trial, std::mt19937& rng) {
    RatioTrialResult result;
    result.b = b;
    result.q = q;
    result.trial = trial;
    
    // Generate two different sets of random nodes in [-b, b]
    std::uniform_real_distribution<double> dist(-b, b);
    
    std::vector<double> nodes_num(q + 1);
    std::vector<double> nodes_den(q + 1);
    
    for (int i = 0; i <= q; ++i) {
        nodes_num[i] = dist(rng);
        nodes_den[i] = dist(rng);
    }
    
    // Sort nodes
    std::sort(nodes_num.begin(), nodes_num.end());
    std::sort(nodes_den.begin(), nodes_den.end());
    
    // === Compute reference (long double McCurdy) ===
    try {
        std::vector<long double> nodes_num_ld(nodes_num.begin(), nodes_num.end());
        std::vector<long double> nodes_den_ld(nodes_den.begin(), nodes_den.end());
        
        long double num_ref = mccurdy_templated::ExpDivDiff<long double>::compute(nodes_num_ld);
        long double den_ref = mccurdy_templated::ExpDivDiff<long double>::compute(nodes_den_ld);
        
        if (std::abs(den_ref) > 1e-100) {
            result.reference_ratio = static_cast<double>(num_ref / den_ref);
        }
    } catch (...) {
        // Reference computation failed - mark as invalid
        return result;
    }
    
    // === McCurdy (double) ===
    try {
        auto start = std::chrono::high_resolution_clock::now();
        
        result.mccurdy_num = mccurdy_templated::ExpDivDiff<double>::compute(nodes_num);
        result.mccurdy_den = mccurdy_templated::ExpDivDiff<double>::compute(nodes_den);
        
        auto end = std::chrono::high_resolution_clock::now();
        result.mccurdy_time_us = std::chrono::duration<double, std::micro>(end - start).count();
        
        if (std::abs(result.mccurdy_den) > 1e-100) {
            result.mccurdy_ratio = result.mccurdy_num / result.mccurdy_den;
            result.mccurdy_valid = true;
            
            if (std::abs(result.reference_ratio) > 1e-100) {
                result.mccurdy_error = std::abs((result.mccurdy_ratio - result.reference_ratio) 
                                                / result.reference_ratio);
            }
        }
    } catch (...) {
        result.mccurdy_valid = false;
    }
    
    // === Chebyshev (full evaluation) ===
    try {
        chebyshev::ExpDivDiff<double> solver(-b, b);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        int n_terms_num = 0, n_terms_den = 0;
        result.cheby_full_num = solver.evaluate(nodes_num, &n_terms_num);
        result.cheby_full_den = solver.evaluate(nodes_den, &n_terms_den);
        
        auto end = std::chrono::high_resolution_clock::now();
        result.cheby_full_time_us = std::chrono::duration<double, std::micro>(end - start).count();
        
        result.cheby_n_terms_num = n_terms_num;
        result.cheby_n_terms_den = n_terms_den;
        
        if (std::abs(result.cheby_full_den) > 1e-100) {
            result.cheby_full_ratio = result.cheby_full_num / result.cheby_full_den;
            result.cheby_full_valid = true;
            
            if (std::abs(result.reference_ratio) > 1e-100) {
                result.cheby_full_error = std::abs((result.cheby_full_ratio - result.reference_ratio) 
                                                   / result.reference_ratio);
            }
        }
    } catch (...) {
        result.cheby_full_valid = false;
    }
    
    // === Chebyshev (normalized evaluation) ===
    // KEY: Scaling factors cancel in the ratio!
    try {
        chebyshev::ExpDivDiff<double> solver(-b, b);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        int n_terms_num = 0, n_terms_den = 0;
        result.cheby_norm_num = solver.evaluate_normalized(nodes_num, &n_terms_num);
        result.cheby_norm_den = solver.evaluate_normalized(nodes_den, &n_terms_den);
        
        auto end = std::chrono::high_resolution_clock::now();
        result.cheby_normalized_time_us = std::chrono::duration<double, std::micro>(end - start).count();
        
        if (std::abs(result.cheby_norm_den) > 1e-100) {
            result.cheby_normalized_ratio = result.cheby_norm_num / result.cheby_norm_den;
            result.cheby_normalized_valid = true;
            
            if (std::abs(result.reference_ratio) > 1e-100) {
                result.cheby_normalized_error = std::abs((result.cheby_normalized_ratio - result.reference_ratio) 
                                                        / result.reference_ratio);
            }
        }
    } catch (...) {
        result.cheby_normalized_valid = false;
    }
    
    return result;
}

/**
 * @brief Test ratios of divided differences
 */
inline void test_ratio_case(double b, int q, int n_trials = 50, 
                           unsigned int random_seed = 12345,
                           std::ostream& os = std::cout) {
    os << "\n";
    os << "╔════════════════════════════════════════════════════════════════════════════╗\n";
    os << "║  Ratio Test: exp[nodes1] / exp[nodes2]                                    ║\n";
    os << "╚════════════════════════════════════════════════════════════════════════════╝\n";
    os << "\n";
    
    os << "Configuration:\n";
    os << "  Interval: [-" << b << ", " << b << "]\n";
    os << "  Divided difference order: q = " << q << " (" << (q+1) << " nodes)\n";
    os << "  Number of trials: " << n_trials << "\n";
    os << "  Random seed: " << random_seed << "\n\n";
    
    os << "Testing advantage of normalized Chebyshev:\n";
    os << "  For ratios with same interval, scaling factors cancel:\n";
    os << "  ratio = (exp(d)*I0(c)/c^q * S1) / (exp(d)*I0(c)/c^q * S2)\n";
    os << "        = S1 / S2  (normalized form!)\n\n";
    
    std::mt19937 rng(random_seed);
    std::vector<RatioTrialResult> results;
    
    os << "Running trials...\n";
    for (int trial = 0; trial < n_trials; ++trial) {
        RatioTrialResult result = run_ratio_trial(b, q, trial, rng);
        results.push_back(result);
        
        if ((trial + 1) % 10 == 0 || trial == 0) {
            os << "\rProgress: " << ((trial + 1) * 100 / n_trials) << "% "
               << "(" << (trial + 1) << "/" << n_trials << ")" << std::flush;
        }
    }
    os << "\n\n";
    
    // Print sample results
    os << "Sample Results (first 15 trials):\n";
    os << std::setw(6) << "Trial"
       << std::setw(18) << "McCurdy Ratio"
       << std::setw(18) << "Cheby Norm"
       << std::setw(18) << "Cheby Full"
       << std::setw(12) << "Err M"
       << std::setw(12) << "Err CN"
       << std::setw(12) << "Err CF"
       << "\n";
    os << std::string(96, '-') << "\n";
    
    for (size_t i = 0; i < std::min(size_t(15), results.size()); ++i) {
        const auto& r = results[i];
        os << std::setw(6) << i;
        
        if (r.mccurdy_valid) {
            os << std::scientific << std::setprecision(6) << std::setw(18) << r.mccurdy_ratio;
        } else {
            os << std::setw(18) << "FAILED";
        }
        
        if (r.cheby_normalized_valid) {
            os << std::scientific << std::setprecision(6) << std::setw(18) << r.cheby_normalized_ratio;
        } else {
            os << std::setw(18) << "FAILED";
        }
        
        if (r.cheby_full_valid) {
            os << std::scientific << std::setprecision(6) << std::setw(18) << r.cheby_full_ratio;
        } else {
            os << std::setw(18) << "FAILED";
        }
        
        if (r.mccurdy_valid) {
            os << std::scientific << std::setprecision(2) << std::setw(12) << r.mccurdy_error;
        } else {
            os << std::setw(12) << "N/A";
        }
        
        if (r.cheby_normalized_valid) {
            os << std::scientific << std::setprecision(2) << std::setw(12) << r.cheby_normalized_error;
        } else {
            os << std::setw(12) << "N/A";
        }
        
        if (r.cheby_full_valid) {
            os << std::scientific << std::setprecision(2) << std::setw(12) << r.cheby_full_error;
        } else {
            os << std::setw(12) << "N/A";
        }
        
        os << "\n";
    }
    os << "\n";
    
    // Compute statistics
    int mccurdy_success = 0, cheby_norm_success = 0, cheby_full_success = 0;
    double mccurdy_avg_error = 0, cheby_norm_avg_error = 0, cheby_full_avg_error = 0;
    double mccurdy_max_error = 0, cheby_norm_max_error = 0, cheby_full_max_error = 0;
    
    for (const auto& r : results) {
        if (r.mccurdy_valid) {
            mccurdy_success++;
            mccurdy_avg_error += r.mccurdy_error;
            mccurdy_max_error = std::max(mccurdy_max_error, r.mccurdy_error);
        }
        if (r.cheby_normalized_valid) {
            cheby_norm_success++;
            cheby_norm_avg_error += r.cheby_normalized_error;
            cheby_norm_max_error = std::max(cheby_norm_max_error, r.cheby_normalized_error);
        }
        if (r.cheby_full_valid) {
            cheby_full_success++;
            cheby_full_avg_error += r.cheby_full_error;
            cheby_full_max_error = std::max(cheby_full_max_error, r.cheby_full_error);
        }
    }
    
    if (mccurdy_success > 0) mccurdy_avg_error /= mccurdy_success;
    if (cheby_norm_success > 0) cheby_norm_avg_error /= cheby_norm_success;
    if (cheby_full_success > 0) cheby_full_avg_error /= cheby_full_success;
    
    // Compute accurate digits
    auto compute_digits = [](double error) {
        if (error <= 0 || std::isinf(error)) return 0;
        return std::max(0, -static_cast<int>(std::log10(error)));
    };
    
    int mccurdy_avg_digits = compute_digits(mccurdy_avg_error);
    int cheby_norm_avg_digits = compute_digits(cheby_norm_avg_error);
    int cheby_full_avg_digits = compute_digits(cheby_full_avg_error);
    
    // Print summary
    os << "Summary Statistics:\n";
    os << std::string(80, '-') << "\n";
    
    os << "Success rates:\n";
    os << "  McCurdy:             " << mccurdy_success << " / " << n_trials 
       << " (" << (100.0 * mccurdy_success / n_trials) << "%)\n";
    os << "  Chebyshev Normalized: " << cheby_norm_success << " / " << n_trials 
       << " (" << (100.0 * cheby_norm_success / n_trials) << "%)\n";
    os << "  Chebyshev Full:      " << cheby_full_success << " / " << n_trials 
       << " (" << (100.0 * cheby_full_success / n_trials) << "%)\n\n";
    
    os << "Accuracy (accurate decimal digits):\n";
    os << "  McCurdy:             avg = " << mccurdy_avg_digits 
       << ", max_error = " << std::scientific << std::setprecision(3) << mccurdy_max_error << "\n";
    os << "  Chebyshev Normalized: avg = " << cheby_norm_avg_digits
       << ", max_error = " << std::scientific << std::setprecision(3) << cheby_norm_max_error << "\n";
    os << "  Chebyshev Full:      avg = " << cheby_full_avg_digits
       << ", max_error = " << std::scientific << std::setprecision(3) << cheby_full_max_error << "\n\n";
    
    // Highlight improvement
    if (cheby_norm_avg_digits > cheby_full_avg_digits) {
        os << "✅ NORMALIZED ADVANTAGE: " << (cheby_norm_avg_digits - cheby_full_avg_digits) 
           << " extra digits!\n";
        os << "   Normalized: " << cheby_norm_avg_digits << " digits\n";
        os << "   Full:       " << cheby_full_avg_digits << " digits\n\n";
    } else if (cheby_norm_avg_digits == cheby_full_avg_digits) {
        os << "≈ Same accuracy for normalized vs full (both work well)\n\n";
    } else {
        os << "⚠️  Normalized slightly worse (unexpected)\n\n";
    }
    
    if (cheby_norm_avg_digits > mccurdy_avg_digits) {
        os << "✅ Chebyshev Normalized outperforms McCurdy by "
           << (cheby_norm_avg_digits - mccurdy_avg_digits) << " digits!\n\n";
    }
    
    // Error improvement factor
    if (cheby_full_avg_error > 0 && cheby_norm_avg_error > 0) {
        double improvement = cheby_full_avg_error / cheby_norm_avg_error;
        if (improvement > 1.5) {
            os << "📊 Normalized is " << std::fixed << std::setprecision(1) 
               << improvement << "× more accurate than full!\n\n";
        }
    }
    
    os << "╔════════════════════════════════════════════════════════════════════════════╗\n";
    os << "║  Test Complete                                                             ║\n";
    os << "╚════════════════════════════════════════════════════════════════════════════╝\n";
    os << "\n";
}

} // namespace test_comparison_ratios

#endif // TEST_COMPARISON_RATIOS_HPP
