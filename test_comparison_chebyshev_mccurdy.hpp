#ifndef TEST_COMPARISON_HPP
#define TEST_COMPARISON_HPP

#include "mccurdy.hpp"
#include "chebyshev_exp_divdiff.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <set>

/**
 * @file test_comparison.hpp
 * @brief Comprehensive comparison of McCurdy vs Chebyshev exponential divided difference methods
 *
 * Test structure:
 * - Loop over interval widths b (testing [-b, b])
 * - For each b, initialize Chebyshev object
 * - Loop over divided difference orders q (q+1 nodes)
 * - For each q, run Ntrials random trials
 * - Each trial: generate q+1 random nodes in [-b, b]
 * - Compute result with both methods and compare
 *
 * Usage:
 *   #include "test_comparison.hpp"
 *   int main() {
 *       test_comparison::run_all_tests("comparison_results.csv");
 *       return 0;
 *   }
 */

namespace test_comparison {

/**
 * @brief Single test result for one trial
 */
struct TrialResult {
    double b;              // Interval half-width
    int q;                 // Divided difference order
    int trial;             // Trial number
    
    double mccurdy_result;
    double chebyshev_result;
    double reference_result;  // Use long double McCurdy as reference
    
    double mccurdy_error;
    double chebyshev_error;
    
    int chebyshev_n_terms;  // Number of Chebyshev terms used
    
    double mccurdy_time_us;   // Microseconds
    double chebyshev_time_us;
    
    bool mccurdy_valid;
    bool chebyshev_valid;
    
    TrialResult() : b(0), q(0), trial(0),
                    mccurdy_result(0), chebyshev_result(0), reference_result(0),
                    mccurdy_error(0), chebyshev_error(0),
                    chebyshev_n_terms(0),
                    mccurdy_time_us(0), chebyshev_time_us(0),
                    mccurdy_valid(false), chebyshev_valid(false) {}
};

/**
 * @brief Summary statistics for a (b, q) combination
 */
struct ComboStats {
    double b;
    int q;
    int n_trials;
    
    // Accuracy statistics
    double mccurdy_avg_error;
    double mccurdy_max_error;
    double mccurdy_avg_digits;
    
    double chebyshev_avg_error;
    double chebyshev_max_error;
    double chebyshev_avg_digits;
    
    // Performance statistics
    double mccurdy_avg_time_us;
    double chebyshev_avg_time_us;
    double speedup_factor;  // mccurdy_time / chebyshev_time
    
    double chebyshev_avg_n_terms;
    
    int mccurdy_failures;
    int chebyshev_failures;
    
    ComboStats() : b(0), q(0), n_trials(0),
                   mccurdy_avg_error(0), mccurdy_max_error(0), mccurdy_avg_digits(0),
                   chebyshev_avg_error(0), chebyshev_max_error(0), chebyshev_avg_digits(0),
                   mccurdy_avg_time_us(0), chebyshev_avg_time_us(0), speedup_factor(0),
                   chebyshev_avg_n_terms(0),
                   mccurdy_failures(0), chebyshev_failures(0) {}
};

/**
 * @brief Test configuration
 */
struct TestConfig {
    std::vector<double> b_values;
    std::vector<int> q_values;
    int n_trials;
    unsigned int random_seed;
    
    static TestConfig create_default() {
        TestConfig config;
        
        // Interval half-widths: very small to large
        config.b_values = {0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0};
        
        // Divided difference orders
        config.q_values = {0, 1, 2, 3, 5, 10, 15, 20, 30, 50};
        
        config.n_trials = 100;  // Trials per (b, q) combination
        config.random_seed = 12345;
        
        return config;
    }
    
    static TestConfig create_quick() {
        TestConfig config;
        
        config.b_values = {0.1, 1.0, 10.0};
        config.q_values = {1, 5, 10, 20};
        config.n_trials = 20;
        config.random_seed = 12345;
        
        return config;
    }
    
    static TestConfig create_comprehensive() {
        TestConfig config;
        
        // Very fine grid of b values
        for (double b = 0.01; b <= 0.1; b += 0.01) {
            config.b_values.push_back(b);
        }
        for (double b = 0.2; b <= 1.0; b += 0.1) {
            config.b_values.push_back(b);
        }
        for (double b = 2.0; b <= 10.0; b += 1.0) {
            config.b_values.push_back(b);
        }
        for (double b = 15.0; b <= 50.0; b += 5.0) {
            config.b_values.push_back(b);
        }
        config.b_values.push_back(100.0);
        
        // Dense q values
        for (int q = 0; q <= 10; ++q) {
            config.q_values.push_back(q);
        }
        for (int q = 12; q <= 30; q += 2) {
            config.q_values.push_back(q);
        }
        for (int q = 35; q <= 100; q += 5) {
            config.q_values.push_back(q);
        }
        
        config.n_trials = 50;
        config.random_seed = 12345;
        
        return config;
    }
};

/**
 * @brief Compute accurate digits from relative error
 */
inline int count_accurate_digits(double computed, double reference) {
    if (std::abs(reference) < 1e-100) return 0;
    double rel_error = std::abs((computed - reference) / reference);
    if (rel_error == 0) return 16;
    return std::max(0, -static_cast<int>(std::log10(rel_error)));
}

/**
 * @brief Generate random nodes in [-b, b]
 */
inline std::vector<double> generate_random_nodes(int count, double b, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(-b, b);
    std::vector<double> nodes(count);
    for (int i = 0; i < count; ++i) {
        nodes[i] = dist(rng);
    }
    return nodes;
}

/**
 * @brief Run a single trial
 */
inline TrialResult run_single_trial(double b, int q, int trial_num, std::mt19937& rng) {
    TrialResult result;
    result.b = b;
    result.q = q;
    result.trial = trial_num;
    
    // Generate random nodes
    std::vector<double> nodes = generate_random_nodes(q + 1, b, rng);
    std::vector<long double> nodes_ld(nodes.begin(), nodes.end());
    
    // Compute reference using long double McCurdy
    try {
        result.reference_result = static_cast<double>(
            mccurdy_templated::ExpDivDiff<long double>::compute(nodes_ld)
        );
    } catch (...) {
        // If reference fails, this trial is invalid
        return result;
    }
    
    // Test McCurdy (double)
    try {
        auto start = std::chrono::high_resolution_clock::now();
        result.mccurdy_result = mccurdy_templated::ExpDivDiff<double>::compute(nodes);
        auto end = std::chrono::high_resolution_clock::now();
        
        result.mccurdy_time_us = std::chrono::duration<double, std::micro>(end - start).count();
        result.mccurdy_valid = !std::isnan(result.mccurdy_result) && !std::isinf(result.mccurdy_result);
        
        if (result.mccurdy_valid) {
            result.mccurdy_error = std::abs((result.mccurdy_result - result.reference_result) / result.reference_result);
        }
    } catch (...) {
        result.mccurdy_valid = false;
    }
    
    // Test Chebyshev
    try {
        chebyshev::ExpDivDiff<double> chebyshev_solver(-b, b);
        
        auto start = std::chrono::high_resolution_clock::now();
        int n_terms = 0;
        result.chebyshev_result = chebyshev_solver.evaluate(nodes, &n_terms);
        auto end = std::chrono::high_resolution_clock::now();
        
        result.chebyshev_time_us = std::chrono::duration<double, std::micro>(end - start).count();
        result.chebyshev_n_terms = n_terms;
        result.chebyshev_valid = !std::isnan(result.chebyshev_result) && !std::isinf(result.chebyshev_result);
        
        if (result.chebyshev_valid) {
            result.chebyshev_error = std::abs((result.chebyshev_result - result.reference_result) / result.reference_result);
        }
    } catch (...) {
        result.chebyshev_valid = false;
    }
    
    return result;
}

/**
 * @brief Compute statistics for a (b, q) combination
 */
inline ComboStats compute_combo_stats(const std::vector<TrialResult>& trials) {
    if (trials.empty()) return ComboStats();
    
    ComboStats stats;
    stats.b = trials[0].b;
    stats.q = trials[0].q;
    stats.n_trials = trials.size();
    
    int mccurdy_valid_count = 0;
    int chebyshev_valid_count = 0;
    
    for (const auto& trial : trials) {
        if (trial.mccurdy_valid) {
            stats.mccurdy_avg_error += trial.mccurdy_error;
            stats.mccurdy_max_error = std::max(stats.mccurdy_max_error, trial.mccurdy_error);
            stats.mccurdy_avg_time_us += trial.mccurdy_time_us;
            mccurdy_valid_count++;
        } else {
            stats.mccurdy_failures++;
        }
        
        if (trial.chebyshev_valid) {
            stats.chebyshev_avg_error += trial.chebyshev_error;
            stats.chebyshev_max_error = std::max(stats.chebyshev_max_error, trial.chebyshev_error);
            stats.chebyshev_avg_time_us += trial.chebyshev_time_us;
            stats.chebyshev_avg_n_terms += trial.chebyshev_n_terms;
            chebyshev_valid_count++;
        } else {
            stats.chebyshev_failures++;
        }
    }
    
    if (mccurdy_valid_count > 0) {
        stats.mccurdy_avg_error /= mccurdy_valid_count;
        stats.mccurdy_avg_time_us /= mccurdy_valid_count;
        stats.mccurdy_avg_digits = -std::log10(stats.mccurdy_avg_error + 1e-16);
    }
    
    if (chebyshev_valid_count > 0) {
        stats.chebyshev_avg_error /= chebyshev_valid_count;
        stats.chebyshev_avg_time_us /= chebyshev_valid_count;
        stats.chebyshev_avg_n_terms /= chebyshev_valid_count;
        stats.chebyshev_avg_digits = -std::log10(stats.chebyshev_avg_error + 1e-16);
    }
    
    if (stats.chebyshev_avg_time_us > 0) {
        stats.speedup_factor = stats.mccurdy_avg_time_us / stats.chebyshev_avg_time_us;
    }
    
    return stats;
}

/**
 * @brief Print progress update
 */
inline void print_progress(int current, int total, const std::string& label) {
    int percent = (100 * current) / total;
    std::cout << "\r" << label << ": " << percent << "% (" << current << "/" << total << ")" << std::flush;
}

/**
 * @brief Write detailed CSV output
 */
inline void write_detailed_csv(const std::vector<TrialResult>& results, const std::string& filename) {
    std::ofstream csv(filename);
    if (!csv.is_open()) {
        std::cerr << "Warning: Could not open " << filename << " for writing\n";
        return;
    }
    
    csv << "# Detailed trial results: McCurdy vs Chebyshev\n";
    csv << "b,q,trial,";
    csv << "mccurdy_result,chebyshev_result,reference_result,";
    csv << "mccurdy_error,chebyshev_error,";
    csv << "mccurdy_valid,chebyshev_valid,";
    csv << "mccurdy_time_us,chebyshev_time_us,";
    csv << "chebyshev_n_terms\n";
    
    for (const auto& r : results) {
        csv << std::scientific << std::setprecision(16);
        csv << r.b << "," << r.q << "," << r.trial << ",";
        csv << r.mccurdy_result << "," << r.chebyshev_result << "," << r.reference_result << ",";
        csv << r.mccurdy_error << "," << r.chebyshev_error << ",";
        csv << r.mccurdy_valid << "," << r.chebyshev_valid << ",";
        csv << r.mccurdy_time_us << "," << r.chebyshev_time_us << ",";
        csv << r.chebyshev_n_terms << "\n";
    }
    
    csv.close();
}

/**
 * @brief Write summary CSV output
 */
inline void write_summary_csv(const std::vector<ComboStats>& stats, const std::string& filename) {
    std::ofstream csv(filename);
    if (!csv.is_open()) {
        std::cerr << "Warning: Could not open " << filename << " for writing\n";
        return;
    }
    
    csv << "# Summary statistics per (b, q) combination\n";
    csv << "b,q,n_trials,";
    csv << "mccurdy_avg_error,mccurdy_max_error,mccurdy_avg_digits,";
    csv << "chebyshev_avg_error,chebyshev_max_error,chebyshev_avg_digits,";
    csv << "mccurdy_avg_time_us,chebyshev_avg_time_us,speedup_factor,";
    csv << "chebyshev_avg_n_terms,";
    csv << "mccurdy_failures,chebyshev_failures\n";
    
    for (const auto& s : stats) {
        csv << std::scientific << std::setprecision(6);
        csv << s.b << "," << s.q << "," << s.n_trials << ",";
        csv << s.mccurdy_avg_error << "," << s.mccurdy_max_error << "," << s.mccurdy_avg_digits << ",";
        csv << s.chebyshev_avg_error << "," << s.chebyshev_max_error << "," << s.chebyshev_avg_digits << ",";
        csv << s.mccurdy_avg_time_us << "," << s.chebyshev_avg_time_us << "," << s.speedup_factor << ",";
        csv << s.chebyshev_avg_n_terms << ",";
        csv << s.mccurdy_failures << "," << s.chebyshev_failures << "\n";
    }
    
    csv.close();
}

/**
 * @brief Print summary to console
 */
inline void print_summary(const std::vector<ComboStats>& stats, std::ostream& os = std::cout) {
    os << "\n\n";
    os << "╔════════════════════════════════════════════════════════════════════════════╗\n";
    os << "║  McCurdy vs Chebyshev: Summary Statistics                                 ║\n";
    os << "╚════════════════════════════════════════════════════════════════════════════╝\n\n";
    
    // Overall statistics
    int total_trials = 0;
    double total_mccurdy_time = 0, total_chebyshev_time = 0;
    int total_mccurdy_failures = 0, total_chebyshev_failures = 0;
    
    for (const auto& s : stats) {
        total_trials += s.n_trials;
        total_mccurdy_time += s.mccurdy_avg_time_us * s.n_trials;
        total_chebyshev_time += s.chebyshev_avg_time_us * s.n_trials;
        total_mccurdy_failures += s.mccurdy_failures;
        total_chebyshev_failures += s.chebyshev_failures;
    }
    
    os << "Total Trials: " << total_trials << "\n";
    
    // Count unique b and q values
    std::set<double> unique_b;
    std::set<int> unique_q;
    for (const auto& s : stats) {
        unique_b.insert(s.b);
        unique_q.insert(s.q);
    }
    
    os << "Total b values: " << unique_b.size() << "\n";
    os << "Total q values: " << unique_q.size() << "\n\n";
    
    os << "Success Rates:\n";
    os << "  McCurdy:   " << (total_trials - total_mccurdy_failures) << " / " << total_trials
       << " (" << std::fixed << std::setprecision(2)
       << (100.0 * (total_trials - total_mccurdy_failures) / total_trials) << "%)\n";
    os << "  Chebyshev: " << (total_trials - total_chebyshev_failures) << " / " << total_trials
       << " (" << std::fixed << std::setprecision(2)
       << (100.0 * (total_trials - total_chebyshev_failures) / total_trials) << "%)\n\n";
    
    double avg_speedup = total_mccurdy_time / total_chebyshev_time;
    os << "Performance:\n";
    os << "  Average speedup: " << std::fixed << std::setprecision(2) << avg_speedup << "×\n";
    os << "  Total McCurdy time:   " << std::scientific << std::setprecision(3)
       << total_mccurdy_time / 1e6 << " s\n";
    os << "  Total Chebyshev time: " << std::scientific << std::setprecision(3)
       << total_chebyshev_time / 1e6 << " s\n\n";
    
    // Sample results
    os << "Sample Results (first 10 combinations):\n";
    os << std::setw(8) << "b" << std::setw(6) << "q"
       << std::setw(12) << "McCurdy" << std::setw(12) << "Chebyshev"
       << std::setw(10) << "Speedup" << "\n";
    os << std::string(58, '-') << "\n";
    
    for (size_t i = 0; i < std::min(size_t(10), stats.size()); ++i) {
        const auto& s = stats[i];
        os << std::fixed << std::setprecision(2) << std::setw(8) << s.b
           << std::setw(6) << s.q
           << std::setw(12) << s.mccurdy_avg_digits
           << std::setw(12) << s.chebyshev_avg_digits
           << std::setw(10) << s.speedup_factor << "×\n";
    }
    
    os << "\n";
}

/**
 * @brief Main entry point - run all tests
 *
 * @param output_prefix Prefix for output files (default: "comparison")
 * @param config Test configuration (default: TestConfig::create_default())
 * @param os Output stream for progress (default: std::cout)
 */
inline void run_all_tests(const std::string& output_prefix = "comparison",
                          const TestConfig& config = TestConfig::create_default(),
                          std::ostream& os = std::cout) {
    os << "\n";
    os << "╔════════════════════════════════════════════════════════════════════════════╗\n";
    os << "║  McCurdy vs Chebyshev: Comprehensive Comparison                           ║\n";
    os << "╚════════════════════════════════════════════════════════════════════════════╝\n";
    os << "\n";
    
    os << "Configuration:\n";
    os << "  b values: " << config.b_values.size() << " (from "
       << config.b_values.front() << " to " << config.b_values.back() << ")\n";
    os << "  q values: " << config.q_values.size() << " (from "
       << config.q_values.front() << " to " << config.q_values.back() << ")\n";
    os << "  Trials per (b,q): " << config.n_trials << "\n";
    os << "  Total combinations: " << config.b_values.size() * config.q_values.size() << "\n";
    os << "  Expected total trials: "
       << config.b_values.size() * config.q_values.size() * config.n_trials << "\n\n";
    
    os << "Running tests...\n";
    
    std::mt19937 rng(config.random_seed);
    std::vector<TrialResult> all_results;
    std::vector<ComboStats> all_stats;
    
    int total_combos = config.b_values.size() * config.q_values.size();
    int combo_count = 0;
    
    for (double b : config.b_values) {
        for (int q : config.q_values) {
            std::vector<TrialResult> combo_results;
            
            for (int trial = 0; trial < config.n_trials; ++trial) {
                TrialResult result = run_single_trial(b, q, trial, rng);
                combo_results.push_back(result);
                all_results.push_back(result);
            }
            
            ComboStats stats = compute_combo_stats(combo_results);
            all_stats.push_back(stats);
            
            combo_count++;
            print_progress(combo_count, total_combos, "Progress");
        }
    }
    
    os << "\n\nTests completed!\n\n";
    
    // Write outputs
    std::string detailed_csv = output_prefix + "_detailed.csv";
    std::string summary_csv = output_prefix + "_summary.csv";
    
    write_detailed_csv(all_results, detailed_csv);
    write_summary_csv(all_stats, summary_csv);
    
    os << "Output files:\n";
    os << "  Detailed results: " << detailed_csv << "\n";
    os << "  Summary stats:    " << summary_csv << "\n\n";
    
    print_summary(all_stats, os);
    
    os << "╔════════════════════════════════════════════════════════════════════════════╗\n";
    os << "║  Test Complete                                                             ║\n";
    os << "╚════════════════════════════════════════════════════════════════════════════╝\n";
    os << "\n";
}

/**
 * @brief Simple test for specific b and q values
 *
 * Generates 100 random input sets of size q+1 in [-b, b] and prints results
 *
 * @param b Interval half-width (testing [-b, b])
 * @param q Divided difference order (q+1 nodes)
 * @param n_trials Number of random trials (default: 100)
 * @param random_seed Random seed (default: 12345)
 * @param os Output stream (default: std::cout)
 */
inline void test_single_case(double b, int q, int n_trials = 100,
                             unsigned int random_seed = 12345,
                             std::ostream& os = std::cout) {
    os << "\n";
    os << "╔════════════════════════════════════════════════════════════════════════════╗\n";
    os << "║  McCurdy vs Chebyshev: Single Case Test                                   ║\n";
    os << "╚════════════════════════════════════════════════════════════════════════════╝\n";
    os << "\n";
    
    os << "Configuration:\n";
    os << "  Interval: [-" << b << ", " << b << "]\n";
    os << "  Divided difference order: q = " << q << " (" << (q+1) << " nodes)\n";
    os << "  Number of trials: " << n_trials << "\n";
    os << "  Random seed: " << random_seed << "\n\n";
    
    std::mt19937 rng(random_seed);
    std::vector<TrialResult> results;
    
    os << "Running trials...\n";
    for (int trial = 0; trial < n_trials; ++trial) {
        TrialResult result = run_single_trial(b, q, trial, rng);
        results.push_back(result);
        
        if ((trial + 1) % 10 == 0 || trial == 0) {
            print_progress(trial + 1, n_trials, "Progress");
        }
    }
    os << "\n\n";
    
    // Print results table
    os << "Results:\n";
    os << std::setw(8) << "Trial"
       << std::setw(18) << "McCurdy"
       << std::setw(18) << "Chebyshev"
       << std::setw(15) << "Rel Error M"
       << std::setw(15) << "Rel Error C"
       << std::setw(10) << "Time M"
       << std::setw(10) << "Time C"
       << "\n";
    os << std::string(96, '-') << "\n";
    
    for (size_t i = 0; i < std::min(size_t(20), results.size()); ++i) {
        const auto& r = results[i];
        os << std::setw(8) << i;
        
        if (r.mccurdy_valid) {
            os << std::scientific << std::setprecision(6)
               << std::setw(18) << r.mccurdy_result;
        } else {
            os << std::setw(18) << "FAILED";
        }
        
        if (r.chebyshev_valid) {
            os << std::scientific << std::setprecision(6)
               << std::setw(18) << r.chebyshev_result;
        } else {
            os << std::setw(18) << "FAILED";
        }
        
        if (r.mccurdy_valid) {
            os << std::scientific << std::setprecision(3)
               << std::setw(15) << r.mccurdy_error;
        } else {
            os << std::setw(15) << "N/A";
        }
        
        if (r.chebyshev_valid) {
            os << std::scientific << std::setprecision(3)
               << std::setw(15) << r.chebyshev_error;
        } else {
            os << std::setw(15) << "N/A";
        }
        
        os << std::fixed << std::setprecision(2)
           << std::setw(10) << r.mccurdy_time_us
           << std::setw(10) << r.chebyshev_time_us
           << "\n";
    }
    
    if (results.size() > 20) {
        os << "... (" << (results.size() - 20) << " more trials)\n";
    }
    os << "\n";
    
    // Compute and print statistics
    ComboStats stats = compute_combo_stats(results);
    
    os << "Summary Statistics:\n";
    os << std::string(60, '-') << "\n";
    
    os << "Success rates:\n";
    os << "  McCurdy:   " << (n_trials - stats.mccurdy_failures) << " / " << n_trials
       << " (" << std::fixed << std::setprecision(1)
       << (100.0 * (n_trials - stats.mccurdy_failures) / n_trials) << "%)\n";
    os << "  Chebyshev: " << (n_trials - stats.chebyshev_failures) << " / " << n_trials
       << " (" << std::fixed << std::setprecision(1)
       << (100.0 * (n_trials - stats.chebyshev_failures) / n_trials) << "%)\n\n";
    
    os << "Accuracy (accurate decimal digits):\n";
    os << "  McCurdy:   avg = " << std::fixed << std::setprecision(2) << stats.mccurdy_avg_digits
       << ", max_error = " << std::scientific << std::setprecision(3) << stats.mccurdy_max_error << "\n";
    os << "  Chebyshev: avg = " << std::fixed << std::setprecision(2) << stats.chebyshev_avg_digits
       << ", max_error = " << std::scientific << std::setprecision(3) << stats.chebyshev_max_error << "\n\n";
    
    os << "Performance (microseconds):\n";
    os << "  McCurdy:   avg = " << std::fixed << std::setprecision(2) << stats.mccurdy_avg_time_us << " μs\n";
    os << "  Chebyshev: avg = " << std::fixed << std::setprecision(2) << stats.chebyshev_avg_time_us << " μs\n";
    os << "  Speedup:   " << std::fixed << std::setprecision(2) << stats.speedup_factor << "×";
    if (stats.speedup_factor > 1.0) {
        os << " (McCurdy is faster)";
    } else {
        os << " (Chebyshev is faster)";
    }
    os << "\n\n";
    
    os << "Chebyshev details:\n";
    os << "  Avg Chebyshev terms: " << std::fixed << std::setprecision(1)
       << stats.chebyshev_avg_n_terms << "\n\n";
    
    os << "╔════════════════════════════════════════════════════════════════════════════╗\n";
    os << "║  Test Complete                                                             ║\n";
    os << "╚════════════════════════════════════════════════════════════════════════════╝\n";
    os << "\n";
}

} // namespace test_comparison

#endif // TEST_COMPARISON_HPP
