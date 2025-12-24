#ifndef COMPARISON_TEST_HPP
#define COMPARISON_TEST_HPP

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <algorithm>

#include "chebyshev_exp_divdiff.hpp"
#include "incremental_chebyshev_exp_divdiff.hpp"

namespace comparison_test {

/**
 * @brief Configuration for comparison tests
 */
struct ComparisonConfig {
    double a = -1.0;                    // Lower bound of interval
    std::vector<double> b_values;       // Upper bounds to test
    std::vector<int> q_values;          // Orders to test
    int num_trials = 5;                 // Number of random trials per (b,q) pair
    unsigned int random_seed = 42;      // Seed for reproducibility
    bool verbose = false;               // Print detailed output
    
    ComparisonConfig() {
        // Default test ranges
        b_values = {1.0, 2.0, 5.0, 10.0};
        q_values = {2, 5, 10, 20};
    }
};

/**
 * @brief Results for a single test case
 */
struct TestResult {
    double b;
    int q;
    int trial;
    
    double batch_value;
    double incr_value;
    double batch_normalized;
    double incr_normalized;
    
    int batch_iterations;
    int incr_cached_layers;
    
    double relative_error;
    double normalized_relative_error;
    
    std::vector<double> nodes;
};

/**
 * @brief Print table header
 */
inline void print_header() {
    std::cout << std::string(140, '=') << std::endl;
    std::cout << std::setw(4) << "b"
              << std::setw(5) << "q"
              << std::setw(7) << "Trial"
              << std::setw(24) << "Batch Value"
              << std::setw(24) << "Incremental Value"
              << std::setw(15) << "Rel Error"
              << std::setw(12) << "Batch n"
              << std::setw(12) << "Incr N"
              << std::setw(12) << "Status"
              << std::endl;
    std::cout << std::string(140, '-') << std::endl;
}

/**
 * @brief Print a single result row
 */
inline void print_result(const TestResult& result) {
    std::cout << std::fixed << std::setprecision(1);
    std::cout << std::setw(4) << result.b
              << std::setw(5) << result.q
              << std::setw(7) << result.trial;
    
    std::cout << std::scientific << std::setprecision(15);
    std::cout << std::setw(24) << result.batch_value
              << std::setw(24) << result.incr_value;
    
    std::cout << std::scientific << std::setprecision(3);
    std::cout << std::setw(15) << result.relative_error;
    
    std::cout << std::fixed << std::setprecision(0);
    std::cout << std::setw(12) << result.batch_iterations
              << std::setw(12) << result.incr_cached_layers;
    
    // Status indicator
    std::string status;
    if (result.relative_error < 1e-14) {
        status = "✓ PASS";
    } else if (result.relative_error < 1e-10) {
        status = "~ GOOD";
    } else {
        status = "✗ FAIL";
    }
    std::cout << std::setw(12) << status << std::endl;
}

/**
 * @brief Print detailed information for a test case
 */
inline void print_detailed(const TestResult& result) {
    std::cout << "\n--- Detailed Results ---" << std::endl;
    std::cout << "Configuration: b=" << result.b << ", q=" << result.q 
              << ", trial=" << result.trial << std::endl;
    
    std::cout << "\nNodes: [";
    for (size_t i = 0; i < result.nodes.size(); ++i) {
        if (i > 0) std::cout << ", ";
        if (i >= 5 && result.nodes.size() > 10) {
            std::cout << "..., " << result.nodes.back();
            break;
        }
        std::cout << std::setprecision(4) << std::fixed << result.nodes[i];
    }
    std::cout << "]" << std::endl;
    
    std::cout << std::scientific << std::setprecision(16);
    std::cout << "\nFull Values:" << std::endl;
    std::cout << "  Batch:       " << result.batch_value << std::endl;
    std::cout << "  Incremental: " << result.incr_value << std::endl;
    std::cout << "  Rel Error:   " << result.relative_error << std::endl;
    
    std::cout << "\nNormalized Values:" << std::endl;
    std::cout << "  Batch:       " << result.batch_normalized << std::endl;
    std::cout << "  Incremental: " << result.incr_normalized << std::endl;
    std::cout << "  Rel Error:   " << result.normalized_relative_error << std::endl;
    
    std::cout << "\nConvergence:" << std::endl;
    std::cout << "  Batch iterations:    " << result.batch_iterations << std::endl;
    std::cout << "  Incremental layers:  " << result.incr_cached_layers << std::endl;
    std::cout << std::endl;
}

/**
 * @brief Generate random nodes in interval [a, b]
 */
inline std::vector<double> generate_random_nodes(double a, double b, int q, 
                                                   std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(a, b);
    std::vector<double> nodes(q + 1);
    for (int i = 0; i <= q; ++i) {
        nodes[i] = dist(rng);
    }
    // Sort for better numerical behavior
    std::sort(nodes.begin(), nodes.end());
    return nodes;
}

/**
 * @brief Run a single comparison test
 */
inline TestResult run_single_test(double a, double b, int q, int trial,
                                   std::mt19937& rng, bool verbose = false) {
    TestResult result;
    result.b = b;
    result.q = q;
    result.trial = trial;
    
    // Generate random nodes
    result.nodes = generate_random_nodes(a, b, q, rng);
    
    // Batch evaluation
    chebyshev::ExpDivDiff<double> batch_eval(a, b);
    result.batch_value = batch_eval.evaluate(result.nodes, &result.batch_iterations);
    result.batch_normalized = batch_eval.evaluate_normalized(result.nodes);
    
    // Incremental evaluation
    chebyshev_incremental::IncrementalExpDivDiff<double> incr_eval(a, b);
    result.incr_value = incr_eval.initialize(result.nodes);
    result.incr_normalized = incr_eval.get_normalized_value();
    result.incr_cached_layers = incr_eval.num_cached_layers();
    
    // Compute errors
    result.relative_error = std::abs(result.batch_value - result.incr_value) / 
                           std::max(std::abs(result.batch_value), 1e-100);
    result.normalized_relative_error = std::abs(result.batch_normalized - result.incr_normalized) / 
                                       std::max(std::abs(result.batch_normalized), 1e-100);
    
    if (verbose) {
        print_detailed(result);
    }
    
    return result;
}

/**
 * @brief Compute statistics over multiple trials
 */
struct Statistics {
    double mean;
    double max;
    double min;
    int count;
    
    static Statistics compute(const std::vector<double>& values) {
        Statistics stats;
        stats.count = values.size();
        if (values.empty()) {
            stats.mean = stats.max = stats.min = 0.0;
            return stats;
        }
        
        stats.min = *std::min_element(values.begin(), values.end());
        stats.max = *std::max_element(values.begin(), values.end());
        
        double sum = 0.0;
        for (double v : values) sum += v;
        stats.mean = sum / values.size();
        
        return stats;
    }
};

/**
 * @brief Print summary statistics
 */
inline void print_summary(const std::vector<TestResult>& results, 
                         const ComparisonConfig& config) {
    std::cout << "\n" << std::string(140, '=') << std::endl;
    std::cout << "SUMMARY STATISTICS" << std::endl;
    std::cout << std::string(140, '=') << std::endl;
    
    // Group by (b, q) pairs
    for (double b : config.b_values) {
        for (int q : config.q_values) {
            std::vector<double> errors;
            std::vector<int> batch_iters;
            std::vector<int> incr_layers;
            
            for (const auto& r : results) {
                if (r.b == b && r.q == q) {
                    errors.push_back(r.relative_error);
                    batch_iters.push_back(r.batch_iterations);
                    incr_layers.push_back(r.incr_cached_layers);
                }
            }
            
            if (errors.empty()) continue;
            
            auto error_stats = Statistics::compute(errors);
            std::vector<double> batch_iters_double(batch_iters.begin(), batch_iters.end());
            std::vector<double> incr_layers_double(incr_layers.begin(), incr_layers.end());
            auto batch_stats = Statistics::compute(batch_iters_double);
            auto incr_stats = Statistics::compute(incr_layers_double);
            
            std::cout << "\nb=" << std::fixed << std::setprecision(1) << b 
                      << ", q=" << q << " (" << error_stats.count << " trials):" << std::endl;
            
            std::cout << std::scientific << std::setprecision(3);
            std::cout << "  Relative Error:  min=" << error_stats.min 
                      << ", max=" << error_stats.max 
                      << ", mean=" << error_stats.mean << std::endl;
            
            std::cout << std::fixed << std::setprecision(1);
            std::cout << "  Batch iters:     min=" << batch_stats.min 
                      << ", max=" << batch_stats.max 
                      << ", mean=" << batch_stats.mean << std::endl;
            
            std::cout << "  Incr layers:     min=" << incr_stats.min 
                      << ", max=" << incr_stats.max 
                      << ", mean=" << incr_stats.mean << std::endl;
            
            // Overall assessment
            if (error_stats.max < 1e-14) {
                std::cout << "  Status: ✓ EXCELLENT (all errors < 1e-14)" << std::endl;
            } else if (error_stats.max < 1e-10) {
                std::cout << "  Status: ✓ GOOD (all errors < 1e-10)" << std::endl;
            } else {
                std::cout << "  Status: ⚠ WARNING (max error = " 
                          << std::scientific << error_stats.max << ")" << std::endl;
            }
        }
    }
}

/**
 * @brief Main comparison test function
 * 
 * This is the single function to call from main() to run all tests.
 * 
 * @param config Configuration for the tests (optional)
 * @return true if all tests passed, false otherwise
 */
inline bool run_comparison_tests(const ComparisonConfig& config = ComparisonConfig()) {
    std::cout << "\n";
    std::cout << std::string(140, '=') << std::endl;
    std::cout << "BATCH vs INCREMENTAL IMPLEMENTATION COMPARISON TEST" << std::endl;
    std::cout << std::string(140, '=') << std::endl;
    
    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Interval: [" << config.a << ", b] where b varies" << std::endl;
    std::cout << "  b values: [";
    for (size_t i = 0; i < config.b_values.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << config.b_values[i];
    }
    std::cout << "]" << std::endl;
    
    std::cout << "  q values: [";
    for (size_t i = 0; i < config.q_values.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << config.q_values[i];
    }
    std::cout << "]" << std::endl;
    
    std::cout << "  Trials per (b,q): " << config.num_trials << std::endl;
    std::cout << "  Random seed: " << config.random_seed << std::endl;
    std::cout << "  Total tests: " << config.b_values.size() * config.q_values.size() * config.num_trials << std::endl;
    std::cout << std::endl;
    
    std::mt19937 rng(config.random_seed);
    std::vector<TestResult> all_results;
    
    print_header();
    
    bool all_passed = true;
    
    // Run tests for each (b, q) combination
    for (double b : config.b_values) {
        for (int q : config.q_values) {
            for (int trial = 1; trial <= config.num_trials; ++trial) {
                TestResult result = run_single_test(config.a, b, q, trial, rng, config.verbose);
                all_results.push_back(result);
                print_result(result);
                
                if (result.relative_error >= 1e-10) {
                    all_passed = false;
                }
            }
        }
    }
    
    std::cout << std::string(140, '=') << std::endl;
    
    // Print summary statistics
    print_summary(all_results, config);
    
    // Overall result
    std::cout << "\n" << std::string(140, '=') << std::endl;
    if (all_passed) {
        std::cout << "OVERALL RESULT: ✓ ALL TESTS PASSED" << std::endl;
    } else {
        std::cout << "OVERALL RESULT: ⚠ SOME TESTS HAD LARGE ERRORS" << std::endl;
    }
    std::cout << std::string(140, '=') << std::endl;
    std::cout << std::endl;
    
    return all_passed;
}

/**
 * @brief Quick test with default parameters
 */
inline bool quick_test() {
    ComparisonConfig config;
    config.b_values = {1.0, 5.0};
    config.q_values = {2, 10};
    config.num_trials = 3;
    return run_comparison_tests(config);
}

/**
 * @brief Comprehensive test with wide parameter range
 */
inline bool comprehensive_test() {
    ComparisonConfig config;
    config.b_values = {1.0, 2.0, 5.0, 10.0, 20.0};
    config.q_values = {1, 2, 5, 10, 15, 20, 30};
    config.num_trials = 5;
    return run_comparison_tests(config);
}

/**
 * @brief Test with challenging parameters (large intervals, high orders)
 */
inline bool stress_test() {
    ComparisonConfig config;
    config.b_values = {10.0, 20.0, 50.0};
    config.q_values = {20, 30, 40, 50};
    config.num_trials = 3;
    return run_comparison_tests(config);
}

/**
 * @brief Test a specific (b, q) pair with verbose output
 */
inline bool detailed_test(double b, int q, int num_trials = 5) {
    ComparisonConfig config;
    config.b_values = {b};
    config.q_values = {q};
    config.num_trials = num_trials;
    config.verbose = true;
    return run_comparison_tests(config);
}

} // namespace comparison_test

#endif // COMPARISON_TEST_HPP
