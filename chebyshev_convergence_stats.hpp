#ifndef CHEBYSHEV_CONVERGENCE_STATS_HPP
#define CHEBYSHEV_CONVERGENCE_STATS_HPP

#include "chebyshev_exp_divdiff.hpp"
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>

/**
 * @file chebyshev_convergence_stats.hpp
 * @brief Statistical analysis of Chebyshev method convergence
 *
 * Provides utilities to analyze how many Chebyshev terms are needed
 * for convergence as a function of interval size b and order q.
 */

namespace chebyshev_stats {

/**
 * @brief Get median number of Chebyshev terms needed for convergence
 *
 * Runs n_trials with random node configurations and returns the median
 * number of terms required for the Chebyshev series to converge.
 *
 * @param b Interval half-width (interval is [-b, b])
 * @param q Divided difference order (q+1 nodes)
 * @param n_trials Number of random trials to run
 * @param random_seed Seed for random number generator (default: 12345)
 * @return Median number of Chebyshev terms used
 *
 * @example
 *   int median = chebyshev_stats::medianNterms(5.0, 10, 100);
 *   std::cout << "Median terms needed: " << median << std::endl;
 */
inline int medianNterms(double b, int q, int n_trials = 100,
                       unsigned int random_seed = 12345) {
    if (n_trials <= 0) {
        throw std::invalid_argument("n_trials must be positive");
    }
    if (b <= 0) {
        throw std::invalid_argument("b must be positive");
    }
    if (q < 0) {
        throw std::invalid_argument("q must be non-negative");
    }
    
    std::mt19937 rng(random_seed);
    std::uniform_real_distribution<double> dist(-b, b);
    
    std::vector<int> n_terms_list;
    n_terms_list.reserve(n_trials);
    
    // Initialize Chebyshev solver
    chebyshev::ExpDivDiff<double> solver(-b, b);
    
    // Run trials
    for (int trial = 0; trial < n_trials; ++trial) {
        // Generate random nodes
        std::vector<double> nodes(q + 1);
        for (int i = 0; i <= q; ++i) {
            nodes[i] = dist(rng);
        }
        
     
        
        // Evaluate and capture number of terms
        int n_terms = 0;
        try {
            solver.evaluate(nodes, &n_terms);
            n_terms_list.push_back(n_terms);
        } catch (...) {
            // If evaluation fails, skip this trial
            continue;
        }
    }
    
    if (n_terms_list.empty()) {
        throw std::runtime_error("All trials failed");
    }
    
    // Compute median
    std::sort(n_terms_list.begin(), n_terms_list.end());
    size_t n = n_terms_list.size();
    
    if (n % 2 == 0) {
        // Even number of elements: average of middle two
        return (n_terms_list[n/2 - 1] + n_terms_list[n/2]) / 2;
    } else {
        // Odd number of elements: middle element
        return n_terms_list[n/2];
    }
}

/**
 * @brief Get full statistics for Chebyshev convergence
 */
struct ConvergenceStats {
    int n_trials;
    int successful_trials;
    
    int median_terms;
    int min_terms;
    int max_terms;
    double mean_terms;
    double std_dev_terms;
    
    // Percentiles
    int p25_terms;  // 25th percentile
    int p75_terms;  // 75th percentile
    int p90_terms;  // 90th percentile
    int p95_terms;  // 95th percentile
    int p99_terms;  // 99th percentile
    
    ConvergenceStats() : n_trials(0), successful_trials(0),
                        median_terms(0), min_terms(0), max_terms(0),
                        mean_terms(0), std_dev_terms(0),
                        p25_terms(0), p75_terms(0), p90_terms(0),
                        p95_terms(0), p99_terms(0) {}
};

/**
 * @brief Compute comprehensive convergence statistics
 *
 * @param b Interval half-width
 * @param q Divided difference order
 * @param n_trials Number of trials
 * @param random_seed Random seed
 * @return ConvergenceStats structure with detailed statistics
 */
inline ConvergenceStats getConvergenceStats(double b, int q, int n_trials = 100,
                                           unsigned int random_seed = 12345) {
    ConvergenceStats stats;
    stats.n_trials = n_trials;
    
    std::mt19937 rng(random_seed);
    std::uniform_real_distribution<double> dist(-b, b);
    
    std::vector<int> n_terms_list;
    n_terms_list.reserve(n_trials);
    
    chebyshev::ExpDivDiff<double> solver(-b, b);
    
    // Run trials
    for (int trial = 0; trial < n_trials; ++trial) {
        std::vector<double> nodes(q + 1);
        for (int i = 0; i <= q; ++i) {
            nodes[i] = dist(rng);
        }
        
        
        int n_terms = 0;
        try {
            solver.evaluate(nodes, &n_terms);
            n_terms_list.push_back(n_terms);
        } catch (...) {
            continue;
        }
    }
    
    if (n_terms_list.empty()) {
        throw std::runtime_error("All trials failed");
    }
    
    stats.successful_trials = n_terms_list.size();
    
    // Sort for percentile calculations
    std::sort(n_terms_list.begin(), n_terms_list.end());
    
    size_t n = n_terms_list.size();
    
    // Min and max
    stats.min_terms = n_terms_list.front();
    stats.max_terms = n_terms_list.back();
    
    // Mean
    double sum = 0;
    for (int val : n_terms_list) {
        sum += val;
    }
    stats.mean_terms = sum / n;
    
    // Standard deviation
    double sq_sum = 0;
    for (int val : n_terms_list) {
        double diff = val - stats.mean_terms;
        sq_sum += diff * diff;
    }
    stats.std_dev_terms = std::sqrt(sq_sum / n);
    
    // Median
    if (n % 2 == 0) {
        stats.median_terms = (n_terms_list[n/2 - 1] + n_terms_list[n/2]) / 2;
    } else {
        stats.median_terms = n_terms_list[n/2];
    }
    
    // Percentiles
    auto percentile = [&](double p) -> int {
        size_t idx = static_cast<size_t>(p * (n - 1));
        return n_terms_list[idx];
    };
    
    stats.p25_terms = percentile(0.25);
    stats.p75_terms = percentile(0.75);
    stats.p90_terms = percentile(0.90);
    stats.p95_terms = percentile(0.95);
    stats.p99_terms = percentile(0.99);
    
    return stats;
}

/**
 * @brief Print convergence statistics in human-readable format
 */
inline void printConvergenceStats(const ConvergenceStats& stats,
                                  double b, int q,
                                  std::ostream& os = std::cout) {
    os << "\n";
    os << "╔════════════════════════════════════════════════════════════════════════════╗\n";
    os << "║  Chebyshev Convergence Statistics                                         ║\n";
    os << "╚════════════════════════════════════════════════════════════════════════════╝\n";
    os << "\n";
    
    os << "Configuration:\n";
    os << "  Interval: [-" << b << ", " << b << "]\n";
    os << "  Order: q = " << q << " (" << (q+1) << " nodes)\n";
    os << "  Trials: " << stats.n_trials << " (successful: " << stats.successful_trials << ")\n";
    os << "\n";
    
    os << "Convergence (number of Chebyshev terms):\n";
    os << "  Mean:   " << std::fixed << std::setprecision(2) << stats.mean_terms << "\n";
    os << "  Median: " << stats.median_terms << "\n";
    os << "  Std Dev: " << std::fixed << std::setprecision(2) << stats.std_dev_terms << "\n";
    os << "  Min:    " << stats.min_terms << "\n";
    os << "  Max:    " << stats.max_terms << "\n";
    os << "\n";
    
    os << "Percentiles:\n";
    os << "  25th: " << stats.p25_terms << "\n";
    os << "  50th: " << stats.median_terms << " (median)\n";
    os << "  75th: " << stats.p75_terms << "\n";
    os << "  90th: " << stats.p90_terms << "\n";
    os << "  95th: " << stats.p95_terms << "\n";
    os << "  99th: " << stats.p99_terms << "\n";
    os << "\n";
    
    os << "Interpretation:\n";
    os << "  - 50% of cases converge in ≤ " << stats.median_terms << " terms\n";
    os << "  - 90% of cases converge in ≤ " << stats.p90_terms << " terms\n";
    os << "  - Worst case needed " << stats.max_terms << " terms\n";
    os << "\n";
}

/**
 * @brief Analyze convergence scaling with b
 *
 * Tests how median convergence changes as interval size increases.
 */
inline void analyzeScalingWithB(int q, int n_trials = 100,
                                std::ostream& os = std::cout) {
    os << "\n";
    os << "╔════════════════════════════════════════════════════════════════════════════╗\n";
    os << "║  Convergence Scaling with Interval Size (b)                               ║\n";
    os << "╚════════════════════════════════════════════════════════════════════════════╝\n";
    os << "\n";
    
    os << "Configuration: q = " << q << ", n_trials = " << n_trials << "\n\n";
    
    std::vector<double> b_values = {0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0};
    
    os << std::setw(10) << "b"
       << std::setw(12) << "Median"
       << std::setw(12) << "Mean"
       << std::setw(12) << "P90"
       << std::setw(12) << "Max"
       << "\n";
    os << std::string(58, '-') << "\n";
    
    for (double b : b_values) {
        try {
            auto stats = getConvergenceStats(b, q, n_trials);
            
            os << std::setw(10) << b
               << std::setw(12) << stats.median_terms
               << std::setw(12) << std::fixed << std::setprecision(1) << stats.mean_terms
               << std::setw(12) << stats.p90_terms
               << std::setw(12) << stats.max_terms
               << "\n";
        } catch (...) {
            os << std::setw(10) << b << "  FAILED\n";
        }
    }
    os << "\n";
}

/**
 * @brief Analyze convergence scaling with q
 *
 * Tests how median convergence changes as order increases.
 */
inline void analyzeScalingWithQ(double b, int n_trials = 100,
                                std::ostream& os = std::cout) {
    os << "\n";
    os << "╔════════════════════════════════════════════════════════════════════════════╗\n";
    os << "║  Convergence Scaling with Order (q)                                       ║\n";
    os << "╚════════════════════════════════════════════════════════════════════════════╝\n";
    os << "\n";
    
    os << "Configuration: b = " << b << ", n_trials = " << n_trials << "\n\n";
    
    std::vector<int> q_values = {1, 2, 5, 10, 15, 20, 25, 30};
    
    os << std::setw(10) << "q"
       << std::setw(12) << "Median"
       << std::setw(12) << "Mean"
       << std::setw(12) << "P90"
       << std::setw(12) << "Max"
       << "\n";
    os << std::string(58, '-') << "\n";
    
    for (int q : q_values) {
        try {
            auto stats = getConvergenceStats(b, q, n_trials);
            
            os << std::setw(10) << q
               << std::setw(12) << stats.median_terms
               << std::setw(12) << std::fixed << std::setprecision(1) << stats.mean_terms
               << std::setw(12) << stats.p90_terms
               << std::setw(12) << stats.max_terms
               << "\n";
        } catch (...) {
            os << std::setw(10) << q << "  FAILED\n";
        }
    }
    os << "\n";
}

/**
 * @brief Create a convergence heatmap (text-based)
 *
 * Shows median convergence as a function of both b and q.
 */
inline void printConvergenceHeatmap(int n_trials = 50,
                                    std::ostream& os = std::cout) {
    os << "\n";
    os << "╔════════════════════════════════════════════════════════════════════════════╗\n";
    os << "║  Convergence Heatmap: Median Terms by (b, q)                              ║\n";
    os << "╚════════════════════════════════════════════════════════════════════════════╝\n";
    os << "\n";
    
    std::vector<double> b_values = {0.5, 1.0, 2.0, 5.0, 10.0, 20.0};
    std::vector<int> q_values = {1, 5, 10, 15, 20, 25, 30};
    
    os << "Rows = b (interval half-width), Columns = q (order)\n";
    os << "Values = Median number of Chebyshev terms\n\n";
    
    // Header
    os << std::setw(8) << "b \\ q";
    for (int q : q_values) {
        os << std::setw(6) << q;
    }
    os << "\n";
    os << std::string(8 + 6*q_values.size(), '-') << "\n";
    
    // Data rows
    for (double b : b_values) {
        os << std::setw(8) << b;
        
        for (int q : q_values) {
            try {
                int median = medianNterms(b, q, n_trials);
                os << std::setw(6) << median;
            } catch (...) {
                os << std::setw(6) << "X";
            }
        }
        os << "\n";
    }
    os << "\n";
    
    os << "Legend: X = Failed to converge or error\n";
    os << "\n";
}

/**
 * @brief Export quartile statistics to file for multiple (b, q) combinations
 *
 * For each (b, q) pair, computes convergence statistics over N_trials and writes
 * the first quartile (Q1/25th percentile), median (Q2/50th percentile), and
 * third quartile (Q3/75th percentile) to a file.
 *
 * @param filename Output filename
 * @param n_trials Number of random trials per (b, q) combination
 * @param b_values Vector of interval half-widths to test
 * @param q_values Vector of orders to test
 * @param random_seed Random seed for reproducibility (default: 12345)
 *
 * Output format (CSV):
 *   b,q,Q1,Q2,Q3
 *   0.5,5,16,17,18
 *   1.0,10,20,21,22
 *   ...
 *
 * @example
 *   std::vector<double> b_vals = {0.5, 1.0, 5.0, 10.0};
 *   std::vector<int> q_vals = {5, 10, 15, 20};
 *   chebyshev_stats::exportQuartilesToFile("quartiles.csv", 100, b_vals, q_vals);
 */
inline void exportQuartilesToFile(const std::string& filename,
                                   int n_trials,
                                   const std::vector<double>& b_values,
                                   const std::vector<int>& q_values,
                                   unsigned int random_seed = 12345) {
    if (n_trials <= 0) {
        throw std::invalid_argument("n_trials must be positive");
    }
    if (b_values.empty()) {
        throw std::invalid_argument("b_values cannot be empty");
    }
    if (q_values.empty()) {
        throw std::invalid_argument("q_values cannot be empty");
    }
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    
    // Write header
    file << "b,q,Q1,Q2,Q3\n";
    
    std::mt19937 rng(random_seed);
    
    // Progress tracking
    size_t total_combinations = b_values.size() * q_values.size();
    size_t current = 0;
    
    std::cout << "Exporting quartile data to " << filename << "\n";
    std::cout << "Total combinations: " << total_combinations << "\n";
    std::cout << "Trials per combination: " << n_trials << "\n\n";
    
    // Loop over all (b, q) combinations
    for (double b : b_values) {
        for (int q : q_values) {
            current++;
            
            // Progress indicator
            if (current % 10 == 0 || current == total_combinations) {
                std::cout << "\rProgress: " << current << "/" << total_combinations
                         << " (" << (100 * current / total_combinations) << "%)"
                         << std::flush;
            }
            
            try {
                // Collect convergence data
                std::vector<int> n_terms_list;
                n_terms_list.reserve(n_trials);
                
                chebyshev::ExpDivDiff<double> solver(-b, b);
                std::uniform_real_distribution<double> dist(-b, b);
                
                for (int trial = 0; trial < n_trials; ++trial) {
                    // Generate random nodes
                    std::vector<double> nodes(q + 1);
                    for (int i = 0; i <= q; ++i) {
                        nodes[i] = dist(rng);
                    }
                    
                    // Evaluate and capture number of terms
                    int n_terms = 0;
                    try {
                        solver.evaluate(nodes, &n_terms);
                        n_terms_list.push_back(n_terms);
                    } catch (...) {
                        // Skip failed trials
                        continue;
                    }
                }
                
                if (n_terms_list.empty()) {
                    // All trials failed - write NaN or skip
                    file << b << "," << q << ",NaN,NaN,NaN\n";
                    continue;
                }
                
                // Sort to compute quartiles
                std::sort(n_terms_list.begin(), n_terms_list.end());
                size_t n = n_terms_list.size();
                
                // Compute quartiles
                // Q1 = 25th percentile (index n/4)
                // Q2 = 50th percentile (median, index n/2)
                // Q3 = 75th percentile (index 3n/4)
                
                auto quartile = [&](double p) -> int {
                    size_t idx = static_cast<size_t>(p * (n - 1));
                    return n_terms_list[idx];
                };
                
                int Q1 = quartile(0.25);
                int Q2 = quartile(0.50);  // median
                int Q3 = quartile(0.75);
                
                // Write to file
                file << b << "," << q << "," << Q1 << "," << Q2 << "," << Q3 << "\n";
                
            } catch (const std::exception& e) {
                // Handle errors gracefully
                file << b << "," << q << ",ERROR,ERROR,ERROR\n";
                std::cerr << "\nError at b=" << b << ", q=" << q << ": "
                         << e.what() << "\n";
            }
        }
    }
    
    std::cout << "\n\nDone! Results written to " << filename << "\n";
    file.close();
}

} // namespace chebyshev_stats

#endif // CHEBYSHEV_CONVERGENCE_STATS_HPP
