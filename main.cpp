#include "test_chebyshev_bessel_Rn.hpp"
#include "test_chebyshev_bessel_I0.hpp"
#include "test_comparison.hpp"
#include "test_comparison_ratios.hpp"
#include "chebyshev_convergence_stats.hpp"
#include "comparison_test.hpp"
#include <iostream>

int main() {
    
   //return test_bessel_ratios::run_all_tests();
    
    //test_chebyshev_bessel_I0::run_all_tests("/mnt/user-data/outputs/I0_test_results.csv");
    
    //test_comparison::test_single_case(5., 30);
    //test_comparison_ratios::test_ratio_case(1.0, 5, 50);
    
    // Quick test (fewer combinations for fast testing)
    //test_comparison::run_all_tests("quick_comparison", test_comparison::TestConfig::create_quick());
    
    // Default test (balanced coverage)
    //test_comparison::run_all_tests("comparison", test_comparison::TestConfig::create_default());
    
    // Comprehensive test (extensive coverage - takes longer)
    // test_comparison::run_all_tests("comprehensive_comparison", test_comparison::TestConfig::create_comprehensive());
   /*
    std::vector<double> b_values = {0.5, 1.0, 5.0, 10.0, 20.0};
       std::vector<int> q_values = {5, 10, 15, 20, 25, 30};
       
       chebyshev_stats::exportQuartilesToFile(
           "quartiles.csv",
           100,           // 100 trials per combination
           b_values,
           q_values
       );
    */
    
    comparison_test::ComparisonConfig custom_config;
        custom_config.a = -2.0;
        custom_config.b_values = {2.0, 4.0, 8.0};
        custom_config.q_values = {3, 7, 15};
        custom_config.num_trials = 10;
        custom_config.random_seed = 12345;
        custom_config.verbose = false;
        
        std::cout << "\n\nRunning custom configuration..." << std::endl;
        run_comparison_tests(custom_config);
    
       return 0;
}
