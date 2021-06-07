#pragma once

#include "test.hpp"
#include "kernels/computeSED.hpp"

class Test_ComputeSED : public Test
{
private:
    static const int perf_test_N = 64;
    static const int perf_in_dim = 2;
    double X[perf_test_N * perf_in_dim] __attribute__ ((aligned (32)));
    typedef void(*comp_func)(const double*, int, int, double*);
    comp_func func;
    double DD[perf_test_N * perf_test_N] __attribute__ ((aligned (32)));
    double baseDD[perf_test_N * perf_test_N] __attribute__ ((aligned (32)));

public:
    Test_ComputeSED(comp_func fn) : func(fn) {}
    ~Test_ComputeSED() = default;
    
    virtual void init_perf() {    
        rands(X, perf_test_N, perf_in_dim);
    }

    virtual void init_validate() {
        rands(X, perf_test_N, perf_in_dim);
    }

    virtual double perf_test_2(const int N_){return 0.0;};
    virtual void perf_test() {
        init_perf();
        double cycles = 0.;
        long num_runs = 100;
        double multiplier = 1;
        unsigned long long start, end;

        do {
            num_runs = num_runs * multiplier;
            start = start_tsc();
            for (size_t i = 0; i < num_runs; i++) {
                (*func)(X, perf_test_N, perf_in_dim, DD);      
            }
            end = stop_tsc(start);

            cycles = (double)end;
            multiplier = (CYCLES_REQUIRED) / (cycles);
            
        } while (multiplier > 2);

        double total_cycles = 0;
        for (size_t j = 0; j < REP; j++) {
            start = start_tsc();
            for (size_t i = 0; i < num_runs; ++i) {
                (*func)(X, perf_test_N, perf_in_dim, DD);      
            }
            end = stop_tsc(start);

            cycles = ((double)end) / num_runs;
            total_cycles += cycles;
        }
        cycles = total_cycles / REP;
        print_perf(cycles, num_runs);
    }
    
    virtual void validate() 
    {
        init_validate();
        double error = .0;

        (*func)(X, perf_test_N, perf_in_dim, DD);
        computeSEDv1::computeSquaredEuclideanDistance(X, perf_test_N, perf_in_dim, baseDD);
        error = nrm_sqr_diff(DD, baseDD, perf_test_N * perf_test_N);
        
        print_error(error);
    }

    virtual void print_perf(double cycles, long num_runs) {
        cout << "\nPerformance Test Report\n";
        cout << "Number of samples = \t" << perf_test_N << endl;
        cout << "Number of dimension = \t" << perf_in_dim << endl;
        cout << "Repeat " << num_runs << " runs for " << REP << " times.\n";
        cout << "Avg cycles = " << cycles << endl; 
        cout << endl; 
    }
};
