#pragma once

#include "test.hpp"
#include "kernels/computeSED.hpp"

class Test_ComputeSED : public Test
{
private:
    double X[N * in_dim] __attribute__ ((aligned (32)));
    typedef void(*comp_func)(const double*, int, int, double*);
    comp_func func;
    int perf_test_N = 128;
    double DD[N * N] __attribute__ ((aligned (32)));
    double baseDD[N * N] __attribute__ ((aligned (32)));

public:
    Test_ComputeSED(comp_func fn) : func(fn) {}
    ~Test_ComputeSED() = default;
    
    virtual void init_perf() {    
        rands(X, N, in_dim);
    }

    virtual void init_validate() {
        rands(X, N, in_dim);
    }


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
                (*func)(X, perf_test_N, in_dim, DD);      
            }
            end = stop_tsc(start);

            cycles = (double)end;
            multiplier = (CYCLES_REQUIRED) / (cycles);
            
        } while (multiplier > 2);

        double total_cycles = 0;
        for (size_t j = 0; j < REP; j++) {
            start = start_tsc();
            for (size_t i = 0; i < num_runs; ++i) {
                (*func)(X, perf_test_N, in_dim, DD);      
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

        (*func)(X, N, in_dim, DD);
        computeSEDv1::computeSquaredEuclideanDistance(X, N, in_dim, baseDD);
        error = nrm_sqr_diff(DD, baseDD, N * N);
        
        print_error(error);
    }

    virtual void print_perf(double cycles, long num_runs) {
        cout << "\nPerformance Test Report\n";
        cout << "Number of samples = \t" << perf_test_N << endl;
        cout << "Repeat " << num_runs << " runs for " << REP << " times.\n";
        cout << "Avg cycles = " << cycles << endl; 
        cout << endl; 
    }
};
