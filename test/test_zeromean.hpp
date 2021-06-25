#pragma once

#include "test.hpp"
#include "kernels/zeroMean.hpp"

class Test_ZeroMean : public Test
{
private:
    /* data */
    int perf_test_N = N;
    double X[N * out_dim] __attribute__ ((aligned (32)));
    double baseX[N * out_dim] __attribute__ ((aligned (32)));
    typedef void(*comp_func)(double*, int, int);
    comp_func func;

public:
    Test_ZeroMean(comp_func fn) : func(fn) {}
    ~Test_ZeroMean() = default;


    virtual void init_perf() {    
        rands(X, N, out_dim);
    }

    virtual void init_validate() {
        rands(X, N, out_dim);
        memcpy(baseX, X, N * out_dim * sizeof(double));
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
                (*func)(X, perf_test_N, out_dim);      
            }
            end = stop_tsc(start);

            cycles = (double)end;
            multiplier = (CYCLES_REQUIRED) / (cycles);
            
        } while (multiplier > 2);

        double total_cycles = 0;
        for (size_t j = 0; j < REP; j++) {
            start = start_tsc();
            for (size_t i = 0; i < num_runs; ++i) {
                (*func)(X, perf_test_N, out_dim);      
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

        (*func)(X, N, out_dim);
        zeroMeanv1::zeroMean(baseX, N, out_dim);
        error = nrm_sqr_diff(X, baseX, N * out_dim);
        
        print_error(error);
    }

    virtual void print_perf(double cycles, long num_runs) {
        cout << "\nPerformance Test Report\n";
        cout << "Number of samples = \t" << perf_test_N << endl;
        cout << "Repeat " << num_runs << " runs for " << REP << " times.\n";
        cout << "Avg cycles = " << cycles << endl; 
        cout << endl; 
    }

    virtual double perf_test_2(const int N_) {
        init_perf();

        double cycles = 0.;
        long num_runs = 100;
        double multiplier = 1.0;
        unsigned long long start, end;

        perf_test_N = N_;      

        //warmup
        do {
            num_runs = num_runs * multiplier;
            start = start_tsc();
            for (size_t i = 0; i < num_runs; i++) {
                (*func)(X, N_, out_dim);      
            }
            end = stop_tsc(start);

            cycles = (double)end;
            multiplier = (CYCLES_REQUIRED) / (cycles);
            
        } while (multiplier > 2);
        


        double total_cycles = 0;
        for (size_t j = 0; j < REP; j++) {
            start = start_tsc();
            for (size_t i = 0; i < num_runs; ++i) {
                (*func)(X, N_, out_dim);      
            }
            end = stop_tsc(start);
            
            cycles = ((double)end) / num_runs;
            total_cycles += cycles;
        }
        cycles = total_cycles / (double)REP;
        return cycles;
    }
};

