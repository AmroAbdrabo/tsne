#pragma once

#include "test.hpp"
#include "kernels/computeGP.hpp"

class Test_ComputeGP : public Test
{
private:
    /* data */
    double X[N * in_dim] __attribute__ ((aligned (32)));
    double P[N * N]      __attribute__ ((aligned (32)));
    double baseP[N * N]  __attribute__ ((aligned (32)));
    int perf_test_N = 64;
    int perf_test_indim = 16;
    typedef void(*comp_func)(const double*, const size_t, const unsigned int, double*, const double);
    comp_func func;

public:
    Test_ComputeGP(comp_func fn) : func(fn) {}
    ~Test_ComputeGP() = default;

    void init() {
        rands(X, N, in_dim);
    }

    virtual void init_perf() {
        init();
    }

    virtual void init_validate() {
        init();
    }

    virtual double perf_test_2(const int N_){
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
                (*func)(X, perf_test_N, perf_test_indim, P, perp);
            }
            end = stop_tsc(start);

            cycles = (double)end;
            multiplier = (CYCLES_REQUIRED) / (cycles);

        } while (multiplier > 2);



        double total_cycles = 0;
        for (size_t j = 0; j < REP; j++) {
            start = start_tsc();
            for (size_t i = 0; i < num_runs; ++i) {
                (*func)(X, perf_test_N, perf_test_indim, P, perp);
            }
            end = stop_tsc(start);

            cycles = ((double)end) / num_runs;
            total_cycles += cycles;
        }
        cycles = total_cycles / (double)REP;
        return cycles;
    };

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
                (*func)(X, perf_test_N, perf_test_indim, P, perp);
            }
            end = stop_tsc(start);

            cycles = (double)end;
            multiplier = (CYCLES_REQUIRED) / (cycles);

        } while (multiplier > 2);

        double total_cycles = 0;
        for (size_t j = 0; j < REP; j++) {
            start = start_tsc();
            for (size_t i = 0; i < num_runs; ++i) {
                (*func)(X, perf_test_N, perf_test_indim, P, perp);
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

        (*func)(X, N, in_dim, P, perp);
        computeGPv1::computeGaussianPerplexity(X, N, in_dim, baseP, perp);
        error = nrm_sqr_diff(P, baseP, N * N);

        print_error(error);
    }

    virtual void print_perf(double cycles, long num_runs) {
        cout << "\nPerformance Test Report\n";
        cout << "Number of samples = \t" << perf_test_N << endl;
        cout << "Input dimension   = \t" << perf_test_indim << endl;
        cout << "Repeat " << num_runs << " runs for " << REP << " times.\n";
        cout << "Avg cycles = " << cycles << endl;
        cout << endl;
    }
};

