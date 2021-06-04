#pragma once

#include "test.hpp"
#include "kernels/updateGradient.hpp"
#include "kernels/computeSED.hpp"

class Test_UpdGradient : public Test
{
private:
    /* data */
    double momentum = .5;
    double eta = 200;
    int perf_test_N = 256;     

    double* P = static_cast<double *>(aligned_alloc(32, N * N * sizeof(double)));
    double* Y = static_cast<double *>(aligned_alloc(32, N * out_dim * sizeof(double)));
    double* dY = static_cast<double *>(aligned_alloc(32, N * out_dim * sizeof(double)));
    double* uY = static_cast<double *>(aligned_alloc(32, N * out_dim * sizeof(double)));
    double* gains = static_cast<double *>(aligned_alloc(32, N * out_dim * sizeof(double)));
    double* baseY = static_cast<double *>(aligned_alloc(32, N * out_dim * sizeof(double)));
    double* basedY = static_cast<double *>(aligned_alloc(32, N * out_dim * sizeof(double)));
    double* baseuY = static_cast<double *>(aligned_alloc(32, N * out_dim * sizeof(double)));
    double* basegains = static_cast<double *>(aligned_alloc(32, N * out_dim * sizeof(double)));

    double* DD = static_cast<double *>(aligned_alloc(32, N * N * sizeof(double)));
    double* Q  = static_cast<double *>(aligned_alloc(32, N * N * sizeof(double)));
    
    
    typedef void(*comp_func)(const double*, double*, int, int, double*, double*, double*, const double, const double, const double*, double*);
    comp_func func;

public:
    Test_UpdGradient(comp_func fn) : func(fn) {}
    ~Test_UpdGradient(){
        free(P); P = NULL;
        free(Y); Y = NULL;
        free(dY); dY = NULL;
        free(uY); uY = NULL;
        free(gains); gains = NULL;
        free(baseY); baseY = NULL;
        free(basedY); basedY = NULL;
        free(baseuY); baseuY = NULL;
        free(basegains); basegains = NULL;

        free(DD); DD = NULL;
        free(Q); Q = NULL;
    }

    virtual void init_perf() {
        rands(P, N, N);
        rands(Y, N, out_dim);
        rands(dY, N, out_dim);
        rands(uY, N, out_dim);
        rands(gains, N, out_dim);

        symmetrize(P, N);   //symmetrizes the squuared matrix

        computeSEDv1::computeSquaredEuclideanDistance(Y, N, out_dim, DD);
    }

    virtual void init_validate() {
        rands(P, N, N);
        rands(Y, N, out_dim);
        rands(dY, N, out_dim);
        rands(uY, N, out_dim);
        rands(gains, N, out_dim);
        memcpy(baseY, Y, out_dim * N * sizeof(double));
        memcpy(basedY, dY, out_dim * N * sizeof(double));
        memcpy(baseuY, uY, out_dim * N * sizeof(double));
        memcpy(basegains, gains, out_dim * N * sizeof(double));

        symmetrize(P, N);   //symmetrizes the squared matrix

        computeSEDv1::computeSquaredEuclideanDistance(Y, N, out_dim, DD);
    }
    
    virtual void perf_test() {
        init_perf();

        double cycles = 0.;
        long num_runs = 100;
        double multiplier = 1.0;
        unsigned long long start, end;      

        //warmup
        do {
            num_runs = num_runs * multiplier;
            start = start_tsc();
            for (size_t i = 0; i < num_runs; i++) {
                (*func)(P, Y, perf_test_N, out_dim, dY, uY, gains, momentum, eta, DD, Q);      
            }
            end = stop_tsc(start);

            cycles = (double)end;
            multiplier = (CYCLES_REQUIRED) / (cycles);
            
        } while (multiplier > 2);
        

        // cout << "Finished warming up, num_runs =  " << num_runs << ", cycles = " << cycles << endl;

        double total_cycles = 0;
        for (size_t j = 0; j < REP; j++) {
            start = start_tsc();
            for (size_t i = 0; i < num_runs; ++i) {
                // cout << "Run: REP = " << j << " run = " << i << endl;
                (*func)(P, Y, perf_test_N, out_dim, dY, uY, gains, momentum, eta, DD, Q);      
            }
            end = stop_tsc(start);
            
            cycles = ((double)end) / num_runs;
            total_cycles += cycles;
        }
        cycles = total_cycles / (double)REP;
        print_perf(cycles, num_runs);
    }
    
    virtual void validate() 
    {
        init_validate();
        double error = .0;

        (*func)(P, Y, N, out_dim, dY, uY, gains, momentum, eta, DD, Q);
        updateGradientv1::updateGradient(P, baseY, N, out_dim, basedY, baseuY, basegains, momentum, eta, DD, Q);
        error = (nrm_sqr_diff(Y, baseY, N * out_dim) + nrm_sqr_diff(dY, basedY, N * out_dim) + 
                nrm_sqr_diff(uY, baseuY, N * out_dim) + nrm_sqr_diff(gains, basegains, N * out_dim)) / 4;
        
        print_error(error);
    }

    virtual void print_perf(double cycles, long num_runs) {
        cout << "\nPerformance Test Report\n";
        cout << "Number of samples = \t" << perf_test_N << endl;
        cout << "Output dimension  = \t" << out_dim << endl;
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
                (*func)(P, Y, N_, out_dim, dY, uY, gains, momentum, eta, DD, Q);      
            }
            end = stop_tsc(start);

            cycles = (double)end;
            multiplier = (CYCLES_REQUIRED) / (cycles);
            
        } while (multiplier > 2);
        


        double total_cycles = 0;
        for (size_t j = 0; j < REP; j++) {
            start = start_tsc();
            for (size_t i = 0; i < num_runs; ++i) {
                (*func)(P, Y, N_, out_dim, dY, uY, gains, momentum, eta, DD, Q);      
            }
            end = stop_tsc(start);
            
            cycles = ((double)end) / num_runs;
            total_cycles += cycles;
        }
        cycles = total_cycles / (double)REP;
        return cycles;
    }
};

