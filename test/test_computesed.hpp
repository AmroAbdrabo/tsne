#pragma once

#include "test.hpp"
#include "kernels/computeSED.hpp"
#include <vector>
#include <string>

class Test_ComputeSED : public Test
{
public:
    typedef void(*comp_func)(const double*, int, int, double*);
private:
    static const int perf_test_N = 8192;
    static const int perf_in_dim = 2;
    double X[perf_test_N * perf_in_dim] __attribute__ ((aligned (32)));
    comp_func func;
    double DD[perf_test_N * perf_test_N] __attribute__ ((aligned (32)));
    double baseDD[perf_test_N * perf_test_N] __attribute__ ((aligned (32)));

public:
    Test_ComputeSED(comp_func fn) : func(fn) {}
    ~Test_ComputeSED() = default;

    virtual void init_perf() {    
        rands(X, perf_test_N, perf_in_dim);
    }
    
    virtual void init_perf(double* X, int perf_test_N, int perf_in_dim) {    
        rands(X, perf_test_N, perf_in_dim);
    }

    virtual void init_validate() {
        rands(X, perf_test_N, perf_in_dim);
    }

    virtual void sweep_input_size() {
        int dim = 2;
        int N = 64;
        std::vector<int> size_ls;
        while(N <= 8192) {
           size_ls.push_back(N);
           N *= 2;
        }

        for(auto NN : size_ls) {
            // allocate the memory
            std::cout << "Running N = " << NN << std::endl;
            double * X = static_cast<double*>(aligned_alloc(32, dim*NN*sizeof(double)));
            double * DD = static_cast<double*>(aligned_alloc(32, NN*NN*sizeof(double)));
            perf_test(this->func, X, NN, dim, DD);
            delete X;
            delete DD;
        }
    }

    void perf_test(comp_func func, double * X, int perf_test_N, int perf_in_dim, double * DD) {
        init_perf(X, perf_test_N, perf_in_dim);
        double cycles = 0.;
        long num_runs = 8;
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
        //print_perf(cycles, num_runs);
        printElement(cycles);
        std::cout << std::endl;
    }

    void sweep(std::vector<comp_func>& funcs_to_test, 
            std::vector<std::string>& names, std::vector<int>& size_to_test) {

        int dim = 2;
        printElement("Function");
        printElement("N");
        printElement("Cycles");
        std::cout << std::endl;

        for(int i = 0; i < funcs_to_test.size(); i++) {
            comp_func to_test = funcs_to_test[i];
            for(int N : size_to_test) {
                printElement(names[i]);
                printElement(N);
                double * X = static_cast<double*>(aligned_alloc(32, dim*N*sizeof(double)));
                double * DD = static_cast<double*>(aligned_alloc(32, N*N*sizeof(double)));
                perf_test(to_test, X, N, dim, DD);
                delete X;
                delete DD;
            }
        }
    }

    void sweep_block_size() {
        std::vector<int> blocksizes_to_test {4, 8, 16, 32, 64, 128, 256};
        int perf_test_N = 1024; 
        int perf_test_dim = 2;
        double * X = static_cast<double*>(aligned_alloc(32, perf_test_dim*perf_test_N*sizeof(double)));
        double * DD = static_cast<double*>(aligned_alloc(32, perf_test_N*perf_test_N*sizeof(double)));
        printElement("BlockSize");
        printElement("Cycles");
        std::cout << std::endl;
        for(auto b : blocksizes_to_test) {
            init_perf(X, perf_test_N, perf_in_dim);
            double cycles = 0.;
            long num_runs = 8;
            double multiplier = 1;
            unsigned long long start, end;

            do {
                num_runs = num_runs * multiplier;
                start = start_tsc();
                for (size_t i = 0; i < num_runs; i++) {
                    computeSEDv2d2ru::computeSquaredEuclideanDistance(X, perf_test_N, perf_in_dim, DD, b);      
                }
                end = stop_tsc(start);

                cycles = (double)end;
                multiplier = (CYCLES_REQUIRED) / (cycles);
                
            } while (multiplier > 2);

            double total_cycles = 0;
            for (size_t j = 0; j < REP; j++) {
                start = start_tsc();
                for (size_t i = 0; i < num_runs; ++i) {
                    computeSEDv2d2ru::computeSquaredEuclideanDistance(X, perf_test_N, perf_in_dim, DD, b);      
                }
                end = stop_tsc(start);

                cycles = ((double)end) / num_runs;
                total_cycles += cycles;
            }
            cycles = total_cycles / REP;
            printElement(b);
            printElement(cycles);
            std::cout << std::endl;
        }
    }


    virtual void perf_test() {
        perf_test(this->func, this->X, perf_test_N, perf_in_dim, this->DD);
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

    virtual double perf_test_2(const int N_) { /* no impl */ return -1; }

    virtual void print_perf(double cycles, long num_runs) {
        cout << "\nPerformance Test Report\n";
        cout << "Number of samples = \t" << perf_test_N << endl;
        cout << "Number of dimension = \t" << perf_in_dim << endl;
        cout << "Repeat " << num_runs << " runs for " << REP << " times.\n";
        cout << "Avg cycles = " << cycles << endl; 
        cout << endl; 
    }
};
