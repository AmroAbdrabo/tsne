#pragma once
#include <stdio.h>
#include <cstdlib>
#include <cmath>
#include <random>
#include <iostream>
#include <cstring>
#include <iomanip>
using namespace std;

#include "common/tsc_x86.h"

constexpr int NR = 32;
constexpr int CYCLES_REQUIRED = (1*1e8);
constexpr int REP = 50;
constexpr double EPS = (1e-3);

constexpr int N = 2048;
constexpr int out_dim = 2;
constexpr int in_dim = 128;
constexpr double perp = 128;

typedef enum {
    COMPUTE_GP,
    UPD_GRADIENT,
    ZERO_MEAN,
} kernel_t;

template<typename T>
void symmetrize(T* X, const size_t N){
    for(size_t n = 0; n < N; ++n){
        for(size_t m = n + 1; m < N; ++m){
            X[m*N + n] = X[n*N + m];
        }
    }
}

template<typename T>
void rands(T * m, size_t row, size_t col)
{
    std::random_device rd;
    //std::mt19937 gen{rd()};
    std::mt19937 gen(42);
    std::uniform_real_distribution<T> dist(-1.0, 1.0);
    for (size_t i = 0; i < row*col; ++i)  
        m[i] = dist(gen);
}

template<typename T>
T nrm_sqr_diff(T *x, T *y, int n) {
    T nrm_sqr = 0.0;
    for(int i = 0; i < n; i++) {
        nrm_sqr += (x[i] - y[i]) * (x[i] - y[i]);
    }
    
    if (isnan(nrm_sqr)) {
      nrm_sqr = INFINITY;
    }
    
    return nrm_sqr;
}

template<typename T> void printElement(T t) {
    std::cout << std::left << std::setw(20) << std::setfill(' ') << t;
}

class Test {
    public:
        virtual void perf_test() = 0;
        virtual double perf_test_2(const int N_) = 0;
        virtual void validate() = 0;
        virtual void init_perf() = 0;
        virtual void init_validate() = 0;
        void print_error(double error) {
            if(error > EPS) {
                cout << error << endl;
                cout << "ERROR!!!!" << endl;
            } else {
                cout << "Validation is successful! [error = " << error << "]\n";
            }
        }
        
        virtual void print_perf(double cycles, long num_runs) {
            cout << "Repeat " << num_runs << " runs for " << REP << " times.\n";
            cout << "Avg cycles = " << cycles << endl; 
        }
};
