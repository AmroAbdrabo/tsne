#pragma once

#include "immintrin.h"

/**
 * Zero out the mean for each dimension
 *
 * @param X data matrix, dim=N*D, row major order
 * @param N number of data points in X
 * @param D dimensions of data in X
 */

namespace zeroMeanv1 {
    inline void zeroMean(double* X, int N, int D) {
        // Compute data mean
        double* mean = (double*) calloc(D, sizeof(double));
        if(mean == NULL) { printf("Memory allocation failed!\n"); exit(1); }
        int nD = 0;
        for(int n = 0; n < N; n++) {
            for(int d = 0; d < D; d++) {
                mean[d] += X[nD + d];
            }
            nD += D;
        }
        for(int d = 0; d < D; d++) {
            mean[d] /= (double) N;
        }

        // Subtract data mean
        nD = 0;
        for(int n = 0; n < N; n++) {
            for(int d = 0; d < D; d++) {
                X[nD + d] -= mean[d];
            }
            nD += D;
        }
        free(mean); mean = NULL;
    }
}


namespace zeroMeanv2 { // scalar replacement
    constexpr int cacheline = 64; // cacheline is 64 bytes
    constexpr int cacheline_doubles = 8; // cacheline can take 8 doubles

    //typedef __m256d d256;

    inline void zeroMean(double* X, int N, int D) {
        // Compute data mean
        double* mean = (double*) calloc(D, sizeof(double));
        if(mean == NULL) { printf("Memory allocation failed!\n"); exit(1); }
        int nD = 0;
        for(int d = 0; d < D; d++) {
            double temp = mean[d];
            for(int n = 0; n < N; n++) {
                temp += X[nD + d];
                 nD += D;
            }
            mean[d] = temp;
           
        }
        for(int d = 0; d < D; d++) {
            mean[d] /= (double) N;
        }

        // Subtract data mean
        nD = N * D - D; // from end to the front to reuse the cache better
        for(int d = 0; d < D; d++) {
            double temp = mean[d];
            for(int n = 0; n < N; n++) {
                X[nD + d] -= temp;
                 nD += D;
            }
        }
        free(mean); mean = NULL;
    }
}

namespace zeroMeanv3 { // AVX2
    constexpr int cacheline = 64; // cacheline is 64 bytes
    constexpr int cacheline_doubles = 8; // cacheline can take 8 doubles

    typedef __m256d d256;

    inline void zeroMean(double* X, int N, int D) {
        // Compute data mean
        double* mean = (double*) calloc(D, sizeof(double));
        if(mean == NULL) { printf("Memory allocation failed!\n"); exit(1); }
        int nD = 0;
        
        // assumption that D is divisible by 8
        int columns_vec = D/8;
        
        for (int colblock = 0; colblock < columns_vec; colblock++){
            d256 m1 = _mm256_setzero_pd();
            d256 m2 = _mm256_setzero_pd();
            for(int n = 0; n < N; n++) {
                d256 x1 = _mm256_load_pd(X + n*D + (colblock << 3));
                d256 x2 = _mm256_load_pd(X + n*D + (colblock << 3) + 4);
                m1 = _mm256_add_pd(m1, x1);
                m2 = _mm256_add_pd(m2, x2);
            }
            _mm256_store_pd(mean + (colblock << 3), m1);
            _mm256_store_pd(mean + (colblock << 3) + 4, m2);
            
        }
        
        d256 constN = _mm256_set1_pd((double)N);
        for(int d = 0; d < columns_vec; d++) {
            d256 m1 = _mm256_load_pd(mean + (d << 2));
            m1 = _mm256_div_pd(m1, constN);
            _mm256_store_pd(mean + (d << 2), m1);
        }

        for (int colblock = 0; colblock < columns_vec; colblock++){
            d256 m1 = _mm256_load_pd(mean + (colblock << 3));
            d256 m2 = _mm256_load_pd(mean + (colblock << 3) + 4);
            for(int n = 0; n < N; n++) {
                d256 x1 = _mm256_load_pd(X + n*D + (colblock << 3));
                d256 x2 = _mm256_load_pd(X + n*D + (colblock << 3) + 4);
                x1 = _mm256_sub_pd(x1, m1);
                x2 = _mm256_sub_pd(x2, m2);
                _mm256_store_pd(X + n*D + (colblock << 3), x1);
                _mm256_store_pd(X + n*D + (colblock << 3) + 4, x2);
            }
        }
        
        free(mean); mean = NULL;
    }
}
