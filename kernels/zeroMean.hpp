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

/*
namespace zeroMeanv2 { // unrolling
    constexpr int cacheline = 64; // cacheline is 64 bytes
    constexpr int cacheline_doubles = 8; // cacheline can take 8 doubles

    typedef __m256d d256;

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
        nD = N * D - D; // from end to the front to reuse the cache better
        for(int n = 0; n < N; n++) {
            for(int d = 0; d < D; d++) {
                X[nD + d] -= mean[d];
            }
            nD -= D;
        }
        free(mean); mean = NULL;
    }
}
*/
