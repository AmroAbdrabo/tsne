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

namespace zeroMeanv4 {

    // Assuming that D=2
    inline void zeromeanvec2(double *X, int N, int D){
    
        // Assumption is that D = 2
        int nbr_el = N*D;
        int limit = nbr_el - 31; // below you will see why 31 was chosen
    
    
        // Throughput of add_pd is 2 on Sky/IceLake and 1 on Haswell, Broadwell, and Ivy Bridge
        // Latency of 4 and 3 on Sky/IceLake and (Has/broadwell, Bridge) respectively
        // ---> 8 and 3 accumulators respectively (my computer uses Sky hence 8)
        d256 m1 = _mm256_setzero_pd();
        d256 m2 =  m1;
        d256 m3 =  m1;
        d256 m4 =  m1;
    
        d256 m5 = m1;
        d256 m6 = m1;
        d256 m7 = m1;
        d256 m8 = m1;
    
        d256 temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8;
    
        // In one iter, read 32 values
        int i;
        for (i = 0; i < limit; i+=32){
            temp1 = _mm256_load_pd(X+i);
            temp2 = _mm256_load_pd(X+i+4);
            temp3 = _mm256_load_pd(X+i+8);
            temp4 = _mm256_load_pd(X+i+12);
            temp5 = _mm256_load_pd(X+i+16);
            temp6 = _mm256_load_pd(X+i+20);
            temp7 = _mm256_load_pd(X+i+24);
            temp8 = _mm256_load_pd(X+i+28);
        
            m1 = _mm256_add_pd(m1, temp1);
            m2 = _mm256_add_pd(m2, temp2);
            m3 = _mm256_add_pd(m3, temp3);
            m4 = _mm256_add_pd(m4, temp4);
            m5 = _mm256_add_pd(m5, temp5);
            m6 = _mm256_add_pd(m6, temp6);
            m7 = _mm256_add_pd(m7, temp7);
            m8 = _mm256_add_pd(m8, temp8);
        }
    
        // Finish residuals
        double even_acc = 0;
        double odd_acc = 0;
    
        // nbr_el is even as D=2 so no need to worry about i+1 going out of bounds
        for (; i < nbr_el; i+=2){
            even_acc += X[i];
            odd_acc  += X[i+1];
        }
    
    
        // Add the 8 accumulators
        d256 nvec = _mm256_set1_pd(N);
        m1 = _mm256_add_pd(m1, m2);
        m1 = _mm256_add_pd(m1, m3);
        m1 = _mm256_add_pd(m1, m4);
        m1 = _mm256_add_pd(m1, m5);
        m1 = _mm256_add_pd(m1, m6);
        m1 = _mm256_add_pd(m1, m7);
        m1 = _mm256_add_pd(m1, m8);
        m1 = _mm256_div_pd(m1, nvec);
    
        // m1 is now of the form A1 | B1 | A2 | B2 where A1+A2 gives sum of the first feature (for all observations) and B1+B2 is likewise the sum of the 2nd feature
    
        
        // If we draw out the array X in memory as (a0, a1, a2, ...) and m1 as (d0, d1, d2, d3)
        // a0  a1  a2  a3 |  a4  a5  a6  a7 | a8   a9  a10  a11 | a12
        // d2  d3         |                 |                   |
        // d0  d1  d2  d3 |                 |                   |
        //         d0  d1 |  d2  d3         |                   |
        //                |  d0  d1  d2  d3 |                   |
        //                           d0  d1 | d2   d3           |
        //                                  | d0   d1  d2   d3  |
        //                                             d0   d1  |
        // we clearly see we need a vector (0, 0, d0, d1) and (d2, d3, 0, 0)
    
        double arr[4];
    
        _mm256_store_pd(arr, m1);
        arr[0] = arr[0]+(even_acc/N);
        arr[1] = arr[1] + (odd_acc/N);
    
        // If someone has a better idea to get 0, 0, d0, d1 and d2, d3, 0, 0 please go ahead
        temp2 = _mm256_set_pd(arr[1], arr[0], 0, 0);
        temp3 = _mm256_set_pd(0, 0, arr[3], arr[2]);
    
        limit =  nbr_el - 3;
    
        for (i = 0; i < limit; i+=4){
            temp1 =_mm256_load_pd(X+i);
            temp1 = _mm256_sub_pd(temp1, m1);
            temp1 = _mm256_sub_pd(temp1, temp2);
            temp1 = _mm256_sub_pd(temp1, temp3);
            _mm256_store_pd(X+i, temp1);
        }
    
        // scalar replace
        double r0 = arr[0], r1 = arr[1], r2 = arr[2], r3 = arr[3];
    
        for ( ;i < nbr_el; i+=2){
            X[i] -= (r0 + r2);
            X[i+1] -= (r1 + r3);
        }
    }

}
