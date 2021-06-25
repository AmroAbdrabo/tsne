#pragma once

#include "immintrin.h"
#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <stdlib.h>

typedef __m256d d256;

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

    inline void zeroMean(double *__restrict__ X, int N, int D){ // was initally named zeromeanvec

        if (D%8 != 0){
            fprintf(stderr, "ERROR: D is not divisible by 8, consider using zeromeanblocked instead");
        }
        double* mean = static_cast<double*>(aligned_alloc(32, D*sizeof(double))); //(double*) calloc(D, sizeof(double)); requires std=c++17
        if(mean == NULL) { fprintf(stderr, "Memory allocation failed!\n"); exit(1); }
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


        int limit = 2*columns_vec;
        for(int d = 0; d < limit; d++) {
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
    inline void zeroMean(double *__restrict__ X, int N, int D){  // originally called zeromeanvec2

        if (D != 2){
            fprintf(stderr, "ERROR: Dimenion variable D must be 2");
            return;
        }
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

        d256 temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9;

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
        temp4 = _mm256_set_pd(arr[3], arr[2], arr[1], arr[0]);

        limit =  nbr_el - 31;


         for (i = 0; i < limit; i+=32){
             temp1 =_mm256_load_pd(X+i);
             temp3 =_mm256_load_pd(X+i+4);
             temp4 =_mm256_load_pd(X+i+8);
             temp5 =_mm256_load_pd(X+i+12);
             temp6 =_mm256_load_pd(X+i+16);
             temp7 =_mm256_load_pd(X+i+20);
             temp8 =_mm256_load_pd(X+i+24);
             temp9 =_mm256_load_pd(X+i+28);


             temp1 = _mm256_sub_pd(temp1, temp2);
             temp3 = _mm256_sub_pd(temp3, temp2);
             temp4 = _mm256_sub_pd(temp4, temp2);
             temp5 = _mm256_sub_pd(temp5, temp2);
             temp6 = _mm256_sub_pd(temp6, temp2);
             temp7 = _mm256_sub_pd(temp7, temp2);
             temp8 = _mm256_sub_pd(temp8, temp2);
             temp9 = _mm256_sub_pd(temp9, temp2);

             _mm256_store_pd(X+i, temp1);
             _mm256_store_pd(X+i+4, temp3);
             _mm256_store_pd(X+i+8, temp4);
             _mm256_store_pd(X+i+12, temp5);
             _mm256_store_pd(X+i+16, temp6);
             _mm256_store_pd(X+i+20, temp7);
             _mm256_store_pd(X+i+24, temp8);
             _mm256_store_pd(X+i+28, temp9);
         }

         // scalar replace
         double r0 = arr[0], r1 = arr[1], r2 = arr[2], r3 = arr[3];

         for ( ;i < nbr_el; i+=2){
             X[i] -= (r0 + r2);
             X[i+1] -= (r1 + r3);
         }



    }

}

// Blocked version
namespace zeroMeanv5 {
    inline void zeroMean(double *__restrict__ X, int N, int D){ // zeromeanblocked previously

        //constexpr int cacheline = 64; // cacheline is 64 bytes
        //constexpr int cacheline_doubles = 8; // cacheline can take 8 doubles

        // Our working set uses n1 x d (n1 is rows) block of X requires d elements of mean
        // Main advantage of this method is better ILP (more additions can occur without the dependency on mean[d] introduced by the expression mean[d]). Ignoring the precise history the miss rate of X and mean remains roughly the same. Unrolling factors (outer loop) can be tuned and number of accumulators (inner loop) depends on ceil(throughtput*latency of add) = 6 on zen3. Problem is that D may be 4 or 2, hence the residual (since D = 4 and 2 is common I may do a separate method for this case and use d+=2 instead of d+=8)


        double* mean = (double*) calloc(D, sizeof(double));
        if(mean == NULL) { printf("Memory allocation failed!\n"); exit(1); }
        int nD = 0;
        int n;
        int limit1 = N-3;
        int limit2 = D-5;
        for(n = 0; n < limit1; n+=4) {

            int d;
            for(d = 0; d < limit2; d+=6) {

                // Accessing in jumps of 4 reduces cache look-ups for mean[d] by a factor of 4
                // Accessing mean[d+1], mean[d+2], ...  improves ILP by doing more adds per cycle
                int center = nD+d;
                double t = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0;

                int c1 = center + D;
                int c2 = center + D*2;
                int c3 = center + D*3;

                t += X[center];
                t1 +=  X[center + 1];
                t2 +=  X[center + 2];
                t3 +=  X[center + 3];
                t4 +=  X[center + 4];
                t5 +=  X[center + 5];


                t += X[c1];
                t1 +=  X[c1 + 1];
                t2 +=  X[c1 + 2];
                t3 +=  X[c1 + 3];
                t4 +=  X[c1 + 4];
                t5 +=  X[c1 + 5];


                t += X[c2];
                t1 +=  X[c2 + 1];
                t2 +=  X[c2 + 2];
                t3 +=  X[c2 + 3];
                t4 +=  X[c2 + 4];
                t5 +=  X[c2 + 5];


                t += X[c3];
                t1 +=  X[c3 + 1];
                t2 +=  X[c3 + 2];
                t3 +=  X[c3 + 3];
                t4 +=  X[c3 + 4];
                t5 +=  X[c3 + 5];


                mean[d] += t;
                mean[d+1] += t1;
                mean[d+2] += t2;
                mean[d+3] += t3;
                mean[d+4] += t4;
                mean[d+5] += t5;

            }


            // The residual
            for (; d < D; d++){
                mean[d] += (X[nD + d] + X[nD + d + D] +X[nD + d + 2*D] + X[nD + d + 3*D]);
            }

            nD += (4*D);
        }

        // The residual
        for (; n < N; n++){
            for (int d = 0; d < D; ++d){
                mean[d] += X[nD+d];
            }
            nD += D;
        }

        double cast_n = (double)N;

        // I think its best to keep division separate for numeric stability (i.e avoid "zeroing out")
        for(int d = 0; d < D; d++) {
            mean[d] /= cast_n;
        }

        // Subtract data mean
        nD = 0;
        limit1 = N-5;
        // For this computation some cache lookups can be spared by unrolling n by 4
        for(n = 0; n < limit1; n+=6) {
            for(int d = 0; d < D; d++) {
                double temp = mean[d];
                int center = nD+d;
                X[center] -= temp;
                X[center + D] -= temp;
                X[center + 2*D] -= temp;
                X[center + 3*D] -= temp;
                X[center + 4*D] -= temp;
                X[center + 5*D] -= temp;
            }
            nD += (6*D);
        }

        // residual
        for (; n < N; ++n){
            for (int d = 0; d < D; ++d){
                X[nD+d] -= mean[d];
            }
            nD += D;
        }

        free(mean); mean = NULL;

    }
}



// Vectorized for D=2
namespace zeroMeanv6 {

    // assume that D=2 and Zen 3 microarchitecture
    inline void zeroMean(double *__restrict__ X, int N, int D){  //zeromeanvec2_zen3

        if (D != 2){
             fprintf(stderr, "ERROR: Dimenion variable D must be 2");
             return;
         }
         // Assumption is that D = 2
         int nbr_el = N*D;
         int limit = nbr_el - 23; // below you will see why 31 was chosen


         // Throughput of add_pd is 2 on Sky/IceLake and 1 on Haswell, Broadwell, and Ivy Bridge
         // Latency of 4 and 3 on Sky/IceLake and (Has/broadwell, Bridge) respectively
         // ---> 8 and 3 accumulators respectively (my computer uses Sky hence 8)
         // On AMD Zen 3, throughoput of ADD/SUBPD is 2 and latency is 3 so 6 accumulators for each loop which only does add/sub operations (ignoring the loads)
         d256 m1 = _mm256_setzero_pd();
         d256 m2 =  m1;
         d256 m3 =  m1;
         d256 m4 =  m1;

         d256 m5 = m1;
         d256 m6 = m1;

         d256 temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

         // In one iter, read 32 values
         int i;
         for (i = 0; i < limit; i+=24){

             temp1 = _mm256_load_pd(X+i);
             temp2 = _mm256_load_pd(X+i+4);
             temp3 = _mm256_load_pd(X+i+8);
             temp4 = _mm256_load_pd(X+i+12);
             temp5 = _mm256_load_pd(X+i+16);
             temp6 = _mm256_load_pd(X+i+20);




             m1 = _mm256_add_pd(m1, temp1);
             m2 = _mm256_add_pd(m2, temp2);
             m3 = _mm256_add_pd(m3, temp3);
             m4 = _mm256_add_pd(m4, temp4);
             m5 = _mm256_add_pd(m5, temp5);
             m6 = _mm256_add_pd(m6, temp6);


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

         double* arr =static_cast<double*>(aligned_alloc(32, 4*sizeof(double)));;

         _mm256_store_pd(arr, m1);
         arr[0] = arr[0]+(even_acc/N);
         arr[1] = arr[1] + (odd_acc/N);

         // If someone has a better idea to get 0, 0, d0, d1 and d2, d3, 0, 0 please go ahead
         temp2 = _mm256_set_pd(arr[1], arr[0], 0, 0);
         temp3 = _mm256_set_pd(0, 0, arr[3], arr[2]);
         temp4 = _mm256_set_pd(arr[3], arr[2], arr[1], arr[0]);
         temp2 = _mm256_add_pd(temp2, temp3);
         temp2 = _mm256_add_pd(temp2, temp4);

         limit =  nbr_el - 23;


         for (i = 0; i < limit; i+=24){
             temp1 =_mm256_load_pd(X+i);
             temp3 =_mm256_load_pd(X+i+4);
             temp4 =_mm256_load_pd(X+i+8);
             temp5 =_mm256_load_pd(X+i+12);
             temp6 =_mm256_load_pd(X+i+16);
             temp7 =_mm256_load_pd(X+i+20);



             temp1 = _mm256_sub_pd(temp1, temp2);
             temp3 = _mm256_sub_pd(temp3, temp2);
             temp4 = _mm256_sub_pd(temp4, temp2);
             temp5 = _mm256_sub_pd(temp5, temp2);
             temp6 = _mm256_sub_pd(temp6, temp2);
             temp7 = _mm256_sub_pd(temp7, temp2);

             _mm256_store_pd(X+i, temp1);
             _mm256_store_pd(X+i+4, temp3);
             _mm256_store_pd(X+i+8, temp4);
             _mm256_store_pd(X+i+12, temp5);
             _mm256_store_pd(X+i+16, temp6);
             _mm256_store_pd(X+i+20, temp7);
         }

         // scalar replace
         double r0 = arr[0], r1 = arr[1], r2 = arr[2], r3 = arr[3];

         for ( ;i < nbr_el; i+=2){
             X[i] -= (r0 + r2);
             X[i+1] -= (r1 + r3);
         }


    }
}

// Vectorized for D=3
namespace zeroMeanv7 {
    // Assume that D=3 - optimized for Zen 3 microarchitecture
    inline void zeroMean(double *__restrict__ X, int N, int D){   // zeromeanvec3_zen3

        if (D != 3){
            fprintf(stderr, "ERROR: Dimenion variable D must be 3");
            return;
        }
        // Assumption is that D = 3
        int nbr_el = N*D;
        int limit = nbr_el - 23; // below you will see why 31 was chosen


        d256 temp1, temp2, temp3, temp4, temp5, temp6;

        //d256 temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8;
        d256 acc1 = _mm256_setzero_pd();
        d256 acc2 = acc1, acc3 = acc1;  // first one stores in order, the columns 0 1 2 0, the second one 1 2 0 1 and the third one 2 0 1 2

        // In one iter, read 12 values
        int i;
        for (i = 0; i < limit; i+=24){ // notice also that 12  = gcd(3, 4),  4 = vec size, 3 = nbr cols --> the idea can be extended to further dimensions

            temp1 = _mm256_load_pd(X+i);
            temp2 = _mm256_load_pd(X+i+4);
            temp3 = _mm256_load_pd(X+i+8);
            temp4 = _mm256_load_pd(X+i+12);
            temp5 = _mm256_load_pd(X+i+16);
            temp6 = _mm256_load_pd(X+i+20);


            acc1 = _mm256_add_pd(acc1, temp1);
            acc2 = _mm256_add_pd(acc2, temp2);
            acc3 = _mm256_add_pd(acc3, temp3);
            acc1 = _mm256_add_pd(acc1, temp4);
            acc2 = _mm256_add_pd(acc2, temp5);
            acc3 = _mm256_add_pd(acc3, temp6);
        }

        // Finish residuals
        double mod0 = 0;
        double mod1 = 0;
        double mod2 = 0;

        // nbr_el is divisible by 3 as D=3 so no need to worry about i+1 going out of bounds
        for (; i < nbr_el; i+=3){
            mod0 += X[i];
            mod1  += X[i+1];
            mod2  += X[i+2];
        }

        double* accs = static_cast<double*>(aligned_alloc(32, 12*sizeof(double)));
        _mm256_store_pd(accs, acc1);
        _mm256_store_pd(accs+4, acc2);
        _mm256_store_pd(accs+8, acc3);

        accs[0]+= mod0;
        accs[1] += mod1;
        accs[2] += mod2;

        // In order to avoid unaligned memory accesses, we need vectors of the form ((0), (1), (2), (0)) and ((1), (2), (0), (1)), and ((2), (0), (1), (2))
        // where (i) represents sum of ith column
        // in the first pass we use the first vector, second pass the second vector, and so on rotating

        double dN = (double)N;

        double sum0 = (accs[0]+accs[3] + accs[6]+ accs[9])/dN;
        double sum1 = (accs[1]+accs[4] + accs[7]+ accs[10])/dN;
        double sum2 = (accs[2]+accs[5] + accs[8]+ accs[11])/dN;

        d256 v1 = _mm256_set_pd(sum0, sum2, sum1, sum0);
        d256 v2 = _mm256_set_pd(sum1, sum0, sum2, sum1);
        d256 v3 = _mm256_set_pd(sum2, sum1, sum0, sum2);

        int prec;

        limit = nbr_el - 23;
        for (i = 0; i < limit; i+=24){
            temp1 =_mm256_load_pd(X+i);
            temp2 =_mm256_load_pd(X+i+4);
            temp3 =_mm256_load_pd(X+i+8);
            temp4 =_mm256_load_pd(X+i+12);
            temp5 =_mm256_load_pd(X+i+16);
            temp6 =_mm256_load_pd(X+i+20);


            prec = i%3;
            if (prec == 0){
                temp1 = _mm256_sub_pd(temp1, v1);
                temp2 = _mm256_sub_pd(temp2, v2);
                temp3 = _mm256_sub_pd(temp3, v3);
                temp4 = _mm256_sub_pd(temp4, v1);
                temp5 = _mm256_sub_pd(temp5, v2);
                temp6 = _mm256_sub_pd(temp6, v3);
            }
            else if (prec == 1){
                 temp1 = _mm256_sub_pd(temp1, v2);
                 temp2 = _mm256_sub_pd(temp2, v3);
                 temp3 = _mm256_sub_pd(temp3, v1);
                 temp4 = _mm256_sub_pd(temp4, v2);
                 temp5 = _mm256_sub_pd(temp5, v3);
                 temp6 = _mm256_sub_pd(temp6, v1);
            }
            else{
                temp1 = _mm256_sub_pd(temp1, v3);
                temp2 = _mm256_sub_pd(temp2, v1);
                temp3 = _mm256_sub_pd(temp3, v2);
                temp4 = _mm256_sub_pd(temp4, v3);
                temp5 = _mm256_sub_pd(temp5, v1);
                temp6 = _mm256_sub_pd(temp6, v2);
            }

            _mm256_store_pd(X+i, temp1);
            _mm256_store_pd(X+i+4, temp2);
            _mm256_store_pd(X+i+8, temp3);
            _mm256_store_pd(X+i+12, temp4);
            _mm256_store_pd(X+i+16, temp5);
            _mm256_store_pd(X+i+20, temp6);
        }

        double sums[3] =  {sum0, sum1, sum2};
        for ( ;i < nbr_el; ++i){

            X[i] -= sums[i%3];
        }


    }

}
