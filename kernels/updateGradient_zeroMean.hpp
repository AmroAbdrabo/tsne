#pragma once
#include "computeSED.hpp"
#include "zeroMean.hpp"

#include <immintrin.h>


/**
 * compute the gradient and update the output data for each iteration
 * @param P N*N symmetric matrix, P[i][j] is the distance between X[i] and X[j]
 * @param Y output data whose dim=N*out_dim, row major order
 * @param N number of data
 * @param out_dim dimension of output data
 * @param dY gradient of this iteration, dim=N*out_dim
 * @param uY decaying history gradient, dim=N*out_dim
 * @param gains dim=N*out_dim
 * @param momentum the coefficient of the momentum term
 * @param eta learning rate
 */

//standart version
namespace updateGradient_zeroMeanv1 {
    
    inline void updateGradient_zeroMean(const double* P, double* Y, int N, int out_dim, double* dY, 
                                double* uY, double* gains, const double momentum, const double eta) {
        // Make sure the current gradient contains zeros
        for(int i = 0; i < N * out_dim; i++) dY[i] = 0.0;

        // Compute the squared Euclidean distance matrix
        double* DD = (double*) malloc(N * N * sizeof(double));
        if(DD == NULL) { printf("Memory allocation failed!\n"); exit(1); }
        computeSEDv1::computeSquaredEuclideanDistance(Y, N, out_dim, DD);

        // Compute Q-matrix and normalization sum
        double* Q    = (double*) malloc(N * N * sizeof(double));
        if(Q == NULL) { printf("Memory allocation failed!\n"); exit(1); }
        double sum_Q = 0.0;
        //int cnt = 0;
        int nN = 0;
        for(int n = 0; n < N; n++) {
            for(int m = 0; m < N; m++) {
                if(n != m) {
                    Q[nN + m] = 1.0 / (1.0 + DD[nN + m]);
                    sum_Q += Q[nN + m];
                    //++cnt;
                }
            }
            nN += N;
        }

        // Perform the computation of the gradient
        nN = 0;
        int nD = 0;
        for(int n = 0; n < N; n++) {
            int mD = 0;
            for(int m = 0; m < N; m++) {
                if(n != m) {
                    double mult = (P[nN + m] - (Q[nN + m] / sum_Q)) * Q[nN + m];
                    for(int d = 0; d < out_dim; d++) {
                        dY[nD + d] += (Y[nD + d] - Y[mD + d]) * mult;
                    }
                }
                mD += out_dim;
            }
            nN += N;
            nD += out_dim;
        }

        // Free memory
        free(DD); DD = NULL;
        free(Q);  Q  = NULL;

        // Update gains
        for(int i = 0; i < N * out_dim; i++) gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + .2) : (gains[i] * .8);
        for(int i = 0; i < N * out_dim; i++) if(gains[i] < .01) gains[i] = .01;

        // Perform gradient update (with momentum and gains)
        for(int i = 0; i < N * out_dim; i++) uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
		for(int i = 0; i < N * out_dim; i++)  Y[i] = Y[i] + uY[i];

        zeroMeanv1::zeroMean(Y, N, out_dim);    //only for validation
    }
}

//unpacked SED & zeroMean into updateGradient (all v1) (only for validation)
namespace updateGradient_zeroMeanv2 {
    
    inline void updateGradient_zeroMean(const double* P, double* Y, int N, int out_dim, double* dY, 
                                double* uY, double* gains, const double momentum, const double eta) {
        // Make sure the current gradient contains zeros
        for(int i = 0; i < N * out_dim; i++) dY[i] = 0.0;

        // Compute the squared Euclidean distance matrix
        double* DD = (double*) malloc(N * N * sizeof(double));
        if(DD == NULL) { printf("Memory allocation failed!\n"); exit(1); }

        /****SED****/
        const double* YnD = Y;
        for(int n = 0; n < N; ++n, YnD += out_dim) { // point[n]
            const double* YmD = YnD + out_dim; // point[n+1]
            double* curr_elem = &DD[n*N + n]; // DD[n,n]
            *curr_elem = 0.0; // DD[n,n] = 0
            double* curr_elem_sym = curr_elem + N; // DD[n+1,n] = dist(point[n], point[n+1])
            for(int m = n + 1; m < N; ++m, YmD+=out_dim, curr_elem_sym+=N) {
                *(++curr_elem) = 0.0;
                for(int d = 0; d < out_dim; ++d) {
                    *curr_elem += (YnD[d] - YmD[d]) * (YnD[d] - YmD[d]); // DD[n,m] = dist(point[n], point[m])
                }
                *curr_elem_sym = *curr_elem; // DD[m,n] = DD[n,m]
            }
        }
        /****SED****/

        // Compute Q-matrix and normalization sum
        double* Q    = (double*) malloc(N * N * sizeof(double));
        if(Q == NULL) { printf("Memory allocation failed!\n"); exit(1); }
        double sum_Q = .0;
        int nN = 0;
        for(int n = 0; n < N; n++) {
            for(int m = 0; m < N; m++) {
                if(n != m) {
                    Q[nN + m] = 1 / (1 + DD[nN + m]);
                    sum_Q += Q[nN + m];
                }
            }
            nN += N;
        }

        // Perform the computation of the gradient
        nN = 0;
        int nD = 0;
        for(int n = 0; n < N; n++) {
            int mD = 0;
            for(int m = 0; m < N; m++) {
                if(n != m) {
                    double mult = (P[nN + m] - (Q[nN + m] / sum_Q)) * Q[nN + m];
                    for(int d = 0; d < out_dim; d++) {
                        dY[nD + d] += (Y[nD + d] - Y[mD + d]) * mult;
                    }
                }
                mD += out_dim;
            }
            nN += N;
            nD += out_dim;
        }

        // Free memory
        free(DD); DD = NULL;
        free(Q);  Q  = NULL;

        // Update gains
        for(int i = 0; i < N * out_dim; i++) gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + .2) : (gains[i] * .8);
        for(int i = 0; i < N * out_dim; i++) if(gains[i] < .01) gains[i] = .01;

        // Perform gradient update (with momentum and gains)
        for(int i = 0; i < N * out_dim; i++) uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
		for(int i = 0; i < N * out_dim; i++)  Y[i] = Y[i] + uY[i];


        /****zeroMean****/
        double* mean = (double*) calloc(out_dim, sizeof(double));
        if(mean == NULL) { printf("Memory allocation failed!\n"); exit(1); }
        nD = 0;
        for(int n = 0; n < N; n++) {
            for(int d = 0; d < out_dim; d++) {
                mean[d] += Y[nD + d];
            }
            nD += out_dim;
        }
        for(int d = 0; d < out_dim; d++) {
            mean[d] /= (double) N;
        }

        // Subtract data mean
        nD = 0;
        for(int n = 0; n < N; n++) {
            for(int d = 0; d < out_dim; d++) {
                Y[nD + d] -= mean[d];
            }
            nD += out_dim;
        }
        free(mean); mean = NULL;
        /****zeroMean****/
    }
}

//combining all optimized kernels (any out_dim)
namespace updateGradient_zeroMeanv3_dx {

    inline void updateGradient_zeroMean(const double* P, double* Y, int N, int out_dim, double* dY, 
                                double* uY, double* gains, const double momentum, const double eta) {

        // Make sure the current gradient contains zeros    (at every iteration!)
        const int ND = N * out_dim;
        int i = 0;
        for(; i < ND - 3; i += 4){
            _mm256_store_pd(dY + i, zero_vec);
        }
        for(; i < ND; ++i) dY[i] = 0.0;
        
        // Compute the squared Euclidean distance matrix
        double* DD = static_cast<double *>(aligned_alloc(32, N * N * sizeof(double)));  //use aligned_alloc for intel intinsics!
        if(DD == NULL) { printf("Memory allocation failed!\n"); exit(1); }
        /****SED****/
        computeSEDv1::computeSquaredEuclideanDistance(Y, N, out_dim, DD);
        /****SED****/

        // Compute Q-matrix and normalization sum
        double* Q = static_cast<double *>(aligned_alloc(32, N * N * sizeof(double)));
        if(Q == NULL) { printf("Memory allocation failed!\n"); exit(1); }
        double sum_Q = 0.0;


        __m256d sum_Q_vec = _mm256_setzero_pd(); 
        __m256d Q_vec, DD_vec;
        
        const int N2 = N*N;
        for(i = 0; i < N2 - 3; i += 4) {
            DD_vec = _mm256_load_pd(DD + i);
            DD_vec = _mm256_add_pd(one_vec, DD_vec);
            Q_vec  = _mm256_div_pd(one_vec, DD_vec);
            
            _mm256_store_pd(Q + i, Q_vec);

            sum_Q_vec = _mm256_add_pd(sum_Q_vec, Q_vec); 
        }
        for(; i < N2; i += 2){
            Q[i] = 1.0 / (1.0 + DD[i]);
            Q[i+1] = 1.0 / (1.0 + DD[i+1]);
            sum_Q += Q[i] + Q[i+1];
        }
        sum_Q_vec = _mm256_hadd_pd(sum_Q_vec, sum_Q_vec);
        sum_Q += sum_Q_vec[0] + sum_Q_vec[2];
        
        // Perform the computation of the gradient
        const double sum_Q_inv = 1.0 / (sum_Q - double(N));

        int nN = 0;
        int mD;
        int nD = 0;
        for(int n = 0; n < N; ++n) {
            mD = 0;
            for(int m = 0; m < N; ++m) {
                if(n != m) {
                    double mult = (P[nN + m] - (Q[nN + m] * sum_Q_inv)) * Q[nN + m];
                    for(int d = 0; d < out_dim; ++d) {
                        dY[nD + d] += (Y[nD + d] - Y[mD + d]) * mult;
                    }
                }
                mD += out_dim;
            }
            nN += N;
            nD += out_dim;
        }

        // Free memory
        free(DD); DD = NULL;
        free(Q);  Q  = NULL;


        const __m256d eta_vec  = _mm256_set1_pd(eta);
        const __m256d mom_vec  = _mm256_set1_pd(momentum);
        __m256d gain_vec, dY_vec, Y_vec, uY_vec, sign_dY_vec, sign_uY_vec, gain_cmp, eta_dY_vec;

        for(i = 0; i < ND - 3; i += 4){
            gain_vec = _mm256_load_pd(gains + i);
            dY_vec   = _mm256_load_pd(dY + i);
            uY_vec   = _mm256_load_pd(uY + i);
            Y_vec    = _mm256_load_pd(Y + i);

            eta_dY_vec = _mm256_mul_pd(eta_vec, dY_vec);

            gain_cmp = _mm256_cmp_pd(sign(dY_vec) , sign(uY_vec), _CMP_NEQ_OQ);
            gain_vec = _mm256_blendv_pd(_mm256_mul_pd(gain_vec, z8_vec), _mm256_add_pd(gain_vec, z2_vec), gain_cmp);
            gain_cmp = _mm256_cmp_pd(zz1_vec, gain_vec, _CMP_GT_OQ);
            gain_vec = _mm256_blendv_pd(gain_vec, zz1_vec, gain_cmp);

            _mm256_store_pd(gains + i, gain_vec);

            uY_vec = _mm256_fmsub_pd(mom_vec, uY_vec, _mm256_mul_pd(gain_vec, eta_dY_vec));
            
            _mm256_store_pd(uY + i, uY_vec);
            
            Y_vec  = _mm256_add_pd(Y_vec, uY_vec);

            _mm256_store_pd(Y + i, Y_vec);
        }
        for(; i < ND; ++i){
            gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + 0.2) : (gains[i] * 0.8);
            if(gains[i] < 0.01) gains[i] = 0.01;

            uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
            Y[i] = Y[i] + uY[i];
        }

        /****zeroMean****/
        zeroMeanv5::zeroMean(Y, N, out_dim);
        /****zeroMean****/
    }
}


//combining all optimized kernels (out_dim=2)
namespace updateGradient_zeroMeanv3_d2 {

    inline void updateGradient_zeroMean(const double* P, double* Y, int N, int out_dim, double* dY, 
                                double* uY, double* gains, const double momentum, const double eta) {
        // Compute the squared Euclidean distance matrix
        double* DD = static_cast<double *>(aligned_alloc(32, N * N * sizeof(double)));  //use aligned_alloc for intel intinsics!
        if(DD == NULL) { printf("Memory allocation failed!\n"); exit(1); }

        /****SED****/
        computeSEDv2d2buf::computeSquaredEuclideanDistance(Y, N, out_dim, DD);
        /****SED****/

        // Compute Q-matrix and normalization sum
        double* Q = static_cast<double *>(aligned_alloc(32, N * N * sizeof(double)));
        if(Q == NULL) { printf("Memory allocation failed!\n"); exit(1); }
        double sum_Q = 0.0;


        __m256d sum_Q_vec = _mm256_setzero_pd(); 
        __m256d Q_vec, DD_vec;
        
        int i;
        const int N2 = N*N;
        for(i = 0; i < N2 - 3; i += 4) {
            DD_vec = _mm256_load_pd(DD + i);
            DD_vec = _mm256_add_pd(one_vec, DD_vec);
            Q_vec  = _mm256_div_pd(one_vec, DD_vec);
            
            _mm256_store_pd(Q + i, Q_vec);

            sum_Q_vec = _mm256_add_pd(sum_Q_vec, Q_vec); 
        }
        for(; i < N2; i += 2){
            Q[i] = 1.0 / (1.0 + DD[i]);
            Q[i+1] = 1.0 / (1.0 + DD[i+1]);
            sum_Q += Q[i] + Q[i+1];
        }
        sum_Q_vec = _mm256_hadd_pd(sum_Q_vec, sum_Q_vec);
        sum_Q += sum_Q_vec[0] + sum_Q_vec[2];
        
        // Perform the computation of the gradient
        const double sum_Q_inv = 1.0 / (sum_Q - double(N));
        const __m256d sum_Q_inv_vec = _mm256_set1_pd(sum_Q_inv);

        __m256d P_vec, dY_vec1, dY_vec2, dY_vec1_sum, dY_vec2_sum, Y_nD_vec1, Y_nD_vec2, Y_mD_vec1, Y_mD_vec2, Y_mD_vec3, Y_mD_vec4, mult_vec;

        double dY_temp[2];

        int nN = 0;
        int m, mD;
        int nD = 0;
        for(int n = 0; n < N; ++n) {
            mD = 0;

            dY_temp[0] = 0.0;
            dY_temp[1] = 0.0;
            dY_vec1_sum = _mm256_setzero_pd();
            dY_vec2_sum = _mm256_setzero_pd();
            Y_nD_vec1  = _mm256_broadcast_sd(Y + nD);
            Y_nD_vec2  = _mm256_broadcast_sd(Y + nD + 1);

            for(m = 0; m < N - 3; m += 4) {
                P_vec = _mm256_load_pd(P + nN + m);
                Q_vec = _mm256_load_pd(Q + nN + m);
                
                mult_vec  = _mm256_mul_pd(_mm256_fmsub_pd(Q_vec, sum_Q_inv_vec, P_vec), Q_vec); //actually this is -mult

                Y_mD_vec1 = _mm256_load_pd(Y + mD);
                Y_mD_vec2 = _mm256_load_pd(Y + mD + 4);

                Y_mD_vec3 = _mm256_unpacklo_pd(Y_mD_vec1, Y_mD_vec2);
                Y_mD_vec4 = _mm256_unpackhi_pd(Y_mD_vec1, Y_mD_vec2);

                Y_mD_vec1 = _mm256_permute4x64_pd(Y_mD_vec3, 0b11011000);
                Y_mD_vec2 = _mm256_permute4x64_pd(Y_mD_vec4, 0b11011000);

                dY_vec1 = _mm256_sub_pd(Y_mD_vec1, Y_nD_vec1);
                dY_vec2 = _mm256_sub_pd(Y_mD_vec2, Y_nD_vec2);

                dY_vec1 = _mm256_mul_pd(dY_vec1, mult_vec);
                dY_vec2 = _mm256_mul_pd(dY_vec2, mult_vec);

                dY_vec1_sum = _mm256_add_pd(dY_vec1_sum, dY_vec1);
                dY_vec2_sum = _mm256_add_pd(dY_vec2_sum, dY_vec2);

                mD += 8;
            }
            for(; m < N; ++m) {
                if(n != m) {
                    double mult = (P[nN + m] - (Q[nN + m] * sum_Q_inv)) * Q[nN + m];

                    dY_temp[0] += (Y[nD    ] - Y[mD    ]) * mult;
                    dY_temp[1] += (Y[nD + 1] - Y[mD + 1]) * mult;
                }
                mD += 2;
            }
            
            dY_vec1_sum = _mm256_hadd_pd(dY_vec1_sum, dY_vec1_sum);
            dY_vec2_sum = _mm256_hadd_pd(dY_vec2_sum, dY_vec2_sum);

            dY_temp[0] += ((double*)&dY_vec1_sum)[0] + ((double*)&dY_vec1_sum)[2];
            dY_temp[1] += ((double*)&dY_vec2_sum)[0] + ((double*)&dY_vec2_sum)[2];
            
            dY[nD    ] = dY_temp[0];
            dY[nD + 1] = dY_temp[1];
            
            
            nN += N;
            nD += 2;
        }
        

        // Free memory
        free(DD); DD = NULL;
        free(Q);  Q  = NULL;


        const __m256d eta_vec  = _mm256_set1_pd(eta);
        const __m256d mom_vec  = _mm256_set1_pd(momentum);
        __m256d gain_vec, dY_vec, Y_vec, uY_vec, sign_dY_vec, sign_uY_vec, gain_cmp, eta_dY_vec;

        const int ND = N * out_dim;
        double gains1, gains2, dY1, dY2, uY1, uY2;
        for(i = 0; i < ND - 3; i += 4){
            gain_vec = _mm256_load_pd(gains + i);
            dY_vec   = _mm256_load_pd(dY + i);
            uY_vec   = _mm256_load_pd(uY + i);
            Y_vec    = _mm256_load_pd(Y + i);

            eta_dY_vec = _mm256_mul_pd(eta_vec, dY_vec);

            gain_cmp = _mm256_cmp_pd(sign(dY_vec) , sign(uY_vec), _CMP_NEQ_OQ);
            gain_vec = _mm256_blendv_pd(_mm256_mul_pd(gain_vec, z8_vec), _mm256_add_pd(gain_vec, z2_vec), gain_cmp);
            gain_cmp = _mm256_cmp_pd(zz1_vec, gain_vec, _CMP_GT_OQ);
            gain_vec = _mm256_blendv_pd(gain_vec, zz1_vec, gain_cmp);

            _mm256_store_pd(gains + i, gain_vec);

            uY_vec = _mm256_fmsub_pd(mom_vec, uY_vec, _mm256_mul_pd(gain_vec, eta_dY_vec));
            
            _mm256_store_pd(uY + i, uY_vec);
            
            Y_vec  = _mm256_add_pd(Y_vec, uY_vec);

            _mm256_store_pd(Y + i, Y_vec);
        }
        for(; i < ND; i += 2){
            gains1 = gains[i];
            gains2 = gains[i+1];

            dY1 = dY[i];
            dY2 = dY[i+1];

            uY1 = uY[i];
            uY2 = uY[i+1];

            gains1 = (sign(dY1) != sign(uY1)) ? (gains1 + 0.2) : (gains1 * 0.8);
            gains2 = (sign(dY2) != sign(uY2)) ? (gains2 + 0.2) : (gains2 * 0.8);
            gains[i]   = (gains1 < 0.01) ? 0.01 : gains1;
            gains[i+1] = (gains2 < 0.01) ? 0.01 : gains2;

            uY[i]   = momentum * uY1 - eta * gains[i] * dY1;
            uY[i+1] = momentum * uY2 - eta * gains[i+1] * dY2;

            Y[i] = Y[i] + uY[i];
            Y[i+1] = Y[i+1] + uY[i+1];
        } 

        /****zeroMean****/
        zeroMeanv6::zeroMean(Y, N, out_dim);
        /****zeroMean****/
    }
}

//unpack all optimized kernels and optimize (out_dim=2)
namespace updateGradient_zeroMeanv4_d2 {
    typedef __m256d d256;
    inline void updateGradient_zeroMean(const double* P, double* Y, int N, int out_dim, double* dY, 
                                double* uY, double* gains, const double momentum, const double eta) {
        
        // Compute the squared Euclidean distance matrix
        // Compute Q-matrix and normalization sum
        double* Q = static_cast<double *>(aligned_alloc(32, N * N * sizeof(double)));  //use aligned_alloc for intel intrinsics!
        if(Q == NULL) { printf("Memory allocation failed!\n"); exit(1); }

        /****SED****/
        //int cnt = 0;
        __m256d sum_Q_vec = _mm256_setzero_pd(); 
        __m256d Q_vec;

        const int b = 16; // block size for cache
        const int rbi = 4; // block size for registers, in the following unrolling, assume rbi = 4, o/w it won't work
        const int rbj = 16; // block size for registers
        
        const double* Xi = Y;
        for(int i = 0; i < N - b + 1; i += b, Xi += b * out_dim) {
            const double* Xj = Y + i * out_dim;
            for(int j = i; j < N - b + 1; j += b, Xj += b * out_dim) {
                const double* Xii = Xi;
                for(int ii = i; ii < i + b - rbi + 1; ii += rbi, Xii += rbi * out_dim) {
                    const double* Xjj = Xj;
                    
                    __m256d x01 = _mm256_load_pd(Xii);
                    __m256d x23 = _mm256_load_pd(Xii + 4);
                    __m256d xvecd0 = _mm256_unpacklo_pd(x01, x23); // dim 0 
                    __m256d xvecd1 = _mm256_unpackhi_pd(x01, x23); // dim 1
                    xvecd0 = _mm256_permute4x64_pd(xvecd0, 0b11011000);
                    xvecd1 = _mm256_permute4x64_pd(xvecd1, 0b11011000);

                    for(int jj = j; jj < j + b; jj++, Xjj += out_dim) {
                        if(ii == jj) {
                            Q[ii * N + jj] = 0.0;    
                            continue;
                        }

                        // load y
                        __m256d yvecd0 = _mm256_set1_pd(Xjj[0]);
                        __m256d yvecd1 = _mm256_set1_pd(Xjj[1]);

                        __m256d xyd0  = _mm256_sub_pd(xvecd0, yvecd0);
                        __m256d xyd1  = _mm256_sub_pd(xvecd1, yvecd1);

                        xyd0 = _mm256_mul_pd(xyd0, xyd0);
                        xyd1 = _mm256_mul_pd(xyd1, xyd1);

                        __m256d xy = _mm256_add_pd(xyd0, xyd1);

                        xy = _mm256_add_pd(one_vec, xy);
                        xy = _mm256_div_pd(one_vec, xy);       //xy = 1.0/(1.0 + xy)

                        sum_Q_vec = _mm256_add_pd(sum_Q_vec, xy);
                        //cnt += 8;

                        const int symm_base = jj * N + ii;
                        _mm256_store_pd(Q + symm_base, xy);

                        int base = ii * N + jj;
                        Q[base] = Q[symm_base    ]; base += N;
                        Q[base] = Q[symm_base + 1]; base += N;
                        Q[base] = Q[symm_base + 2]; base += N;
                        Q[base] = Q[symm_base + 3];
                    }
                }
            }
        }
        /****SED****/

        sum_Q_vec = _mm256_hadd_pd(sum_Q_vec, sum_Q_vec);

        //printf("v4 [%.3f, %.3f, %.3f, %.3f] (%f, %d) \n", Q[N-4], Q[N-3], Q[N-2], Q[N-1], 2.0*sum_Q, cnt);
        
        const double sum_Q_inv = 0.5 / (sum_Q_vec[0] + sum_Q_vec[2]);
        const __m256d sum_Q_inv_vec = _mm256_set1_pd(sum_Q_inv);

        __m256d P_vec, dY_vec1, dY_vec2, dY_vec1_sum, dY_vec2_sum, Y_nD_vec1, Y_nD_vec2, Y_mD_vec1, Y_mD_vec2, Y_mD_vec3, Y_mD_vec4, mult_vec;

        double dY_temp[2];

        int nN = 0;
        int m, mD;
        int nD = 0;
        for(int n = 0; n < N; ++n) {
            mD = 0;

            dY_temp[0] = 0.0;
            dY_temp[1] = 0.0;
            dY_vec1_sum = _mm256_setzero_pd();
            dY_vec2_sum = _mm256_setzero_pd();
            Y_nD_vec1  = _mm256_broadcast_sd(Y + nD);
            Y_nD_vec2  = _mm256_broadcast_sd(Y + nD + 1);

            for(m = 0; m < N - 3; m += 4) {
                P_vec = _mm256_load_pd(P + nN + m);
                Q_vec = _mm256_load_pd(Q + nN + m);
                
                mult_vec  = _mm256_mul_pd(_mm256_fmsub_pd(Q_vec, sum_Q_inv_vec, P_vec), Q_vec); //actually this is -mult

                Y_mD_vec1 = _mm256_load_pd(Y + mD);
                Y_mD_vec2 = _mm256_load_pd(Y + mD + 4);

                Y_mD_vec3 = _mm256_unpacklo_pd(Y_mD_vec1, Y_mD_vec2);
                Y_mD_vec4 = _mm256_unpackhi_pd(Y_mD_vec1, Y_mD_vec2);

                Y_mD_vec1 = _mm256_permute4x64_pd(Y_mD_vec3, 0b11011000);
                Y_mD_vec2 = _mm256_permute4x64_pd(Y_mD_vec4, 0b11011000);

                dY_vec1 = _mm256_sub_pd(Y_mD_vec1, Y_nD_vec1);
                dY_vec2 = _mm256_sub_pd(Y_mD_vec2, Y_nD_vec2);

                dY_vec1 = _mm256_mul_pd(dY_vec1, mult_vec);
                dY_vec2 = _mm256_mul_pd(dY_vec2, mult_vec);

                dY_vec1_sum = _mm256_add_pd(dY_vec1_sum, dY_vec1);
                dY_vec2_sum = _mm256_add_pd(dY_vec2_sum, dY_vec2);

                mD += 8;
            }
            for(; m < N; ++m) {
                if(n != m) {
                    double mult = (P[nN + m] - (Q[nN + m] * sum_Q_inv)) * Q[nN + m];

                    dY_temp[0] += (Y[nD    ] - Y[mD    ]) * mult;
                    dY_temp[1] += (Y[nD + 1] - Y[mD + 1]) * mult;
                }
                mD += 2;
            }
            
            dY_vec1_sum = _mm256_hadd_pd(dY_vec1_sum, dY_vec1_sum);
            dY_vec2_sum = _mm256_hadd_pd(dY_vec2_sum, dY_vec2_sum);

            dY_temp[0] += dY_vec1_sum[0] + dY_vec1_sum[2];
            dY_temp[1] += dY_vec2_sum[0] + dY_vec2_sum[2];
            
            dY[nD    ] = dY_temp[0];
            dY[nD + 1] = dY_temp[1];
            
            
            nN += N;
            nD += 2;
        }
        

        // Free memory
        free(Q);  Q = NULL;


        const __m256d eta_vec  = _mm256_set1_pd(eta);
        const __m256d mom_vec  = _mm256_set1_pd(momentum);
        __m256d gain_vec, dY_vec, Y_vec, uY_vec, sign_dY_vec, sign_uY_vec, gain_cmp, eta_dY_vec;

        double gains1, gains2, dY1, dY2, uY1, uY2;
        
        int i;
        const int ND = N * out_dim;
        for(i = 0; i < ND - 3; i += 4){
            gain_vec = _mm256_load_pd(gains + i);
            dY_vec   = _mm256_load_pd(dY + i);
            uY_vec   = _mm256_load_pd(uY + i);
            Y_vec    = _mm256_load_pd(Y + i);

            eta_dY_vec = _mm256_mul_pd(eta_vec, dY_vec);

            gain_cmp = _mm256_cmp_pd(sign(dY_vec) , sign(uY_vec), _CMP_NEQ_OQ);
            gain_vec = _mm256_blendv_pd(_mm256_mul_pd(gain_vec, z8_vec), _mm256_add_pd(gain_vec, z2_vec), gain_cmp);
            gain_cmp = _mm256_cmp_pd(zz1_vec, gain_vec, _CMP_GT_OQ);
            gain_vec = _mm256_blendv_pd(gain_vec, zz1_vec, gain_cmp);

            _mm256_store_pd(gains + i, gain_vec);

            uY_vec = _mm256_fmsub_pd(mom_vec, uY_vec, _mm256_mul_pd(gain_vec, eta_dY_vec));
            
            _mm256_store_pd(uY + i, uY_vec);
            
            Y_vec  = _mm256_add_pd(Y_vec, uY_vec);

            _mm256_store_pd(Y + i, Y_vec);
        }
        for(; i < ND; i += 2){
            gains1 = gains[i];
            gains2 = gains[i+1];

            dY1 = dY[i];
            dY2 = dY[i+1];

            uY1 = uY[i];
            uY2 = uY[i+1];

            gains1 = (sign(dY1) != sign(uY1)) ? (gains1 + 0.2) : (gains1 * 0.8);
            gains2 = (sign(dY2) != sign(uY2)) ? (gains2 + 0.2) : (gains2 * 0.8);
            gains[i]   = (gains1 < 0.01) ? 0.01 : gains1;
            gains[i+1] = (gains2 < 0.01) ? 0.01 : gains2;

            uY[i]   = momentum * uY1 - eta * gains[i] * dY1;
            uY[i+1] = momentum * uY2 - eta * gains[i+1] * dY2;

            Y[i] = Y[i] + uY[i];
            Y[i+1] = Y[i+1] + uY[i+1];
        } 

        /****zeroMean****/

        // Y = X, N = N, out_dim = D
        // Assumption is that D = 2
        int limit = ND - 31; // below you will see why 31 was chosen
    
    
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
        for (i = 0; i < limit; i+=32){
            temp1 = _mm256_load_pd(Y+i);
            temp2 = _mm256_load_pd(Y+i+4);
            temp3 = _mm256_load_pd(Y+i+8);
            temp4 = _mm256_load_pd(Y+i+12);
            temp5 = _mm256_load_pd(Y+i+16);
            temp6 = _mm256_load_pd(Y+i+20);
            temp7 = _mm256_load_pd(Y+i+24);
            temp8 = _mm256_load_pd(Y+i+28);
        
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
        for (; i < ND; i+=2){
            even_acc += Y[i];
            odd_acc  += Y[i+1];
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
    
        
        // If we draw out the array Y in memory as (a0, a1, a2, ...) and m1 as (d0, d1, d2, d3)
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
        arr[0] = arr[0] + (even_acc/N);
        arr[1] = arr[1] + (odd_acc/N);
    
        // If someone has a better idea to get 0, 0, d0, d1 and d2, d3, 0, 0 please go ahead
        temp2 = _mm256_set_pd(arr[1], arr[0], 0, 0);
        temp3 = _mm256_set_pd(0, 0, arr[3], arr[2]);
    
        limit =  ND - 3;
        for (i = 0; i < limit; i+=4){
            temp1 =_mm256_load_pd(Y+i);
            temp1 = _mm256_sub_pd(temp1, m1);
            temp1 = _mm256_sub_pd(temp1, temp2);
            temp1 = _mm256_sub_pd(temp1, temp3);
            _mm256_store_pd(Y+i, temp1);
        }
    
        // scalar replace
        double r0 = arr[0], r1 = arr[1], r2 = arr[2], r3 = arr[3];
    
        //printf("v4 [%f, %f] \n", (r0 + r2), (r1 + r3));
        for (; i < ND; i += 2){
            Y[i] -= (r0 + r2);
            Y[i+1] -= (r1 + r3);
        }
        /****zeroMean****/
    }
}

//unpack all optimized kernels and optimize (out_dim=2) (own version of zeroMean)
namespace updateGradient_zeroMeanv5_d2 {

    inline void updateGradient_zeroMean(const double* P, double* Y, int N, int out_dim, double* dY, 
                                double* uY, double* gains, const double momentum, const double eta) {
        
        // Compute the squared Euclidean distance matrix
        // Compute Q-matrix and normalization sum
        double* Q = static_cast<double *>(aligned_alloc(32, N * N * sizeof(double)));  //use aligned_alloc for intel intrinsics!
        if(Q == NULL) { printf("Memory allocation failed!\n"); exit(1); }

        /****SED****/
        double sum_Q = 0.0;
        //int cnt = 0;
        __m256d sum_Q_vec = _mm256_setzero_pd(); 
        __m256d Q_vec;

        const int b = 16; // block size for cache
        const int rbi = 4; // block size for registers, in the following unrolling, assume rbi = 4, o/w it won't work
        const int rbj = 16; // block size for registers
        
        const double* Xi = Y;
        for(int i = 0; i < N - b + 1; i += b, Xi += b * out_dim) {
            const double* Xj = Y + i * out_dim;
            for(int j = i; j < N - b + 1; j += b, Xj += b * out_dim) {
                const double* Xii = Xi;
                for(int ii = i; ii < i + b - rbi + 1; ii += rbi, Xii += rbi * out_dim) {
                    const double* Xjj = Xj;
                    
                    __m256d x01 = _mm256_load_pd(Xii);
                    __m256d x23 = _mm256_load_pd(Xii + 4);
                    __m256d xvecd0 = _mm256_unpacklo_pd(x01, x23); // dim 0 
                    __m256d xvecd1 = _mm256_unpackhi_pd(x01, x23); // dim 1
                    xvecd0 = _mm256_permute4x64_pd(xvecd0, 0b11011000);
                    xvecd1 = _mm256_permute4x64_pd(xvecd1, 0b11011000);

                    for(int jj = j; jj < j + b; ++jj, Xjj += out_dim) {
                        if(ii == jj) {
                            Q[ii * N + jj] = 0.0;    
                            continue;
                        }

                        // load y
                        __m256d yvecd0 = _mm256_set1_pd(Xjj[0]);
                        __m256d yvecd1 = _mm256_set1_pd(Xjj[1]);

                        __m256d xyd0  = _mm256_sub_pd(xvecd0, yvecd0);
                        __m256d xyd1  = _mm256_sub_pd(xvecd1, yvecd1);

                        //xyd0 = _mm256_mul_pd(xyd0, xyd0);
                        xyd1 = _mm256_mul_pd(xyd1, xyd1);

                        __m256d xy = _mm256_fmadd_pd(xyd0, xyd0, xyd1);

                        xy = _mm256_add_pd(one_vec, xy);
                        xy = _mm256_div_pd(one_vec, xy);       //xy = 1.0/(1.0 + xy)

                        sum_Q_vec = _mm256_add_pd(sum_Q_vec, xy);
                        //cnt += 8;

                        const int symm_base = jj * N + ii;
                        _mm256_store_pd(Q + symm_base, xy);

                        int base = ii * N + jj;
                        Q[base] = Q[symm_base    ]; base += N;
                        Q[base] = Q[symm_base + 1]; base += N;
                        Q[base] = Q[symm_base + 2]; base += N;
                        Q[base] = Q[symm_base + 3];
                    }
                }
            }
        }
        /****SED****/

        sum_Q_vec = _mm256_hadd_pd(sum_Q_vec, sum_Q_vec);

        sum_Q += sum_Q_vec[0] + sum_Q_vec[2];

        //printf("v4 [%.3f, %.3f, %.3f, %.3f] (%f, %d) \n", Q[N-4], Q[N-3], Q[N-2], Q[N-1], 2.0*sum_Q, cnt);
        
        const double sum_Q_inv = 0.5 / sum_Q;
        const __m256d sum_Q_inv_vec = _mm256_set1_pd(sum_Q_inv);

        __m256d P_vec, dY_vec1, dY_vec2, dY_vec1_sum, dY_vec2_sum, Y_nD_vec1, Y_nD_vec2, Y_mD_vec1, Y_mD_vec2, Y_mD_vec3, Y_mD_vec4, mult_vec;

        double dY_temp[2];

        int nN = 0;
        int m, mD;
        int nD = 0;
        for(int n = 0; n < N; ++n) {
            mD = 0;

            dY_temp[0] = 0.0;
            dY_temp[1] = 0.0;
            dY_vec1_sum = _mm256_setzero_pd();
            dY_vec2_sum = _mm256_setzero_pd();
            Y_nD_vec1  = _mm256_broadcast_sd(Y + nD);
            Y_nD_vec2  = _mm256_broadcast_sd(Y + nD + 1);

            for(m = 0; m < N; m += 4) {
                P_vec = _mm256_load_pd(P + nN + m);
                Q_vec = _mm256_load_pd(Q + nN + m);
                
                mult_vec  = _mm256_mul_pd(_mm256_fmsub_pd(Q_vec, sum_Q_inv_vec, P_vec), Q_vec); //actually this is -mult

                Y_mD_vec1 = _mm256_load_pd(Y + mD);
                Y_mD_vec2 = _mm256_load_pd(Y + mD + 4);

                Y_mD_vec3 = _mm256_unpacklo_pd(Y_mD_vec1, Y_mD_vec2);
                Y_mD_vec4 = _mm256_unpackhi_pd(Y_mD_vec1, Y_mD_vec2);

                Y_mD_vec1 = _mm256_permute4x64_pd(Y_mD_vec3, 0b11011000);
                Y_mD_vec2 = _mm256_permute4x64_pd(Y_mD_vec4, 0b11011000);

                dY_vec1 = _mm256_sub_pd(Y_mD_vec1, Y_nD_vec1);
                dY_vec2 = _mm256_sub_pd(Y_mD_vec2, Y_nD_vec2);

                // dY_vec1 = _mm256_mul_pd(dY_vec1, mult_vec);
                // dY_vec2 = _mm256_mul_pd(dY_vec2, mult_vec);

                dY_vec1_sum = _mm256_fmadd_pd(mult_vec, dY_vec1, dY_vec1_sum);
                dY_vec2_sum = _mm256_fmadd_pd(mult_vec, dY_vec2, dY_vec2_sum);

                mD += 8;
            }
            for(; m < N; ++m) {
                if(n != m) {
                    double mult = (P[nN + m] - (Q[nN + m] * sum_Q_inv)) * Q[nN + m];

                    dY_temp[0] += (Y[nD    ] - Y[mD    ]) * mult;
                    dY_temp[1] += (Y[nD + 1] - Y[mD + 1]) * mult;
                }
                mD += 2;
            }
            
            dY_vec1_sum = _mm256_hadd_pd(dY_vec1_sum, dY_vec1_sum);
            dY_vec2_sum = _mm256_hadd_pd(dY_vec2_sum, dY_vec2_sum);

            dY_temp[0] += dY_vec1_sum[0] + dY_vec1_sum[2];
            dY_temp[1] += dY_vec2_sum[0] + dY_vec2_sum[2];
            
            dY[nD    ] = dY_temp[0];
            dY[nD + 1] = dY_temp[1];
            
            
            nN += N;
            nD += 2;
        }
        

        // Free memory
        free(Q);  Q = NULL;


        const __m256d eta_vec  = _mm256_set1_pd(eta);
        const __m256d mom_vec  = _mm256_set1_pd(momentum);
        __m256d gain_vec, dY_vec, Y_vec, uY_vec, sign_dY_vec, sign_uY_vec, gain_cmp, eta_dY_vec;

        __m256d mean_vec = _mm256_setzero_pd();

        double gains1, gains2, dY1, dY2, uY1, uY2;
        
        int i;
        const int ND = N * out_dim;
        for(i = 0; i < ND - 3; i += 4){
            gain_vec = _mm256_load_pd(gains + i);
            dY_vec   = _mm256_load_pd(dY + i);
            uY_vec   = _mm256_load_pd(uY + i);
            Y_vec    = _mm256_load_pd(Y + i);

            eta_dY_vec = _mm256_mul_pd(eta_vec, dY_vec);

            gain_cmp = _mm256_cmp_pd(sign(dY_vec) , sign(uY_vec), _CMP_NEQ_OQ);
            gain_vec = _mm256_blendv_pd(_mm256_mul_pd(gain_vec, z8_vec), _mm256_add_pd(gain_vec, z2_vec), gain_cmp);
            gain_cmp = _mm256_cmp_pd(zz1_vec, gain_vec, _CMP_GT_OQ);
            gain_vec = _mm256_blendv_pd(gain_vec, zz1_vec, gain_cmp);

            _mm256_store_pd(gains + i, gain_vec);

            uY_vec = _mm256_fmsub_pd(mom_vec, uY_vec, _mm256_mul_pd(gain_vec, eta_dY_vec));
            
            _mm256_store_pd(uY + i, uY_vec);
            
            Y_vec  = _mm256_add_pd(Y_vec, uY_vec);

            mean_vec = _mm256_add_pd(mean_vec, Y_vec);

            _mm256_store_pd(Y + i, Y_vec);
        }
        for(; i < ND; i += 2){
            gains1 = gains[i];
            gains2 = gains[i+1];

            dY1 = dY[i];
            dY2 = dY[i+1];

            uY1 = uY[i];
            uY2 = uY[i+1];

            gains1 = (sign(dY1) != sign(uY1)) ? (gains1 + 0.2) : (gains1 * 0.8);
            gains2 = (sign(dY2) != sign(uY2)) ? (gains2 + 0.2) : (gains2 * 0.8);
            gains[i]   = (gains1 < 0.01) ? 0.01 : gains1;
            gains[i+1] = (gains2 < 0.01) ? 0.01 : gains2;

            uY[i]   = momentum * uY1 - eta * gains[i] * dY1;
            uY[i+1] = momentum * uY2 - eta * gains[i+1] * dY2;

            Y[i] = Y[i] + uY[i];
            Y[i+1] = Y[i+1] + uY[i+1];

            mean_vec = _mm256_add_pd(mean_vec, _mm256_set_pd(0.0, 0.0, Y[i+1], Y[i]));
        } 

        /****zeroMean****/
        mean_vec = _mm256_mul_pd(mean_vec, _mm256_set1_pd(1.0/double(N)));
        mean_vec = _mm256_permute4x64_pd(mean_vec, 0b11011000);
        mean_vec = _mm256_hadd_pd(mean_vec, mean_vec);
        mean_vec = _mm256_permute4x64_pd(mean_vec, 0b11011000);

        
        for(i = 0; i < ND - 3; i += 4){
            Y_vec = _mm256_load_pd(Y + i);
            Y_vec = _mm256_sub_pd(Y_vec, mean_vec);
            _mm256_store_pd(Y + i, Y_vec);
        }
        for(; i < ND; i += 2){
            Y[i]   -= mean_vec[0];
            Y[i+1] -= mean_vec[1];
        }
        /****zeroMean****/
    }
}

//unpack all optimized kernels and optimize (out_dim=2) (own version of zeroMean) (strict upper matrix)
//ONLY WORKS IF N DIVISBLE BY 4
//does not performe as it should. Still error (because of wrong SED kernel)
namespace updateGradient_zeroMeanv6_d2 {
    inline void updateGradient_zeroMean(const double* P, double* Y, int N, int out_dim, double* dY, 
                                double* uY, double* gains, const double momentum, const double eta) {

        // Compute the squared Euclidean distance matrix
        // Compute Q-matrix and normalization sum
        double* Q = static_cast<double *>(aligned_alloc(32, N * N * sizeof(double)));  //use aligned_alloc for intel intrinsics!
        if(Q == NULL) { printf("Memory allocation failed!\n"); exit(1); }
        
        /****SED****/
        //int cnt = 0;
        __m256d sum_Q_vec = _mm256_setzero_pd(); 
        __m256d Q_vec;

        const int b = 16; // block size for cache
        const int rbi = 4; // block size for registers, in the following unrolling, assume rbi = 4, o/w it won't work
        const int rbj = 16; // block size for registers
        
        const double* Xi = Y;
        for(int i = 0; i < N - b + 1; i += b, Xi += b * out_dim) {
            const double* Xj = Y + i * out_dim;
            for(int j = i; j < N - b + 1; j += b, Xj += b * out_dim) {
                const double* Xii = Xi;
                for(int ii = i; ii < i + b - rbi + 1; ii += rbi, Xii += rbi * out_dim) {
                    const double* Xjj = Xj;
                    
                    __m256d x01 = _mm256_load_pd(Xii);
                    __m256d x23 = _mm256_load_pd(Xii + 4);
                    __m256d xvecd0 = _mm256_unpacklo_pd(x01, x23); // dim 0 
                    __m256d xvecd1 = _mm256_unpackhi_pd(x01, x23); // dim 1
                    xvecd0 = _mm256_permute4x64_pd(xvecd0, 0b11011000);
                    xvecd1 = _mm256_permute4x64_pd(xvecd1, 0b11011000);

                    for(int jj = j; jj < j + b; ++jj, Xjj += out_dim) {
                        if(ii == jj) {
                            Q[ii * N + jj] = 1.0;    
                            continue;
                        }

                        // load y
                        __m256d yvecd0 = _mm256_set1_pd(Xjj[0]);
                        __m256d yvecd1 = _mm256_set1_pd(Xjj[1]);

                        __m256d xyd0  = _mm256_sub_pd(xvecd0, yvecd0);
                        __m256d xyd1  = _mm256_sub_pd(xvecd1, yvecd1);

                        //xyd0 = _mm256_mul_pd(xyd0, xyd0);
                        xyd1 = _mm256_mul_pd(xyd1, xyd1);

                        __m256d xy = _mm256_fmadd_pd(xyd0, xyd0, xyd1);

                        xy = _mm256_add_pd(one_vec, xy);
                        xy = _mm256_div_pd(one_vec, xy);       //xy = 1.0/(1.0 + xy)

                        sum_Q_vec = _mm256_add_pd(sum_Q_vec, xy);
                        //cnt += 8;

                        const int symm_base = jj * N + ii;
                        _mm256_store_pd(Q + symm_base, xy);

                        // int base = ii * N + jj;
                        // Q[base] = Q[symm_base    ]; base += N;
                        // Q[base] = Q[symm_base + 1]; base += N;
                        // Q[base] = Q[symm_base + 2]; base += N;
                        // Q[base] = Q[symm_base + 3];
                    }
                }
            }
        }
        /****SED****/


        sum_Q_vec = _mm256_hadd_pd(sum_Q_vec, sum_Q_vec);

        //printf("v4 [%.3f, %.3f, %.3f, %.3f] (%f, %d) \n", Q[N-4], Q[N-3], Q[N-2], Q[N-1], 2.0*sum_Q, cnt);
        
        const double sum_Q_inv = 0.5 / (sum_Q_vec[0] + sum_Q_vec[2]);
        const __m256d sum_Q_inv_vec = _mm256_set1_pd(sum_Q_inv);

        __m256d P_vec, dY_vec1, dY_vec2, dY_vec1_sum, dY_vec2_sum, Y_nD_vec1, Y_nD_vec2, Y_mD_vec1, Y_mD_vec2, Y_mD_vec3, Y_mD_vec4, mult_vec;

        double dY_temp[2];


        int m;
        int nD = 0;
        for(int n = 0; n < N; ++n) {

            dY_temp[0] = 0.0;
            dY_temp[1] = 0.0;
            dY_vec1_sum = _mm256_setzero_pd();
            dY_vec2_sum = _mm256_setzero_pd();
            Y_nD_vec1  = _mm256_broadcast_sd(Y + n*2);
            Y_nD_vec2  = _mm256_broadcast_sd(Y + n*2 + 1);

            //m smaller than n
            for(m = 0; m < n - 2; m += 4) {
                P_vec = _mm256_set_pd(P[(m+3)*N + n], P[(m+2)*N + n], P[(m+1)*N + n], P[(m  )*N + n]);
                Q_vec = _mm256_set_pd(Q[(m+3)*N + n], Q[(m+2)*N + n], Q[(m+1)*N + n], Q[(m  )*N + n]);
                
                mult_vec  = _mm256_mul_pd(_mm256_fmsub_pd(Q_vec, sum_Q_inv_vec, P_vec), Q_vec); //actually this is -mult

                Y_mD_vec1 = _mm256_load_pd(Y + m*2);
                Y_mD_vec2 = _mm256_load_pd(Y + m*2 + 4);

                Y_mD_vec3 = _mm256_unpacklo_pd(Y_mD_vec1, Y_mD_vec2);
                Y_mD_vec4 = _mm256_unpackhi_pd(Y_mD_vec1, Y_mD_vec2);

                Y_mD_vec1 = _mm256_permute4x64_pd(Y_mD_vec3, 0b11011000);
                Y_mD_vec2 = _mm256_permute4x64_pd(Y_mD_vec4, 0b11011000);

                dY_vec1 = _mm256_sub_pd(Y_mD_vec1, Y_nD_vec1);      //extra inverted because of -mult
                dY_vec2 = _mm256_sub_pd(Y_mD_vec2, Y_nD_vec2);

                dY_vec1_sum = _mm256_fmadd_pd(mult_vec, dY_vec1, dY_vec1_sum);
                dY_vec2_sum = _mm256_fmadd_pd(mult_vec, dY_vec2, dY_vec2_sum);
            }

            //special cases [n - 2, n + 4] cases
            if(m == n){
                P_vec = _mm256_load_pd(P + n*N + m);
                Q_vec = _mm256_load_pd(Q + n*N + m);

                mult_vec  = _mm256_mul_pd(_mm256_fmsub_pd(Q_vec, sum_Q_inv_vec, P_vec), Q_vec); //actually this is -mult

                Y_mD_vec1 = _mm256_load_pd(Y + m*2);
                Y_mD_vec2 = _mm256_load_pd(Y + m*2 + 4);

                Y_mD_vec3 = _mm256_unpacklo_pd(Y_mD_vec1, Y_mD_vec2);
                Y_mD_vec4 = _mm256_unpackhi_pd(Y_mD_vec1, Y_mD_vec2);

                Y_mD_vec1 = _mm256_permute4x64_pd(Y_mD_vec3, 0b11011000);
                Y_mD_vec2 = _mm256_permute4x64_pd(Y_mD_vec4, 0b11011000);

                dY_vec1 = _mm256_sub_pd(Y_mD_vec1, Y_nD_vec1);      //extra inverted because of -mult
                dY_vec2 = _mm256_sub_pd(Y_mD_vec2, Y_nD_vec2);

                dY_vec1_sum = _mm256_fmadd_pd(mult_vec, dY_vec1, dY_vec1_sum);
                dY_vec2_sum = _mm256_fmadd_pd(mult_vec, dY_vec2, dY_vec2_sum);

                m += 4;
            }
            else if(m == n - 1){                                    //actually 0.0 here
                P_vec = _mm256_set_pd(P[(n)*N + m+3], P[(n)*N + m+2], P[(n)*N + m+1], P[(n-1)*N + m+1]);
                Q_vec = _mm256_set_pd(Q[(n)*N + m+3], Q[(n)*N + m+2], Q[(n)*N + m+1], Q[(n-1)*N + m+1]);

                mult_vec  = _mm256_mul_pd(_mm256_fmsub_pd(Q_vec, sum_Q_inv_vec, P_vec), Q_vec); //actually this is -mult

                Y_mD_vec1 = _mm256_load_pd(Y + m*2);
                Y_mD_vec2 = _mm256_load_pd(Y + m*2 + 4);

                Y_mD_vec3 = _mm256_unpacklo_pd(Y_mD_vec1, Y_mD_vec2);
                Y_mD_vec4 = _mm256_unpackhi_pd(Y_mD_vec1, Y_mD_vec2);

                Y_mD_vec1 = _mm256_permute4x64_pd(Y_mD_vec3, 0b11011000);
                Y_mD_vec2 = _mm256_permute4x64_pd(Y_mD_vec4, 0b11011000);

                dY_vec1 = _mm256_sub_pd(Y_mD_vec1, Y_nD_vec1);      //extra inverted because of -mult
                dY_vec2 = _mm256_sub_pd(Y_mD_vec2, Y_nD_vec2);

                dY_vec1_sum = _mm256_fmadd_pd(mult_vec, dY_vec1, dY_vec1_sum);
                dY_vec2_sum = _mm256_fmadd_pd(mult_vec, dY_vec2, dY_vec2_sum);

                m += 4;
            }else if(m == n - 2){
                P_vec = _mm256_set_pd(P[(n)*N + m+3], P[(n)*N + m+2], P[(n-1)*N + m+2], P[(n-2)*N + m+2]);
                Q_vec = _mm256_set_pd(Q[(n)*N + m+3], Q[(n)*N + m+2], Q[(n-1)*N + m+2], Q[(n-2)*N + m+2]);

                mult_vec  = _mm256_mul_pd(_mm256_fmsub_pd(Q_vec, sum_Q_inv_vec, P_vec), Q_vec); //actually this is -mult

                Y_mD_vec1 = _mm256_load_pd(Y + m*2);
                Y_mD_vec2 = _mm256_load_pd(Y + m*2 + 4);

                Y_mD_vec3 = _mm256_unpacklo_pd(Y_mD_vec1, Y_mD_vec2);
                Y_mD_vec4 = _mm256_unpackhi_pd(Y_mD_vec1, Y_mD_vec2);

                Y_mD_vec1 = _mm256_permute4x64_pd(Y_mD_vec3, 0b11011000);
                Y_mD_vec2 = _mm256_permute4x64_pd(Y_mD_vec4, 0b11011000);

                dY_vec1 = _mm256_sub_pd(Y_mD_vec1, Y_nD_vec1);      //extra inverted because of -mult
                dY_vec2 = _mm256_sub_pd(Y_mD_vec2, Y_nD_vec2);

                dY_vec1_sum = _mm256_fmadd_pd(mult_vec, dY_vec1, dY_vec1_sum);
                dY_vec2_sum = _mm256_fmadd_pd(mult_vec, dY_vec2, dY_vec2_sum);

                m += 4;
            }
                
            

            //m [n, N - 3]
            for(; m < N - 3; m += 4) {
                P_vec = _mm256_load_pd(P + n*N + m);
                Q_vec = _mm256_load_pd(Q + n*N + m);
                
                mult_vec  = _mm256_mul_pd(_mm256_fmsub_pd(Q_vec, sum_Q_inv_vec, P_vec), Q_vec); //actually this is -mult

                Y_mD_vec1 = _mm256_load_pd(Y + m*2);
                Y_mD_vec2 = _mm256_load_pd(Y + m*2 + 4);

                Y_mD_vec3 = _mm256_unpacklo_pd(Y_mD_vec1, Y_mD_vec2);
                Y_mD_vec4 = _mm256_unpackhi_pd(Y_mD_vec1, Y_mD_vec2);

                Y_mD_vec1 = _mm256_permute4x64_pd(Y_mD_vec3, 0b11011000);
                Y_mD_vec2 = _mm256_permute4x64_pd(Y_mD_vec4, 0b11011000);

                dY_vec1 = _mm256_sub_pd(Y_mD_vec1, Y_nD_vec1);      //extra inverted because of -mult
                dY_vec2 = _mm256_sub_pd(Y_mD_vec2, Y_nD_vec2);

                dY_vec1_sum = _mm256_fmadd_pd(mult_vec, dY_vec1, dY_vec1_sum);
                dY_vec2_sum = _mm256_fmadd_pd(mult_vec, dY_vec2, dY_vec2_sum);

            }
            for(; m < N; ++m) {
                if(n != m) {
                    double mult = (P[n*N + m] - (Q[n*N + m] * sum_Q_inv)) * Q[n*N + m];

                    dY_temp[0] += (Y[n*2    ] - Y[m*2    ]) * mult;
                    dY_temp[1] += (Y[n*2 + 1] - Y[m*2 + 1]) * mult;
                }
            }
            
            dY_vec1_sum = _mm256_hadd_pd(dY_vec1_sum, dY_vec1_sum);
            dY_vec2_sum = _mm256_hadd_pd(dY_vec2_sum, dY_vec2_sum);

            dY_temp[0] += dY_vec1_sum[0] + dY_vec1_sum[2];
            dY_temp[1] += dY_vec2_sum[0] + dY_vec2_sum[2];
            
            dY[nD    ] = dY_temp[0];
            dY[nD + 1] = dY_temp[1];
            

            nD += 2;
        }
        

        // Free memory
        free(Q);  Q = NULL;


        const __m256d eta_vec  = _mm256_set1_pd(eta);
        const __m256d mom_vec  = _mm256_set1_pd(momentum);
        __m256d gain_vec, dY_vec, Y_vec, uY_vec, sign_dY_vec, sign_uY_vec, gain_cmp, eta_dY_vec;

        __m256d mean_vec = _mm256_setzero_pd();

        double gains1, gains2, dY1, dY2, uY1, uY2;
        
        int i;
        const int ND = N * out_dim;
        for(i = 0; i < ND - 3; i += 4){
            gain_vec = _mm256_load_pd(gains + i);
            dY_vec   = _mm256_load_pd(dY + i);
            uY_vec   = _mm256_load_pd(uY + i);
            Y_vec    = _mm256_load_pd(Y + i);

            eta_dY_vec = _mm256_mul_pd(eta_vec, dY_vec);

            gain_cmp = _mm256_cmp_pd(sign(dY_vec) , sign(uY_vec), _CMP_NEQ_OQ);
            gain_vec = _mm256_blendv_pd(_mm256_mul_pd(gain_vec, z8_vec), _mm256_add_pd(gain_vec, z2_vec), gain_cmp);
            gain_cmp = _mm256_cmp_pd(zz1_vec, gain_vec, _CMP_GT_OQ);
            gain_vec = _mm256_blendv_pd(gain_vec, zz1_vec, gain_cmp);

            _mm256_store_pd(gains + i, gain_vec);

            uY_vec = _mm256_fmsub_pd(mom_vec, uY_vec, _mm256_mul_pd(gain_vec, eta_dY_vec));
            
            _mm256_store_pd(uY + i, uY_vec);
            
            Y_vec  = _mm256_add_pd(Y_vec, uY_vec);

            mean_vec = _mm256_add_pd(mean_vec, Y_vec);

            _mm256_store_pd(Y + i, Y_vec);
        }
        for(; i < ND; i += 2){
            gains1 = gains[i];
            gains2 = gains[i+1];

            dY1 = dY[i];
            dY2 = dY[i+1];

            uY1 = uY[i];
            uY2 = uY[i+1];

            gains1 = (sign(dY1) != sign(uY1)) ? (gains1 + 0.2) : (gains1 * 0.8);
            gains2 = (sign(dY2) != sign(uY2)) ? (gains2 + 0.2) : (gains2 * 0.8);
            gains[i]   = (gains1 < 0.01) ? 0.01 : gains1;
            gains[i+1] = (gains2 < 0.01) ? 0.01 : gains2;

            uY[i]   = momentum * uY1 - eta * gains[i] * dY1;
            uY[i+1] = momentum * uY2 - eta * gains[i+1] * dY2;

            Y[i] = Y[i] + uY[i];
            Y[i+1] = Y[i+1] + uY[i+1];

            mean_vec = _mm256_add_pd(mean_vec, _mm256_set_pd(0.0, 0.0, Y[i+1], Y[i]));
        } 

        /****zeroMean****/
        mean_vec = _mm256_mul_pd(mean_vec, _mm256_set1_pd(1.0/double(N)));
        mean_vec = _mm256_permute4x64_pd(mean_vec, 0b11011000);
        mean_vec = _mm256_hadd_pd(mean_vec, mean_vec);
        mean_vec = _mm256_permute4x64_pd(mean_vec, 0b11011000);

        
        for(i = 0; i < ND - 3; i += 4){
            Y_vec = _mm256_load_pd(Y + i);
            Y_vec = _mm256_sub_pd(Y_vec, mean_vec);
            _mm256_store_pd(Y + i, Y_vec);
        }
        for(; i < ND; i += 2){
            Y[i]   -= mean_vec[0];
            Y[i+1] -= mean_vec[1];
        }
        /****zeroMean****/
    }
}


//unpack all optimized kernels and optimize (out_dim=2) (own version of zeroMean) (strict upper matrix)
//ONLY WORKS IF N DIVISBLE BY 4
//does not performe as it should. Still error (because of wrong SED kernel)
namespace updateGradient_zeroMeanv7_d2 {
    inline void updateGradient_zeroMean(const double* P, double* Y, int N, int out_dim, double* dY, 
                                double* uY, double* gains, const double momentum, const double eta) {

        // Compute the squared Euclidean distance matrix
        // Compute Q-matrix and normalization sum
        double* Q = static_cast<double *>(aligned_alloc(32, N * N * sizeof(double)));  //use aligned_alloc for intel intrinsics!
        if(Q == NULL) { printf("Memory allocation failed!\n"); exit(1); }
        
        /****SED****/
        double sum_Q = 0.0;

        const int b = 32; // block size for cache
        double buf[b * b] __attribute__ ((aligned (32)));

        const double* Xi = Y;
        for(int i = 0; i < N - b + 1; i += b, Xi += b * 2) {
            const double* Xj = Y + i * 2;
            for(int j = i; j < N - b + 1; j += b, Xj += b * 2) {
                const double* Xii = Xi;
                if(i == j) { // diagonal block
                    for(int ii = i; ii < i + b; ii++, Xii += 2) {
                        const double* Xjj = Xj;
                        double xii0 = Xii[0], xii1 = Xii[1];
                        int base = ii * N;
                        for(int jj = j; jj < j + b; jj++, Xjj += 2) {
                                if(ii == jj) {
                                    // DD[base + jj] = 0.0;
                                    Q[base + jj] = 1.0;
                                    continue;
                                }

                                double tmp1 = xii0 - Xjj[0];
                                double tmp2 = xii1 - Xjj[1];
                                double dist = tmp1 * tmp1 + tmp2 * tmp2;

                                dist = 1.0 / (1.0 + dist);
                                
                                Q[base + jj] = dist;
                                sum_Q += dist;
                        }
                    }
                    continue;
               }

                for(int ii = i; ii < i + b; ii++, Xii += 2) {
                    const double* Xjj = Xj;
                    double xii0 = Xii[0], xii1 = Xii[1];
                    int base = ii * N;
                    int shift = ii - i;
                    int buf_base = (ii - i) * b;
                    for(int jj = j; jj < j + b; jj++, Xjj += 2) {
                        // for dim = 2
                        double tmp1 = xii0 - Xjj[0];
                        double tmp2 = xii1 - Xjj[1];
                        double dist = tmp1 * tmp1 + tmp2 * tmp2;

                        dist = 1.0 / (1.0 + dist);
                        
                        Q[base + jj] = dist;
                        sum_Q += 2.0*dist;
                        
                        //buf[(jj-j) * b + shift] = dist;
                    }
                }

                // copy the buffer back to DD
                // for(int jj = j; jj < j + b; jj++) { 
                //     int base = jj * N;
                //     int buf_base = (jj - j) * b;
                //     for(int ii = i; ii < i + b; ii++) {
                //         Q[base + ii] = buf[buf_base + ii - i];
                //     }
                // }
           }
        }
        /****SED****/
        
        const double sum_Q_inv = 1.0 / sum_Q;
        const __m256d sum_Q_inv_vec = _mm256_set1_pd(sum_Q_inv);

        __m256d Q_vec, P_vec, dY_vec1, dY_vec2, dY_vec1_sum, dY_vec2_sum, Y_nD_vec1, Y_nD_vec2, Y_mD_vec1, Y_mD_vec2, Y_mD_vec3, Y_mD_vec4, mult_vec;

        double dY_temp[2];


        int m;
        int nD = 0;
        for(int n = 0; n < N; ++n) {

            dY_temp[0] = 0.0;
            dY_temp[1] = 0.0;
            dY_vec1_sum = _mm256_setzero_pd();
            dY_vec2_sum = _mm256_setzero_pd();
            Y_nD_vec1  = _mm256_broadcast_sd(Y + n*2);
            Y_nD_vec2  = _mm256_broadcast_sd(Y + n*2 + 1);

            //m smaller than n
            for(m = 0; m < n - 2; m += 4) {
                P_vec = _mm256_set_pd(P[(m+3)*N + n], P[(m+2)*N + n], P[(m+1)*N + n], P[(m  )*N + n]);
                Q_vec = _mm256_set_pd(Q[(m+3)*N + n], Q[(m+2)*N + n], Q[(m+1)*N + n], Q[(m  )*N + n]);
                
                mult_vec  = _mm256_mul_pd(_mm256_fmsub_pd(Q_vec, sum_Q_inv_vec, P_vec), Q_vec); //actually this is -mult

                Y_mD_vec1 = _mm256_load_pd(Y + m*2);
                Y_mD_vec2 = _mm256_load_pd(Y + m*2 + 4);

                Y_mD_vec3 = _mm256_unpacklo_pd(Y_mD_vec1, Y_mD_vec2);
                Y_mD_vec4 = _mm256_unpackhi_pd(Y_mD_vec1, Y_mD_vec2);

                Y_mD_vec1 = _mm256_permute4x64_pd(Y_mD_vec3, 0b11011000);
                Y_mD_vec2 = _mm256_permute4x64_pd(Y_mD_vec4, 0b11011000);

                dY_vec1 = _mm256_sub_pd(Y_mD_vec1, Y_nD_vec1);      //extra inverted because of -mult
                dY_vec2 = _mm256_sub_pd(Y_mD_vec2, Y_nD_vec2);

                dY_vec1_sum = _mm256_fmadd_pd(mult_vec, dY_vec1, dY_vec1_sum);
                dY_vec2_sum = _mm256_fmadd_pd(mult_vec, dY_vec2, dY_vec2_sum);
            }

            //special cases [n - 2, n + 4] cases
            if(m == n){
                P_vec = _mm256_load_pd(P + n*N + m);
                Q_vec = _mm256_load_pd(Q + n*N + m);

                mult_vec  = _mm256_mul_pd(_mm256_fmsub_pd(Q_vec, sum_Q_inv_vec, P_vec), Q_vec); //actually this is -mult

                Y_mD_vec1 = _mm256_load_pd(Y + m*2);
                Y_mD_vec2 = _mm256_load_pd(Y + m*2 + 4);

                Y_mD_vec3 = _mm256_unpacklo_pd(Y_mD_vec1, Y_mD_vec2);
                Y_mD_vec4 = _mm256_unpackhi_pd(Y_mD_vec1, Y_mD_vec2);

                Y_mD_vec1 = _mm256_permute4x64_pd(Y_mD_vec3, 0b11011000);
                Y_mD_vec2 = _mm256_permute4x64_pd(Y_mD_vec4, 0b11011000);

                dY_vec1 = _mm256_sub_pd(Y_mD_vec1, Y_nD_vec1);      //extra inverted because of -mult
                dY_vec2 = _mm256_sub_pd(Y_mD_vec2, Y_nD_vec2);

                dY_vec1_sum = _mm256_fmadd_pd(mult_vec, dY_vec1, dY_vec1_sum);
                dY_vec2_sum = _mm256_fmadd_pd(mult_vec, dY_vec2, dY_vec2_sum);

                m += 4;
            }
            else if(m == n - 1){                                    //actually 0.0 here
                P_vec = _mm256_set_pd(P[(n)*N + m+3], P[(n)*N + m+2], P[(n)*N + m+1], P[(n-1)*N + m+1]);
                Q_vec = _mm256_set_pd(Q[(n)*N + m+3], Q[(n)*N + m+2], Q[(n)*N + m+1], Q[(n-1)*N + m+1]);

                mult_vec  = _mm256_mul_pd(_mm256_fmsub_pd(Q_vec, sum_Q_inv_vec, P_vec), Q_vec); //actually this is -mult

                Y_mD_vec1 = _mm256_load_pd(Y + m*2);
                Y_mD_vec2 = _mm256_load_pd(Y + m*2 + 4);

                Y_mD_vec3 = _mm256_unpacklo_pd(Y_mD_vec1, Y_mD_vec2);
                Y_mD_vec4 = _mm256_unpackhi_pd(Y_mD_vec1, Y_mD_vec2);

                Y_mD_vec1 = _mm256_permute4x64_pd(Y_mD_vec3, 0b11011000);
                Y_mD_vec2 = _mm256_permute4x64_pd(Y_mD_vec4, 0b11011000);

                dY_vec1 = _mm256_sub_pd(Y_mD_vec1, Y_nD_vec1);      //extra inverted because of -mult
                dY_vec2 = _mm256_sub_pd(Y_mD_vec2, Y_nD_vec2);

                dY_vec1_sum = _mm256_fmadd_pd(mult_vec, dY_vec1, dY_vec1_sum);
                dY_vec2_sum = _mm256_fmadd_pd(mult_vec, dY_vec2, dY_vec2_sum);

                m += 4;
            }else if(m == n - 2){
                P_vec = _mm256_set_pd(P[(n)*N + m+3], P[(n)*N + m+2], P[(n-1)*N + m+2], P[(n-2)*N + m+2]);
                Q_vec = _mm256_set_pd(Q[(n)*N + m+3], Q[(n)*N + m+2], Q[(n-1)*N + m+2], Q[(n-2)*N + m+2]);

                mult_vec  = _mm256_mul_pd(_mm256_fmsub_pd(Q_vec, sum_Q_inv_vec, P_vec), Q_vec); //actually this is -mult

                Y_mD_vec1 = _mm256_load_pd(Y + m*2);
                Y_mD_vec2 = _mm256_load_pd(Y + m*2 + 4);

                Y_mD_vec3 = _mm256_unpacklo_pd(Y_mD_vec1, Y_mD_vec2);
                Y_mD_vec4 = _mm256_unpackhi_pd(Y_mD_vec1, Y_mD_vec2);

                Y_mD_vec1 = _mm256_permute4x64_pd(Y_mD_vec3, 0b11011000);
                Y_mD_vec2 = _mm256_permute4x64_pd(Y_mD_vec4, 0b11011000);

                dY_vec1 = _mm256_sub_pd(Y_mD_vec1, Y_nD_vec1);      //extra inverted because of -mult
                dY_vec2 = _mm256_sub_pd(Y_mD_vec2, Y_nD_vec2);

                dY_vec1_sum = _mm256_fmadd_pd(mult_vec, dY_vec1, dY_vec1_sum);
                dY_vec2_sum = _mm256_fmadd_pd(mult_vec, dY_vec2, dY_vec2_sum);

                m += 4;
            }
                
            

            //m [n, N - 3]
            for(; m < N - 3; m += 4) {
                P_vec = _mm256_load_pd(P + n*N + m);
                Q_vec = _mm256_load_pd(Q + n*N + m);
                
                mult_vec  = _mm256_mul_pd(_mm256_fmsub_pd(Q_vec, sum_Q_inv_vec, P_vec), Q_vec); //actually this is -mult

                Y_mD_vec1 = _mm256_load_pd(Y + m*2);
                Y_mD_vec2 = _mm256_load_pd(Y + m*2 + 4);

                Y_mD_vec3 = _mm256_unpacklo_pd(Y_mD_vec1, Y_mD_vec2);
                Y_mD_vec4 = _mm256_unpackhi_pd(Y_mD_vec1, Y_mD_vec2);

                Y_mD_vec1 = _mm256_permute4x64_pd(Y_mD_vec3, 0b11011000);
                Y_mD_vec2 = _mm256_permute4x64_pd(Y_mD_vec4, 0b11011000);

                dY_vec1 = _mm256_sub_pd(Y_mD_vec1, Y_nD_vec1);      //extra inverted because of -mult
                dY_vec2 = _mm256_sub_pd(Y_mD_vec2, Y_nD_vec2);

                dY_vec1_sum = _mm256_fmadd_pd(mult_vec, dY_vec1, dY_vec1_sum);
                dY_vec2_sum = _mm256_fmadd_pd(mult_vec, dY_vec2, dY_vec2_sum);

            }
            for(; m < N; ++m) {
                if(n != m) {
                    double mult = (P[n*N + m] - (Q[n*N + m] * sum_Q_inv)) * Q[n*N + m];

                    dY_temp[0] += (Y[n*2    ] - Y[m*2    ]) * mult;
                    dY_temp[1] += (Y[n*2 + 1] - Y[m*2 + 1]) * mult;
                }
            }
            
            dY_vec1_sum = _mm256_hadd_pd(dY_vec1_sum, dY_vec1_sum);
            dY_vec2_sum = _mm256_hadd_pd(dY_vec2_sum, dY_vec2_sum);

            dY_temp[0] += dY_vec1_sum[0] + dY_vec1_sum[2];
            dY_temp[1] += dY_vec2_sum[0] + dY_vec2_sum[2];
            
            dY[nD    ] = dY_temp[0];
            dY[nD + 1] = dY_temp[1];
            

            nD += 2;
        }
        

        // Free memory
        free(Q);  Q = NULL;


        const __m256d eta_vec  = _mm256_set1_pd(eta);
        const __m256d mom_vec  = _mm256_set1_pd(momentum);
        __m256d gain_vec, dY_vec, Y_vec, uY_vec, sign_dY_vec, sign_uY_vec, gain_cmp, eta_dY_vec;

        __m256d mean_vec = _mm256_setzero_pd();

        double gains1, gains2, dY1, dY2, uY1, uY2;
        
        int i;
        const int ND = N * out_dim;
        for(i = 0; i < ND - 3; i += 4){
            gain_vec = _mm256_load_pd(gains + i);
            dY_vec   = _mm256_load_pd(dY + i);
            uY_vec   = _mm256_load_pd(uY + i);
            Y_vec    = _mm256_load_pd(Y + i);

            eta_dY_vec = _mm256_mul_pd(eta_vec, dY_vec);

            gain_cmp = _mm256_cmp_pd(sign(dY_vec) , sign(uY_vec), _CMP_NEQ_OQ);
            gain_vec = _mm256_blendv_pd(_mm256_mul_pd(gain_vec, z8_vec), _mm256_add_pd(gain_vec, z2_vec), gain_cmp);
            gain_cmp = _mm256_cmp_pd(zz1_vec, gain_vec, _CMP_GT_OQ);
            gain_vec = _mm256_blendv_pd(gain_vec, zz1_vec, gain_cmp);

            _mm256_store_pd(gains + i, gain_vec);

            uY_vec = _mm256_fmsub_pd(mom_vec, uY_vec, _mm256_mul_pd(gain_vec, eta_dY_vec));
            
            _mm256_store_pd(uY + i, uY_vec);
            
            Y_vec  = _mm256_add_pd(Y_vec, uY_vec);

            mean_vec = _mm256_add_pd(mean_vec, Y_vec);

            _mm256_store_pd(Y + i, Y_vec);
        }
        for(; i < ND; i += 2){
            gains1 = gains[i];
            gains2 = gains[i+1];

            dY1 = dY[i];
            dY2 = dY[i+1];

            uY1 = uY[i];
            uY2 = uY[i+1];

            gains1 = (sign(dY1) != sign(uY1)) ? (gains1 + 0.2) : (gains1 * 0.8);
            gains2 = (sign(dY2) != sign(uY2)) ? (gains2 + 0.2) : (gains2 * 0.8);
            gains[i]   = (gains1 < 0.01) ? 0.01 : gains1;
            gains[i+1] = (gains2 < 0.01) ? 0.01 : gains2;

            uY[i]   = momentum * uY1 - eta * gains[i] * dY1;
            uY[i+1] = momentum * uY2 - eta * gains[i+1] * dY2;

            Y[i] = Y[i] + uY[i];
            Y[i+1] = Y[i+1] + uY[i+1];

            mean_vec = _mm256_add_pd(mean_vec, _mm256_set_pd(0.0, 0.0, Y[i+1], Y[i]));
        } 

        /****zeroMean****/
        mean_vec = _mm256_mul_pd(mean_vec, _mm256_set1_pd(1.0/double(N)));
        mean_vec = _mm256_permute4x64_pd(mean_vec, 0b11011000);
        mean_vec = _mm256_hadd_pd(mean_vec, mean_vec);
        mean_vec = _mm256_permute4x64_pd(mean_vec, 0b11011000);

        
        for(i = 0; i < ND - 3; i += 4){
            Y_vec = _mm256_load_pd(Y + i);
            Y_vec = _mm256_sub_pd(Y_vec, mean_vec);
            _mm256_store_pd(Y + i, Y_vec);
        }
        for(; i < ND; i += 2){
            Y[i]   -= mean_vec[0];
            Y[i+1] -= mean_vec[1];
        }
        /****zeroMean****/
    }
}
