#pragma once
#include "computeSED.hpp"

#include <immintrin.h>

const __m256d zero_vec = _mm256_setzero_pd();
const __m256d one_vec  = _mm256_set1_pd(1.0);
const __m256d none_vec = _mm256_set1_pd(-1.0);

/// return the sign of the a double, 0 if x = 0, x/abs(x) otherwise
static inline double sign(double x) {
    return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0));
}

static inline __m256d sign(__m256d x) {
    
    __m256d out_vec, zero_cmp, gt_cmp;
    zero_cmp = _mm256_cmp_pd(x, zero_vec, _CMP_EQ_OQ);
    gt_cmp   = _mm256_cmp_pd(x, zero_vec, _CMP_GT_OQ);
    
    out_vec = _mm256_blendv_pd(none_vec, one_vec, gt_cmp);
    return _mm256_blendv_pd(out_vec, zero_vec, zero_cmp);
}

//why check for 0.0 even if it is very uncommon?
static inline __m256d sign_fast(__m256d x) {
    return _mm256_cmp_pd(x, zero_vec, _CMP_GT_OQ);
}

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

namespace updateGradientv1 {
    
    inline void updateGradient(const double* P, double* Y, int N, int out_dim, double* dY, 
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
    }
}


namespace updateGradientv2 {
    
    inline void updateGradient(const double* P, double* Y, int N, int out_dim, double* dY, 
                                double* uY, double* gains, const double momentum, const double eta) {

        // Make sure the current gradient contains zeros    (at every iteration!)
        for(int i = 0; i < N * out_dim; ++i) dY[i] = 0.0;

        // Compute the squared Euclidean distance matrix
        double* DD = (double*) malloc(N * N * sizeof(double));
        if(DD == NULL) { printf("Memory allocation failed!\n"); exit(1); }
        computeSEDv1::computeSquaredEuclideanDistance(Y, N, out_dim, DD);

        // Compute Q-matrix and normalization sum
        double* Q = (double*) malloc(N * N * sizeof(double));
        if(Q == NULL) { printf("Memory allocation failed!\n"); exit(1); }

        double sum_Q = 0.0;


        const int N2 = N*N;
        for(int n = 0; n < N2; ++n) {
            Q[n] = 1.0 / (1.0 + DD[n]);
            sum_Q += Q[n];
        }
        const double sum_Q_inv = 1.0 / (sum_Q - double(N));
        


        // Perform the computation of the gradient
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

        /*
        for(int i = 0; i < N * out_dim; ++i){
            gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + 0.2) : (gains[i] * 0.8);
            if(gains[i] < 0.01) gains[i] = 0.01;

            uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
            Y[i] = Y[i] + uY[i];
        }
        */

        
        double gains_;
        for(int i = 0; i < N * out_dim; ++i){
            gains_ = gains[i];
            gains_ = (sign(dY[i]) != sign(uY[i])) ? (gains_ + 0.2) : (gains_ * 0.8);
            gains[i] = gains_ < 0.01 ? 0.01 : gains_;

            uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
            Y[i] = Y[i] + uY[i];
        }
        
    }
}


namespace updateGradientv2_2_outdim {
    
    inline void updateGradient(const double* P, double* Y, int N, int out_dim, double* dY, 
                                double* uY, double* gains, const double momentum, const double eta) {

        // Compute the squared Euclidean distance matrix
        double* DD = (double*) malloc(N * N * sizeof(double));
        if(DD == NULL) { printf("Memory allocation failed!\n"); exit(1); }
        computeSEDv1::computeSquaredEuclideanDistance(Y, N, out_dim, DD);

        // Compute Q-matrix and normalization sum
        double* Q = (double*) malloc(N * N * sizeof(double));
        if(Q == NULL) { printf("Memory allocation failed!\n"); exit(1); }

        double sum_Q = 0.0;


        const int N2 = N*N;
        for(int n = 0; n < N2; ++n) {
            Q[n] = 1.0 / (1.0 + DD[n]);
            sum_Q += Q[n];
        }

        // Perform the computation of the gradient
        const double sum_Q_inv = 1.0 / (sum_Q - double(N));
        double dY0, dY1;

        int nN = 0;
        int m, mD;
        int nD = 0;
        for(int n = 0; n < N; ++n) {
            mD = 0;
            dY0 = 0.0;
            dY1 = 0.0;

            for(m = 0; m < N; m += 4) {
                
                double mult0 = (P[nN + m] - (Q[nN + m] * sum_Q_inv)) * Q[nN + m];
                double mult1 = (P[nN + m + 1] - (Q[nN + m + 1] * sum_Q_inv)) * Q[nN + m + 1];
                double mult2 = (P[nN + m + 2] - (Q[nN + m + 2] * sum_Q_inv)) * Q[nN + m + 2];
                double mult3 = (P[nN + m + 3] - (Q[nN + m + 3] * sum_Q_inv)) * Q[nN + m + 3];

                dY0 += (Y[nD    ] - Y[mD    ]) * mult0;
                dY0 += (Y[nD    ] - Y[mD + 2    ]) * mult1;
                dY0 += (Y[nD    ] - Y[mD + 4    ]) * mult2;
                dY0 += (Y[nD    ] - Y[mD + 6    ]) * mult3;

                dY1 += (Y[nD + 1] - Y[mD + 1]) * mult0;
                dY1 += (Y[nD + 1] - Y[mD + 2 + 1]) * mult1;
                dY1 += (Y[nD + 1] - Y[mD + 4 + 1]) * mult2;
                dY1 += (Y[nD + 1] - Y[mD + 6 + 1]) * mult3;

                
                mD += 4*out_dim;
            }
            for(; m < N; ++m) {
                if(n != m) {
                    double mult = (P[nN + m] - (Q[nN + m] * sum_Q_inv)) * Q[nN + m];

                    dY0 += (Y[nD    ] - Y[mD    ]) * mult;
                    dY1 += (Y[nD + 1] - Y[mD + 1]) * mult;
                }
                mD += out_dim;
            }
            dY[nD    ] = dY0;
            dY[nD + 1] = dY1;
            

            nN += N;
            nD += out_dim;
        }
        

        // Free memory
        free(DD); DD = NULL;
        free(Q);  Q  = NULL;

        /*
        for(int i = 0; i < N * out_dim; ++i){
            gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + 0.2) : (gains[i] * 0.8);
            if(gains[i] < 0.01) gains[i] = 0.01;

            uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
            Y[i] = Y[i] + uY[i];
        }
        */

        
        double gains_;
        for(int i = 0; i < N * out_dim; ++i){
            gains_ = gains[i];
            gains_ = (sign(dY[i]) != sign(uY[i])) ? (gains_ + 0.2) : (gains_ * 0.8);
            gains[i] = gains_ < 0.01 ? 0.01 : gains_;

            uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
            Y[i] = Y[i] + uY[i];
        }
        
    }
}


namespace updateGradientv3_x_outdim {
    
    inline void updateGradient(const double* P, double* Y, int N, int out_dim, double* dY, 
                                double* uY, double* gains, const double momentum, const double eta) {
        
        // Make sure the current gradient contains zeros    (at every iteration!)
        int i = 0;
        for(; i < N * out_dim; i += 4){
            _mm256_store_pd(dY + i, zero_vec);
        }
        for(; i < N * out_dim; ++i) dY[i] = 0.0;
        
        // Compute the squared Euclidean distance matrix
        double* DD = static_cast<double *>(aligned_alloc(32, N * N * sizeof(double)));  //use aligned_alloc for intel intinsics!
        if(DD == NULL) { printf("Memory allocation failed!\n"); exit(1); }
        computeSEDv1::computeSquaredEuclideanDistance(Y, N, out_dim, DD);

        // Compute Q-matrix and normalization sum
        double* Q = static_cast<double *>(aligned_alloc(32, N * N * sizeof(double)));
        if(Q == NULL) { printf("Memory allocation failed!\n"); exit(1); }
        double sum_Q = 0.0;


        __m256d sum_Q_vec, Q_vec, DD_vec;
        
        const int N2 = N*N;
        for(i = 0; i < N2; i += 4) {
            DD_vec = _mm256_load_pd(DD + i);
            DD_vec = _mm256_add_pd(one_vec, DD_vec);
            Q_vec  = _mm256_div_pd(one_vec, DD_vec);
            
            _mm256_store_pd(Q + i, Q_vec);

            sum_Q_vec = _mm256_hadd_pd(Q_vec, Q_vec);
            sum_Q += ((double*)&sum_Q_vec)[0] + ((double*)&sum_Q_vec)[2];
        }
        for(; i < N2; ++i){
            Q[i] = 1.0 / (1.0 + DD[i]);
            sum_Q += Q[i];
        }
        
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


        const __m256d zz1_vec  = _mm256_set1_pd(0.01);
        const __m256d z8_vec   = _mm256_set1_pd(0.8);
        const __m256d z2_vec   = _mm256_set1_pd(0.2);
        const __m256d eta_vec  = _mm256_set1_pd(eta);
        const __m256d mom_vec  = _mm256_set1_pd(momentum);
        __m256d gain_vec, dY_vec, Y_vec, uY_vec, sign_dY_vec, sign_uY_vec, gain_cmp, eta_dY_vec;

        for(i = 0; i < N * out_dim; i += 4){
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
        for(; i < N * out_dim; ++i){
            gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + 0.2) : (gains[i] * 0.8);
            if(gains[i] < 0.01) gains[i] = 0.01;

            uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
            Y[i] = Y[i] + uY[i];
        } 
    }
}


namespace updateGradientv3_2_outdim {
    
    inline void updateGradient(const double* P, double* Y, int N, int out_dim, double* dY, 
                                double* uY, double* gains, const double momentum, const double eta) {
        
        // Compute the squared Euclidean distance matrix
        double* DD = static_cast<double *>(aligned_alloc(32, N * N * sizeof(double)));  //use aligned_alloc for intel intinsics!
        if(DD == NULL) { printf("Memory allocation failed!\n"); exit(1); }
        computeSEDv1::computeSquaredEuclideanDistance(Y, N, out_dim, DD);

        // Compute Q-matrix and normalization sum
        double* Q = static_cast<double *>(aligned_alloc(32, N * N * sizeof(double)));
        if(Q == NULL) { printf("Memory allocation failed!\n"); exit(1); }
        double sum_Q = 0.0;


        __m256d sum_Q_vec, Q_vec, DD_vec;
        
        int i;
        const int N2 = N*N;
        for(i = 0; i < N2; i += 4) {
            DD_vec = _mm256_load_pd(DD + i);
            DD_vec = _mm256_add_pd(one_vec, DD_vec);
            Q_vec  = _mm256_div_pd(one_vec, DD_vec);
            
            _mm256_store_pd(Q + i, Q_vec);

            sum_Q_vec = _mm256_hadd_pd(Q_vec, Q_vec);
            sum_Q += ((double*)&sum_Q_vec)[0] + ((double*)&sum_Q_vec)[2];
        }
        for(; i < N2; ++i){
            Q[i] = 1.0 / (1.0 + DD[i]);
            sum_Q += Q[i];
        }
        
        // Perform the computation of the gradient
        const double sum_Q_inv = 1.0 / (sum_Q - double(N));
        const __m256d sum_Q_inv_vec = _mm256_set1_pd(sum_Q_inv);

        __m256d P_vec, Y_nD_vec1, Y_nD_vec2, Y_mD_vec1, Y_mD_vec2, Y_mD_vec3, Y_mD_vec4, mult_vec;

        double dY_temp[2];

        int nN = 0;
        int m, mD;
        int nD = 0;
        for(int n = 0; n < N; ++n) {
            mD = 0;

            dY_temp[0] = 0.0;
            dY_temp[1] = 0.0;
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

                Y_mD_vec1 = _mm256_sub_pd(Y_mD_vec1, Y_nD_vec1);
                Y_mD_vec2 = _mm256_sub_pd(Y_mD_vec2, Y_nD_vec2);

                Y_mD_vec1 = _mm256_mul_pd(Y_mD_vec1, mult_vec);
                Y_mD_vec2 = _mm256_mul_pd(Y_mD_vec2, mult_vec);

                Y_mD_vec3 = _mm256_hadd_pd(Y_mD_vec1, Y_mD_vec1);
                Y_mD_vec4 = _mm256_hadd_pd(Y_mD_vec2, Y_mD_vec2);

                dY_temp[0] += ((double*)&Y_mD_vec3)[0] + ((double*)&Y_mD_vec3)[2];
                dY_temp[1] += ((double*)&Y_mD_vec4)[0] + ((double*)&Y_mD_vec4)[2];

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
            dY[nD    ] = dY_temp[0];
            dY[nD + 1] = dY_temp[1];
            
            
            nN += N;
            nD += 2;
        }
        

        // Free memory
        free(DD); DD = NULL;
        free(Q);  Q  = NULL;


        const __m256d zz1_vec  = _mm256_set1_pd(0.01);
        const __m256d z8_vec   = _mm256_set1_pd(0.8);
        const __m256d z2_vec   = _mm256_set1_pd(0.2);
        const __m256d eta_vec  = _mm256_set1_pd(eta);
        const __m256d mom_vec  = _mm256_set1_pd(momentum);
        __m256d gain_vec, dY_vec, Y_vec, uY_vec, sign_dY_vec, sign_uY_vec, gain_cmp, eta_dY_vec;

        
        for(i = 0; i < N * out_dim; i += 4){
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
        for(; i < N * out_dim; ++i){
            gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + 0.2) : (gains[i] * 0.8);
            if(gains[i] < 0.01) gains[i] = 0.01;

            uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
            Y[i] = Y[i] + uY[i];
        } 
    }
}