#pragma once
#include "computeSED.hpp"

/// return the sign of the a double, 0 if x = 0, x/abs(x) otherwise
static inline double sign(double x) { return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0)); }

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

namespace upgradeGradientv1 {
    using namespace computeSEDv1;
    
    inline void upgradeGradient(const double* P, double* Y, int N, int out_dim, double* dY, 
                                double* uY, double* gains, const int momentum, const int eta) {
        // Make sure the current gradient contains zeros
        for(int i = 0; i < N * out_dim; i++) dY[i] = 0.0;

        // Compute the squared Euclidean distance matrix
        double* DD = (double*) malloc(N * N * sizeof(double));
        if(DD == NULL) { printf("Memory allocation failed!\n"); exit(1); }
        computeSquaredEuclideanDistance(Y, N, out_dim, DD);

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
