#pragma once
#include "computeSED.hpp"
#include "memory/Pool.h"
#include <cfloat>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

/**
 * compute variance for each input data point, calculate pairwise distances, and symmetrize the matrix
 *
 * @param X input data whose dim = N*in_dim, stored in row major order
 * @param N number of input data points
 * @param in_dim input data dimension
 * @param P out-parameter for storing the pairwise distance, symmetric, dim=N*N
 * @param perp user-defined perplexity to determine the optimal variance
 */

namespace computeGPv1 {

    inline void computeGaussianPerplexity(const double* X, const size_t N, const unsigned int in_dim, double* P, const double perp) {
        // Compute the squared Euclidean distance matrix

        double* DD = (double*) memory::Pool::getMemory(N * N * sizeof(double));
        if(DD == NULL) { printf("Memory allocation failed!\n"); exit(1); }
        computeSEDv1::computeSquaredEuclideanDistance(X, N, in_dim, DD);

        // Compute the Gaussian kernel row by row
        int nN = 0;
        for(int n = 0; n < N; n++) {

            // Initialize some variables
            bool found = false;
            double beta = 1.0;
            double min_beta = -DBL_MAX;
            double max_beta =  DBL_MAX;
            double tol = 1e-5;
            double sum_P;

            // Iterate until we found a good perplexity
            int iter = 0;
            while(!found && iter < 200) {

                // Compute Gaussian kernel row
                for(int m = 0; m < N; m++) P[nN + m] = exp(-beta * DD[nN + m]);
                P[nN + n] = DBL_MIN;

                // Compute entropy of current row
                sum_P = DBL_MIN;
                for(int m = 0; m < N; m++) sum_P += P[nN + m];
                double H = 0.0;
                for(int m = 0; m < N; m++) H += beta * (DD[nN + m] * P[nN + m]);
                H = (H / sum_P) + log(sum_P);

                // Evaluate whether the entropy is within the tolerance level
                double Hdiff = H - log(perp);
                if(Hdiff < tol && -Hdiff < tol) {
                    found = true;
                }
                else {
                    if(Hdiff > 0) {
                        min_beta = beta;
                        if(max_beta == DBL_MAX || max_beta == -DBL_MAX)
                            beta *= 2.0;
                        else
                            beta = (beta + max_beta) / 2.0;
                    }
                    else {
                        max_beta = beta;
                        if(min_beta == -DBL_MAX || min_beta == DBL_MAX)
                            beta /= 2.0;
                        else
                            beta = (beta + min_beta) / 2.0;
                    }
                }

                // Update iteration counter
                iter++;
            }

            // Row normalize P
            for(int m = 0; m < N; m++) P[nN + m] /= sum_P;
            nN += N;
        }

        // Clean up memory
        // free(DD); DD = NULL;

        // Symmetrize input similarities
        nN = 0;
        for(int n = 0; n < N; n++) {
            int mN = (n + 1) * N;
            for(int m = n + 1; m < N; m++) {
                P[nN + m] += P[mN + n];
                P[mN + n]  = P[nN + m];
                mN += N;
            }
            nN += N;
        }
        double sum_P = .0;
        for(int i = 0; i < N * N; i++) sum_P += P[i];
        for(int i = 0; i < N * N; i++) P[i] /= sum_P;
    }
}
