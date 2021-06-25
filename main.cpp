#include <stdio.h>
#include <cstdlib>
#include <cmath>
#include <random>

#include "kernels/computeGP.hpp"
#include "kernels/updateGradient.hpp"
#include "kernels/zeroMean.hpp"
#include "kernels/updateGradient_zeroMean.hpp"
#include "parameters.hpp"
#include "utility.hpp"
#include "memory/Pool.h"

using namespace computeGPv1;
using namespace updateGradientv3_2_outdim;
using namespace zeroMeanv6;
using namespace updateGradient_zeroMeanv5_d2;

int main(int argc, char** argv) {
    memory::Pool::allocate((in_dim + 4 * out_dim + 2*N) * 1.1 * N * sizeof(double));

    double* X     = (double*) memory::Pool::getMemory(N * in_dim * sizeof(double));
    double* Y     = (double*) memory::Pool::getMemory(N * out_dim * sizeof(double));
    double* dY    = (double*) memory::Pool::getMemory(N * out_dim * sizeof(double));
    double* uY    = (double*) memory::Pool::getMemory(N * out_dim * sizeof(double));
    double* gains = (double*) memory::Pool::getMemory(N * out_dim * sizeof(double));
    double* P     = (double*) memory::Pool::getMemory(N * N * sizeof(double));
    if(dY == NULL || uY == NULL || gains == NULL || P == NULL) { printf("Memory allocation failed!\n"); exit(1); }

    // initialization
    srand(random_seed);
    for(int i = 0; i < N * out_dim; i++)    uY[i] =  0.0;
    for(int i = 0; i < N * out_dim; i++) gains[i] = 1.0;
    for(int i = 0; i < N * out_dim; i++)     Y[i] = randn() * 0.0001;

    load_data(X);

    // preprocess input data
    zeroMean(X, N, in_dim);

    double max_X = .0;
    for(int i = 0; i < N * in_dim; i++) {
        if(fabs(X[i]) > max_X) max_X = fabs(X[i]);
    }
    for(int i = 0; i < N * in_dim; i++) X[i] /= max_X;

    computeGaussianPerplexity(X, N, out_dim, P, perp); // namespace computeGPvx

    // Lie about the P-values
    for(int i = 0; i < N * N; i++) P[i] *= 12.0;

    // Training
    for(int iter = 0; iter < first_phase_iter; iter++) {
        updateGradient_zeroMean(P, Y, N, out_dim, dY, uY, gains, momentum, eta);
        //updateGradient(P, Y, N, out_dim, dY, uY, gains, momentum, eta); // compute gradient and upgrade
        //zeroMean(Y, N, out_dim);
        if (iter > 0 && (iter % 50 == 0 || iter == max_iter - 1)) {
            double C = .0;
            C = evaluateError(P, Y, N, out_dim);
            printf("Iteration %d: error is %f \n", iter, C);
        }
    }
    //std::cout << Y[3] << std::endl;

    for(int i = 0; i < N * N; i++)  P[i] /= 12.0;

    for(int iter = 0; iter < second_phase_iter; iter++) {
        updateGradient_zeroMean(P, Y, N, out_dim, dY, uY, gains, final_momentum, eta);
        //updateGradient(P, Y, N, out_dim, dY, uY, gains, final_momentum, eta);
        //zeroMean(Y, N, out_dim);
        if (iter > 0 && (iter % 50 == 0 || iter == max_iter - 1)) {
            double C = .0;
            C = evaluateError(P, Y, N, out_dim);
            printf("Iteration %d: error is %f \n", iter, C);
        }
    }
    //std::cout << Y[3] << std::endl;

    // Save data for visualization
    save_data(Y);

    // Clean up memory
    free(X);
    free(Y);
    free(dY);
    free(uY);
    free(gains);
    free(P);
}
