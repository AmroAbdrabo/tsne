#include <stdio.h>
#include <cstdlib>
#include <cmath>
#include <random>

#include "kernels/computeGP.hpp"
#include "kernels/upgradeGradient.hpp"
#include "kernels/zeroMean.hpp"
#include "parameters.hpp"
#include "utility.hpp"

using namespace computeGPv1;
using namespace upgradeGradientv1;
using namespace zeroMeanv1;

int main(int argc, char** argv) {
    double* X     = (double*) malloc(N * in_dim  * sizeof(double));
    double* Y     = (double*) malloc(N * out_dim * sizeof(double));
    double* dY    = (double*) malloc(N * out_dim * sizeof(double));
    double* uY    = (double*) malloc(N * out_dim * sizeof(double));
    double* gains = (double*) malloc(N * out_dim * sizeof(double));
    double* P     = (double*) malloc(N * N       * sizeof(double));
    if(dY == NULL || uY == NULL || gains == NULL || P == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    
    // initialization
    srand(random_seed);
    for(int i = 0; i < N * out_dim; i++)    uY[i] =  .0;
    for(int i = 0; i < N * out_dim; i++) gains[i] = 1.0;
    for(int i = 0; i < N * out_dim; i++)     Y[i] = randn() * .0001;

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
    std::cout << Y[3] << std::endl;
    for(int iter = 0; iter < first_phase_iter; iter++) {
        upgradeGradient(P, Y, N, out_dim, dY, uY, gains, momentum, eta); // compute gradient and upgrade
        zeroMean(Y, N, out_dim);
        if (iter > 0 && (iter % 50 == 0 || iter == max_iter - 1)) {
            double C = .0;
            C = evaluateError(P, Y, N, out_dim);
            printf("Iteration %d: error is %f \n", iter, C);
        }
    }
    //std::cout << Y[3] << std::endl;

    for(int i = 0; i < N * N; i++)  P[i] /= 12.0;

    for(int iter = 0; iter < second_phase_iter; iter++) {
        upgradeGradient(P, Y, N, out_dim, dY, uY, gains, final_momentum, eta);
        zeroMean(Y, N, out_dim);
        if (iter > 0 && (iter % 50 == 0 || iter == max_iter - 1)) {
            double C = .0;
            C = evaluateError(P, Y, N, out_dim);
            printf("Iteration %d: error is %f \n", iter, C);
        }
    }
    //std::cout << Y[3] << std::endl;


    save_data(Y);
    // Clean up memory
    free(dY);
    free(uY);
    free(gains);
    free(P);
}
