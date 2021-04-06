#pragma once
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