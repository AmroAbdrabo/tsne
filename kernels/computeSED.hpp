#pragma once

/**
 * compute the pairwise squared Euclidean Distance
 *
 * @param X data to be computed, dim=N*D, row major order
 * @param N number of data points
 * @param D dimension of data
 * @param DD output parameter, the results of computation, symmetric, dim=N*N
 */

namespace computeSEDv1{
    void computeSquaredEuclideanDistance(const double* X, int N, int D, double* DD) {
        const double* XnD = X;
        for(int n = 0; n < N; ++n, XnD += D) { // point[n]
            const double* XmD = XnD + D; // point[n+1]
            double* curr_elem = &DD[n*N + n]; // DD[n,n]
            *curr_elem = 0.0; // DD[n,n] = 0
            double* curr_elem_sym = curr_elem + N; // DD[n+1,n] = dist(point[n], point[n+1])
            for(int m = n + 1; m < N; ++m, XmD+=D, curr_elem_sym+=N) {
                *(++curr_elem) = 0.0;
                for(int d = 0; d < D; ++d) {
                    *curr_elem += (XnD[d] - XmD[d]) * (XnD[d] - XmD[d]); // DD[n,m] = dist(point[n], point[m])
                }
                *curr_elem_sym = *curr_elem; // DD[m,n] = DD[n,m]
            }
        }
    }
}


namespace computeSEDv2{ // with blocking
    void computeSquaredEuclideanDistance(const double* X, int N, int D, double* DD) {
       const int b = 8; // block size

       for(int i = 0; i < N; i += b) {
           for(int j = i; j < N; j += b) {
               for(int ii = i; ii < i + b; ii++) {
                   const double* Xii = X + ii*D;
                   for(int jj = j; jj < j + b; jj++) {
                       // compute distance
                       if(ii == jj) {
                           DD[ii*N + jj] = 0.0;
                           continue;
                       }
                       
                       const double* Xjj = X + jj*D;
                       double dist = 0.0;

                       for(int d = 0; d < D; d++) {
                           double tmp = Xii[d] - Xjj[d];
                           dist += tmp * tmp;
                       }
                       
                       DD[ii*N + jj] = dist;
                       DD[jj*N + ii] = dist;
                   }
               }
           }
       }
    }
}

