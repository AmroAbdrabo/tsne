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
                // double tmp1 = XnD[0] - XmD[0];
                // double tmp2 = XnD[1] - XmD[1];
                // *curr_elem += tmp1 * tmp1 + tmp2 * tmp2;

                *curr_elem_sym = *curr_elem; // DD[m,n] = DD[n,m]
            }
        }
    }
}


namespace computeSEDv2d2{ // with blocking
    void computeSquaredEuclideanDistance(const double* X, int N, int D, double* DD) {
       const int b = 16; // block size for cache

       for(int i = 0; i < N; i += b) {
           const double* Xi = X + i * D;
           for(int j = i; j < N; j += b) {
               const double* Xii = Xi;
               const double* Xj = X + j * D;
               for(int ii = i; ii < i + b; ii++, Xii += D) {
                   const double* Xjj = Xj;
                   for(int jj = j; jj < j + b; jj++, Xjj += D) {

                        // compute distance
                        if(ii == jj) {
                            DD[ii * N + jj] = 0.0;
                            continue;
                        }

                        // for dim = 2
                        double tmp1 = Xii[0] - Xjj[0];
                        double tmp2 = Xii[1] - Xjj[1];
                        double dist = tmp1 * tmp1 + tmp2 * tmp2;
                        
                        DD[ii * N + jj] = dist;
                        DD[jj * N + ii] = dist;
                   }
               }
           }
       }
    }
}

namespace computeSEDv2d2r{ // with blocking for cache AND register wo unrolling
    void computeSquaredEuclideanDistance(const double* X, int N, int D, double* DD) {
       const int b = 16; // block size for cache
       const int rbi = 4; // block size for registers
       const int rbj = 16; // block size for registers

        const double* Xi = X;
        for(int i = 0; i < N; i += b, Xi += b * D) {
            const double* Xj = X + i * D;
            for(int j = i; j < N; j += b, Xj += b * D) {
                const double* Xii = Xi;
                for(int ii = i; ii < i + b; ii += rbi, Xii += rbi * D) {
                    const double* Xjj = Xj;
                    for(int jj = j; jj < j + b; jj += rbj, Xjj += rbj * D) {
                        const double* Xri = Xii;
                        for(int ri = ii; ri < ii + rbi; ri++, Xri += D) {
                            const double* Xrj = Xjj;
                            for(int rj = jj; rj < jj + rbj; rj++, Xrj += D) {
                                
                                if(ri == rj) {
                                    DD[ri * N + rj] = 0.0;
                                    continue;
                                }
                                
                                double tmp1 = Xri[0] - Xrj[0];
                                double tmp2 = Xri[1] - Xrj[1];
                                double dist = tmp1 * tmp1 + tmp2 * tmp2;
                                    
                                DD[ri * N + rj] = dist;
                                DD[rj * N + ri] = dist;
                            }
                        }
                    }
                }
            }
        }
    }
}

namespace computeSEDv2d2ru{ // with blocking for cache AND register w unrolling
    void computeSquaredEuclideanDistance(const double* X, int N, int D, double* DD) {
       const int b = 16; // block size for cache
       const int rbi = 4; // block size for registers
       const int rbj = 16; // block size for registers

        const double* Xi = X;
        for(int i = 0; i < N; i += b, Xi += b * D) {
            const double* Xj = X + i * D;
            for(int j = i; j < N; j += b, Xj += b * D) {
                const double* Xii = Xi;
                for(int ii = i; ii < i + b; ii += rbi, Xii += rbi * D) {
                    const double* Xjj = Xj;
                    double x0d0 = Xii[0], x0d1 = Xii[1];
                    double x1d0 = Xii[2], x1d1 = Xii[3];
                    double x2d0 = Xii[4], x2d1 = Xii[5];
                    double x3d0 = Xii[6], x3d1 = Xii[7];
                    for(int jj = j; jj < j + b; jj += rbj, Xjj += rbj * D) {
                        const double* Xrj = Xjj;
                        for(int rj = jj; rj < jj + rbj; rj++, Xrj += D) {
                            
                            if(ii == rj) {
                                DD[ii * N + rj] = 0.0;
                                continue;
                            }

                            int c = ii;
                            
                            double tmp1, tmp2, dist;
                            
                            double yd0 = Xrj[0];
                            double yd1 = Xrj[1];
                            
                            tmp1 = x0d0 - yd0;
                            tmp2 = x0d1 - yd1;
                            dist = tmp1 * tmp1 + tmp2 * tmp2;
                            DD[c * N + rj] = dist;
                            DD[rj * N + c] = dist;
                            c++;
                            
                            tmp1 = x1d0 - yd0;
                            tmp2 = x1d1 - yd1;
                            dist = tmp1 * tmp1 + tmp2 * tmp2;
                            DD[c * N + rj] = dist;
                            DD[rj * N + c] = dist;
                            c++;

                            tmp1 = x2d0 - yd0;
                            tmp2 = x2d1 - yd1;
                            dist = tmp1 * tmp1 + tmp2 * tmp2;
                            DD[c * N + rj] = dist;
                            DD[rj * N + c] = dist;
                            c++;

                            tmp1 = x3d0 - yd0;
                            tmp2 = x3d1 - yd1;
                            dist = tmp1 * tmp1 + tmp2 * tmp2;
                            DD[c * N + rj] = dist;
                            DD[rj * N + c] = dist;
                        }
                    }
                }
            }
        }
    }
}
namespace computeSEDv2dx{ // with blocking
    void computeSquaredEuclideanDistance(const double* X, int N, int D, double* DD) {
       const int b = 16; // block size

       for(int i = 0; i < N; i += b) {
           const double* Xi = X + i * D;
           for(int j = i; j < N; j += b) {
               const double* Xii = Xi;
               const double* Xj = X + j * D;
               for(int ii = i; ii < i + b; ii++, Xii += D) {
                   const double* Xjj = Xj;
                   for(int jj = j; jj < j + b; jj++, Xjj += D) {

                        // compute distance
                        if(ii == jj) {
                            DD[ii * N + jj] = 0.0;
                            continue;
                        }
                       
                        double dist = 0.0;

                        // for general dimensions
                       for(int d = 0; d < D; d++) {
                           double tmp = Xii[d] - Xjj[d];
                           dist += tmp * tmp;
                       }
                        
                        DD[ii * N + jj] = dist;
                        DD[jj * N + ii] = dist;
                   }
               }
           }
       }
    }
}