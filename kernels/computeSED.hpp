#pragma once
#include "immintrin.h"

/**
 * compute the pairwise squared Euclidean Distance
 *
 * @param[in] X data to be computed, dim=N*D, row major order
 * @param[in] N number of data points
 * @param[in] D dimension of data
 * @param[out] DD output parameter, the results of computation, symmetric, dim=N*N
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

       for(int i = 0; i < N - b + 1; i += b) {
           const double* Xi = X + i * D;
           for(int j = i; j < N - b + 1; j += b) {
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

namespace computeSEDv2d2ru{ // with blocking for cache AND register w unrolling
    void computeSquaredEuclideanDistance(const double* X, int N, int D, double* DD) {
       const int b = 16; // block size for cache
       const int rbi = 4; // block size for registers
       const int rbj = 16; // block size for registers

        const double* Xi = X;
        for(int i = 0; i < N - b + 1; i += b, Xi += b * D) {
            const double* Xj = X + i * D;
            for(int j = i; j < N - b + 1; j += b, Xj += b * D) {
                const double* Xii = Xi;
                for(int ii = i; ii < i + b - rbi + 1; ii += rbi, Xii += rbi * D) {
                    const double* Xjj = Xj;

                    // stay in registers for reuse
                    double x0d0 = Xii[0], x0d1 = Xii[1];
                    double x1d0 = Xii[2], x1d1 = Xii[3];
                    double x2d0 = Xii[4], x2d1 = Xii[5];
                    double x3d0 = Xii[6], x3d1 = Xii[7];
                    
                    for(int jj = j; jj < j + b; jj += 1, Xjj += D) {
                        if(ii == jj) {
                            DD[ii * N + jj] = 0.0;
                            continue;
                        }

                        int c = ii;
                        
                        double tmp1, tmp2, dist; // dynamic register renaming
                        
                        // stay in registers for reuse
                        double yd0 = Xjj[0];
                        double yd1 = Xjj[1];

                        int base = ii * N + jj;
                        int symm_base = jj * N + ii;
                        
                        tmp1 = x0d0 - yd0;
                        tmp2 = x0d1 - yd1;
                        dist = tmp1 * tmp1 + tmp2 * tmp2;
                        DD[base] = dist;
                        DD[symm_base] = dist;
                        base += N;
                        
                        tmp1 = x1d0 - yd0;
                        tmp2 = x1d1 - yd1;
                        dist = tmp1 * tmp1 + tmp2 * tmp2;
                        DD[base] = dist;
                        DD[symm_base + 1] = dist;
                        base += N;

                        tmp1 = x2d0 - yd0;
                        tmp2 = x2d1 - yd1;
                        dist = tmp1 * tmp1 + tmp2 * tmp2;
                        DD[base] = dist;
                        DD[symm_base + 2] = dist;
                        base += N;

                        tmp1 = x3d0 - yd0;
                        tmp2 = x3d1 - yd1;
                        dist = tmp1 * tmp1 + tmp2 * tmp2;
                        DD[base] = dist;
                        DD[symm_base + 3] = dist;
                    }
                }
            }
        }
    }
}

namespace computeSEDv2d2ruvec{ // with blocking for cache AND register w unrolling
    typedef __m256d d256;
    void computeSquaredEuclideanDistance(const double* X, int N, int D, double* DD) {
       const int b = 16; // block size for cache
       const int rbi = 4; // block size for registers, in the following unrolling, assume rbi = 4, o/w it won't work
       const int rbj = 16; // block size for registers

        const double* Xi = X;
        for(int i = 0; i < N - b + 1; i += b, Xi += b * D) {
            const double* Xj = X + i * D;
            for(int j = i; j < N - b + 1; j += b, Xj += b * D) {
                const double* Xii = Xi;
                for(int ii = i; ii < i + b - rbi + 1; ii += rbi, Xii += rbi * D) {
                    const double* Xjj = Xj;
                    
                    d256 x01 = _mm256_load_pd(Xii);
                    d256 x23 = _mm256_load_pd(Xii + 4);
                    d256 xvecd0 = _mm256_unpacklo_pd(x01, x23); // dim 0 
                    d256 xvecd1 = _mm256_unpackhi_pd(x01, x23); // dim 1
                    xvecd0 = _mm256_permute4x64_pd(xvecd0, 0b11011000);
                    xvecd1 = _mm256_permute4x64_pd(xvecd1, 0b11011000);

                    for(int jj = j; jj < j + b; jj++, Xjj += D) {
                        if(ii == jj) {
                            DD[ii * N + jj] = 0.0;
                            continue;
                        }

                        // load y
                        d256 yvecd0 = _mm256_set1_pd(Xjj[0]);
                        d256 yvecd1 = _mm256_set1_pd(Xjj[1]);

                       d256 xyd0  = _mm256_sub_pd(xvecd0, yvecd0);
                        d256 xyd1  = _mm256_sub_pd(xvecd1, yvecd1);

                        xyd0 = _mm256_mul_pd(xyd0, xyd0);
                        xyd1 = _mm256_mul_pd(xyd1, xyd1);

                        d256 xy = _mm256_add_pd(xyd0, xyd1);

                        _mm256_store_pd(DD + jj * N + ii, xy);

                        int base = ii * N + jj;
                        const int symm_base = jj * N + ii;
                        DD[base] = DD[symm_base    ]; base += N;
                        DD[base] = DD[symm_base + 1]; base += N;
                        DD[base] = DD[symm_base + 2]; base += N;
                        DD[base] = DD[symm_base + 3];
                    }
                }
            }
        }
    }
}

namespace computeSEDv2dx{ // with blocking, any dimension
    void computeSquaredEuclideanDistance(const double* X, int N, int D, double* DD) {
       const int b = 16; // block size

       for(int i = 0; i < N - b + 1; i += b) {
           const double* Xi = X + i * D;
           for(int j = i; j < N - b + 1; j += b) {
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
namespace computeSEDv2d2r{ // with blocking for cache AND register wo unrolling
    void computeSquaredEuclideanDistance(const double* X, int N, int D, double* DD) {
       const int b = 16; // block size for cache
       const int rbi = 4; // block size for registers
       const int rbj = 16; // block size for registers

        const double* Xi = X;
        for(int i = 0; i < N - b + 1; i += b, Xi += b * D) {
            const double* Xj = X + i * D;
            for(int j = i; j < N - b + 1; j += b, Xj += b * D) {
                const double* Xii = Xi;
                for(int ii = i; ii < i + b - rbi + 1; ii += rbi, Xii += rbi * D) {
                    const double* Xjj = Xj;
                    for(int jj = j; jj < j + b - rbj + 1; jj += rbj, Xjj += rbj * D) {
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
namespace computeSEDv2d2ru_depr{ // with blocking for cache AND register w unrolling
    void computeSquaredEuclideanDistance(const double* X, int N, int D, double* DD) {
       const int b = 16; // block size for cache
       const int rbi = 4; // block size for registers
       const int rbj = 16; // block size for registers

        const double* Xi = X;
        for(int i = 0; i < N - b + 1; i += b, Xi += b * D) {
            const double* Xj = X + i * D;
            for(int j = i; j < N - b + 1; j += b, Xj += b * D) {
                const double* Xii = Xi;
                for(int ii = i; ii < i + b - rbi + 1; ii += rbi, Xii += rbi * D) {
                    const double* Xjj = Xj;
                    double x0d0 = Xii[0], x0d1 = Xii[1];
                    double x1d0 = Xii[2], x1d1 = Xii[3];
                    double x2d0 = Xii[4], x2d1 = Xii[5];
                    double x3d0 = Xii[6], x3d1 = Xii[7];
                    for(int jj = j; jj < j + b - rbj + 1; jj += rbj, Xjj += rbj * D) {
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