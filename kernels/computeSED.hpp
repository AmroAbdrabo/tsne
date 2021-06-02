#pragma once
#include "immintrin.h"
#include <iostream>

/**
 * compute the pairwise squared Euclidean Distance
 *
 * @param[in] X data to be computed, dim=N*D, row major order
 * @param[in] N number of data points
 * @param[in] D dimension of data
 * @param[out] DD output parameter, the results of computation, symmetric, dim=N*N
 */

namespace computeSEDv1{
    void computeSquaredEuclideanDistance(const double* X, int N,  int D, double* DD) {
        const double* XnD = X;
        for(int n = 0; n < N; ++n, XnD += D) { // point[n]
            const double* XmD = XnD + D; // point[n+1]
            double* curr_elem = &DD[n*N + n]; // DD[n,n]
            *curr_elem = 0.0; // DD[n,n] = 0
            double* curr_elem_sym = curr_elem + N; // DD[n+1,n] = dist(point[n], point[n+1])
            for(int m = n + 1; m < N; ++m, XmD+=D, curr_elem_sym+=N) {
                // *(++curr_elem) = 0.0;
                // for(int d = 0; d < D; ++d) {
                //     *curr_elem += (XnD[d] - XmD[d]) * (XnD[d] - XmD[d]); // DD[n,m] = dist(point[n], point[m])
                // }
                 double tmp1 = XnD[0] - XmD[0];
                 double tmp2 = XnD[1] - XmD[1];
                 
                 *(++curr_elem) = tmp1 * tmp1 + tmp2 * tmp2;

                *curr_elem_sym = *curr_elem; // DD[m,n] = DD[n,m]
            }
        }
    }
}

namespace computeSEDv2d2{ // with blocking
    void computeSquaredEuclideanDistance(const double* X, int N,  int D, double* DD) {
       const int b = 32; // block size for cache

        const double* Xi = X;
       for(int i = 0; i < N - b + 1; i += b, Xi += b * D) {
            const double* Xj = X + i * D;
           for(int j = i; j < N - b + 1; j += b, Xj += b * D) {
               const double* Xii = Xi;
               if(i == j) {
                   for(int ii = i; ii < i + b; ii++, Xii += D) {
                        const double* Xjj = Xj;
                        double xii0 = Xii[0], xii1 = Xii[1];
                        int base = ii * N;
                        for(int jj = j; jj < j + b; jj++, Xjj += D) {
                                if(ii == jj) {
                                    // DD[base + jj] = 0.0;
                                    DD[base + jj] = 0.0;
                                    continue;
                                }

                                double tmp1 = xii0 - Xjj[0];
                                double tmp2 = xii1 - Xjj[1];
                                double dist = tmp1 * tmp1 + tmp2 * tmp2;
                                
                                DD[base + jj] = dist;
                        }
                    }
                    continue;
               }

               for(int ii = i; ii < i + b; ii++, Xii += D) {
                   const double* Xjj = Xj;
                   double xii0 = Xii[0], xii1 = Xii[1];
                   int base = ii * N;
                   for(int jj = j; jj < j + b; jj++, Xjj += D) {

                        // compute distance
                        if(ii == jj) {
                            DD[base + jj] = 0.0;
                            continue;
                        }

                        // for dim = 2
                        double tmp1 = xii0 - Xjj[0];
                        double tmp2 = xii1 - Xjj[1];
                        double dist = tmp1 * tmp1 + tmp2 * tmp2;
                        
                        DD[base + jj] = dist;
                        DD[jj * N + ii] = dist;
                   }
               }
           }
       }
    }
}

namespace computeSEDv2d2buf{ // with blocking w buffering
    void computeSquaredEuclideanDistance(const double* X, int N,  int D, double* DD) {
       const int b = 32; // block size for cache
       double buf[b * b] __attribute__ ((aligned (32)));

        const double* Xi = X;
       for(int i = 0; i < N - b + 1; i += b, Xi += b * D) {
            const double* Xj = X + i * D;
           for(int j = i; j < N - b + 1; j += b, Xj += b * D) {
               const double* Xii = Xi;
               if(i == j) { // diagonal block
                    for(int ii = i; ii < i + b; ii++, Xii += D) {
                        const double* Xjj = Xj;
                        double xii0 = Xii[0], xii1 = Xii[1];
                        int base = ii * N;
                        for(int jj = j; jj < j + b; jj++, Xjj += D) {
                                if(ii == jj) {
                                    // DD[base + jj] = 0.0;
                                    DD[base + jj] = 0.0;
                                    continue;
                                }

                                double tmp1 = xii0 - Xjj[0];
                                double tmp2 = xii1 - Xjj[1];
                                double dist = tmp1 * tmp1 + tmp2 * tmp2;
                                
                                DD[base + jj] = dist;
                        }
                    }
                    continue;
               }

               for(int ii = i; ii < i + b; ii++, Xii += D) {
                   const double* Xjj = Xj;
                   double xii0 = Xii[0], xii1 = Xii[1];
                   int base = ii * N;
                   int shift = ii - i;
                   int buf_base = (ii - i) * b;
                   for(int jj = j; jj < j + b; jj++, Xjj += D) {
                        // for dim = 2
                        double tmp1 = xii0 - Xjj[0];
                        double tmp2 = xii1 - Xjj[1];
                        double dist = tmp1 * tmp1 + tmp2 * tmp2;
                        
                        DD[base + jj] = dist;
                        buf[(jj-j) * b + shift] = dist;
                   }
               }

               // copy the buffer back to DD
                for(int jj = j; jj < j + b; jj++) { 
                    int base = jj * N;
                    int buf_base = (jj - j) * b;
                    for(int ii = i; ii < i + b; ii++) {
                        DD[base + ii] = buf[buf_base + ii - i];
                    }
                }
           }
       }
    }
}

namespace computeSEDv2d2ru{ // with blocking for cache AND register w unrolling wo buffering
    void computeSquaredEuclideanDistance(const double* X, int N,  int D, double* DD) {
       const int b = 32; // block size for cache
       const int rbi = 4; // block size for registers

        const double* Xi = X;
        for(int i = 0; i < N - b + 1; i += b, Xi += b * D) {
            const double* Xj = X + i * D;
            for(int j = i; j < N - b + 1; j += b, Xj += b * D) {
                const double* Xii = Xi;
                if(i == j) { // diagonal block
                    for(int ii = i; ii < i + b; ii++, Xii += D) {
                        const double* Xjj = Xj;
                        double xii0 = Xii[0], xii1 = Xii[1];
                        int base = ii * N;
                        for(int jj = j; jj < j + b; jj++, Xjj += D) {
                                if(ii == jj) {
                                    // DD[base + jj] = 0.0;
                                    DD[base + jj] = 0.0;
                                    continue;
                                }

                                double tmp1 = xii0 - Xjj[0];
                                double tmp2 = xii1 - Xjj[1];
                                double dist = tmp1 * tmp1 + tmp2 * tmp2;
                                
                                DD[base + jj] = dist;
                        }
                    }
                    continue;
               }

                for(int ii = i; ii < i + b - rbi + 1; ii += rbi, Xii += rbi * D) {
                    const double* Xjj = Xj;

                    // stay in registers for reuse
                    double x0d0 = Xii[0], x0d1 = Xii[1];
                    double x1d0 = Xii[2], x1d1 = Xii[3];
                    double x2d0 = Xii[4], x2d1 = Xii[5];
                    double x3d0 = Xii[6], x3d1 = Xii[7];
                    
                    for(int jj = j; jj < j + b - 1; jj += 2, Xjj += D * 2) {
                            double tmp1, tmp2, dist, tmp3, tmp4, ddist; // dynamic register renaming
                            
                            // stay in registers for reuse
                            double yd0 = Xjj[0];
                            double yd1 = Xjj[1];
                            double yyd0 = Xjj[2];
                            double yyd1 = Xjj[3];

                            int base = ii * N + jj;
                            int symm_base = jj * N + ii;
                            int bbase = base + 1; 
                            int ssymm_base = symm_base + N; 
                            
                            tmp1 = x0d0 - yd0;
                            tmp2 = x0d1 - yd1;
                            tmp3 = x0d0 - yyd0;
                            tmp4 = x0d1 - yyd1;
                            dist = tmp1 * tmp1 + tmp2 * tmp2;
                            ddist = tmp3 * tmp3 + tmp4 * tmp4;
                            DD[base] = dist;
                            DD[symm_base] = dist;
                            DD[bbase] = ddist;
                            DD[ssymm_base] = ddist;
                            base += N;
                            bbase += N;
                            
                            tmp1 = x1d0 - yd0;
                            tmp2 = x1d1 - yd1;
                            tmp3 = x1d0 - yyd0;
                            tmp4 = x1d1 - yyd1;
                            dist = tmp1 * tmp1 + tmp2 * tmp2;
                            ddist = tmp3 * tmp3 + tmp4 * tmp4;
                            DD[base] = dist;
                            DD[symm_base + 1] = dist;
                            DD[bbase] = ddist;
                            DD[ssymm_base + 1] = ddist;
                            base += N;
                            bbase += N;

                            tmp1 = x2d0 - yd0;
                            tmp2 = x2d1 - yd1;
                            tmp3 = x2d0 - yyd0;
                            tmp4 = x2d1 - yyd1;
                            dist = tmp1 * tmp1 + tmp2 * tmp2;
                            ddist = tmp3 * tmp3 + tmp4 * tmp4;
                            DD[base] = dist;
                            DD[symm_base + 2] = dist;
                            DD[bbase] = ddist;
                            DD[ssymm_base + 2] = ddist;
                            base += N;
                            bbase += N;

                            tmp1 = x3d0 - yd0;
                            tmp2 = x3d1 - yd1;
                            tmp3 = x3d0 - yyd0;
                            tmp4 = x3d1 - yyd1;
                            dist = tmp1 * tmp1 + tmp2 * tmp2;
                            ddist = tmp3 * tmp3 + tmp4 * tmp4;
                            DD[base] = dist;
                            DD[symm_base + 3] = dist;
                            DD[bbase] = ddist;
                            DD[ssymm_base + 3] = ddist;
                    }
                }
            }
        }
    }

    void computeSquaredEuclideanDistance(const double* X, int N,  int D, double* DD, const int b) {
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
                    
                    for(int jj = j; jj < j + b - 1; jj += 2, Xjj += D * 2) {
                        if(ii == jj) {
                            DD[ii * N + jj] = 0.0;
                            double tmp3, tmp4, ddist; // dynamic register renaming
                            
                            // stay in registers for reuse
                            double yyd0 = Xjj[2];
                            double yyd1 = Xjj[3];

                            int bbase = ii * N + jj + 1; 
                            int ssymm_base = jj * N + ii + N; 
                            
                            tmp3 = x0d0 - yyd0;
                            tmp4 = x0d1 - yyd1;
                            ddist = tmp3 * tmp3 + tmp4 * tmp4;
                            DD[bbase] = ddist;
                            DD[ssymm_base] = ddist;
                            bbase += N;
                            
                            tmp3 = x1d0 - yyd0;
                            tmp4 = x1d1 - yyd1;
                            ddist = tmp3 * tmp3 + tmp4 * tmp4;
                            DD[bbase] = ddist;
                            DD[ssymm_base + 1] = ddist;
                            bbase += N;

                            tmp3 = x2d0 - yyd0;
                            tmp4 = x2d1 - yyd1;
                            ddist = tmp3 * tmp3 + tmp4 * tmp4;
                            DD[bbase] = ddist;
                            DD[ssymm_base + 2] = ddist;
                            bbase += N;

                            tmp3 = x3d0 - yyd0;
                            tmp4 = x3d1 - yyd1;
                            ddist = tmp3 * tmp3 + tmp4 * tmp4;
                            DD[bbase] = ddist;
                            DD[ssymm_base + 3] = ddist;
                        } else { 
                            double tmp1, tmp2, dist, tmp3, tmp4, ddist; // dynamic register renaming
                            
                            // stay in registers for reuse
                            double yd0 = Xjj[0];
                            double yd1 = Xjj[1];
                            double yyd0 = Xjj[2];
                            double yyd1 = Xjj[3];

                            int base = ii * N + jj;
                            int symm_base = jj * N + ii;
                            int bbase = base + 1; 
                            int ssymm_base = symm_base + N; 
                            
                            tmp1 = x0d0 - yd0;
                            tmp2 = x0d1 - yd1;
                            tmp3 = x0d0 - yyd0;
                            tmp4 = x0d1 - yyd1;
                            dist = tmp1 * tmp1 + tmp2 * tmp2;
                            ddist = tmp3 * tmp3 + tmp4 * tmp4;
                            DD[base] = dist;
                            DD[symm_base] = dist;
                            DD[bbase] = ddist;
                            DD[ssymm_base] = ddist;
                            base += N;
                            bbase += N;
                            
                            tmp1 = x1d0 - yd0;
                            tmp2 = x1d1 - yd1;
                            tmp3 = x1d0 - yyd0;
                            tmp4 = x1d1 - yyd1;
                            dist = tmp1 * tmp1 + tmp2 * tmp2;
                            ddist = tmp3 * tmp3 + tmp4 * tmp4;
                            DD[base] = dist;
                            DD[symm_base + 1] = dist;
                            DD[bbase] = ddist;
                            DD[ssymm_base + 1] = ddist;
                            base += N;
                            bbase += N;

                            tmp1 = x2d0 - yd0;
                            tmp2 = x2d1 - yd1;
                            tmp3 = x2d0 - yyd0;
                            tmp4 = x2d1 - yyd1;
                            dist = tmp1 * tmp1 + tmp2 * tmp2;
                            ddist = tmp3 * tmp3 + tmp4 * tmp4;
                            DD[base] = dist;
                            DD[symm_base + 2] = dist;
                            DD[bbase] = ddist;
                            DD[ssymm_base + 2] = ddist;
                            base += N;
                            bbase += N;

                            tmp1 = x3d0 - yd0;
                            tmp2 = x3d1 - yd1;
                            tmp3 = x3d0 - yyd0;
                            tmp4 = x3d1 - yyd1;
                            dist = tmp1 * tmp1 + tmp2 * tmp2;
                            ddist = tmp3 * tmp3 + tmp4 * tmp4;
                            DD[base] = dist;
                            DD[symm_base + 3] = dist;
                            DD[bbase] = ddist;
                            DD[ssymm_base + 3] = ddist;
                        }
                    }
                }
            }
        }
    }
}

namespace computeSEDv2d2rubuf{ // with blocking for cache AND register w unrolling w buffering
    void computeSquaredEuclideanDistance(const double* X, int N,  int D, double* DD) {
       const int b = 32; // block size for cache
       const int rbi = 4; // block size for registers
       double buf[b * b] __attribute__ ((aligned (32)));


        const double* Xi = X;
        for(int i = 0; i < N - b + 1; i += b, Xi += b * D) {
            const double* Xj = X + i * D;
            for(int j = i; j < N - b + 1; j += b, Xj += b * D) {
                const double* Xii = Xi;
                if(i == j) { // diagonal block
                    for(int ii = i; ii < i + b; ii++, Xii += D) {
                        const double* Xjj = Xj;
                        double xii0 = Xii[0], xii1 = Xii[1];
                        int base = ii * N;
                        for(int jj = j; jj < j + b; jj++, Xjj += D) {
                                if(ii == jj) {
                                    // DD[base + jj] = 0.0;
                                    DD[base + jj] = 0.0;
                                    continue;
                                }

                                double tmp1 = xii0 - Xjj[0];
                                double tmp2 = xii1 - Xjj[1];
                                double dist = tmp1 * tmp1 + tmp2 * tmp2;
                                
                                DD[base + jj] = dist;
                        }
                    }
                    continue;
               }

                for(int ii = i; ii < i + b - rbi + 1; ii += rbi, Xii += rbi * D) {
                    const double* Xjj = Xj;

                    // stay in registers for reuse
                    double x0d0 = Xii[0], x0d1 = Xii[1];
                    double x1d0 = Xii[2], x1d1 = Xii[3];
                    double x2d0 = Xii[4], x2d1 = Xii[5];
                    double x3d0 = Xii[6], x3d1 = Xii[7];
                    
                    for(int jj = j; jj < j + b - 1; jj += 2, Xjj += D * 2) {
                            double tmp1, tmp2, dist, tmp3, tmp4, ddist; // dynamic register renaming
                            
                            // stay in registers for reuse
                            double yd0 = Xjj[0];
                            double yd1 = Xjj[1];
                            double yyd0 = Xjj[2];
                            double yyd1 = Xjj[3];

                            int base = ii * N + jj;
                            int buf_base = (jj-j) * b + ii - i;
                            int bbase = base + 1; 
                            int bbuf_base = buf_base + b; 
                            
                            tmp1 = x0d0 - yd0;
                            tmp2 = x0d1 - yd1;
                            tmp3 = x0d0 - yyd0;
                            tmp4 = x0d1 - yyd1;
                            dist = tmp1 * tmp1 + tmp2 * tmp2;
                            ddist = tmp3 * tmp3 + tmp4 * tmp4;
                            DD[base] = dist;
                            buf[buf_base] = dist;
                            DD[bbase] = ddist;
                            buf[bbuf_base] = ddist;
                            base += N;
                            bbase += N;
                            
                            tmp1 = x1d0 - yd0;
                            tmp2 = x1d1 - yd1;
                            tmp3 = x1d0 - yyd0;
                            tmp4 = x1d1 - yyd1;
                            dist = tmp1 * tmp1 + tmp2 * tmp2;
                            ddist = tmp3 * tmp3 + tmp4 * tmp4;
                            DD[base] = dist;
                            buf[buf_base + 1] = dist;
                            DD[bbase] = ddist;
                            buf[bbuf_base + 1] = ddist;
                            base += N;
                            bbase += N;

                            tmp1 = x2d0 - yd0;
                            tmp2 = x2d1 - yd1;
                            tmp3 = x2d0 - yyd0;
                            tmp4 = x2d1 - yyd1;
                            dist = tmp1 * tmp1 + tmp2 * tmp2;
                            ddist = tmp3 * tmp3 + tmp4 * tmp4;
                            DD[base] = dist;
                            buf[buf_base + 2] = dist;
                            DD[bbase] = ddist;
                            buf[bbuf_base + 2] = ddist;
                            base += N;
                            bbase += N;

                            tmp1 = x3d0 - yd0;
                            tmp2 = x3d1 - yd1;
                            tmp3 = x3d0 - yyd0;
                            tmp4 = x3d1 - yyd1;
                            dist = tmp1 * tmp1 + tmp2 * tmp2;
                            ddist = tmp3 * tmp3 + tmp4 * tmp4;
                            DD[base] = dist;
                            buf[buf_base + 3] = dist;
                            DD[bbase] = ddist;
                            buf[bbuf_base + 3] = ddist;
                    }
                }

                // copy the buffer back to DD
                for(int jj = j; jj < j + b; jj++) { 
                    int base = jj * N;
                    int buf_base = (jj - j) * b;
                    for(int ii = i; ii < i + b; ii++) {
                        DD[base + ii] = buf[buf_base + ii - i];
                    }
                }
            }
        }
    }
}

namespace computeSEDv2d2rubuf_depr{ // with blocking for cache AND register w unrolling w buffering
    void computeSquaredEuclideanDistance(const double* X, int N,  int D, double* DD) {
       const int b = 32; // block size for cache
       const int rbi = 8; // block size for registers
       double buf[b * b] __attribute__ ((aligned (32)));


        const double* Xi = X;
        for(int i = 0; i < N - b + 1; i += b, Xi += b * D) {
            const double* Xj = X + i * D;
            for(int j = i; j < N - b + 1; j += b, Xj += b * D) {
                const double* Xii = Xi;
                if(i == j) { // diagonal block
                    for(int ii = i; ii < i + b; ii++, Xii += D) {
                        const double* Xjj = Xj;
                        double xii0 = Xii[0], xii1 = Xii[1];
                        int base = ii * N;
                        for(int jj = j; jj < j + b; jj++, Xjj += D) {
                                if(ii == jj) {
                                    // DD[base + jj] = 0.0;
                                    DD[base + jj] = 0.0;
                                    continue;
                                }

                                double tmp1 = xii0 - Xjj[0];
                                double tmp2 = xii1 - Xjj[1];
                                double dist = tmp1 * tmp1 + tmp2 * tmp2;
                                
                                DD[base + jj] = dist;
                        }
                    }
                    continue;
               }

                for(int ii = i; ii < i + b - rbi + 1; ii += rbi, Xii += rbi * D) {
                    const double* Xjj = Xj;

                    // stay in registers for reuse
                    double x0d0 = Xii[0], x0d1 = Xii[1];
                    double x1d0 = Xii[2], x1d1 = Xii[3];
                    double x2d0 = Xii[4], x2d1 = Xii[5];
                    double x3d0 = Xii[6], x3d1 = Xii[7];
                    double x4d0 = Xii[8], x4d1 = Xii[9];
                    double x5d0 = Xii[10], x5d1 = Xii[11];
                    double x6d0 = Xii[12], x6d1 = Xii[13];
                    double x7d0 = Xii[14], x7d1 = Xii[15];
                    
                    for(int jj = j; jj < j + b; jj += 1, Xjj += D) {
                            // stay in registers for reuse
                            double yd0 = Xjj[0];
                            double yd1 = Xjj[1];

                            int buf_base = (ii-i) * b + jj - j;
                            int base = jj * N + ii;
                            
                            double tmp1 = x0d0 - yd0;
                            double tmp2 = x0d1 - yd1;
                            double dist1 = tmp1 * tmp1 + tmp2 * tmp2;
                            
                            double tmp3 = x1d0 - yd0;
                            double tmp4 = x1d1 - yd1;
                            double dist2 = tmp3 * tmp3+ tmp4 * tmp4;
                            
                            double tmp5 = x2d0 - yd0;
                            double tmp6 = x2d1 - yd1;
                            double dist3 = tmp5 * tmp5 + tmp6 * tmp6;

                            double tmp7 = x3d0 - yd0;
                            double tmp8 = x3d1 - yd1;
                            double dist4 = tmp7 * tmp7 + tmp8 * tmp8;

                            double tmp9 = x4d0 - yd0;
                            double tmp10 = x4d1 - yd1;
                            double dist5 = tmp9 * tmp9 + tmp10 * tmp10;
                            
                            double tmp11 = x5d0 - yd0;
                            double tmp12 = x5d1 - yd1;
                            double dist6 = tmp11 * tmp11+ tmp12 * tmp12;
                            
                            double tmp13 = x6d0 - yd0;
                            double tmp14 = x6d1 - yd1;
                            double dist7 = tmp13 * tmp13 + tmp14 * tmp14;

                            double tmp15 = x7d0 - yd0;
                            double tmp16 = x7d1 - yd1;
                            double dist8 = tmp15 * tmp15 + tmp16 * tmp16;
                            
                            DD[base] = dist1;
                            DD[base + 1] = dist2;
                            DD[base + 2] = dist3;
                            DD[base + 3] = dist4;
                            DD[base + 4] = dist5;
                            DD[base + 5] = dist6;
                            DD[base + 6] = dist7;
                            DD[base + 7] = dist8;

                            buf[buf_base] = dist1; buf_base += b;
                            buf[buf_base] = dist2; buf_base += b;
                            buf[buf_base] = dist3; buf_base += b;
                            buf[buf_base] = dist4; buf_base += b;
                            buf[buf_base] = dist5; buf_base += b;
                            buf[buf_base] = dist6; buf_base += b;
                            buf[buf_base] = dist7; buf_base += b;
                            buf[buf_base] = dist8; 
                    }
                }

                // copy the buffer back to DD
                for(int ii = i; ii < i + b; ii++) {
                    int base = ii * N;
                    int buf_base = (ii - i) * b;
                    for(int jj = j; jj < j + b; jj++) { 
                        DD[base + jj] = buf[buf_base + jj - j];
                    }
                }
            }
        }
    }
}

namespace computeSEDv2d2ruvec{ // with blocking for cache AND register w unrolling
    typedef __m256d d256;
    void computeSquaredEuclideanDistance(const double* X, int N,  int D, double* DD) {
       const int b = 16; // block size for cache
       const int rbi = 4; // block size for registers, in the following unrolling, assume rbi = 4, o/w it won't work
       const int rbj = 16; // block size for registers

        const double* Xi = X;
        for(int i = 0; i < N - b + 1; i += b, Xi += b * D) {
            const double* Xj = X + i * D;
            for(int j = i; j < N - b + 1; j += b, Xj += b * D) {
                const double* Xii = Xi;
                if(i == j) { // diagonal block
                    for(int ii = i; ii < i + b; ii++, Xii += D) {
                        const double* Xjj = Xj;
                        double xii0 = Xii[0], xii1 = Xii[1];
                        int base = ii * N;
                        for(int jj = j; jj < j + b; jj++, Xjj += D) {
                                if(ii == jj) {
                                    // DD[base + jj] = 0.0;
                                    DD[base + jj] = 0.0;
                                    continue;
                                }

                                double tmp1 = xii0 - Xjj[0];
                                double tmp2 = xii1 - Xjj[1];
                                double dist = tmp1 * tmp1 + tmp2 * tmp2;
                                
                                DD[base + jj] = dist;
                        }
                    }
                    continue;
               }
                for(int ii = i; ii < i + b - rbi + 1; ii += rbi, Xii += rbi * D) {
                    const double* Xjj = Xj;
                    
                    d256 x01 = _mm256_load_pd(Xii);
                    d256 x23 = _mm256_load_pd(Xii + 4);
                    d256 xvecd0 = _mm256_unpacklo_pd(x01, x23); // dim 0 
                    d256 xvecd1 = _mm256_unpackhi_pd(x01, x23); // dim 1
                    xvecd0 = _mm256_permute4x64_pd(xvecd0, 0b11011000);
                    xvecd1 = _mm256_permute4x64_pd(xvecd1, 0b11011000);

                    for(int jj = j; jj < j + b; jj++, Xjj += D) {
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

namespace computeSEDv2d2ruvecbuf{ // with blocking for cache AND register w unrolling
    typedef __m256d d256;
    void computeSquaredEuclideanDistance(const double* X, int N,  int D, double* DD) {
       const int b = 32; // block size for cache
       const int rbi = 4; // block size for registers, in the following unrolling, assume rbi = 4, o/w it won't work
       const int rbj = 16; // block size for registers
       double buf[b * b] __attribute__ ((aligned (32)));

        const double* Xi = X;
        for(int i = 0; i < N - b + 1; i += b, Xi += b * D) {
            const double* Xj = X + i * D;
            for(int j = i; j < N - b + 1; j += b, Xj += b * D) {
                const double* Xii = Xi;
                if(i == j) { // diagonal block
                    for(int ii = i; ii < i + b; ii++, Xii += D) {
                        const double* Xjj = Xj;
                        double xii0 = Xii[0], xii1 = Xii[1];
                        int base = ii * N;
                        for(int jj = j; jj < j + b; jj++, Xjj += D) {
                                if(ii == jj) {
                                    // DD[base + jj] = 0.0;
                                    DD[base + jj] = 0.0;
                                    continue;
                                }

                                double tmp1 = xii0 - Xjj[0];
                                double tmp2 = xii1 - Xjj[1];
                                double dist = tmp1 * tmp1 + tmp2 * tmp2;
                                
                                DD[base + jj] = dist;
                        }
                    }
                    continue;
               }
                for(int ii = i; ii < i + b - rbi + 1; ii += rbi, Xii += rbi * D) {
                    const double* Xjj = Xj;
                    
                    d256 x01 = _mm256_load_pd(Xii);
                    d256 x23 = _mm256_load_pd(Xii + 4);
                    d256 xvecd0 = _mm256_unpacklo_pd(x01, x23); // dim 0 
                    d256 xvecd1 = _mm256_unpackhi_pd(x01, x23); // dim 1
                    xvecd0 = _mm256_permute4x64_pd(xvecd0, 0b11011000);
                    xvecd1 = _mm256_permute4x64_pd(xvecd1, 0b11011000);

                    for(int jj = j; jj < j + b; jj++, Xjj += D) {
                        // load y
                        d256 yvecd0 = _mm256_set1_pd(Xjj[0]);
                        d256 yvecd1 = _mm256_set1_pd(Xjj[1]);

                       d256 xyd0  = _mm256_sub_pd(xvecd0, yvecd0);
                        d256 xyd1  = _mm256_sub_pd(xvecd1, yvecd1);

                        xyd0 = _mm256_mul_pd(xyd0, xyd0);
                        xyd1 = _mm256_mul_pd(xyd1, xyd1);

                        d256 xy = _mm256_add_pd(xyd0, xyd1);

                        const int symm_base = (jj - j) * b + ii - i;
                        _mm256_store_pd(buf + symm_base, xy);

                        int base = ii * N + jj;
                        DD[base] = buf[symm_base    ]; base += N;
                        DD[base] = buf[symm_base + 1]; base += N;
                        DD[base] = buf[symm_base + 2]; base += N;
                        DD[base] = buf[symm_base + 3];
                    }
                }

                // copy the buffer back to DD
                for(int jj = j; jj < j + b; jj++) { 
                    int base = jj * N;
                    int buf_base = (jj - j) * b;
                    for(int ii = i; ii < i + b; ii++) {
                        DD[base + ii] = buf[buf_base + ii - i];
                    }
                }
            }
        }
    }
}