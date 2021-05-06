#ifndef WIN32
#include "./../kernels/computeGP.hpp"
#include "./../kernels/computeSED.hpp"
#include <sys/time.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "../common/tsc_x86.h"
#include "../parameters.hpp"

#define NUM_RUNS 1
#define CYCLES_REQUIRED 1e8
#define FREQUENCY 2.3e9
#define CALIBRATE


double rdtsc(const double* X, double* P, int n) {
    int i, num_runs;
    myInt64 cycles;
    myInt64 start;
    num_runs = NUM_RUNS;
    double temp;

    /*
     * The CPUID instruction serializes the pipeline.
     * Using it, we can create execution barriers around the code we want to time.
     * The calibrate section is used to make the computation large enough so as to
     * avoid measurements bias due to the timing overhead.
     */
#ifdef CALIBRATE
    while(num_runs < (1 << 14)) {
        start = start_tsc();
        for (i = 0; i < num_runs; ++i) {
            computeSEDv1::computeSquaredEuclideanDistance(X, n, in_dim, P);
        }
        cycles = stop_tsc(start);

        if(cycles >= CYCLES_REQUIRED) break;

        num_runs *= 2;
    }
#endif

    start = start_tsc();
    for (i = 0; i < num_runs; ++i) {
         computeSEDv1::computeSquaredEuclideanDistance(X, n, in_dim, P);
    }

    cycles = stop_tsc(start)/num_runs;
    for (int i = 0; i < 10; ++i){
        temp += P[i];
    }
    printf("done with %lf \n", temp);
   
    return (double) cycles;
}


int main(int argc, char **argv) {
    if (argc!=3) {
        printf("Please enter filename and input size: example ./a.out 100 file.txt \n");
        return -1;
        
    }
    int N = atoi(argv[1]);
    char *filename = argv[2];
    
    
    
    double* X = (double*)malloc(N*in_dim*sizeof(double));
    double* P = (double*)malloc(N*N*sizeof(double));
    init_rand_vec(X, N*in_dim);
    init_rand_vec(P, N*N);
    
    FILE *fptr;
    fptr = fopen(filename,"a+");
    
    double r = rdtsc(X, P, N);
    fprintf(fptr,"%lf\n",r);
    fclose(fptr);
    
    free(X);
    free(P);
    return 0;
}



