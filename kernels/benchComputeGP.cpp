#ifndef WIN32
#include "computeGP.hpp"
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
            computeGPv1::computeGaussianPerplexity(X, N, in_dim, P, perp);
        }
        cycles = stop_tsc(start);

        if(cycles >= CYCLES_REQUIRED) break;

        num_runs *= 2;
    }
#endif

    start = start_tsc();
    for (i = 0; i < num_runs; ++i) {
         computeGPv1::computeGaussianPerplexity(X, N, in_dim, P, perp);
    }

    cycles = stop_tsc(start)/num_runs;
    printf("done with %lf ", temp);
   
    return (double) cycles;
}


int main(int argc, char **argv) {
    if (argc!=2) {
        printf("Please enter filename: example ./a.out file.txt \n");
        return -1;
        
    }
    //int n = atoi(argv[1]);
    char *filename = argv[1];
    
    double* X = (double*)malloc(N*in_dim*sizeof(double));
    double* P = (double*)malloc(N*N*sizeof(double));
    
    FILE *fptr;
    fptr = fopen(filename,"a+");
    
    double r = rdtsc(X, P, N);
    fprintf(fptr,"%lf\n",r);
    fclose(fptr);
    
    free(X);
    free(P);
    return 0;
}


