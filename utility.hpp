#pragma once
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <random>
#include <iostream>
#include <fstream>
#include "parameters.hpp"
#include "kernels/computeSED.hpp"

double randn() {
	double x, y, radius;
	do {
		x = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
		y = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
		radius = (x * x) + (y * y);
	} while((radius >= 1.0) || (radius == 0.0));
	radius = sqrt(-2 * log(radius) / radius);
	x *= radius;
	y *= radius;
	return x;
}

bool load_data(double* data) {
    printf("Begin loading data\n");

    // generate data instead
    using namespace std;
    int num_clusters = 8;
    double stddev = 1.0;

    default_random_engine generator;
    for(int i = 0, c = 0; i < N; i++, c = (c+1) % num_clusters) { // generate in a round robin manner
        normal_distribution<double> distribution(c*10.0+1, stddev);
        for(int j = 0; j < in_dim; j++) {
            data[i*in_dim+j] = distribution(generator);
        }
    }

    printf("Finish generating data\n");

    return true;
}

// Function that saves map to a t-SNE file
void save_data(double* data) {
    printf("Begin saving data\n");

    using namespace std;
    ofstream myfile;
    myfile.open ("output.txt");
    for(int i = 0; i < N; i++) {
        myfile << data[i*out_dim] << " " << data[i*out_dim+1] << endl;
    }
    myfile.close();
    printf("End saving data\n");
}

double evaluateError(double* P, double* Y, int N, int D) {

    // Compute the squared Euclidean distance matrix
    double* DD = (double*) malloc(N * N * sizeof(double));
    double* Q = (double*) malloc(N * N * sizeof(double));
    if(DD == NULL || Q == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    computeSEDv1::computeSquaredEuclideanDistance(Y, N, D, DD);

    // Compute Q-matrix and normalization sum
    int nN = 0;
    double sum_Q = DBL_MIN;
    for(int n = 0; n < N; n++) {
    	for(int m = 0; m < N; m++) {
            if(n != m) {
                Q[nN + m] = 1 / (1 + DD[nN + m]);
                sum_Q += Q[nN + m];
            }
            else Q[nN + m] = DBL_MIN;
        }
        nN += N;
    }
    for(int i = 0; i < N * N; i++) Q[i] /= sum_Q;

    // Sum t-SNE error
    double C = .0;
	for(int n = 0; n < N * N; n++) {
        C += P[n] * log((P[n] + FLT_MIN) / (Q[n] + FLT_MIN));
	}

    // Clean up memory
    free(DD);
    free(Q);
	return C;
}
