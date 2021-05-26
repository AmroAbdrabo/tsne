#include "test/test.hpp"
#include "kernels/computeSED.hpp"
using namespace computeSEDv2d2rubuf;
//using namespace computeSEDv1;

int main(int argc, char* argv[]) {
    int N = 16384;
    int dim = 2;
    double * X = static_cast<double*>(aligned_alloc(32, sizeof(double) * N * dim));
    rands(X, N, dim);
    double * DD = static_cast<double*>(aligned_alloc(32, sizeof(double) * N * N));
    //computeSEDv1::computeSquaredEuclideanDistance(X, N, dim, DD);
    //computeSEDv1::computeSquaredEuclideanDistance(X, N, dim, DD);
    //computeSEDv1::computeSquaredEuclideanDistance(X, N, dim, DD);
    //computeSEDv1::computeSquaredEuclideanDistance(X, N, dim, DD);
    //computeSEDv1::computeSquaredEuclideanDistance(X, N, dim, DD);
    //computeSEDv1::computeSquaredEuclideanDistance(X, N, dim, DD);

    computeSquaredEuclideanDistance(X, N, dim, DD);
    free(X);
    free(DD);
    return 0;
}
