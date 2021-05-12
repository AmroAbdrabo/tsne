#include "test/test.hpp"
#include "kernels/computeSED.hpp"
using namespace computeSEDv2d2ru;
//using namespace computeSEDv1;

int main() {
    int N = 8192;
    int dim = 2;
    double * X = static_cast<double*>(aligned_alloc(32, sizeof(double) * N * dim));
    rands(X, N, dim);
    double * DD = static_cast<double*>(aligned_alloc(32, sizeof(double) * N * N));
    computeSquaredEuclideanDistance(X, N, dim, DD);
    return 0;
}
