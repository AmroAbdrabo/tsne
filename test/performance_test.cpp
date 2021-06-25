#include "test.hpp"
#include "test_updgradient.hpp"
#include "test_zeromean.hpp"
//#include "test_computegp.hpp"
#include "test_computesed.hpp"
#include "test_updgradient_zeromean.hpp"
#include "../memory/Pool.h"

/// plug in your implementation using corresponding namespaces
//using namespace computeGPv1;
using namespace updateGradientv4_2_outdim;
using namespace zeroMeanv6;
using namespace computeSEDv1;
using namespace updateGradient_zeroMeanv7_d2;

void usage() {
    cout << "[Usage] ./test <kernel_name1> <kernel_name2> ...\n";
    cout << "Valid kernel names : computegp/updgradient/zeromean/computesed\n";
}

int main(int argc, char** argv) {
    if(argc < 2) {
        cout << "require at least one kernel to test!" << endl;
        usage();
        exit(1);
    }
    Test* test = nullptr;
    
    
    string to_test(argv[1]);
    cout << "Begin Test " << to_test << endl;
    // if(to_test == "computegp") {
    //     test = new Test_ComputeGP(computeGaussianPerplexity);
    // }
    if(to_test == "updgradient") {
        test = new Test_UpdGradient(updateGradient);
    }
    else if(to_test == "zeromean") {
        test = new Test_ZeroMean(zeroMean);
    }
    else if(to_test == "computesed") {
        test = new Test_ComputeSED(computeSquaredEuclideanDistance);
    }
    else if(to_test == "updgradient_zeromean"){
        test = new Test_UpdGradient_ZeroMean(updateGradient_zeroMean);
    }
    else {
        cout << "Invalid kernel name!\n";
        usage();
        exit(1);
    }

    const int reps = 2;
    double total_cycles[18];
    double cycles;
    int N_;
    int j = 0;
    for(N_ = 16; N_ <= N; N_ *= 2){
        printf("\n[N_ = %i]: ", N_);
        for(int i = 0; i < reps; ++i){
            cycles = test->perf_test_2(N_);
            total_cycles[j] += cycles;
            printf("%f, ", cycles);
        }
        ++j;
        total_cycles[j] /= reps;
    }
    
    //write into file
    FILE *pFile;
    pFile = fopen("plots/zMv6.txt","w");
    N_ = 16;
    for(int i = 0; i < 18; ++i){
        fprintf(pFile, "%i,%e\n", N_, total_cycles[i]);
        N_ *= 2;
    }
    fclose (pFile);

    free(test);
    test = nullptr;
}
