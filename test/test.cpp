#include "test.hpp"
#include "test_updgradient.hpp"
#include "test_zeromean.hpp"
#include "test_computegp.hpp"

void usage() {
    cout << "[Usage] ./test <kernel_name1> <kernel_name2> ...\n";
    cout << "Valid kernel names : computegp/updgradient/zeromean\n";
}

int main(int argc, char** argv) {
    if(argc < 2) {
        cout << "require at least one kernel to test!" << endl;
        usage();
        exit(1);
    }
    Test* test = nullptr;
    
    for(int i = 1; i < argc; i++) {
        string to_test(argv[i]);
        cout << "Begin Test " << to_test << endl;
        if(to_test == "computegp") {
            test = new Test_ComputeGP(computeGPv1::computeGaussianPerplexity);
        }
        else if(to_test == "updgradient") {
            test = new Test_UpdGradient(upgradeGradientv1::upgradeGradient);
        }
        else if(to_test == "zeromean") {
            test = new Test_ZeroMean(zeroMeanv1::zeroMean);
        }
        else {
            cout << "Invalid kernel name!\n";
            usage();
            exit(1);
        }
        test->validate();
        test->perf_test();
        free(test);
        test = nullptr;
    }
}