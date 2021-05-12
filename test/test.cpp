#include "test.hpp"
#include "test_updgradient.hpp"
#include "test_zeromean.hpp"
#include "test_computegp.hpp"
#include "test_computesed.hpp"

/// plug in your implementation using corresponding namespaces
using namespace computeGPv1;
using namespace updateGradientv3;
using namespace zeroMeanv1;
//using namespace computeSEDv2d2ru;
using namespace computeSEDv1;

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
    
    for(int i = 1; i < argc; i++) {
        string to_test(argv[i]);
        cout << "Begin Test " << to_test << endl;
        if(to_test == "computegp") {
            test = new Test_ComputeGP(computeGaussianPerplexity);
        }
        else if(to_test == "updgradient") {
            test = new Test_UpdGradient(updateGradient);
        }
        else if(to_test == "zeromean") {
            test = new Test_ZeroMean(zeroMean);
        }
        else if(to_test == "computesed") {
            test = new Test_ComputeSED(computeSquaredEuclideanDistance);
            std::vector<std::string> names {"Baseline", "Blocking", "Blocking++", "Vectorization"}; 
            std::vector<Test_ComputeSED::comp_func> func_to_test {
                computeSEDv1::computeSquaredEuclideanDistance,
                computeSEDv2d2::computeSquaredEuclideanDistance,
                computeSEDv2d2ru::computeSquaredEuclideanDistance,
                computeSEDv2d2ruvec::computeSquaredEuclideanDistance
            };
            std::vector<int> size_to_test {64, 128, 256, 512, 1024, 2048};
            //static_cast<Test_ComputeSED*>(test)->sweep_input_size();
            static_cast<Test_ComputeSED*>(test)->sweep(func_to_test, names, size_to_test);
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
