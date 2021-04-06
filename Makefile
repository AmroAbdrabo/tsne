CXX=g++
CXXFLAGS=-O3 -march=native -std=c++14
INC=-I.

tsne: init main.cpp
	$(CXX) -o $@ main.cpp $(CXXFLAGS)
	mv $@ build/

kernel_test: init test/test.cpp
	$(CXX) -o $@ test/test.cpp $(CXXFLAGS) $(INC)
	mv $@ build/

init:
	mkdir -p build/

.PHONY: init

# test_computegp: kernel_test.cpp
# 	$(CXX) -o $@ $< $(CXXFLAGS) -DTEST_COMPUTEGP

# test_updgradient: kernel_test.cpp
# 	$(CXX) -o $@ $< $(CXXFLAGS) -DTEST_UPDGRADIENT

# test_zeromean: kernel_test.cpp
# 	$(CXX) -o $@ $< $(CXXFLAGS) -DTEST_ZEROMEAN