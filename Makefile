CXX=g++-10
CXXFLAGS=-O3 -mavx2  -march=native -mtune=native -mfma -mtune=native -fno-tree-vectorize -std=c++17
INC=-I.

tsne: init main.cpp
	$(CXX) -o $@ main.cpp memory/Pool.cpp $(CXXFLAGS) $(INC)
	mv $@ build/

kernel_test: init test/test.cpp
	$(CXX) -o $@ test/test.cpp memory/Pool.cpp $(CXXFLAGS) $(INC)
	mv $@ build/

performance_test: init test/performance_test.cpp
	$(CXX) -o $@ test/performance_test.cpp $(CXXFLAGS) $(INC)
	mv $@ build/

init:
	mkdir -p build/

.PHONY: init
