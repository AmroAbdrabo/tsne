CXX=g++-10
CXXFLAGS=-O3 -ffast-math -march=native -mavx2  -mfma -mtune=native -fno-tree-vectorize -std=c++14
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

