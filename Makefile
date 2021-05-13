CXX=g++
CXXFLAGS=-O3 -march=native -ffast-math -std=c++17 -fno-tree-vectorize
INC=-I.

tsne: init main.cpp
	$(CXX) -o $@ main.cpp $(CXXFLAGS) $(INC)
	mv $@ build/

kernel_test: init test/test.cpp
	$(CXX) -o $@ test/test.cpp $(CXXFLAGS) $(INC)
	mv $@ build/

init:
	mkdir -p build/

.PHONY: init
