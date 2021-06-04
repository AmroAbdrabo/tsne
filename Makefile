CXX=g++
CXXFLAGS=-O3 -march=native -mtune=native -mavx2 -mfma -ffast-math -fno-tree-vectorize -std=c++17 -g
INC=-I.

tsne: init main.cpp
	$(CXX) -o $@ main.cpp $(CXXFLAGS)
	mv $@ build/

kernel_test: init test/test.cpp
	$(CXX) -o $@ test/test.cpp $(CXXFLAGS) $(INC)
	mv $@ build/

performance_test: init test/performance_test.cpp
	$(CXX) -o $@ test/performance_test.cpp $(CXXFLAGS) $(INC)
	mv $@ build/

init:
	mkdir -p build/

.PHONY: init

