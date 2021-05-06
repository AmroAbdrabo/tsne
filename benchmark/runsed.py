import numpy as np
import time
from subprocess import call
import os
import sys


#os.system("gcc -O3 -fno-tree-vectorize maxperformance.c")
os.system("g++ -std=c++11 benchComputeSED.cpp -o bcsed")
input_sizes = np.arange(100, 4000, 150)
## get the name of the file to write to as an argument to pass to the c code
for i in input_sizes:
    for j in range(5):
        call(["./bcsed", str(i), str(sys.argv[1])])
        


