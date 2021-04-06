import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    f = open("output.txt")
    x = []
    y = []
    for line in f.readlines():
        line = line.strip()
        entries = line.split(" ")
        x.append(float(entries[0]))
        y.append(float(entries[1]))
    plt.plot(x, y, ".")
    plt.show()