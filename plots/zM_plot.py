import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

sns.set_theme()
sns.set(font="sans serif")

ax = plt.gca()

plt.rcParams['font.size'] = 10

def flops(n):
    return 4.0*n + 2.0

if __name__ == "__main__":

    #data
    n   = np.loadtxt("zMv1.txt", dtype=np.float64, delimiter=',')[:,0][:13]
    i1  = np.loadtxt("zMv1.txt", dtype=np.float64, delimiter=',')[:,1][:13]
    i12 = np.loadtxt("zMv1_d2.txt", dtype=np.float64, delimiter=',')[:,1][:13]
    #i3  = np.loadtxt("zMv3.txt", dtype=np.float64, delimiter=',')[:,1]
    #i4  = np.loadtxt("zMv4.txt", dtype=np.float64, delimiter=',')[:,1]
    #i5  = np.loadtxt("zMv5.txt", dtype=np.float64, delimiter=',')[:,1]
    i6  = np.loadtxt("zMv6.txt", dtype=np.float64, delimiter=',')[:,1][:13]


    plt.plot(n, flops(n)/i1  , 'o--', label='Base implementation')
    plt.plot(n, flops(n)/i12 , 'o--', label='Base implementation unrolled for D=2')
    #plt.plot(n, flops(n)/i3 , 'o--', label='AVX (D=x)')
    #plt.plot(n, flops(n)/i4  , 'o--', label='AVX optimized for Sky/IceLake')
    #plt.plot(n, flops(n)/i5 , 'o--', label='blocked (without AVX)')
    plt.plot(n, flops(n)/i6  , 'ro-', label='AVX optimized for Zen3', linewidth=3.0)


    plt.vlines(2000, ymin=0.0, ymax=3.0)
    plt.vlines(32000, ymin=0.0, ymax=3.0)
    #plt.vlines(2000000, ymin=0.0, ymax=3.0)

    shift = 1.4
    plt.text(2000 / shift, 2.9, 'L1')
    plt.text(32000 / shift, 2.9, 'L2')
    #plt.text(2000000 - shift, 0.1, 'L3')

    plt.xlabel('Input size N')
    plt.ylabel("Performance [flops/cycle]", rotation=0)

    ax.yaxis.set_label_coords(0.05,1.01)
    ax.xaxis.grid()
    ax.set_xscale('log', basex=2)

    plt.title('zeroMean kernel on Ryzen 9 5950X 3.4GHz', loc='left', pad=25, fontweight="bold")
    
    #plt.legend()
    ax.legend().set_visible(False)
    plt.plot()
    plt.show()
    #plt.savefig(sys.argv[2])
    plt.close()
