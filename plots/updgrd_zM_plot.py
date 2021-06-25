import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

sns.set_theme()
sns.set(font="sans serif")

ax = plt.gca()

plt.rcParams['font.size'] = 10

def flops(n):
    return 14.5*n*n + 13.5*n + 2

if __name__ == "__main__":

    #data
    n   = np.loadtxt("updGr_zM_v1.txt", dtype=np.float64, delimiter=',')[:,0]
    i1  = np.loadtxt("updGr_zM_v1.txt", dtype=np.float64, delimiter=',')[:,1]
    i2  = np.loadtxt("updGr_zM_v2.txt", dtype=np.float64, delimiter=',')[:,1]
    i32 = np.loadtxt("updGr_zM_v3_d2.txt", dtype=np.float64, delimiter=',')[:,1]
    i3x = np.loadtxt("updGr_zM_v3_dx.txt", dtype=np.float64, delimiter=',')[:,1]
    i4  = np.loadtxt("updGr_zM_v4_d2.txt", dtype=np.float64, delimiter=',')[:,1]
    i5   = np.loadtxt("updGr_zM_v5_d2.txt", dtype=np.float64, delimiter=',')[:,1]
    i7  = np.loadtxt("updGr_zM_v7_d2.txt", dtype=np.float64, delimiter=',')[:,1]


    plt.plot(n, (flops(n)-12*n)/i1  , 'o--', label='Base implementation')
    #plt.plot(n, (flops(n)-12*n)/i2 , 'o--', label='v2 d=x')
    plt.plot(n, flops(n)/i32, 'o--', label='All optimized kernels (without unwrapping)')
    #plt.plot(n, flops(n)/i3x , 'o--', label='v3 d=x')
    #plt.plot(n, flops(n)/i4  , 'o--', label='v4 d=2')
    plt.plot(n, flops(n)/i5 , 'ro-', label='Combine and optimize kernels', linewidth=3.0)
    plt.plot(n, flops(n)/i7 , 'o--', label='Strict upper triangular matrices')

    plt.vlines(35, ymin=0.0, ymax=3.0)
    plt.vlines(144, ymin=0.0, ymax=3.0)
    plt.vlines(1153, ymin=0.0, ymax=3.0)

    shift = 1.4
    plt.text(35 / shift, 0.05, 'L1')
    plt.text(144 / shift, 0.05, 'L2')
    plt.text(1153 / shift, 0.05, 'L3')


    plt.xlabel('Input size N')
    plt.ylabel("Performance [flops/cycle]", rotation=0)

    ax.yaxis.set_label_coords(0.05,1.01)
    ax.xaxis.grid()
    ax.set_xscale('log', basex=2)

    plt.title('updateGradient_zeroMean kernel on Ryzen 9 5950X 3.4GHz', loc='left', pad=25, fontweight="bold")
    
    #plt.legend()
    ax.legend().set_visible(False)
    plt.plot()
    plt.show()
    #plt.savefig(sys.argv[2])
    plt.close()
