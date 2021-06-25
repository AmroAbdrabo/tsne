import matplotlib
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


    # plt.plot(n, (flops(n)-12*n)/i1  , 'o--', label='Base implementation')
    # #plt.plot(n, (flops(n)-12*n)/i2 , 'o--', label='v2 d=x')
    # plt.plot(n, flops(n)/i32, 'o--', label='All optimized kernels (without unwrapping)')
    # #plt.plot(n, flops(n)/i3x , 'o--', label='v3 d=x')
    # plt.plot(n, flops(n)/i4  , 'ro-', label='Combine and optimize kernels', linewidth=3.0)
    # #plt.plot(n, flops(n)/i5 , 'o--', label='v5 d=2')
    # plt.plot(n, flops(n)/i7 , 'o--', label='Strict upper triangular matrices')

    pi   = 6            #Peak performance [flops/cycle]
    beta = 6.474        #Memory bandwidth [bytes/cycle]

    pi_beta = pi/beta   #[flops/cycle]

    mb_intensity   = np.linspace(1.0/32.0, pi_beta, 2, dtype=np.float64)
    mb_performance = beta*mb_intensity

    cb_intensity   = np.linspace(pi_beta, 4.0, 2, dtype=np.float64)
    cb_performance = np.full((2,), pi)

    ax.plot(mb_intensity, mb_performance, 'k-')
    ax.plot(cb_intensity, cb_performance, 'k-')

    ax.set_yscale("log")
    ax.set_xscale("log")

    
    ax.set_yticks([2e-4, 2e-3, 2e-2, 2e-1, 2e0, 2e1, 2e2, 2e3])
    ax.set_xticks([2e-5, 2e-4, 2e-3, 2e-2, 2e-1, 2e0, 2e1, 2e2])

    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    #ax.get_xaxis().get_major_formatter().labelOnlyBase = False

    plt.xlabel('I(N) [flops/byte]')
    plt.ylabel("P(N) [flops/cycle]", rotation=0)

    ax.yaxis.set_label_coords(0.05,1.01)
    ax.xaxis.grid()

    plt.title('Roofline plot of updateGradient_zeroMean kernel', loc='left', pad=25, fontweight="bold")
    
    plt.legend()
    plt.plot()
    plt.show()
    #plt.savefig(sys.argv[2])
    plt.close()
