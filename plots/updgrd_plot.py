import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

sns.set_theme()
sns.set(font="sans serif")

ax = plt.gca()

plt.rcParams['font.size'] = 10



if __name__ == "__main__":

    #data
    n   = np.loadtxt("performance_plot_v1.txt", dtype=np.float64, delimiter=',')[:,0]
    i   = np.loadtxt("performance_plot_v1.txt", dtype=np.float64, delimiter=',')[:,1]
    ii  = np.loadtxt("performance_plot_v2_2_outdim.txt", dtype=np.float64, delimiter=',')[:,1]
    iii = np.loadtxt("performance_plot_v2_x_outdim.txt", dtype=np.float64, delimiter=',')[:,1]
    iv  = np.loadtxt("performance_plot_v3_2_outdim.txt", dtype=np.float64, delimiter=',')[:,1]
    v   = np.loadtxt("performance_plot_v3_x_outdim.txt", dtype=np.float64, delimiter=',')[:,1]
    vi  = np.loadtxt("performance_plot_v4_2_outdim.txt", dtype=np.float64, delimiter=',')[:,1]


    plt.plot(n, (12*n*n)/i  , 'o--', label='Base implementation')
    plt.plot(n, (12*(n*n + n))/ii , 'o--', label='Optimized (without AVX)')
    #plt.plot(n, (12*n*n + 3*n)/iii, 'o--', label='v2 d=x')
    plt.plot(n, (12*(n*n + n))/iv , 'ro-', label='AVX', linewidth=3.0)
    #plt.plot(n, (12*n*n + 3*n)/v  , 'o--', label='v3 d=x')
    plt.plot(n, (12*(n*n + n))/vi , 'o--', label='Strict upper triangular matrices')

    plt.vlines(35, ymin=0.0, ymax=2.5)
    plt.vlines(144, ymin=0.0, ymax=2.5)
    plt.vlines(1153, ymin=0.0, ymax=2.5)

    shift = 1.4
    plt.text(35 / shift, 0.1, 'L1')
    plt.text(144 / shift, 0.1, 'L2')
    plt.text(1153 / shift, 0.1, 'L3')

    

    plt.xlabel('Input size N')
    plt.ylabel("Performance [flops/cycle]", rotation=0)

    ax.yaxis.set_label_coords(0.05,1.01)
    ax.xaxis.grid()
    ax.set_xscale('log', basex=2)

    plt.title('updateGradient kernel on Ryzen 9 5950X 3.4GHz', loc='left', pad=25, fontweight="bold")
    
    #plt.legend()
    ax.legend().set_visible(False)
    plt.plot()
    plt.show()
    #plt.savefig(sys.argv[2])
    plt.close()
