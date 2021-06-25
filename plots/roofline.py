import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

"""
sns.set_theme()
sns.set(font="sans serif")

ax = plt.gca()

plt.rcParams['font.size'] = 10
"""

def flops(n):
    return 14.5*n*n + 13.5*n + 2

if __name__ == "__main__":
    fig, ax = plt.subplots()

    pi   = 6            #Peak performance [flops/cycle]
    beta = 6.474        #Memory bandwidth [bytes/cycle]
    #beta = 32

    pi_beta = pi/beta   #[flops/cycle]

    mb_intensity   = np.linspace(1.0/32.0, pi_beta, 2, dtype=np.float64)
    mb_performance = beta*mb_intensity

    cb_intensity   = np.linspace(pi_beta, 4.0, 2, dtype=np.float64)
    cb_performance = np.full((2,), pi)

    ax.plot(mb_intensity, mb_performance, 'k-')
    ax.plot(cb_intensity, cb_performance, 'k-')

    ax.plot(0.25, 0.95, 'yx', label="zeroMean")
    ax.plot(0.375, 1.625, 'cx', label="updateGradient")
    ax.plot(0.303, 0.91, 'mx', label="computeSED")
    ax.plot(0.598, 0.62, 'rx', label="SED + upGr + zMean")

    ax.set_yscale("log", basey=2)
    ax.set_xscale("log", basex=2)

    

    plt.xlabel('I(N) [flops/byte]')
    plt.ylabel("P(N) [flops/cycle]")

    # plt.yaxis.set_label_coords(0.05,1.01)
    # plt.xaxis.grid()

    plt.title('Roofline plot of Ryzen 9 5950X 3.4GHz', loc='left', pad=25, fontweight="bold")
    
    plt.legend()
    plt.grid()
    plt.plot()
    plt.show()
    #plt.savefig(sys.argv[2])
    plt.close()
