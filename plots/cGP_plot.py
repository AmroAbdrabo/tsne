import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

sns.set_theme()
sns.set(font="sans serif")

ax = plt.gca()

plt.rcParams['font.size'] = 10

def flops(n):
    iter = 200.0
    #return n*iter*(7.0*n + 9.0) + 3.0*n*n*n
    return n * iter * (21*n + 27) + 3* n**3

if __name__ == "__main__":

    #data
    n   = np.array([16, 32, 64, 128, 256, 512, 1024])
    i0  = np.array([1298593.395600, 5105709.790000, 21019049.872800, 86413815.288400, 339342477.631200, 1365672515.568800, 5474817792.133201])
    i1  = np.array([1182623.632000, 4791367.842400, 19037700.981200, 76249158.778000, 294747944.854400, 1166518718.101200, 4505519622.244800])
    #i2  = np.array([487600.528670, 1440422.076800, 4928710.618000, 18631271.971600])
    i3  = np.array([492277.516505, 1443944.687600, 4947015.422400, 18667820.033600, 81611942.704000, 338074369.832400, 1347571232.095200])



    plt.plot(n, flops(n)/i0 , 'o-', label='Base implementation')
    plt.plot(n, flops(n)/i1 , 'o-', label='not vectorized-all other optimizations')
    #plt.plot(n[:4], flops(n[:4])/i2, 'o--', label='vectorized')
    plt.plot(n, flops(n)/i3 , 'ro-', label='vectorized', linewidth=3.0)
    

    plt.vlines(44, ymin=0.0, ymax=6)
    plt.vlines(178, ymin=0.0, ymax=6)
    plt.vlines(1414, ymin=0.0, ymax=6)

    shift = 1.4
    plt.text(44 / shift, 0.1, 'L1')
    plt.text(178 / shift, 0.1, 'L2')
    plt.text(1414 / shift, 0.1, 'L3')


    plt.xlabel('Input size N')
    plt.ylabel("Performance [flops/cycle]", rotation=0)

    ax.yaxis.set_label_coords(0.05,1.01)
    ax.xaxis.grid()
    ax.set_xscale('log', basex=2)

    plt.title('computeGP kernel on Ryzen 9 5950X 3.4GHz', loc='left', pad=25, fontweight="bold")
    
    #plt.legend()
    ax.legend().set_visible(False)
    plt.plot()
    plt.show()
    #plt.savefig(sys.argv[2])
    plt.close()
