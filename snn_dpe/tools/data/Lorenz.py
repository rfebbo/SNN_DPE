import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp

# import openml

# Create an image of the Lorenz attractor.
# The maths behind this code is described in the scipython blog article
# at https://scipython.com/blog/the-lorenz-attractor/
# Christian Hill, January 2016.
# Updated, January 2021 to use scipy.integrate.solve_ivp.

def lorenz( t, Crd, sigma, beta, rho):
    """The Lorenz equations from https://en.wikipedia.org/wiki/Lorenz_system"""
    x, y, z = Crd
    dxdt = sigma*(y-x)
    dydt = x*(rho-z) - y
    dzdt = x*y-beta*z 
    return dxdt,dydt,dzdt

def create_dataset(x0=0,y0=1,z0=1.05,tmax=100,n = 10000, normalize=True, sigma=10,beta=8/3,rho=28):
    "Inspired From: https://scipython.com/blog/the-lorenz-attractor/"
    # Create n Time space from 0 to 100
    
    t = np.linspace(0, tmax, n)

    # Integrate the Lorenz equations.
    soln = solve_ivp(lorenz, (0, tmax), (x0, y0, z0), args=(sigma, beta, rho),
                        dense_output=True)

    """X,Y,Z: The array of coordinate X,Y,X for all time step"""

    X, Y, Z = soln.sol(t)
    
    full_data=np.zeros((n,3))
    
    full_data[:,0],full_data[:,1],full_data[:,2]=X,Y,Z

    if normalize:
        n_d = 3

        for dim in range(n_d):
            dmin = np.min(full_data[:,dim])
            dmax = np.max(full_data[:,dim])
            normalized_data = []
            for d in full_data[:,dim]:
                v = (d - dmin) / (dmax - dmin)
                normalized_data.append(v)

            full_data[:,dim] = np.asarray(normalized_data)

    return full_data
    
def plot_data(data, te_data=None, plt_len=200, WIDTH=1000, HEIGHT=750, DPI=100,s = 1,save=False, Filename='lorenz.png'):
    X = data[:, 0]
    Y = data[:, 1]
    Z = data[:, 2]

    if te_data is not None:
        Xte = te_data[:, 0]
        Yte = te_data[:, 1]
        Zte = te_data[:, 2]
        
    # Plot the Lorenz attractor using a Matplotlib 3D projection.
    fig = plt.figure(facecolor='w', figsize=(WIDTH/DPI, HEIGHT/DPI))
    ax = plt.axes(projection='3d')
    ax.set_facecolor('w')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # Make the line multi-coloured by plotting it in segments of length s which
    # change in colour across the whole time series.
    cmap = plt.cm.summer
    cmapte = plt.cm.winter
    # grab every 's' points
    for i in range(0,plt_len-s,s):
        ax.plot3D(X[i:i+s+1], Y[i:i+s+1], Z[i:i+s+1], color=cmap(i/plt_len), alpha=0.7)
        if te_data is not None:
            ax.plot3D(Xte[i:i+s+1], Yte[i:i+s+1], Zte[i:i+s+1], color=cmapte(i/plt_len), alpha=0.7)
    # Remove all the axis clutter, leaving just the curve.
    # ax.set_axis_off()
    
    if save==True:
        plt.savefig(Filename, dpi=DPI)
    plt.show()
    