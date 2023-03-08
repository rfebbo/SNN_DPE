import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import openml

# Create an image of the Lorenz attractor.
# The maths behind this code is described in the scipython blog article
# at https://scipython.com/blog/the-lorenz-attractor/
# Christian Hill, January 2016.
# Updated, January 2021 to use scipy.integrate.solve_ivp.

class Lorenz_Dataset:
    
    def __init__(self,sigma=10,beta=8/3,rho=28): 

        # Lorenz paramters and initial conditions.
        self.sigma=sigma
        self.beta=beta
        self.rho = rho

    def lorenz(self, t, Crd, sigma, beta, rho):
        """The Lorenz equations from https://en.wikipedia.org/wiki/Lorenz_system"""
        x, y, z = Crd
        dxdt = sigma*(y-x)
        dydt = x*(rho-z) - y
        dzdt = x*y-beta*z 
        return dxdt,dydt,dzdt

    def create_dataset(self,x0=0,y0=1,z0=1.05,tmax=100,n = 10000):
        "Inspired From: https://scipython.com/blog/the-lorenz-attractor/"
        # Create n Time space from 0 to 100
        
        t = np.linspace(0, tmax, n)

        # Integrate the Lorenz equations.
        soln = solve_ivp(self.lorenz, (0, tmax), (x0, y0, z0), args=(self.sigma, self.beta, self.rho),
                         dense_output=True)

        """X,Y,Z: The array of coordinate X,Y,X for all time step"""

        self.X, self.Y, self.Z = soln.sol(t)
        
        self.full_data=np.zeros((n,3))
        
        self.full_data[:,0],self.full_data[:,1],self.full_data[:,2]=self.X,self.Y,self.Z
        
    def plot_data(self, WIDTH=1000, HEIGHT=750, DPI=100,s = 10,save=False, Filename='lorenz.png'):
        
        # Plot the Lorenz attractor using a Matplotlib 3D projection.
        fig = plt.figure(facecolor='k', figsize=(WIDTH/DPI, HEIGHT/DPI))
        ax = plt.axes(projection='3d')
        ax.set_facecolor('k')
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        # Make the line multi-coloured by plotting it in segments of length s which
        # change in colour across the whole time series.
        cmap = plt.cm.winter
        for i in range(0,len(self.X)-s,s):
            ax.plot3D(self.X[i:i+s+1], self.Y[i:i+s+1], self.Z[i:i+s+1], color=cmap(i/len(self.X)), alpha=0.4)

        # Remove all the axis clutter, leaving just the curve.
        ax.set_axis_off()
        
        if save==True:
            plt.savefig(Filename, dpi=DPI)
        plt.show()
        
    def create_openml_lorenz_data(self):
        self.openml_full_data = openml.datasets.get_dataset(42182).get_data(dataset_format='array')[0]   