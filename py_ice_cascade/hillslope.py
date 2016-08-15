"""
Python ICE-CASCADE hillslope erosion-deposition model component

References:

    (1) Becker, T. W., & Kaus, B. J. P. (2016). Numerical Modeling of Earth
    Systems: Lecture Notes for USC GEOL557 (1.1.4)
"""

import numpy as np
import matplotlib.pyplot as plt
import time

class ftcs_openbnd():
    """
    Hillslope diffusion model using forward-time center-space (FTCS) finite
    diffence scheme with "open" boundary conditions. 
    
    For an overview of FTCS see reference (1). 
    
    At the "open" boundaries, incoming and outgoing flux normal to the boundary
    is equal. In other words, material is allowed to pass through the boundary
    node. This condition means dq/dx = 0, and the boundary-normal component of
    the diffusion equation goes to 0 as well. Note that boundary-parallel flux
    gradients are not necessarily 0, and so boundary heights may not be
    constant. 
    """

    def __init__(self, height, delta, kappa):
        """
        Arguments:
            height = 2D Numpy array, surface elevation in model domain, [m]
            delta = Scalar double, grid spacing, assumed square, [m]
            kappa = Scalar double, diffusion coefficient, [m**2/a]
        """
        
        # declare attributes
        self.height = None # elevation grid
        self.delta = None  # grid spacing
        self.kappa = None  # diffusion parameter
        
        # set attributes
        self.delta = np.double(delta)
        self.kappa = np.double(kappa) 
        self.set_height(height) 
        
    def set_height(self, new):
        self.height = np.atleast_2d(np.array(new, dtype=np.double))

    def get_height(self):
        return self.height

    def run(self, run_time):
        """
        Run numerical integration for specified time period

        Arguments:
            run_time = Scalar double, model run time, [a]
        """
        
        run_time = np.double(run_time)
        time = np.double(0.0)    
        max_step = 0.95*self.delta*self.delta/(4.0*self.kappa) # stable time step, note ref (1) has error
        ddhx = np.zeros(self.height.shape) # arrays for 2nd derivative terms
        ddhy = np.zeros(self.height.shape)

        while time < run_time:
            step = min(run_time-time, max_step)
            ddhx[:,1:-1] = self.height[:,2:] - 2.0*self.height[:,1:-1] + self.height[:,:-2]
            ddhy[1:-1,:] = self.height[2:,:] - 2.0*self.height[1:-1,:] + self.height[:-2,:]
            coeff = step*self.kappa/self.delta/self.delta
            self.height += coeff*(ddhx+ddhy)
            time += step

if __name__ == '__main__':
    
    # basic usage example and "smell test": relaxation to height==0 steady state
    # # initialize model
    nx = 100
    ny = 100
    max_time = 5.0
    time_step = 0.05
    h0 = np.random.rand(nx, ny).astype(np.double)-0.5
    h0[:,0] = np.double(0.0) # constant values are compatible with "open" BC treatment
    h0[:,-1] = np.double(0.0)
    h0[0,:] = np.double(0.0)
    h0[-1,:] = np.double(0.0)
    dd = np.double(1.0)
    kk = np.double(1.0)
    model = ftcs_openbnd(h0, dd, kk)
    # # update and plot model
    plt.imshow(model.get_height(), interpolation='nearest', clim=(-0.5,0.5))
    plt.colorbar()
    plt.ion()
    time = 0.0
    while time < max_time: 
        model.run(time_step)
        time += time_step
        plt.cla()
        plt.imshow(model.get_height(), interpolation='nearest', clim=(-0.5,0.5))
        plt.title("TIME = {:.2f}".format(time))
        plt.pause(0.05)
