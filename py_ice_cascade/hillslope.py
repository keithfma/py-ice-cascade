"""
Python ICE-CASCADE hillslope erosion-deposition model component

References:

    (1) Becker, T. W., & Kaus, B. J. P. (2016). Numerical Modeling of Earth
    Systems: Lecture Notes for USC GEOL557 (1.1.4)
"""

# TODO: How to treat wet or icy points?

import numpy as np

class diffuse_ftcs_open():
    """
    Hillslope diffusion model using forward-time center-space (FTCS) finite
    diffence scheme with "open" boundary conditions. 
    
    For an overview of FTCS see reference (1). 
    
    At the "open" boundaries, incoming and outgoing flux normal to the boundary
    is equal. In other words, material is allowed to pass through the boundary
    node. This condition means dq/dx = 0, and the boundary-normal component of
    the diffusion equation goes to 0 as well. Note that boundary-parallel flux
    gradients are not necessarily 0, and so boundary heights are not constant. 
    """

    def __init__(self, height, delta, kappa):
        """
        Initialize new model object

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
        max_step = 0.5*self.delta*self.delta/self.kappa # stable time step, see ref (1)
        ddhx = np.zeros(self.height.shape) # arrays for 2nd derivative terms
        ddhy = np.zeros(self.height.shape)

        while time < run_time:
            step = min(run_time-time, max_step)
            ddhx[:,1:-1] = self.height[:,2:] - 2.0*self.height[:,1:-1] + self.height[:,:-2]
            ddhy[1:-1,:] = self.height[2:,:] - 2.0*self.height[1:-1,:] + self.height[:-2,:]
            cc = self.kappa*step/self.delta/self.delta
            self.height += cc*(ddhx+ddhy) 
            time += step

# basic usage example 
if __name__ == '__main__':
    
    hh = np.ones((100,100), dtype=np.double)
    dd = np.double(10.0)
    kk = np.double(1.0)

    model = diffuse_ftcs_open(hh, dd, kk)
    model.run(100)
