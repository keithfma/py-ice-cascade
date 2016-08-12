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
        self.solver = None # numerical solver name
        
        # set attributes
        self.set_height(height) 
        self.set_delta(delta)
        self.set_kappa(kappa)
        
    def set_height(self, new):
        self.height = np.atleast_2d(np.array(new_height, dtype=np.double))

    def set_delta(self, new):
        self.delta = np.double(new)

    def set_kappa(self, new):
        self.kappa = np.double(new_kappa) 

    def get_height(self):
        return self.height

    def get_delta(self):
        return self.delta

    def get_kappa(self):
        return self.kappa

    def run(self, run_time):
        """
        Run numerical integration for specified time period

        Arguments:
            run_time = Scalar double, model run time, [a]
        """
        pass
