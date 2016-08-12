"""
Python ICE-CASCADE hillslope erosion-deposition model component
"""

# TODO: How to treat wet or icy points?

import numpy as np

class hillslope_model():
    """
    Hillslope diffusion model state and methods
    """

    def __init__(self, height, delta, kappa):
        """
        Initialize new model object

        Arguments:
            height = 2D Numpy array, surface elevation in model domain, [m]
            delta = Scalar double, grid spacing, assumed square, [m]
            kappa = Scalar double, diffusion coefficient, [m**2/s]
        """

        self.set_param(new_height=height, new_delta=delta, new_kappa=kappa) 
        return
            
    def set_param(self, new_height = None, new_delta = None, new_kappa = None):
        """
        Set model state and parameters as expected types. See __init__() for
        argument definitions.
        """

        if new_height is not None:
            self.height = np.atleast_2d(np.array(new_height, dtype=np.double))
        if new_delta is not None:
            self.delta = np.double(new_delta)
        if new_kappa is not None:
            self.kappa = np.double(new_kappa) 
        return


        
        
        
