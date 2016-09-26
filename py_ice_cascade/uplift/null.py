"""
Python ICE-CASCADE tectonic uplift-subsidence model component

Null model: Defines do-nothing methods for required interface, used to disable
the uplift-subsidence model component
"""

from .base import base_model
import numpy as np

class null_model(base_model):
    """Do-nothing class to be used for disabled uplift component"""

    def __init__(self):
        pass

    def set_height(self, new):
        self._height = np.copy(np.double(new))
    
    def get_height(self):
        return np.copy(self._height)
    
    def init_netcdf(self, nc, *args):
        nc.createVariable('uplift_model', np.dtype('i1')) # scalar
        nc['uplift_model'][...] = False 
        nc['uplift_model'].type = self.__class__.__name__ 
    
    def to_netcdf(*args):
        pass
    
    def run(*args):
        pass
