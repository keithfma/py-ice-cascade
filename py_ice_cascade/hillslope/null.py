from .base import base_model
import numpy as np

class null_model(base_model):
    """
    Do-nothing class to be used for disabled hillslope component

    Internal height grid is set and returned unchanged
    """
    
    def __init__(self):
        """Not used"""
        pass

    def set_height(self, new):
        """Store height grid as numpy array"""
        self._height = np.double(new) 
    
    def get_height(self):
        """Return height grid as numpy array"""
        return self._height
    
    def set_mask(*args):
        """Not used"""
        pass
    
    def init_netcdf(self, nc, *args):
        """
        Set name, status of hillslope model component to False, Null

        Arguments:
            nc: netCDF Dataset object, open for writing
            *args: other ignored arguments
        """
        nc.createVariable('hill_model', np.dtype('i1')) # scalar
        nc['hill_model'][...] = False 
        nc['hill_model'].type = self.__class__.__name__ 
    
    def to_netcdf(*args):
        """Not used"""
        pass
    
    def run(*args):
        """Not used"""
        pass
