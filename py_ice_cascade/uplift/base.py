"""
Python ICE-CASCADE tectonic uplift-subsidence model component

Base class: Defines required interface methods such that a NotImplementedError
exception is raised if they are not defined in the child class.
"""

class base_model():
    """Base class for uplift model components"""

    def __init__(self):
        pass
    
    def set_height(self, new):
        raise NotImplementedError
    
    def get_height(self):
        raise NotImplementedError
    
    def init_netcdf(self, nc, zlib, complevel, shuffle, chunksizes):
        raise NotImplementedError
    
    def to_netcdf(self, nc, time_idx):
        raise NotImplementedError
    
    def run(self, t_start, t_end):
        raise NotImplementedError
