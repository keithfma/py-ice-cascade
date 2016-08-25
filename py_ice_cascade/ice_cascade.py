"""
Python ICE-CASCADE combined glacial-fluvial-hillslope landscape evolution model
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import py_ice_cascade
import netCDF4
import sys

class model():
    """
    Composite landscape evolution model. Integrates glacial, fluvial, and
    hillslope model components and handles input-output.

    Arguments:
        x: numpy vector, x-coordinate, [m]
        y: numpy vector, y-coordinate, [m]
        zrx: grid, initial bedrock elevation, [m]
        time_start: scalar, starting time, [a]
        time_step: scalar, topographic model time step, [a]
        num_steps: scalar, total steps in simulation, i.e. duration, [1]
        out_steps: list, step numbers to write output, 0 is initial state, [1]
        hill_on: scalar bool, set True to enable hillslope model
        hill_kappa_active: scalar, hillslope diffusivity where active, [m^2 / a]
        hill_kappa_inactive: scalar, hillslope diffusivity where inactive, [m^2 / a]
        hill_bc: list, hillslope model boundary conditions at [y[0],
            y[end], x[0], x[end]. See hilllslope.py for details.
        uplift_on: scalar bool, set True to enable uplift component model
        uplift_start: grid, uplift rate at time_start [m / a]
        uplift_end: grid uplift rate at time_end [m / a]
        verbose: Boolean, set True to show verbose messages
    """

    def __init__(self, x=None, y=None, zrx=None, time_start=None,
        time_step=None, num_steps=None, out_steps=None, hill_on=None,
        hill_kappa_active=None, hill_kappa_inactive=None, hill_bc=None,
        uplift_on=None, uplift_start=None, uplift_end=None,
        verbose=False):

        if verbose:
            print("ice_cascade.model: setting model parameters")

        # user-defined parameters
        self._x = np.copy(x)
        self._y = np.copy(y) 
        self._zrx = np.copy(zrx) 
        self._time_start = time_start 
        self._time_step = time_step 
        self._num_steps = num_steps 
        self._out_steps = np.copy(out_steps) 
        self._hill_on = bool(hill_on)
        if self._hill_on:
            self._hill_kappa_active = np.copy(hill_kappa_active)
            self._hill_kappa_inactive = np.copy(hill_kappa_inactive)
            self._hill_bc = list(hill_bc)
        self._uplift_on = bool(uplift_on)
        if self._uplift_on:
            self._uplift_start = uplift_start
            self._uplift_end = uplift_end
        # automatic parameters
        self._delta = None
        self._time = None
        self._step = None
        self._model_hill = None

    def _create_netcdf(self, file_name, verbose=False):
        """
        Create new (empty) netCDF for model state and parameters
        
        Includes an option to save as an input file (*as_input*), which sets
        the length of the time dimension to 1.

        Arguments:
            file_name: String, path to which file should be saved 
            verbose: Bool, set True to enable verbose messages
        """

        if verbose:
            print("ice_cascade.model._create_netcdf : creating input file "+file_name)
        
        # compression/chunking parameters for time-dependant grid vars
        zlib = False
        complevel = 1 # 1->fastest, 9->best
        shuffle = True 
        chunksizes = (1, self._y.size, self._x.size)
        
        # create file
        nc = netCDF4.Dataset(file_name, "w", format="NETCDF4", clobber=False)
       
        # global attributes: on/off switches for model components
        nc.hillslope_on = int(self._hill_on)
        nc.uplift_on = int(self._uplift_on)

        # create dimensions
        nc.createDimension('x', size=self._x.size)
        nc.createDimension('y', size=self._y.size)
        nc.createDimension('time', size=self._out_steps.size)
        nc.createDimension('bc', size=4)

        # create variables, populate constants
        nc.createVariable('x', np.double, dimensions=('x'))
        nc['x'].long_name = 'x coordinate'
        nc['x'].units = 'm'
        nc['x'][:] = self._x

        nc.createVariable('y', np.double, dimensions=('y'))
        nc['y'].long_name = 'y coordinate'
        nc['y'].units = 'm'
        nc['y'][:] = self._y
        
        nc.createVariable('time', np.double, dimensions=('time'))
        nc['time'].long_name = 'time coordinate'
        nc['time'].units = 'a'
        nc['time'].start = self._time_start
        nc['time'].step = self._time_step

        nc.createVariable('step', np.int64, dimensions=('time'))
        nc['step'].long_name = 'model time step'
        nc['step'].units = '1'
        nc['step'].num_steps = self._num_steps
        nc['step'].out_steps = self._out_steps
        
        nc.createVariable('zrx', np.double, dimensions=('time', 'y', 'x'),
            zlib=zlib, complevel=complevel, shuffle=shuffle, chunksizes=chunksizes)
        nc['zrx'].long_name = 'bedrock surface elevation' 
        nc['zrx'].units = 'm' 

        if self._hill_on:        
            nc.createVariable('hill_kappa', np.double, dimensions=('time', 'y', 'x'), 
                zlib=zlib, complevel=complevel, shuffle=shuffle, chunksizes=chunksizes)
            nc['hill_kappa'].long_name = 'hillslope diffusivity'
            nc['hill_kappa'].units = 'm^2 / a'
            nc['hill_kappa'].active = self._hill_kappa_active
            nc['hill_kappa'].inactive = self._hill_kappa_inactive
            
            nc.createVariable('hill_bc', str, dimensions=('bc'))
            for ii in range(4):
                nc['hill_bc'][ii] = self._hill_bc[ii]

        if self._uplift_on:
            nc.createVariable('uplift_rate', np.double, dimensions=('time', 'y', 'x'), 
                zlib=zlib, complevel=complevel, shuffle=shuffle, chunksizes=chunksizes)
            nc['uplift_rate'].long_name = 'tectonic rock uplift rate'
            nc['uplift_rate'].units = 'm / a'

        # finalize
        nc.close()

    def _to_netcdf(self, file_name, verbose=False):
        """
        Append model state and parameters to netCDF file
        
        Arguments:
            file_name: String, path to which file should be saved 
            verbose: Bool, set True to enable verbose messages
        """

        if verbose:
            print("ice_cascade.model._to_netcdf: write time = {:.2f}, step = {}".format(
                self._time, self._step))

        ii = list(self._out_steps).index(self._step) 
        nc = netCDF4.Dataset(file_name, "a")
        nc['time'][ii] = self._time
        nc['step'][ii] = self._step
        nc['zrx'][ii,:,:] = self._zrx
        if self._hill_on:
            nc['hill_kappa'][ii,:,:] = self._hill_kappa
        if self._uplift_on:
            nc['uplift_rate'][ii,:,:] = self._model_uplift.get_uplift_rate(self._time)
        nc.close()

    def _initialize_component_models(self):
        """Initialize component model objects"""

        # hillslope component
        if self._hill_on:
            self._hill_kappa = np.ones(self._zrx.shape)*self._hill_kappa_active
            self._model_hill = py_ice_cascade.hillslope.ftcs(self._zrx, 
                self._delta, self._hill_kappa, self._hill_bc)
        else: 
            self._model_hill = py_ice_cascade.hillslope.null()

        # uplift component
        if self._uplift_on:
            self._model_uplift = py_ice_cascade.uplift.linear(self._uplift_start, 
                self._uplift_end, self._time_start, self._time_end)
        else:
            self._model_uplift = py_ice_cascade.uplift.null(self._zrx.shape)

    def run(self, file_name, verbose=False):
        """
        Run model simulation, save results to file

        Arguments:
            file_name: String, path to which results should be saved 
            verbose: Bool, set True to show verbose messages
        """

        if verbose:
            print("ice_cascade.model.run: initializing simulation")

        # init automatic parameters
        self._delta = np.abs(self._x[1]-self._x[0])
        self._time_end = self._time_start+self._time_step*(self._num_steps-1)
        self._initialize_component_models()
        
        # TODO: test for a regular grid here

        # init model integration loop
        self._time = self._time_start
        self._step = 0
        self._create_netcdf(file_name)
        self._to_netcdf(file_name)
        
        while self._step < self._num_steps:

            # synchronize model components
            self._model_hill.set_height(self._zrx)

            # run climate component simulations

            # gather climate component results

            # run erosion-deposition component simulations
            self._model_hill.run(self._time_step)

            # gather erosion-deposition-uplift component results
            d_zrx_hill = self._model_hill.get_height() - self._zrx
            d_zrx_uplift = self._model_uplift.get_uplift(self._time, self._time+self._time_step)

            # run isostasy component simulations

            # gather isostasy results

            # advance time step 
            self._zrx += d_zrx_hill+d_zrx_uplift
            self._time += self._time_step
            self._step += 1

            # write output and/or display model state
            if self._step in self._out_steps:
                self._to_netcdf(file_name)

        if verbose:
            print("ice_cascade.model.run: simulation complete")

if __name__ == '__main__':
    pass
