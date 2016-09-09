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
        hillslope: initialized hillslope model component, expect child of
            py_ice_cascade.hillslope.model class 
        uplift: initialized uplift model component, expect child of
            py_ice_cascade.uplift.model class
        x: numpy vector, x-coordinate, [m]
        y: numpy vector, y-coordinate, [m]
        z_rx: grid, initial bedrock elevation, [m]
        time_start: scalar, starting time, [a]
        time_step: scalar, topographic model time step, [a]
        num_steps: scalar, total steps in simulation, i.e. duration, [1]
        out_steps: list, step numbers to write output, 0 is initial state, [1]
        verbose: Boolean, set True to show verbose messages
    """

    def __init__(self, hillslope, uplift,
        x, y, z_rx, time_start, time_step, num_steps, out_steps, verbose=False):

        if verbose:
            print("ice_cascade.model.__init__: setting model parameters")

        # user-defined parameters
        self._model_hill = hillslope 
        self._model_uplift = uplift
        self._x = np.copy(x)
        self._y = np.copy(y) 
        self._z_rx = np.copy(z_rx) 
        self._time_start = time_start 
        self._time_step = time_step 
        self._num_steps = num_steps 
        self._out_steps = np.copy(out_steps) 
        self._verbose = verbose
        # automatic parameters
        self._delta = None
        self._time = None
        self._step = None

        # TODO: test for a regular grid here

    def _create_netcdf(self, file_name):
        """
        Create new (empty) netCDF for model state and parameters
        
        Arguments:
            file_name: String, path to which file should be saved 
        
        Model components are responsible for initializing thier own output
        variables, using the expected .init_netcdf method.
        """

        if self._verbose:
            print("ice_cascade.model._create_netcdf : creating input file "+file_name)
        
        # compression/chunking parameters for time-dependant grid vars
        zlib = False
        complevel = 1 # 1->fastest, 9->best
        shuffle = True 
        chunksizes = (1, self._y.size, self._x.size)
        
        # create file
        nc = netCDF4.Dataset(file_name, "w", format="NETCDF4", clobber=False)
       
        # global attributes
        nc.version = py_ice_cascade.__version__
        nc.time_start = self._time_start
        nc.time_step = self._time_step
        nc.num_steps = self._num_steps
        nc.out_steps = "see step variable"

        # create dimensions
        nc.createDimension('x', size=self._x.size)
        nc.createDimension('y', size=self._y.size)
        nc.createDimension('time', size=self._out_steps.size)

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
        
        nc.createVariable('z_rx', np.double, dimensions=('time', 'y', 'x'),
            zlib=zlib, complevel=complevel, shuffle=shuffle, chunksizes=chunksizes)
        nc['z_rx'].long_name = 'bedrock surface elevation' 
        nc['z_rx'].units = 'm' 

        # initialize output for component models
        self._model_hill.init_netcdf(nc, zlib, complevel, shuffle, chunksizes)
        self._model_uplift.init_netcdf(nc, zlib, complevel, shuffle, chunksizes)

        # finalize
        nc.close()

    def _to_netcdf(self, file_name):
        """
        Append model state and parameters to netCDF file
        
        Arguments:
            file_name: String, path to which file should be saved 
        """

        if self._step in self._out_steps:
            if self._verbose:
                print("ice_cascade.model._to_netcdf: write time = {:.2f}, step = {}".format(
                    self._time, self._step))

            ii = list(self._out_steps).index(self._step) 
            nc = netCDF4.Dataset(file_name, "a")
            nc['time'][ii] = self._time
            nc['step'][ii] = self._step
            nc['z_rx'][ii,:,:] = self._z_rx

            # write data for model components
            self._model_hill.to_netcdf(nc, ii)
            self._model_uplift.to_netcdf(nc, ii)

            # finalize
            nc.close()

    def run(self, file_name):
        """
        Run model simulation, save results to file

        Arguments:
            file_name: String, path to which results should be saved 
        """

        if self._verbose:
            print("ice_cascade.model.run: initializing simulation")

        # init automatic parameters
        self._delta = np.abs(self._x[1]-self._x[0])
        self._time_end = self._time_start+self._time_step*(self._num_steps-1)

        # init model integration loop
        self._time = self._time_start
        self._step = 0
        self._create_netcdf(file_name)
        self._to_netcdf(file_name)

        while self._step < self._num_steps:

            # synchronize model components
            self._model_hill.set_height(self._z_rx)
            self._model_uplift.set_height(self._z_rx)

            # run climate component simulations

            # gather climate component results

            # run erosion-deposition component simulations
            self._model_hill.run(self._time_step)
            self._model_uplift.run(self._time, self._time+self._time_step)

            # gather erosion-deposition-uplift component results
            dzdt = (self._model_hill.get_height()
                + self._model_uplift.get_height()
                - 2*self._z_rx)

            # run isostasy component simulations

            # gather isostasy results

            # advance time step 
            self._z_rx += dzdt
            self._time += self._time_step
            self._step += 1

            # write output and/or display model state
            self._to_netcdf(file_name)

        if self._verbose:
            print("ice_cascade.model.run: simulation complete")
