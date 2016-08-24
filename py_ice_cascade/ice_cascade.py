"""
Python ICE-CASCADE combined glacial-fluvial-hillslope landscape evolution model
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import py_ice_cascade
import netCDF4
import sys

# NOTE: in __init__(), create the output_file and populate
# the 0th time step. This can serve as an input file too.

class model():
    """
    Composite landscape evolution model. Integrates glacial, fluvial, and
    hillslope model components and handles input-output.
    """

    def __init__(self):
        """Define internal attributes to be populated by set_param*()"""
        # user-defined parameters
        self._x = None
        self._y = None
        self._zrx = None
        self._time_start = None
        self._time_step = None
        self._num_steps = None
        self._out_steps = None
        self._hill_on = None
        self._hill_kappa = None
        self._hill_bc = None
        # automatic parameters
        self._delta = None
        self._time = None
        self._step = None
        self._model_hill = None

    def _create_netcdf(self, file_name, as_input=False, clobber=False, verbose=False):
        """
        Create new (empty) netCDF for model state and parameters
        
        Includes an option to save as an input file (*as_input*), which sets
        the length of the time dimension to 1.

        Arguments:
            file_name = String, path to which file should be saved 
            as_input = Bool, set True save as a model input file
            clobber = Bool, set True to enable overwriting file_name
            verbose = Bool, set True to enable verbose messages
        """

        if verbose:
            print("ice_cascade.model._create_netcdf : creating input file "+file_name)
        
        # compression/chunking parameters for time-dependant grid vars
        zlib = False
        complevel = 1 # 1->fastest, 9->best
        shuffle = True 
        chunksizes = (1, self._y.size, self._x.size)
        
        # create file
        nc = netCDF4.Dataset(file_name, "w", format="NETCDF4", clobber=clobber)
       
        # global attributes: on/off switches for model components
        nc.hillslope_on = int(self._hill_on)

        # create dimensions
        nc.createDimension('x', size=self._x.size)
        nc.createDimension('y', size=self._y.size)
        nc.createDimension('time', size=(1 if as_input else self._out_steps.size))
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
        
        nc.createVariable('hill_kappa', np.double, dimensions=('time', 'y', 'x'), 
            zlib=zlib, complevel=complevel, shuffle=shuffle, chunksizes=chunksizes)
        nc['hill_kappa'].long_name = 'hillslope diffusivity'
        nc['hill_kappa'].units = 'm^2 / a'
        
        nc.createVariable('hill_bc', str, dimensions=('bc'))
        for ii in range(4):
            nc['hill_bc'][ii] = self._hill_bc[ii]

        # finalize
        nc.close()

    def _to_netcdf(self, file_name, verbose=False):
        """
        Append model state and parameters to netCDF file
        
        Arguments:
            file_name = String, path to which file should be saved 
            verbose = Bool, set True to enable verbose messages
        """

        if verbose:
            print("ice_cascade.model._to_netcdf: write time = {:.2f}, step = {}".format(
                self._time, self._step))

        ii = list(self._out_steps).index(self._step) 
        nc = netCDF4.Dataset(file_name, "a")
        nc['time'][ii] = self._time
        nc['step'][ii] = self._step
        nc['zrx'][ii,:,:] = self._zrx
        nc['hill_kappa'][ii,:,:] = self._hill_kappa
        nc.close()

    def _display(self):
        """Display model state, for debugging / demonstration"""
        plt.clf()
        plt.ion()
        plt.imshow(self._zrx, extent=[np.min(self._x), np.max(self._x), 
            np.min(self._y), np.max(self._y)], interpolation='nearest', 
            aspect='equal')
        plt.colorbar()
        plt.title('Bedrock elev, step = {}, time = {:.3f}'.format(
            self._step, self._time))
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show(block=False)
        plt.waitforbuttonpress()

    def _initialize_components(self):
        """Initialize component model objects"""

        self._delta = np.abs(self._x[1]-self._x[0])
        # TODO: test for a regular grid here

        if self._hill_on:
            self._model_hill = py_ice_cascade.hillslope.ftcs(self._zrx, 
                self._delta, self._hill_kappa, self._hill_bc)
        else: 
            self._model_hill = py_ice_cascade.hillslope.null()


    def set_param_from_var(self, x=None, y=None, zrx=None, time_start=None,
        time_step=None, num_steps=None, out_steps=None, hill_on=None,
        hill_kappa=None, hill_bc=None, verbose=False):
        """
        Initialize model state and parameters from argument variables

        All input arguments are optional, and only the supplied model
        state/parameter attributes will be set.

        Arguments:
            x = 1 x nx numpy vector, x-coordinate, [m]
            y = ny x 1 numpy vector, y-coordinate, [m]
            zrx = grid, initial bedrock elevation, [m]
            time_start = scalar, starting time, [a]
            time_step = scalar, topographic model time step, [a]
            num_steps = scalar, total steps in simulation, i.e. duration, [1]
            out_steps = list, step numbers to write output, 0 is initial state, [1]
            hill_on = scalar, boolean flag, True to enable hillslope model
            hill_kappa = grid, hillslope diffusivity, [m^2 / a]
            hill_bc = list, hillslope model boundary conditions at [y[0],
                y[end], x[0], x[end]. See hilllslope.py for details.
            verbose = Boolean, set True to show verbose messages
        """

        if verbose:
            print("set_param_from_var: setting model parameters")

        if x is not None: 
            self._x = np.copy(x)
        if y is not None: 
            self._y = np.copy(y) 
        if zrx is not None: 
            self._zrx = np.copy(zrx) 
        if time_start is not None: 
            self._time_start = time_start 
        if time_step is not None: 
            self._time_step = time_step 
        if num_steps is not None: 
            self._num_steps = num_steps 
        if out_steps is not None: 
            self._out_steps = np.copy(out_steps) 
        if hill_on is not None: 
            self._hill_on = bool(hill_on)
        if hill_kappa is not None: 
            self._hill_kappa = np.copy(hill_kappa)
        if hill_bc is not None: 
            self._hill_bc = list(hill_bc)

    def set_param_from_file(self, file_name, step=0, verbose=False):
        """
        Initialize model state and parameters from input file

        The input file is in netCDF4 format, and must have all model variables
        defined (i.e. as generated by a previous model run() or by the
        to_input_file() method). All model state/parameter attributes will be
        set.

        Arguments:
            file_name = String, name of input file
            step = Int, index of time step to read for time-dependent vars
            verbose = Bool, set to True for verbose output
        """

        if verbose:
            print("ice_cascade.model.set_param_from_file: read from "+file_name)

        nc = netCDF4.Dataset(file_name, "r")
        self._x = nc['x'][:]
        self._y = nc['y'][:]
        self._zrx = nc['zrx'][step,:,:]
        self._time_start = np.asscalar(nc['time'].start)
        self._time_step = np.asscalar(nc['time'].step)
        self._num_steps = np.asscalar(nc['step'].num_steps)
        self._out_steps = nc['step'].out_steps
        self._hill_on = np.asscalar(nc.hillslope_on)
        self._hill_kappa = nc['hill_kappa'][step,:,:]
        self._hill_bc = nc['hill_bc'][:]
        nc.close()

    def to_input_file(self, file_name, clobber=False, verbose=False):
        """
        Write current model state and parameters as a netCDF input file

        Provides the ability to save a model definition to file. Initializing
        and running a new model from the generated file would be equivalent to
        running the current model. It only makes sense to call this method
        *before* calling run().

        Arguments:
            file_name = String, path to which file should be saved 
            clobber = Bool, set True to enable overwriting file_name
            verbose = Boolean, set True to show verbose messages
        """

        if verbose:
            print("ice_cascade.model.to_input_file: dumping model to "+file_name)

        self._time = self._time_start
        self._step = 0
        self._create_netcdf(file_name, as_input=True, clobber=clobber, verbose=verbose)
        self._to_netcdf(file_name, verbose=verbose)

    def run(self, file_name, clobber=False, verbose=False, display=False):
        """
        Run initialized model simulation, save results to file

        Arguments:
            file_name = String, path to which results should be saved 
            clobber = Bool, set True to enable overwriting file_name
            verbose = Bool, set True to show verbose messages
            display = Bool, set True to show debugging plots
        """

        if verbose:
            print("ice_cascade.model.run: initializing simulation")

        self._initialize_components()
        self._time = self._time_start
        self._step = 0
        self._create_netcdf(file_name, clobber=clobber)
        self._to_netcdf(file_name)
        
        while self._step < self._num_steps:

            # synchronize model components
            self._model_hill.set_height(self._zrx)

            # run climate component simulations

            # gather climate component results

            # run erosion-deposition component simulations
            self._model_hill.run(self._time_step)

            # gather erosion-deposition-uplift component results
            delta_zrx = self._model_hill.get_height() - self._zrx

            # run isostasy component simulations

            # gather isostasy results

            # advance time step 
            self._zrx += delta_zrx
            self._time += self._time_step
            self._step += 1

            # write output and/or display model state
            if self._step in self._out_steps:
                self._to_netcdf(file_name)
                if display:
                    self._display()

        if verbose:
            print("ice_cascade.model.run: simulation complete")

def cli():
    """
    Command-line front-end for Python ICE-CASCADE glacial-fluvial-hillslope
    landscape evolution model. 
    
    Parses command-line arguments and runs the ICE-CASCADE model. Installed as
    a console-script called *ice-cascade*.  Additional help can be accessed with
    the command `ice-cascade -h`.
    """

    # get command line arguments
    parser = argparse.ArgumentParser(description='Command-line front-end for '
        'Python ICE-CASCADE glacial-fluvial-hillslope landscape evolution model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input_file', type=str, default='in.nc',
        help='input netCDF file name')
    parser.add_argument('-o', '--output_file', type=str, default='out.nc',
        help='output netCDF file name')
    parser.add_argument('-v', '--verbose', action='store_true',
        help='show verbose progress messages')
    parser.add_argument('-d', '--display', action='store_true',
        help='plot model state at output time-steps, for debugging or demonstration')
    args = parser.parse_args()

    # init and run model
    mod = model()
    mod.set_param_from_file(args.input_file, verbose=args.verbose) 
    mod.run(args.output_file, verbose=args.verbose, display=args.display)
