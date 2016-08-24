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

    def __init__(self, input_file=None, output_file=None, x=None, y=None, 
        zrx=None, time_start=None, time_step=None, num_steps=None, 
        out_steps=None, hill_on=None, hill_kappa=None, hill_bc=None, 
        verbose=False, display=False):
        """
        Initialize model state and parameters from file and variables. The
        input file is parsed first, then overwritten by any supplied variables.
         
        Arguments:
            input_file = String, filename of netCDF file containing initial
                model state and parameters.
            output_file = String, filename of netCDF file containing model
                state and parameters
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
            verbose = Boolean, show verbose progress messages
            display = Boolean, plot model state at output time-steps, for
                debugging or demonstration
        """
        
        # store select arguments
        self._input_file = input_file
        self._output_file = output_file
        self._verbose = verbose
        self._display = display

        # init from file
        if input_file is not None:
            self._from_netcdf()
        
        # init from optional vars, overwriting
        if x is not None: self._x = np.copy(x)
        if y is not None: self._y = np.copy(y) 
        if zrx is not None: self._zrx = np.copy(zrx) 
        if time_start is not None: self._time_start = time_start 
        if time_step is not None: self._time_step = time_step 
        if num_steps is not None: self._num_steps = num_steps 
        if out_steps is not None: self._out_steps = np.copy(out_steps) 
        if hill_on is not None: self._hill_on = bool(hill_on)
        if hill_kappa is not None: self._hill_kappa = np.copy(hill_kappa)
        if hill_bc is not None: self._hill_bc = list(hill_bc)
        
        # initialize component models (which include additional checks)
        if self._hill_on:
            self._model_hill = py_ice_cascade.hillslope.ftcs(self._zrx, 
                self._delta, self._hill_kappa, self._hill_bc)
        else: 
            self._model_hill = py_ice_cascade.hillslope.null()

        # init model time
        self._time = self._time_start
        self._step = 0

        # create ouput file and write step 0
        self._create_netcdf()
        self._to_netcdf()

    def _create_netcdf(self):
        """Create new (empty) netCDF for model state and parameters"""

        # create file
        nc = netCDF4.Dataset(self._output_file, "w", format="NETCDF4", clobber=False)
        
        # create dimensions
        nc.createDimension('x', size = self._nx)
        nc.createDimension('y', size = self._ny)
        nc.createDimension('time', size = len(list(self._out_steps)))

        # create variables
        nc.createVariable('x', np.double, dimensions=('x'))
        nc['x'].long_name = 'x coordinate'
        nc['x'].units = 'm'
        nc.createVariable('y', np.double, dimensions=('y'))
        nc['y'].long_name = 'y coordinate'
        nc['y'].units = 'm'
        nc.createVariable('time', np.double, dimensions=('time'))
        nc['time'].long_name = 'time coordinate'
        nc['time'].units = 'a'
        nc.createVariable('time_step', np.double, dimensions=())
        nc['time_step'].long_name = 'time step'
        nc['time_step'].units = 'a'
        nc.createVariable('num_steps', np.int64, dimensions=())
        nc['num_steps'].long_name = 'number of time steps'
        nc['num_steps'].units = '1'
        nc.createVariable('out_steps', np.int64, dimensions=('time'))
        nc['out_steps'].long_name = 'model ouput step indices'
        nc['out_steps'].units = '1'
        nc.createVariable('zrx', np.double, dimensions=('time', 'y', 'x'))
        nc['zrx'].long_name = 'bedrock surface elevation' 
        nc['zrx'].units = 'm' 
        nc.createVariable('hill_on', np.int, dimensions=())
        nc['hill_on'].long_name = 'hillslope model on/off flag'
        nc.createVariable('hill_kappa', np.double, dimensions=('time', 'y', 'x')) # scalar
        nc['hill_kappa'].long_name = 'hillslope diffusivity'
        nc['hill_kappa'].units = 'm^2 / a'
        nc.createVariable('hill_bc_y0', str, dimensions=())
        nc['hill_bc_y0'].long_name = 'hillslope boundary condition at y[0]' 
        nc.createVariable('hill_bc_y1', str, dimensions=())
        nc['hill_bc_y1'].long_name = 'hillslope boundary condition at y[end]' 
        nc.createVariable('hill_bc_x0', str, dimensions=())
        nc['hill_bc_x0'].long_name = 'hillslope boundary condition at x[0]' 
        nc.createVariable('hill_bc_x1', str, dimensions=())
        nc['hill_bc_x1'].long_name = 'hillslope boundary condition at x[end]' 
        
        # populate constant variables   
        nc['x'][:] = self._x
        nc['y'][:] = self._y
        nc['time_step'][...] = self._time_step
        nc['num_steps'][...] = self._num_steps
        nc['out_steps'][:] = self._out_steps
        nc['hill_on'][...] = self._hill_on
        nc['hill_bc_y0'][...] = self._hill_bc[0]
        nc['hill_bc_y1'][...] = self._hill_bc[1]
        nc['hill_bc_x0'][...] = self._hill_bc[2]
        nc['hill_bc_x1'][...] = self._hill_bc[3]

        # finalize
        nc.close()

    def _to_netcdf(self):
        """Append model state and parameters to netCDF file"""

        if self._verbose:
            print("ice-cascade: write model state at time = {:.2f}, step = {}".format(
                self._time, self._step))

        ii = list(self._out_steps).index(self._step) 
        nc = netCDF4.Dataset(self._output_file, "a")
        nc['time'][ii] = self._time
        nc['zrx'][ii,:,:] = self._zrx
        nc['hill_kappa'][ii,:,:] = self._hill_kappa
        nc.close()

        # plot model state for debugging
        if self._display:
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

    def _from_netcdf(self):
        """Read model state and parameters from first time step in netCDF file"""

        if self._verbose:
            print("ice-cascade: read model state and parameters from netCDF file")

        nc = netCDF4.Dataset(self._input_file, "r")
        self._x = nc['x'][:]
        self._nx = self._x.size
        self._x = self._x.reshape(1, self._nx)
        self._y = nc['y'][:]
        self._ny = self._y.size
        self._y = self._y.reshape(self._ny, 1)
        self._time_start = nc['time'][0]
        self._time_step = np.asscalar(nc['time_step'][...])
        self._num_steps = np.asscalar(nc['num_steps'][...])
        self._out_steps = nc['out_steps'][:] 
        self._zrx = nc['zrx'][0,:,:]
        self._hill_on = np.asscalar(nc['hill_on'][...])
        self._hill_kappa = nc['hill_kappa'][0,:,:]
        self._hill_bc = [nc['hill_bc_y0'][...], nc['hill_bc_y1'][...], 
            nc['hill_bc_x0'][...], nc['hill_bc_x1'][...]]
        nc.close()

    def run(self):
        """Run model simulation and optionally write results to file"""

        if self._verbose:
            print("ice-cascade: running simulation")
        
        while self._step < self._num_steps:

            # synchronize model components
            self._model_hill.set_height(self._zrx)

            # run climate component simulations

            # gather climate component results

            # run erosion-deposition component simulations
            self._model_hill.run(self._time_step)

            # gather erosion-deposition component results
            delta_zrx = self._model_hill.get_height() - self._zrx

            # run uplift-subsidence component simulations

            # gather uplift-subsidence results

            # advance time step 
            self._zrx += delta_zrx
            self._time += self._time_step
            self._step += 1

            # write output and/or display model state
            if self._step in self._out_steps:
                self._to_netcdf()

        if self._verbose:
            print("ice-cascade: simulation complete")

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
    mod = model(input_file=args.input_file, output_file=args.output_file, 
        verbose=args.verbose, display=args.display)
    mod.run()
