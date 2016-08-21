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

    def __init__(self, input_file=None, output_file=None, x=None, y=None, zrx=None,
        time_start=None, time_step=None, num_steps=None, out_steps=None,
        hill_on=None, hill_kappa=None, hill_bc=None, verbose=False, display=False):
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
        self._output_file = output_file
        self._verbose = verbose
        self._display = display

        # init from file
        if input_file is not None:
            self.from_netcdf(input_file)
        
        # init from optional vars, overwriting
        if x is not None: self._x = x
        if y is not None: self._y = y 
        if zrx is not None: self._zrx = zrx 
        if time_start is not None: self._time_start = time_start 
        if time_step is not None: self._time_step = time_step 
        if num_steps is not None: self._num_steps = num_steps 
        if out_steps is not None: self._out_steps = out_steps 
        if hill_on is not None: self._hill_on = hill_on 
        if hill_kappa is not None: self._hill_kappa = hill_kappa 
        if hill_bc is not None: self._hill_bc = hill_bc 

        # confirm valid state and parameters
        if self._output_file is None:
            self._abort("missing output file name") 
        if type(self._x) is not np.ndarray:
            self._abort("expect x to be a numpy array")
        self._nx = self._x.size
        if self._x.shape != (1, self._nx):
            self._abort("expect x to be a row vector")
        self._delta = np.abs(self._x[0,1]-self._x[0,0])
        if type(self._y) is not np.ndarray:
            self._abort("expect y to be a numpy array")
        self._ny = self._y.size
        if self._nx<3 or self._ny<3:
            self._abort("minimum grid dimension is 3 x 3")
        if not np.allclose(np.abs(self._x[0,1:]-self._x[0,:-1]), self._delta):
            self._abort("invalid spacing in x-coordinate")
        if not np.allclose(np.abs(self._y[1:,0]-self._y[:-1,0]), self._delta):
            self._abort("invalid spacing in y-coordinate")
        if type(self._zrx) is not np.ndarray:
            self._abort("expect zrx to be a numpy array")
        if self._zrx.shape != (self._ny, self._nx): 
            self._abort("zrx shape does not match coordinate dimensions")
        if self._time_start is None:
            self._abort("missing required parameter 'time_start'")
        if hasattr(self._time_start, '__len__'):
            self._abort("expected scalar for time_start")
        if self._time_step is None:
            self._abort("missing required parameter 'time_step'")
        if hasattr(self._time_step, '__len__'):
            self._abort("expected scalar for time_step")
        if self._num_steps is None:
            self._abort("missing required parameter 'num_steps'")
        if hasattr(self._num_steps, '__len__'):
            self._abort("expected scalar for num_steps")
        if self._num_steps < 2:
            self._abort("must have at least two steps")
        if self._out_steps.size < 2:
            self._abort("expected numpy array with at least 2 elements for out_steps")
        if self._out_steps[0] != 0:
            self._abort("output steps must begin with 0th")
        if self._out_steps[-1] != self._num_steps-1:
            self._abort("output steps must end with last")
        if self._hill_on not in [True, False]:
            self._abort("hill_on parameter must be boolean")
        if type(self._hill_kappa) is not np.ndarray:
            self._abort("expected hill_kappa to be a numpy array")
        if self._hill_kappa.shape != (self._ny, self._nx):
            self._abort("invalid shape for hill_kappa grid")
        if len(self._hill_bc) != 4:
            self._abort("expect list of 4 boundary conditions in hill_bc")
        
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

    def _abort(self, msg):
        """Print message and abort"""
        print("ice-cascade: " + msg)
        sys.exit()

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

    def _from_netcdf(self):
        """Read model state and parameters from first time step in netCDF file"""

        if self._verbose:
            print("ice-cascade: read model state and parameters from netCDF file")

        nc = netCDF4.Dataset(self._input_file, "r")
        self._x = nc['x'][:]
        self._y = nc['y'][:]
        self._time_step = nc['time_step']
        self._num_steps = nc['num_steps']
        self._out_steps = nc['out_steps'][:] 
        self._zrx = nc['zrx'][0,:,:]
        self._hill_on = nc['hill_on'][...]
        self._hill_kappa = nc['hill_kappa'][0,:,:]
        self._hill_bc = [nc['hill_bc_y0'][...], nc['hill_bc_y1'][...], 
            nc['hill_bc_x0'][...], nc['hill_bc_x1'][...]]
        nc.close()

    def run(self):
        """Run model simulation and optionally write results to file"""

        if self._verbose:
            print("ice-cascade: running simulation")
        
        for self._step in range(self._num_steps):

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

            # write output and/or display model state
            if self._step in self._out_steps:
                self._to_netcdf()
                if self._display:
                    self._plot()

        if self._verbose:
            print("ice-cascade: simulation complete")
    
    def _plot(self):
        """Plot model state for debugging and demonstration purposes"""
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
    mod = model(args.input_file, args.output_file, verbose=args.verbose, 
        display=args.display)
    mod.run()

# example usage and "smell test"
if __name__ == '__main__':

    # parameters
    ny = 50
    nx = 100
    lx = 1.0
    delta = lx/(nx-1) 
    ly = delta*(ny-1)
    x = np.linspace(0, lx, nx).reshape(1,nx)
    y = np.linspace(0, ly, ny).reshape(ny, 1)
    zrx = np.pad(np.random.rand(ny-2, nx-2), 1, 'constant', constant_values=0)
    time_start = 0.0
    time_step = 0.1
    num_steps = 10
    out_steps = np.array([0,9]) 
    hill_on = True
    hill_kappa = 0.01*np.ones((ny, nx))
    hill_bc = ['constant']*4

    # Example 1: init model directly and run
    mod1 = model(output_file='ex1.out.nc', x=x, y=y, zrx=zrx,
        time_start=time_start, time_step=time_step, num_steps=num_steps,
        out_steps=out_steps, hill_on=hill_on, hill_kappa=hill_kappa,
        hill_bc=hill_bc, verbose=True)
    mod1.run()

    # Example 2: init model, generating input file but not running
    mod2 = model(output_file='ex2.out.nc', x=x, y=y, zrx=zrx,
        time_start=time_start, time_step=time_step, num_steps=num_steps,
        out_steps=out_steps, hill_on=hill_on, hill_kappa=hill_kappa,
        hill_bc=hill_bc, verbose=True)

    # # Example 3: run model from input file
    # mod3 = model(input_file='ex2.out.nc', output_file='ex3.out.nc', verbose=True)

