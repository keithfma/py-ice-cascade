"""
Python ICE-CASCADE combined glacial-fluvial-hillslope landscape evolution model
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import py_ice_cascade
import netCDF4

class model():
    """
    Composite landscape evolution model. Integrates glacial, fluvial, and
    hillslope model components and handles input-output.
    """

    # TODO: Document details of input file and output file formatting

    # TODO: Make it possible to directly initialize the ice-cascade model from
    # variables. In this way, the _create_output function (or some variant) can
    # be used to create an input file by dumping the model state. The big
    # advantage is the the IO formatting will ALL be contained in this model
    # class --- so orothogonal!

    def __init__(self, input_file, output_file, verbose=False, display=False):
        """
        Initialize model from netCDF file containing initial data and model
        parameters. See class docstring for input / output file format
        details
        
        Arguments:
            input_file = String, filename of input netCDF file
            output_file = String, filename of output netCDF file
            verbose = Boolean, show verbose progress messages
            display = Boolean, plot model state at output time-steps, for
                debugging or demonstration
        """
        
        self._input_file = input_file
        self._output_file = output_file
        self._verbose = verbose
        self._display = display
        self._read_input()
        self._time = self._time_start
        self._step = 0

        self._model_hill = py_ice_cascade.hillslope.ftcs(self._zrx, 
            self._delta, self._hill_kappa, self._hill_bc)
        
        # prep output file 
        self._create_output()

    def _read_input(self):
        """Read initial data and model parameters from netCDF file"""

        if self._verbose:
            print("ice-cascade: read input file")

        # DEBUG: test pipeline by setting attributes directly for now
        # # common
        self._ny = 50
        self._nx = 100
        lx = 1.0
        self._delta = lx/(self._nx-1) 
        ly = self._delta*(self._ny-1)
        self._x = np.linspace(0, lx, self._nx).reshape(1,self._nx)
        self._y = np.linspace(0, ly, self._ny).reshape(self._ny, 1)
        self._zrx = np.pad(np.random.rand(self._ny-2, self._nx-2), 1, 'constant', constant_values=0)
        self._time_start = 0.0
        self._time_step = 0.1
        self._num_steps = 10 # must be >= 2
        self._out_steps = [0,9] # must include 0
        # # hillslope
        self._hill_kappa = 0.01*np.ones((self._ny, self._nx))
        self._hill_bc = ['constant']*4

    def _create_output(self):
        """Create output netCDF file for model state/results"""

        if self._verbose:
            print("ice-cascade: create output file")

        # create file
        nc = netCDF4.Dataset(self._output_file, "w", format="NETCDF4", clobber=False)
        
        # create dimensions
        dim_x    = nc.createDimension('x', size = self._nx)
        dim_y    = nc.createDimension('y', size = self._ny)
        dim_time = nc.createDimension('time', size = len(self._out_steps))

        # create and populate variables
        var_x = nc.createVariable('x', np.double, dimensions=('x'))
        var_x.long_name = 'x coordinate'
        var_x.units = 'm'
        var_x[:] = self._x

        var_y = nc.createVariable('y', np.double, dimensions=('y'))
        var_y.long_name = 'y coordinate'
        var_y.units = 'm'
        var_y[:] = self._y

        var_time = nc.createVariable('time', np.double, dimensions=('time'))
        var_time.long_name = 'time coordinate'
        var_time.units = 'a'
        var_time.start = self._time_start
        var_time.step = self._time_step
        var_time.num_steps = self._num_steps
        var_time.out_steps = self._out_steps 
        var_time[0] = self._time_start
    
        var_zrx = nc.createVariable('zrx', np.double, dimensions=('y', 'x', 'time'))
        var_zrx.long_name = 'bedrock surface elevation' 
        var_zrx.units = 'm' 
        var_zrx[:,:,0] = self._zrx 

        var_hill_kappa = nc.createVariable('hill_kappa', np.double, dimensions=('y', 'x', 'time')) # scalar
        var_hill_kappa.long_name = 'hillslope diffusivity'
        var_hill_kappa.units = 'm^2 / a'
        var_hill_kappa[:,:,0] = self._hill_kappa
        
        var_hill_bc_x0 = nc.createVariable('hill_bc_x0', str, dimensions=())
        var_hill_bc_x0.long_name = 'hillslope boundary condition at x[0]' 
        var_hill_bc_x0[...] = self._hill_bc[0]

        var_hill_bc_x1 = nc.createVariable('hill_bc_x1', str, dimensions=())
        var_hill_bc_x1.long_name = 'hillslope boundary condition at x[end]' 
        var_hill_bc_x1[...] = self._hill_bc[1]
        
        var_hill_bc_y0 = nc.createVariable('hill_bc_y0', str, dimensions=())
        var_hill_bc_y0.long_name = 'hillslope boundary condition at y[0]' 
        var_hill_bc_y0[...] = self._hill_bc[2]

        var_hill_bc_y1 = nc.createVariable('hill_bc_y1', str, dimensions=())
        var_hill_bc_y1.long_name = 'hillslope boundary condition at y[end]' 
        var_hill_bc_y1[...] = self._hill_bc[3]

        # finalize
        nc.close()

    def _write_output(self):
        """Write output step to file"""

        if self._verbose:
            print("ice-cascade: write output for time = {:.3f}".format(self._time))

    def run(self):
        """Run model simulation and write results to file"""

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
                self._write_output()
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

    input_file = None
    output_file = "junk.out.nc"
    verbose = True
    display = True

    mod = model(input_file, output_file, verbose, display)
    mod.run()
