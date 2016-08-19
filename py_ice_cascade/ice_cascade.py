"""
Python ICE-CASCADE combined glacial-fluvial-hillslope landscape evolution model
"""

import argparse
import numpy as np
import py_ice_cascade

class model():
    """
    Composite landscape evolution model. Integrates glacial, fluvial, and
    hillslope model components and handles input-output.
    """

    # TODO: Document details of input file and output file formatting

    def __init__(self, input_file, output_file, verbose=False):
        """
        Initialize model from netCDF file containing initial data and model
        parameters. See class docstring for input / output file format
        details
        
        Arguments:
            input_file = String, filename of input netCDF file
            output_file = String, filename of output netCDF file
        """
        
        self._input_file = input_file
        self._output_file = output_file
        self._verbose = verbose
        self._read_input()
        self._time = self._time_start

        self._model_hill = py_ice_cascade.hillslope.ftcs(self._zrx_init, 
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
        self._delta = 1.0/(self._nx-1) 
        self._zrx_init = np.pad(np.random.rand(self._ny-2, self._nx-2), 1, 'constant', constant_values=0)
        self._time_start = 0.0
        self._time_end = 1.0
        self._time_step = 0.1
        # # hillslope
        self._hill_kappa = np.ones((self._ny, self._nx))
        self._hill_bc = ['constant']*4

    def _create_output(self):
        """Create output netCDF file for model results"""

        if self._verbose:
            print("ice-cascade: create output file")
        
        # NOTE: is there an easy way to print an overview of the output file,
        # like ncdump -h? This would be a useful addition to verbose output

    def _write_output(self):
        """Write output step to file"""

        if self._verbose:
            print("ice-cascade: write output for time = {:.2f}".format(self._time))

    def run(self):
        """Run model simulation and write results to file"""

        if self._verbose:
            print("ice-cascade: running simulation")
        
        while self._time <= self._time_end:

            # synchronize model components

            # run climate component simulations

            # gather climate component results

            # run erosion-deposition component simulations
            self._model_hill.run(self._time_step)

            # gather erosion-deposition component results

            # run uplift-subsidence component simulations

            # gather uplift-subsidence results

            # advance time step 
            self._time += self._time_step

            # write output if requested
            self._write_output()

        if self._verbose:
            print("ice-cascade: simulation complete")

def cli():
    """
    Command-line front-end for Python ICE-CASCADE glacial-fluvial-hillslope
    landscape evolution model. 
    
    Parses command-line arguments and runs the
    ICE-CASCADE model. Installed as a console-script called "ice-cascade".
    Additional help can be accessed with the command `ice-cascade -h`.
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
        help='verbose progress messages')
    args = parser.parse_args()

    # init and run model
    mod = model(args.input_file, args.output_file, verbose=args.verbose)
    mod.run()
