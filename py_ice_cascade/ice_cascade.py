"""
Python ICE-CASCADE combined glacial-fluvial-hillslope landscape evolution model
"""

import argparse

class model():
    """
    Composite landscape evolution model. Integrates glacial, fluvial, and
    hillslope model components and handles input-output.
    """

    # TODO: Document details of input file and output file formatting

    def __init__(self, input_file, output_file):
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
        self._read_input()
        self._create_output()

    def _read_input(self):
        """Read initial data and model parameters from netCDF file"""
        print("READ INPUT: "+self._input_file)

    def _create_output(self):
        """Create output netCDF file for model results"""
        print("CREATE OUTPUT: "+self._output_file)

    def _write_output(self):
        """Write output step to file"""
        print("WRITE OUTPUT STEP: "+self._output_file)

    def run(self):
        """Run model simulation and write results to file"""
        print("RUNNING SIMULATION")
        self._write_output()

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
        help='input netCDF file name', required=True)
    parser.add_argument('-o', '--output_file', type=str, default='out.nc',
        help='output netCDF file name', required=True)
    args = parser.parse_args()

    # init and run model
    mod = model(args.input_file, args.output_file)
    mod.run()
