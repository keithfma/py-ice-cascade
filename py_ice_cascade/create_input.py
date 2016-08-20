"""
Console script with options to create input files for example cases
"""

import argparse
import os
import sys
import netCDF4

def template(filename, clobber=False):
    """
    Create a template input file for the Python ICE-CASCADE
    landscape evolution model. The template defines the expected dimensions,
    variables, and attributes.  
    
    To set up a model run, create a template file, then open it for editing and
    populate the values. Using the netCDF4 package (http://unidata.github.io/netcdf4-python/):

    .. code-block:: python
       import py_ice_cascade
       import netCDF4
       import shutil
       py_ice_cascade.create_input.template("my_experiment.in.nc", clobber=True)
       rootgrp = netCDF4.Dataset("my_experiment.in.nc", mode="a")
       # ...populate values here...
       rootgrp.close()
    """

def hill_only(clobber):
    """Create input file for hillslope diffusion only example case"""

    # create/open new file
    filename = 'hill_only.in.nc'
    rootgrp = netCDF4.Dataset(filename, "w", format="NETCDF4", clobber=clobber)

    # create dimensions

    # create variables
    
    # create attributes


def main():
    """Select case from command line arguments and generate input file"""

    # get command line arguments
    parser = argparse.ArgumentParser(description='Create input files for '
        'various ICE-CASCADE example cases',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('case', type=str, choices=['hill_only'],
        help='example case to generate')
    parser.add_argument('-c', '--clobber', action='store_true',
        help='allow overwriting existing files')
    args = parser.parse_args()

    # create input file for selected example case
    if args.case == 'hill_only':
        hill_only(args.clobber)

if __name__ == '__main__':
    main()
