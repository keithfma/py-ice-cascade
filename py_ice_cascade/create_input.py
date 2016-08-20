"""
Console script with options to create input files for example cases
"""

import argparse
import sys
import netCDF4

def create_template(filename, clobber=False):
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

    # create/open new file, handle errors with user-friendly message
    try:
        rootgrp = netCDF4.Dataset(filename, "w", format="NETCDF4", clobber=clobber)
    except RuntimeError as err:
        if str(err)  == "NetCDF: File exists && NC_NOCLOBBER":
            print("create_input: file exists and clobber is False, aborting")
        else:
            print("create_input: unknown error creating new netCDF file, aborting")
            print(err)
        sys.exit()

    # create dimensions

    # create variables
    
    # create attributes

    # close file
    rootgrp.close()

def create_hill_only(filename, clobber=False):
    """Create input file for hillslope diffusion only example case"""

    # create/open new file with expected dimensions, variables, and attributes
    create_template(filename, clobber=clobber)
    rootgrp = netCDF4.Dataset(filename, "a", format="NETCDF4", clobber=clobber)

    # populate attributes

    # populate variables

    # close file
    rootgrp.close()

def cli():
    """
    Command-line tool to generate valid input files for Python ICE-CASCADE
    glacial-fluvial-hillslope landscape evolution model. Input cases include a
    blank template and various simple examples. 

    Installed as a console-script called *ice-cascade-create-input*.
    Additional help can be accessed with the *-h* flag.
    """
    valid_cases = ['template', 'hill_only']

    # get command line arguments
    parser = argparse.ArgumentParser(description='Create input files for '
        'various ICE-CASCADE example cases',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('case', type=str, choices=valid_cases,
        help='example case to generate')
    parser.add_argument('-f', '--filename', type=str, default="input.nc",
        help='output filename')
    parser.add_argument('-c', '--clobber', action='store_true',
        help='allow overwriting existing files')
    args = parser.parse_args()

    # create input file for selected example case
    if args.case == 'template':
        create_template(args.filename, clobber=args.clobber)
    if args.case == 'hill_only':
        create_hill_only(args.filename, clobber=args.clobber)

if __name__ == '__main__':
    main()
