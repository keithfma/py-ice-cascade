"""
Console script with options to create input files for example cases
"""

import argparse
import os
import sys
import netCDF4

def hill_only(clobber):
    """Create input file for hillslope diffusion only example case"""

    # create/open new file
    filename = 'hill_only.in.nc'
    rootgrp = netCDF4.Dataset(filename, "w", format="NETCDF4", clobber=clobber)

    


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
