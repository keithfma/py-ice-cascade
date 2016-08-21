"""
Console script with options to create input files for example cases
"""

import argparse
import sys
import netCDF4

# TODO: Update to make use of new ice-cascade model class. This file should
# include a few functions to generate input & output files for example cases.

def hill_only(clobber=False):
    """Generate input and output files for hillslope-diffusion-only example"""

def cli():
    """
    Command-line tool to run example cases for for Python ICE-CASCADE glacial-
    fluvial-hillslope landscape evolution model. Each case generates both an
    input and an output file. 
    
    Installed as a console-script called *ice-cascade-example*. Additional help
    can be accessed with the *-h* flag.
    """
    valid_cases = ['hill_only']

    # get command line arguments
    parser = argparse.ArgumentParser(description='Run various ICE-CASCADE '
        'example cases, and generate both input and output files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('case', type=str, choices=valid_cases,
        help='example case to generate')
    parser.add_argument('-c', '--clobber', action='store_true',
        help='allow overwriting existing files')
    args = parser.parse_args()

    # create input file for selected example case
    if args.case == 'hill_only':
        create_hill_only(clobber=args.clobber)

if __name__ == '__main__':
    main()
