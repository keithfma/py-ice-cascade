"""
Console script with options to create input files for example cases
"""

import argparse

def hill_only(filename):
    """Create input file for hillslope diffusion only example case"""
    print(filename)

def main():
    """Select case from command line arguments and generate input file"""

    # get command line arguments
    parser = argparse.ArgumentParser(description='Create input files for '
        'various ICE-CASCADE example cases',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('case', type=str, choices=['hill_only'],
        help='example case to generate')
    parser.add_argument('filename', type=str, default='example_in.nc',
        help='example case input file name')
    args = parser.parse_args()

    # create input file for selected example case
    if args.case == 'hill_only':
        hill_only(args.filename)

if __name__ == '__main__':
    main()
