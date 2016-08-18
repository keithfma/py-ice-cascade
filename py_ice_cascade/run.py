"""
Command-line front-end for Python ICE-CASCADE glacial-fluvial-hillslope
landscape evolution model. 
"""

import argparse

def main():
    """
    Parse command-line arguments and run the ICE-CASCADE model. This function
    is installed as a console-script called "ice-cascade". Additional help can
    be accessed with the command `ice-cascade -h`.
    """

    # init commmand line interface
    parser = argparse.ArgumentParser(description='Command-line front-end for '
        'Python ICE-CASCADE glacial-fluvial-hillslope landscape evolution model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input_file', type=str, default='in.nc',
        help='output netCDF file name')
    parser.add_argument('-o', '--output_file', type=str, default='out.nc',
        help='output netCDF file name')

    # get command line arguments    
    args = parser.parse_args()

if __name__ == '__main__':
    main()
