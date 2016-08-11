"""
Command-line front-end for Python ICE-CASCADE hillslope erosion-deposition
model component

usage: run_hillslope.py [-h] [-o OUTPUT_FILE]

Command-line front-end for Python ICE-CASCADE hillslope erosion-deposition
model component

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT_FILE, --output_file OUTPUT_FILE
                        Output file name (netCDF) (default: hillslope_out.nc)
"""

import argparse

def run():

	# get commmand line inputs
	parser = argparse.ArgumentParser(description=__doc__, 
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-o', '--output_file', type=str, default='hillslope_out.nc',
		help='Output netCDF file name')
	args = parser.parse_args()


if __name__ == '__main__':
	run()
