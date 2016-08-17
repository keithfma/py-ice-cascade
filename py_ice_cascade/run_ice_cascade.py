"""
Command-line front-end for Python ICE-CASCADE glacial-fluvial-hillslope
landscape evolution model.
"""

# #
# usage: run_ice_cascade.py [-h] [-o OUTPUT_FILE]
# 
# Command-line front-end for Python ICE-CASCADE glacial-fluvial-hillslope
# landscape evolution model.
# 
# optional arguments:
#   -h, --help            show this help message and exit
#   -o OUTPUT_FILE, --output_file OUTPUT_FILE
#                         Output netCDF file name (default: model_out.nc)
# #

import argparse

def run():

	# get commmand line inputs
	parser = argparse.ArgumentParser(description=__doc__, 
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-o', '--output_file', type=str, default='model_out.nc',
		help='Output netCDF file name')
	args = parser.parse_args()


if __name__ == '__main__':
	run()
