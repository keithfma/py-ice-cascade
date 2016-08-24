"""
Console script with options to create input files for example cases
"""

import argparse
import sys
import netCDF4
import py_ice_cascade
import shutil
import numpy as np

def hill_only(clobber=False):
    """Generate input and output files for hillslope-diffusion-only example"""

    ny = 50
    nx = 100
    lx = 1.0
    delta = lx/(nx-1) 
    ly = delta*(ny-1)
    x = np.linspace(0, lx, nx)
    y = np.linspace(0, ly, ny)
    zrx = np.pad(np.random.rand(ny-2, nx-2), 1, 'constant', constant_values=0)
    time_start = 0.0
    time_step = 0.1
    num_steps = 10
    out_steps = np.arange(0,num_steps)
    hill_on = True
    hill_kappa = 0.01*np.ones((ny, nx))
    hill_bc = ['constant']*4
    
    mod = py_ice_cascade.ice_cascade.model()
    mod.set_param_from_var(x=x, y=y, zrx=zrx, time_start=time_start,
        time_step=time_step, num_steps=num_steps, out_steps=out_steps,
        hill_on=hill_on, hill_kappa=hill_kappa, hill_bc=hill_bc, verbose=True)
    mod.to_input_file('ex.hill_only.in.nc', clobber=True, verbose=True)
    mod.run('ex.hill_only.out.nc', clobber=True, verbose=True, display=False)

def uplift_only(clobber=False):
    """Generate input and output files for uplift-only example"""
    raise NotImplementedError

def hill_with_uplift(clobber=False):
    """Generate input and output files for hillslope diffusion with uplift example"""
    raise NotImplementedError

def cli():
    """
    Command-line tool to run example cases for for Python ICE-CASCADE glacial-
    fluvial-hillslope landscape evolution model. Each case generates both an
    input and an output file. 
    
    Installed as a console-script called *ice-cascade-example*. Additional help
    can be accessed with the *-h* flag.
    """
    valid_cases = ['hill_only', 'uplift_only', 'hill_with_uplift']

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
        hill_only(clobber=args.clobber)
    elif args.case == 'uplift_only':
        uplift_only(clobber=args.clobber)
    elif args.case == 'hill_with_uplift':
        hill_with_uplift(clobber=args.clobber)

if __name__ == '__main__':
    main()
