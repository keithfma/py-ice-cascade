"""
Example cases for for Python ICE-CASCADE glacial-fluvial-hillslope landscape
evolution model. 

Examples can be run either directly in the python interpreter, e.g.:

.. code-block:: python
   import py_ice_cascade
   py_ice_cascade.example.hill_only

or from the command line by executing the module as a script, e.g.:

.. code-block:: bash
   python -m py_ice_cascade.example -c hill_only

In the latter case, use the *-h* flag to get additional help
"""

import argparse
import py_ice_cascade
import numpy as np

def hill_only():
    """hillslope-diffusion-only example"""

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
    hill_kappa_active = 0.01
    hill_kappa_inactive = 0.0
    hill_bc = ['constant']*4
    
    mod = py_ice_cascade.ice_cascade.model()
    mod.set_param_from_var(x=x, y=y, zrx=zrx, time_start=time_start,
        time_step=time_step, num_steps=num_steps, out_steps=out_steps,
        hill_on=hill_on, hill_kappa_active=hill_kappa_active,
        hill_kappa_inactive=hill_kappa_inactive, hill_bc=hill_bc, verbose=True)
    mod.run('example.hill_only.out.nc', verbose=True)

def uplift_only():
    """uplift-only example"""
    raise NotImplementedError

def hill_with_uplift():
    """hillslope diffusion with uplift example"""
    raise NotImplementedError

# command line interface
if __name__ == '__main__':

    valid_cases = ['hill_only', 'uplift_only', 'hill_with_uplift']

    parser = argparse.ArgumentParser(description='Run example cases for for '
        'Python ICE-CASCADE glacial-fluvial-hillslope landscape evolution model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', type=str, choices=valid_cases,
        help='example case to generate', required=True)
    args = parser.parse_args()

    if args.c == 'hill_only':
        hill_only()
    elif args.c == 'uplift_only':
        uplift_only()
    elif args.c == 'hill_with_uplift':
        hill_with_uplift()
