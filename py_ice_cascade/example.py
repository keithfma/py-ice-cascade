"""
Example cases for for Python ICE-CASCADE glacial-fluvial-hillslope landscape
evolution model. 

Examples can be run either directly in the python interpreter, e.g.:

.. code-block:: python

   import py_ice_cascade
   py_ice_cascade.example.hill_only

or from the command line by executing the module as a script, e.g.:

.. code-block:: bash

   python -m py_ice_cascade.example hill_only

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
    mask = np.ones((ny, nx))
    kappa_active = 0.01
    kappa_inactive = 0.0
    bcs = ['constant']*4
    time_start = 0.0
    time_step = 0.1
    num_steps = 10
    out_steps = np.arange(0,num_steps)

    hill = py_ice_cascade.hillslope.ftcs(zrx, mask, delta, kappa_active,
        kappa_inactive, bcs)

    uplift = py_ice_cascade.uplift.null()
    
    mod = py_ice_cascade.ice_cascade.model(hill, uplift, 
        x, y, zrx, time_start, time_step, num_steps, out_steps, verbose=True)
    mod.run('example.hill_only.out.nc')

def uplift_only():
    """uplift-only example"""

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
    time_end = time_start + time_step*(num_steps-1)
    out_steps = np.arange(0,num_steps)
    uplift_start = np.zeros((ny,nx), dtype=np.double)
    uplift_end = np.ones((ny,nx), dtype=np.double)

    hill = py_ice_cascade.hillslope.null() 

    uplift = py_ice_cascade.uplift.linear(zrx, uplift_start, uplift_end, time_start, time_end)

    mod = py_ice_cascade.ice_cascade.model(hill, uplift, 
        x, y, zrx, time_start, time_step, num_steps, out_steps, verbose=True)
    mod.run('example.uplift_only.out.nc')

def hill_uplift():
    """hillslope diffusion with uplift example"""

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
    time_end = time_start + time_step*(num_steps-1)
    out_steps = np.arange(0,num_steps)
    hill_mask = np.ones((ny, nx))
    hill_kappa_active = 0.01
    hill_kappa_inactive = 0.0
    hill_bc = ['constant']*4
    uplift_start = np.zeros((ny,nx), dtype=np.double)
    uplift_end = np.ones((ny,nx), dtype=np.double)

    hill = py_ice_cascade.hillslope.ftcs(zrx, hill_mask, delta, hill_kappa_active,
        hill_kappa_inactive, hill_bc)

    uplift = py_ice_cascade.uplift.linear(zrx, uplift_start, uplift_end, time_start, time_end)

    mod = py_ice_cascade.ice_cascade.model(hill, uplift, 
        x, y, zrx, time_start, time_step, num_steps, out_steps, verbose=True)
    mod.run('example.hill_uplift.out.nc')

# command line interface
if __name__ == '__main__':

    valid_cases = ['hill_only', 'uplift_only', 'hill_uplift']

    parser = argparse.ArgumentParser(description='Run example cases for for '
        'Python ICE-CASCADE glacial-fluvial-hillslope landscape evolution model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('case', type=str, choices=valid_cases,
        help='example case to run')
    args = parser.parse_args()

    if args.case == 'hill_only':
        hill_only()
    elif args.case == 'uplift_only':
        uplift_only()
    elif args.case == 'hill_uplift':
        hill_uplift()
