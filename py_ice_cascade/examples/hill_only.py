"""
Hillslope-only example case for PY_ICE_CASCADE landscape evolution model
"""

import py_ice_cascade
import numpy as np

def run_example():
    """
    Run hillslope-diffusion-only example, save results to file, and return
    output file name
    """
 
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
    output_file = 'example.hill_only.out.nc'

    hill = py_ice_cascade.hillslope.ftcs_model(zrx, mask, delta, kappa_active,
        kappa_inactive, bcs)

    uplift = py_ice_cascade.uplift.null_model()
    
    mod = py_ice_cascade.main_model(hill, uplift, 
        x, y, zrx, time_start, time_step, num_steps, out_steps, verbose=True)
    mod.run(output_file, clobber=True)

    return output_file

if __name__ == '__main__':
    run_example()
