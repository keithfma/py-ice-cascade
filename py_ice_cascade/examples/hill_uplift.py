"""
Hillslope & uplift example case for PY_ICE_CASCADE landscape evolution model
"""

import py_ice_cascade
import numpy as np

def run_example():
    """
    Run hillslope diffusion with uplift example, save results to file, and
    return output file name
    """

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
    output_filename = 'example.hill_uplift.out.nc'

    hill = py_ice_cascade.hillslope.ftcs_model(zrx, hill_mask, delta, hill_kappa_active,
        hill_kappa_inactive, hill_bc)

    uplift = py_ice_cascade.uplift.linear_model(zrx, uplift_start, uplift_end, time_start, time_end)

    mod = py_ice_cascade.main_model(hill, uplift, 
        x, y, zrx, time_start, time_step, num_steps, out_steps, verbose=True)
    mod.run(output_filename, clobber=True)

    return output_filename

if __name__ == '__main__':
    run_example()
