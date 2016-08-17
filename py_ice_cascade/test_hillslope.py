"""
Unit tests for Python ICE-CASCADE hillslope erosion-deposition model component

References:

(1) Holman, J. P. (2002). Heat transfer (pp. 75) 
"""

import unittest
import numpy as np
import py_ice_cascade

import matplotlib.pyplot as plt # debug only

class ftcs_TestCase(unittest.TestCase):
    """Tests for hillslope ftcs model component"""

    def test_steady_bc_constant(self):
        """
        Compare against exact solution for Laplace equation with sinusoid at
        y=max and zero at other boundaries
        """
        
        # parameters
        h0 = 1.0
        nx = 100
        ny = 50
        lx = 1.0
        delta = lx/(nx-1)
        ly = delta*(ny-1)

        # exact solution
        xx = np.linspace(0, lx, nx, dtype=np.double).reshape(( 1,nx))
        yy = np.linspace(0, ly, ny, dtype=np.double).reshape((ny, 1))
        h_exact = h0/np.sinh(np.pi*ly/lx)*np.sin(np.pi*xx/lx)*np.sinh(np.pi*yy/lx)

        # numerical solution
        h_init = np.zeros((ny, nx))
        h_init[-1,:] = h0*np.sin(np.pi*xx/lx)
        kappa = np.ones((ny,nx))
        bcs = ['constant']*4
        model = py_ice_cascade.hillslope.ftcs(h_init, delta, kappa, bcs)
        model.run(0.25)

        # check errors
        h_error = np.abs(model.get_height()-h_exact)
        self.assertTrue(np.max(h_error) < 0.001)

    def test_steady_layered_kappa(self):
        """Compare against exact solution for diffusion in 2 layered material"""

        # parameters
        nx = 100
        ny = 5 
        lx = 1.0
        delta = lx/(nx-1)
        xx = np.linspace(0, lx, nx, dtype=np.double).reshape((1,nx))*np.ones((ny,1))
        l0 = 0.5*(xx[0,50]+xx[0,51]) # transition at midpoint
        l1 = lx-l0
        h0 = 1.0
        h1 = 0.0
        k0 = 1.0
        k1 = 0.5
        t_end = 1.5
        epsilon = 0.001

        # exact solution (resistance = l/k in series) 
        qq = (h0-h1)/(l0/k0+l1/k1)
        hb = h0-qq*l0/k0 # or: hb = qq*l1/k1-h1 
        xx = np.linspace(0, lx, nx, dtype=np.double).reshape((1,nx))*np.ones((ny,1))
        h_exact = np.where(xx <= l0, h0+(hb-h0)/l0*xx, hb+(h1-hb)/l1*(xx-l0))

        # numerical solution
        h_init = np.zeros((ny, nx))
        h_init[:,0] = h0
        h_init[:,-1] = h1
        kappa = np.where(xx <= l0, k0, k1)
        bcs = ['closed', 'closed', 'constant', 'constant'] # TODO: must be no-flux at steady state 
        model = py_ice_cascade.hillslope.ftcs(h_init, delta, kappa, bcs)
        model.run(t_end)

        # check errors
        h_error = np.abs(model.get_height()-h_exact)
        self.assertTrue(np.max(h_error) < epsilon)

if __name__ == '__main__':
    unittest.main()

