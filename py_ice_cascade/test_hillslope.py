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

    def test_steady_dirichlet(self):
        """
        Compare against exact solution for laplace equation with sinusoid at
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
        self.assertTrue(np.max(h_error) < 0.0001)

if __name__ == '__main__':
    unittest.main()

