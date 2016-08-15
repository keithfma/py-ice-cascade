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
    """Tests for ftcs_openbnd model component"""

    def test_steady_1d(self):
        """1D fixed boundaries, converge to constant gradient"""
        
        # define parameters
        lx = 1.0
        npts = 100
        h0 = 0.0
        h1 = 1.0
        kappa = 1.0
        epsilon = 0.001 # arbitrary tolerance
        
        # init and run model
        delta = lx/(npts-1)
        height = np.random.rand(1,npts)
        height[ 0] = h0
        height[-1] = h1
        kappa = kappa*np.ones((1,npts))
        model = py_ice_cascade.hillslope.ftcs(height, delta, kappa, ['constant']*4)
        model.run(1)
        
        # compute solution
        dh_expected = (h1-h0)/(npts-1)

        # check results
        dh_observed = model.get_height()[0,1:]-model.get_height()[0,:-1]
        dh_error = np.abs(dh_observed-dh_expected)
        self.assertTrue(np.all(dh_error < epsilon))

    # # NOTE: Exact solution is working, but this solution requires Dirichlet
    # #   boundary conditions which are not yet implemented
    # def test_steady_2d(self):
    #     """2D fixed unequal boundaries, converge to solution in ref (1)"""
    #     
    #     # define parameters
    #     nx = 100
    #     ny = 50
    #     lx = 1.0
    #     h1 = 1.0
    #     h2 = 0.0
    #     kappa = 1.0
    #     epsilon = 0.001 # arbitrary tolerance

    #     # init and run model
    #     delta = lx/(nx-1)
    #     ly = delta*(ny-1)
    #     height = np.random.rand(ny, nx).astype(np.double)
    #     height[ :, 0] = h1
    #     height[ :,-1] = h1
    #     height[ 0, :] = h1
    #     height[-1, :] = h2
    #     model = py_ice_cascade.hillslope.ftcs_openbnd(height, delta, kappa)
    #     model.run(0.01)

    #     # compute solution
    #     xx = np.linspace(0, lx, nx, dtype=np.double).reshape(( 1,nx))
    #     yy = np.linspace(0, ly, ny, dtype=np.double).reshape((ny, 1))
    #     total = np.zeros((ny,nx), dtype=np.double)
    #     for nn in range(1,400,2): # note: becomes unstable as nn->inf, why?
    #         total += (np.power(-1.0, nn+1)+1.0)/nn * \
    #             np.sin(nn*np.pi*xx/lx) * \
    #             np.sinh(nn*np.pi*yy/lx) / \
    #             np.sinh(nn*np.pi*ly/lx)
    #     height_exact = h1 + 2.0*(h2-h1)/np.pi*total

    #     # check results

    #     # debug
    #     plt.imshow(height, interpolation='nearest')
    #     plt.colorbar()
    #     plt.show()
    #     plt.imshow(model.get_height(), interpolation='nearest')
    #     plt.colorbar()
    #     plt.show()

    def test_transient_1d(self):
        pass

    def test_transient_2d(self):
        pass

if __name__ == '__main__':
    unittest.main()

