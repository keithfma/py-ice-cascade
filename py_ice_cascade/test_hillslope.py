"""
Unit tests for Python ICE-CASCADE hillslope erosion-deposition model component
"""

import unittest
import numpy as np
import py_ice_cascade

import matplotlib.pyplot as plt # debug only

class ftcs_openbnd_TestCase(unittest.TestCase):
    """Tests for ftcs_openbnd model component"""

    def test_steady_1d(self):
        """1D fixed boundaries, converge to constant gradient"""
        # define parameters
        lx = 1.0
        npts = 100
        h0 = 0.0
        h1 = 1.0
        dh_expected = (h1-h0)/(npts-1)
        epsilon = 0.001 # tolerance for error in dh
        kappa = 1.0 # arbitrary
        # init and run model
        delta = lx/(npts-1)
        height = np.random.rand(npts)
        height[0] = h0
        height[-1] = h1
        model = py_ice_cascade.hillslope.ftcs_openbnd(height, delta, kappa)
        model.run(1)
        # check results
        dh_observed = model.get_height()[0,1:]-model.get_height()[0,:-1]
        dh_error = np.abs(dh_observed-dh_expected)
        self.assertTrue(np.all(dh_error < epsilon))

    def test_steady_2d(self):
        pass

    def test_transient_1d(self):
        pass

    def test_transient_2d(self):
        pass
