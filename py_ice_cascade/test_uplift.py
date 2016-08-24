"""
Unit tests for Python ICE-CASCADE tectonic uplift-subsidence model component
"""

import unittest
import numpy as np
from py_ice_cascade import uplift 

class linear_TestCase(unittest.TestCase):

    def test_uplift_dims(self):
        """Initial and final uplift dims must match"""
        uplift.linear(np.random.rand(10,10), np.random.rand(10,10), 0, 1)
        self.assertRaises(ValueError, uplift.linear, np.random.rand(10,10), np.random.rand(11,11), 0, 1)

    def test_time_scalar(self):
        """Time bounds must be scalars"""
        uplift.linear(np.random.rand(10,10), np.random.rand(10,10), 0, 1)
        self.assertRaises(ValueError, uplift.linear, np.random.rand(10,10), np.random.rand(10,10), [0,0], 1)
        self.assertRaises(ValueError, uplift.linear, np.random.rand(10,10), np.random.rand(10,10), 0, [1,1])

    def test_time_incr(self):
        """Time bounds must be increasing"""
        uplift.linear(np.random.rand(10,10), np.random.rand(10,10), 0, 1)
        self.assertRaises(ValueError, uplift.linear, np.random.rand(10,10), np.random.rand(10,10), 1, 0)

    def test_random_uplift(self):
        """Check uplift rate and total uplift for random uplift field"""
        nx = ny = 100
        ui = np.random.rand(nx, ny)
        uf = -ui
        ti = 0.0
        tf = 1.0
        epsilon = 0.00001
        model = uplift.linear(ui, uf, ti, tf)

        # correct uplift rate at t = ti, (ti+tf)/2, tf ?
        error_ui = np.abs(model.get_uplift_rate(ti)-ui)
        self.assertTrue(np.all(error_ui < epsilon))

        error_um = np.abs(model.get_uplift_rate((ti+tf)/2.0) - (ui+uf)/2.0)
        self.assertTrue(np.all(error_um < epsilon))

        error_uf = np.abs(model.get_uplift_rate(tf)-uf)
        self.assertTrue(np.all(error_uf < epsilon))

        # correct total uplift at t = ti, tf ?
        error_uti = np.abs(model.get_uplift(ti, ti) - 0.0)
        self.assertTrue(np.all(error_uti < epsilon))
        
        error_utf = np.abs(model.get_uplift(ti, tf) - 0.0)
        self.assertTrue(np.all(error_utf < epsilon))

if __name__ == '__main__':
    unittest.main()
