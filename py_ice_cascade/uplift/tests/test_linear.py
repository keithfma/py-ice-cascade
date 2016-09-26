"""
Unit tests for Python ICE-CASCADE tectonic uplift-subsidence linear model
component
"""

import unittest
import numpy as np
from py_ice_cascade import uplift 

class linear_TestCase(unittest.TestCase):

    def test_uplift_dims(self):
        """Grid dimensions must match"""
        grid = np.random.rand(10,10)
        uplift.linear_model(grid, grid, grid, 0, 1)
        self.assertRaises(ValueError, uplift.linear_model,np.random.rand(11,11), grid, grid, 0, 1)
        self.assertRaises(ValueError, uplift.linear_model,grid, np.random.rand(11,11), grid, 0, 1)
        self.assertRaises(ValueError, uplift.linear_model,grid, grid, np.random.rand(11,11), 0, 1)

    def test_protect_model_dims(self):
        """Attempt to set model grid with incorrect size array throw error"""
        grid = np.random.rand(10,10)
        model = uplift.linear_model(grid, grid, grid, 0, 1)
        self.assertRaises(ValueError, model.set_height, np.random.rand(11,11))

    def test_time_scalar(self):
        """Time bounds must be scalars"""
        grid = np.random.rand(10,10)
        uplift.linear_model(grid, grid, grid, 0, 1)
        self.assertRaises(ValueError, uplift.linear_model, grid, grid, grid, [0,0], 1    )
        self.assertRaises(ValueError, uplift.linear_model, grid, grid, grid,     0, [1,1])

    def test_time_incr(self):
        """Time bounds must be increasing"""
        grid = np.random.rand(10,10)
        uplift.linear_model(grid, grid, grid, 0, 1)
        self.assertRaises(ValueError, uplift.linear_model, grid, grid, grid, 1, 0)

    def test_random_uplift(self):
        """Check total uplift for random uplift field"""
        nx = ny = 100
        h0 = np.zeros((ny, nx))
        ui = np.random.rand(nx, ny)
        uf = -ui # uplift integrates to 0 over [ti, tf] interval
        ti = 0.0
        tf = 1.0
        epsilon = 0.00001

        model = uplift.linear_model(h0, ui, uf, ti, tf)
        model.run(ti, tf)
        error_htf = np.abs(h0-model.get_height())
        self.assertTrue(np.all(error_htf < epsilon))

if __name__ == '__main__':
    unittest.main()
