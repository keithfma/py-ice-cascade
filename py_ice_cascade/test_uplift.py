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

if __name__ == '__main__':
    unittest.main()
