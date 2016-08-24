"""
Unit tests for Python ICE-CASCADE tectonic uplift-subsidence model component
"""

import unittest
from py_ice_cascade import uplift 

class linear_TestCase(unittest.TestCase):

    def test_uplift_dims(self):
        """Initial and final uplift dims must match"""
        pass

    def test_time_bnds(self):
        """Time bounds must be increasing scalars"""
        pass  

if __name__ == '__main__':
    unittest.main()
