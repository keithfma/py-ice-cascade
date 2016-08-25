"""
Unit tests for Python ICE-CASCADE core model
"""

import unittest
import numpy as np
from py_ice_cascade import ice_cascade

class model_TestCase(unittest.TestCase):

    def test_irregular_grid(self):
        """Expect failure if grid is not regular"""
        pass

    def test_output_steps(self):
        """Expect output steps to include 1st and last step"""
        pass

    def test_disable_hillslope(self):
        """Confirm ability to disable the hillslope component model"""
        pass

    def test_disable_uplift(self):
        """Confirm ability to disable the uplift component model"""
        pass

if __name__ == '__main__':
    unittest.main()
