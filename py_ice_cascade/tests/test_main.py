"""
Unit tests for Python ICE-CASCADE core model
"""

import unittest
import numpy as np
import py_ice_cascade

class model_TestCase(unittest.TestCase):

    def test_irregular_grid(self):
        """Expect failure if grid is not regular"""
        # TODO: implement
        pass

    def test_output_steps(self):
        """Expect output steps to include 1st and last step"""
        # TODO: implement
        pass

    def test_disable_hillslope(self):
        """Confirm ability to disable the hillslope component model"""
        # TODO: implement
        pass

    def test_disable_uplift(self):
        """Confirm ability to disable the uplift component model"""
        # TODO: implement
        pass

if __name__ == '__main__':
    unittest.main()
