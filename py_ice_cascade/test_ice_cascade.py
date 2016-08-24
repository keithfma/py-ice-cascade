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

if __name__ == '__main__':
    unittest.main()
