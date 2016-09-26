"""
Unit tests for Python ICE-CASCADE uplift-only example case
"""

import os 
import unittest
import py_ice_cascade.examples

class uplift_only_TestCase(unittest.TestCase):

    def test_run_successfully(self):
        """Confirm the example runs without error"""
        file_name = py_ice_cascade.examples.uplift_only()
        os.remove(file_name)
