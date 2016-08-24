"""
Unit tests for Python ICE-CASCADE core model
"""

import unittest
import numpy as np
from py_ice_cascade import ice_cascade

class model_TestCase(unittest.TestCase):

    def test_arg_sanity(self):
        """Confirm expected failures on init for insane inputs"""
        pass

        # # confirm valid state and parameters
        # if self._output_file is None:
        #     self._abort("missing output file name") 
        # if type(self._x) is not np.ndarray:
        #     self._abort("expect x to be a numpy array")
        # self._nx = self._x.size
        # if self._x.shape != (1, self._nx):
        #     self._abort("expect x to be a row vector")
        # self._delta = np.abs(self._x[0,1]-self._x[0,0])
        # if type(self._y) is not np.ndarray:
        #     self._abort("expect y to be a numpy array")
        # self._ny = self._y.size
        # if self._nx<3 or self._ny<3:
        #     self._abort("minimum grid dimension is 3 x 3")
        # if not np.allclose(np.abs(self._x[0,1:]-self._x[0,:-1]), self._delta):
        #     self._abort("invalid spacing in x-coordinate")
        # if not np.allclose(np.abs(self._y[1:,0]-self._y[:-1,0]), self._delta):
        #     self._abort("invalid spacing in y-coordinate")
        # if type(self._zrx) is not np.ndarray:
        #     self._abort("expect zrx to be a numpy array")
        # if self._zrx.shape != (self._ny, self._nx): 
        #     self._abort("zrx shape does not match coordinate dimensions")
        # if self._time_start is None:
        #     self._abort("missing required parameter 'time_start'")
        # if hasattr(self._time_start, '__len__'):
        #     self._abort("expected scalar for time_start")
        # if self._time_step is None:
        #     self._abort("missing required parameter 'time_step'")
        # if hasattr(self._time_step, '__len__'):
        #     self._abort("expected scalar for time_step")
        # if self._num_steps is None:
        #     self._abort("missing required parameter 'num_steps'")
        # if hasattr(self._num_steps, '__len__'):
        #     self._abort("expected scalar for num_steps")
        # if self._num_steps < 2:
        #     self._abort("must have at least two steps")
        # if self._out_steps.size < 2:
        #     self._abort("expected numpy array with at least 2 elements for out_steps")
        # if self._out_steps[0] != 0:
        #     self._abort("output steps must begin with 0th")
        # if self._out_steps[-1] != self._num_steps:
        #     self._abort("output steps must end with last")
        # if self._hill_on not in [True, False]:
        #     self._abort("hill_on parameter must be boolean")
        # if type(self._hill_kappa) is not np.ndarray:
        #     self._abort("expected hill_kappa to be a numpy array")
        # if self._hill_kappa.shape != (self._ny, self._nx):
        #     self._abort("invalid shape for hill_kappa grid")
        # if len(self._hill_bc) != 4:
        #     self._abort("expect list of 4 boundary conditions in hill_bc")

if __name__ == '__main__':
    unittest.main()