"""
Example cases for for Python ICE-CASCADE glacial-fluvial-hillslope landscape
evolution model. 

Examples can be run either directly in the python interpreter, e.g.:

.. code-block:: python

   import py_ice_cascade.examples
   py_ice_cascade.examples.hill_only()

or from the command line by executing the module as a script, e.g.:

.. code-block:: bash

   python -m py_ice_cascade.examples.hill_only
"""

from .hill_only import run_example as hill_only
from .uplift_only import run_example as uplift_only
from .hill_uplift import run_example as hill_uplift
