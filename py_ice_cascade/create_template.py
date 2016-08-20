"""
Create a template input file *template.in.nc* for the Python ICE-CASCADE
landscape evolution model. The template defines the expected dimensions,
variables, and attributes.  To set up a model run, make a copy of the template
file, then open it for editing and populate the values. Using the netCDF4
package (http://unidata.github.io/netcdf4-python/), the commands would be:

.. code-block:: python
   import netCDF4
   import shutil
   shutil.copy("template.in.nc", "my_experiment.in.nc")
   rootgrp = netCDF4.Dataset("my_experiment.in.nc", mode="a")
   # ...populate values here...
   rootgrp.close()
"""

