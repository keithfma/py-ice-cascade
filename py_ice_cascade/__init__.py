# shared project metadata
# # use __XXX__ for standard metadata vars
# # use _XXX for custom metadata vars
__name__ = 'py_ice_cascade'
__version__ = '0.0.1'
_description = 'Python implementation of ICE-CASCADE landscape evolution model'
_url = 'https://github.com/keithfma/py_ice_cascade'
_author = 'Keith F. Ma'
_author_email = 'keithfma@gmail.com'

# load modules
from .main import main_model
from . import hillslope
from . import uplift

# setup package-level logger
import logging
import sys
logger = logging.getLogger(name=__name__) 
handler = logging.StreamHandler(stream=sys.stdout)
formatter = logging.Formatter(
    '%(name)s | %(levelname)s | %(message)s | %(asctime)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
