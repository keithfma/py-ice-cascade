"""
Python ICE-CASCADE tectonic uplift-subsidence model component

Linear mode: Uplift rate is linearly interpolated between pre-defined initial
and final states.
"""

from .base import base_model
import numpy as np

class linear_model(base_model):
    r"""
    Tectonic uplift model in which uplift is linearly interpolated between a
    pre-defined initial and final state. 

    Arguments:
        height: 2D numpy array, surface elevation in model domain, [m]
        ui: 2D numpy array, initial uplift rate, [m/a]
        uf: 2D numpy array, final uplift rate, [m/a]
        ti: Scalar, initial time, [a]
        tf: Scalar, final time, [a]

    Let :math:`u(x,y,t)` be the uplift rate as a function of space
    (:math:`x,y`) and time (:math:`t`), and let the initial and final time be
    :math:`t_i, t_f`, respectively. This class defines :math:`u` as:

    .. math::
       u(x,y,t) = u(x,y,t_i) + \frac{u(x,y,t_f) - u(x,y,t_i)}{t_f-t_i} \left( t-t_i \right)

    Omiting the :math:`x,y` coordinates:

    .. math::
       u(t) = u(t_i) + \frac{u(t_f) - u(t_i)}{t_f-t_i} (t - t_i)
       = u_i + \frac{u_f - u_i}{t_f - t_i} (t - t_i)

    It is useful redefine the slope term as :math:`b` and gather up constants
    into a new term :math:`a`, like so:

    .. math::
       u(t) = u_i + b(t - t_i) = (u_i - b t_i) + b t = a + b t

    The total uplift over some time interval :math:`[t_{start},t_{end}]`. This
    is simply the definite integral of the uplift rate:

    .. math::
       \int_{t_{start}}^{t_{end}} (a + b t) dt &= (a t + \frac{1}{2} b t^2 + c)|_{t_{end}} - (a t + \frac{1}{2} b t^2 + c)|_{t_{start}} \\
       &= (a t_{end} + \frac{1}{2} b t_{end}^2) - (a t_{start} + \frac{1}{2} b t_{start}^2) \\
       &= a(t_{end}-t_{start}) + \frac{b}{2} (t_{end}^2 - t_{start}^2)

    Since :math:`a` and :math:`b` are constant coefficients, they are
    precomputed for efficiency.
    """

    def __init__(self, height, ui, uf, ti, tf):

        self.set_height(height)
        self._check_dims(ui)
        self._ui = np.copy(np.double(ui))
        self._check_dims(uf)
        self._uf = np.copy(np.double(uf))
        self._ti = np.asscalar(np.copy(np.double(ti)))
        self._tf = np.asscalar(np.copy(np.double(tf)))
        self._uplift_rate = np.zeros((self._nx, self._ny), dtype=np.double)

        # precompute constant terms, see class docstring
        self._b = (self._uf-self._ui)/(self._tf-self._ti)
        self._a = self._ui-self._b*self._ti

        if self._tf <= self._ti:
            raise ValueError("Initial time must be before final time")

    def _check_dims(self, array):
        """Check that array dims match model dims, or set model dims"""
        if hasattr(self, '_nx') and array.shape != (self._ny, self._nx):
            raise ValueError("Array dims do not match model dims")
        else:
            self._ny, self._nx = array.shape

    def set_height(self, new):
        """Set height grid"""
        new_array = np.copy(np.double(new))
        self._check_dims(new_array)
        self._height = new_array

    def get_height(self):
        """Return height grid as 2D numpy array"""
        return np.copy(self._height)

    def run(self, t_start, t_end):
        """
        Compute total(integrated) uplift over the time interval and update height grid
        
        Arguments:
            t_start: time interval start, [a]
            t_end: time interval end, [a]
        """
        uplift = self._a*(t_end-t_start) + 0.5*self._b*(t_end*t_end-t_start*t_start)
        self._height += uplift 
        self._uplift_rate = uplift/(t_end-t_start)

    def init_netcdf(self, nc, zlib, complevel, shuffle, chunksizes):
        """
        Initialize model-specific variables and attributes in output file
        
        Arguments: 
            nc: netCDF4 Dataset object, output file open for writing 
            zlib: see http://unidata.github.io/netcdf4-python/#netCDF4.Dataset.createVariable
            complevel: " " 
            shuffle: " " 
            chunksizes: " " 
        """

        nc.createVariable('uplift_model', np.dtype('i1')) # scalar
        nc['uplift_model'][...] = True
        nc['uplift_model'].type = self.__class__.__name__ 
        nc['uplift_model'].time_initial = self._ti
        nc['uplift_model'].time_final = self._tf
        nc['uplift_model'].uplift_initial = 'see uplift_dzdt_initial variable'
        nc['uplift_model'].uplift_final = 'see uplift_dzdt_final variable'

        nc.createVariable('uplift_dzdt_initial', np.double, dimensions=('y', 'x'))
        nc['uplift_dzdt_initial'].long_name = 'initial uplift rate'
        nc['uplift_dzdt_initial'].units = 'm / a'
        nc['uplift_dzdt_initial'][:,:] = self._ui

        nc.createVariable('uplift_dzdt_final', np.double, dimensions=('y', 'x'))
        nc['uplift_dzdt_final'].long_name = 'final uplift rate'
        nc['uplift_dzdt_final'].units = 'm / a'
        nc['uplift_dzdt_final'][:,:] = self._uf

        nc.createVariable('uplift_dzdt', np.double, dimensions=('time', 'y', 'x'), 
            zlib=zlib, complevel=complevel, shuffle=shuffle, chunksizes=chunksizes)
        nc['uplift_dzdt'].long_name = 'average tectonic uplift rate'
        nc['uplift_dzdt'].units = 'm / a'

    def to_netcdf(self, nc, time_idx):
        """
        Write model-specific state variables to output file

        Arguments:
            nc: netCDF4 Dataset object, output file open for writing 
            time_idx: integer time index to write to
        """

        nc['uplift_dzdt'][time_idx,:,:] = self._uplift_rate
