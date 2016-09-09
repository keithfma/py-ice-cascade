"""
Python ICE-CASCADE tectonic uplift-subsidence model component
"""

import numpy as np
import matplotlib.pyplot as plt

class model():
    """Base class for uplift model components"""
    def __init__(self):
        pass
    def set_height(self, new):
        raise NotImplementedError
    def get_height(self):
        raise NotImplementedError
    def init_netcdf(self, nc, zlib, complevel, shuffle, chunksizes):
        raise NotImplementedError
    def to_netcdf(self, nc, time_idx):
        raise NotImplementedError
    def run(self, t_start, t_end):
        raise NotImplementedError

class null(model):
    """
    Do-nothing class to be used for disabled uplift component

    Internal height grid is set and returned unchanged
    """

    def __init__(self):
        pass
    def set_height(self, new):
        self._height = np.copy(np.double(new))
    def get_height(self):
        return np.copy(self._height)
    def init_netcdf(self, nc, *args):
        nc.createVariable('uplift_model', np.dtype('i1')) # scalar
        nc['uplift_model'][...] = False 
        nc['uplift_model'].type = self.__class__.__name__ 
    def to_netcdf(*args):
        pass
    def run(*args):
        pass

class linear(model):
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

    The method *get_uplift_rate()* returns the above uplift rate. The method
    *get_uplift()* instead returns the total uplift over some time interval
    :math:`[t_{start},t_{end}]`. This is simply the definite integral of the
    uplift rate:

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
        nc['uplift_model'].uplift_initial = 'see uplift_rate_initial variable'
        nc['uplift_model'].uplift_final = 'see uplift_rate_final variable'

        nc.createVariable('uplift_rate_initial', np.double, dimensions=('y', 'x'))
        nc['uplift_rate_initial'].long_name = 'initial uplift rate'
        nc['uplift_rate_initial'].units = 'm / a'
        nc['uplift_rate_initial'][:,:] = self._ui

        nc.createVariable('uplift_rate_final', np.double, dimensions=('y', 'x'))
        nc['uplift_rate_final'].long_name = 'final uplift rate'
        nc['uplift_rate_final'].units = 'm / a'
        nc['uplift_rate_final'][:,:] = self._uf

        nc.createVariable('uplift_rate', np.double, dimensions=('time', 'y', 'x'), 
            zlib=zlib, complevel=complevel, shuffle=shuffle, chunksizes=chunksizes)
        nc['uplift_rate'].long_name = 'average tectonic uplift rate'
        nc['uplift_rate'].units = 'm / a'

    def to_netcdf(self, nc, time_idx):
        """
        Write model-specific state variables to output file

        Arguments:
            nc: netCDF4 Dataset object, output file open for writing 
            time_idx: integer time index to write to
        """

        nc['uplift_rate'][time_idx,:,:] = self._uplift_rate

# TODO: move to examples and delete
if __name__ == '__main__':

    # Basic usage example and "smell test": 
    # # linear transition between negative and positive ramp functions, "hinge"
    # # is at x=0. The expected result is for the uplift rate to transition from
    # # initial to final values, and the total erosion to decrease to a minimum
    # # of -0.25, then return to 0.

    ny = nx = 101 
    u0 = -1.0*np.linspace(0.0, 1.0, nx).reshape(1,nx)*np.ones((ny,1))
    u1 =  1.0*np.linspace(0.0, 1.0, nx).reshape(1,nx)*np.ones((ny,1))
    t0 = 0.0
    t1 = 1.0
    model = linear(u0, u1, t0, t1)

    plt.subplot(2,2,1)
    plt.imshow(u0, interpolation='nearest', vmin=-1.0, vmax=1.0)
    plt.title('u(x,y,t_i)')
    plt.colorbar()

    plt.subplot(2,2,2)
    plt.imshow(u1, interpolation='nearest', vmin=-1.0, vmax=1.0)
    plt.title('u(x,y,t_f)')
    plt.colorbar()

    plt.subplot(2,2,3)
    plt.imshow(model.get_uplift_rate(t0), interpolation='nearest', vmin=-1.0, vmax=1.0)
    plt.title("u(x,y,{:.2f})".format(t0))
    plt.colorbar()
    
    plt.subplot(2,2,4)
    plt.imshow(model.get_uplift(t0, t0), interpolation='nearest', vmin=-0.25, vmax=0.0)
    plt.title("u_total(x,y,{:.2f})".format(t0))
    plt.colorbar()
    
    for time in np.linspace(t0, t1, 20):
    
        plt.subplot(2,2,3)
        plt.cla()
        plt.imshow(model.get_uplift_rate(time), interpolation='nearest', vmin=-1.0, vmax=1.0)
        plt.title("u(x,y,{:.2f})".format(time))
        
        plt.subplot(2,2,4)
        plt.cla()
        plt.imshow(model.get_uplift(t0, time), interpolation='nearest', vmin=-0.25, vmax=0.0)
        plt.title("u_total(x,y,{:.2f})".format(time))
        
        plt.pause(0.20)

    plt.show()
