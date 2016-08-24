"""
Python ICE-CASCADE tectonic uplift-subsidence model component
"""

import numpy as np
import matplotlib.pyplot as plt

class null():
    """
    Do-nothing class to be used for disabled uplift component
    """
    
    def __init__(self, *args):
        pass
    def get_uplift_rate(self, *args):
        pass
    def get_uplift(self, *args):
        pass

class linear():
    r"""
    Tectonic uplift model in which uplift is linearly interpolated between a
    pre-defined initial and final state. 

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

    def __init__(self, ui, uf, ti, tf):
        """
        Initialize model attributes
        
        Arguments:
            ui: 2D numpy array, initial uplift rate, [m/a]
            uf: 2D numpy array, final uplift rate, [m/a]
            ti: Scalar, initial time, [a]
            tf: Scalar, final time, [a]
        """

        self._ui = np.copy(np.double(ui))
        self._uf = np.copy(np.double(uf))
        self._ti = np.asscalar(np.copy(np.double(ti)))
        self._tf = np.asscalar(np.copy(np.double(tf)))

        # precompute constant terms, see class docstring
        self._b = (uf-ui)/(tf-ti)
        self._a = ui-self._b*ti

        # NOTE: this first test may not be needed, precomputation should catch errors
        if self._ui.shape != self._uf.shape:
            raise ValueError("Unequal dimensions for initial and final arrays")
        if self._tf <= self._ti:
            raise ValueError("Initial time must be before final time")

    def get_uplift_rate(self, time):
        """Return the uplift rate at time = time"""
        return self._a+self._b*time 
    
    def get_uplift(self, t_start, t_end):
        """Return total (integrated) uplift over the interval [t_start, t_end]"""
        return self._a*(t_end-t_start) + 0.5*self._b*(t_end*t_end-t_start*t_start)

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
