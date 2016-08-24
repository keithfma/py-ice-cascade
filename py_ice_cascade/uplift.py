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

        if self._ui.shape != self._uf.shape:
            raise ValueError("Unequal dimensions for initial and final arrays")
        if self._tf <= self._ti:
            raise ValueError("Initial time must be before final time")

    def get_uplift_rate(self, time):
        """Return the uplift rate at time = time"""
        time = np.asscalar(np.copy(np.double(time)))
        return self._ui+(time-self._ti)*(self._uf-self._ui)/(self._tf-self._ti)
    
    def get_uplift(self, start, end):
        """Return total (integrated) uplift over the interval start -> end"""
        # THE BELOW IS INCORRECT, REDO THE INTEGRAL
        start = np.asscalar(np.copy(np.double(start)))
        end = np.asscalar(np.copy(np.double(end)))
        return end*self.get_uplift_rate(end)-start*self.get_uplift_rate(start)

if __name__ == '__main__':

    # basic usage example and "smell test": linear transition between ramp functions
    ny = nx = 101 
    u0 = -1.0*np.linspace(0.0, 1.0, nx).reshape(1,nx)*np.ones((ny,1))
    u1 =  1.0*np.linspace(0.0, 1.0, nx).reshape(1,nx)*np.ones((ny,1))
    t0 = 0.0
    t1 = 1.0
    model = linear(u0, u1, t0, t1)

    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(u0, interpolation='nearest', vmin=-1.0, vmax=1.0)
    plt.title('U(x,y,t) at t = 0')
    plt.subplot(2,2,2)
    plt.imshow(u1, interpolation='nearest', vmin=-1.0, vmax=1.0)
    plt.title('U(x,y,t) at t = end')
    plt.subplot(2,2,4)
    plt.colorbar()
    plt.ion()
    for time in np.linspace(t0, t1, 20):
        plt.subplot(2,2,3)
        plt.cla()
        plt.imshow(model.get_uplift_rate(time), interpolation='nearest', vmin=-1.0, vmax=1.0)
        plt.title("U(x,y,t) at t = {:.2f}".format(time))
        plt.subplot(2,2,4)
        plt.cla()
        plt.imshow(model.get_uplift(t0, time), interpolation='nearest')
        plt.title("U total at t = {:.2f}".format(time))
        plt.pause(0.20)
        
