"""
Python ICE-CASCADE tectonic uplift-subsidence model component
"""

import numpy as np
import matplotlib.pyplot as plt

class null():
    """Do-nothing class to be used for disabled uplift component"""
    def __init__(self):
        pass

class linear():
    """
    Tectonic uplift model in which uplift (:math:`U = f(x,y,t)`) is linearly
    interpolated between a pre-defined initial and final state.
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
        self._ti = np.copy(np.double(ti))
        self._tf = np.copy(np.double(tf))

        if self._ui.shape != self._uf.shape:
            raise ValueError("Unequal dimensions for initial and final arrays")
        if self._ti.isscalar() is not True or self._tf.isscalar() is not True:
            raise ValueError("Initial and final times must be scalar")
        if self._tf <= self.ti:
            raise ValueError("Initial time must be before final time")

if __name__ == '__main__':

    # basic usage example and "smell test": linear transition between ramp functions
    ny = nx = 101 
    u0 = -1.0*np.linspace(0.0, 1.0, nx).reshape(1,nx)*np.ones((ny,1))
    u1 =  1.0*np.linspace(0.0, 1.0, nx).reshape(1,nx)*np.ones((ny,1))

    # init figure
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(u0, interpolation='nearest', vmin=-1.0, vmax=1.0)
    plt.title('U(x,y,x) at time = 0')
    plt.subplot(1,3,2)
    plt.imshow(u1, interpolation='nearest', vmin=-1.0, vmax=1.0)
    plt.title('U(x,y,x) at time = end')
    plt.show(block=False)
