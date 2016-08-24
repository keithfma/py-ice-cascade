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
    def __init__(self):
        pass

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
