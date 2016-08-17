"""
Python ICE-CASCADE hillslope erosion-deposition model component

References:

    (1) Becker, T. W., & Kaus, B. J. P. (2016). Numerical Modeling of Earth
    Systems: Lecture Notes for USC GEOL557 (1.1.4)

    (2) Spiegelman, M. (2000). Myths & Methods in Modeling. Columbia University
    Course Lecture Notes, 202. http://doi.org/10.1007/BF00763500
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import sys

class ftcs():
    r"""
    Hillslope diffusion model using forward-time center-space (FTCS) finite
    diffence scheme. 
    
    Overview of FTCS scheme with spatially variable diffusivity (see reference
    (1) and (2)). Starting with the basic diffusion equation:  
    
    .. math::
       \frac{\partial H}{\partial t} = \nabla \cdot q

    Discretize derivative terms using FTCS scheme:
    
    .. math::
       \frac{H^{n+1}_{i,j} - H^{n}_{i,j}}{\Delta_t} = 
           \frac{(q_x)^{n}_{i,j+1/2} - (q_x)^{n}_{i,j-1/2}}{\Delta_x} + 
           \frac{(q_y)^{n}_{i+1/2,j} - (q_y)^{n}_{i-1/2,j}}{\Delta_y} 

    Expand the fluxes in terms of height gradient:
    
    .. math::
       \frac{1}{\Delta_t} \left( H^{n+1}_{i,j} - H^{n}_{i,j} \right) = 
           \frac{1}{\Delta_x} \left[ 
           \kappa_{i,j+1/2} \frac{H^{n}_{i,j+1}-H^{n}_{i,j}}{\Delta_x} - 
           \kappa_{i,j-1/2} \frac{H^{n}_{i,j}-H^{n}_{i,j-1}}{\Delta_x}
           \right] + \\
           \frac{1}{\Delta_y} \left[ 
           \kappa_{i+1/2,j} \frac{H^{n}_{i+1,j}-H^{n}_{i,j}}{\Delta_y} - 
           \kappa_{i-1/2,j} \frac{H^{n}_{i,j}-H^{n}_{i-1,j}}{\Delta_y}
           \right]

    Use the mean diffusivity at midpoints:

    .. math::
       \frac{1}{\Delta_t} \left( H^{n+1}_{i,j} - H^{n}_{i,j} \right) = 
           \frac{1}{\Delta_x} \left[ 
           \frac{\kappa_{i,j+1} + \kappa_{i,j}}{2} \frac{H^{n}_{i,j+1}-H^{n}_{i,j}}{\Delta_x} - 
           \frac{\kappa_{i,j} + \kappa_{i,j-1}}{2} \frac{H^{n}_{i,j}-H^{n}_{i,j-1}}{\Delta_x}
           \right] + \\
           \frac{1}{\Delta_y} \left[ 
           \frac{\kappa_{i+1,j} + \kappa_{i,j}}{2} \frac{H^{n}_{i+1,j}-H^{n}_{i,j}}{\Delta_y} - 
           \frac{\kappa_{i,j} + \kappa_{i-1,j}}{2} \frac{H^{n}_{i,j}-H^{n}_{i-1,j}}{\Delta_y}
           \right]

    Assume equal x- and y- grid spacing and collect coefficients:
        
    .. math::
       \frac{1}{\Delta_t} \left( H^{n+1}_{i,j} - H^{n}_{i,j} \right) = 
           \frac{1}{2 \Delta_{xy}^2} \left[ 
           (\kappa_{i,j+1} + \kappa_{i,j}) (H^{n}_{i,j+1}-H^{n}_{i,j}) - 
           (\kappa_{i,j} + \kappa_{i,j-1}) (H^{n}_{i,j}-H^{n}_{i,j-1}) + \\
           (\kappa_{i+1,j} + \kappa_{i,j}) (H^{n}_{i+1,j}-H^{n}_{i,j}) - 
           (\kappa_{i,j} + \kappa_{i-1,j}) (H^{n}_{i,j}-H^{n}_{i-1,j})
           \right]

    Expand and collect terms:

    .. math::
       H^{n+1}_{i,j} =  H^{n}_{i,j} + \frac{\Delta_t}{2 \Delta_{xy}^2} \left[ 
       H^{n}_{i,j+1} (\kappa_{i,j+1} + \kappa_{i,j}) +  
       H^{n}_{i,j-1} (\kappa_{i,j} + \kappa_{i,j-1}) +
       H^{n}_{i+1,j} (\kappa_{i+1,j} + \kappa_{i,j}) + \\ 
       H^{n}_{i-1,j} (\kappa_{i,j} + \kappa_{i-1,j}) - 
       H^{n}_{i,j} (4 \kappa_{i,j} + \kappa_{i,j+1} + \kappa_{i,j-1} + \kappa_{i+1,j} + \kappa_{i-1,j})
       \right]

    The above scheme is modified at boundary points. Supported boundary conditions are:
    
    * *constant*: :math:`\frac{\partial H}{\partial t} = 0`
    
    * *closed*: no flux out of boundary (e.g. :math:`(q_x)_{i,j+1/2} = 0` at
      :math:`x_{max}`)

    * *open*: no flux gradient normal to boundary, material passes through
      (e.g. :math:`\frac{\partial q_x}{\partial x} = 0` at :math:`x_{max}`)
    
    * *cyclic*: flux at opposing boundaries is equal (e.g.
      :math:`(q_x)_{i,-1/2} = (q_x)_{i,\text{end}+1/2}`)
    
    * *mirror*: boundary flux is equal and opposite incoming flux (e.g.
      :math:`(q_x)_{i,j+1/2} = -(q_x)_{i,j-1/2}` at :math:`x_{max}`)

    """

    # NOTE: attributes and methods with the "_" prefix are considered private,
    #       use outside the object at your own risk

    def __init__(self, height, delta, kappa, bc):
        """
        Arguments:
            height = 2D numpy array, surface elevation in model domain, [m]
            delta = Scalar double, grid spacing, assumed square, [m]
            kappa = 2D numpy array, diffusion coefficient, [m**2/a]
            bc = List of boundary conditions names for [y=0, y=end, x=0, x=end]
        """

        self._height = None
        self._delta = None
        self._kappa = None
        self._nx = None
        self._ny = None
        self._bc_x0 = None
        self._bc_x1 = None
        self._bc_y0 = None
        self._bc_y1 = None
        self._valid_bcs = set(['constant', 'closed', 'open', 'cyclic', 'mirror'])

        self.set_height(height)
        self.set_diffusivity(kappa)
        self._delta = np.copy(np.double(delta))
        self._bc = list(bc)
        self._set_coeff_matrix()

    def set_height(self, new):
        """Set height grid internal attribute"""
        new_array = np.copy(np.double(new))
        if new_array.ndim != 2:
            print("hillslope: height is not a 2D array"); sys.exit()
        if (self._height != None) and (new_array.shape != (self._ny, self._nx)):
            print("hillslope: cannot change shape of height grid"); sys.exit()
        self._ny, self._nx = new_array.shape
        self._height = np.ravel(new_array, order='C')

    def get_height(self):
        """Return height grid as 2D numpy array"""
        return np.copy(self._height.reshape((self._ny, self._nx), order='C'))

    def set_diffusivity(self, new):
        """Set diffusivity grid internal attribute"""
        self._kappa = np.copy(np.double(new))
        if self._kappa.ndim != 2:
            print("hillslope: diffusivity is not a 2D array"); sys.exit()
        if self._kappa.shape != (self._ny, self._nx):
            print("hillslope: diffusitity grid dims do not match height grid"); sys.exit()

    def _set_coeff_matrix(self):
        """Define sparse coefficient matrix for dHdt stencil"""

        # NOTE: FTCS is a 5-point stencil, since diffusivity is a grid, all
        # coefficients are potentially unique. 

        # init variables
        A = scipy.sparse.lil_matrix((self._ny*self._nx, self._ny*self._nx), dtype=np.double) # lil format is fast to populate
        c = 1.0/(2.0*self._delta*self._delta)
        kappa = self._kappa # alias for convenience 
        k = lambda row, col: row*self._nx+col # map subscripts (row, col) to linear index (k) in row-major order

        # populate interior points
        for i in range(1,self._ny-1):
            for j in range(1,self._nx-1):
                A[k(i,j), k(i  ,j  )] = -c*(4.0*kappa[i,j]+kappa[i-1,j]+kappa[i+1,j]+kappa[i,j-1]+kappa[i,j+1])
                A[k(i,j), k(i-1,j  )] = c*(kappa[i,j]+kappa[i-1,j  ]) 
                A[k(i,j), k(i+1,j  )] = c*(kappa[i,j]+kappa[i+1,j  ])
                A[k(i,j), k(i  ,j-1)] = c*(kappa[i,j]+kappa[i  ,j-1]) 
                A[k(i,j), k(i  ,j+1)] = c*(kappa[i,j]+kappa[i  ,j+1])

        # NOTE: BC treatment handles corners by adding the dqx/dx and dqy/dy
        # terms separately. This makes it possible for *one* BC to be applied
        # at edge points, and *both* BCs to be applied at corner points.

        # populate boundary at y=0
        i = 0
        if self._bc[0] == 'constant':
            pass 
        elif self._bc[0]  == 'closed':
            for j in range(0,self._nx): # dqy/dy term
                A[k(i,j), k(i  ,j)] += -c*(kappa[i,j]+kappa[i+1,j])
                A[k(i,j), k(i+1,j)] +=  c*(kappa[i,j]+kappa[i+1,j])
            for j in range(1,self._nx-1): # dqx/dx term
                A[k(i,j), k(i,j  )] += -c*(2.0*kappa[i,j]+kappa[i,j-1]+kappa[i,j+1])
                A[k(i,j), k(i,j-1)] +=  c*(kappa[i,j]+kappa[i  ,j-1]) 
                A[k(i,j), k(i,j+1)] +=  c*(kappa[i,j]+kappa[i  ,j+1])
        elif self._bc[0] == 'open':
            for j in range(1,self._nx-1): # dqx/dx term only
                A[k(i,j), k(i,j  )] += -c*(2.0*kappa[i,j]+kappa[i,j-1]+kappa[i,j+1])
                A[k(i,j), k(i,j-1)] +=  c*(kappa[i,j]+kappa[i  ,j-1]) 
                A[k(i,j), k(i,j+1)] +=  c*(kappa[i,j]+kappa[i  ,j+1])
        elif self._bc[0] == 'cyclic':
            print("hillslope: cyclic BC not implemented"); sys.exit()
        elif self._bc[0] == 'mirror':
            print("hillslope: mirror BC not implemented"); sys.exit()
        else:
            print("hillslope: invalid boundary condition at y=0"); sys.exit()

        # populate boundary at y=end
        i = self._ny-1
        if self._bc[1] == 'constant':
            pass 
        elif self._bc[1]  == 'closed':
            for j in range(0,self._nx): # dqy/dy term
                A[k(i,j), k(i  ,j  )] += -c*(kappa[i,j]+kappa[i-1,j])
                A[k(i,j), k(i-1,j  )] +=  c*(kappa[i,j]+kappa[i-1,j]) 
            for j in range(1,self._nx-1): # dqx/dx term
                A[k(i,j), k(i,j  )] += -c*(2.0*kappa[i,j]+kappa[i,j-1]+kappa[i,j+1])
                A[k(i,j), k(i,j-1)] +=  c*(kappa[i,j]+kappa[i,j-1]) 
                A[k(i,j), k(i,j+1)] +=  c*(kappa[i,j]+kappa[i,j+1])
        elif self._bc[1] == 'open':
            for j in range(1,self._nx-1): # dqx/dx term only
                A[k(i,j), k(i,j  )] += -c*(2.0*kappa[i,j]+kappa[i,j-1]+kappa[i,j+1])
                A[k(i,j), k(i,j-1)] +=  c*(kappa[i,j]+kappa[i,j-1]) 
                A[k(i,j), k(i,j+1)] +=  c*(kappa[i,j]+kappa[i,j+1])
        elif self._bc[1] == 'cyclic':
            print("hillslope: cyclic BC not implemented"); sys.exit()
        elif self._bc[1]  == 'mirror':
            print("hillslope: mirror BC not implemented"); sys.exit()
        else:
            print("hillslope: invalid boundary condition at y=end"); sys.exit()

        # populate boundary at x=0
        j = 0
        if self._bc[2] == 'constant':
            pass 
        elif self._bc[2]  == 'closed':
            for i in range(0,self._ny): # dqx/dx term
                A[k(i,j), k(i,j  )] += -c*(kappa[i,j]+kappa[i,j+1])
                A[k(i,j), k(i,j+1)] +=  c*(kappa[i,j]+kappa[i,j+1])
            for i in range(1,self._ny-1): # dqy/dy term
                A[k(i,j), k(i  ,j)] += -c*(2.0*kappa[i,j]+kappa[i-1,j]+kappa[i+1,j])
                A[k(i,j), k(i-1,j)] +=  c*(kappa[i,j]+kappa[i-1,j]) 
                A[k(i,j), k(i+1,j)] +=  c*(kappa[i,j]+kappa[i+1,j])
        elif self._bc[2] == 'open':
            for i in range(1,self._ny-1): # dqy/dy term only
                A[k(i,j), k(i  ,j)] += -c*(2.0*kappa[i,j]+kappa[i-1,j]+kappa[i+1,j])
                A[k(i,j), k(i-1,j)] +=  c*(kappa[i,j]+kappa[i-1,j]) 
                A[k(i,j), k(i+1,j)] +=  c*(kappa[i,j]+kappa[i+1,j])
        elif self._bc[2] == 'cyclic':
            print("hillslope: cyclic BC not implemented"); sys.exit()
        elif self._bc[2]  == 'mirror':
            print("hillslope: mirror BC not implemented"); sys.exit()
        else:
            print("hillslope: invalid boundary condition at x=0"); sys.exit()

        # populate boundary at x=end
        j = self._nx-1
        if self._bc[3] == 'constant':
            pass 
        elif self._bc[3]  == 'closed':
            for i in range(0,self._ny): # dqx/dx term
                A[k(i,j), k(i,j  )] += -c*(kappa[i,j]+kappa[i,j-1])
                A[k(i,j), k(i,j-1)] +=  c*(kappa[i,j]+kappa[i,j-1]) 
            for i in range(1,self._ny-1): # dqy/dy term
                A[k(i,j), k(i  ,j)] += -c*(2.0*kappa[i,j]+kappa[i-1,j]+kappa[i+1,j])
                A[k(i,j), k(i-1,j)] +=  c*(kappa[i,j]+kappa[i-1,j]) 
                A[k(i,j), k(i+1,j)] +=  c*(kappa[i,j]+kappa[i+1,j])
        elif self._bc[3] == 'open':
            for i in range(1,self._ny-1): # dqy/dy term only
                A[k(i,j), k(i  ,j)] += -c*(2.0*kappa[i,j]+kappa[i-1,j]+kappa[i+1,j])
                A[k(i,j), k(i-1,j)] +=  c*(kappa[i,j]+kappa[i-1,j]) 
                A[k(i,j), k(i+1,j)] +=  c*(kappa[i,j]+kappa[i+1,j])
        elif self._bc[3] == 'cyclic':
            print("hillslope: cyclic BC not implemented"); sys.exit()
        elif self._bc[3]  == 'mirror':
            print("hillslope: mirror BC not implemented"); sys.exit()
        else:
            print("hillslope: invalid boundary condition at x=end"); sys.exit()

        # store results in effcient format for matrix*vector product
        self._A = A.tocsr()

    def run(self, run_time):
        """
        Run numerical integration for specified time period

        Arguments:
            run_time = Scalar double, model run time, [a]
        """
        
        run_time = np.double(run_time)
        time = np.double(0.0)    
        max_step = 0.95*self._delta*self._delta/(4.0*np.max(self._kappa)) # stable time step, note ref (1) has error

        while time < run_time:
            step = min(run_time-time, max_step)
            self._height += step*self._A.dot(self._height)
            time += step

if __name__ == '__main__':
    
    # basic usage example and "smell test": relaxation to height==0 steady state
    # # initialize model
    nx = 100
    ny = 100
    max_time = 50.0
    time_step = 0.5
    h0 = np.random.rand(ny, nx).astype(np.double)
    h0[:,0] = np.double(0.0) 
    h0[:,-1] = np.double(0.5)
    h0[0,:] = np.double(0.5)
    h0[-1,:] = np.double(0.5)
    dd = np.double(1.0)
    kk = np.ones((ny, nx), dtype=np.double)
    bcs = ['open', 'constant', 'constant', 'constant']
    model = ftcs(h0, dd, kk, bcs)
    # # update and plot model
    plt.imshow(model.get_height(), interpolation='nearest', clim=(-0.5,0.5))
    plt.colorbar()
    plt.ion()
    time = 0.0
    while time < max_time: 
        model.run(time_step)
        time += time_step
        plt.cla()
        plt.imshow(model.get_height(), interpolation='nearest', clim=(-0.5,0.5))
        plt.title("TIME = {:.2f}".format(time))
        plt.pause(0.05)
