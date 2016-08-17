"""
Python ICE-CASCADE hillslope erosion-deposition model component

References:

    (1) Becker, T. W., & Kaus, B. J. P. (2016). Numerical Modeling of Earth
    Systems: Lecture Notes for USC GEOL557 (1.1.4)
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import sys

class ftcs():
    """
    Hillslope diffusion model using forward-time center-space (FTCS) finite
    diffence scheme. 
    
    For an overview of FTCS see reference (1). 

    Supported boundary conditions are:
    
        constant: height is constant in time at the given value 
        
        open: incoming and outgoing flux normal to the boundary is equal. In other
        words, material is allowed to pass through the boundary node. This
        condition means dq/dx = 0, and the boundary-normal component of the
        diffusion equation goes to 0 as well. Note that boundary-parallel flux
        gradients are not necessarily 0, and so boundary heights may not be
        constant. 
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

        self._valid_bcs = set(['constant', 'closed', 'open', 'cyclic', 'mirror'])

        # populate boundary at y=0
        name = self._bc[0] 
        i = 0
        for j in range(0,self._nx):
            if name == 'constant':
                pass # dhdt = 0 
            elif name  == 'closed':
                print("hillslope: BC not implemented"); sys.exit()
            elif name == 'open':
                print("hillslope: BC not implemented"); sys.exit()
            elif name == 'cyclic':
                print("hillslope: BC not implemented"); sys.exit()
            elif name  == 'mirror':
                print("hillslope: BC not implemented"); sys.exit()
            else:
                print("hillslope: invalid boundary condition at y=0"); sys.exit()

        # populate boundary at y=end
        name = self._bc[1] 
        i = self._ny-1
        for j in range(0,self._nx):
            if name == 'constant':
                pass # dhdt = 0 
            elif name  == 'closed':
                print("hillslope: BC not implemented"); sys.exit()
            elif name == 'open':
                print("hillslope: BC not implemented"); sys.exit()
            elif name == 'cyclic':
                print("hillslope: BC not implemented"); sys.exit()
            elif name  == 'mirror':
                print("hillslope: BC not implemented"); sys.exit()
            else:
                print("hillslope: invalid boundary condition at y=end"); sys.exit()

        # populate boundary at x=0
        name = self._bc[2] 
        j = 0
        for i in range(0,self._ny):
            if name == 'constant':
                pass # dhdt = 0 
            elif name  == 'closed':
                print("hillslope: BC not implemented"); sys.exit()
            elif name == 'open':
                print("hillslope: BC not implemented"); sys.exit()
            elif name == 'cyclic':
                print("hillslope: BC not implemented"); sys.exit()
            elif name  == 'mirror':
                print("hillslope: BC not implemented"); sys.exit()
            else:
                print("hillslope: invalid boundary condition at x=0"); sys.exit()

        # populate boundary at x=end
        name = self._bc[3] 
        j = self._nx-1
        for i in range(0,self._ny):
            if name == 'constant':
                pass # dhdt = 0 
            elif name  == 'closed':
                print("hillslope: BC not implemented"); sys.exit()
            elif name == 'open':
                print("hillslope: BC not implemented"); sys.exit()
            elif name == 'cyclic':
                print("hillslope: BC not implemented"); sys.exit()
            elif name  == 'mirror':
                print("hillslope: BC not implemented"); sys.exit()
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
    max_time = 5.0
    time_step = 0.05
    h0 = np.random.rand(ny, nx).astype(np.double)-0.5
    h0[:,0] = np.double(0.0) 
    h0[:,-1] = np.double(0.0)
    h0[0,:] = np.double(0.0)
    h0[-1,:] = np.double(0.0)
    dd = np.double(1.0)
    kk = np.ones((ny, nx), dtype=np.double)
    bcs = ['constant', 'constant', 'constant', 'constant']
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
