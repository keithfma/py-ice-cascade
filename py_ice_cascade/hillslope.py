"""
Python ICE-CASCADE hillslope erosion-deposition model component

References:

    (1) Becker, T. W., & Kaus, B. J. P. (2016). Numerical Modeling of Earth
    Systems: Lecture Notes for USC GEOL557 (1.1.4)
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.sparse

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

    def __init__(self, height, delta, kappa, bcs):
        """
        Arguments:
            height = 2D Numpy array, surface elevation in model domain, [m]
            delta = Scalar double, grid spacing, assumed square, [m]
            kappa = Scalar double, diffusion coefficient, [m**2/a]
            bcs = List of boundary conditions names for [x=0, x=end, y=0, y=end]
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
        self._valid_bcs = set(['constant'])

        self.set_height(height)
        self.set_diffusivity(kappa)
        self._delta = np.copy(np.double(delta))
        self._set_bcs(bcs)
        self._set_coeff_matrix() # TODO: use BC names to adapt boundary coefficients

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

    def _set_bcs(self, bcs):
        """Check and store BC names"""
        if len(bcs) != 4:
            print("hillslope: incorrect number of boundary conditions supplied"); sys.exit()
        if not set(bcs).issubset(self._valid_bcs):
            print("hillslope: invalid boundary condition name"); sys.exit()
        self._bc_x0 = bcs[0]  
        self._bc_x1 = bcs[1] 
        self._bc_y0 = bcs[2] 
        self._bc_y1 = bcs[3] 

    def _set_coeff_matrix(self):
        """Define sparse coefficient matrix for dHdt stencil"""
        # NOTE: FTCS is a 5-point stencil, since diffusivity is a grid, all
        # coefficients are potentially unique. The approach here is to compute
        # coefficients for each diagonal as a matrix including BCs, then
        # convert to a sparse penta-diagonal. Special care is needed to allow
        # for the general case where some diagonals may wrap around.
        
        # declare coefficient arrays
        i_j   = np.zeros((self._ny, self._nx), dtype = np.double)
        im1_j = np.zeros((self._ny, self._nx), dtype = np.double)
        ip1_j = np.zeros((self._ny, self._nx), dtype = np.double)
        i_jm1 = np.zeros((self._ny, self._nx), dtype = np.double)
        i_jp1 = np.zeros((self._ny, self._nx), dtype = np.double)

        # populate coefficients for interior points
        inv2delta2 = 1.0/(2.0*self._delta*self._delta)
        kappa_i_j   = self._kappa[1:-1,1:-1] # just views, should be efficient
        kappa_ip1_j = self._kappa[2:  ,1:-1]
        kappa_im1_j = self._kappa[ :-2,1:-1]
        kappa_i_jp1 = self._kappa[1:-1,2:  ]
        kappa_i_jm1 = self._kappa[1:-1, :-2]

        i_j[1:-1,1:-1] = -inv2delta2*(4.0*kappa_i_j + kappa_im1_j + kappa_ip1_j + kappa_i_jm1 + kappa_i_jp1)
        im1_j[1:-1,1:-1] = inv2delta2*(kappa_i_j + kappa_im1_j)
        ip1_j[1:-1,1:-1] = inv2delta2*(kappa_i_j + kappa_ip1_j)
        i_jm1[1:-1,1:-1] = inv2delta2*(kappa_i_j + kappa_i_jm1)
        i_jp1[1:-1,1:-1] = inv2delta2*(kappa_i_j + kappa_i_jp1)

        # populate coeff for boundary at y=0
        if self._bc_x0 == 'constant':
            pass

        # populate coeff for boundary at y=end
        if self._bc_x1 == 'constant':
            pass

        # populate coeff for boundary at x=0
        if self._bc_y0 == 'constant':
            pass

        # populate coeff for boundary at x=max
        if self._bc_y1 == 'constant':
            pass

        # convert coeffs to vectors
        i_j = np.ravel(i_j, order='C')
        ip1_j = np.ravel(ip1_j, order='C')
        im1_j = np.ravel(im1_j, order='C')
        i_jp1 = np.ravel(i_jp1, order='C')
        i_jm1 = np.ravel(i_jm1, order='C')

        # NOTE: leaving boundary coefficients as 0 is equivalent to a constant
        # boundary condition. I leave this for now and move on to test the
        # whole pipeline.

        # construct sparse coefficient matrix
        dd = [];                       kk = [] 
        # # center diagonal 
        dd.append(i_j);                kk.append(0)
        # # upper diagonal, last element wraps
        dd.append(i_jp1[:-1]);         kk.append(1) 
        dd.append(i_jp1[-1:]);         kk.append(-self._ny*self._nx+1)  
        # # lower diagonal, first element wraps
        dd.append(i_jm1[1:]);          kk.append(-1) 
        dd.append(i_jm1[:1]);          kk.append(self._ny*self._nx-1)
        # # outer upper diagonal, last nx elements wrap
        dd.append(ip1_j[0:-self._nx]); kk.append(self._nx)
        dd.append(ip1_j[-self._nx:]);  kk.append(-self._ny*self._nx+self._nx)
        # # outer lower diagonal, first nx elements wrap 
        dd.append(im1_j[self._nx:]);   kk.append(-self._nx)
        dd.append(im1_j[:self._nx]);   kk.append(self._ny*self._nx-self._nx)
        # # construct compressed-row matrix from diagonals 
        self._A = scipy.sparse.diags(dd, offsets=kk, format="csr", dtype=np.double)


        # TEST: populate matrix using for loops instead, this may be more clear and maintainable
        A = scipy.sparse.lil_matrix((self._ny*self._nx, self._ny*self._nx), dtype=np.double) # lil format is fast to populate
        c = 1.0/(2.0*self._delta*self._delta)
        kappa = self._kappa # alias for convenience 
        k = lambda row, col: row*self._nx+col # sub2ind

        # interior points
        for i in range(1,self._ny-1):
            for j in range(1,self._nx-1):
                A[k(i,j), k(i  ,j  )] = -c*(4.0*kappa[i,j]+kappa[i-1,j]+kappa[i+1,j]+kappa[i,j-1]+kappa[i,j+1])
                A[k(i,j), k(i-1,j  )] = c*(kappa[i,j]+kappa[i-1,j  ]) 
                A[k(i,j), k(i+1,j  )] = c*(kappa[i,j]+kappa[i+1,j  ])
                A[k(i,j), k(i  ,j-1)] = c*(kappa[i,j]+kappa[i  ,j-1]) 
                A[k(i,j), k(i  ,j+1)] = c*(kappa[i,j]+kappa[i  ,j+1])

        self._A = A.tocsr() # csr format is fast for matrix*vector

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
