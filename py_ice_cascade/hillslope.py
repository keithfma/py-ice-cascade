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
import netCDF4

class model():
    """Base class for hillslope model components"""
    def __init__(self):
        pass
    def set_height(self, new):
        raise NotImplementedError
    def get_height(self):
        raise NotImplementedError
    def set_mask(self, new):
        raise NotImplementedError
    def init_netcdf(self, nc, zlib, complevel, shuffle, chunksizes):
        raise NotImplementedError
    def to_netcdf(self, nc, time_idx):
        raise NotImplementedError
    def run(self, run_time):
        raise NotImplementedError

class null(model):
    """
    Do-nothing class to be used for disabled hillslope component

    Internal height grid is set and returned unchanged
    """
    def __init__(self):
        pass
    def set_height(self, new):
        self._height = np.double(new) 
    def get_height(self):
        return self._height
    def set_mask(*args):
        pass
    def init_netcdf(self, nc, *args):
        nc.createVariable('hill_model', np.dtype('i1')) # scalar
        nc['hill_model'][...] = False 
        nc['hill_model'].type = self.__class__.__name__ 
    def to_netcdf(*args):
        pass
    def run(*args):
        pass

class ftcs(model):
    r"""
    Hillslope diffusion model using forward-time center-space (FTCS) finite
    diffence scheme. 

    Arguments:
        height: 2D numpy array, surface elevation in model domain, [m]
        mask: 2D numpy array, True where hillslope diffusion is active, False
            where inactive (e.g. under water or ice)
        delta: Scalar double, grid spacing, assumed square, [m]
        kappa_active: active diffusion coefficient, used where mask is 
            True, [m**2/a]
        kappa_inactive: inactive diffusion coefficient, used where mask is
            False, [m**2/a]
        bc: List of boundary conditions names for [y=0, y=end, x=0, x=end]

    Supported boundary conditions are:
    
    * *constant*: :math:`\frac{\partial H}{\partial t} = 0`
    
    * *closed*: no flux out of boundary (e.g. :math:`(q_x)_{i,j+1/2} = 0` at
      :math:`x_{max}`)

    * *open*: no flux gradient normal to boundary, material passes through
      (e.g. :math:`\frac{\partial q_x}{\partial x} = 0` at :math:`x_{max}`)
    
    * *cyclic*: flux at opposing boundaries is equal (e.g.
      :math:`(q_x)_{i,-1/2} = (q_x)_{i,\text{end}+1/2}`)
    
    * *mirror*: boundary flux is equal and opposite incoming flux (e.g.
      :math:`(q_x)_{i,j+1/2} = -(q_x)_{i,j-1/2}` at :math:`x_{max}`)

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

    The above scheme is modified at boundary points.     
    """


    def __init__(self, height, mask, delta, kappa_active, kappa_inactive, bc):
        # set attributes
        self._valid_bc = set(['constant', 'closed', 'open', 'cyclic', 'mirror'])
        self._delta = np.copy(np.double(delta))
        self._kappa_active = np.copy(np.double(kappa_active))
        self._kappa_inactive = np.copy(np.double(kappa_inactive))
        self._bc = list(bc)
        self.set_height(height)
        self.set_mask(mask)
        self._dhdt = np.zeros((self._ny, self._nx), dtype=np.double)

    def _check_dims(self, array):
        """Check that array dims match model dims, or set model dims"""
        if hasattr(self, '_nx') and array.shape != (self._ny, self._nx):
            raise ValueError("Array dims do not match model dims")
        else:
            self._ny, self._nx = array.shape

    def _check_bcs(self):
        """Check for valid boundary condition selections"""
        if not set(self._bc).issubset(self._valid_bc):
            raise ValueError("Invalid boundary condition name")
        y_num_cyclic = sum([self._bc[ii]=='cyclic' for ii in range(0,2)])
        x_num_cyclic = sum([self._bc[ii]=='cyclic' for ii in range(2,4)])
        if y_num_cyclic==1 or x_num_cyclic==1:
            raise ValueError("Unmatched cyclic boundary condition") 

    def set_height(self, new):
        """Set height grid"""
        new_array = np.copy(np.double(new))
        self._check_dims(new_array)
        self._height = np.ravel(new_array, order='C') # stored as vector 

    def get_height(self):
        """Return height grid as 2D numpy array"""
        return np.copy(self._height.reshape((self._ny, self._nx), order='C'))

    def set_mask(self, new):
        """Set diffusivity grid from mask and update dheight/dt coeff matrix"""
        new_array = np.where(new, True, False)
        self._check_dims(new_array)
        self._mask = new_array
        self._kappa = np.where(self._mask, self._kappa_active, self._kappa_inactive)
        self._set_coeff_matrix()

    def _set_coeff_matrix(self):
        """Define sparse coefficient matrix for dHdt stencil"""

        # NOTE: FTCS is a 5-point stencil, since diffusivity is a grid, all
        # coefficients are potentially unique. 

        # NOTE: corners are handled corners by adding the dqx/dx and dqy/dy
        # terms separately. This makes it possible for *one* BC to be applied
        # at edge points, and *both* BCs to be applied at corner points.

        # init variables
        self._check_bcs()
        A = scipy.sparse.lil_matrix((self._ny*self._nx, self._ny*self._nx), dtype=np.double) # lil format is fast to populate
        c = 1.0/(2.0*self._delta*self._delta)
        kappa = self._kappa # alias for convenience 
        k = lambda row, col: row*self._nx+col # map subscripts (row, col) to linear index (k) in row-major order

        # populate interior points
        for i in range(1,self._ny-1): # add dqy/dy tern
            for j in range(0,self._nx):
                A[k(i,j),k(i  ,j)] += -c*(kappa[i-1,j]+2*kappa[i,j]+kappa[i+1,j])
                A[k(i,j), k(i-1,j)] +=  c*(kappa[i-1,j]+kappa[i,j]) 
                A[k(i,j), k(i+1,j)] +=  c*(kappa[i,j]+kappa[i+1,j])
        for i in range(0,self._ny): # add dqx/dx term
            for j in range(1,self._nx-1):
                A[k(i,j), k(i,j  )] += -c*(kappa[i,j-1]+2*kappa[i,j]+kappa[i,j+1])
                A[k(i,j), k(i,j-1)] +=  c*(kappa[i,j-1]+kappa[i,j]) 
                A[k(i,j), k(i,j+1)] +=  c*(kappa[i,j]+kappa[i,j+1])

        # populate boundary at y=0
        i = 0
        if self._bc[0] == 'constant':
            for j in range(0,self._nx): # all coeff -> 0
                A[k(i,j),:] = 0
        elif self._bc[0]  == 'closed':
            for j in range(0,self._nx): # add qy/dy term
                A[k(i,j), k(i  ,j)] += -c*(kappa[i,j]+kappa[i+1,j])
                A[k(i,j), k(i+1,j)] +=  c*(kappa[i,j]+kappa[i+1,j])
        elif self._bc[0] == 'open':
            pass
        elif self._bc[0] == 'cyclic':
            for j in range(0,self._nx): # add qy/dy term
                A[k(i,j), k(i  ,j)]        += -c*(kappa[self._ny-1,j]+2*kappa[i,j]+kappa[i+1,j])
                A[k(i,j), k(self._ny-1,j)] +=  c*(kappa[self._ny-1,j]+kappa[i,j]) 
                A[k(i,j), k(i+1,j)]        +=  c*(kappa[i,j]+kappa[i+1,j])
        elif self._bc[0] == 'mirror':
            for j in range(0,self._nx): # dqy/dy term
                A[k(i,j), k(i  ,j)] += -2.0*c*(kappa[i,j]+kappa[i+1,j])
                A[k(i,j), k(i+1,j)] +=  2.0*c*(kappa[i,j]+kappa[i+1,j])

        # populate boundary at y=end
        i = self._ny-1
        if self._bc[1] == 'constant':
            for j in range(0,self._nx): # all coeff -> 0
                A[k(i,j),:] = 0
        elif self._bc[1]  == 'closed':
            for j in range(0,self._nx): # dqy/dy term
                A[k(i,j), k(i  ,j  )] += -c*(kappa[i,j]+kappa[i-1,j])
                A[k(i,j), k(i-1,j  )] +=  c*(kappa[i,j]+kappa[i-1,j]) 
        elif self._bc[1] == 'open':
            pass
        elif self._bc[1] == 'cyclic': # dqy/dy term
            for j in range(0,self._nx):
                A[k(i,j), k(i  ,j)] += -c*(kappa[i-1,j]+2*kappa[i,j]+kappa[0,j])
                A[k(i,j), k(i-1,j)] +=  c*(kappa[i-1,j]+kappa[i,j]) 
                A[k(i,j), k(0  ,j)] +=  c*(kappa[i,j]+kappa[0,j])
        elif self._bc[1]  == 'mirror':
            for j in range(0,self._nx): # dqy/dy term
                A[k(i,j), k(i  ,j  )] += -2.0*c*(kappa[i,j]+kappa[i-1,j])
                A[k(i,j), k(i-1,j  )] +=  2.0*c*(kappa[i,j]+kappa[i-1,j]) 

        # populate boundary at x=0
        j = 0
        if self._bc[2] == 'constant':
            for i in range(0,self._ny): # all coeff -> 0
                A[k(i,j),:] = 0
        elif self._bc[2]  == 'closed':
            for i in range(0,self._ny): # dqx/dx term
                A[k(i,j), k(i,j  )] += -c*(kappa[i,j]+kappa[i,j+1])
                A[k(i,j), k(i,j+1)] +=  c*(kappa[i,j]+kappa[i,j+1])
        elif self._bc[2] == 'open':
            pass
        elif self._bc[2] == 'cyclic':
            for i in range(0,self._ny): # dqx/dx term
                A[k(i,j), k(i,j  )]        += -c*(kappa[i,self._nx-1]+2*kappa[i,j]+kappa[i,j+1])
                A[k(i,j), k(i,self._nx-1)] +=  c*(kappa[i,j]+kappa[i,self._nx-1])
                A[k(i,j), k(i,j+1)]        +=  c*(kappa[i,j]+kappa[i,j+1])
        elif self._bc[2]  == 'mirror':
            for i in range(0,self._ny): # dqx/dx term
                A[k(i,j), k(i,j  )] += -2.0*c*(kappa[i,j]+kappa[i,j+1])
                A[k(i,j), k(i,j+1)] +=  2.0*c*(kappa[i,j]+kappa[i,j+1])

        # populate boundary at x=end
        j = self._nx-1
        if self._bc[3] == 'constant':
            for i in range(0,self._ny): # all coeff -> 0
                A[k(i,j),:] = 0
        elif self._bc[3]  == 'closed':
            for i in range(0,self._ny): # dqx/dx term
                A[k(i,j), k(i,j  )] += -c*(kappa[i,j]+kappa[i,j-1])
                A[k(i,j), k(i,j-1)] +=  c*(kappa[i,j]+kappa[i,j-1]) 
        elif self._bc[3] == 'open':
            pass
        elif self._bc[3] == 'cyclic':
            for i in range(0,self._ny): # dqx/dx term
                A[k(i,j), k(i,j  )] += -c*(kappa[i,j-1]+2*kappa[i,j]+kappa[i,0])
                A[k(i,j), k(i,j-1)] +=  c*(kappa[i,j-1]+kappa[i,j]) 
                A[k(i,j), k(i,0)] +=  c*(kappa[i,j]+kappa[i,0])
        elif self._bc[3]  == 'mirror':
            for i in range(0,self._ny): # dqx/dx term
                A[k(i,j), k(i,j  )] += -2.0*c*(kappa[i,j]+kappa[i,j-1])
                A[k(i,j), k(i,j-1)] +=  2.0*c*(kappa[i,j]+kappa[i,j-1]) 

        # store results in effcient format for matrix*vector product
        self._A = A.tocsr()

    def run(self, run_time):
        """
        Run numerical integration for specified time period

        Arguments:
            run_time: Scalar double, model run time, [a]
        """
        
        run_time = np.double(run_time)
        time = np.double(0.0)    
        max_step = 0.95*self._delta*self._delta/(4.0*np.max(self._kappa)) # stable time step, note ref (1) has error
        h0 = np.copy(self._height)

        while time < run_time:
            step = min(run_time-time, max_step)
            self._height += step*self._A.dot(self._height)
            time += step

        self._dhdt = (self._height-h0)/run_time

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

        nc.createVariable('hill_model', np.dtype('i1')) # scalar
        nc['hill_model'][...] = True
        nc['hill_model'].type = self.__class__.__name__ 
        nc['hill_model'].kappa_active = self._kappa_active 
        nc['hill_model'].kappa_inactive = self._kappa_inactive 
        nc['hill_model'].bc_yi = self._bc[0]
        nc['hill_model'].bc_yf = self._bc[1]
        nc['hill_model'].bc_xi = self._bc[2]
        nc['hill_model'].bc_xf = self._bc[3]

        nc.createVariable('hill_kappa', np.double, dimensions=('time', 'y', 'x'), 
            zlib=zlib, complevel=complevel, shuffle=shuffle, chunksizes=chunksizes)
        nc['hill_kappa'].long_name = 'hillslope diffusivity'
        nc['hill_kappa'].units = 'm^2 / a'

        nc.createVariable('hill_dzdt', np.double, dimensions=('time', 'y', 'x'), 
            zlib=zlib, complevel=complevel, shuffle=shuffle, chunksizes=chunksizes)
        nc['hill_dzdt'].long_name = 'average hillslope bedrock elevation rate of change'
        nc['hill_dzdt'].units = 'm / a'

    def to_netcdf(self, nc, time_idx):
        """
        Write model-specific state variables to output file

        Arguments:
            nc: netCDF4 Dataset object, output file open for writing 
            time_idx: integer time index to write to
        """

        nc['hill_kappa'][time_idx,:,:] = self._kappa
        nc['hill_dzdt'][time_idx,:,:] = self._dhdt
