import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs

def inner_prod(gparams, wf1, wf2):
    inn_prod = np.trapz(np.multiply(wf1.conj(),wf2), x=gparams.x)
            
    return inn_prod

def build_1DSE_hamiltonian(consts, gparams):
    '''
    
    Build a single electron Hamilonian for the 1-dimensional potential 
    specified in the gparams class. The laplacian operator is approximated by
    using a 1D 3-point stencil. The Hamilonian assumes a natural ordering 
    format along the main diagonal.

    Parameters
    ----------
    consts : Constants class
        Contains constants value for material system.
    
    gparams : GridParameters class
        Contains grid and potential information

    Returns
    -------
    ham_1D : sparse 2D array
        1-dimensional Hamtilonian. The diagonal elements are in natural
        ordering format

    '''
    
    # Build potential energy hamiltonian term
    PE_1D = sparse.diags(gparams.potential)
    
    # Build the kinetic energy hamiltonian term
    
    # Construct dummy block matrix B
    KE_1D = sparse.eye(gparams.nx)*(-2/(gparams.dx**2))
    # Add the +/-1 off diagonal entries for the 1/dx^2 elements
    KE_1D = KE_1D + sparse.diags(np.ones(gparams.nx-1)/(gparams.dx**2),-1)
    KE_1D = KE_1D + sparse.diags(np.ones(gparams.nx-1)/(gparams.dx**2),1)
    
    # Multiply by unit coefficients
    if consts.units == 'Ry':
        KE_1D = -KE_1D
    else:
        KE_1D = -consts.hbar**2/(2*consts.me)*KE_1D    
        
    # Assemble the full Hamiltonian with potential and kinetic terms
    ham_1D = PE_1D + KE_1D
    
    return ham_1D

def solve_schrodinger_eq(consts, gparams, n_sols=1):
    '''
    
    Solve the time-independent Schrodinger-Equation H|Y> = E|Y> where H is
    the single-electron 1-dimensional Hamiltonian.

    Parameters
    ----------
    consts : Constants class
        Contains constants value for material system.
    
    gparams : GridParameters class
        Contains grid and potential information.
        
    n_sols: int
        Number of eigenvectors and eigenenergies to return (default is 1).

    Returns
    -------
    eig_ens : complex 1D array
        Lowest eigenenergies sorted in ascending order.

    eig_vecs : complex 2D array
        Corresponding eigenvectors in natural order format. eig_vecs[:,i] is 
        the eigenvector for eigenvalue eig_ens[i].
        

    '''
    
    # Determine if a 1D or 2D grid and build the respective Hamiltonian
    hamiltonian = build_1DSE_hamiltonian(consts, gparams)
        
    # Solve the schrodinger equation (eigenvalue problem)
    eig_ens, eig_vecs = eigs(hamiltonian.tocsc(), k=n_sols, M=None,
                                           sigma=gparams.potential.min())
    
    # Sort the eigenvalues in ascending order (if not already)
    idx = eig_ens.argsort()   
    eig_ens = eig_ens[idx]
    eig_vecs = eig_vecs[:,idx]
    
    for idx in range(n_sols):
        curr_wf = eig_vecs[:,idx]
        norm_val = inner_prod(gparams, curr_wf, curr_wf)
        eig_vecs[:,idx] = curr_wf/np.sqrt(norm_val)
    
    return eig_ens, eig_vecs



class Constants:
    def __init__(self):
        # Default units are SI [International system]
        self.units = "SI"
        # Mathematical constants
        self.pi = 3.141592653589793         # pi
        
        # Physical constants
        self.h = 6.62607015E-34              # Planck's constant [J*s]
        self.hbar = self.h/(2*self.pi)      # Reduced Planck's constant [J*s]
        self.e = 1.602176634*1E-19          # electron charge [C]
        self.m0 = 9.10938356E-31            # free electron mass [kg]
        self.c = 2.99792458E8               # speed of light [m/s]
        self.muB = 9.274009994E-24          # Bohr magneton [J/T]
        self.eps0 = 8.85418782E-12          # Vacuum permitivity [F/m]
        self.epsR = 7.8                 # Dielectric constant
        self.eps = self.eps0*self.epsR  # Permitivity [F/m]
        self.me = self.m0*0.191         # Effective mass [kg]
  
class GridParameters:
    '''
    
    Initialize the grid parameter class. Handles all things related to the 
    grid settings for the simulations.
        
    '''
    
    def __init__(self, x, y=None, potential=np.array([])):
        '''
        
        Parameters
        ----------
        x : array
            Grid coordinates along x with uniform spacing.
        y : array
            Grid coordinates along y with uniform spacing.
        potential : array
            Potential values along x-y coordinates where 2DEG is formed. Must
            be in meshgrid format (y,x).

        Returns
        -------
        None.

        '''
        self.potential = np.array(potential)
        
        self.x = np.array(x)
        self.dx = x[1] - x[0]
        self.nx = len(x)  
        self.grid_type = '1D'
        
        # Check that coordinate data matches potential but ignore if the 
        # potential is not defined
        if potential.shape[0] != 0:
            if self.nx != self.potential.shape[0]:
                raise ValueError("x coordinate grid points do not match"\
                                " number of potential x-coordinates.")
