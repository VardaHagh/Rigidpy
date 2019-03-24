from __future__ import division, print_function, absolute_import

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh, cg, inv
from scipy.spatial.distance import cdist
from numpy.linalg import norm, eig, qr, eigh
from itertools import product

class Framework(object):
    '''
    Base class for a framework
    A framework at a minimum needs an array of coordinates for vertices/sites and
    a list of edges/bonds where each element represents an bond. Additional
    information such as boundary conditions, additional constriants
    such pinning specific vertices, spring constants, etc. are optional.
    For the complete list of the variables, see the following.

    Parameters
    ----------
    coordinates: (N, d) array
        Vertex/site coordinates. For ``N`` sites in ``d`` dimensions, the shape is ``(N,d)``.
    bonds: (M, 2) tuple or array
        Edge list. For ``M`` bonds, its shape is ``(M,2)``.
    basis: (d,d) array, optional
        List of basis/repeat/lattice vectors, Default: ``None``.
        If ``None`` or array of zero vectors, the system is assumed to be finite.
    pins: array/list of int, optional
        List of sites to be immobilized.
    k: int/float or array of int/floats, optional
        Spring constant/stiffness. Default: `1`.
        If an array is supplied, the shape should be ``(M,2)``.
    equlengths: int/float or array of int/floats, optional
        Equilibrium/rest length of bonds, used for systems with pre-stress.
    mass: int/float or array of int/floats, optional
        Mass of sites. Default: `1`.
    varcell: (d*d,) array of booleans/int
        A list of basis vector components allowed to change (1/True) or
        fixed (0/False). Example: ``[0,1,0,0]`` or ``[False, True, False, False]``
        both mean that in two dimensions, only second element of first
        basis vector is allowed to change.

    Returns
    -------
    out : framework object
        A framework object used to study vibrational and rigidity properties.

    See Also
    --------
    Configuration

    Notes
    -----


    Examples
    --------
    >>> import numpy as np
    >>> import rigidpy as rp
    >>> coordinates = np.array([[-1,0], [0,0], [1,0], [0,-1]])
    >>> bonds = np.array([[0,1],[1,2],[1,3]])
    >>> basis = [[0,0],[0,0]]
    >>> pins = [0,2,3]
    >>> F = rp.Framework(coordinates, bonds, basis=basis, pins=pins)
    >>> F.RigidityMatrix()
    >>> np.loadtxt(c)
    array([ [ 1., -0.],
            [-1.,  0.],
            [ 0.,  1.]])
    '''

    def __init__(self, coordinates, bonds, basis=None, pins=None, k=1, equlengths=None, mass=1, varcell=None):

        # Number of sites and spatial dimensions
        self.coordinates = np.array(coordinates)
        self.N, self.dim = self.coordinates.shape

        # Number of bonds and bond list
        self.bonds = np.array(bonds)
        self.E, self.C = self.bonds.shape

        # Basis vectors and their norms
        if basis is None:
            basis = np.zeros(shape=(self.dim,self.dim))
            self.boundary = 'free'
        else:
            self.basis = basis
            self.boundary = 'periodic'
        self.nbasis = nbasis = len(basis) # number of basis vectors
        self.basisNorm = np.array([norm(base) for base in basis])
        self.cutoff = 0.5*np.amin(self.basisNorm)

        if pins is None:
            self.pins = []
        else:
            self.pins = pins
            # update the boundary conditions
            if self.boundary=='periodic':
                self.boundary='periodic+pinned'
            else:
                self.boundary='anchored'

        # index of pinned and non-pinned sites in rigidity matrix
        dim_idx = np.arange(0,self.dim)
        if len(self.pins)!=0:
            self.pins_idx = [pin*self.dim+dim_idx for pin in self.pins]
            self.keep_idx = np.setdiff1d(np.arange(0,self.dim*self.N), self.pins_idx)

        # whether cell can deform
        self.varcell = varcell

        # volume of the box/cell
        if nbasis == 1:
            self.volume = self.basisNorm
        else:
            self.volume = np.abs(np.product(eig(basis)[0]))

        # froce constant matrix
        if isinstance(k, (int, float)):
            self.K = np.diagflat(k*np.ones(self.E))
        else:
            self.K = np.diagflat(k)
        self.KS = csr_matrix(self.K) #sparse spring constants

        # Cartesian product for periodic box
        regionIndex = np.array(list(product([-1,0,1], repeat=nbasis)))
        transVectors = np.einsum('ij,jk->ik',regionIndex,basis)

        # Identify long bonds
        if self.C not in [2,3]:
            raise ValueError('Second dimension should be 2 or 3.')
        elif self.C is 2:
            # vector from node i to node j if bond is (i,j)
            dr = -np.diff(coordinates[bonds[:,0:2]],axis=1).reshape(-1,self.dim)
            # length of dr
            lengths = norm(dr,axis=1)
            # which bonds are long == cross the boundary
            self.indexLong = indexLong = np.nonzero(lengths > self.cutoff)[0]
            # two ends of long bonds
            longBonds = bonds[indexLong]
            # index of neiboring boxes for long bonds only
            index = [np.argmin(norm(item[0]-item[1]-transVectors,axis=1)) for item in coordinates[longBonds]]
            dr[indexLong] -= transVectors[index]
            # negihbor is in which neighboring box
            self.mn = regionIndex[index]
            # correct vector from particle 1 to particle 2
            self.dr = dr
        else:
            pass
            # which bonds are long == cross the boundary
            #indexLong = np.nonzero(lengths > self.cutoff)
            # feature for future release

        # Equilibrium or rest length of springs
        if equlengths is None:
            self.L0 = norm(self.dr,axis=1)
        else:
            self.L0 = equlengths

        # Tension spring stiffness
        # by convention: compression has postive tension
        self.tension = np.dot(self.K, self.L0 - norm(self.dr,axis=1) )
        self.KP = np.diag(self.tension/self.L0)

    def EdgeLengths(self):
        """Compute the length of all bonds."""
        return norm(self.dr,axis=1)

    def RigidityMatrixStable(self):
        """Calculate rigidity matrix of the graph.
        Elements are normalized position difference of connected
        coordinates."""
        N,M,d=self.N,self.E,self.dim
        drNorm = self.L0[:,np.newaxis]
        dr = self.dr/drNorm # normalized dr
        # find row and col for non zero values
        row = np.repeat(np.arange(M),2)
        col = self.bonds.reshape(-1)
        val = np.column_stack((dr,-dr)).reshape(-1,self.dim)
        R = np.zeros([M,N,self.dim])
        R[row,col]=val
        R = R.reshape(M,-1)

        if self.varcell is not None:
            conditions = np.array(self.varcell, dtype=bool)
            #cellDim = np.sum(conditions!=0)
            #RCell = np.zeros([M,cellDim])
            #print(RCell)
            RCell = np.zeros([M,self.nbasis,self.dim])
            valsCell = np.einsum('ij,ik->ijk',self.mn,dr[self.indexLong])
            RCell[self.indexLong] = valsCell
            RCell = RCell.reshape(M,-1)
            # select only specified components
            RCell = RCell[:, conditions]
            R = np.append(R, RCell,axis=1)

        if len(self.pins)!=0:
            return R[:, self.keep_idx]

        return R

    def RigidityMatrixAxis(self, i):
        '''
        Calculate rigidity matrix of the graph along an axis
        in d dimensions. Elements are unit vectors.
        '''
        N,M,d=self.N,self.E,self.dim
        row = np.repeat(np.arange(M),2)
        col = self.bonds.reshape(-1)
        dr = np.repeat([np.eye(self.dim)[i]],M,axis=0)
        val = np.column_stack((dr,-dr)).reshape(-1,self.dim)
        R = np.zeros([M,N,self.dim])
        R[row,col]=val
        R = R.reshape(M,-1)

        if self.varcell is not None:
            conditions = np.array(self.varcell, dtype=bool)
            RCell = np.zeros([M,self.nbasis,self.dim])
            valsCell = np.einsum('ij,ik->ijk',self.mn,dr[self.indexLong])
            RCell[self.indexLong] = valsCell
            RCell = RCell.reshape(M,-1)
            # select only specified components
            RCell = RCell[:, conditions]
            R = np.append(R, RCell,axis=1)

        if len(self.pins)!=0:
            return R[:, self.keep_idx]

        return R

    def RigidityMatrix(self):
        '''
        Calculate rigidity matrix of the graph. For now, it simply returns
        stable rigidity matrix.

        .. todolist::
            Update the function to include the second-order term, if required.
        '''
        R = self.RigidityMatrixStable()
        return R

    def HessianMatrixStable(self):
        '''calculate stable Hessian.'''
        R = self.RigidityMatrixStable()
        H = np.dot(R.T, np.dot(self.K, R))
        return H

    def HessianMatrixDestable(self):
        '''calculate destable Hessian.'''
        KP = self.KP
        R = self.RigidityMatrixStable()
        H = np.dot(R.T, np.dot(KP, R))
        for i in range(self.dim):
            Raxis = self.RigidityMatrixAxis(i)
            H -= np.dot(Raxis.T, np.dot(KP, Raxis))
        return H

    def HessianMatrix(self):
        '''calculate total Hessian = stable + unstable'''
        Hstable = self.HessianMatrixStable()
        Hdestable = self.HessianMatrixDestable()
        Htotal = Hstable + Hdestable
        return Htotal

    def __RigidityMatrixSparse(self):
        N,M,d=self.N,self.E,self.dim
        drNorm = self.L0[:,np.newaxis]
        dr = self.dr/drNorm # normalized dr
        row = np.repeat(np.arange(M),2*d)
        index_range = np.arange(0,d).reshape(-1,1)
        additive_idx = np.repeat(index_range,2*M,axis=-1)
        col = (d*self.bonds.reshape(-1)+additive_idx).T.reshape(-1)
        val = np.column_stack((dr,-dr)).reshape(-1)
        # form matrix
        R = csr_matrix((val,(row,col)), shape=(M,N*d))
        return R

    def __HessianMatrixSparse(self):
        '''
        calculate Hessian or dynamical matrix
        Shape is (dim*V-P) by (dim*V-P).
        '''
        RS = self.RigidityMatrixSparse()
        RST = RS.transpose()
        HS = RST.dot((self.KS).dot(RS))
        return HS

    def Eigenspace(self,eigvals=(0,4)):
        """Returns sorted eigenvalues and eigenvectors of
        total Hessian matrix.

        Parameters
        ----------
        eigvals: tuple
            Index of eigenvalues to be returned.
            Default: `(0,4)` equivalent to first 5 eigenvalues.

        Returns
        -------
        eigval, eigvecs:
            Eigenvalues and eigenvectors.
        """

        H = self.HessianMatrix()
        evalues, evectors = eigh(H,eigvals=eigvals)

        """if self.pins:
            evectors = np.insert(arr=evectors, obj=np.ravel(self.pins_idx),
                                values=0, axis=0)"""

        return evalues, evectors.T

    def __EigenspaceSparse(self,eigvals=4,sigma=1e-12,which='LM',mode='normal'):
        """
        sorted eigenvalues and eigenvectors
        assumes the hessian is symmetric (by design)
        """
        H = self.HessianMatrixSparse()
        evalues, evectors = eigsh(H,k=eigvals,sigma=sigma,which=which,mode=mode)
        args = np.argsort(np.abs(evalues))
        return evalues[args], evectors.T[args]

    def ForceMatrix(self, L):
        """
        A force matrix showing the tension in each bond
        """
        N,M,d=self.N,self.E,self.dim
        l = self.L0
        deltaL = (l-L)/l
        vals = np.multiply(deltaL.reshape(M,-1),self.dr)
        vals = np.dot(self.K,vals)
        # form the matrix
        row,col = self.bonds.T
        Force = np.zeros((N,N,d),float)
        Force[row,col] = vals
        Force[col,row] = -vals
        return Force

    def ForceAlongBond(self, bondId, forceScale = 1e-2):
        """
        Generate a force along the bond with index equal to `BondId` with
        the norm equal to `forceScale`.
        """
        N,d = self.N, self.dim
        Fext = np.zeros([N,d])
        e0, e1 = self.bonds[bondId]
        forceVector = forceScale*self.dr[bondId]
        Fext[e0], Fext[e1] = forceVector, -forceVector
        return Fext.reshape(-1,)

    def CouplingMatrix(self):
        H = self.HessianMatrix()
        Hinv = np.linalg.pinv(H)
        R = self.RigidityMatrix()
        RT = R.T
        G = np.dot(R,np.dot(Hinv,RT))
        return G

    def __CouplingMatrixSparse(self):
        H = self.HessianMatrixSparse()
        Hinv = inv(H)
        R = self.RigidityMatrixSparse()
        RT = R.transpose()
        G = R.dot(Hinv.dot(RT))
        return G

    def SelfStress(self):
        '''Return states of self-stress (SSS).'''
        RT = self.RigidityMatrix().T
        u, s, vh = np.linalg.svd(RT)
        nullity = self.E - np.sum(s>1e-10)
        SSS = vh[-nullity:].T
        return SSS

    def ElasticModulus(self, strainMatrix):
        """Return the elastic modulus for the supplied strain matrix

        In ``d`` dimensions, the strain matrix is a ``(d,d)`` array. To compute
        bulk and shear moduli, use ``bulkModulus`` and ``shearModulus`` functions.

        """
        transformedBasis = np.einsum('ik,lk->li', strainMatrix, self.basis)
        transformedCoordinates = np.einsum('ik,lk->li', strainMatrix, self.coordinates)

        volume = self.volume
        K = self.K # spring constant matrix
        H = self.HessianMatrix()
        R = self.RigidityMatrix()
        RT = R.T

        F2 = Framework(transformedCoordinates, self.bonds, basis=transformedBasis, k=np.diag(K))
        dl = F2.EdgeLengths() - self.EdgeLengths()
        Force = -np.dot(RT,K.dot(dl))
        u = np.linalg.lstsq(H,Force,rcond=-1)
        Dl = np.dot(R,u[0])
        energy = 0.5 * np.sum(np.dot(K,(Dl + dl)**2))
        modulus = energy/volume
        return modulus

    def BulkModulus(self,eps=1e-6):
        """Returns the bulk modulus."""
        strainArray = np.full((self.nbasis,), 1-eps)
        strainMatrix = np.diag(strainArray)
        bulk = self.ElasticModulus(strainMatrix)/(2*eps*eps)
        return bulk

    def ShearModulus(self,eps=1e-6):
        """Returns the shear modulus."""
        strainArray = np.empty((self.nbasis,))
        strainArray[::2] = 1-eps
        strainArray[1::2] = 1+eps
        strainMatrix = np.diag(strainArray)
        shear = self.ElasticModulus(strainMatrix)/(2*eps*eps)
        return shear
