from __future__ import division, print_function, absolute_import

import numpy as np
import scipy.linalg as SLA
from scipy.linalg import qr
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh,cg,inv
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from numpy.linalg import norm
from itertools import product

class Framework(object):
    '''Base class for a framework
    '''
    def __init__(self, coordinates, edges, basis, k=1, equlengths=None, varCell=None):

        '''Number of sites and spatial dimensions'''
        self.V, self.dim = coordinates.shape
        self.coordinates = coordinates

        '''Number of edges and edge list'''
        self.E, self.C = edges.shape
        self.edges = edges

        '''Basis vectors and their norms'''
        self.basis = basis
        self.nbasis = nbasis = len(basis) # number of basis vectors
        self.basisNorm = np.array([norm(base) for base in basis])
        self.cutoff = 0.5*np.amin(self.basisNorm)

        '''whether cell can deform'''
        self.varCell = varCell

        '''
        volume of the container box
        '''
        self.volume = np.abs(np.product(np.linalg.eig(basis)[0]))

        # froce constant matrix
        if isinstance(k, (int, float)):
            self.K = np.diagflat(k*np.ones(self.E))
        else:
            self.K = np.diagflat(k)
        self.KS = csr_matrix(self.K) #sparse spring constants

        '''Cartesian product for periodic box'''
        regionIndex = np.array(list(product([-1,0,1], repeat=nbasis)))
        transVectors = np.einsum('ij,jk->ik',regionIndex,basis)

        '''Identify long bonds'''
        if self.C not in [2,3]:
            raise ValueError('Second dimension should be 2 or 3.')
        elif self.C is 2:
            #dr = vector from node i to node j if edge is (i,j)
            dr = -np.diff(coordinates[edges[:,0:2]],axis=1).reshape(-1,self.dim)
            #length of dr
            lengths = norm(dr,axis=1)
            # which bonds are long == cross the boundary
            self.indexLong = indexLong = np.nonzero(lengths > self.cutoff)
            # two ends of long bonds
            longBonds = edges[indexLong]
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

        '''Equilibrium or rest length of springs'''
        if equlengths is None:
            self.L0 = norm(self.dr,axis=1)
        else:
            self.L0 = equlengths

        '''Tension spring stiffness
        by convention: compression has postive tension
        '''
        self.tension = np.dot(self.K, self.L0 - norm(self.dr,axis=1) )
        self.KP = np.diag(self.tension/self.L0)

    def EdgeLengths(self):
        return norm(self.dr,axis=1)

    def RigidityMatrixStable(self):
        '''
        Calculate rigidity matrix of the graph.
        Elements are normalized position difference of connected
        coordinates.
         make sure edge list, first element < second element
        '''
        #self.dr
        N,M=self.V,self.E
        drNorm = self.L0[:,np.newaxis]
        dr = self.dr/drNorm # normalized dr
        # find row and col for non zero values
        row = np.repeat(np.arange(M),2)
        col = self.edges.reshape(-1)
        val = np.column_stack((dr,-dr)).reshape(-1,self.dim)
        R = np.zeros([M,N,self.dim])
        R[row,col]=val
        R = R.reshape(M,-1)

        if self.varCell is not None:
            conditions = np.array(self.varCell,dtype=bool)
            #cellDim = np.sum(conditions!=0)
            #RCell = np.zeros([M,cellDim])
            #print(RCell)
            RCell = np.zeros([M,self.nbasis,self.dim])
            valsCell = np.einsum('ij,ik->ijk',self.mn,dr[self.indexLong])
            RCell[self.indexLong] = valsCell
            RCell = RCell.reshape(M,-1)
            # select only specified components
            RCell = RCell[:, conditions]
            R = np.append(R,RCell,axis=1)

        return R

    def RigidityMatrixAxis(self, i):
        '''
        Calculate rigidity matrix of the graph along an axis
        in d dimensions. Elements are unit vectors.
        '''
        N,M=self.V,self.E
        row = np.repeat(np.arange(M),2)
        col = self.edges.reshape(-1)
        dr = np.repeat([np.eye(self.dim)[i]],M,axis=0)
        val = np.column_stack((dr,-dr)).reshape(-1,self.dim)
        R = np.zeros([M,N,self.dim])
        R[row,col]=val
        R = R.reshape(M,-1)
        return R

    def RigidityMatrix(self):
        '''
        Calculate rigidity matrix of the graph. For now, it simply returns
        stable rigidity matrix.
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
        R = self.RigidityMatrixStable()
        H = np.dot(R.T, np.dot(self.KP, R))
        for i in range(self.dim):
            Raxis = self.RigidityMatrixAxis(i)
            H -= np.dot(Raxis.T, np.dot(self.KP, Raxis))
        return H

    def HessianMatrix(self):
        '''calculate total Hessian = stable + unstable'''
        Hstable = self.HessianMatrixStable()
        Hdestable = self.HessianMatrixDestable()
        Htotal = Hstable + Hdestable
        return Htotal

    def RigidityMatrixSparse(self):
        N,M,d=self.V,self.E,self.dim
        drNorm = self.L0[:,np.newaxis]
        dr = self.dr/drNorm # normalized dr
        row = np.repeat(np.arange(M),2*d)
        index_range = np.arange(0,d).reshape(-1,1)
        additive_idx = np.repeat(index_range,2*M,axis=-1)
        col = (d*self.edges.reshape(-1)+additive_idx).T.reshape(-1)
        val = np.column_stack((dr,-dr)).reshape(-1)
        # form matrix
        R = csr_matrix((val,(row,col)), shape=(M,N*d))
        return R

    def HessianMatrixSparse(self):
        '''
        calculate Hessian or dynamical matrix
        Shape is (dim*V-P) by (dim*V-P).
        '''
        RS = self.RigidityMatrixSparse()
        RST = RS.transpose()
        HS = RST.dot((self.KS).dot(RS))
        return HS

    def Eigenspace(self,eigvals=(0,4)):
        '''
        Returns sorted eigenvalues and eigenvectors of
        total Hessian matrix.
        '''
        H = self.HessianMatrix()
        evalues, evectors = SLA.eigh(H,eigvals=eigvals)
        return evalues, evectors.T

    def EigenspaceSparse(self,eigvals=4,sigma=1e-12,which='LM',mode='normal'):
        '''
        sorted eigenvalues and eigenvectors
        assumes the hessian is symmetric (by design)
        '''
        H = self.HessianMatrixSparse()
        evalues, evectors = eigsh(H,k=eigvals,sigma=sigma,which=which,mode=mode)
        args = np.argsort(np.abs(evalues))
        return evalues[args], evectors.T[args]

    def ForceMatrix(self, L):
        N,M,d=self.V,self.E,self.dim
        l = self.L0
        deltaL = (l-L)/l
        vals = np.multiply(deltaL.reshape(M,-1),self.dr)
        vals = np.dot(self.K,vals)
        # form the matrix
        row,col = self.edges.T
        Force = np.zeros((N,N,d),float)
        Force[row,col] = vals
        Force[col,row] = -vals
        return Force

    def CouplingMatrix(self):
        H = self.HessianMatrix()
        Hinv = np.linalg.pinv(H)
        R = self.RigidityMatrix()
        RT = R.T
        G = np.dot(R,np.dot(Hinv,RT))
        return G

    def CouplingMatrixSparse(self):
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
        '''Return the elastic modulus for the supplied strain matrix'''
        transformedBasis = np.multiply(strainMatrix, self.basis)
        transformedCoordinates = np.multiply(strainMatrix, self.coordinates)

        volume = self.volume
        K = self.K # spring constant matrix
        H = self.HessianMatrix()
        R = self.RigidityMatrix()
        RT = R.T

        F2 = Framework(transformedCoordinates, self.edges, transformedBasis, np.diag(K))
        dl = F2.EdgeLengths() - self.EdgeLengths()
        Force = -np.dot(RT,dl)
        u = np.linalg.lstsq(H,Force,rcond=-1)
        Dl = np.dot(R,u[0])
        energy = 0.5 * np.sum(np.dot(K,(Dl + dl)**2))
        modulus = energy/(2*volume)
        return modulus

    def BulkModulus(self,eps=1e-6):
        '''Returns the bulk modulus.'''
        strainMatrix = np.full((self.nbasis,), 1-eps)
        bulk = self.ElasticModulus(strainMatrix)/(eps*eps)
        return bulk

    def ShearModulus(self,eps=1e-6):
        '''Returns the bulk modulus.'''
        strainMatrix = np.empty((self.nbasis,))
        strainMatrix[::2] = 1-eps
        strainMatrix[1::2] = 1+eps
        shear = self.ElasticModulus(strainMatrix)/(eps*eps)
        return shear
