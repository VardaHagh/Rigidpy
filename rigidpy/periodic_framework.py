from __future__ import division, print_function, absolute_import

import numpy as np
import networkx as nx
import scipy.linalg as SLA
from scipy.linalg import qr
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.spatial import distance

class Periodic_Framework(object):
    '''
    Base class for a framework

    A framework is defined as a graph in addition
    to a configuration:
    .. math::
       framwork = graph + configuration
    Graph :math:`G = (V, E)`  is a set :math:`V` of
    vertices and a set of :math:`E` edges such
    that :math:`|V|=N` and :math:`|E|=C`.

    To create a framework, it is
    necessary to provide connectivity and position of
    coordinates.
    '''
    def __init__(self, coordinates, edges,  x, y, dim=2):
        '''
        Create a graph
        '''
        self.edgelist = [tuple(e) for e in edges]
        self.graph = nx.Graph(self.edgelist)
        self.E = self.graph.number_of_edges()
        self.V = self.graph.number_of_nodes()
        self.dim = dim
        self.x = x
        self.y = y
        '''
        Have access to coordinates and edges
        as numpy arrays
        '''
        self.edges = np.array(edges,int)
        self.coordinates = np.array(coordinates,float)

    def adjacency_matrix(self):
        '''
        Calculate the adjaceny matrix of the graph. 1 when two coordinates
        are connected, 0 when not connected.
        '''
        A = nx.adjacency_matrix(self.graph)
        return A

    def lenref(self):
        a1 = self.x
        a2 = self.y
        index = np.array([[-1,-1],[0,-1],[1,-1],[-1,0],[0,0],[1,0],[-1,1],[0,1],[1,1]],dtype=np.longdouble)
        transvec = np.outer(index[:,0],a1)+ np.outer(index[:,1],a2)
        return transvec

    def EdgeLengths(self):

        lenref = self.lenref()
        coordinates=self.coordinates
        points=self.edges
        a1 = self.x
        a2 = self.y

        # Fist find the cutoff length (half of the smaller lattice vector)
        cutoff = np.amin(np.array([np.sqrt(np.dot(a1,a1)),np.sqrt(np.dot(a2,a2))]))/2.
        # Lengths of all the bonds
        d = coordinates[points]
        lengths = np.linalg.norm(d[:,0]-d[:,1],axis=-1)
        long_bonds = points[lengths > cutoff]
        pos = np.where(lengths > cutoff)[0]

        new_lengths = [np.amin(distance.cdist(item[0]+[[0.,0.]],item[1]+lenref, 'euclidean')) for item in coordinates[long_bonds]]
        lengths[pos] = new_lengths

        return lengths

    def RigidityMatrix(self):
        '''
        Calculate rigidity matrix of the graph.
        Elements are position difference of connected
        coordinates. Pinned points are excluded as they are
        fixed.
        The shape is E by (dim*V-P).
         make sure edge list, first element < second element
        '''
        N,M=self.V,self.E
        coordinates=self.coordinates
        points=self.edges
        a1 = self.x
        a2 = self.y
        lenref = self.lenref()
        cutoff = np.amin(np.array([np.sqrt(np.dot(a1,a1)),np.sqrt(np.dot(a2,a2))]))/2.

        dr = -np.diff(coordinates[points],axis=1).reshape(M,-1)
        lengths = np.linalg.norm(dr,axis=1)
        long_bonds = points[lengths > cutoff]
        positions = np.where(lengths > cutoff)[0]
        index = [np.argmin(distance.cdist(item[0]+[[0.,0.]],item[1]+lenref, 'euclidean')) for item in coordinates[long_bonds]]
        new_dr = [item[0]+[[0.,0.]] - (item[1]+lenref[index][i]) for i,item in enumerate(coordinates[long_bonds])]
        new_dr = np.array(new_dr).reshape(-1,2)
        dr[positions] = new_dr
        dr = dr/np.linalg.norm(dr,axis=1)[:,np.newaxis]
        # find row and col for non zero values
        row = np.repeat(np.arange(M),2)
        col = points.reshape(-1)
        val = np.column_stack((-dr,dr)).reshape(-1,2)
        # form matrix
        R = np.zeros([M,N,self.dim])
        R[row,col]=val
        R = R.reshape(M,-1) #flatten matrix
        return R

    def HessianMatrix(self):
        '''
        calculate Hessian or dynamical matrix
        Shape is (dim*V-P) by (dim*V-P).
        '''
        R = self.RigidityMatrix()
        D = np.dot(R.T,R)
        return D

    def ForceMatrix(self):
        '''
        force matrix
        '''
        R = self.RigidityMatrix()
        F = np.dot(R,R.T) # force
        return F

    """def VibrationalSpectrum(self):
        '''
        sorted eigenvalues and eigenvectors
        assumes the hessian is symmetric (by design)
        '''
        D = self.HessianMatrix()
        evalues, evectors = np.linalg.eigh(D)
        return evalues, evectors"""

    def Eigenspace(self,eigvals=(0,5)):
        '''
        sorted eigenvalues and eigenvectors
        assumes the hessian is symmetric (by design)
        '''
        D = self.HessianMatrix()
        evalues, evectors = SLA.eigh(D,eigvals=eigvals)
        return evalues, evectors.T
