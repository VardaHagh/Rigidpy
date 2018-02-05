from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist,pdist
import networkx as nx
from .periodic_framework import Periodic_Framework
import scipy.optimize as opt

class Periodic_Configuration(object):
    '''
    takes in a strcuture, returns optimized structure
    '''
    def __init__(self, dim=2):
        self.dim = dim
        self.initialenergy = 0
        self.finalenergy = 0
        self.report = None

    def energy(self, coordinates, edges, a1, a2, L, k=1):
        '''
        find energy of spring network

        paramters
        ---------
        L: rest length
        k : spring constant
        '''
        # The argument P is a vector (flattened matrix).
        # We convert it to a matrix here.
        P = np.array(coordinates)
        E = np.array(edges,int)
        P = P.reshape((-1, self.dim))
        # length of all edges
        PF = Periodic_Framework(P,E,a1,a2)
        lengths = PF.EdgeLengths()

        return (0.5 * (k * (lengths - L)**2).sum())

    def forces(self, coordinates, edges, a1, a2, L, k=1):

        cutoff = np.amin(np.array([np.sqrt(np.dot(a1,a1)),np.sqrt(np.dot(a2,a2))]))/2.
        E = np.array(edges,int)
        P = np.array(coordinates)
        P = P.reshape((-1, self.dim))
        Ns,Nb = len(P),len(E)
        l0 = L
        relaxed_F = Periodic_Framework(P, E, a1, a2)
        l = relaxed_F.EdgeLengths()
        lenref = relaxed_F.lenref()
        Rij = -np.diff(P[E],axis=1).reshape(Nb,-1)
        lengths = np.linalg.norm(Rij,axis=1)
        long_bonds = E[lengths > cutoff]
        positions = np.where(lengths > cutoff)[0]
        index = [np.argmin(cdist(item[0]+[[0.,0.]],item[1]+lenref,'euclidean')) for item in P[long_bonds]]
        new_r = [item[0]+[[0.,0.]] - (item[1]+lenref[index][i]) for i,item in enumerate(P[long_bonds])]
        new_r = np.array(new_r).reshape(-1,self.dim)
        Rij[positions] = new_r
        deltal = (l-l0)/l
        val = np.multiply(deltal.reshape(Nb,-1),Rij)

        Force = np.zeros((Ns,Ns,self.dim),float)
        row,col = E.T

        Force[row,col] = k*val
        Force[col,row] = -k*val

        return Force.sum(axis=1).reshape(-1,)


    def hessian(self, coordinates, edges, a1, a2, L, k=1):
        E = np.array(edges,int)
        P = np.array(coordinates)
        P = P.reshape((-1, self.dim))
        F = Periodic_Framework(P, E, a1, a2)
        H = F.HessianMatrix()
        return H

    def energy_minimize_BFGS(self, coordinates, edges, a1, a2, L, k=1):
        P = np.array(coordinates)
        E = np.array(edges,int)
        self.initialenergy =self.energy(P.ravel(), E, a1, a2, L, k)
        P1 = opt.minimize(self.energy, P.ravel(), args = (E, a1, a2, L, k), method='L-BFGS-B',
                          options={'disp': None, 'maxls': 20, 'iprint': -1,
                                   'gtol': 1e-10, 'eps': 1e-10, 'maxiter': 50000,
                                   'ftol': 1e-10,'maxcor': 30,
                                   'maxfun': 50000})
        self.report = P1
        self.finalenergy = P1.fun
        P1 = P1.x.reshape((-1, self.dim))
        return P1

    def energy_minimize_Newton(self, coordinates, edges, a1, a2, L, k=1):
        P = np.array(coordinates)
        E = np.array(edges,int)
        self.initialenergy =self.energy(P.ravel(), E, a1, a2, L, k)
        P1 = opt.minimize(self.energy, P.ravel(), args = (E, a1, a2, L, k), method='Newton-CG',
        jac = self.forces,hess=self.hessian ,options={'disp': False, 'xtol': 1e-5,'return_all': False, 'maxiter': None})
        self.report = P1
        self.finalenergy = P1.fun
        P1 = P1.x.reshape((-1, self.dim))
        return P1
