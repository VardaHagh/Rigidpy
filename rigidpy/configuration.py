from __future__ import division, print_function, absolute_import

import numpy as np
from .framework import Framework
import scipy.optimize as opt

class Configuration(object):
    '''
    takes in a strcuture, returns optimized structure
    '''
    def __init__(self, coordinates, bonds, basis, k=1, dim=2):
        self.dim = dim
        self.x0 = coordinates.ravel()
        self.bonds = bonds
        self.basis = basis
        self.k = k
        self.initialenergy = 0
        self.finalenergy = 0
        self.report = None
        self.framework = None
        self.lengths = None

    def Energy(self, P, L, restlengths):
        '''
        find energy of spring network

        Paramters
        ---------
        L: rest length
        k : spring constant
        restlengths: equilengths


        Returns
        -------
        Energy of the network

        '''
        # The argument P is a vector (flattened matrix).We convert it to a matrix here.
        coordinates = P.reshape((-1, self.dim))
        PF = Framework(coordinates, self.bonds, basis=self.basis, k=self.k, restlengths=restlengths)
        self.framework = PF
        lengths = PF.EdgeLengths() # length of all bonds
        self.lengths = lengths
        energy = 0.5 * np.sum(np.dot(PF.K,(lengths - L)**2))
        return energy

    def Forces(self, P, L, restlengths):
        coordinates = P.reshape((-1, self.dim))
        Ns,Nb = len(coordinates),len(self.bonds)
        PF = self.framework
        lengths = self.lengths # length of all bonds
        deltaL = (lengths-L)/lengths
        vals = np.multiply(deltaL.reshape(Nb,-1),PF.dr)
        vals = np.dot(PF.K,vals)
        Force = np.zeros((Ns,Ns,self.dim),float)
        row,col = self.bonds.T
        Force[row,col] = vals
        Force[col,row] = -vals
        return Force.sum(axis=1).reshape(-1,)

    def Hessian(self, P, L, restlengths):
        coordinates = P.reshape((-1, self.dim))
        PF = self.framework
        H = PF.HessianMatrix()
        return H

    def energy_minimize_Newton(self, L, restlengths):
        E = np.array(self.bonds,int)
        self.initialenergy =self.Energy(self.x0, L, restlengths)
        report = opt.minimize(fun=self.Energy, x0=self.x0, args = (L, restlengths),
                              method='Newton-CG', jac = self.Forces, hess=self.Hessian,
                              options={'disp': False, 'xtol': 1e-7,'return_all': False, 'maxiter': None})
        self.report = report
        self.finalenergy = report.fun
        P1 = report.x.reshape((-1, self.dim))
        return P1

    """def energy_minimize_BFGS(self, coordinates, bonds, a1, a2, L, k=1):
        P = np.array()
        E = np.array(self.bonds,int)
        self.initialenergy =self.energy(P.ravel(), E, a1, a2, L, k)
        report = opt.minimize(self.energy, P.ravel(), args = (E, a1, a2, L, k), method='L-BFGS-B',
                          options={'disp': None, 'maxls': 20, 'iprint': -1,
                                   'gtol': 1e-10, 'eps': 1e-10, 'maxiter': 50000,
                                   'ftol': 1e-10,'maxcor': 30,
                                   'maxfun': 50000})
        self.report = report
        self.finalenergy = report.fun
        P1 = report.x.reshape((-1, self.dim))
        return P1"""
