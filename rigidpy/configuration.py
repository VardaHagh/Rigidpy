from __future__ import division, print_function, absolute_import

import numpy as np
from .framework import Framework
import scipy.optimize as opt

from typing import Union


class Configuration(object):
    """Optimized a configuration.

    Args:
        coordinates (Union[np.array, list]): [description]
        bonds (Union[np.array, list]): [description]
        basis (Union[np.array, list], optional): [description]. Defaults to None.
        k (Union[np.array, float], optional): [description]. Defaults to 1.0.
        dim (int, optional): [description]. Defaults to 2.
    """

    def __init__(
        self,
        coordinates: Union[np.array, list],
        bonds: Union[np.array, list],
        basis: Union[np.array, list] = None,
        k: Union[np.array, float] = 1.0,
        dim: int = 2,
    ):
        self.dim = dim
        self.Ns, self.Nb = len(coordinates), len(bonds)
        self.coordinates = coordinates
        self.x0 = coordinates.ravel()
        self.bonds = bonds
        self.basis = basis
        self.k = k
        self.initialenergy = 0
        self.finalenergy = 0
        self.report = None
        self.framework = None
        self.lengths = None

    def Energy(
        self,
        P: np.ndarray,
        L: Union[np.array, list, float],
        restlengths: np.array = None,
    ) -> float:
        """Find energy of spring network.

        Args:
            P (np.ndarray): positions of sites
            L (Union[np.array, list, float]): Target lengths
            restlengths (Union[np.array, list, float], optional): Natural length of
                bonds/springs. Defaults to None.

        Returns:
            float: energy
        """
        # The argument P is a vector (flattened matrix). We convert it to a matrix here.
        coordinates = P.reshape((-1, self.dim))
        PF = Framework(
            coordinates, self.bonds, basis=self.basis, k=self.k, restlengths=restlengths
        )
        self.framework = PF
        lengths = PF.EdgeLengths()  # length of all bonds
        self.lengths = lengths
        energy = 0.5 * np.sum(np.dot(PF.K, (lengths - L) ** 2))
        return energy

    def Forces(
        self,
        L: np.ndarray,
    ) -> np.ndarray:
        """Compute force on sites.

        Args:
            L (np.array, optional): Target lengths. Defaults to None.

        Returns:
            np.ndarray: force
        """
        # coordinates = P.reshape((-1, self.dim))
        PF = self.framework
        lengths = self.lengths  # length of all bonds
        deltaL = (lengths - L) / lengths
        vals = np.multiply(deltaL.reshape(self.Nb, -1), PF.dr)
        vals = np.dot(PF.K, vals)
        Force = np.zeros((self.Ns, self.Ns, self.dim), float)
        row, col = self.bonds.T
        Force[row, col] = vals
        Force[col, row] = -vals
        return Force.sum(axis=1).reshape(
            -1,
        )

    def Hessian(self) -> np.ndarray:
        """Computes Hessian of framework.

        Returns:
            np.ndarray: Hessian matrix.
        """
        PF = self.framework
        H = PF.HessianMatrix()
        return H

    def energy_minimize_Newton(
        self, L: np.ndarray, restlengths: np.ndarray
    ) -> np.ndarray:
        """Minimizes energy uing Newton method.

        Args:
            L (np.ndarray): Target lengths
            restlengths (np.ndarray): natural lengths

        Returns:
            np.ndarray: optimized site positions
        """
        # E = np.array(self.bonds, int)
        self.initialenergy = self.Energy(self.x0, L, restlengths)
        report = opt.minimize(
            fun=self.Energy,
            x0=self.x0,
            args=(L, restlengths),
            method="Newton-CG",
            jac=self.Forces,
            hess=self.Hessian,
            options={"disp": False, "xtol": 1e-7, "return_all": False, "maxiter": None},
        )

        self.report = report
        self.finalenergy = report.fun
        P1 = report.x.reshape((-1, self.dim))
        return P1

    def energy_minimize_LBFGSB(
        self,
        L: np.ndarray,
        restlengths: np.ndarray,
        pins: Union[np.array, list] = None,
        maxIteration: int = 1000,
    ) -> np.ndarray:
        """Minimizes energy uing LBFGSB method.

        Args:
            L (np.ndarray): target lengths
            restlengths (np.ndarray): natural lengths
            pins (Union[np.array, list], optional): immobile sites. Defaults to None.
            maxIteration (int, optional): maximum optimization iterations. Defaults to 1000.

        Returns:
            np.ndarray: optimized site positions
        """
        # E = np.array(self.bonds, int)
        self.initialenergy = self.Energy(self.x0, L, restlengths)
        # ensure pinned nodes don't move
        bounds = [(None, None)] * (self.dim * self.Ns)
        P_repeat = np.repeat(self.coordinates, self.dim).reshape(-1, self.dim)
        if pins:
            for pin in pins:
                for i in range(self.dim):
                    val = P_repeat[pin * self.dim + i].tolist()
                    bounds[pin * self.dim + i] = val

        report = opt.minimize(
            fun=self.Energy,
            x0=self.x0,
            args=(L, restlengths),
            method="L-BFGS-B",
            bounds=bounds,
            options={"disp": False, "maxiter": maxIteration},
        )
        self.report = report
        self.finalenergy = report.fun
        P1 = report.x.reshape((-1, self.dim))
        return P1
