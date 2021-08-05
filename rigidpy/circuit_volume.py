from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from .framework import framework
from .configuration import configuration
import time

from typing import Union, Dict


class circuitVolume(object):
    """
    Set up a circuit

    Parameters
    ----------
    coordinates: sequence of d floats
        Point coordinates in d dimension.
    bonds: tuple or array
        Edge list
    basis: array
        List of lattice vectors
    k: int/float or array of int/floats
        Spring constant
    varcell: array of booleans
        A list of lattice vector components
        allowed to change (1/True) or fixed (0/False).

    Returns
    -------
    The class does not return anything.
    """

    def __init__(
        self,
        coordinates: Union[np.array, list],
        bonds: Union[np.array, list],
        basis: Union[np.array, list] = None,
        k: Union[np.array, float] = 1.0,
        varcell=None,
    ):

        self.N, self.dim = coordinates.shape
        self.nbasis = len(basis)  # number of lattice vectors
        self.nchange = np.sum(varcell)  # number of components allowed to change
        self.coordinates = coordinates
        self.bonds = bonds
        self.basis = basis
        self.varcell = varcell
        self.k = k
        self.center = np.mean(coordinates, axis=0)  # center of mass
        PF = framework(coordinates, bonds, basis=basis, k=k, varcell=varcell)
        self.K = PF.K
        self.restLengths = PF.edgeLengths()
        self.results = None
        print("Building a circuit by changing the volume...")

    def point(
        self,
        coordinates: Union[np.array, list],
        bonds: Union[np.array, list],
        basis: Union[np.array, list] = None,
    ):

        d = {
            "coordinates": None,
            "lengths": None,
            "direction": None,
            "eigenvalue": None,
            "eigenvector": None,
        }
        dim = self.dim
        N = self.N
        nchange = self.nchange

        """create a framework, currently works only for 2D"""
        PF = framework(coordinates, bonds, basis=basis, k=self.k, varcell=self.varcell)
        evalues, evectors = PF.eigenspace(eigvals=(0, dim + 1))
        nzeros = np.sum(evalues < 1e-14)  # number of zero eigenvalues
        if nzeros > dim + 1:
            print("Warning: Rigidity rank dropped!")

        # removing translations
        rhatCoordinates = np.tile(np.eye(dim), N)
        rhatCell = np.zeros([dim, nchange], int)
        rhat = np.append(rhatCoordinates, rhatCell, axis=1) / np.sqrt(N + nchange)
        e = evectors[0]
        projecions = np.multiply(e, rhat).sum(axis=1)  # projection along each axis
        enet = e - np.dot(projecions, rhat)
        enet = enet / norm(enet)

        d["coordinates"] = coordinates
        d["lengths"] = PF.EdgeLengths()
        d["eigenvalue"] = evalues[0]
        d["eigenvector"] = evectors[0]
        d["direction"] = enet
        return d

    def nextPoint(self, coordinates, bonds, basis, direction, stepSize, threshold=0.99):

        # eigenvector for points and lattice vectors
        ## and change in lattice vectors
        eCoordinates = direction[: -self.nchange].reshape(-1, self.dim)
        eCell = direction[-self.nchange :]
        latticeU = np.zeros_like(self.varcell, dtype=float)
        latticeU[self.varcell] = eCell
        eCell = latticeU.reshape(-1, self.dim)

        # initial step
        scale = stepSize
        nextP = coordinates + scale * eCoordinates
        nextBasis = basis + scale * eCell
        nextPoint = self.point(nextP, bonds, basis)
        test = np.abs(np.vdot(direction, nextPoint["direction"]))
        """If dot product is less than threshold, decrease
        the step size until it is larger."""
        while test < threshold:
            scale *= 0.5
            if scale < 10 ** (-10):
                break
            nextP = coordinates + scale * eCoordinates
            nextBasis = basis + scale * eCell
            nextPoint = self.point(nextP, bonds, basis)
            test = np.abs(np.vdot(direction, nextPoint["direction"]))
        return nextPoint, nextBasis

    def follow(
        self,
        stepSize: float = 1e-3,
        iteration: int = 10,
        relaxStep: int = 5,
        report: bool = True,
        optimization: bool = False,
        kick: int = 1,
    ) -> Dict:
        """Follow the circuit."""
        d = {
            "coordinates": [],
            "volume": [],
            "distance": [],
            "basis": [],
            "energy": [],
            "nsteps": None,
            "stepsize": stepSize,
            "time": None,
        }

        p = np.copy(self.coordinates)
        bonds = self.bonds
        basis = self.basis
        restLengths = self.restLengths
        center = self.center
        dim = self.dim
        # N = self.N

        timei = time.time()
        for i in range(iteration):

            # find direction for next step
            currentPoint = self.point(p, bonds, basis)
            translationVec = currentPoint["direction"]
            lengths = currentPoint["lengths"]
            eigval = currentPoint["eigenvalue"]

            # make sure we move forward
            if i == 0:
                translationVecOld = kick * translationVec
            projection = np.vdot(translationVec, translationVecOld)
            direction = np.sign(projection) * translationVec
            translationVecOld = direction

            # current volume, distance from center of mass, energy
            volume = np.abs(np.product(np.linalg.eig(basis)[0]))
            dist = np.mean(norm(p - center, axis=1))
            energy = 0.5 * np.dot(self.K, (restLengths - lengths) ** 2)

            # append current data
            d["coordinates"].append(p)
            d["volume"].append(volume)
            d["distance"].append(dist)
            d["basis"].append(basis)
            d["energy"].append(energy)

            # update
            nextP, nextBasis = self.nextPoint(p, bonds, basis, direction, stepSize)
            p = nextP["coordinates"]
            basis = nextBasis

            if optimization and i != 0:
                if i % relaxStep == 0:
                    c = configuration(p, bonds, basis, self.K, dim)
                    p = c.energyMinimizeNewton(restLengths, restLengths)
        d["nsteps"] = i + 1
        timef = time.time()
        totalTime = timef - timei
        d["time"] = totalTime
        if report:
            print("Total time={:<.4f} seconds".format(totalTime))
        self.results = d
        return d

    def detectPassagePoints(self) -> np.ndarray:
        """
        This function determines if the volume of
        of the cell returns to its original
        length.
        """
        lens = self.results["volume"]
        arr = lens - lens[0]
        # find when sign flips
        mask = np.diff(np.sign(arr)) != 0
        return np.nonzero(mask)[0]

    def dotProduct(self, save: bool = False, name: str = None) -> plt.Figure:
        results = self.results
        nsteps = results["nsteps"]
        results
        m, n = results["coordinates"][0].shape
        P = np.array(results["coordinates"]).reshape(-1, m * n)
        T_diff = P[1:] - P[:-1]
        T_diff = T_diff / norm(T_diff, axis=1)[:, np.newaxis]

        delta = np.diag(np.dot(T_diff[1:, :], T_diff[:-1, :].T))
        mag = norm(P, axis=1)

        fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.plot(np.arange(1, nsteps - 1), delta, "-r")
        ax1.set_xlabel("Iteration")
        # Make the y-axis label, ticks and tick labels match the line color.
        ax1.set_ylabel("Dot product", color="r")
        ax1.tick_params("y", colors="r")
        ax1.ticklabel_format(useOffset=False)
        # ax1.set_ylim(0.99,1)

        ax2 = ax1.twinx()
        ax2.plot(np.arange(nsteps), mag, "-ob", markersize=0.1)
        ax2.plot(np.arange(nsteps), np.ones(nsteps) * mag[0], "--k")
        ax2.set_ylabel("Distance from origin", color="b")
        ax2.tick_params("y", colors="b")
        ax2.ticklabel_format(useOffset=False)

        ax3.plot(np.arange(nsteps), results["volume"], "-ok", markersize=0.1)
        ax3.plot(np.arange(nsteps), np.ones(nsteps) * results["volume"][0], "--k")
        ax3.set_ylabel("Cell Volume", color="k")
        ax3.set_xlabel("Iteration")

        ax4 = ax3.twinx()
        dists = norm(P - P[0], axis=1)
        ax4.plot(np.arange(nsteps), dists, "-og", markersize=0.1)
        ax4.set_ylabel("Distance from starting point", color="g")
        ax4.tick_params("y", colors="g")
        ax4.ticklabel_format(useOffset=False)

        fig.tight_layout()
        # now = datetime.datetime.now()
        # variables = map(str,[results['bond'],np.mean(lave),nsteps])
        # vars_strings = "_".join(variables)
        # name = 'trihex_lengths_'+vars_strings+'.pdf'
        # plt.title(name)
        plt.tight_layout()
        if save:
            plt.savefig(name, dpi=100)
            plt.close()
        else:
            plt.show(fig)
