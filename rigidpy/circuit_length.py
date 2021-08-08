from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from .framework import framework
from .configuration import configuration
import time
from typing import Union, Dict


class circuitLength(object):
    """Set up a circuit where degrees of freedom are lengths.

    Args:
        coordinates (Union[np.array, list]): Vertex/site coordinates. For
            ``N`` sites in ``d`` dimensions, the shape is ``(N,d)``.
        bonds (Union[np.array, list]): Edge list. For ``M`` bonds, its
            shape is ``(M,2)``.
        basis (Union[np.array, list], optional): List of basis/repeat/lattice
            vectors, Default: ``None``.If ``None`` or array of zero vectors,
            the system is assumed to be finite. Defaults to None.
        k (Union[np.array, float], optional): Spring constant/stiffness.
            If an array is supplied, the shape should be ``(M,2)``.
            Defaults to 1.0.
        varcell (Union[np.array, list], optional): (d*d,) array of booleans/int
            A list of basis vector components allowed to change (1/True) or
            fixed (0/False). Example: ``[0,1,0,0]`` or ``[False, True, False, False]``
            both mean that in two dimensions, only second element of first
            basis vector is allowed to change.. Defaults to None.
    """

    def __init__(
        self,
        coordinates: Union[np.array, list],
        bonds: Union[np.array, list],
        basis: Union[np.array, list] = None,
        k: Union[np.array, float] = 1.0,
        varcell: Union[np.array, list] = None,
    ):

        self.N, self.dim = coordinates.shape
        self.nbasis = len(basis)  # number of lattice vectors
        if varcell is None:
            self.nchange = 0
        else:
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
        print("Building a circuit by cutting a bond...")

    def point(
        self,
        coordinates: Union[np.array, list],
        bonds: Union[np.array, list],
        basis: Union[np.array, list] = None,
    ) -> Dict:
        """Generate a point in phase space.

        Args:
            coordinates (Union[np.array, list]): Vertex/site coordinates. For
            ``N`` sites in ``d`` dimensions, the shape is ``(N,d)``.
            bonds (Union[np.array, list]): Edge list. For ``M`` bonds, its
                shape is ``(M,2)``.
            basis (Union[np.array, list], optional): List of basis/repeat/lattice
                vectors, Default: ``None``.If ``None`` or array of zero vectors,
                the system is assumed to be finite. Defaults to None.
            k (Union[np.array, float], optional): Spring constant/stiffness.
                If an array is supplied, the shape should be ``(M,2)``.
                Defaults to 1.0.

        Returns:
            Dict: dictionary of infomation
        """

        d = {
            "coordinates": None,
            "lengths": None,
            "direction": None,
            "eigenvalue": None,
            "eigenvector": None,
        }
        dim = self.dim
        N = self.N
        # nchange = self.nchange

        """create a framework"""
        PF = framework(coordinates, bonds, basis=basis, k=self.k, varcell=self.varcell)
        evalues, evectors = PF.eigenspace(eigvals=(0, dim + 2))
        nzeros = np.sum(evalues < 1e-14)  # number of zero eigenvalues
        if nzeros > dim + 1:
            print("Warning: Rigidity rank dropped!")

        # removing translations
        rhatCoordinates = np.tile(np.eye(dim), N)
        rhat = rhatCoordinates / np.sqrt(N)
        e = evectors[0]
        projecions = np.multiply(e, rhat).sum(axis=1)  # projection along each axis
        enet = e - np.dot(projecions, rhat)
        enet = enet / norm(enet)

        d["coordinates"] = coordinates
        d["lengths"] = PF.edgeLengths()
        d["eigenvalue"] = evalues[0]
        d["eigenvector"] = evectors[0]
        d["direction"] = enet
        return d

    def nextPoint(
        self,
        coordinates: Union[np.array, list],
        bonds: Union[np.array, list],
        basis: Union[np.array, list],
        direction: np.ndarray,
        stepsize: float,
        threshold: float = 0.99,
    ) -> np.ndarray:

        """Find the next right point in traversing the phase space"""

        # eigenvector for points and lattice vectors
        ## and change in lattice vectors
        eCoordinates = direction.reshape(-1, self.dim)

        # initial step
        scale = stepsize
        nextP = coordinates + scale * eCoordinates
        nextPoint = self.point(nextP, bonds, basis, self.k)
        test = np.abs(np.vdot(direction, nextPoint["direction"]))
        """If dot product is less than threshold, decrease
        the step size until it is larger."""
        while test < threshold:
            scale *= 0.5
            if scale < 10 ** (-10):
                break
            nextP = coordinates + scale * eCoordinates
            nextPoint = self.point(nextP, bonds, basis, self.k)
            test = np.abs(np.vdot(direction, nextPoint["direction"]))
        return nextPoint, basis

    def follow(
        self,
        bondId: int,
        stepsize: float = 1e-3,
        iteration: int = 10,
        radius: float = 0.1,
        relaxStep: int = 5,
        report: bool = True,
        optimization: bool = False,
        lazySearch: bool = False,
        kick: int = 1,
    ) -> Dict:
        """Follow the circuit."""
        d = {
            "coordinates": [],
            "length": [],
            "distance": [],
            "basis": [],
            "energy": [],
            "nsteps": None,
            "stepsize": stepsize,
            "time": None,
        }

        orignalCoordinates = self.coordinates
        p = np.copy(self.coordinates)
        bonds = self.bonds
        M = len(bonds)
        basis = self.basis
        restLengths = self.restLengths
        center = self.center
        N, dim = self.N, self.dim

        # removing the specified bonds from the bond list
        bondsAfterCut = np.copy(bonds, int)
        mask = np.ones(M, dtype=bool)
        mask[bondId] = False
        bondsAfterCut = bondsAfterCut[mask]

        # remove the spring constant for bondId
        kAfterCut = np.copy(np.diag(self.K), int)
        mask = np.ones(M, dtype=bool)
        mask[bondId] = False
        kAfterCut = kAfterCut[mask]
        KAfterCut = np.diag(kAfterCut)

        # remove the equilibrium length for bondId
        restlengthsAfterCut = np.copy(self.restlengths)
        mask = np.ones(M, dtype=bool)
        mask[bondId] = False
        restlengthsAfterCut = restlengthsAfterCut[mask]

        timei = time.time()
        for i in range(iteration):

            # find direction for next step
            currentPoint = self.point(p, bondsAfterCut, basis, kAfterCut)
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
            dist = np.mean(norm(p - center, axis=1))
            energy = 0.5 * np.dot(KAfterCut, (restlengthsAfterCut - lengths) ** 2)

            # length of cut bonds
            tempF = framework(p, bonds, basis)
            length = tempF.EdgeLengths()[bondId]

            # append current data
            d["coordinates"].append(p)
            d["length"].append(length)
            d["distance"].append(dist)
            d["basis"].append(basis)
            d["energy"].append(energy)

            # update
            nextP, nextBasis = self.nextPoint(
                p, bondsAfterCut, basis, kAfterCut, direction, stepsize
            )
            p = nextP["coordinates"]
            basis = nextBasis

            if optimization and i != 0:
                if i % relaxStep == 0:
                    C = configuration(p, bondsAfterCut, basis, kAfterCut, dim)
                    p = C.energyMinimizeNewton(restLengths, restLengths)

            """if path wants to repeat itself, break the loop.The
            idea is to check when previous step gets closer to initial
            point but next step gets far. This does happen only close
            near starting point. In addition we check if the mid point
            is closer than a cutoff to initial point."""

            if i > 2:
                dist1 = norm(d["coordinates"][-2] - orignalCoordinates)
                dist2 = norm(d["coordinates"][-1] - orignalCoordinates)
                dist3 = norm(p - orignalCoordinates)
                turning = np.vdot(dist2 - dist1, dist3 - dist2)
                if (i != 0) and turning < 0 and dist2 < radius:
                    print(
                        """Tracking stopped: The last point was closer than
                    radius from the starting point!"""
                    )
                    break

            if lazySearch:
                passage = self.DetectPassagePoints(d["length"])
                if len(passage) > 1:
                    print("Tracking stopped: The first conjugate point is found!")
                    break

        if i + 1 == iteration:
            print("Tracking stopped: Maximum number of iterations is used!")
        d["nsteps"] = i + 1
        timef = time.time()
        totalTime = timef - timei
        d["time"] = totalTime
        if report:
            print("Total time={:<.4f} seconds".format(totalTime))
        self.results = d
        return d

    def DetectPassagePoints(
        self, lengths: Union[np.ndarray, list] = None
    ) -> np.ndarray:
        """
        This function determines if the volume of
        of the cell returns to its original
        length.
        """
        if lengths is None:
            lens = self.results["length"]
        else:
            lens = lengths
        arr = lens - lens[0]
        # find when sign flips
        mask = np.diff(np.sign(arr)) != 0
        return np.nonzero(mask)[0]

    def circuitRealization(
        self, save: bool = False, name: str = None, **kwds
    ) -> plt.Figure:
        results = self.results
        datax, datay = results["length"], results["distance"]

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.ticklabel_format(useOffset=False)
        ax.set_ylabel("Average Distance from Center of Mass")
        ax.set_xlabel("Length of the removed edge")
        ax.plot(datax, datay, "-", color="k", zorder=1, **kwds)
        ax.scatter(datax[0], datay[0], marker="*", color="r", zorder=2, s=200, **kwds)
        plt.tight_layout()
        if save:
            plt.savefig(name, dpi=100)
            plt.close()
        else:
            return plt.show(fig)

    def plotRealization(
        self, save: bool = False, name: str = None, **kwds
    ) -> plt.Figure:
        """Draw the configurational path after removal of degrees of freedom.

        Args:
            save (bool, optional): whether to save the plot. Defaults to False.
            name (str, optional): name of the plot to be saved. Defaults to None.

        Returns:
            : [description]
        """
        return self.circuitRealization(save=save, name=name, **kwds)

    def dotProduct(self, save: bool = False, name: bool = None) -> plt.Figure:
        """ """
        results = self.results
        nsteps = results["nsteps"]
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

        ax3.plot(np.arange(nsteps), results["length"], "-ok", markersize=0.1)
        ax3.plot(np.arange(nsteps), np.ones(nsteps) * results["length"][0], "--k")
        ax3.set_ylabel("Length of cut bond", color="k")
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
