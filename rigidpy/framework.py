from __future__ import division, print_function, absolute_import

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh, inv
from numpy.linalg import norm, eig
from scipy.linalg import eigh
from itertools import product

from typing import Union


class framework(object):
    """Base class for a framework
    A framework at a minimum needs an array of coordinates for vertices/sites and
    a list of edges/bonds where each element represents an bond. Additional
    information such as boundary conditions, additional constriants
    such pinning specific vertices, spring constants, etc. are optional.
    For the complete list of the variables, see the following.

        Args:
            coordinates (Union[np.array, list]): Vertex/site coordinates. For
                ``N`` sites in ``d`` dimensions, the shape is ``(N,d)``.
            bonds (Union[np.array, list]): Edge list. For ``M`` bonds, its
                shape is ``(M,2)``.
            basis (Union[np.array, list], optional): List of basis/repeat/lattice
                vectors, Default: ``None``.If ``None`` or array of zero vectors,
                the system is assumed to be finite. Defaults to None.
            pins (Union[np.array, list], optional): array/list of int,
                List of sites to be immobilized. Defaults to None.
            k (Union[np.array, float], optional): Spring constant/stiffness.
                If an array is supplied, the shape should be ``(M,2)``.
                Defaults to 1.0.
            restLengths (Union[np.array, list, float], optional): Equilibrium
                or rest length of bonds, used for systems with pre-stress.
                Defaults to None.
            varcell (Union[np.array, list], optional): (d*d,) array of booleans/int
                A list of basis vector components allowed to change (1/True) or
                fixed (0/False). Example: ``[0,1,0,0]`` or ``[False, True, False, False]``
                both mean that in two dimensions, only second element of first
                basis vector is allowed to change.. Defaults to None.
            power (int, optional): Power of potential energy.
                power=2 is Hookean, power=5/2 is Hertzian. For
                non-Hookean potentials, make sure to supply restLengths non-equal to the
                current length of the bonds, otherwise the calculations will be wrong.
                Defaults to 2.

        Raises:
            ValueError: The bond list should have two columns corresponding to
            two ends of bonds.

        Examples:
            ```python
            >>> import numpy as np
            >>> import rigidpy as rp
            >>> coordinates = np.array([[-1,0], [1,0], [0,1]])
            >>> bonds = np.array([[0,1],[1,2],[0,2]])
            >>> basis = [[0,0],[0,0]]
            >>> pins = [0]
            >>> F = rp.Framework(coordinates, bonds, basis=basis, pins=pins)
            >>> print ("rigidity matrix:\n",F.RigidityMatrix())
            >>> eigvals, eigvecs = F.eigenspace(eigvals=(0,3))
            >>> print("vibrational eigenvalues:\n",eigvals)
    """

    def __init__(
        self,
        coordinates: Union[np.array, list],
        bonds: Union[np.array, list],
        basis: Union[np.array, list] = None,
        pins: Union[np.array, list] = None,
        k: Union[np.array, float] = 1.0,
        restLengths: Union[np.array, list, float] = None,
        # mass=1,
        varcell: Union[np.array, list] = None,
        power: int = 2,
    ):

        # Number of sites and spatial dimensions
        self.coordinates = np.array(coordinates)
        self.N, self.dim = self.coordinates.shape

        # Number of bonds and bond list
        self.bonds = np.array(bonds)
        self.NB, self.C = self.bonds.shape

        # Basis vectors and their norms
        if basis is None:
            basis = np.zeros(shape=(self.dim, self.dim))
            self.boundary = "free"
        else:
            self.basis = basis
            self.boundary = "periodic"
        self.nbasis = nbasis = len(basis)  # number of basis vectors
        self.basisNorm = np.array([norm(base) for base in basis])
        self.cutoff = 0.5 * np.amin(self.basisNorm)

        if pins is None:
            self.pins = []
        else:
            self.pins = pins
            # update the boundary conditions
            if self.boundary == "periodic":
                self.boundary = "periodic+pinned"
            else:
                self.boundary = "anchored"

        # index of pinned and non-pinned sites in rigidity matrix
        dim_idx = np.arange(0, self.dim)
        if len(self.pins) != 0:
            self.pins_idx = [pin * self.dim + dim_idx for pin in self.pins]
            self.keepIdx = np.setdiff1d(np.arange(0, self.dim * self.N), self.pins_idx)

        # whether cell can deform
        self.varcell = varcell

        # volume of the box/cell
        if nbasis == 1:
            # in case system is 1D
            self.volume = self.basisNorm
        else:
            volume = np.abs(np.product(eig(basis)[0]))
            if volume:
                self.volume = volume
            else:
                self.volume = 1

        # froce constant matrix
        if isinstance(k, (int, float)):
            self.K = np.diagflat(k * np.ones(self.NB))
        else:
            self.K = np.diagflat(k)
        self.KS = csr_matrix(self.K)  # sparse spring constants

        # Cartesian product for periodic box
        regionIndex = np.array(list(product([-1, 0, 1], repeat=nbasis)))
        transVectors = np.einsum("ij,jk->ik", regionIndex, basis)

        # Identify long bonds
        if self.C not in [2, 3]:
            raise ValueError("Second dimension should be 2 or 3.")
        elif self.C == 2:
            # vector from node i to node j if bond is (i,j)
            dr = -np.diff(coordinates[bonds[:, 0:2]], axis=1).reshape(-1, self.dim)
            # length of dr
            lengths = norm(dr, axis=1)
            # which bonds are long == cross the boundary
            self.indexLong = indexLong = np.nonzero(lengths > self.cutoff)[0]
            # two ends of long bonds
            longBonds = bonds[indexLong]
            # index of neiboring boxes for long bonds only
            index = [
                np.argmin(norm(item[0] - item[1] - transVectors, axis=1))
                for item in coordinates[longBonds]
            ]
            dr[indexLong] -= transVectors[index]
            # negihbor is in which neighboring box
            self.mn = regionIndex[index]
            # correct vector from particle 1 to particle 2
            self.dr = dr
        else:
            pass
            # which bonds are long == cross the boundary
            # indexLong = np.nonzero(lengths > self.cutoff)
            # feature for future release

        # Equilibrium or rest length of springs
        if restLengths is None:
            self.L0 = norm(self.dr, axis=1)
        else:
            self.L0 = restLengths

        # Tension spring stiffness
        # by convention: compression has postive tension
        seperation_norm = self.L0 - norm(self.dr, axis=1)
        self.tension = np.dot(self.K, seperation_norm ** (power - 1))
        self.KP = np.diag(self.tension / norm(self.dr, axis=1))

        ### effective stiffness to use in non-Hookean cases
        if power == 2:
            self.Ke = self.K
        else:
            self.Ke = np.diag(np.dot(self.K, seperation_norm ** (power - 2)))

    def edgeLengths(self) -> np.ndarray:
        """Compute the length of all bonds.

        Returns:
            np.array: bond lengths]
        """
        return norm(self.dr, axis=1)

    def rigidityMatrixGeometric(self) -> np.ndarray:
        """Calculate rigidity matrix of the graph.
        Elements are normalized position difference of connected
        coordinates.

        Returns:
            np.ndarray: rigidity matrix.
        """
        N, M = self.N, self.NB
        L = norm(self.dr, axis=1)
        drNorm = L[:, np.newaxis]
        dr = self.dr / drNorm  # normalized dr
        # find row and col for non zero values
        row = np.repeat(np.arange(M), 2)
        col = self.bonds.reshape(-1)
        val = np.column_stack((dr, -dr)).reshape(-1, self.dim)
        R = np.zeros([M, N, self.dim])
        R[row, col] = val
        R = R.reshape(M, -1)

        if self.varcell is not None:
            conditions = np.array(self.varcell, dtype=bool)
            # cellDim = np.sum(conditions!=0)
            # RCell = np.zeros([M,cellDim])
            # print(RCell)
            rCell = np.zeros([M, self.nbasis, self.dim])
            valsCell = np.einsum("ij,ik->ijk", self.mn, dr[self.indexLong])
            rCell[self.indexLong] = valsCell
            rCell = rCell.reshape(M, -1)
            # select only specified components
            rCell = rCell[:, conditions]
            R = np.append(R, rCell, axis=1)

        if len(self.pins) != 0:
            return R[:, self.keepIdx]

        return R

    def rigidityMatrixAxis(self, i: int) -> np.ndarray:
        """Calculate rigidity matrix of the graph along an axis
        in d dimensions. Elements are unit vectors.

        Args:
            i (int): index of dimensions

        Returns:
            np.ndarray: rigidity matrix
        """
        N, M = self.N, self.NB
        row = np.repeat(np.arange(M), 2)
        col = self.bonds.reshape(-1)
        dr = np.repeat([np.eye(self.dim)[i]], M, axis=0)
        val = np.column_stack((dr, -dr)).reshape(-1, self.dim)
        R = np.zeros([M, N, self.dim])
        R[row, col] = val
        R = R.reshape(M, -1)

        if self.varcell is not None:
            conditions = np.array(self.varcell, dtype=bool)
            rCell = np.zeros([M, self.nbasis, self.dim])
            valsCell = np.einsum("ij,ik->ijk", self.mn, dr[self.indexLong])
            rCell[self.indexLong] = valsCell
            rCell = rCell.reshape(M, -1)
            # select only specified components
            rCell = rCell[:, conditions]
            R = np.append(R, rCell, axis=1)

        if len(self.pins) != 0:
            return R[:, self.keepIdx]

        return R

    def rigidityMatrix(self) -> np.ndarray:
        """Calculate rigidity matrix of the graph. For now, it simply returns
        geometric rigidity matrix.

        Returns:
            np.ndarray: rigidity matrix

        Todo:
            Update the function to include the second-order term, if required.
        """
        R = self.rigidityMatrixGeometric()
        return R

    def hessianMatrixGeometric(self) -> np.ndarray:
        """calculate geometric Hessian."""
        R = self.rigidityMatrixGeometric()
        H = np.dot(R.T, np.dot(self.Ke, R))
        return H

    def hessianMatrixPrestress(self) -> np.ndarray:
        """calculate prestress Hessian."""
        KP = self.KP
        R = self.rigidityMatrixGeometric()
        H = np.dot(R.T, np.dot(KP, R))
        for i in range(self.dim):
            rAxis = self.rigidityMatrixAxis(i)
            H -= np.dot(rAxis.T, np.dot(KP, rAxis))
        return H

    def hessianMatrix(self) -> np.ndarray:
        """calculate total Hessian = geometric + prestress"""
        hGeometric = self.hessianMatrixGeometric()
        hPrestress = self.hessianMatrixPrestress()
        hTotal = hGeometric + hPrestress
        return hTotal

    def __rigidityMatrixSparse(self) -> np.ndarray:
        N, M, d = self.N, self.NB, self.dim
        L = norm(self.dr, axis=1)
        drNorm = L[:, np.newaxis]
        dr = self.dr / drNorm  # normalized dr
        row = np.repeat(np.arange(M), 2 * d)
        indexRange = np.arange(0, d).reshape(-1, 1)
        additiveIdx = np.repeat(indexRange, 2 * M, axis=-1)
        col = (d * self.bonds.reshape(-1) + additiveIdx).T.reshape(-1)
        val = np.column_stack((dr, -dr)).reshape(-1)
        # form matrix
        R = csr_matrix((val, (row, col)), shape=(M, N * d))
        return R

    def __hessianMatrixSparse(self) -> np.ndarray:
        """
        calculate Hessian or dynamical matrix
        Shape is (dim*V-P) by (dim*V-P).
        """
        RS = self.rigidityMatrixSparse()
        RST = RS.transpose()
        HS = RST.dot((self.KS).dot(RS))
        return HS

    def eigenSpace(self, eigvals=(0, 4)) -> np.ndarray:
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

        H = self.hessianMatrix()
        evalues, evectors = eigh(H, eigvals=eigvals)

        """if self.pins:
            evectors = np.insert(arr=evectors, obj=np.ravel(self.pins_idx),
                                values=0, axis=0)"""

        return evalues, evectors.T

    def __eigenSpaceSparse(
        self, eigvals=4, sigma=1e-12, which="LM", mode="normal"
    ) -> np.ndarray:
        """
        sorted eigenvalues and eigenvectors
        assumes the hessian is symmetric (by design)
        """
        H = self.hessianMatrixSparse()
        evalues, evectors = eigsh(H, k=eigvals, sigma=sigma, which=which, mode=mode)
        args = np.argsort(np.abs(evalues))
        return evalues[args], evectors.T[args]

    def forceMatrix(self, L: Union[np.ndarray, list]) -> np.ndarray:
        """
        A force matrix showing the tension in each bond
        """
        N, M, d = self.N, self.NB, self.dim
        l = self.L0
        deltaL = (l - L) / l
        vals = np.multiply(deltaL.reshape(M, -1), self.dr)
        vals = np.dot(self.K, vals)
        # form the matrix
        row, col = self.bonds.T
        Force = np.zeros((N, N, d), float)
        Force[row, col] = vals
        Force[col, row] = -vals
        return Force

    def forceAlongBond(self, bondId: int, forceScale: float = 1e-2) -> np.ndarray:
        """Generate a normalized force along a given bond.

        Args:
            bondId (int): bond index.
            forceScale (float, optional): The force norm. Defaults to 1e-2.

        Returns:
            np.ndarray: normlazed force.
        """
        N, d = self.N, self.dim
        fExt = np.zeros([N, d])
        e0, e1 = self.bonds[bondId]
        forceVector = forceScale * self.dr[bondId]
        fExt[e0], fExt[e1] = forceVector, -forceVector
        return fExt.reshape(
            -1,
        )

    def couplingMatrix(self) -> np.ndarray:

        H = self.hessianMatrix()
        Hinv = np.linalg.pinv(H)
        R = self.rigidityMatrix()
        RT = R.T
        G = np.dot(R, np.dot(Hinv, RT))
        return G

    def __couplingMatrixSparse(self) -> np.ndarray:
        H = self.hessianMatrixSparse()
        Hinv = inv(H)
        R = self.rigidityMatrixSparse()
        RT = R.transpose()
        G = R.dot(Hinv.dot(RT))
        return G

    def selfStress(self) -> np.ndarray:
        """Compute states of self-stress (SSS).

        Returns:
            np.ndarray: The output array shape is (NB, R) where R is the number of states of
        self-stress (or redundant bonds) and NB is the number of bonds.
        If there's no SSS, it returns `0`.
        """
        RT = self.rigidityMatrix().T
        u, s, vh = np.linalg.svd(RT)
        nullity = self.NB - np.sum(s > 1e-10)
        SSS = vh[-nullity:].T
        if nullity == 0:
            return 0
        else:
            return vh[-nullity:].T

    def elasticModulus(self, strainMatrix: np.ndarray) -> np.ndarray:
        """Return the elastic modulus for the given strain matrix.

        Note:
            To compute bulk and shear moduli, use ``bulkModulus`` and ``shearModulus`` functions.

        Args:
            strainMatrix (np.ndarray): In ``d`` dimensions, the strain matrix is a ``(d,d)`` array.

        Returns:
            np.array: elastic moduli
        """
        transformedBasis = np.einsum("ik,lk->li", strainMatrix, self.basis)
        transformedCoordinates = np.einsum("ik,lk->li", strainMatrix, self.coordinates)

        volume = self.volume
        K = self.K  # spring constant matrix
        H = self.hessianMatrix()
        R = self.rigidityMatrix()
        RT = R.T

        F2 = framework(
            transformedCoordinates, self.bonds, basis=transformedBasis, k=np.diag(K)
        )
        dl = F2.edgeLengths() - self.edgeLengths()
        Force = -np.dot(RT, K.dot(dl))
        u = np.linalg.lstsq(H, Force, rcond=-1)
        Dl = np.dot(R, u[0])
        energy = 0.5 * np.sum(np.dot(K, (Dl + dl) ** 2))
        modulus = energy / volume
        return modulus

    def bulkModulus(self, eps: float = 1e-6) -> float:
        """Compute shear modulus.

        Args:
            eps (float, optional): strain amount. Defaults to 1e-6.

        Returns:
            float: bulk
        """
        strainArray = np.full((self.nbasis,), 1 - eps)
        strainMatrix = np.diag(strainArray)
        bulk = self.elasticModulus(strainMatrix) / (2 * eps * eps)
        return bulk

    def shearModulus(self, eps: float = 1e-6) -> float:
        """Compute (pure) shear modulus.

        Args:
            eps (float, optional): strain amount. Defaults to 1e-6.

        Returns:
            float: shear
        """
        strainArray = np.empty((self.nbasis,))
        strainArray[::2] = 1 - eps
        strainArray[1::2] = 1 + eps
        strainMatrix = np.diag(strainArray)
        shear = self.elasticModulus(strainMatrix) / (2 * eps * eps)
        return shear
