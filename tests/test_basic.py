from context import rigidpy as rp
import numpy as np
import os


def read_data(datasize=96):
    """The library has two different networks to test. One with 96 sites
    and one with 6 sites.

    Args:
        datasize (int, optional): The number of sites in the file. Defaults to 96.

    Returns:
        np.array: coordinates
        np.array: bonds
        np.array: basis
    """
    CUR_DIR = os.path.dirname(__file__)
    RESOURCES_DIR = f"{CUR_DIR}/data_{datasize}"
    coordinates = np.loadtxt(f"{RESOURCES_DIR}/coordinates.dat")
    bonds = np.loadtxt(f"{RESOURCES_DIR}/bonds.dat", int)
    basis = np.loadtxt(f"{RESOURCES_DIR}/basis.dat")
    return coordinates, bonds, basis


def test_framework_triangle():
    # position of sites
    coordinates = np.array([[-1, 0], [1, 0], [0, 1]])
    # list of sites between sites
    bonds = np.array([[0, 1], [1, 2], [0, 2]])
    # repetition of unit-cell
    basis = [[0, 0], [0, 0]]
    # which site is pinned in space
    pins = [0]
    # create a Framework object
    F = rp.framework(coordinates, bonds, basis=basis)
    assert F.bulkModulus() < 1e-20, "Bulk modulus is not zero."
    assert F.shearModulus() < 1e-20, "Bulk modulus is not zero."


def test_framework_periodic():
    """Test Framework module."""
    coordinates, bonds, basis = read_data(96)
    # create a Framework object
    F = rp.framework(coordinates, bonds, basis=basis)
    # calculate the rigidity matrix
    rigidity_matrix_shape = F.rigidityMatrix().shape
    # calculate the eigenvalues of Hessian/dynamical matrix
    eigvals, eigvecs = F.eigenSpace(eigvals=(0, 5))

    assert len(eigvals) == 6, "Number of returned eigenvalues is not 6."
    assert np.sum(eigvals < 1e-10) == 3, "Number of zero-modes is not 3."
    assert rigidity_matrix_shape == (
        F.NB,
        F.N * F.dim,
    ), "Shape of rigidity matrix is not correct."
    assert F.bulkModulus() < 1e-10, "Bulk modulus is not zero."
    assert F.shearModulus() < 1e-10, "Shear modulus is not zero."
