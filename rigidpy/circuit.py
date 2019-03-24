from __future__ import division, print_function, absolute_import


def Circuit(coordinates, bonds, basis, mode, k=1, varcell=None):
    """

    Set up a circuit

    Parameters
    ----------
    coordinates: (N, d) array
        Vertex/site coordinates. For ``N`` sites in ``d`` dimensions, the shape is ``(N,d)``.
    bonds: (M, 2) tuple or array
        Edge list. For ``M`` bonds, its shape is ``(M,2)``.
    basis: (d,d) array, optional
        List of basis/repeat/lattice vectors, Default: ``None``.
        If ``None`` or array of zero vectors, the system is assumed to be finite.
    mode: string
        mode of traversal in the configuration space. Options are "length" or "volume".
    k: int/float or array of int/floats, optional
        Spring constant/stiffness. Default: `1`.
        If an array is supplied, the shape should be ``(M,2)``.
    varcell: (d*d,) array of booleans/int
        A list of basis vector components allowed to change (1/True) or
        fixed (0/False). Example: ``[0,1,0,0]`` or ``[False, True, False, False]``
        both mean that in two dimensions, only second element of first
        basis vector is allowed to change.

    Returns
    -------
    A circuit object based on the selected mode.
    """
    if mode == "length":
        from .circuit_length import circuit_length
        circuit = circuit_length(coordinates, bonds, basis, k=k, varcell=None)
        return circuit
    elif mode == "volume":
        from .circuit_volume import circuit_volume
        circuit = circuit_volume(coordinates, bonds, basis, k=k, varcell=varcell)
        return circuit
    else:
        raise TypeError("You should provide a mode variable, possible values: 'length' and 'volume'.")
