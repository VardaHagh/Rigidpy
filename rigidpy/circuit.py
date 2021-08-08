from __future__ import division, print_function, absolute_import
from typing import Union
import numpy as np
from rigidpy.circuit_length import circuitLength
from rigidpy.circuit_volume import circuitVolume


def circuit(
    coordinates: Union[np.array, list],
    bonds: Union[np.array, list],
    basis: Union[np.array, list],
    mode: str = "length",
    k: Union[np.array, float] = 1.0,
    varcell: Union[np.array, list] = None,
) -> Union[circuitVolume, circuitLength]:
    """Set up a circuit

    Args:
        coordinates (Union[np.array, list]): Vertex/site coordinates. For
                ``N`` sites in ``d`` dimensions, the shape is ``(N,d)``.
        bonds (Union[np.array, list]): Edge list. For ``M`` bonds, its
            shape is ``(M,2)``.
        basis (Union[np.array, list], optional): List of basis/repeat/lattice
            vectors, Default: ``None``.If ``None`` or array of zero vectors,
            the system is assumed to be finite. Defaults to None.
        mode (str, optional): mode of traversal in the configuration space.
            Options are "length" or "volume". Defaults to "length".
        k (Union[np.array, float], optional): Spring constant/stiffness.
                If an array is supplied, the shape should be ``(M,2)``.
                Defaults to 1.0.
        varcell (Union[np.array, list], optional): (d*d,) array of booleans/int
                A list of basis vector components allowed to change (1/True) or
                fixed (0/False). Example: ``[0,1,0,0]`` or ``[False, True, False, False]``
                both mean that in two dimensions, only second element of first
                basis vector is allowed to change.. Defaults to None.

    Raises:
        TypeError: If mode is not selected or is not among options.

    Returns:
        Union[circuit_volume, circuit_length]: A circuit object based on the selected mode.

    """
    if mode == "length":
        from .circuit_length import circuitLength

        circuit = circuitLength(coordinates, bonds, basis, k=k, varcell=None)
        return circuit
    elif mode == "volume":
        from .circuit_volume import circuitVolume

        circuit = circuitVolume(coordinates, bonds, basis, k=k, varcell=varcell)
        return circuit
    else:
        raise TypeError(
            "You should provide a mode variable, possible values: 'length' and 'volume'."
        )
