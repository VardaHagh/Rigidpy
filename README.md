# RigidPy

`RigidPy` is a library written in python with minimal dependency to compute
rigidity properties of a graph/network. The namespace is designed to
resemble mathematical terminology unifying the research themes in mathematics,
physics and biology.
The library can be used to compute the rigidity matrix, vibrational properties,
and elastic properties of the spring networks. The library also offers a set of
tools to find different realizations of a graph for a given topology.

### Example

```python
import numpy as np
import rigidpy as rp
# position of sites
coordinates = np.array([[-1,0], [1,0], [0,1]])
# list of sites between sites
bonds = np.array([[0,1],[1,2],[0,2]])
# repetition of unit-cell
basis = [[0,0],[0,0]]
# which site is pinned in space
pins = [0]
# create a Framework object
F = rp.Framework(coordinates, bonds, basis=basis, pins=pins)
# calculate the rigidity matrix
print ("rigidity matrix:\n",F.RigidityMatrix())
# calculate the eigenvalues of Hessian/dynamical matrix
eigvals, eigvecs = F.Eigenspace(eigvals=(0,3))
print("vibrational eigenvalues:\n",eigvals)
```

Output:
```python
rigidity matrix:
 [[ 1.         -0.          0.          0.        ]
 [ 0.70710678 -0.70710678 -0.70710678  0.70710678]
 [ 0.          0.          0.70710678  0.70710678]]
vibrational eigenvalues:
 [2.22044605e-16 6.33974596e-01 1.00000000e+00 2.36602540e+00]

```


### Requirements

* `Python` 3.5+
* `Numpy` (>= 1.9.3)
* `Scipy` (>= 0.14.0)
* `Matplotlib` (>= 1.4.3) (only required for plotting)


### Authors:

* [Varda Faghir Hagh](https://github.com/vfaghirh)
* [Mahdi Sadjadi](https://github.com/Mahdisadjadi)

### License
This project is licensed under the MIT License - see the LICENSE.md file for details.
