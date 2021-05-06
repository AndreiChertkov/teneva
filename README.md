# teneva


## Description

This python package, named **teneva** (**ten**sor **eva**luation), provides very compact implementation for the multidimensional cross approximation algorithm in the tensor-train (TT) format.
This package also contains a function for quickly calculating (using numba package) the values of the constructed low-rank tensor approximation, as well as a number of auxiliary useful utilities for rounding, adding, multiplying TT-tensors, etc.

Note that:
1. This compact implementation does not require a fortran compiler to be installed, unlike the original [ttpy](https://github.com/oseledets/ttpy) python package.
1. The program code is organized within a functional paradigm. Most functions take `Y` - a list of the TT-cores (3D numpy arrays) - as an input argument and return its updated representation as a new list of TT-cores or some related scalar values (mean, norm, etc.).
1. The simple form of the code presented in this repository allows in the future to rewrite it using popular jax framework.


## Requirements

1. [Python](https://www.python.org) programming language (version >= 3.6).
1. "Standard" python packages [numpy](https://numpy.org) and [scipy](https://www.scipy.org) (all of them are included in [anaconda](https://www.anaconda.com/download/) distribution).
1. Python package [numba](https://github.com/numba/numba).
    > With this package, the tensor values at the given points will be calculated an order of magnitude faster.

> All of these dependencies must be manually installed prior to installing this package.


## Installation

1. Install **python** (version >= 3.6) and "standard" python packages listed in the section **Requirements** above. The best way is to install only **anaconda** distribution which includes all the packages.
1. Install **numba** python package according to instructions from the corresponding repository.
1. Install this package via pip: `pip install teneva`.
    > You can also download the repository [teneva](https://github.com/AndreiChertkov/teneva) and run `python setup.py install` from the root folder of the project.
1. To uninstall this package from the system run `pip uninstall teneva`.


## Examples

- See the jupyter notebook `example.ipynb` in the repository, which contains various useful examples.
- See the colab notebook [teneva_demo](https://colab.research.google.com/drive/1tRlJGk497N0UpBkR4bhCmymO9lPEnQmY?usp=sharing) with the basic example.


## Tests

- See the folder `test` with detailed unit tests. Call it as
    ```bash
    python -m unittest test_base test_vs_ttpy
    ```
    > To run the test test_vs_ttpy, you should first install the [ttpy](https://github.com/oseledets/ttpy) python package.
