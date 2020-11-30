# teneva


## Description

This python package, named **teneva** (**ten**sor **eva**luation), provides very compact implementation for the multidimensional cross approximation algorithm in the tensor-train (TT) format.
This package also contains a function for quickly calculating the values of the constructed low-rank tensor approximation, as well as a number of auxiliary useful utilities.

In the current implementation, this software product actually duplicates the functionality related to the `rectcross` algoritm of the popular [ttpy](https://github.com/oseledets/ttpy) python package.
However, this compact implementation does not require a fortran compiler to be installed.


## Requirements

1. [Python](https://www.python.org) programming language (version >= 3.7).
1. "Standard" python packages [numpy](https://numpy.org) and [scipy](https://www.scipy.org) (all of them are included in [anaconda](https://www.anaconda.com/download/) distribution).
1. Python package [numba](https://github.com/numba/numba).
    > With this package, the tensor values at the given points will be calculated an order of magnitude faster.


## Installation

1. Install **python** (version >= 3.7) and "standard" python packages listed in the section **Requirements** above. The best way is to install only **anaconda** distribution which includes all the packages.
1. Install **numba** python package according to instructions from the corresponding repository.
1. Download this repository and run `python setup.py install` from the root folder of the project.
    > You can install this package via pip: `pip install teneva`.
1. To uninstall this package from the system run `pip uninstall teneva`.


## Examples

See this [colab notebook](https://colab.research.google.com/drive/1tRlJGk497N0UpBkR4bhCmymO9lPEnQmY?usp=sharing) with examples.

See also the folder `examples` with some demos in the `jupyter` format.


## Tests

See [colab notebook](https://colab.research.google.com/drive/1ijgeyefhGK3RXS_rnuHqsb_FObRGGSQa?usp=sharing), where the comparison with the package `ttpy` is provided.
