# teneva


## Description

This python package, named **teneva** (**ten**sor **eva**luation), provides very compact implementation for the multidimensional cross approximation algorithm in the tensor-train (TT) format.
This package also contains a function for quickly calculating (using numba package) the values of the constructed low-rank tensor approximation, as well as a number of auxiliary useful utilities for rounding, adding, multiplying TT-tensors, etc.

**Notes**:

- This compact implementation does not require a fortran compiler to be installed, unlike the original [ttpy](https://github.com/oseledets/ttpy) python package.
- The program code is organized within a functional paradigm. Most functions take `Y` - a list of the TT-cores (3D numpy arrays) - as an input argument and return its updated representation as a new list of TT-cores or some related scalar values (mean, norm, etc.).


## Installation

The package (it requires the [Python](https://www.python.org) programming language of the version >= 3.6) can be installed via pip: `pip install teneva`. It can be also downloaded from the repository [teneva](https://github.com/AndreiChertkov/teneva) and installed by `python setup.py install` command from the root folder of the project.

> Required python packages [numpy](https://numpy.org), [scipy](https://www.scipy.org) and [numba](https://github.com/numba/numba) will be automatically installed during the installation of the main software product.


## Examples

See `demo/demo.py` script, which contains code for approximation of the multivariate (100 dimensional) Rosenbrock function with noise on a uniform grid by various methods (TT-ANOVA, TT-ALS, TT-Cross) and by its combinations.


## Tutorials

> All materials at the moment are presented in the form of drafts and are written in Russian.

- Colab-ноутбук [Разложение тензорного поезда](https://colab.research.google.com/drive/1TR-ptUINvglasplQCLXdl2g0F3Nh5AIG?usp=sharing) с подробным описанием специфических особенностей разложения тензорного поезда и демонстрационными примерами.
- Colab-ноутбук [Построение тензорного поезда и округление](https://colab.research.google.com/drive/17yW1ILOTgf1lvJEqUrn6YcHki-WYCozN?usp=sharing) с описанием метода построения TT-разложения для заданного тензора (алгоритм TT-SVD) и метода дополнительного округления (сжатия) TT-разложения, включая программный код и численные примеры.
- Colab-ноутбук [Алгоритмы maxvol и rect_maxvol](https://colab.research.google.com/drive/186ig_CS7RA5WVRwBPzT7Wu-vwKXZrm7m?usp=sharing) с подробным описанием алгоритма `maxvol`, его программным кодом (в том числе на jax) и демонстрационными примерами.
- Colab-ноутбук [Алгоритм TT-cross](https://colab.research.google.com/drive/1zfqwAdHAOiSbbgpPOiufmXgoErukhq4N?usp=sharing) с подробным описанием алгоритма `TT-cross`, его программным кодом и демонстрационными примерами.
- Colab-ноутбук [Алгоритм TT-als](https://colab.research.google.com/drive/1EOAkmwkFcswCGroSvUBaXjgPDZGkkkvJ?usp=sharing) с описанием алгоритма `TT-ALS`, его программным кодом и демонстрационными примерами.
