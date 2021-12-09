# teneva


## Description

This python package, named **teneva** (**ten**sor **eva**luation), provides a very compact implementation of basic operations in the tensor-train (TT) format, including TT-SVD, TT-ALS, TT-ANOVA, TT-cross, TT-truncate, "add", "mul", "norm", "mean", etc. The program code is organized within a functional paradigm and it is very easy to learn and use.


## Installation

The package can be installed via pip: `pip install teneva` (it requires the [Python](https://www.python.org) programming language of the version >= 3.6). It can be also downloaded from the repository [teneva](https://github.com/AndreiChertkov/teneva) and installed by `python setup.py install` command from the root folder of the project.

> Required python packages [numpy](https://numpy.org), [scipy](https://www.scipy.org) and [numba](https://github.com/numba/numba) will be automatically installed during the installation of the main software product.


## Examples

- See the colab notebook [teneva_demo](https://colab.research.google.com/drive/1tRlJGk497N0UpBkR4bhCmymO9lPEnQmY?usp=sharing) with brief description and demonstration of the capabilities of each function from the `teneva` package, including the basic examples of using of the TT-ALS, TT-ANOVA and TT-cross for multidimensional function approximation (we approximate the 100 dimensional Rosenbrock function for the simple demonstration).


## Tutorials

- Colab-notebook [Tensor train basics](https://colab.research.google.com/drive/1TR-ptUINvglasplQCLXdl2g0F3Nh5AIG?usp=sharing) with a detailed description of the specific features of the tensor train decomposition and demos.
- Colab-notebook [Build and truncate the tensor train](https://colab.research.google.com/drive/17yW1ILOTgf1lvJEqUrn6YcHki-WYCozN?usp=sharing) with a description of the method for constructing a TT-decomposition for a given tensor (TT-SVD algorithm) and a method for additional rounding (compression, truncation) of the TT-decomposition, including program code and numerical examples.
- Colab-notebook [Maxvol and Maxvol_rect algorithms](https://colab.research.google.com/drive/186ig_CS7RA5WVRwBPzT7Wu-vwKXZrm7m?usp=sharing) with a detailed description of the `maxvol` algorithm to efficiently find the maximum volume submatrix in a given matrix, its program code (including jax-draft) and demo examples.
- Colab-notebook [Black box approximation with tensor train](https://colab.research.google.com/drive/1zfqwAdHAOiSbbgpPOiufmXgoErukhq4N?usp=sharing) with examples for multidimensional function (black box) approximation with the TT-ALS, TT-ANOVA and TT-cross approaches.


## Authors

- [Andrei Chertkov](https://github.com/AndreiChertkov) (a.chertkov@skoltech.ru);
- [Gleb Ryzhakov](https://github.com/G-Ryzhakov) (g.ryzhakov@skoltech.ru);
- [Ivan Oseledets](https://github.com/oseledets) (i.oseledets@skoltech.ru).
