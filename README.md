# teneva


## Description

This python package, named **teneva** (**ten**sor **eva**luation), provides a very compact implementation of basic operations in the tensor-train (TT) format, including TT-SVD, TT-ALS, TT-ANOVA, TT-cross, TT-truncate, "add", "mul", "norm", "mean", Chebyshev interpolation, etc. This approach can be used for approximation of multidimensional arrays and multivariate functions, as well as for efficient implementation of various operations of linear algebra in the low rank format. The program code is organized within a functional paradigm and it is very easy to learn and use.


## Installation

> Current version "0.12.4".

The package can be installed via pip: `pip install teneva` (it requires the [Python](https://www.python.org) programming language of the version >= 3.6). It can be also downloaded from the repository [teneva](https://github.com/AndreiChertkov/teneva) and installed by `python setup.py install` command from the root folder of the project. Required python packages [numpy](https://numpy.org), [scipy](https://www.scipy.org), [numba](https://github.com/numba/numba) and [matplotlib](https://matplotlib.org/) will be automatically installed during the installation of the main software product.


## Documentation and examples

- See detailed [online documentation](https://teneva.readthedocs.io) for a description of each function and numerical examples.
- See the jupyter notebooks in the `./demo` folder with brief description and demonstration of the capabilities of each function from the `teneva` package, including the basic examples of using the TT-ALS, TT-ANOVA and TT-cross for approximation of the multivariable functions. Note that all examples from this folder are also presented in the online documentation.


## Authors

- [Andrei Chertkov](https://github.com/AndreiChertkov)
- [Gleb Ryzhakov](https://github.com/G-Ryzhakov)
- [Ivan Oseledets](https://github.com/oseledets)

> ✭ The stars that you give to **teneva**, motivate us to develop faster and add new interesting features to the code 😃


## Citation

If you find our approach and/or code useful in your research, please consider citing:

```bibtex
@article{chertkov2022black,
    author    = {Chertkov, Andrei and Ryzhakov, Gleb and Oseledets, Ivan},
    year      = {2022},
    title     = {Black box approximation in the tensor train format initialized by ANOVA decomposition},
    journal   = {arXiv preprint arXiv:2208.03380},
    doi       = {10.48550/ARXIV.2208.03380},
    url       = {https://arxiv.org/abs/2208.03380}
}
```
