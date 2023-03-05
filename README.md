# teneva


## Description

This python package, named **teneva** (**ten**sor **eva**luation), provides a very compact implementation of basic operations in the tensor-train (TT) format, including TT-SVD, TT-ALS, TT-ANOVA, TT-cross, TT-truncate, "add", "mul", "norm", "mean", Chebyshev interpolation, etc. This approach can be used for approximation of multidimensional arrays and multivariate functions, as well as for efficient implementation of various operations of linear algebra in the low rank format. The program code is organized within a functional paradigm and it is very easy to learn and use.

##### JAX-version (NEW)

All basic functionality (`teneva/core`) is written within the framework of standard python libraries (`numpy`, `scipy`). In this case, the d-dimensional tensor is represented as a list of its TT-cores (i.e., the list of d various 3D arrays). At the same time, we working on the alternative version (`teneva/core_jax`) within the `jax` machine learning framework. As part of the `jax` version, in order to speed up (by orders of magnitude) code compilation (i.e., `jax.jit`), we only support tensors of constant mode size (n) and TT-rank (r). In this case, the tensor (`d > 2`) is represented as a list of three jax arrays: the first TT-core (3D array of the shape `1xnxr`), an array of all internal TT-cores (4D array of the shape `(d-2)xrxnxr`), and the last core (3D array of the shape `rxnx1`). Please also note that in all demo for jax-version we perfrom the following imports: `import teneva.core_jax as teneva` and `import teneva as teneva_base`.


## Installation

> Current version "0.13.1".

The package can be installed via pip: `pip install teneva` (it requires the [Python](https://www.python.org) programming language of the version >= 3.8). It can be also downloaded from the repository [teneva](https://github.com/AndreiChertkov/teneva) and installed by `python setup.py install` command from the root folder of the project.

> Required python packages [numpy](https://numpy.org) (1.22+), [scipy](https://www.scipy.org) (1.8+) and [opt_einsum](https://github.com/dgasmith/opt_einsum) (3.3+) will be automatically installed during the installation of the main software product. However, it is recommended that you manually install them first.

> If you are going to use the jax version of the code, please install manually [jax[cpu]](https://github.com/google/jax) (`jax[cpu]>=0.4.3`).


## Documentation and examples

- See detailed [online documentation](https://teneva.readthedocs.io) for a description of each function and various numerical examples for each (!) function.
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

```bibtex
@article{chertkov2022optimization,
    author    = {Chertkov, Andrei and Ryzhakov, Gleb and Novikov, Georgii and Oseledets, Ivan},
    year      = {2022},
    title     = {Optimization of functions given in the tensor train format},
    journal   = {arXiv preprint arXiv:2209.14808},
    doi       = {10.48550/ARXIV.2209.14808},
    url       = {https://arxiv.org/abs/2209.14808}
}
```
