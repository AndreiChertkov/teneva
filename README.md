# teneva


## Description

This python package, named **teneva** (**ten**sor **eva**luation), provides a very compact implementation of basic operations in the tensor-train (TT) format, including TT-SVD, TT-ALS, TT-ANOVA, TT-cross, TT-truncate, Chebyshev interpolation, "add", "mul", "norm", "mean", "sample", etc. Our approach can be used for approximation of multidimensional arrays and multivariate functions, as well as for efficient implementation of various operations of linear algebra in the low rank TT-format. The program code is organized within a functional paradigm and it is very easy to learn and use. Each function has detailed documentation and various usage demos.


## Installation

> Current version "0.14.3".

The package can be installed via pip: `pip install teneva` (it requires the [Python](https://www.python.org) programming language of the version >= 3.8). It can be also downloaded from the repository [teneva](https://github.com/AndreiChertkov/teneva) and installed by `python setup.py install` command from the root folder of the project.


## Documentation, examples and tests

- See detailed [online documentation](https://teneva.readthedocs.io) for a description of each function and various numerical examples for each function.

- See the jupyter notebooks in the `demo` folder of the repository [teneva](https://github.com/AndreiChertkov/teneva) with brief description and demonstration of the capabilities of each function from the `teneva` package, including the basic examples of using the TT-ALS, TT-ANOVA and TT-cross for approximation of the multivariable functions. Note that all examples from this folder are also presented in the online documentation.

- See [changelog.md](https://github.com/AndreiChertkov/teneva/blob/master/changelog.md) file with a description of the changes made for new package versions and [workflow.md](https://github.com/AndreiChertkov/teneva/blob/master/workflow.md) file with a description of the rules we use to work on the code (draft!).

- Run all the tests (based on the `unittest` framework) from the root of the repository [teneva](https://github.com/AndreiChertkov/teneva) as `python test/test.py` (draft!).


## Useful links

- The github repository [teneva_jax](https://github.com/AndreiChertkov/teneva_jax) with the fast `jax` version of the `teneva` code.

- The github repository [teneva_bm](https://github.com/AndreiChertkov/teneva_bm) with benchmarks library for testing multidimensional approximation and optimization methods.

- The github repository [ttopt](https://github.com/AndreiChertkov/ttopt) with the gradient-free optimization method `TTOpt` for multivariable functions based on the TT-format and maximal-volume principle (see also [NeurIPS-2022 paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a730abbcd6cf4a371ca9545db5922442-Abstract-Conference.html)).

- The github repository [PROTES](https://github.com/anabatsh/PROTES) with the optimization method `PROTES` (PRobability Optimizer with TEnsor Sampling) for derivative-free optimization of the multidimensional arrays and discretized multivariate functions based on the TT-format (see also [arxiv paper](https://arxiv.org/pdf/2301.12162.pdf)).

- The github repository [Constructive-TT](https://github.com/G-Ryzhakov/Constructive-TT) with the method for constructive TT-representation of the tensors given as index interaction functions (see also [ICLR-2023 paper](https://openreview.net/forum?id=yLzLfM-Esnu)).

- Dissertation work [Computational tensor methods and their applications](https://disk.yandex.ru/i/JEQXcFQlGuntyQ) (in Russian only), in which the TT-decomposition is proposed and all its properties are described in detail.

- Dissertation work [Tensor methods for multidimensional differential equations](https://www.hse.ru/sci/diss/847453144) (in Russian only), which presents various new algorithms in the TT-format for problems of multidimensional approximation, optimization and solution of differential equations.


## Authors

- [Andrei Chertkov](https://github.com/AndreiChertkov)
- [Gleb Ryzhakov](https://github.com/G-Ryzhakov)
- [Ivan Oseledets](https://github.com/oseledets)

> ✭__🚂  The stars that you give to **teneva**, motivate us to develop faster and add new interesting features to the code 😃


## Citation

If you find our approach and/or code useful in your research, please consider citing:

```bibtex
@article{chertkov2023black,
    author    = {Chertkov, Andrei and Ryzhakov, Gleb and Oseledets, Ivan},
    year      = {2023},
    title     = {Black box approximation in the tensor train format initialized by {ANOVA} decomposition},
    journal   = {arXiv preprint arXiv:2208.03380 (accepted into the SIAM Journal on Scientific Computing)},
    doi       = {10.48550/ARXIV.2208.03380},
    url       = {https://arxiv.org/pdf/2208.03380.pdf}
}
```

```bibtex
@article{chertkov2022optimization,
    author    = {Chertkov, Andrei and Ryzhakov, Gleb and Novikov, Georgii and Oseledets, Ivan},
    year      = {2022},
    title     = {Optimization of functions given in the tensor train format},
    journal   = {arXiv preprint arXiv:2209.14808},
    doi       = {10.48550/ARXIV.2209.14808},
    url       = {https://arxiv.org/pdf/2209.14808.pdf}
}
```
