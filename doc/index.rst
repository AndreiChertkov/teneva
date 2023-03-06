Documentation
=============

Python package **teneva** (**ten**\ sor **eva**\ luation) provides a very compact implementation of basic operations in the tensor-train (TT) format, including TT-SVD, TT-ALS, TT-ANOVA, TT-CROSS, TT-truncate, "add", "mul", "norm", "mean", Chebyshev interpolation, etc. This approach can be used for approximation of multidimensional arrays and multivariate functions, as well as for efficient implementation of various operations of linear algebra in the low rank format. The core program code is organized within a functional paradigm and it is very easy to learn and use.

-----

Current version "0.13.2".

-----

Below, we provide a brief description and demonstration of the capabilities of each function from the package. Most functions take "Y" - a list of the TT-cores "G1", "G2", ..., "Gd" (3D numpy arrays) - as an input argument and return its updated representation as a new list of TT-cores or some related scalar values (mean, norm, etc.). Sometimes to demonstrate a specific function, it is also necessary to use some other functions from the package, in this case we do not provide comments for the auxiliary function, however all related information can be found in the relevant subsection.

-----

.. toctree::
  :maxdepth: 1

  core/index
  core_jax/index
  func/index

-----

Please, note that all demos from "teneva/core" assume the following imports:

  .. code-block:: python

    import numpy as np
    import teneva
    from time import perf_counter as tpc
    np.random.seed(42)

All demos from "teneva/core_jax" assume the following imports:

  .. code-block:: python

    import jax
    import jax.numpy as np
    import teneva as teneva_base
    import teneva.core_jax as teneva
    from time import perf_counter as tpc
    rng = jax.random.PRNGKey(42)

In most cases, we use the following notation for function input arguments and intermediate variables:

- "d" - number of dimensions of the tensor (multidimensional array) or of the multivariable function's input;
- "f" - multivariable function (black box), which is the real function of the "d"-dimensional argument;
- "Y", "Z" - "d"-dimensional TT-tensor, which is python list of the length "d" of the TT-cores (3-dimensional numpy arrays);
- "G", "Q" - TT-core, which is 3-dimensional numpy array;
- "a" - rectangular grid lower bounds for each dimension of the multivariable function input (list of length "d");
- "b" - rectangular grid upper bounds for each dimension of the multivariable function input (list of length "d");
- "n" - tensor size for each mode (list of length "d");
- "m" - sample/batch size (int);
- "i" - a tensor multi-index (1-dimensional numpy array of the length "d";
- "I" - a set of tensor multi-indices (2-dimensional numpy array of the shape "samples" x "d", where "samples" is the number of samples/indices/points). We also use postfixes "_trn", "_vld" and "_tst" for the training, validation, and test datasets, respectively;
- "X_data" - a set of function inputs (2-dimensional numpy array of the shape "samples" x "d", where "samples" is the number of samples/points). We also use postfixes "_trn", "_vld" and "_tst" for the training, validation, and test datasets, respectively;
- "y_data" - a set of function outputs (1-dimensional numpy array of the shape "samples", where "samples" is the number of samples/points). We also use postfixes "_trn", "_vld" and "_tst" for the training, validation, and test datasets, respectively.

-----

> JAX-version (NEW)

All basic functionality ("teneva/core") is written within the framework of standard python libraries ("numpy", "scipy"). In this case, the d-dimensional tensor is represented as a list of its TT-cores (i.e., the list of d various 3D arrays). At the same time, we working on the alternative version ("teneva/core_jax") within the "jax" machine learning framework. As part of the "jax" version, in order to speed up (by orders of magnitude) code compilation (i.e., "jax.jit"), we only support tensors of constant mode size ("n") and TT-rank ("r"). In this case, the tensor ("d > 2") is represented as a list of three jax arrays: the first TT-core (3D array of the shape "1xnxr"), an array of all internal TT-cores (4D array of the shape "(d-2)xrxnxr"), and the last core (3D array of the shape "rxnx1"). Please also note that in all demo for jax-version we perfrom the following imports: "import teneva.core_jax as teneva" and "import teneva as teneva_base".

-----

- :ref:`genindex`
- :ref:`modindex`
- :ref:`search`
