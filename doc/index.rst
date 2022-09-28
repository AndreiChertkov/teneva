Documentation
=============

Python package **teneva** (**ten**\ sor **eva**\ luation) provides a very compact implementation of basic operations in the tensor-train (TT) format, including TT-SVD, TT-ALS, TT-ANOVA, TT-CROSS, TT-truncate, "add", "mul", "norm", "mean", Chebyshev interpolation, etc. This approach can be used for approximation of multidimensional arrays and multivariate functions, as well as for efficient implementation of various operations of linear algebra in the low rank format. The core program code is organized within a functional paradigm and it is very easy to learn and use.

-----

Current version "0.12.2".

-----

Below, we provide a brief description and demonstration of the capabilities of each function from the package. Most functions take "Y" - a list of the TT-cores "G1", "G2", ..., "Gd" (3D numpy arrays) - as an input argument and return its updated representation as a new list of TT-cores or some related scalar values (mean, norm, etc.). Sometimes to demonstrate a specific function, it is also necessary to use some other functions from the package, in this case we do not provide comments for the auxiliary function, however all related information can be found in the relevant subsection.

In most cases, we use the following notation:

- "d" - number of dimensions of the tensor (multidimensional array) or of the multivariable function's input;
- "f" - multivariable function (black box), which is the real function of the "d"-dimensional argument;
- "Y", "Z" - "d"-dimensional TT-tensor, which is python list of the length "d" of the TT-cores (3-dimensional numpy arrays);
- "G", "Q" - TT-core, which is 3-dimensional numpy array;
- "a" - rectangular grid lower bounds for each dimension of the multivariable function input (list of length "d");
- "b" - rectangular grid upper bounds for each dimension of the multivariable function input (list of length "d");
- "n" - tensor size for each mode (list of length "d");
- "m" - sample/batch size (int);
- "I" - a set of tensor multi-indices (2-dimensional numpy array of the shape "samples" x "d", where "samples" is the number of samples/indices/points). We also use postfixes "_trn", "_vld" and "_tst" for the training, validation, and validation datasets, respectively;
- "X_data" - a set of function inputs (2-dimensional numpy array of the shape "samples" x "d", where "samples" is the number of samples/points). We also use postfixes "_trn", "_vld" and "_tst" for the training, validation, and validation datasets, respectively;
- "Y_data" - a set of function outputs (1-dimensional numpy array of the shape "samples", where "samples" is the number of samples/points). We also use postfixes "_trn", "_vld" and "_tst" for the training, validation, and validation datasets, respectively.

.. toctree::
  :maxdepth: 1

  core/index
  collection/index
  func/index

-----

- :ref:`genindex`
- :ref:`modindex`
- :ref:`search`
