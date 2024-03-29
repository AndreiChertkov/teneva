Notation and comments
=====================

In this documentation we provide a brief description and demonstration of the capabilities of each function from the package. Most functions take "Y" - a list of the TT-cores "G1", "G2", ..., "Gd" (3D numpy arrays) - as an input argument and return its updated representation as a new list of TT-cores or some related scalar values (mean, norm, etc.). Sometimes to demonstrate a specific function, it is also necessary to use some other functions from the package, in this case we do not provide comments for the auxiliary function, however all related information can be found in the relevant subsection.

-----

Please, note that all demos assume the following imports:

  .. code-block:: python

    import numpy as np
    import teneva
    from time import perf_counter as tpc
    np.random.seed(42)

-----

In most cases, we use the following notation for function input arguments and intermediate variables:

- "d" - number of dimensions of the tensor (multidimensional array) or of the multivariable function's input;
- "f" - multivariable function (black box), which is the real function of the "d"-dimensional argument;
- "Y", "Z" - "d"-dimensional TT-tensor, which is a python list of the length "d" combined of the "d" TT-cores (3-dimensional numpy arrays);
- "G", "Q" - TT-core, which is a 3-dimensional numpy array;
- "a" / "b" - rectangular grid lower / upper bound for each dimension of the multivariable function input (list of the length "d");
- "n" - tensor size for each mode (list of the length "d");
- "m" - sample / batch size or computational budget (int);
- "i" - a tensor multi-index (1-dimensional numpy array of the length "d");
- "I" - a set of tensor multi-indices (2-dimensional numpy array of the shape "samples" x "d", where "samples" is the number of samples/indices/points). We also use postfixes "_trn", "_vld" and "_tst" for the training, validation, and test datasets, respectively;
- "X_data" - a set of function inputs (2-dimensional numpy array of the shape "samples" x "d", where "samples" is the number of samples/points). We also use postfixes "_trn", "_vld" and "_tst" for the training, validation, and test datasets, respectively;
- "y_data" - a set of function outputs (1-dimensional numpy array of the shape "samples", where "samples" is the number of samples/points). We also use postfixes "_trn", "_vld" and "_tst" for the training, validation, and test datasets, respectively.
