Notation and general comments
=============================

[THIS IS A DRAFT OF THE TEXT !]

In most cases, we use the following notation for function input arguments and intermediate variables:

- "d" - number of dimensions of the tensor (multidimensional array) or of the multivariable function's input;
- "f" - multivariable function (black box), which is the real function of the "d"-dimensional argument;
- "Y", "Z" - "d"-dimensional TT-tensor, which is a python list of the length "d" combined of the "d" TT-cores (3-dimensional numpy arrays);
- "G", "Q" - TT-core, which is a 3-dimensional numpy array;
- "a" / "b" - rectangular grid lower / upper bound for each dimension of the multivariable function input (list of the length "d");
- "n" - tensor size for each mode (list of the length "d");
- "m" - sample / batch size (int);
- "i" - a tensor multi-index (1-dimensional numpy array of the length "d");
- "I" - a set of tensor multi-indices (2-dimensional numpy array of the shape "samples" x "d", where "samples" is the number of samples/indices/points). We also use postfixes "_trn", "_vld" and "_tst" for the training, validation, and test datasets, respectively;
- "X_data" - a set of function inputs (2-dimensional numpy array of the shape "samples" x "d", where "samples" is the number of samples/points). We also use postfixes "_trn", "_vld" and "_tst" for the training, validation, and test datasets, respectively;
- "y_data" - a set of function outputs (1-dimensional numpy array of the shape "samples", where "samples" is the number of samples/points). We also use postfixes "_trn", "_vld" and "_tst" for the training, validation, and test datasets, respectively.
