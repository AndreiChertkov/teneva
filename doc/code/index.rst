Description of functions and examples
=====================================

Below, we provide a brief description and demonstration of the capabilities of each function from the package. Most functions take "Y" - a list of the TT-cores "G1", "G2", ..., "Gd" (3D numpy arrays) - as an input argument and return its updated representation as a new list of TT-cores or some related scalar values (mean, norm, etc.). Sometimes to demonstrate a specific function, it is also necessary to use some other functions from the package, in this case we do not provide comments for the auxiliary function, however all related information can be found in the relevant subsection.

Please, note that all demos assume the following imports:

  .. code-block:: python

    import numpy as np
    import teneva
    from time import perf_counter as tpc
    np.random.seed(42)

-----

.. toctree::
  :maxdepth: 4

  act_one
  act_two
  act_many
  als
  als_func
  anova
  core
  cross
  cross_act
  data
  func
  func_full
  grid
  matrices
  maxvol
  optima
  optima_func
  props
  sample
  sample_func
  stat
  svd
  tensors
  transformation
  vectors
  vis
