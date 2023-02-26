Module data: functions for working with datasets
------------------------------------------------


.. automodule:: teneva.core_jax.data


-----




|
|

.. autofunction:: teneva.core_jax.data.accuracy_on_data

  **Examples**:

  Let generate a random TT-tensor:

  .. code-block:: python

    d = 20  # Dimension of the tensor
    n = 10  # Mode size of the tensor
    r = 2   # TT-rank of the tensor

  .. code-block:: python

    rng, key = jax.random.split(rng)
    Y = teneva.rand(d, n, r, key)

  Then we generate some random multi-indices, compute related
    tensor values and add some noise:

  .. code-block:: python

    m = 100 # Size of the dataset
    I_data = teneva_base.sample_lhs([n]*d, m)
    y_data = teneva.get_many(Y, I_data)
    
    rng, key = jax.random.split(rng)
    y_data = y_data + 1.E-5*jax.random.normal(key, (m, ))

  And then let compute the accuracy:

  .. code-block:: python

    eps = teneva.accuracy_on_data(Y, I_data, y_data)
    
    print(f'Accuracy     : {eps:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Accuracy     : 4.11e-04
    # 




|
|

