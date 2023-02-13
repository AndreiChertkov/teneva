Module data: functions for working with datasets
------------------------------------------------


.. automodule:: teneva.core.data


-----




|
|

.. autofunction:: teneva.accuracy_on_data

  **Examples**:

  .. code-block:: python

    m = 100       # Size of the dataset
    n = [5] * 10  # Shape of the tensor
    
    # Random TT-tensor with TT-rank 2:
    Y = teneva.tensor_rand(n, 2)
    
    # Let build toy dataset:
    I_data = teneva.sample_lhs(n, m)
    Y_data = [teneva.get(Y, i) for i in I_data]
    Y_data = np.array(Y_data)
    
    # Add add some noise:
    Y_data = Y_data + 1.E-3*np.random.randn(m)
    
    # Compute the accuracy:
    eps = teneva.accuracy_on_data(Y, I_data, Y_data)
    
    print(f'Accuracy     : {eps:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Accuracy     : 1.35e-04
    # 




|
|

