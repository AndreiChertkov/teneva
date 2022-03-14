maxvol: compute the maximal-volume submatrix
--------------------------------------------


.. automodule:: teneva.core.maxvol


-----


.. autofunction:: teneva.maxvol

  **Examples**:

  .. code-block:: python

    n = 5000                        # Number of rows
    r = 50                          # Number of columns
    A = np.random.randn(n, r)       # Random tall matrix

  .. code-block:: python

    e = 1.01                        # Accuracy parameter
    k = 500                         # Maximum number of iterations

  .. code-block:: python

    I, B = teneva.maxvol(A, e, k)   # Compute row numbers and coefficient matrix
    C = A[I, :]                     # Maximal-volume square submatrix

  .. code-block:: python

    print(f'|Det C|        : {np.abs(np.linalg.det(C)):-10.2e}')
    print(f'Max |B|        : {np.max(np.abs(B)):-10.2e}')
    print(f'Max |A - B C|  : {np.max(np.abs(A - B @ C)):-10.2e}')
    print(f'Selected rows  : {I.size:-10d} > ', np.sort(I))

    # >>> ----------------------------------------
    # >>> Output:

    # |Det C|        :   2.41e+40
    # Max |B|        :   1.01e+00
    # Max |A - B C|  :   1.20e-14
    # Selected rows  :         50 >  [  70  138  169  230  239  278  346  387  393  416  549  670  673  821
    #   931 1007 1195 1278 1281 1551 1658 1822 1823 1927 2312 2335 2381 2529
    #  2570 2634 2757 2818 3208 3239 3408 3626 3688 3739 3822 3833 3834 4079
    #  4144 4197 4529 4627 4874 4896 4905 4977]
    # 


-----


.. autofunction:: teneva.maxvol_rect

  **Examples**:

  .. code-block:: python

    n = 5000                        # Number of rows
    r = 50                          # Number of columns
    A = np.random.randn(n, r)       # Random tall matrix

  .. code-block:: python

    e = 1.01                        # Accuracy parameter
    dr_min = 2                      # Minimum number of added rows
    dr_max = 8                      # Maximum number of added rows
    e0 = 1.05                       # Accuracy parameter for the original maxvol algorithm
    k0 = 50                         # Maximum number of iterations for the original maxvol algorithm

  .. code-block:: python

    I, B = teneva.maxvol_rect(A, e,
        dr_min, dr_max, e0, k0)     # Row numbers and coefficient matrix
    C = A[I, :]                     # Maximal-volume rectangular submatrix

  .. code-block:: python

    print(f'Max |B|        : {np.max(np.abs(B)):-10.2e}')
    print(f'Max |A - B C|  : {np.max(np.abs(A - B @ C)):-10.2e}')
    print(f'Selected rows  : {I.size:-10d} > ', np.sort(I))

    # >>> ----------------------------------------
    # >>> Output:

    # Max |B|        :   1.00e+00
    # Max |A - B C|  :   9.71e-15
    # Selected rows  :         58 >  [ 233  294  306  553  564  566  574  623  732  739  754  899  901 1095
    #  1142 1190 1275 1316 1416 1560 1605 1622 2028 2051 2084 2085 2108 2293
    #  2339 2519 2574 2667 2705 2757 2782 2975 3147 3159 3170 3251 3330 3360
    #  3499 3564 3599 3627 3641 3849 3893 4135 4274 4453 4549 4740 4819 4837
    #  4891 4933]
    # 

  We may select "dr_max" as None. In this case the number of added rows will be determined by the precision parameter "e" (the resulting submatrix can even has the same size as the original matrix "A"):

  .. code-block:: python

    e = 1.01                        # Accuracy parameter
    dr_max = None                   # Maximum number of added rows
    I, B = teneva.maxvol_rect(A, e,
        dr_min, dr_max, e0, k0)     # Compute row numbers and coefficient matrix
    C = A[I, :]                     # Maximal-volume rectangular submatrix
    
    print(f'Max |B|        : {np.max(np.abs(B)):-10.2e}')
    print(f'Max |A - B C|  : {np.max(np.abs(A - B @ C)):-10.2e}')
    print(f'Selected rows  : {I.size:-10d} > ', np.sort(I))

    # >>> ----------------------------------------
    # >>> Output:

    # Max |B|        :   1.00e+00
    # Max |A - B C|  :   7.27e-15
    # Selected rows  :         93 >  [ 233  281  294  306  362  526  553  564  566  574  608  623  642  732
    #   739  745  754  761  899  901 1095 1102 1142 1190 1219 1275 1283 1316
    #  1416 1560 1605 1622 1955 1968 2028 2051 2084 2085 2108 2214 2243 2292
    #  2293 2339 2409 2422 2507 2519 2566 2574 2643 2661 2665 2667 2705 2757
    #  2782 2864 2975 3147 3159 3170 3251 3258 3330 3360 3487 3499 3506 3532
    #  3564 3599 3627 3641 3849 3893 3907 4066 4115 4135 4201 4274 4453 4502
    #  4526 4549 4740 4767 4819 4837 4891 4933 4979]
    # 

  .. code-block:: python

    A = np.random.randn(20, 5)      # Random tall matrix
    e = 0.1                         # Accuracy parameter (we select very small value here)
    dr_max = None                   # Maximum number of added rows
    I, B = teneva.maxvol_rect(A, e,
        dr_min, dr_max, e0, k0)     # Row numbers and coefficient matrix
    C = A[I, :]                     # Maximal-volume rectangular submatrix
    
    print(f'Max |B|        : {np.max(np.abs(B)):-10.2e}')
    print(f'Max |A - B C|  : {np.max(np.abs(A - B @ C)):-10.2e}')
    print(f'Selected rows  : {I.size:-10d} > ', np.sort(I))

    # >>> ----------------------------------------
    # >>> Output:

    # Max |B|        :   1.00e+00
    # Max |A - B C|  :   0.00e+00
    # Selected rows  :         20 >  [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
    # 


