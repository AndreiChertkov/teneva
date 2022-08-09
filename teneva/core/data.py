"""Package teneva, module core.data: functions for working with datasets.

This module contains functions for working with datasets, including
"accuracy_on_data" function.

"""
import numpy as np


from .act_one import getter
from .transformation import truncate


def accuracy_on_data(Y, I_data, Y_data, e_trunc=None):
    """Compute the relative error of TT-tensor on the dataset.

    Args:
        I_data (np.ndarray): multi-indices for items of dataset in the form of
            array of the shape [samples, d].
        Y_data (np.ndarray): values for items related to I_data of dataset in
            the form of array of the shape [samples].
        e_trunc (float): optional truncation accuracy (> 0). If this parameter
            is set, then sampling will be performed from the rounded TT-tensor.

    Returns:
        float: the relative error.

    Note:
        If "I_data" or "Y_data" is not provided, the function will return "-1".

    """
    if I_data is None or Y_data is None:
        return -1.

    I_data = np.asanyarray(I_data, dtype=int)
    Y_data = np.asanyarray(Y_data, dtype=float)

    if e_trunc is not None:
        get = getter(truncate(Y, e_trunc))
    else:
        get = getter(Y)

    Z = np.array([get(i) for i in I_data])
    e = np.linalg.norm(Z - Y_data)
    e /= np.linalg.norm(Y_data)
    return e
