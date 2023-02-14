"""Package teneva, module core.data: functions for working with datasets.

This module contains functions for working with datasets, including
"accuracy_on_data" function.

"""
import numpy as np
import teneva


def accuracy_on_data(Y, I_data, y_data, e_trunc=None):
    """Compute the relative error of TT-tensor on the dataset.

    Args:
        I_data (np.ndarray): multi-indices for items of dataset in the form of
            array of the shape [samples, d].
        y_data (np.ndarray): values for items related to I_data of dataset in
            the form of array of the shape [samples].
        e_trunc (float): optional truncation accuracy. If this parameter is
            set, then sampling will be performed from the rounded TT-tensor.

    Returns:
        float: the relative error.

    Note:
        If I_data or y_data is not provided, the function will return -1.

    """
    if I_data is None or y_data is None:
        return -1.

    I_data = np.asanyarray(I_data, dtype=int)
    y_data = np.asanyarray(y_data, dtype=float)

    if e_trunc is not None:
        Y = teneva.truncate(Y, e_trunc)

    y = teneva.get_many(Y, I_data)
    return np.linalg.norm(y - y_data) / np.linalg.norm(y_data)
