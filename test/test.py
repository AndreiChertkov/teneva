"""Tests for teneva (in progress).

Run it from the root of the project as "clear && python test/test.py".

"""
import unittest


from test_act_one import *
from test_als import *
from test_cross import *
from test_func import *
from test_grid import *
from test_maxvol import *
from test_tensors import *


if __name__ == '__main__':
    np.random.seed(42)
    unittest.main()
