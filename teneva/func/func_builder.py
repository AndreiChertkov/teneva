"""Package teneva, module func.func_builder: helpers to build benchmarks.

This module contains functions, which build one or many (with filters)
benchmarks (model functions).

"""
import teneva


def func_demo(d, name, dy=0.):
    """Build class instance for demo function by name.

    Args:
        d (int): number of dimensions.
        name (str): function name (in any register). The following functions
            are available: "ackley", "alpine", "dixon", "exponential",
            "grienwank", "michalewicz", "piston", "rastrigin", "rosenbrock",
            "schaffer" and "schwefel".
        dy (float): optional function shift (y -> y + dy).

    Returns:
        Func: the class instance for demo function.

    """
    funcs = func_demo_all(d, [name], dy)
    if len(funcs) == 0:
        raise ValueError(f'Unknown function "{name}"')
    return funcs[0]


def func_demo_all(d, names=None, dy=0., with_piston=False, only_with_cores=False, only_with_min=False, only_with_min_x=False):
    """Build list of class instances for all demo functions.

    Args:
        d (int): number of dimensions.
        names (list): optional list of function names (in any register),
            which should be added to the resulting list of class instances.
            The following functions are available: "ackley", "alpine", "dixon",
            "exponential", "grienwank", "michalewicz", "piston", "rastrigin",
            "rosenbrock", "schaffer" and "schwefel".
        dy (float): optional function shift (y -> y + dy).
        with_piston (bool): If True, then Piston function will be also
            added to the list. Note that this function is 7-dimensional,
            hence given argument "d" will be ignored for this function. The
            value of this flag does not matter if the names ("names" argument)
            of the added functions are explicitly specified.
        only_with_cores (bool): If True, then only functions with known TT-cores
            will be returned.
        only_with_min (bool): If True, then only functions with known exact
            y_min value will be returned.
        only_with_min_x (bool): If True, then only functions with known exact
            x_min value will be returned.

    Returns:
        list: the list of class instances for all demo functions.

    """
    funcs_all = [
        teneva.FuncDemoAckley(d, dy),
        teneva.FuncDemoAlpine(d, dy),
        teneva.FuncDemoDixon(d, dy),
        teneva.FuncDemoExponential(d, dy),
        teneva.FuncDemoGrienwank(d, dy),
        teneva.FuncDemoMichalewicz(d, dy),
        teneva.FuncDemoPiston(7, dy),
        teneva.FuncDemoQing(d, dy),
        teneva.FuncDemoRastrigin(d, dy),
        teneva.FuncDemoRosenbrock(d, dy),
        teneva.FuncDemoSchaffer(d, dy),
        teneva.FuncDemoSchwefel(d, dy),
    ]

    if names is not None:
        names = [name.lower() for name in names]

    funcs = []
    for func in funcs_all:
        if names is not None:
            if func.name.lower() not in names:
                continue
        else:
            if not with_piston and func.name == 'Piston':
                continue
            if only_with_cores and not func.with_cores:
                continue
            if only_with_min and not func.with_min:
                continue
            if only_with_min_x and not func.with_min_x:
                continue

        funcs.append(func)

    return funcs
