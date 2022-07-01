from .func_demo_ackley import FuncDemoAckley
from .func_demo_alpine import FuncDemoAlpine
from .func_demo_brown import FuncDemoBrown
from .func_demo_dixon import FuncDemoDixon
from .func_demo_exponential import FuncDemoExponential
from .func_demo_grienwank import FuncDemoGrienwank
from .func_demo_michalewicz import FuncDemoMichalewicz
from .func_demo_piston import FuncDemoPiston
from .func_demo_qing import FuncDemoQing
from .func_demo_rastrigin import FuncDemoRastrigin
from .func_demo_rosenbrock import FuncDemoRosenbrock
from .func_demo_schaffer import FuncDemoSchaffer
from .func_demo_schwefel import FuncDemoSchwefel


def func_demo_all(d, names=None, dy=0., with_piston=False):
    """Build list of class instances for all demo functions.

    Args:
        d (int): number of dimensions.
        dy (float): optional function shift (y -> y + dy).
        names (list): optional list of function names (in any register),
            which should be added to the resulting list of class instances.
            The following functions are available: "ackley", "brown",
            "grienwank", "michalewicz", "piston", "rastrigin", "rosenbrock",
            "schaffer" and "schwefel".
        with_piston (bool): If True, then Piston function will be also
            added to the list. Note that this function is 7-dimensional,
            hence given argument "d" will be ignored for this function. The
            value of this flag does not matter if the names ("names" argument)
            of the added functions are explicitly specified.

    Returns:
        list: the list of class instances for all demo functions.

    """
    funcs_all = [
        FuncDemoAckley(d, dy),
        FuncDemoAlpine(d, dy),
        FuncDemoBrown(d, dy),
        FuncDemoDixon(d, dy),
        FuncDemoExponential(d, dy),
        FuncDemoGrienwank(d, dy),
        FuncDemoMichalewicz(d, dy),
        FuncDemoPiston(7, dy),
        FuncDemoQing(d, dy),
        FuncDemoRastrigin(d, dy),
        FuncDemoRosenbrock(d, dy),
        FuncDemoSchaffer(d, dy),
        FuncDemoSchwefel(d, dy),
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

        funcs.append(func)

    return funcs
