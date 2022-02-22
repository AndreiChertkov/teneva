from .func_demo_ackley import FuncDemoAckley
from .func_demo_brown import FuncDemoBrown
from .func_demo_grienwank import FuncDemoGrienwank
from .func_demo_michalewicz import FuncDemoMichalewicz
from .func_demo_piston import FuncDemoPiston
from .func_demo_rastrigin import FuncDemoRastrigin
from .func_demo_rosenbrock import FuncDemoRosenbrock
from .func_demo_schaffer import FuncDemoSchaffer
from .func_demo_schwefel import FuncDemoSchwefel


def func_demo_all(d, with_piston=False):
    """Build list of class instances for all demo functions.

    Args:
        d (int): number of dimensions.
        with_piston (bool): If True, then Piston function will be also
            added to the list. Note that this function is 7-dimensional,
            hence given argument "d" will be ignored for this function.

    Returns:
        list: the list of class instances for all demo functions.

    """
    funcs = [
        FuncDemoAckley(d),
        FuncDemoBrown(d),
        FuncDemoGrienwank(d),
        FuncDemoMichalewicz(d),
        FuncDemoRastrigin(d),
        FuncDemoRosenbrock(d),
        FuncDemoSchaffer(d),
        FuncDemoSchwefel(d),
    ]

    if with_piston:
        func = FuncDemoPiston(d=7)
        funcs.insert(3, func)

    return funcs
