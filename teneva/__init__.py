__version__ = '0.12.8'


from .collection import *
from .core import *
from .func import *


# We add utilities for convenience to the common namespace:
from .core.utils import _is_num
from .core.utils import _maxvol
from .core.utils import _ones
from .core.utils import _range
from .core.utils import _reshape
