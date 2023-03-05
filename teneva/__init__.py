__version__ = '0.13.1'


from .core import *

# We add utilities for convenience to the common namespace:
from .core.utils import _info_appr
from .core.utils import _is_num
from .core.utils import _maxvol
from .core.utils import _ones
from .core.utils import _range
from .core.utils import _reshape
from .core.utils import _vector_index_expand
from .core.utils import _vector_index_prepare


from .func import *
