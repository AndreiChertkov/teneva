__version__ = '0.7.1'


from .als import als
from .als import als2


from .anova import ANOVA


from .cross import cross


from .grid import ind2poi
from .grid import ind2str
from .grid import sample_lhs
from .grid import sample_tt
from .grid import str2ind


from .maxvol import maxvol
from .maxvol import maxvol_rect


from .svd import matrix_skeleton
from .svd import matrix_svd
from .svd import svd
from .svd import svd_incomplete


from .tensor import add
from .tensor import erank
from .tensor import full
from .tensor import get
from .tensor import getter
from .tensor import mean
from .tensor import mul
from .tensor import norm
from .tensor import rand
from .tensor import show
from .tensor import sum
from .tensor import truncate


from .utils import confidence
from .utils import get_cdf
