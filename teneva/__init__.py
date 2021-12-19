__version__ = '0.7.2'


from .als import als
from .als import als2


from .anova import ANOVA
from .anova import anova


from .cheb import cheb_bld
from .cheb import cheb_get
from .cheb import cheb_get_full
from .cheb import cheb_ind
from .cheb import cheb_int
from .cheb import cheb_pol
from .cheb import cheb_sum


from .cross import cross


from .fpe import fpe


from .grid import ind2poi
from .grid import ind2poi_cheb
from .grid import ind2str
from .grid import sample_lhs
from .grid import sample_tt
from .grid import str2ind


from .maxvol import maxvol
from .maxvol import maxvol_rect


from .stat import confidence
from .stat import get_cdf


from .svd import matrix_skeleton
from .svd import matrix_svd
from .svd import svd
from .svd import svd_incomplete


from .tensor import accuracy
from .tensor import add
from .tensor import add_many
from .tensor import copy
from .tensor import erank
from .tensor import full
from .tensor import get
from .tensor import getter
from .tensor import mean
from .tensor import mul
from .tensor import norm
from .tensor import orthogonalize
from .tensor import rand
from .tensor import show
from .tensor import sum
from .tensor import truncate


from .utils import core_one
