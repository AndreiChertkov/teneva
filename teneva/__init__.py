__version__ = '0.14.6'


from .act_many import add_many
from .act_many import outer_many


from .act_one import copy
from .act_one import interface
from .act_one import get
from .act_one import get_and_grad
from .act_one import get_many
from .act_one import getter
from .act_one import mean
from .act_one import norm
from .act_one import qtt_to_tt
from .act_one import sum
from .act_one import tt_to_qtt


from .act_two import accuracy
from .act_two import add
from .act_two import mul
from .act_two import mul_scalar
from .act_two import outer
from .act_two import sub


from .als import als


from .als_func import als_func


from .anova import ANOVA
from .anova import anova


from .core import core_dot
from .core import core_dot_inv
from .core import core_dot_maxvol
from .core import core_qr_rand
from .core import core_qtt_to_tt
from .core import core_stab
from .core import core_tt_to_qtt


from .cross import cross


from .cross_act import cross_act


from .data import accuracy_on_data
from .data import cache_to_data


from .func import func_basis
from .func import func_diff_matrix
from .func import func_diff_matrix_apply
from .func import func_get
from .func import func_gets
from .func import func_int
from .func import func_int_general
from .func import func_sum


from .func_full import func_get_full
from .func_full import func_gets_full
from .func_full import func_int_full
from .func_full import func_sum_full


from .grid import grid_flat
from .grid import grid_prep_opt
from .grid import grid_prep_opts
from .grid import ind_qtt_to_tt
from .grid import ind_to_poi
from .grid import ind_tt_to_qtt
from .grid import poi_scale
from .grid import poi_to_ind


from .matrices import matrix_delta


from .maxvol import maxvol
from .maxvol import maxvol_rect


from .optima import optima_qtt
from .optima import optima_tt
from .optima import optima_tt_beam
from .optima import optima_tt_max


from .optima_func import optima_func_tt_beam


from .props import erank
from .props import ranks
from .props import shape
from .props import size


from .sample import sample
from .sample import sample_square
from .sample import sample_lhs
from .sample import sample_rand
from .sample import sample_tt


from .sample_func import sample_func


from .stat import cdf_confidence
from .stat import cdf_getter


from .svd import matrix_skeleton
from .svd import matrix_svd
from .svd import svd
from .svd import svd_matrix
from .svd import svd_incomplete


from .tensors import const
from .tensors import delta
from .tensors import poly
from .tensors import rand
from .tensors import rand_custom
from .tensors import rand_norm


from .transformation import full
from .transformation import full_matrix
from .transformation import orthogonalize
from .transformation import orthogonalize_left
from .transformation import orthogonalize_right
from .transformation import truncate


from .utils import _info_appr
from .utils import _is_num
from .utils import _maxvol
from .utils import _ones
from .utils import _rand
from .utils import _range
from .utils import _reshape
from .utils import _vector_index_expand
from .utils import _vector_index_prepare


from .vectors import vector_delta


from .vis import show
