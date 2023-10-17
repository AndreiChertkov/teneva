"""List of all modules/functions/classes for documentation.

Note:
    Use "True" for functions and "False" for classes.

"""
MAP = {
    'modules': {
        'act_one': {
            'title': 'single TT-tensor operations',
            'items': {
                'copy': True,
                'get': True,
                'get_and_grad': True,
                'get_many': True,
                'getter': True,
                'interface': True,
                'mean': True,
                'norm': True,
                'qtt_to_tt': True,
                'sum': True,
                'tt_to_qtt': True,
            },
        },
        'act_two': {
            'title': 'operations with a pair of TT-tensors',
            'items': {
                'accuracy': True,
                'add': True,
                'mul': True,
                'mul_scalar': True,
                'outer': True,
                'sub': True,
            },
        },
        'act_many': {
            'title': 'operations with a set of TT-tensors',
            'items': {
                'add_many': True,
                'outer_many': True,
            },
        },
        'als': {
            'title': 'construct TT-tensor by TT-ALS',
            'items': {
                'als': True,
            },
        },
        'als_func': {
            'title': 'construct TT-tensor of coefficients',
            'items': {
                'als_func': True,
            },
        },
        'anova': {
            'title': 'construct TT-tensor by TT-ANOVA',
            'items': {
                'anova': True,
            },
        },
        'anova_func': {
            'title': 'construct TT-tensor of interpolation coefs by TT-ANOVA',
            'items': {
                'anova_func': True,
            },
        },
        'core': {
            'title': 'operations with individual TT-cores',
            'items': {
                'core_qtt_to_tt': True,
                'core_stab': True,
                'core_tt_to_qtt': True,
            },
        },
        'cross': {
            'title': 'construct TT-tensor by TT-cross',
            'items': {
                'cross': True,
            },
        },
        'cross_act': {
            'title': 'compute user-specified function of TT-tensors',
            'items': {
                'cross_act': True,
            },
        },
        'data': {
            'title': 'functions for working with datasets',
            'items': {
                'accuracy_on_data': True,
                'cache_to_data': True,
            },
        },
        'func': {
            'title': 'Functional TT-format including Chebyshev interpolation',
            'items': {
                'func_basis': True,
                'func_diff_matrix': True,
                'func_get': True,
                'func_gets': True,
                'func_int': True,
                'func_int_general': True,
                'func_sum': True,
            },
        },
        'func_full': {
            'title': 'Functional full format including Chebyshev interpolation',
            'items': {
                'func_get_full': True,
                'func_gets_full': True,
                'func_int_full': True,
                'func_sum_full': True,
            },
        },
        'grid': {
            'title': 'create and transform multidimensional grids',
            'items': {
                'grid_flat': True,
                'grid_prep_opt': True,
                'grid_prep_opts': True,
                'ind_qtt_to_tt': True,
                'ind_to_poi': True,
                'ind_tt_to_qtt': True,
                'poi_scale': True,
                'poi_to_ind': True,
            },
        },
        'matrices': {
            'title': 'collection of explicit useful QTT-matrices',
            'items': {
                'matrix_delta': True,
            },
        },
        'maxvol': {
            'title': 'compute the maximal-volume submatrix',
            'items': {
                'maxvol': True,
                'maxvol_rect': True,
            },
        },
        'optima': {
            'title': 'estimate min and max value of the tensor',
            'items': {
                'optima_qtt': True,
                'optima_tt': True,
                'optima_tt_beam': True,
                'optima_tt_max': True,
            },
        },
        'optima_func': {
            'title': 'estimate max for function',
            'items': {
                'optima_func_tt_beam': True,
            },
        },
        'props': {
            'title': 'various simple properties of TT-tensors',
            'items': {
                'erank': True,
                'ranks': True,
                'shape': True,
                'size': True,
            },
        },
        'sample': {
            'title': 'random sampling for/from the TT-tensor',
            'items': {
                'sample': True,
                'sample_square': True,
                'sample_lhs': True,
                'sample_rand': True,
                'sample_rand_poi': True,
                'sample_tt': True,
            }
        },
        'sample_func': {
            'title': 'random sampling from the functional TT-tensor',
            'items': {
                'sample_func': True,
            }
        },
        'stat': {
            'title': 'helper functions for processing statistics',
            'items': {
                'cdf_confidence': True,
                'cdf_getter': True,
            }
        },
        'svd': {
            'title': 'SVD-based algorithms for matrices and tensors',
            'items': {
                'matrix_skeleton': True,
                'matrix_svd': True,
                'svd': True,
                'svd_matrix': True,
                'svd_incomplete': True,
            },
        },
        'tensors': {
            'title': 'collection of explicit useful TT-tensors',
            'items': {
                'const': True,
                'delta': True,
                'poly': True,
                'rand': True,
                'rand_custom': True,
                'rand_norm': True,
                'rand_stab': True,
            },
        },
        'transformation': {
            'title': 'orthogonalization, truncation and other transformations of the TT-tensors',
            'items': {
                'full': True,
                'full_matrix': True,
                'orthogonalize': True,
                'orthogonalize_left': True,
                'orthogonalize_right': True,
                'truncate': True,
            },
        },
        'vectors': {
            'title': 'collection of explicit useful QTT-vectors',
            'items': {
                'vector_delta': True,
            },
        },
        'vis': {
            'title': 'visualization methods for tensors',
            'items': {
                'show': True,
            },
        },
    },
}
