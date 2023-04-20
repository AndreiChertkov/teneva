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
                'tt_to_func': True,
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
        'als_contin': {
            'title': 'construct TT-tensor of coefficients',
            'items': {
                'als_contin': True,
            },
        },
        'anova': {
            'title': 'construct TT-tensor by TT-ANOVA',
            'items': {
                'anova': True,
            },
        },
        'cheb': {
            'title': 'Chebyshev interpolation in the TT-format',
            'items': {
                'cheb_diff_matrix': True,
                'cheb_get': True,
                'cheb_gets': True,
                'cheb_int': True,
                'cheb_pol': True,
                'cheb_sum': True,
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
            'title': 'compute function of TT-tensors (draft!)',
            'items': {
                'cross_act': True,
            },
        },
        'data': {
            'title': 'functions for working with datasets',
            'items': {
                'accuracy_on_data': True,
            },
        },
        'grid': {
            'title': 'create and transform multidimensional grids',
            'items': {
                'cache_to_data': True,
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
            'title': 'collection of explicit useful QTT-matrices (draft)',
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
        'optima_contin': {
            'title': 'estimate max for function',
            'items': {
                'optima_contin_tt_beam': True,
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
                'sample_tt': True,
            }
        },
        #'sample_contin': {
        #    'title': 'random sampling from the functional TT-tensor',
        #    'items': {
        #        'sample_contin': True,
        #    }
        #},
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
            'title': 'collection of explicit useful QTT-vectors (draft)',
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
