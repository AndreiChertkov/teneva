"""List of all modules/functions/classes for documentation.

Note:
    Use "True" for functions and "False" for classes.

"""
MAP = {
    'title': 'code',
    'modules': {
        'core': {
            'title': 'implementation of basic operations in the TT-format',
            'modules': {
                'act_one': {
                    'title': 'single TT-tensor operations',
                    'items': {
                        'copy': True,
                        'get': True,
                        'get_many': True,
                        'getter': True,
                        'mean': True,
                        'norm': True,
                        'sum': True,
                    },
                },
                'act_two': {
                    'title': 'operations with a pair of TT-tensors',
                    'items': {
                        'accuracy': True,
                        'add': True,
                        'mul': True,
                        'mul_scalar': True,
                        'sub': True,
                    },
                },
                'act_many': {
                    'title': 'operations with a set of TT-tensors',
                    'items': {
                        'add_many': True,
                    },
                },
                'als': {
                    'title': 'construct TT-tensor by TT-ALS',
                    'items': {
                        'als': True,
                        'als2': True,
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
                        'cheb_bld': True,
                        'cheb_get': True,
                        'cheb_gets': True,
                        'cheb_int': True,
                        'cheb_pol': True,
                        'cheb_sum': True,
                    },
                },
                'cheb_full': {
                    'title': 'Chebyshev interpolation in the full format',
                    'items': {
                        'cheb_bld_full': True,
                        'cheb_get_full': True,
                        'cheb_gets_full': True,
                        'cheb_int_full': True,
                        'cheb_sum_full': True,
                    },
                },
                'core': {
                    'title': 'operations with individual TT-cores',
                    'items': {
                        'core_stab': True,
                        'core_tt_to_qtt': True,
                    },
                },
                'cross': {
                    'title': 'construct TT-tensor by TT-CROSS',
                    'items': {
                        'cross': True,
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
                        'ind_to_poi': True,
                        'poi_to_ind': True,
                        'sample_lhs': True,
                        'sample_tt': True,
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
                    'title': 'estimate min and max value of tensor',
                    'items': {
                        #'optima_tt': True,
                        #'optima_tt_max': True,
                    },
                },
                'props': {
                    'title': 'various properties (mean, norm, etc.) of TT-tensors',
                    'items': {
                        'erank': True,
                        'ranks': True,
                        'shape': True,
                        'size': True,
                    },
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
                        'svd_incomplete': True,
                    },
                },
                'tensor': {
                    'title': 'basic operations with TT-tensors',
                    'items': {
                        'rand': True,
                    },
                },
                'transformation': {
                    'title': 'orthogonalization, truncation and other transformations of the TT-tensors',
                    'items': {
                        'full': True,
                        'orthogonalize': True,
                        'orthogonalize_left': True,
                        'orthogonalize_right': True,
                        'truncate': True,
                    },
                },
                'vis': {
                    'title': 'visualization methods for tensors',
                    'items': {
                        'show': True,
                    },
                },
            },
        },
        'collection': {
            'title': 'collection of various explicit TT/QTT-tensors',
            'modules': {
                'tensors': {
                    'title': 'collection of explicit useful TT-tensors',
                    'items': {
                        'tensor_const': True,
                        'tensor_delta': True,
                        'tensor_poly': True,
                    },
                },
                'vectors': {
                    'title': 'collection of explicit useful QTT-vectors',
                    'items': {
                        'vector_delta': True,
                    },
                },
                'matrices': {
                    'title': 'collection of explicit useful QTT-matrices',
                    'items': {
                        'matrix_delta': True,
                    },
                },
            },
        },
        'func': {
            'title': 'wrapper class for functions and benchmarks',
            'modules': {
                'func': {
                    'title': 'wrapper for multivariable function with approximation methods',
                    'items': {
                        'func': False,
                    },
                },
                'demo': {
                    'title': 'analytical functions for demo and tests',
                    'modules': {
                        'func_demo_ackley': {
                            'title': 'Ackley function for demo and tests',
                            'items': {
                                'func_demo_ackley': False,
                            },
                        },
                        'func_demo_alpine': {
                            'title': 'Alpine function for demo and tests',
                            'items': {
                                'func_demo_alpine': False,
                            },
                        },
                        'func_demo_dixon': {
                            'title': 'Dixon function for demo and tests',
                            'items': {
                                'func_demo_dixon': False,
                            },
                        },
                        'func_demo_exponential': {
                            'title': 'Exponential function for demo and tests',
                            'items': {
                                'func_demo_exponential': False,
                            },
                        },
                        'func_demo_grienwank': {
                            'title': 'Grienwank function for demo and tests',
                            'items': {
                                'func_demo_grienwank': False,
                            },
                        },
                        'func_demo_michalewicz': {
                            'title': 'Michalewicz function for demo and tests',
                            'items': {
                                'func_demo_michalewicz': False,
                            },
                        },
                        'func_demo_piston': {
                            'title': 'Piston function for demo and tests',
                            'items': {
                                'func_demo_piston': False,
                            },
                        },
                        'func_demo_qing': {
                            'title': 'Qing function for demo and tests',
                            'items': {
                                'func_demo_qing': False,
                            },
                        },
                        'func_demo_rastrigin': {
                            'title': 'Rastrigin function for demo and tests',
                            'items': {
                                'func_demo_rastrigin': False,
                            },
                        },
                        'func_demo_rosenbrock': {
                            'title': 'Rosenbrock function for demo and tests',
                            'items': {
                                'func_demo_rosenbrock': False,
                            },
                        },
                        'func_demo_schaffer': {
                            'title': 'Schaffer function for demo and tests',
                            'items': {
                                'func_demo_schaffer': False,
                            },
                        },
                        'func_demo_schwefel': {
                            'title': 'Schwefel function for demo and tests',
                            'items': {
                                'func_demo_schwefel': False,
                            },
                        },
                    },
                },
                'utils': {
                    'title': 'helper methods to build model functions',
                    'items': {
                        'func_demo': True,
                        'func_demo_all': True,
                    },
                },
            },
        },
    },
}
