"""Build the doc files with numerical examples from ipynb file.

Note:
    Run "sphinx-build ./doc ./doc/_build/html" after this module call.

"""
import json
import os


structure = {
    'core': {
        'als': {
            '_title': 'construct TT-tensor by TT-ALS',
            'als': True,
            'als2': True,
        },
        'anova': {
            '_title': 'construct TT-tensor by TT-ANOVA',
            'anova': True,
        },
        'cheb': {
            '_title': 'Chebyshev interpolation in the TT-format',
            'cheb_bld': True,
            'cheb_get': True,
            'cheb_get_full': True,
            'cheb_int': True,
            'cheb_pol': True,
            'cheb_sum': True,
        },
        'cross': {
            '_title': 'construct TT-tensor by TT-CAM',
            'cross': True,
        },
        'grid': {
            '_title': 'create and transform multidimensional grids',
            'grid_flat': True,
            'grid_prep_opts': True,
            'ind2poi': True,
            'ind2str': True,
            'sample_lhs': True,
            'sample_tt': True,
            'str2ind': True,
        },
        'maxvol': {
            '_title': 'compute the maximal-volume submatrix',
            'maxvol': True,
            'maxvol_rect': True,
        },
        'stat': {
            '_title': 'helper functions for processing statistics',
            'cdf_confidence': True,
            'cdf_getter': True,
        },
        'svd': {
            '_title': 'SVD-based algorithms for matrices and tensors',
            'matrix_skeleton': True,
            'matrix_svd': True,
            'svd': True,
            'svd_incomplete': True,
        },
        'tensor': {
            '_title': 'basic operations with TT-tensors',
            'accuracy': True,
            'add': True,
            'add_many': True,
            'const': True,
            'copy': True,
            'erank': True,
            'full': True,
            'get': True,
            'getter': True,
            'mean': True,
            'mul': True,
            'mul_scalar': True,
            'norm': True,
            'orthogonalize': True,
            'rand': True,
            'ranks': True,
            'shape': True,
            'show': True,
            'size': True,
            'sub': True,
            'sum': True,
            'truncate': True,
        }
    },
}


def find(data, func_name):
    func_pref = '### Function'
    func_str = f'{func_pref} `{func_name}`'

    i1, i2 = None, None
    for i, cell in enumerate(data['cells']):
        source = cell.get('source', [])
        if len(source) != 1:
            continue
        if source[0] == func_str:
            i1 = i
            continue
        elif i1 is not None and (func_pref in source[0] or '# Modules, functions and classes' in source[0] or '## Module `' in source[0]):
            i2 = i
            break

    if i1 is None or i2 is None:
        raise ValueError(f'Can not find function "{func_name}"')

    if i2 - i1 < 3:
        print(f'WRN !!! Empty function description "{func_name}"')
        return None

    res = []
    for i in range(i1+2, i2):
        cell = data['cells'][i]
        inp = cell.get('source', [])
        inp = ''.join(inp)

        out = cell.get('outputs', [])
        if len(out) > 0:
            out = out[0]
        if isinstance(out, list):
            out = '\n'.join(out)
        if isinstance(out, dict):
            out = out.get('text', [])
            out = ''.join(out)

        md = cell.get('cell_type') == 'markdown'

        res.append({'inp': inp, 'out': out, 'md': md})

    return res


def load():
    with open('./demo.ipynb', 'r') as f:
        data = json.load(f)
    return data


def run():
    data = load()

    for name_block, item_block in structure.items():
        fold_block = os.path.join('./doc/code', name_block)
        if not os.path.isdir(fold_block):
            os.mkdir(fold_block)
            print(f'The folder "{fold_block}" is created')

        res = ''
        res += name_block + '\n'
        res += '='*len(name_block) + '\n'
        res += '\n\n'
        res += '.. toctree::\n  :maxdepth: 4\n\n'
        res += '  ' + '\n  '.join(list(item_block.keys()))

        file_block = os.path.join(fold_block, 'index.rst')
        with open(file_block, 'w') as f:
            f.write(res)
        print(f'The file "{file_block}" is created')

        for name_module, item_module in item_block.items():
            title = name_module + ': ' + item_module['_title']

            res = ''
            res += title + '\n'
            res += '-'*len(title) + '\n'
            res += '\n\n'
            res += f'.. automodule:: teneva.{name_block}.{name_module}\n'
            res += '\n'

            for name_func, item_func in item_module.items():
                if not item_func or name_func[0] == '_':
                    continue

                res += '\n\n'
                res += '-----\n\n'
                res += f'.. autofunction:: '
                #res += f'teneva.{name_block}.{name_module}.{name_func}\n'
                res += f'teneva.{name_func}\n'

                cont = find(data, name_func)
                if cont and len(cont):
                    res += '\n  **Examples**:\n'
                for item in (cont or []):
                    res += '\n'
                    if item['md']:
                        res += '  ' + item['inp'].replace('\n', '\n  ')
                    else:
                        res += '  .. code-block:: python\n\n'
                        #res += '    :linenothreshold: 1\n\n'

                        res += '    ' + item['inp'].replace('\n', '\n    ')

                        if item['out']:

                            # res += '\n  .. code-block:: bash\n\n'
                            res += '\n\n'
                            res += '    # >>> ' + '-' * 40 + '\n'
                            res += '    # >>> Output:\n\n'
                            res += '    # ' + item['out'].replace('\n', '\n    # ')

                    res += '\n'

                res += '\n'

            file_module = os.path.join(fold_block, name_module + '.rst')
            with open(file_module, 'w') as f:
                f.write(res)
            print(f'The file "{file_module}" is created')


if __name__ == '__main__':
    run()
