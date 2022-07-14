"""Build the doc files with numerical examples from ipynb-files.

Run this module as "python doc/build.py" from the root of the project. The
prepared documentation in the html-format will be in the "doc/_build/html"
folder.

Note:
    This script will generate subfolders in the "doc/code" folders with
    documentation for all modules/functions/classes from the "structure"
    dictionary. At the same time,  "doc/code/index.rst" file should be prepared
    manually.

"""
import json
import os
import re
import subprocess


# Package name:
PACK = 'teneva'


# List of all modules/functions/classes for documentation
# (use "True" for functions and "False" for classes):
structure = {
    'collection': {
        'matrices': {
            '_title': 'collection of explicit useful QTT-matrices',
            'matrix_delta': True,
        },
        'tensors': {
            '_title': 'collection of explicit useful TT-tensors',
            'tensor_const': True,
            'tensor_delta': True,
            'tensor_poly': True,
        },
        'vectors': {
            '_title': 'collection of explicit useful QTT-vectors',
            'vector_delta': True,
        },
    },
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
            'cheb_gets': True,
            'cheb_int': True,
            'cheb_pol': True,
            'cheb_sum': True,
        },
        'cheb_full': {
            '_title': 'Chebyshev interpolation in the full format',
            'cheb_bld_full': True,
            'cheb_get_full': True,
            'cheb_gets_full': True,
            'cheb_int_full': True,
            'cheb_sum_full': True,
        },
        'cross': {
            '_title': 'construct TT-tensor by TT-CROSS',
            'cross': True,
        },
        'grid': {
            '_title': 'create and transform multidimensional grids',
            'cache_to_data': True,
            'grid_flat': True,
            'grid_prep_opt': True,
            'grid_prep_opts': True,
            'ind_to_poi': True,
            'poi_to_ind': True,
            'sample_lhs': True,
            'sample_tt': True,
        },
        'maxvol': {
            '_title': 'compute the maximal-volume submatrix',
            'maxvol': True,
            'maxvol_rect': True,
        },
        'optima': {
            '_title': 'estimate min and max value of tensor',
            'optima_tt': True,
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
            'accuracy_on_data': True,
            'add': True,
            'add_many': True,
            'copy': True,
            'erank': True,
            'full': True,
            'get': True,
            'getter': True,
            'mean': True,
            'mul': True,
            'mul_scalar': True,
            'norm': True,
            'rand': True,
            'ranks': True,
            'shape': True,
            'show': True,
            'size': True,
            'sub': True,
            'sum': True,
        },
        'transformation': {
            '_title': 'orthogonalization and truncation of TT-tensors',
            'orthogonalize': True,
            'orthogonalize_left': True,
            'orthogonalize_right': True,
            'truncate': True,
        }
    },
    'func': {
        'func': {
            '_title': 'wrapper for multivariable function with approximation methods',
            'func': False,
        },
        'demo': {
            '_title': 'analytical functions for demo and tests',
            '_virt': True,
            'func_demo': True,
            'func_demo_all': True,
            'func_demo_ackley': False,
            'func_demo_alpine': False,
            'func_demo_dixon': False,
            'func_demo_exponential': False,
            'func_demo_grienwank': False,
            'func_demo_michalewicz': False,
            'func_demo_piston': False,
            'func_demo_qing': False,
            'func_demo_rastrigin': False,
            'func_demo_rosenbrock': False,
            'func_demo_schaffer': False,
            'func_demo_schwefel': False,
        }
    }
}


def build():
    build_version()
    for name_block, item_block in structure.items():
        fold_block = build_block(name_block, item_block)
        for name_module, item_module in item_block.items():
            build_module(name_module, item_module, name_block, fold_block)


def build_block(name, item):
    fold = os.path.join('./doc/code', name)
    if not os.path.isdir(fold):
        os.mkdir(fold)
        print(f'The folder "{fold}" is created')

    res = name + '\n' + '='*len(name) + '\n\n\n'
    res += '.. toctree::\n  :maxdepth: 4\n\n'
    res += '  ' + '\n  '.join(list(item.keys()))

    fpath = os.path.join(fold, 'index.rst')
    with open(fpath, 'w') as f:
        f.write(res)
    print(f'The file "{fpath}" is created')

    return fold


def build_module(name, item, name_block, fold):
    data = load(name_block, name)

    title = name + ': ' + item.get('_title', 'module with code')

    res = title + '\n' + '-'*len(title) + '\n\n\n'
    if not item.get('_virt'):
        res += f'.. automodule:: {PACK}.{name_block}.{name}\n\n\n-----\n\n\n'

    for name_obj, is_func in item.items():
        if name_obj[0] == '_':
            continue

        if is_func:
            name_obj_disp = name_obj
        else:
            name_obj_disp = ''.join(x.capitalize() or '_'
                for x in name_obj.split('_'))

        res += '.. autofunction:: ' if is_func else '.. autoclass:: '
        res += f'{PACK}.{name_obj_disp}\n'
        if not is_func:
            res += '  :members: \n'

        cont = find(data, name_obj, is_func)
        if cont and len(cont):
            sp = '  ' if is_func else ''
            res += '\n' + sp +  '**Examples**:\n'
            for item in (cont or []):
                res += '\n'
                if item['md']:
                    res += sp + item['inp'].replace('\n', '\n  ' + sp)
                else:
                    res += sp + '.. code-block:: python\n\n'
                    res += sp + '  ' + item['inp'].replace('\n', '\n  ' + sp)
                    if item['out']:
                        res += '\n\n'
                        res += sp + '  # >>> ' + '-' * 40 + '\n'
                        res += sp + '  # >>> Output:\n\n'
                        res += sp + '  # '
                        res += item['out'].replace('\n', '\n  ' + sp + '# ')
                    if item['img']:
                        # TODO: Add support for image
                        res += '\n\n'
                        res += sp + '  # >>> ' + '-' * 40 + '\n'
                        res += sp + '  # >>> Output:\n\n'
                        res += sp + '  # '
                        res += 'Display of images is not supported in the docs.'
                        res += ' See related ipynb file.'
                res += '\n'

        res += '\n\n' + '-----\n\n\n'
    res = res[:-8]

    fpath = os.path.join(fold, name + '.rst')
    with open(fpath, 'w') as f:
        f.write(res)
    print(f'The file "{fpath}" is created')


def build_version():
    with open(f'./{PACK}/__init__.py', 'r', encoding='utf-8') as f:
        text = f.read()
        version = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", text, re.M)
        version = version.group(1)

    with open('./doc/index.rst', 'r') as f:
        text = f.read()

    version_old = text.split('Current version ')[1].split('"')[1]
    text = text.replace(version_old, version)

    with open('./doc/index.rst', 'w') as f:
        f.write(text)


def find(data, name, is_func=True):
    name_pref = '## Function' if is_func else '## Class'
    name_pref_alt = '## Function' if not is_func else '## Class'
    if not is_func:
        name = ''.join(x.capitalize() or '_' for x in name.split('_'))
    name_str = f'{name_pref} `{name}`'

    i1, i2 = None, None
    for i, cell in enumerate(data['cells']):
        source = cell.get('source', [])
        if len(source) != 1:
            continue

        if source[0] == name_str:
            i1 = i
        elif i1 is not None:
            is_end = name_pref in source[0] or name_pref_alt in source[0]
            is_end = is_end or source[0] == '---'
            if is_end:
                i2 = i
                break

    if i1 is None or i2 is None:
        raise ValueError(f'Can not find function/class "{name}"')
    if i2 - i1 < 3:
        raise ValueError(f'Empty function/class description "{name}"')

    res = []
    for i in range(i1+2, i2):
        cell = data['cells'][i]
        inp = cell.get('source', [])
        inp = ''.join(inp)

        out = cell.get('outputs', [])
        img = ''
        if len(out) > 0:
            out = out[0]
        if isinstance(out, list):
            out = '\n'.join(out)
        if isinstance(out, dict):
            img = out.get('data', {}).get('image/png', '')
            if 'text' in out:
                out = out.get('text')
            else:
                out = out.get('data', {}).get('text/plain', [])
                out.append('\n')
            out = ''.join(out or [])

        md = cell.get('cell_type') == 'markdown'

        res.append({'inp': inp, 'out': out, 'img': img, 'md': md})

    return res


def load(name, name_item):
    with open(f'./demo/{name}_{name_item}.ipynb', 'r') as f:
        data = json.load(f)
    return data


if __name__ == '__main__':
    build()
    print(f'\n\n>>> The documentation rst-files are prepared by script\n\n')

    cmd = 'sphinx-build ./doc ./doc/_build/html'
    res = subprocess.run(cmd, shell=True)
    print(f'\n\n>>> The html-documentation is built by sphinx\n\n')
