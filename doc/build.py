"""Build the doc files with numerical examples from ipynb-files.

Run this module as "python doc/build.py" from the root of the project. The
prepared documentation in the html-format will be in the "doc/_build/html"
folder.

Note:
    This script will generate subfolders in the "doc/code" folders with
    documentation for all modules/functions/classes from the "MAP" dictionary
    (see "map.py" file). At the same time, "doc/code/index.rst" file should be
    prepared manually.

"""
import json
import os
import re
import shutil
import subprocess


from map import MAP


# Package name:
PACK = 'teneva'


def build_item(name, is_func, tree=[]):
    cont = parse_jupyter(load_jupyter(tree), name, is_func)
    text = ''

    if is_func:
        meth = name
    else:
        meth = ''.join(x.capitalize() or '_' for x in name.split('_'))

    text += '.. autofunction:: ' if is_func else '.. autoclass:: '
    text += f'{PACK}.{meth}\n'
    if not is_func:
        text += '  :members: \n'

    sp = '  ' if is_func else ''
    text += '\n' + sp +  '**Examples**:\n'
    for item in (cont or []):
        text += '\n'
        if item['md']:
            text += sp + item['inp'].replace('\n', '\n  ' + sp)
        else:
            text += sp + '.. code-block:: python\n\n'
            text += sp + '  ' + item['inp'].replace('\n', '\n  ' + sp)
            if item['out']:
                text += '\n\n'
                text += sp + '  # >>> ' + '-' * 40 + '\n'
                text += sp + '  # >>> Output:\n\n'
                text += sp + '  # '
                text += item['out'].replace('\n', '\n  ' + sp + '# ')
            if item['img']:
                # TODO: Add support for image
                text += '\n\n'
                text += sp + '  # >>> ' + '-' * 40 + '\n'
                text += sp + '  # >>> Output:\n\n'
                text += sp + '  # '
                text += 'Display of images is not supported in the docs.'
                text += ' See related ipynb file.'
        text += '\n'

    text += '\n\n' + '-----\n\n\n'
    text = text[:-8]

    return text


def build_module(obj, name, tree=[]):
    tree = tree + [name]
    title = 'Module ' + name + ': ' + obj.get('title', {})
    modules = obj.get('modules')
    items = obj.get('items')

    if 'modules' in obj:
        create_folder('/'.join(tree))
        create_index(title, modules, tree)

        for name, item in obj['modules'].items():
            build_module(item, name, tree)

    elif 'items' in obj:
        text = title + '\n' + '-'*len(title) + '\n\n\n'
        text += f'.. automodule:: {PACK}.{".".join(tree)}\n\n\n-----\n\n\n'

        if len(list(obj['items'])) == 0:
            text += '\n\n TODO \n\n'
        else:
            for name, is_func in obj['items'].items():
                text += build_item(name, is_func, tree)

        with open('./doc/' + '/'.join(tree) + '.rst', 'w') as f:
            f.write(text)

    else:
        raise ValueError('Invalid map for module')


def build():
    for name, item in MAP.get('modules', {}).items():
        shutil.rmtree(f'./doc/{name}', ignore_errors=True)
        build_module(item, name)

    build_version()


def create_folder(fold):
    fold = os.path.join('./doc', fold)

    if not os.path.isdir(fold):
        os.mkdir(fold)


def create_index(name, children, tree):
    res = name + '\n' + '='*len(name) + '\n\n\n'

    res += '.. toctree::\n  :maxdepth: 4\n\n'
    for name_item, item in (children or {}).items():
        link = name_item
        if 'modules' in item:
            link += '/index'
        res += '  ' + link + '\n'

    fpath = os.path.join('./doc', '/'.join(tree), 'index.rst')
    with open(fpath, 'w') as f:
        f.write(res)


def build_version():
    with open(f'./{PACK}/__init__.py', 'r', encoding='utf-8') as f:
        text = f.read()
        version = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", text, re.M)
        version = version.group(1)

    # Add version into docs index page:

    with open('./doc/index.rst', 'r') as f:
        text = f.read()

    version_old = text.split('Current version ')[1].split('"')[1]
    text = text.replace(version_old, version)

    with open('./doc/index.rst', 'w') as f:
        f.write(text)

    # Add version into README.md file:

    # TODO

    return


def load_jupyter(tree):
    with open('./demo/' + '/'.join(tree) + '.ipynb', 'r') as f:
        data = json.load(f)
    return data


def parse_jupyter(data, name, is_func=True):
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


if __name__ == '__main__':
    build()
    print(f'\n\n>>> The documentation rst-files are prepared by script\n\n')

    cmd = 'sphinx-build ./doc ./doc/_build/html'
    res = subprocess.run(cmd, shell=True)
    print(f'\n\n>>> The html-documentation is built by sphinx\n\n')
