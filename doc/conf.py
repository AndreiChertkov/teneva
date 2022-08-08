import os
import sys


sys.path.insert(0, os.path.abspath('..'))


project = 'teneva'
copyright = '2020-2022'
author = 'Andrei Chertkov'
language = 'en'
html_theme = 'alabaster'
html_favicon = '_static/favicon.ico'
html_theme_options = {
    'logo': 'favicon.ico',
    'logo_name': False,
    'page_width': '80%',
    'sidebar_width': '20%',
    'show_powered_by': False,
    'show_relbars': False,
    'extra_nav_links': {
        'Repository on github': 'https://github.com/AndreiChertkov/teneva',
    },
    'sidebar_collapse': True,
    'fixed_sidebar': False,
    'nosidebar': False,
}
extensions = [
    'sphinx.ext.imgmath',
    'sphinx.ext.graphviz',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.viewcode',
]
templates_path = [
    '_templates',
]
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
]
html_static_path = [
    '_static',
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

todo_include_todos = True
