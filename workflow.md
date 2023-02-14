# workflow

> This is a draft version of the workflow instructions for teneva developers.


## How to install the current local version

1. Install [python](https://www.python.org) (version 3.8; you may use [anaconda](https://www.anaconda.com) package manager);

2. Create a virtual environment:
    ```bash
    conda create --name teneva python=3.8 -y
    ```

3. Activate the environment:
    ```bash
    conda activate teneva
    ```

4. Install special dependencies (for developers):
    ```bash
    pip install sphinx twine jupyterlab
    ```

5. Install teneva:
    ```bash
    python setup.py install
    ```

6. Reinstall teneva (after updates of the code):
    ```bash
    clear && pip uninstall teneva -y && python setup.py install
    ```

7. Delete virtual environment at the end of the work (optional):
    ```bash
    conda activate && conda remove --name teneva --all -y
    ```


## How to add a new function

1. Choose the most suitable module from `core` / `core_jax`

2. Choose the name for function in lowercase

3. Add new function in alphabetical order, separating it with two empty lines from neighboring functions

4. Add function in alphabetical order into `core/__init__.py` or `core_jax/__init__.py`

5. Make documentation for the function similar to other functions

6. Prepare a demo for a function similar to demos for other functions in the jupyter notebook with the same name as a module name (add it in alphabetical order)

7. Add function name into dict in docs `doc/map.py` and rebuild the docs (run `python doc/build.py`), check the result in web browser (see `doc/_build/html/index.html`)

8. Make commit like `[NEW](core.module.funcname) OPTIONAL_COMMENT` (see the next section with details)

9. Add related comment in `changelog.md` into subsection `NEW` for the upcoming version

10. Use it locally until update of the package version

> TODO: add item which relates to testing (for each demo should be also the corresponding autotest)


## How to make commits

For the convenience of tracking changes, it is worth following a certain structure in the naming of commits. The following style is proposed (draft):
```
KIND[func] OPTIONAL_COMMENT
```
For example, `UPG[core_jax.vis.show] Check that r <= n` (i.e., we added new features for the function `show` in the module `core_jax.vis`). The following possible values are suggested for the `KIND`:

- `GLB` - global changes (remove support of `python 3.6`, etc.). Example:
```
GLB[*] Add draft for workflow instructions for teneva developers
```

- `RNM` - renames of modules, functions, etc. Example:
```
RNM[core.tensors.tensor_const -> core.tensors.const] Since it is used very often
```

- `UPG` - upgrade of modules, functions, etc. Example:
```
UPG[core_jax.vis.show] Check that r <= n
```

- `NEW` - new modules, functions, etc. Example:
```
NEW[core_jax.act_one.get] Compute value of the TT-tensor in provided multi-index
```

- `DEM` - changes in demo (jupyter notebooks). Note that the assembly of the documentation must also be performed in this case (`python doc/build.py`). In square brackets, we indicate the corresponding function or module, but not the modified notebook itself. Example:
```
DEM[core_jax.vis.show] Add example for the case r <= n
```

- `FIX` - fixes for small bugs. Example:
```
FIX[core_jax.vis.show] Add mode size value for output
```

- `BUG` - fixes for big bugs. Example:
```
BUG[core_jax.vis.show] Remove invalid ...
```

- `STL` - fixes for style (pep, etc.) of functions and modules. Example:
```
STL[core.vis.show]
```

- `DOC` - updates for content of the docs (it is the text of the documentation, not the descriptions of functions and demonstrations in jupyter notebook)
```
GLB[*] Add description of the new jax version
```

- `DEV` - some code related to the development of new approaches, etc.
```
DEV[core.act_one.super_function] Try to integrate the tensor
```

Note that for "simple" commits we can merge the kinds like this:
```
(UPG,STL,DEM)[core.data.accuracy_on_data] Replace "getter" with "get_many"
```
or even like this:
```
(STL,DEM)[core.matrices,core.vectors] Minor stylistic changes and comments
```

> Note that the same structure should be used in sections of `changelog.md`


## How to update the package version

1. Add record in `changelog.md`

2. Run tests or `clear && python check/check.py` (TODO: add tests)

3. Update version (like `0.13.X`) in the file `teneva/__init__.py`

    > For breaking changes we should increase the major index (`13`), for non-breaking changes we should increase the minor index (`X`)

4. Build the docs `python doc/build.py`

5. Do commit `Update version (0.13.X)` and push

6. Upload new version to `pypi` (login: AndreiChertkov; passw: xxx)
    ```bash
    rm -r ./dist && python setup.py sdist bdist_wheel && twine upload dist/*
    ```

7. Reinstall
    ```bash
    pip install --no-cache-dir --upgrade teneva
    ```

8. Check the [teneva docs build](https://readthedocs.org/projects/teneva/builds/)

9. Check the [teneva docs site](https://teneva.readthedocs.io/)

> TODO: add standard for working with branches (`dev` branch?)
