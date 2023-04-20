# workflow

> Workflow instructions for `teneva` developers.


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

7. Rebuild the docs (after updates of the code):
    ```bash
    python doc/build.py
    ```

8. Delete virtual environment at the end of the work (optional):
    ```bash
    conda activate && conda remove --name teneva --all -y
    ```


## How to add a new function

1. Choose the most suitable module from `teneva` folder

2. Choose the name for function in lowercase

3. Add new function in alphabetical order, separating it with two empty lines from neighboring functions

4. Add function in alphabetical order into `__init__.py`

5. Make documentation (i.e., `docstring`) for the function similar to other functions

6. Prepare a demo for a function (jupyter notebook in the `demo` folder) similar to demos for other functions in the jupyter notebook with the same name as a module name (add it in alphabetical order)
    > Then demo is ready, run `Restart Kernel and Run All Cells` and save the notebook to make sure that the sequence of cells executed in the correct order will be included in the commit (due to this, git will maybe not commit changes to the file on subsequent runs). Note that it's important to use a consistent style for all functions, as the code is then automatically exported from the jupyter notebooks to assemble the online documentation.

7. Add function name into dict in docs `doc/map.py` and rebuild the docs (run `python doc/build.py`), check the result in web browser (see `doc/_build/html/index.html`)

8. Make commit like `[NEW](module.funcname) OPTIONAL_COMMENT` (see the next section with details of commit's message style)

9. Add related comment in `changelog.md` (with the tag `NEW`) for the upcoming version

10. Use it locally until update of the package version

> TODO: add item which relates to testing (for each demo should be also the corresponding autotest)


## How to make commits

For the convenience of tracking changes, it is worth following a certain structure in the naming of commits. The following style is proposed (draft):
```
[KIND](func) OPTIONAL_COMMENT
```
For example, `[UPG](vis.show) Check that r <= n` (i.e., we added new features for the function `show` in the module `vis`).

The following possible values are suggested for the `KIND`:

- `GLB` - global changes (remove support of `python 3.6`, etc.). Example:
    ```
    [GLB] Add draft for workflow instructions for teneva developers
    ```

- `RNM` - renames of existing modules, functions, etc. Example:
    ```
    [RNM](tensors.tensor_const -> core.tensors.const) Since it is used very often
    ```

- `UPG` - upgrade of existing modules, functions, etc. Example:
    ```
    [UPG](vis.show) Check that r <= n
    ```

- `NEW` - new modules, functions, etc. Example:
    ```
    [NEW](act_one.get) Compute value of the TT-tensor in provided multi-index
    ```

- `DEM` - changes in demo (jupyter notebooks). Note that the assembly of the documentation must also be performed in this case (`python doc/build.py`). In the brackets, we indicate the corresponding function or module, but not the modified notebook itself. Example:
    ```
    [DEM](vis.show) Add example for the case r <= n
    ```

- `FIX` - fixes for small bugs. Example:
    ```
    [FIX](vis.show) Add mode size value for output
    ```

- `BUG` - fixes for big bugs. Example:
    ```
    [BUG](vis.show) Remove invalid ...
    ```

- `STL` - fixes for style (pep, etc.) of functions and modules. Example:
    ```
    [STL](vis.show) More accurate docstring
    ```

- `DOC` - updates for content of the docs (it is the text of the documentation, not the descriptions (docstrings) of functions and demonstrations in jupyter notebook)
    ```
    [DOC] Add link to the jax repo
    ```

- `DEV` - some code related to the development of new approaches, etc.
    ```
    [DEV](act_one.super_function) Try to integrate the tensor
    ```

Note that for "simple" commits we can merge the kinds like this:
```
[UPG, STL, DEM](data.accuracy_on_data) Replace "getter" with "get_many"
```
or even like this:
```
[STL, DEM](matrices, vectors) Minor stylistic changes and comments
```

> Note that the same tag names should be used in the `changelog.md`


## How to update the package version

1. Check and modify record in `changelog.md` (remove `upcoming` tag)

2. Run tests or `clear && python check/check.py` (TODO: add tests)

3. Update version (like `0.14.X`) in the file `teneva/__init__.py`

    > For breaking changes we should increase the major index (`14`), for non-breaking changes we should increase the minor index (`X`)

4. Build the docs `python doc/build.py`

5. Do commit `Update version (0.14.X)` and push

6. Upload new version to `pypi` (login: AndreiChertkov; passw: xxx)
    ```bash
    rm -r ./dist && python setup.py sdist bdist_wheel && twine upload dist/*
    ```

7. Reinstall and check that installed version is new
    ```bash
    pip install --no-cache-dir --upgrade teneva
    ```

8. Check the [teneva docs build](https://readthedocs.org/projects/teneva/builds/)

9. Check the [teneva docs site](https://teneva.readthedocs.io/)

> TODO: add standard for working with branches (`dev` branch?)
