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

4. Install special dependencies (for developers only):
    ```bash
    pip install sphinx twine jupyterlab
    ```
    > You will also need `pip install numba==0.57.1` only for the function `act_one.getter`.

5. Switch to the `dev` branch and pull:
    ```bash
    git checkout dev && git pull origin dev
    ```

6. Install `teneva` from the source:
    ```bash
    python setup.py install
    ```

7. Reinstall `teneva` from the source (after updates of the code):
    ```bash
    clear && pip uninstall teneva -y && python setup.py install
    ```

8. Rebuild the docs (after updates of the code):
    ```bash
    python doc/build.py
    ```

9. Run all the tests:
    ```bash
    python test/test.py
    ```

10. Optionally delete the virtual environment at the end of the work:
    ```bash
    conda activate && conda remove --name teneva --all -y
    ```


## How to add a new function

> Note that we carry out the entire development process in the `dev` branch; when we are ready to release, we merge it into the master branch.

1. Choose the most suitable module from `teneva` folder;

2. Choose the name for the new function in lowercase;

3. Add the new function in alphabetical order, separating it with two empty lines from neighboring functions;

4. Add the new function import in alphabetical order into `__init__.py`;

5. Make documentation (i.e., `docstring`) for the new function similar to other functions (note that we use [the guide from google](https://google.github.io/styleguide/pyguide.html));

6. Prepare a demo for the new function (add it in alphabetical order) in the related jupyter notebook (with the same name as a module name) in the `demo` folder similar to demos for other functions in the jupyter notebook with the same name as a module name;
    > Note that it's important to use a consistent style for all functions, as the code is then automatically exported from the jupyter notebooks to assemble the online documentation.

7. Add function name into the dict in docs `doc/map.py` and rebuild the docs (run `python doc/build.py`), check the result in the web browser (see `doc/_build/html/index.html`);

8. [For now, this item can be skipped.] Prepare tests for the new function in the corresponding module inside the `test` folder, and then run all the tests `python test/test.py`;

9. Make commit like `[NEW](MODULE.FUNCTION) OPTIONAL_COMMENT` (see the next section with details of commit's message style);

10. [For now, this item can be skipped.] Add related comment in `changelog.md` (with the tag `NEW`) for the upcoming version;

11. Use the new function locally until update of the package version.


## How to make commits

> Note that we carry out the entire development process in the `dev` branch; when we are ready to release, we merge it into the master branch. Before you commit, please check that you are on the `dev` branch (`git branch -a`).

For the convenience of tracking changes, it is worth following a certain structure in the naming of commits. The following style is proposed:
```
[KIND](func) OPTIONAL_COMMENT
```
```
[KIND1, KIND2](func1, func2, func3) OPTIONAL_COMMENT
```
For example, `[UPG](vis.show) Check that r <= n` (i.e., we added new features for the function `show` in the module `vis`) or `[UPG, STL, DEM](data.accuracy_on_data) Replace "getter" with "get_many"` or even like this: `[STL, DEM](matrices, vectors) Minor stylistic changes and comments`.

The following possible values are suggested for the `KIND`:

- `GLB` - global changes (remove support of `python 3.6`, changing the behavior logic of a large group of functions, etc.). The name of the files in parentheses can be omitted in this case. Example:
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

- `TST` - new tests (or its updates) for modules and functions. Example:
    ```
    [TST](vis.show) Add special tests
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

> Note that the same tag names should be used in the `changelog.md`.


## How to add new tests for function

1. Check that you are on the `dev` branch as `git branch -a`;

2. Select the function `MODULE.FUNCTION` from `teneva`;

3. Check or (optionally) update the style of the function's code (in this case run `clear && pip uninstall teneva -y && python setup.py install`);

4. Check or (optionally) update the function's demo and rerun the demo jupyter;

5. Update the docs as `clear && python doc/build.py`;

6. Write the tests for function in `test/test_MODULE.py` file;

7. Run the tests as `clear && python test/test.py`;

8. Do commit like `[STL, DEM, TST](MODULE.FUNCTION)`.


## How to work with git branches

> Note that we carry out the entire development process in the `dev` branch; when we are ready to release, we merge it into the master branch.

1. Check existing branches:
    ```bash
    git branch -a
    ```

2. Optionally delete the `dev` branch:
    ```bash
    git branch --delete dev
    ```

3. Create a new `dev` branch:
    ```bash
    git branch dev
    ```

4. Select the `dev` branch as a current:
    ```bash
    git checkout dev
    ```

5. Sometimes merge the `dev` branch with the `master`:
    ```bash
    git checkout dev && git merge master
    ```

6. Push the branch `dev` after commits:
    ```bash
    git checkout dev && git push origin dev
    ```


## How to update the package version

1. Pull the `master` branch:
    ```bash
    git checkout master && git pull origin master
    ```

2. Pull the `dev` branch:
    ```bash
    git checkout dev && git pull origin dev
    ```

3. Merge the `dev` branch with the `master` (`master -> dev`):
    ```bash
    git checkout dev && git merge master
    ```

4. Reinstall teneva locally:
    ```bash
    clear && pip uninstall teneva -y && python setup.py install
    ```

5. Run all the tests:
    ```bash
    clear && python test/test.py
    ```

6. Build the docs:
    ```bash
    clear && python doc/build.py
    ```

7. Add a description of the changes made in the `changelog.md`;
    > The command `git log --oneline --decorate` may be helpfull.

8. Do commit `[GLB] Ready to the new version` and push:
    ```bash
    git push origin dev
    ```

9. Merge the `master` branch with the `dev` (`dev -> master`):
    ```bash
    git checkout master && git merge dev
    ```

10. Update the package version (like `0.14.X`) in the file `teneva/__init__.py`;

11. Build the docs:
    ```bash
    clear && python doc/build.py
    ```

12. Remove `upcoming` tag from the new version title and add section for the next version with the `upcoming` tag in the `changelog.md`;

13. Do commit like `[GLB] Update version (0.14.X)` and push:
    ```bash
    git push origin master
    ```

14. Upload the new version to `pypi` (login: AndreiChertkov):
    ```bash
    rm -r ./dist && python setup.py sdist bdist_wheel && twine upload dist/*
    ```

15. Reinstall the package from `pypi` and check that installed version is new:
    ```bash
    pip uninstall teneva -y && pip install --no-cache-dir --upgrade teneva
    ```

16. Check the [teneva docs build](https://readthedocs.org/projects/teneva/builds/);

17. Check the [teneva docs site](https://teneva.readthedocs.io/).

18. Merge the `dev` branch with the `master` (`master -> dev`) and push:
    ```bash
    git checkout dev && git merge master && git push origin dev
    ```
