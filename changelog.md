# changelog

> This is a draft version of the changelog (for developers now). In the future, it will be integrated into the github system.


## Notation

- `GLB` - global changes
- `RNM` - renames of existing modules, functions, etc.
- `UPG` - upgrade of existing modules, functions, etc.
- `NEW` - new modules, functions, etc.
- `DEM` - changes in demo (jupyter notebooks)
- `TST` - new tests (or its updates) for modules and functions
- `FIX` - fixes for small bugs
- `BUG` - fixes for big bugs
- `STL` - fixes for style (pep, etc.) of functions and modules
- `DOC` - updates for content of the docs
- `DEV` - some code related to the development of new approaches


## Version 0.14.3 (upcoming)


## Version 0.14.2

- [GLB] Update `requirements.txt` for correct installation with python 3.8

- [GLB] Small fix for `workflow.md`


## Version 0.14.1

- [GLB] Put back the module `func_full` with Chebyshev interpolation in the full (numpy) format


## Version 0.14.0

- [GLB] remove `core_jax` module, it will be in the separate repo [teneva_jax](https://github.com/AndreiChertkov/teneva_jax)

- [GLB] remove `func` module, it will be in the special repo [teneva_bm](https://github.com/AndreiChertkov/teneva_bm)

- [GLB, RNM] remove `core` module and move its content (submodules) into the root of the package

- [GLB] Remove the module `cheb_full`

- [GLB, TST] Add folder `test` (draft version) with unittests. For example, basic tests for the `maxvol` module have been added, later all modules and functions will be covered by tests

- [GLB, RNM] Rename the `cheb` module into `func`, combine with `sin`, and set new names for inner functions. We also remove the function `cheb.cheb_bld` (Cross / ALS should be used instead)

- [RNM] Rename `*_contin` modules and functions into `*_func`

- [DOC] Big update for the structure of the online docs

- [GLB] Big update for code, demo and docs (there are several breaking changes, especially in the "func" module)
