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


## Version 0.14.4 (upcoming)


## Version 0.14.3

- [GLB] The `seed` argument has been added to all functions that operate with random values (`anova.anova`, `core.core_qr_rand`, `cross_act.cross_act`, `sample.sample`, `sample.sample_lhs`, `sample.sample_rand`, `sample.sample_square`, `sample.sample_tt`, `sample_func.sample_func`, `tensors.rand` and `tensors.rand_norm`), which can be either an integer number (will be used for new numpy generator class) or an instance of the [numpy generator class](https://numpy.org/doc/stable/reference/random/generator.html). Please note that now, on repeated calls without setting a new seed value (default seed is always `42`), these functions will return the same values

- [UPG] Big update for `README.md`: remove irrelevant descriptions, add link to [changelog.md](https://github.com/AndreiChertkov/teneva/blob/master/changelog.md) and [workflow.md](https://github.com/AndreiChertkov/teneva/blob/master/workflow.md), and also add new section with useful links to related repositories and papers

- [NEW] Add new function `sample.sample_rand`

- [FIX] Fix dependencies in `requirements.txt` to better support python 3.9 and also add link to `requirements.txt` into `MANIFEST.in`

- [FIX] Fix small rare bug in `sample.sample_square`


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
