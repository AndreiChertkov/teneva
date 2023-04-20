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


## Version 0.14.0 (upcoming)

- [GLB] remove `core_jax` module, it will be in separate repo [teneva_jax](https://github.com/AndreiChertkov/teneva_jax)

- [GLB] remove `func` module, it will be moved into special repo [teneva_bm](https://github.com/AndreiChertkov/teneva_bm)

- [GLB, RNM] remove `core` module and move its content (submodules) into the root of the package

- [GLB, TST] add folder `test` (draft version) with unittests. For example, basic tests for the `maxvol` module have been added, in the future all modules and functions will be covered by tests
