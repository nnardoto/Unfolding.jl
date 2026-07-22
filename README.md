# Unfolding.jl

Band unfolding in Julia using a code-independent HDF5 input format. Electronic-structure codes are isolated behind converters; the solver and plotting layers never parse native output files.

The first converter imports CP2K real-space matrices written by `KS_CSR_WRITE` and `S_CSR_WRITE`. It does not depend on `MO_KP`, `AO_MATRICES`, or CP2K 2026.2.

## Quick start

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
Pkg.test()
```

Generate the three small graphene HDF5 models (primitive, pristine 2×2, and defective 2×2), then plot the unfolded defective supercell over the primitive bands:

```sh
julia --project=. examples/graphene/generate_hdf5.jl
julia --project=. examples/graphene/plot_unfolding.jl
```

The plot is written to `examples/graphene/output/graphene_unfolding.png` and `.pdf`.

## Data flow

```text
CP2K / another code -> converter -> canonical HDF5 -> bands/unfolding -> Plots.jl
```

The canonical schema and phase convention are documented in [`docs/hdf5-schema.md`](docs/hdf5-schema.md). A detailed equation-by-equation mapping from the reference paper to the implementation is available in [`docs/paper-implementation.md`](docs/paper-implementation.md). The graphene CP2K example and conversion command are in [`examples/graphene/cp2k/README.md`](examples/graphene/cp2k/README.md).

Reference: J. Quan, N. Rybin, M. Scheffler, and C. Carbogno, *Phys. Rev. B* **113**, 085112 (2026), [DOI 10.1103/7xym-7388](https://doi.org/10.1103/7xym-7388).

## Julia API

```julia
model = read_model("calculation.h5")
bands = solve_bands(model, [[0.0, 0.0, 0.0], [2/3, 1/3, 0.0]])
```

Use `convert_cp2k_to_hdf5` for CP2K. All downstream functions consume `RealSpaceModel`, so future ports only need to produce the same canonical model.
