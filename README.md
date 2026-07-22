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

Generate the three small graphene HDF5 models (primitive, pristine 2×2, and defective 2×2), then export the unfolded defective supercell over the primitive bands to a binary HDF5 file:

```sh
julia --project=. examples/graphene/generate_hdf5.jl
julia --project=. examples/graphene/export_unfolding.jl
```

`export_unfolding.jl` only depends on `Unfolding` itself — no plotting library. It writes a single `examples/graphene/output/graphene_unfolding.h5`, documented in [`docs/unfolded-hdf5-schema.md`](docs/unfolded-hdf5-schema.md), bundling the unfolded supercell bands together with the primitive-cell reference bands. Any HDF5-capable tool can turn that into a plot; two examples are provided:

```sh
julia --project=examples/graphene/plotting examples/graphene/plotting/plot_unfolding.jl   # Plots.jl
python examples/graphene/plotting/plot_matplotlib.py                                       # Matplotlib
```

Both write `examples/graphene/output/graphene_unfolding.png` (and the Julia one also a `.pdf`).

## Data flow

```text
CP2K / another code -> converter -> canonical HDF5 -> bands/unfolding -> binary HDF5 export -> any plotting tool
```

The canonical schema and phase convention are documented in [`docs/hdf5-schema.md`](docs/hdf5-schema.md). A detailed equation-by-equation mapping from the reference paper to the implementation is available in [`docs/paper-implementation.md`](docs/paper-implementation.md). The graphene CP2K example and conversion command are in [`examples/graphene/cp2k/README.md`](examples/graphene/cp2k/README.md). For the end-to-end steps to use this on your own structures -- run the calculation, convert, unfold, export, plot -- see [`docs/getting-started.md`](docs/getting-started.md).

Reference: J. Quan, N. Rybin, M. Scheffler, and C. Carbogno, *Phys. Rev. B* **113**, 085112 (2026), [DOI 10.1103/7xym-7388](https://doi.org/10.1103/7xym-7388).

## Julia API

```julia
pc = read_model("primitive.h5")
sc = read_model("supercell.h5")
bands = solve_bands(pc, [[0.0, 0.0, 0.0], [2/3, 1/3, 0.0]])

# End-user entry point: unfolds sc along a primitive-cell k-path in one call.
result = unfold_bandstructure(pc.lattice, sc, [[0.0,0.0,0.0],[2/3,1/3,0.0],[0.5,0.0,0.0]], 61)
write_unfolded_hdf5("unfolded.h5", result)
```

Use `convert_cp2k_to_hdf5` for CP2K. All downstream functions consume `RealSpaceModel`, so future ports only need to produce the same canonical model. Lower-level building blocks (`AtomBasis`, `translation_operators`, `joint_eigen`, `unfold_supercell`, ...) remain available and exported for anyone who needs a mesh, a single k-point, or a custom projector instead of a path.
