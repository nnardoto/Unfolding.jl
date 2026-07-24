# Graphene with CP2K

`graphene_primitive.inp` and `graphene_2x2.inp` are small PBE/DZVP test calculations. They request real-space KS and overlap matrices with `KS_CSR_WRITE` and `S_CSR_WRITE`.

Run CP2K inside a disposable `run` directory so its CSR images remain separated from the example source:

```sh
mkdir -p examples/graphene/cp2k/run
cd examples/graphene/cp2k/run
cp2k.psmp -i ../graphene_primitive.inp -o graphene_primitive.out
```

Then, from the repository root:

```sh
julia --project=. examples/graphene/convert_cp2k.jl primitive \
  examples/graphene/cp2k/run/graphene_primitive.out \
  examples/graphene/cp2k/run/graphene_pc_H \
  examples/graphene/cp2k/run/graphene_pc_S \
  examples/graphene/models/graphene_primitive_cp2k.h5
```

For the supercell, replace `primitive` by `2x2` and use the `graphene_sc_H`/`graphene_sc_S` prefixes.

CP2K writes one CSR file per real-space periodic image. This is temporary converter input: the converter gathers all images into one canonical HDF5 file. The band path and unfolding subsequently read only that HDF5 file.

The example assumes 13 spherical AOs per carbon for `DZVP-MOLOPT-SR-GTH`. If the basis is changed, update `norb` in `convert_cp2k.jl`; the converter checks matrix dimensions and Hermitian reciprocity.

## Lightweight end-to-end debug case

`graphene_debug_2x2.inp` and `graphene_debug_primitive.inp` form a small,
self-checking pair intended for debugging the complete CP2K → CSR → HDF5 →
unfolding path. They use the minimal `SZV-MOLOPT-SR-GTH` basis and equivalent
3×3×1 supercell / 6×6×1 primitive-cell meshes. The 2×2 supercell is unfolded
along Γ–K–M–Γ and plotted over an independent primitive-cell calculation.

From the repository root:

```sh
mkdir -p examples/graphene/cp2k/run_debug
cd examples/graphene/cp2k/run_debug
OMP_NUM_THREADS=1 cp2k.psmp -i ../graphene_debug_2x2.inp -o graphene_debug.out
OMP_NUM_THREADS=1 cp2k.psmp -i ../graphene_debug_primitive.inp -o graphene_debug_pc.out
cd ../../../..
OPENBLAS_NUM_THREADS=1 julia --threads=auto --project=. \
  examples/graphene/cp2k/debug_unfold.jl
```

The Julia script discovers both basis layouts from the CP2K outputs, builds
the canonical models, unfolds 94 path points, solves the primitive-cell bands
on the same path, and checks:

- constant total spectral weight at every k-point;
- every individual weight is in `[0,1]` within numerical tolerance;
- the two occurrences of Γ have identical energies and weights;
- every primitive band has an unfolded counterpart within 20 meV;
- spectral weight farther than 20 meV from every primitive band is reported
  explicitly as leakage (it should vanish for this pristine system).

It writes `graphene_debug_unfolded.h5` (including `/reference/energies`) and
two small plotting CSV files under `run_debug`. The optional Matplotlib plot
overlays the independent primitive bands as cyan lines on the unfolding
weights and does not require h5py:

```sh
python3 examples/graphene/cp2k/plot_debug.py
```

Pass another run directory as the first argument to both scripts. The Julia
script also accepts the number of samples per segment as its second argument.
