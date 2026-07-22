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
