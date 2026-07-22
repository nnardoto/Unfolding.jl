# Canonical HDF5 schema

Schema name: `unfolding.realspace`; version: `1.0`.

The file stores a localized-orbital real-space model. Lattice vectors and atomic positions are columns. AO indices are grouped by atom in the same order as `positions`.

```text
/schema/name                              UInt8 UTF-8 bytes
/schema/version                           Int[2]
/structure/lattice                        Float64[3,3]
/structure/positions                      Float64[3,natom]
/structure/reference_positions            Float64[3,natom]
/structure/atomic_numbers                 Int[natom]
/basis/norb_per_atom                      Int[natom]
/translations/R                           Int[3,nR]
/matrices/overlap/<image>/colptr          Int[nbasis+1]
/matrices/overlap/<image>/rowval          Int[nnz]
/matrices/overlap/<image>/nzval           Float64[nnz]
/matrices/hamiltonian/spin_<s>/<image>/... same CSC datasets
/metadata/energy_unit                     UInt8 UTF-8 bytes
/metadata/length_unit                     UInt8 UTF-8 bytes
/metadata/source                          UInt8 UTF-8 bytes
/metadata/nspin                           Int[1]
```

`<image>` is the one-based column index in `/translations/R`. Sparse matrices use Julia/SuiteSparse CSC indexing, also one-based.

## Mathematical convention

For fractional reciprocal coordinates `k` and integer direct-lattice translation `R`,

```text
H(k) = sum_R exp(+i 2π k·R) H(R)
S(k) = sum_R exp(+i 2π k·R) S(R)
```

Every translation must have its opposite in the file and satisfy

```text
H(-R) = H(R)†
S(-R) = S(R)†.
```

`positions` contains the physical structure. `reference_positions` contains the ideal parent topology used to construct primitive-cell translation operators; for an onsite or substitutional defect with unchanged sites they can be identical. `norb_per_atom` must follow the AO ordering in every matrix.

Version 1.0 stores real-valued real-space matrices. Energies and lengths are not converted silently; their units are mandatory metadata.
