"""
    Unfolding

Band-unfolding tools for localized, nonorthogonal atomic-orbital bases.

The package is split into three layers:

1. `schema.jl` and `bands.jl` define a code-independent real-space model;
2. `geometry.jl`, `jointdiag.jl`, and `unfold.jl` implement the method of
   Quan *et al.*, Phys. Rev. B 113, 085112 (2026);
3. `converters/` contains code-specific importers, initially for CP2K.

See `docs/paper-implementation.md` for the equation-by-equation derivation.
"""
module Unfolding

using HDF5
using LinearAlgebra
using Printf
using Random
using SparseArrays

include("schema.jl")
include("bands.jl")
include("geometry.jl")
include("jointdiag.jl")
include("unfold.jl")
include("converters/cp2k.jl")

export RealSpaceModel, BandData, write_model, read_model, validate_model, model_summary
export fourier_matrix, hamiltonian_at_k, overlap_at_k, solve_bands, interpolate_kpath
export AtomBasis, translation_operator, translation_operators, periodic_frac_distance
export joint_eigen, check_joint_eigen, kfrac_from_lambdas, group_by_kfrac
export hermitian_sqrt, lowdin_transform, unfold_weights, check_sum_rule_over_k
export spectral_function, write_unfolded_csv, unfold_supercell
export read_cp2k_csr, convert_cp2k_to_hdf5

"""
    unfold_supercell(pc_lattice, ab, sc_lattice, Kfrac, S, C; tol=1e-5, rng)

Unfold all columns of the supercell coefficient matrix `C` at one supercell
point `Kfrac`. `S` is the overlap matrix at the same point and `ab` describes
the ideal/reference AO topology of the supercell.

Returns `(W, V, lambdas)` where:

- `W[k]` contains one unfolding weight per supercell band at primitive-cell
  fractional point `k`;
- columns of `V` are common eigenvectors of the three primitive translations;
- `lambdas[i,n]` is the eigenvalue of translation `i` for column `n` of `V`.

The four operations below correspond respectively to the paper's construction
of `T_PC`, simultaneous treatment of Eqs. 46-48, identification of the folded
`k` subspaces, and the Löwdin-projected weight in Eq. 28.
"""
function unfold_supercell(pc_lattice::AbstractMatrix{<:Real},
                          ab::AtomBasis,
                          sc_lattice::AbstractMatrix{<:Real},
                          Kfrac::AbstractVector{<:Real},
                          S::AbstractMatrix{<:Number},
                          C::AbstractMatrix{<:Number};
                          tol::Real=1e-5,
                          rng::AbstractRNG=Random.default_rng())
    # The translation matrices are pure phase-permutation matrices and are
    # therefore unitary. Dense storage is intentional here because their
    # common eigenvectors are also dense.
    Ts = Matrix{ComplexF64}.(translation_operators(pc_lattice, ab, sc_lattice, Kfrac; tol=tol))
    V, lambdas = joint_eigen(Ts; rng=rng)

    # Equal eigenvalue phases identify the degenerate projector subspace for
    # one unfolded primitive-cell k point.
    groups = group_by_kfrac(kfrac_from_lambdas(lambdas))

    # C' = S^(1/2)C (Eq. 24) is formed only once; projection then uses Eq. 28.
    W = unfold_weights(V, groups, lowdin_transform(ComplexF64.(S), ComplexF64.(C)))
    return W, V, lambdas
end

end
