"""
    Geometry / translation-operator construction.

Implements the core idea of Quan, Rybin, Scheffler & Carbogno,
"Efficient band structure unfolding with atom-centered orbitals: General
theory and application", Phys. Rev. B 113, 085112 (2026):

For a *perfect* (or placeholder-extended, see Sec. II C of the paper) atomic
mapping between PC and SC, the Löwdin-transformed PC translational operator
T̃'_PC becomes IDENTICAL to a pure phase-permutation matrix (their Eq. 27,
Eq. 45) — i.e. it does not depend on the overlap matrix S at all, only on:

  1. the atom-to-atom mapping induced by translating every SC atom by one
     PC lattice vector, and
  2. the phase e^{i 2π K·p} picked up whenever that translation crosses the
     SC boundary (p = integer SC-lattice wrap vector).

This lets us build T_i (i = 1,2,3, one per PC lattice vector direction)
purely from geometry, at a given SC K-point, and then jointly diagonalize
the (commuting, unitary) T_1, T_2, T_3 to obtain the PC-translation
eigenvectors F_{k,n} used for projection (their Eq. 28/38).
"""

const Vec3 = AbstractVector{<:Real}
const Mat3 = AbstractMatrix{<:Real}

"""
    wrap_fractional(frac::AbstractVector{<:Real})

Return `(fracmod, p)` such that `frac = fracmod + p`, with `fracmod`
componentwise in `[0, 1)` and `p` an integer vector.
"""
function wrap_fractional(frac::AbstractVector{<:Real})
    p = floor.(frac)
    fracmod = frac .- p
    return fracmod, Int.(round.(p))
end

"""
    periodic_frac_distance(a, b)

Minimal periodic distance between two fractional-coordinate vectors,
componentwise, folded into [-0.5, 0.5).
"""
function periodic_frac_distance(a::AbstractVector{<:Real}, b::AbstractVector{<:Real})
    d = a .- b
    return mod.(d .+ 0.5, 1.0) .- 0.5
end

"""
    match_atom(newpos_cart, sc_positions_cart, Ainv; tol=1e-5)

Given a Cartesian position `newpos_cart` (the result of translating some SC
atom by one PC lattice vector), find which SC atom (column index into
`sc_positions_cart`) it coincides with *modulo an SC lattice vector*, and
return `(jatom, p)` where `p` is the integer SC-lattice wrap vector needed:

    newpos_cart ≈ sc_positions_cart[:, jatom] + A * p

`Ainv` is the inverse of the SC lattice matrix `A` (columns = lattice
vectors), used to go to fractional coordinates.

Throws an error if no atom matches within `tol` (in fractional units) —
this signals either a wrong `M`/mapping, or a defect/vacancy case that
needs placeholder orbitals (not yet implemented here, see Sec. II C of the
paper and the `TODO` in `translation_operator`).
"""
function match_atom(newpos_cart::AbstractVector{<:Real},
                     sc_positions_cart::AbstractMatrix{<:Real},
                     Ainv::AbstractMatrix{<:Real}; tol::Real=1e-5)
    frac_new = Ainv * newpos_cart
    fracmod_new, _ = wrap_fractional(frac_new)

    natoms = size(sc_positions_cart, 2)
    for j in 1:natoms
        frac_j = Ainv * sc_positions_cart[:, j]
        fracmod_j, _ = wrap_fractional(frac_j)
        d = periodic_frac_distance(fracmod_new, fracmod_j)
        if maximum(abs.(d)) < tol
            # p is such that frac_new = frac_j + p  (up to the tol-close match)
            p_exact = frac_new .- frac_j
            p = Int.(round.(p_exact))
            return j, p
        end
    end
    error("match_atom: no SC atom found matching the translated position " *
          "(within tol=$tol). Check the PC lattice vector / atom ordering, " *
          "or, if this is a defect/vacancy supercell, use placeholder " *
          "orbitals (not yet implemented).")
end

"""
    AtomBasis

Bookkeeping for how AOs are laid out per atom in the SC basis.

- `positions`  : (3, Natoms) Cartesian atom positions
- `norb`       : Natoms-vector, number of AOs (basis functions) on each atom
- `offsets`    : Natoms-vector, index (0-based) of the first AO of each atom
                 within the full (Nbasis,) AO ordering
"""
struct AtomBasis
    positions::Matrix{Float64}
    norb::Vector{Int}
    offsets::Vector{Int}
end

function AtomBasis(positions::AbstractMatrix{<:Real}, norb::AbstractVector{<:Integer})
    offsets = cumsum(vcat(0, norb))[1:end-1]
    return AtomBasis(Matrix{Float64}(positions), Vector{Int}(norb), Vector{Int}(offsets))
end

nbasis(ab::AtomBasis) = sum(ab.norb)
natoms(ab::AtomBasis) = length(ab.norb)

"""
    translation_operator(pc_vector, ab::AtomBasis, Ainv, Kfrac; tol=1e-5)

Build the (Nbasis × Nbasis) unitary matrix representing the SC-K-space
Bloch translation operator for translation by one PC lattice vector
`pc_vector` (Cartesian, length-3), evaluated at fractional SC K-point
`Kfrac` (length-3, in the SC reciprocal basis B — i.e. K = Kfrac · B).

This directly builds the Löwdin-transformed operator T'_PC (paper Eq. 27),
so no overlap matrix is needed here — only the atom mapping and phases.
"""
function translation_operator(pc_vector::AbstractVector{<:Real},
                               ab::AtomBasis,
                               Ainv::AbstractMatrix{<:Real},
                               Kfrac::AbstractVector{<:Real};
                               tol::Real=1e-5)
    n = nbasis(ab)
    T = zeros(ComplexF64, n, n)
    for iatom in 1:natoms(ab)
        # The paper defines the translation operator as
        # T|phi_LJ> = |phi_(L-1)J> (Eq. 15): it moves an AO by -a_i.
        # Using +a_i here builds the inverse operator and gives conjugated
        # projector phases (visible already in the Eq. 42 toy model).
        newpos = ab.positions[:, iatom] .- pc_vector
        jatom, p = match_atom(newpos, ab.positions, Ainv; tol=tol)
        if ab.norb[jatom] != ab.norb[iatom]
            error("translation_operator: orbital-count mismatch mapping atom " *
                  "$iatom ($(ab.norb[iatom]) AOs) -> atom $jatom " *
                  "($(ab.norb[jatom]) AOs). This happens for defects/dopants " *
                  "with different basis sets; the paper handles this via " *
                  "placeholder orbitals (Sec. II C) which is a TODO here.")
        end
        # Our Bloch sums use the paper's phase convention.  If the mapped AO
        # differs by the SC lattice vector A*p, the matrix element therefore
        # acquires exp(-i K.A*p).
        phase = cispi(-2 * dot(Kfrac, p))
        for q in 1:ab.norb[iatom]
            T[ab.offsets[jatom]+q, ab.offsets[iatom]+q] = phase
        end
    end
    return T
end

"""
    translation_operators(pc_lattice, ab::AtomBasis, sc_lattice, Kfrac; tol=1e-5)

Convenience wrapper: builds the three translation operators T_1, T_2, T_3
for the three columns of `pc_lattice` (3×3 matrix, PC lattice vectors as
columns), given the SC lattice matrix `sc_lattice` (3×3, columns = SC
lattice vectors) and fractional SC K-point `Kfrac`.
"""
function translation_operators(pc_lattice::AbstractMatrix{<:Real},
                                ab::AtomBasis,
                                sc_lattice::AbstractMatrix{<:Real},
                                Kfrac::AbstractVector{<:Real};
                                tol::Real=1e-5)
    Ainv = inv(Matrix{Float64}(sc_lattice))
    return [translation_operator(pc_lattice[:, i], ab, Ainv, Kfrac; tol=tol) for i in 1:3]
end
