"""
    Löwdin transform, unfolding weights, and spectral function.

Implements the paper's Eqs. (24)-(28), (12):

    C'_{KN} = S_K^{1/2} C_{KN}
    W^k_{KN} = Σ_{n ∈ group(k)} |F_{k,n}† C'_{KN}|^2
    A(k, E) = Σ_{KN} W^k_{KN} δ(E - E_{KN})   (δ → Lorentzian in practice)
"""

"""
    hermitian_sqrt(S::AbstractMatrix{ComplexF64})

Matrix square root of a Hermitian positive-definite matrix `S`, via
eigendecomposition. Symmetrizes `S` first to remove numerical asymmetry.
"""
function hermitian_sqrt(S::AbstractMatrix{ComplexF64})
    Sh = Hermitian((S .+ S') ./ 2)
    vals, vecs = eigen(Sh)
    if any(v -> v <= 0, vals)
        error("hermitian_sqrt: overlap matrix is not positive definite " *
              "(smallest eigenvalue = $(minimum(vals))). Check S_K.")
    end
    return vecs * Diagonal(sqrt.(vals)) * vecs'
end

"""
    lowdin_transform(S::AbstractMatrix{ComplexF64}, C::AbstractMatrix{ComplexF64})

Return `S^{1/2} * C`, i.e. the Löwdin-transformed MO coefficient matrix
(paper Eq. 24). `C` is (Nbasis × Nbands).
"""
function lowdin_transform(S::AbstractMatrix{ComplexF64}, C::AbstractMatrix{ComplexF64})
    return hermitian_sqrt(S) * C
end

"""
    unfold_weights(V::AbstractMatrix{ComplexF64}, groups, Cprime::AbstractMatrix{ComplexF64})

Compute unfolding weights `W^k_{KN}` for every unfolded PC k-point (key of
`groups`, produced by `group_by_kfrac`) and every SC band N (column of
`Cprime`, the Löwdin-transformed SC coefficients at this SC K-point).

Returns `Dict{Vector{Float64}, Vector{Float64}}` mapping each `f_k` to a
length-Nbands weight vector.
"""
function unfold_weights(V::AbstractMatrix{ComplexF64},
                         groups::Dict{Vector{Float64},Vector{Int}},
                         Cprime::AbstractMatrix{ComplexF64})
    nbands = size(Cprime, 2)
    W = Dict{Vector{Float64},Vector{Float64}}()
    for (fk, cols) in groups
        w = zeros(Float64, nbands)
        for c in cols
            Fc = @view V[:, c]
            for n in 1:nbands
                w[n] += abs2(dot(Fc, @view Cprime[:, n]))
            end
        end
        W[fk] = w
    end
    return W
end

"""
    check_sum_rule_over_k(W::Dict)

Sanity check of the two sum rules quoted in the paper (below Eq. 42):
`Σ_kn W^{kn}_{KN} = 1` for each SC band N. Since our `W[fk]` is already
summed over the degenerate group at each `fk`, this reduces to checking
that, for every band index N, `Σ_{fk} W[fk][N] ≈ 1`.

Returns the maximum deviation from 1 found across all bands.
"""
function check_sum_rule_over_k(W::Dict{Vector{Float64},Vector{Float64}})
    nbands = length(first(values(W)))
    totals = zeros(Float64, nbands)
    for w in values(W)
        totals .+= w
    end
    return maximum(abs.(totals .- 1.0))
end

"""
    spectral_function(Egrid, weights::Vector{Float64}, EKN::Vector{Float64}; broadening=0.01)

Evaluate `A(k, E) = Σ_N w_N * L(E - E_N; broadening)` on the energies in
`Egrid`, where `L` is a normalized Lorentzian (paper Eq. 12/52, δ replaced
by a Lorentzian of HWHM `broadening`).
"""
function spectral_function(Egrid::AbstractVector{<:Real},
                            weights::AbstractVector{<:Real},
                            EKN::AbstractVector{<:Real};
                            broadening::Real=0.01)
    A = zeros(Float64, length(Egrid))
    for (wn, En) in zip(weights, EKN)
        wn == 0 && continue
        @. A += wn * (broadening / π) / ((Egrid - En)^2 + broadening^2)
    end
    return A
end

"""
    write_unfolded_csv(path, W, energies)

Write one row per unfolded k-point and SC band with columns
`k1,k2,k3,band,energy,weight`. Energies remain in the unit supplied by the
caller. The file is directly readable by
Julia, Python, gnuplot, or spreadsheet software.
"""
function write_unfolded_csv(path::AbstractString,
                            W::Dict{Vector{Float64},Vector{Float64}},
                            energies::AbstractVector{<:Real})
    all(length(w) == length(energies) for w in values(W)) ||
        throw(DimensionMismatch("every weight vector must match the energy vector"))
    kpoints = sort(collect(keys(W)); by=k -> Tuple(k))
    open(path, "w") do io
        println(io, "k1,k2,k3,band,energy,weight")
        for k in kpoints
            length(k) == 3 || throw(DimensionMismatch("CSV output requires 3D k-points"))
            for band in eachindex(energies)
                @printf(io, "%.12g,%.12g,%.12g,%d,%.16g,%.16g\n",
                        k[1], k[2], k[3], band, energies[band], W[k][band])
            end
        end
    end
    return path
end
