"""
    Joint diagonalization of commuting, unitary translation operators.

T_1, T_2, T_3 (one per PC lattice direction) commute pairwise because
translations along different lattice vectors commute. Each is unitary
(pure phase-permutation). We diagonalize a generic Hermitian combination
of their Hermitian and anti-Hermitian parts. Its eigenvectors are,
generically (i.e. almost surely for random coefficients), the *common*
eigenvectors of T_1, T_2, T_3 simultaneously, while Hermiticity guarantees
an orthonormal basis inside orbital-degenerate eigenspaces. Once we have an
eigenvector v of this combination, its individual
T_i-eigenvalues are recovered exactly via the Rayleigh quotient
λ_i = v† T_i v (exact because v is a true eigenvector of each T_i).

This is a numerical alternative to the paper's analytical orbit solution
(Eqs. 33-38) and sequential 3D diagonalization (Eqs. 46-48). Because the
operators are commuting and normal, both procedures construct the same joint
eigenspaces and therefore the same projectors. The graphene regression tests
in `test/runtests.jl` validate the resulting weights and sum rule.
"""

"""
    joint_eigen(Ts::Vector{<:AbstractMatrix{ComplexF64}}; rng=Random.default_rng())

Return `(V, lambdas)`:

- `V`       : (n × n) matrix whose columns are the common, normalized
              eigenvectors of all matrices in `Ts`.
- `lambdas` : (length(Ts) × n) matrix, `lambdas[i, k]` is the eigenvalue of
              `Ts[i]` for eigenvector `V[:, k]`.
"""
function joint_eigen(Ts::Vector{Matrix{ComplexF64}}; rng::AbstractRNG=Random.default_rng())
    @assert !isempty(Ts)
    n = size(Ts[1], 1)

    # Diagonalize a generic *Hermitian* function of the commuting unitary
    # operators.  A generic complex combination is normal in exact
    # arithmetic, but a general eigensolver need not return an orthonormal
    # basis inside repeated eigenspaces (the usual case with several AOs per
    # atom).  The Hermitian construction makes orthonormality guaranteed by
    # the eigensolver while random real coefficients separate distinct joint
    # eigenvalue tuples with probability one.
    Hcombo = zeros(ComplexF64, n, n)
    for T in Ts
        a, b = randn(rng), randn(rng)
        Hcombo .+= a .* ((T .+ T') ./ 2)
        Hcombo .+= b .* ((T .- T') ./ (2im))
    end

    F = eigen(Hermitian(Hcombo))
    V = Matrix{ComplexF64}(F.vectors)
    for k in 1:n
        V[:, k] ./= norm(@view V[:, k])
    end

    lambdas = zeros(ComplexF64, length(Ts), n)
    for k in 1:n
        vk = @view V[:, k]
        for (i, T) in enumerate(Ts)
            lambdas[i, k] = dot(vk, T * vk)   # <v|T|v>, exact Rayleigh quotient
        end
    end
    return V, lambdas
end

"""
    check_joint_eigen(Ts, V, lambdas; atol=1e-8)

Sanity check: residual ‖T_i v_k - λ_{i,k} v_k‖ for every operator/eigenvector
pair. Returns the maximum residual found (should be ~1e-10 or smaller for
well-separated eigenvalues of the random combination; a larger residual
              usually indicates that the random combination had accidental degeneracies and a different
random seed should be tried).
"""
function check_joint_eigen(Ts::Vector{Matrix{ComplexF64}}, V::AbstractMatrix{ComplexF64},
                            lambdas::AbstractMatrix{ComplexF64})
    n = size(V, 2)
    maxres = 0.0
    for k in 1:n
        vk = @view V[:, k]
        for (i, T) in enumerate(Ts)
            res = norm(T * vk .- lambdas[i, k] .* vk)
            maxres = max(maxres, res)
        end
    end
    return maxres
end

"""
    kfrac_from_lambdas(lambdas::AbstractMatrix{ComplexF64}; digits=6)

Convert eigenvalues (rows = directions, columns = eigenvector index) to PC
fractional k-point coordinates `f_k = angle(λ)/(2π) mod 1`, rounded to
`digits` decimals so that degenerate eigenvectors sharing the same
(unfolded) PC k-point can be grouped by exact key equality.
"""
function kfrac_from_lambdas(lambdas::AbstractMatrix{ComplexF64}; digits::Int=6)
    fk = mod.(angle.(lambdas) ./ (2π), 1.0)
    fk = round.(fk; digits=digits)
    # Floating-point noise can turn an angle infinitesimally below zero into
    # 1.0 after mod.  Fractional coordinates 1 and 0 are identical and must
    # not form separate unfolding groups.
    fk[fk .== 1.0] .= 0.0
    return fk
end

"""
    group_by_kfrac(fk::AbstractMatrix{<:Real})

Group eigenvector column indices by identical (rounded) fractional k-point.
Returns a `Dict{Vector{Float64}, Vector{Int}}`.
"""
function group_by_kfrac(fk::AbstractMatrix{<:Real})
    groups = Dict{Vector{Float64},Vector{Int}}()
    ndir, n = size(fk)
    for col in 1:n
        key = fk[:, col]
        push!(get!(groups, key, Int[]), col)
    end
    return groups
end
