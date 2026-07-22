"""
    BandData

Band solution at a list of fractional reciprocal coordinates. Entry `i` in
each vector belongs to `kpoints_frac[i]`; columns of `coefficients[i]` are
bands and obey `C† S C = I` with `S = overlaps[i]`.
"""
struct BandData
    kpoints_frac::Vector{Vector{Float64}}
    energies::Vector{Vector{Float64}}
    coefficients::Vector{Matrix{ComplexF64}}
    overlaps::Vector{Matrix{ComplexF64}}
    energy_unit::String
end

"""
    fourier_matrix(matrices, R, kfrac)

Reconstruct a reciprocal-space matrix from real-space images using
`A(k) = sum_i exp(+i 2π k⋅R[:,i]) A(R_i)`. Both `kfrac` and `R` are expressed
in mutually dual fractional bases, so their dot product is dimensionless.
"""
function fourier_matrix(matrices::Vector{<:SparseMatrixCSC}, R, kfrac)
    length(matrices) == size(R, 2) || error("matrix/R count mismatch")
    n = size(first(matrices), 1)
    result = zeros(ComplexF64, n, n)
    for i in eachindex(matrices)
        # cispi(x) evaluates exp(iπx) without explicitly forming πx.
        phase = cispi(2 * dot(kfrac, R[:, i]))
        A = matrices[i]
        # Accumulate only stored nonzeros. This is noticeably cheaper than
        # converting every localized real-space block to a dense matrix.
        for col in axes(A, 2), ptr in nzrange(A, col)
            result[A.rowval[ptr], col] += phase * A.nzval[ptr]
        end
    end
    result
end

hamiltonian_at_k(model::RealSpaceModel, k; spin=1) = fourier_matrix(model.H[spin], model.R, k)
overlap_at_k(model::RealSpaceModel, k) = fourier_matrix(model.S, model.R, k)

"""
    solve_bands(model, kpoints; spin=1, validate=true, atol=1e-8)

Solve `H(k)C = S(k)CE` (paper Eq. 18) at each fractional `k` point. The
returned eigenvectors are normalized in the nonorthogonal AO metric.

With `validate=true`, the routine checks Hermiticity before trusting the
symmetrized matrices and verifies the paper's normalization `C†SC=I`.
"""
function solve_bands(model::RealSpaceModel, kpoints; spin=1, validate=true, atol=1e-8)
    1 <= spin <= nspin(model) || throw(ArgumentError("invalid spin channel"))
    ks = [Vector{Float64}(k) for k in kpoints]
    energies = Vector{Float64}[]; coefficients = Matrix{ComplexF64}[]; overlaps = Matrix{ComplexF64}[]
    for (ik, k) in enumerate(ks)
        length(k) == 3 || error("k-point $ik does not have three components")
        Hraw = hamiltonian_at_k(model, k; spin=spin)
        Sraw = overlap_at_k(model, k)
        # Remove roundoff-level anti-Hermitian noise before LAPACK. The raw
        # residual is still checked below so a genuinely bad input is rejected.
        Hk = (Hraw + Hraw') / 2
        Sk = (Sraw + Sraw') / 2
        # A positive-definite overlap is required for a well-posed generalized
        # Hermitian eigenproblem and for the later Löwdin square root.
        minimum(eigvals(Hermitian(Sk))) > 0 || error("S(k) is not positive definite at k-point $ik")
        sol = eigen(Hermitian(Hk), Hermitian(Sk))
        push!(energies, Vector{Float64}(sol.values))
        push!(coefficients, Matrix{ComplexF64}(sol.vectors))
        push!(overlaps, Matrix{ComplexF64}(Sk))
        if validate
            norm(Hraw-Hraw', Inf) <= atol*max(norm(Hraw,Inf),1) || error("non-Hermitian H(k) at $ik")
            norm(Sraw-Sraw', Inf) <= atol*max(norm(Sraw,Inf),1) || error("non-Hermitian S(k) at $ik")
            norm(sol.vectors' * Sk * sol.vectors - I, Inf) <= 10atol || error("C†SC != I at $ik")
        end
    end
    BandData(ks, energies, coefficients, overlaps, model.energy_unit)
end

"""
    interpolate_kpath(points, n_per_segment)

Linearly interpolate a path through fractional reciprocal coordinates. Returns
`(kpoints, distance, ticks)`, where `ticks` marks every supplied vertex. Shared
segment endpoints are emitted once.
"""
function interpolate_kpath(points::Vector{<:AbstractVector}, n_per_segment::Int)
    n_per_segment >= 2 || throw(ArgumentError("n_per_segment must be at least 2"))
    kpoints = Vector{Float64}[]; distance = Float64[0.0]; ticks = Float64[0.0]
    for segment in 1:(length(points)-1)
        start, stop = points[segment], points[segment+1]
        for j in 0:(n_per_segment-1)
            # The previous segment already emitted this shared endpoint.
            segment > 1 && j == 0 && continue
            k = Vector{Float64}(start .+ (j/(n_per_segment-1)) .* (stop .- start))
            if !isempty(kpoints)
                push!(distance, distance[end] + norm(k-kpoints[end]))
            end
            push!(kpoints, k)
        end
        push!(ticks, distance[end])
    end
    return kpoints, distance, ticks
end
