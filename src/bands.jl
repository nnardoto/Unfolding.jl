"""
    BandData

Solução de bandas em uma lista de coordenadas fracionárias recíprocas. A
entrada `i` de cada vetor pertence a `kpoints_frac[i]`; as colunas de
`coefficients[i]` são as bandas e obedecem `C† S C = I`, com
`S = overlaps[i]`.
"""
struct BandData
    kpoints_frac::Vector{Vector{Float64}}
    energies::Vector{Vector{Float64}}
    coefficients::Vector{Matrix{ComplexF64}}
    overlaps::Vector{Matrix{ComplexF64}}
    energy_unit::String
end

"""
Cria um atualizador de progresso seguro para chamadas concorrentes.

Cada atualização ocupa uma linha própria e é descarregada imediatamente no
`io`, permitindo acompanhar a execução com ferramentas como `tail -f`.
"""
function _progress_reporter(label::AbstractString, total::Integer, enabled::Bool, io::IO)
    completed = Ref(0)
    output_lock = ReentrantLock()

    if enabled
        println(io, label, ": 0 / ", total)
        flush(io)
    end

    function report(Δt::Union{Nothing,Real}=nothing)
        enabled || return nothing
        lock(output_lock) do
            completed[] += 1
            if Δt === nothing
                println(io, label, ": ", completed[], " / ", total)
            else
                println(io, label, ": ", completed[], " / ", total,
                        " (", round(Δt; digits=3), "s)")
            end
            flush(io)
        end
        return nothing
    end

    return report
end

"""
    fourier_matrix(matrices, R, kfrac)

Reconstrói uma matriz em espaço recíproco a partir das imagens em espaço
real, usando `A(k) = sum_i exp(+i 2π k⋅R[:,i]) A(R_i)`. Tanto `kfrac`
quanto `R` estão expressos em bases fracionárias mutuamente duais, de modo
que o produto escalar entre eles é adimensional.
"""
function fourier_matrix(matrices::Vector{<:SparseMatrixCSC}, R, kfrac)
    length(matrices) == size(R, 2) || error("matrix/R count mismatch")
    n = size(first(matrices), 1)
    result = zeros(ComplexF64, n, n)
    for i in eachindex(matrices)
        # cispi(x) calcula exp(iπx) sem formar πx explicitamente.
        phase = cispi(2 * dot(kfrac, R[:, i]))
        A = matrices[i]
        # Soma apenas os elementos não-nulos armazenados. Isso é bem mais
        # barato do que converter cada bloco esparso em espaço real para
        # uma matriz densa antes de somar.
        for col in axes(A, 2), ptr in nzrange(A, col)
            result[A.rowval[ptr], col] += phase * A.nzval[ptr]
        end
    end
    result
end

hamiltonian_at_k(model::RealSpaceModel, k; spin=1) = fourier_matrix(model.H[spin], model.R, k)
overlap_at_k(model::RealSpaceModel, k) = fourier_matrix(model.S, model.R, k)

"""
    solve_bands(model, kpoints; spin=1, validate=true, atol=1e-8,
                parallel=Threads.nthreads() > 1, progress=false)

Resolve `H(k)C = S(k)CE` (Eq. 18 do artigo) em cada ponto `k` fracionário
fornecido. Os autovetores retornados são normalizados na métrica não-
ortogonal dos AOs.

Com `validate=true`, a rotina confere a hermiticidade das matrizes antes de
confiar na versão simetrizada, e verifica a normalização `C†SC=I` exigida
pelo artigo.

Os pontos k são independentes. Com `parallel=true`, eles são distribuídos
entre as threads Julia, sem alterar a ordem das entradas no `BandData`.
Use `progress=true` para acompanhar a fração de pontos concluídos.
"""
function solve_bands(model::RealSpaceModel, kpoints; spin=1, validate=true, atol=1e-8,
                     parallel=Threads.nthreads() > 1, progress::Bool=false,
                     progress_io::IO=stderr)
    1 <= spin <= nspin(model) || throw(ArgumentError("invalid spin channel"))
    ks = [Vector{Float64}(k) for k in kpoints]
    nk = length(ks)
    energies = Vector{Vector{Float64}}(undef, nk)
    coefficients = Vector{Matrix{ComplexF64}}(undef, nk)
    overlaps = Vector{Matrix{ComplexF64}}(undef, nk)
    report_progress = _progress_reporter("Bandas", nk, progress && nk > 0, progress_io)

    function solve_one!(ik)
        k = ks[ik]
        length(k) == 3 || error("k-point $ik does not have three components")
        Δt = @elapsed begin
            Hraw = hamiltonian_at_k(model, k; spin=spin)
            Sraw = overlap_at_k(model, k)
            # Remove ruído anti-Hermitiano de arredondamento antes de chamar o
            # LAPACK. O resíduo original ainda é conferido abaixo, então uma
            # entrada de fato inconsistente continua sendo rejeitada.
            Hk = (Hraw + Hraw') / 2
            Sk = (Sraw + Sraw') / 2
            # Um overlap positivo-definido é necessário tanto para o problema de
            # autovalores generalizado Hermitiano quanto para a raiz quadrada de
            # Löwdin usada mais adiante.
            minimum(eigvals(Hermitian(Sk))) > 0 || error("S(k) is not positive definite at k-point $ik")
            sol = eigen(Hermitian(Hk), Hermitian(Sk))
            energies[ik] = Vector{Float64}(sol.values)
            coefficients[ik] = Matrix{ComplexF64}(sol.vectors)
            overlaps[ik] = Matrix{ComplexF64}(Sk)
            if validate
                norm(Hraw-Hraw', Inf) <= atol*max(norm(Hraw,Inf),1) || error("non-Hermitian H(k) at $ik")
                norm(Sraw-Sraw', Inf) <= atol*max(norm(Sraw,Inf),1) || error("non-Hermitian S(k) at $ik")
                norm(sol.vectors' * Sk * sol.vectors - I, Inf) <= 10atol || error("C†SC != I at $ik")
            end
        end
        report_progress(Δt)
        return nothing
    end

    if parallel && nk > 1
        Threads.@threads :dynamic for ik in eachindex(ks)
            solve_one!(ik)
        end
    else
        for ik in eachindex(ks)
            solve_one!(ik)
        end
    end
    BandData(ks, energies, coefficients, overlaps, model.energy_unit)
end

"""
    interpolate_kpath(points, n_per_segment)

Interpola linearmente um caminho por coordenadas recíprocas fracionárias.
Retorna `(kpoints, distance, ticks)`, onde `ticks` marca cada vértice
fornecido. Extremos de segmento compartilhados são emitidos uma única vez.
"""
function interpolate_kpath(points::Vector{<:AbstractVector}, n_per_segment::Int)
    n_per_segment >= 2 || throw(ArgumentError("n_per_segment must be at least 2"))
    kpoints = Vector{Float64}[]; distance = Float64[0.0]; ticks = Float64[0.0]
    for segment in 1:(length(points)-1)
        start, stop = points[segment], points[segment+1]
        for j in 0:(n_per_segment-1)
            # O segmento anterior já emitiu este extremo compartilhado.
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
