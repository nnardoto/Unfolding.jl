"""
    Unfolding

Ferramentas de unfolding de bandas para bases atômicas localizadas e
não-ortogonais.

O pacote está dividido em três camadas:

1. `schema.jl` e `bands.jl` definem um modelo em espaço real, independente
   de código de estrutura eletrônica;
2. `geometry.jl`, `jointdiag.jl` e `unfold.jl` implementam o método de
   Quan *et al.*, Phys. Rev. B 113, 085112 (2026);
3. `converters/` contém os importadores específicos de cada código,
   inicialmente para o CP2K.

Veja `docs/paper-implementation.md` para a derivação equação por equação.
"""
module Unfolding

using Distributed
using HDF5
using LinearAlgebra
using Random
using SparseArrays

include("schema.jl")
include("bands.jl")
include("geometry.jl")
include("jointdiag.jl")
include("unfold.jl")
include("converters/cp2k.jl")

export RealSpaceModel, BandData, write_model, read_model, validate_model, model_summary
export nspin, nbasis, nimages
export fourier_matrix, hamiltonian_at_k, overlap_at_k, solve_bands, interpolate_kpath
export AtomBasis, natoms, translation_operator, translation_operators, periodic_frac_distance
export joint_eigen, check_joint_eigen, kfrac_from_lambdas, group_by_kfrac
export hermitian_sqrt, lowdin_transform, unfold_weights, check_sum_rule_over_k
export spectral_function, write_unfolded_hdf5, read_unfolded_hdf5, unfold_supercell
export CP2KFiles, read_cp2k_csr, convert_cp2k_to_hdf5
export UnfoldedBandStructure, unfold_bandstructure

"""
    unfold_supercell(pc_lattice, ab, sc_lattice, Kfrac, S, C; tol=1e-5, rng)

Desdobra todas as colunas da matriz de coeficientes da supercélula `C`, em
um único ponto `Kfrac` da supercélula. `S` é a matriz de overlap no mesmo
ponto, e `ab` descreve a topologia ideal/de referência dos AOs da
supercélula.

Retorna `(W, V, lambdas)`, onde:

- `W[k]` contém um peso de unfolding por banda da supercélula, no ponto
  fracionário `k` da célula primitiva;
- as colunas de `V` são os autovetores comuns das três translações
  primitivas;
- `lambdas[i,n]` é o autovalor da translação `i` para a coluna `n` de `V`.

As quatro etapas do corpo da função correspondem, em ordem, à construção de
`T_PC` do artigo, ao tratamento simultâneo das Eqs. 46-48, à identificação
dos subespaços `k` desdobrados, e ao peso projetado por Löwdin da Eq. 28.
"""
function unfold_supercell(pc_lattice::AbstractMatrix{<:Real},
                          ab::AtomBasis,
                          sc_lattice::AbstractMatrix{<:Real},
                          Kfrac::AbstractVector{<:Real},
                          S::AbstractMatrix{<:Number},
                          C::AbstractMatrix{<:Number};
                          tol::Real=1e-5,
                          rng::AbstractRNG=Random.default_rng())
    # 1. Construção de T_PC: matrizes de translação puramente de fase-
    #    permutação (portanto unitárias). Armazenamento denso é intencional
    #    aqui, pois seus autovetores comuns também são densos.
    Ts = Matrix{ComplexF64}.(translation_operators(pc_lattice, ab, sc_lattice, Kfrac; tol=tol))

    # 2. Diagonalização conjunta (Eqs. 46-48): obtém autovetores/autovalores
    #    comuns às três translações.
    V, lambdas = joint_eigen(Ts; rng=rng)

    # 3. Identificação dos subespaços k desdobrados: autovalores de mesma
    #    fase identificam o subespaço projetor degenerado de um ponto k da
    #    célula primitiva.
    groups = group_by_kfrac(kfrac_from_lambdas(lambdas))

    # 4. Peso projetado por Löwdin (Eq. 28): C' = S^(1/2)C (Eq. 24) é formado
    #    uma única vez, e a projeção usa esse resultado.
    W = unfold_weights(V, groups, lowdin_transform(ComplexF64.(S), ComplexF64.(C)))
    return W, V, lambdas
end

"""
    UnfoldedBandStructure

Resultado de `unfold_bandstructure`: o unfolding da supercélula ao longo de
um caminho de pontos k da célula de referência. Mesmo layout usado por
`write_unfolded_hdf5`, para que o resultado possa ser exportado diretamente.

- `kpoints_frac`: `3 × nk`, o ponto k fracionário (base da célula de
  referência) requisitado em cada posição do caminho;
- `distance`, `ticks`, `tick_labels`: mesma convenção de `interpolate_kpath`;
- `energies`, `weights`: `nbands × nk`, energia e peso de cada banda da
  supercélula em cada ponto do caminho;
- `energy_unit`: unidade de `energies`, herdada do modelo da supercélula.
"""
struct UnfoldedBandStructure
    kpoints_frac::Matrix{Float64}
    distance::Vector{Float64}
    ticks::Vector{Float64}
    tick_labels::Vector{String}
    energies::Matrix{Float64}
    weights::Matrix{Float64}
    energy_unit::String
end

"""
Calcula somente o vetor de pesos selecionado para um ponto do caminho.

Esta função pequena e sem mutação do resultado global é usada tanto pelas
threads locais quanto pelos workers de `Distributed`.
"""
function _unfold_weight_at_k(pcl, ab, sc_lattice, K, S, C, k, tol, point_rng)
    W, _, _ = unfold_supercell(pcl, ab, sc_lattice, K, S, C;
                               tol=tol, rng=point_rng)
    keys_W = collect(keys(W))
    target = argmin(norm(periodic_frac_distance(key, k)) for key in keys_W)
    return W[keys_W[target]]
end

"""
Executa um bloco independente de pontos dentro de um worker.

Cada bloco recebe somente os overlaps e coeficientes dos seus próprios
pontos. Assim, o conjunto completo de autovetores não é replicado em todos
os processos.
"""
function _distributed_unfold_chunk(pcl, ab, sc_lattice, Ks, Ss, Cs, ks, tol, seeds)
    length(Ks) == length(Ss) == length(Cs) == length(ks) == length(seeds) ||
        throw(DimensionMismatch("distributed unfolding chunk size mismatch"))
    [_unfold_weight_at_k(pcl, ab, sc_lattice, Ks[j], Ss[j], Cs[j], ks[j],
                         tol, MersenneTwister(seeds[j]))
     for j in eachindex(ks)]
end

function _unfold_worker_processes(requested::Int)
    requested >= 0 ||
        throw(ArgumentError("unfold_processes must be nonnegative"))
    requested == 0 && return Int[], Int[]

    available = filter(!=(myid()), workers())
    created = Int[]
    if length(available) < requested
        active_project = Base.active_project()
        project_directory = active_project === nothing ? nothing : dirname(active_project)
        worker_flags = project_directory === nothing ?
            `--threads=1` : `--project=$(project_directory) --threads=1`
        created = addprocs(requested - length(available);
                           dir=pwd(),
                           exeflags=worker_flags,
                           enable_threaded_blas=false)
        append!(available, created)
    end

    selected = available[1:requested]
    try
        # Workers iniciados por `-p` ou já presentes na sessão também precisam
        # carregar o pacote antes de desserializar suas funções internas.
        Distributed.remotecall_eval(Main, selected, :(using Unfolding))
    catch
        isempty(created) || rmprocs(created)
        rethrow()
    end
    return selected, created
end

function _distributed_unfold_weights(pcl, ab, sc_lattice, Ksc, bands, kpc,
                                     tol, point_seeds, worker_ids,
                                     batches_per_process, report_progress)
    batches_per_process >= 1 ||
        throw(ArgumentError("unfold_batches_per_process must be at least 1"))
    nk = length(kpc)
    nchunks = min(nk, length(worker_ids) * batches_per_process)
    chunk_size = cld(nk, nchunks)
    chunks = [collect(first_index:min(first_index + chunk_size - 1, nk))
              for first_index in 1:chunk_size:nk]

    pool = WorkerPool(worker_ids)
    chunk_results = asyncmap(chunks; ntasks=min(length(worker_ids), length(chunks))) do indices
        worker = take!(pool)
        try
            result = remotecall_fetch(
                _distributed_unfold_chunk, worker,
                pcl, ab, sc_lattice,
                Ksc[indices],
                bands.overlaps[indices],
                bands.coefficients[indices],
                kpc[indices],
                tol,
                point_seeds[indices],
            )
            foreach(_ -> report_progress(), indices)
            return result
        finally
            put!(pool, worker)
        end
    end

    weights = Vector{Vector{Float64}}(undef, nk)
    for (indices, results) in zip(chunks, chunk_results)
        weights[indices] = results
    end
    return weights
end

"""
    unfold_bandstructure(pc_lattice, sc::RealSpaceModel, path, n_per_segment;
                        tick_labels=nothing, spin=1, tol=1e-5,
                        rng=Random.default_rng(), parallel=Threads.nthreads() > 1,
                        unfold_processes=0, unfold_batches_per_process=4,
                        keep_processes=false, progress=false)

Camada de conveniência para o caso de uso mais comum: dada a rede da célula
de referência (`pc_lattice`) e uma supercélula já carregada (`sc`), resolve
as bandas da supercélula e as desdobra em cada ponto de um caminho de
pontos k fracionários da célula de referência. Substitui o laço manual de
~15 linhas (interpolar o caminho, mapear cada `k` para o `K` da SC, resolver
`solve_bands`, montar o `AtomBasis`, chamar `unfold_supercell`, escolher o
`k` desdobrado mais próximo do pedido) que antes vivia copiado em cada
script de exemplo.

Esta função cuida apenas do unfolding da supercélula. As bandas de
referência da célula primitiva (para sobrepor num gráfico, por exemplo) não
são calculadas aqui — use `solve_bands(pc, kpoints)` separadamente com
`kpoints = eachcol(result.kpoints_frac)`, e opcionalmente embuta o
resultado no mesmo arquivo com
`write_unfolded_hdf5(path, result; reference_energies=...)`.

Lança um erro cedo se `sc.lattice` não for um múltiplo inteiro de
`pc_lattice`, em vez de deixar o mapeamento atômico falhar mais adiante com
uma mensagem menos direta.

Com `parallel=true`, a solução das bandas usa threads Julia. O unfolding
também usa threads quando `unfold_processes=0`.

Defina `unfold_processes=N`, com `N > 0`, para executar a etapa de unfolding
em `N` processos independentes. Workers existentes são reutilizados e os que
faltarem são criados automaticamente com uma thread e BLAS não paralelizado.
Por padrão, somente os workers criados por esta chamada são encerrados ao
final. Use `keep_processes=true` para mantê-los e amortizar a inicialização
em chamadas posteriores.

`unfold_batches_per_process` controla quantos blocos de trabalho são criados
por processo. Valores maiores balanceiam melhor pontos de custo desigual,
mas aumentam a comunicação. Cada ponto e suas matrizes são enviados a apenas
um worker, independentemente do número de processos.

A ordem do caminho e a reprodutibilidade para um `rng` explícito são
preservadas em todos os modos.

Use `progress=true` para exibir o avanço das etapas `Bandas` e `Unfolding`.

Retorna uma `UnfoldedBandStructure`.
"""
function unfold_bandstructure(pc_lattice::AbstractMatrix{<:Real},
                              sc::RealSpaceModel,
                              path::Vector{<:AbstractVector},
                              n_per_segment::Int;
                              tick_labels::Union{Nothing,Vector{<:AbstractString}}=nothing,
                              spin::Int=1,
                              tol::Real=1e-5,
                              rng::AbstractRNG=Random.default_rng(),
                              parallel::Bool=Threads.nthreads() > 1,
                              unfold_processes::Int=0,
                              unfold_batches_per_process::Int=4,
                              keep_processes::Bool=false,
                              progress::Bool=false,
                              progress_io::IO=stderr)
    unfold_processes >= 0 ||
        throw(ArgumentError("unfold_processes must be nonnegative"))
    unfold_batches_per_process >= 1 ||
        throw(ArgumentError("unfold_batches_per_process must be at least 1"))
    kpc, distance, ticks = interpolate_kpath(path, n_per_segment)
    labels = tick_labels === nothing ? fill("", length(ticks)) : String.(collect(tick_labels))
    length(labels) == length(ticks) || throw(DimensionMismatch("tick_labels/ticks count mismatch"))

    pcl = Matrix{Float64}(pc_lattice)
    transform = round.(Int, pcl \ sc.lattice)
    norm(pcl * transform - sc.lattice, Inf) <= tol * max(norm(sc.lattice, Inf), 1) ||
        error("unfold_bandstructure: sc.lattice is not an integer multiple of pc_lattice " *
              "(check the primitive-cell lattice or the M matrix)")

    ab = AtomBasis(sc)
    nk = length(kpc)
    nb = nbasis(sc)
    kpoints_frac = zeros(Float64, 3, nk)
    energies = zeros(Float64, nb, nk)
    weights = zeros(Float64, nb, nk)
    Ksc = [mod.(transform' * k .+ 0.5, 1.0) .- 0.5 for k in kpc]
    bands = solve_bands(sc, Ksc; spin=spin, parallel=parallel,
                        progress=progress, progress_io=progress_io)
    report_progress = _progress_reporter("Unfolding", nk, progress && nk > 0, progress_io)

    function store_unfolded_point!(i, point_rng)
        k = kpc[i]
        point_weights = _unfold_weight_at_k(
            pcl, ab, sc.lattice, Ksc[i], bands.overlaps[i],
            bands.coefficients[i], k, tol, point_rng)
        kpoints_frac[:, i] = k
        energies[:, i] = bands.energies[i]
        weights[:, i] = point_weights
        report_progress()
        return nothing
    end

    if unfold_processes > 0 && nk > 1
        # As sementes são definidas no processo principal, em ordem, para que
        # o resultado não dependa da ordem em que workers terminam os blocos.
        point_seeds = [rand(rng, UInt64) for _ in 1:nk]
        requested_processes = min(unfold_processes, nk)
        progress && println(progress_io, "Processos: preparando ",
                            requested_processes, " worker(s) para o unfolding")
        worker_ids, created_workers = _unfold_worker_processes(requested_processes)
        progress && println(progress_io, "Processos: ", length(worker_ids),
                            " worker(s) pronto(s), ", length(created_workers),
                            " criado(s) nesta chamada")
        try
            process_weights = _distributed_unfold_weights(
                pcl, ab, sc.lattice, Ksc, bands, kpc, tol, point_seeds,
                worker_ids, unfold_batches_per_process, report_progress)
            for i in eachindex(kpc)
                kpoints_frac[:, i] = kpc[i]
                energies[:, i] = bands.energies[i]
                weights[:, i] = process_weights[i]
            end
        finally
            (!keep_processes && !isempty(created_workers)) &&
                rmprocs(created_workers)
        end
    elseif parallel && nk > 1
        # Um RNG separado por ponto evita compartilhar estado mutável entre
        # threads. As sementes são geradas em ordem, antes do trabalho
        # paralelo, mantendo a execução reprodutível para um `rng` explícito.
        point_rngs = [MersenneTwister(rand(rng, UInt64)) for _ in 1:nk]
        Threads.@threads :dynamic for i in eachindex(kpc)
            store_unfolded_point!(i, point_rngs[i])
        end
    else
        for i in eachindex(kpc)
            store_unfolded_point!(i, rng)
        end
    end

    UnfoldedBandStructure(kpoints_frac, distance, ticks, labels, energies, weights, sc.energy_unit)
end

"""
    unfold_bandstructure(M::AbstractMatrix{<:Integer}, sc::RealSpaceModel, path, n_per_segment; kwargs...)

Mesmo método acima, mas recebendo a matriz de transformação inteira `M`
(`A_sc = A_pc * M`, Eq. 1 do artigo) em vez dos vetores de rede da célula de
referência. A rede de referência é recuperada exatamente por
`pc_lattice = sc.lattice * inv(M)`, útil quando você já conhece `M` (por
exemplo `[2 0 0; 0 2 0; 0 0 1]` para uma supercélula 2×2×1) e não quer
manter os vetores de rede da PC à parte só para chamar esta função.

Como `M` e `pc_lattice` (o método com vetores de rede, em unidades de
comprimento) têm significados físicos completamente diferentes, o
despacho múltiplo aqui é seguro apenas porque os dois têm tipo de elemento
diferente: `M` deve ser uma matriz de inteiros. Se você passar a matriz
errada (por exemplo `M` convertida para `Float64`), a checagem de
consistência já existente no outro método -- de que a rede da supercélula é
um múltiplo inteiro da rede de referência recebida -- captura o erro cedo
em vez de produzir um resultado fisicamente sem sentido.
"""
function unfold_bandstructure(M::AbstractMatrix{<:Integer}, sc::RealSpaceModel,
                              path::Vector{<:AbstractVector}, n_per_segment::Int; kwargs...)
    size(M) == (3, 3) || throw(DimensionMismatch("M must be 3x3"))
    Mf = Matrix{Float64}(M)
    abs(det(Mf)) > eps(Float64) || error("unfold_bandstructure: M must be invertible (det(M) != 0)")
    pc_lattice = sc.lattice * inv(Mf)
    unfold_bandstructure(pc_lattice, sc, path, n_per_segment; kwargs...)
end

"""
    unfold_bandstructure(M, files::CP2KFiles, path, n_per_segment; kwargs...)

Atalho de ponta a ponta para um cálculo CP2K real. Importa o modelo a partir
dos quatro caminhos em `files` e executa o unfolding usando somente a matriz
de transformação inteira `M`, sem transcrição manual de geometria ou base.
"""
function unfold_bandstructure(M::AbstractMatrix{<:Integer}, files::CP2KFiles,
                              path::Vector{<:AbstractVector}, n_per_segment::Int;
                              spins=[1], validate=true, kwargs...)
    length(spins) == 1 ||
        error("unfold_bandstructure(files): select exactly one spin channel")
    sc = read_cp2k_csr(files; spins=spins, validate=validate)
    unfold_bandstructure(M, sc, path, n_per_segment; spin=1, kwargs...)
end

"""
    write_unfolded_hdf5(path, result::UnfoldedBandStructure; reference_energies=nothing)

Grava um `UnfoldedBandStructure` diretamente, sem que o usuário precise
desmontar os campos manualmente. Aceita opcionalmente `reference_energies`
(`nbands_pc × nk`), as bandas da célula de referência no mesmo caminho, que
são embutidas no grupo `/reference` do mesmo arquivo -- ver
`docs/unfolded-hdf5-schema.md`.
"""
function write_unfolded_hdf5(path::AbstractString, result::UnfoldedBandStructure;
                             reference_energies::Union{Nothing,AbstractMatrix{<:Real}}=nothing)
    nk = size(result.energies, 2)
    energies = [result.energies[:, i] for i in 1:nk]
    weights = [result.weights[:, i] for i in 1:nk]
    write_unfolded_hdf5(path, result.kpoints_frac, result.distance, energies, weights;
        ticks=result.ticks, tick_labels=result.tick_labels, energy_unit=result.energy_unit,
        reference_energies=reference_energies)
end

end
