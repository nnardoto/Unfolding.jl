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
export start_unfold_workers, stop_unfold_workers, unfold_worker_status

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
Resolve as bandas e executa o unfolding de um bloco dentro de um worker.

Cada ponto é resolvido e desdobrado imediatamente. As matrizes densas de
overlap e coeficientes permanecem locais ao worker somente durante esse
ponto; apenas energias e pesos retornam ao processo principal.

Se `progress_channel` não for `nothing`, cada ponto publica nele
`(tempo_bandas, tempo_unfolding)` assim que termina, em vez de só ao fim do
bloco inteiro -- é o que dá visibilidade ponto a ponto ao processo
principal mesmo com blocos grandes.
"""
function _distributed_band_unfold_chunk(pcl, ab, sc, Ks, ks, spin, tol, seeds,
                                        progress_channel)
    length(Ks) == length(ks) == length(seeds) ||
        throw(DimensionMismatch("distributed band/unfolding chunk size mismatch"))
    chunk_energies = Vector{Vector{Float64}}(undef, length(ks))
    chunk_weights = Vector{Vector{Float64}}(undef, length(ks))
    for j in eachindex(ks)
        band_dt = @elapsed(point_bands = solve_bands(sc, (Ks[j],);
                                                      spin=spin, parallel=false, progress=false))
        chunk_energies[j] = point_bands.energies[1]
        unfold_dt = @elapsed(chunk_weights[j] = _unfold_weight_at_k(
            pcl, ab, sc.lattice, Ks[j], point_bands.overlaps[1],
            point_bands.coefficients[1], ks[j], tol,
            MersenneTwister(seeds[j])))
        progress_channel === nothing || put!(progress_channel, (band_dt, unfold_dt))
    end
    return (energies=chunk_energies, weights=chunk_weights)
end

const _managed_unfold_workers = Set{Int}()
const _managed_unfold_workers_lock = ReentrantLock()

function _register_unfold_workers(worker_ids)
    lock(_managed_unfold_workers_lock) do
        union!(_managed_unfold_workers, worker_ids)
    end
    return worker_ids
end

function _forget_unfold_workers(worker_ids)
    lock(_managed_unfold_workers_lock) do
        setdiff!(_managed_unfold_workers, worker_ids)
    end
    return worker_ids
end

function _remove_unfold_workers(worker_ids)
    active = intersect(collect(worker_ids), procs())
    isempty(active) || rmprocs(active)
    _forget_unfold_workers(worker_ids)
    return active
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
        _register_unfold_workers(created)
        append!(available, created)
    end

    selected = available[1:requested]
    try
        # Workers iniciados por `-p` ou já presentes na sessão também precisam
        # carregar o pacote antes de desserializar suas funções internas.
        Distributed.remotecall_eval(Main, selected, :(using Unfolding))
    catch
        isempty(created) || _remove_unfold_workers(created)
        rethrow()
    end
    return selected, created
end

"""
    start_unfold_workers(count::Integer)

Prepara `count` processos para bandas e unfolding e retorna seus
identificadores.
Pode ser chamada diretamente em uma célula de notebook. Workers já presentes
na sessão são reutilizados; os que faltarem são criados localmente com uma
thread Julia, BLAS serial e o projeto ativo do notebook.

Os workers criados por esta função permanecem vivos para chamadas seguintes
de `unfold_bandstructure(...; unfold_processes=count)`. Encerre somente os
workers gerenciados pelo pacote com [`stop_unfold_workers`](@ref).
"""
function start_unfold_workers(count::Integer)
    worker_ids, _ = _unfold_worker_processes(Int(count))
    return worker_ids
end

"""
    unfold_worker_status()

Retorna um `NamedTuple` adequado para exibição em notebooks com o número de
threads e threads BLAS do kernel principal, os workers disponíveis e o
subconjunto criado e gerenciado pelo `Unfolding.jl`.
"""
function unfold_worker_status()
    active = filter(!=(myid()), workers())
    managed = lock(_managed_unfold_workers_lock) do
        sort!(collect(intersect(_managed_unfold_workers, Set(active))))
    end
    return (
        julia_threads=Threads.nthreads(),
        blas_threads=LinearAlgebra.BLAS.get_num_threads(),
        available_workers=active,
        managed_workers=managed,
    )
end

"""
    stop_unfold_workers()

Encerra os workers locais criados pelo `Unfolding.jl` e retorna seus
identificadores. Workers que já pertenciam ao notebook, a outro pacote ou a
um cluster não são removidos.
"""
function stop_unfold_workers()
    managed = lock(_managed_unfold_workers_lock) do
        collect(_managed_unfold_workers)
    end
    return _remove_unfold_workers(managed)
end

function _distributed_band_unfold_results(pcl, ab, sc, Ksc, kpc, spin, tol,
                                          point_seeds, worker_ids,
                                          batches_per_process,
                                          report_bands, report_unfolding,
                                          progress::Bool)
    batches_per_process >= 1 ||
        throw(ArgumentError("unfold_batches_per_process must be at least 1"))
    nk = length(kpc)
    nchunks = min(nk, length(worker_ids) * batches_per_process)
    chunk_size = cld(nk, nchunks)
    chunks = [collect(first_index:min(first_index + chunk_size - 1, nk))
              for first_index in 1:chunk_size:nk]

    # Cada worker publica aqui o tempo de bandas/unfolding de cada ponto assim
    # que termina. Sem isso, o progresso só seria conhecido ao fim de cada
    # bloco inteiro (remotecall_fetch síncrono), dando a falsa impressão de
    # blocos de silêncio seguidos de rajadas mesmo com a CPU sempre ocupada.
    progress_channel = nothing
    consumer = nothing
    if progress
        progress_channel = RemoteChannel(() -> Channel{Tuple{Float64,Float64}}(nk))
        consumer = @async begin
            for _ in 1:nk
                band_dt, unfold_dt = take!(progress_channel)
                report_bands(band_dt)
                report_unfolding(unfold_dt)
            end
        end
    end

    # O CachingPool envia este fechamento (e portanto o modelo esparso) uma
    # única vez a cada worker. Chamadas seguintes transmitem apenas os índices
    # do bloco, evitando reenviar `sc` ou matrizes densas a cada lote.
    pool = CachingPool(worker_ids)
    process_chunk = indices -> _distributed_band_unfold_chunk(
        pcl, ab, sc, Ksc[indices], kpc[indices], spin, tol,
        point_seeds[indices], progress_channel)
    chunk_results = try
        asyncmap(chunks; ntasks=min(length(worker_ids), length(chunks))) do indices
            remotecall_fetch(process_chunk, pool, indices)
        end
    finally
        # Libera a cópia do fechamento/modelo quando workers persistentes são
        # mantidos para chamadas futuras.
        clear!(pool)
        # Libera o consumidor de progresso mesmo se algum bloco tiver
        # lançado um erro antes de publicar todos os seus pontos.
        progress_channel === nothing || close(progress_channel)
    end
    progress && try
        wait(consumer)
    catch
        # O canal pode ter sido fechado cedo por causa de um erro acima;
        # a exceção original (se houver) já está se propagando por ali.
    end

    energies = Vector{Vector{Float64}}(undef, nk)
    weights = Vector{Vector{Float64}}(undef, nk)
    for (indices, results) in zip(chunks, chunk_results)
        energies[indices] = results.energies
        weights[indices] = results.weights
    end
    return energies, weights
end

"""
    unfold_bandstructure(pc_lattice, sc::RealSpaceModel, path, n_per_segment;
                        tick_labels=nothing, spin=1, tol=1e-5,
                        rng=Random.default_rng(), parallel=Threads.nthreads() > 1,
                        unfold_processes=0, unfold_batches_per_process=16,
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

Com `parallel=true` e `unfold_processes=0`, a solução das bandas e o
unfolding usam threads Julia.

Defina `unfold_processes=N`, com `N > 0`, para resolver as bandas e executar
o unfolding em `N` processos independentes. Cada worker resolve e desdobra
um ponto antes de receber o próximo bloco, sem devolver as matrizes densas
intermediárias ao processo principal. Workers existentes são reutilizados e
os que faltarem são criados automaticamente com uma thread e BLAS não
paralelizado. Nesse modo, `parallel` não altera o trabalho interno de cada
worker.
Por padrão, somente os workers criados por esta chamada são encerrados ao
final. Use `keep_processes=true` para mantê-los e amortizar a inicialização
em chamadas posteriores.

`unfold_batches_per_process` controla quantos blocos de trabalho são criados
por processo. Valores maiores balanceiam melhor pontos de custo desigual,
mas aumentam o número de chamadas remotas. O modelo fica em cache e as
matrizes densas intermediárias permanecem no worker responsável pelo ponto.

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
                              unfold_batches_per_process::Int=16,
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

    if unfold_processes > 0 && nk > 1
        report_bands = _progress_reporter("Bandas", nk, progress, progress_io)
        report_unfolding = _progress_reporter("Unfolding", nk, progress, progress_io)
        # As sementes são definidas no processo principal, em ordem, para que
        # o resultado não dependa da ordem em que workers terminam os blocos.
        point_seeds = [rand(rng, UInt64) for _ in 1:nk]
        requested_processes = min(unfold_processes, nk)
        progress && println(progress_io, "Processos: preparando ",
                            requested_processes,
                            " worker(s) para bandas + unfolding")
        progress && flush(progress_io)
        worker_ids, created_workers = _unfold_worker_processes(requested_processes)
        progress && println(progress_io, "Processos: ", length(worker_ids),
                            " worker(s) pronto(s), ", length(created_workers),
                            " criado(s) nesta chamada")
        progress && flush(progress_io)
        try
            process_energies, process_weights = _distributed_band_unfold_results(
                pcl, ab, sc, Ksc, kpc, spin, tol, point_seeds,
                worker_ids, unfold_batches_per_process,
                report_bands, report_unfolding, progress)
            for i in eachindex(kpc)
                kpoints_frac[:, i] = kpc[i]
                energies[:, i] = process_energies[i]
                weights[:, i] = process_weights[i]
            end
        finally
            (!keep_processes && !isempty(created_workers)) &&
                _remove_unfold_workers(created_workers)
        end
    else
        bands = solve_bands(sc, Ksc; spin=spin, parallel=parallel,
                            progress=progress, progress_io=progress_io)
        report_unfolding = _progress_reporter(
            "Unfolding", nk, progress && nk > 0, progress_io)

        function store_unfolded_point!(i, point_rng)
            k = kpc[i]
            Δt = @elapsed(point_weights = _unfold_weight_at_k(
                pcl, ab, sc.lattice, Ksc[i], bands.overlaps[i],
                bands.coefficients[i], k, tol, point_rng))
            kpoints_frac[:, i] = k
            energies[:, i] = bands.energies[i]
            weights[:, i] = point_weights
            report_unfolding(Δt)
            return nothing
        end

        if parallel && nk > 1
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
