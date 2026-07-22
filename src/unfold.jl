"""
    Transformação de Löwdin, pesos de unfolding e função espectral.

Implementa as Eqs. (24)-(28) e (12) do artigo:

    C'_{KN} = S_K^{1/2} C_{KN}
    W^k_{KN} = Σ_{n ∈ group(k)} |F_{k,n}† C'_{KN}|^2
    A(k, E) = Σ_{KN} W^k_{KN} δ(E - E_{KN})   (δ → Lorentziana na prática)
"""

# Mudar qualquer um dos dois valores abaixo é uma mudança de formato de
# arquivo, no mesmo espírito de SCHEMA_NAME/SCHEMA_VERSION em schema.jl.
const UNFOLDED_SCHEMA_NAME = "unfolding.spectral"
const UNFOLDED_SCHEMA_VERSION = (1, 0)

"""
    hermitian_sqrt(S::AbstractMatrix{ComplexF64})

Raiz quadrada de uma matriz Hermitiana positivo-definida `S`, via
autodecomposição. `S` é simetrizada antes, para remover assimetria numérica.
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

Retorna `S^{1/2} * C`, isto é, a matriz de coeficientes MO transformada por
Löwdin (Eq. 24 do artigo). `C` tem dimensão (Nbasis × Nbands).
"""
function lowdin_transform(S::AbstractMatrix{ComplexF64}, C::AbstractMatrix{ComplexF64})
    return hermitian_sqrt(S) * C
end

"""
    unfold_weights(V::AbstractMatrix{ComplexF64}, groups, Cprime::AbstractMatrix{ComplexF64})

Calcula os pesos de unfolding `W^k_{KN}` para cada ponto k desdobrado da PC
(chave de `groups`, produzido por `group_by_kfrac`) e cada banda N da SC
(coluna de `Cprime`, os coeficientes da SC transformados por Löwdin neste
ponto K da SC).

Retorna `Dict{Vector{Float64}, Vector{Float64}}` mapeando cada `f_k` a um
vetor de pesos de tamanho Nbands.
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

Checagem de sanidade das duas regras de soma citadas no artigo (logo após a
Eq. 42): `Σ_kn W^{kn}_{KN} = 1` para cada banda N da SC. Como nosso `W[fk]`
já está somado sobre o grupo degenerado em cada `fk`, isso se reduz a
verificar que, para todo índice de banda N, `Σ_{fk} W[fk][N] ≈ 1`.

Retorna o maior desvio de 1 encontrado entre todas as bandas.
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

Avalia `A(k, E) = Σ_N w_N * L(E - E_N; broadening)` nas energias de
`Egrid`, onde `L` é uma Lorentziana normalizada (Eq. 12/52 do artigo, com δ
substituído por uma Lorentziana de meia-largura `broadening`).
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
    write_unfolded_hdf5(path, kpoints_frac, distance, energies, weights;
                        ticks=Float64[], tick_labels=String[], energy_unit="eV",
                        reference_energies=nothing)

Grava um arquivo HDF5 binário e autodescritivo com o resultado do unfolding
ao longo de um caminho de pontos k, documentado em
`docs/unfolded-hdf5-schema.md`. Por ser HDF5 puro — o pacote já depende de
`HDF5.jl` para o modelo canônico — nenhuma biblioteca de plotagem é
necessária no núcleo; o arquivo resultante pode ser lido por qualquer
programa capaz de ler HDF5 (Plots.jl/HDF5.jl em Julia, h5py+Matplotlib em
Python, gnuplot, etc.) para gerar o gráfico final.

- `kpoints_frac`: matriz `3 × nk`, o ponto k fracionário desdobrado (na base
  da célula de referência) usado em cada posição do caminho;
- `distance`: vetor de tamanho `nk`, distância acumulada ao longo do
  caminho (mesma convenção de `interpolate_kpath`);
- `energies`, `weights`: vetores de tamanho `nk`; cada elemento é um vetor
  de tamanho `nbands` com a energia/peso de cada banda da supercélula
  naquele ponto do caminho — todos os pontos devem ter o mesmo `nbands`;
- `ticks`, `tick_labels`: posições e rótulos opcionais dos vértices de alta
  simetria do caminho (por exemplo `Γ`, `K`, `M`);
- `reference_energies`: opcional, `nbands_pc × nk`. Bandas da célula de
  referência no mesmo caminho (por exemplo de `solve_bands(pc, kpoints)`),
  embutidas no mesmo arquivo em vez de exigir um segundo arquivo separado
  para sobrepor no gráfico final.
"""
function write_unfolded_hdf5(path::AbstractString,
                             kpoints_frac::AbstractMatrix{<:Real},
                             distance::AbstractVector{<:Real},
                             energies::AbstractVector{<:AbstractVector{<:Real}},
                             weights::AbstractVector{<:AbstractVector{<:Real}};
                             ticks::AbstractVector{<:Real}=Float64[],
                             tick_labels::AbstractVector{<:AbstractString}=String[],
                             energy_unit::AbstractString="eV",
                             reference_energies::Union{Nothing,AbstractMatrix{<:Real}}=nothing)
    nk = length(distance)
    size(kpoints_frac, 1) == 3 || throw(DimensionMismatch("kpoints_frac must have three rows"))
    size(kpoints_frac, 2) == nk || throw(DimensionMismatch("kpoints_frac/distance count mismatch"))
    length(energies) == nk || throw(DimensionMismatch("energies/distance count mismatch"))
    length(weights) == nk || throw(DimensionMismatch("weights/distance count mismatch"))
    length(ticks) == length(tick_labels) || throw(DimensionMismatch("ticks/tick_labels count mismatch"))
    nbands = length(first(energies))
    all(length(e) == nbands for e in energies) || error("every k-point must carry the same number of bands")
    all(length(w) == nbands for w in weights) || error("every k-point must carry the same number of bands")
    reference_energies === nothing || size(reference_energies, 2) == nk ||
        throw(DimensionMismatch("reference_energies/distance count mismatch"))

    h5open(path, "w") do file
        schema = create_group(file, "schema")
        _write_string(schema, "name", UNFOLDED_SCHEMA_NAME)
        write(schema, "version", collect(UNFOLDED_SCHEMA_VERSION))
        path_group = create_group(file, "path")
        write(path_group, "kpoints_frac", Matrix{Float64}(kpoints_frac))
        write(path_group, "distance", Vector{Float64}(distance))
        write(path_group, "ticks", Vector{Float64}(ticks))
        write(path_group, "tick_labels", String.(collect(tick_labels)))
        data = create_group(file, "data")
        # Layout nbands × nk: cada coluna é um ponto do caminho, cada linha
        # uma banda da supercélula -- direto para qualquer ferramenta plotar.
        write(data, "energies", reduce(hcat, energies))
        write(data, "weights", reduce(hcat, weights))
        if reference_energies !== nothing
            # Bandas exatas da célula de referência: não têm peso associado,
            # servem só de comparação visual com o unfolding acima.
            refgroup = create_group(file, "reference")
            write(refgroup, "energies", Matrix{Float64}(reference_energies))
        end
        metadata = create_group(file, "metadata")
        _write_string(metadata, "energy_unit", String(energy_unit))
    end
    return path
end

"""
    read_unfolded_hdf5(path)

Lê de volta um arquivo escrito por `write_unfolded_hdf5`. Retorna uma
`NamedTuple` com `kpoints_frac`, `distance`, `ticks`, `tick_labels`,
`energies`, `weights` (as duas últimas como matrizes `nbands × nk`),
`energy_unit`, e `reference_energies` (`nbands_pc × nk`, ou `nothing` se o
arquivo não tiver o grupo `/reference` opcional).
"""
function read_unfolded_hdf5(path::AbstractString)
    h5open(path, "r") do file
        _read_string(file["schema"], "name") == UNFOLDED_SCHEMA_NAME || error("unsupported unfolded HDF5 schema")
        Tuple(read(file["schema/version"])) == UNFOLDED_SCHEMA_VERSION || error("unsupported unfolded schema version")
        reference_energies = haskey(file, "reference") ?
            Matrix{Float64}(read(file["reference/energies"])) : nothing
        (kpoints_frac=Matrix{Float64}(read(file["path/kpoints_frac"])),
         distance=Vector{Float64}(read(file["path/distance"])),
         ticks=Vector{Float64}(read(file["path/ticks"])),
         tick_labels=Vector{String}(read(file["path/tick_labels"])),
         energies=Matrix{Float64}(read(file["data/energies"])),
         weights=Matrix{Float64}(read(file["data/weights"])),
         energy_unit=_read_string(file["metadata"], "energy_unit"),
         reference_energies=reference_energies)
    end
end
