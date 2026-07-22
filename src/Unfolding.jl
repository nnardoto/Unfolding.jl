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
export read_cp2k_csr, convert_cp2k_to_hdf5
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
    unfold_bandstructure(pc_lattice, sc::RealSpaceModel, path, n_per_segment;
                        tick_labels=nothing, spin=1, tol=1e-5, rng=Random.default_rng())

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

Retorna uma `UnfoldedBandStructure`.
"""
function unfold_bandstructure(pc_lattice::AbstractMatrix{<:Real},
                              sc::RealSpaceModel,
                              path::Vector{<:AbstractVector},
                              n_per_segment::Int;
                              tick_labels::Union{Nothing,Vector{<:AbstractString}}=nothing,
                              spin::Int=1,
                              tol::Real=1e-5,
                              rng::AbstractRNG=Random.default_rng())
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

    for (i, k) in enumerate(kpc)
        Ksc = mod.(transform' * k .+ 0.5, 1.0) .- 0.5
        bands = solve_bands(sc, [Ksc]; spin=spin)
        W, _, _ = unfold_supercell(pcl, ab, sc.lattice, Ksc, bands.overlaps[1], bands.coefficients[1];
            tol=tol, rng=rng)
        # Um ponto K da SC se desdobra em det(M) pontos k da PC. Mantém o
        # membro que está sobre o caminho de referência pedido.
        target = argmin([norm(periodic_frac_distance(key, k)) for key in keys(W)])
        key = collect(keys(W))[target]
        kpoints_frac[:, i] = k
        energies[:, i] = bands.energies[1]
        weights[:, i] = W[key]
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
