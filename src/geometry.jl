"""
    Construção dos operadores de translação (geometria).

Implementa a ideia central de Quan, Rybin, Scheffler & Carbogno,
"Efficient band structure unfolding with atom-centered orbitals: General
theory and application", Phys. Rev. B 113, 085112 (2026):

Para um mapeamento *perfeito* (ou estendido por placeholders, ver Seç. II C
do artigo) entre célula primitiva (PC) e supercélula (SC), o operador de
translação da PC transformado por Löwdin, T̃'_PC, torna-se IDÊNTICO a uma
matriz de permutação de fase pura (Eqs. 27 e 45 do artigo) — ou seja, não
depende do overlap S, apenas de:

  1. o mapeamento átomo-a-átomo induzido ao transladar cada átomo da SC por
     um vetor de rede da PC, e
  2. a fase e^{i 2π K·p} adquirida sempre que essa translação cruza a
     fronteira da SC (p = vetor inteiro de wrap na rede da SC).

Isso permite construir T_i (i = 1,2,3, um por direção de rede da PC)
puramente a partir da geometria, em um dado ponto K da SC, e depois
diagonalizar conjuntamente os T_1, T_2, T_3 (que comutam e são unitários)
para obter os autovetores de translação da PC, F_{k,n}, usados na projeção
(Eqs. 28/38 do artigo).
"""

const Vec3 = AbstractVector{<:Real}
const Mat3 = AbstractMatrix{<:Real}

"""
    wrap_fractional(frac::AbstractVector{<:Real})

Retorna `(fracmod, p)` tal que `frac = fracmod + p`, com `fracmod`
componente a componente em `[0, 1)` e `p` um vetor inteiro.
"""
function wrap_fractional(frac::AbstractVector{<:Real})
    p = floor.(frac)
    fracmod = frac .- p
    return fracmod, Int.(round.(p))
end

"""
    periodic_frac_distance(a, b)

Distância periódica mínima entre dois vetores de coordenadas fracionárias,
componente a componente, dobrada para o intervalo `[-0.5, 0.5)`.
"""
function periodic_frac_distance(a::AbstractVector{<:Real}, b::AbstractVector{<:Real})
    d = a .- b
    return mod.(d .+ 0.5, 1.0) .- 0.5
end

"""
    match_atom(newpos_cart, sc_positions_cart, Ainv; tol=1e-5)

Dada uma posição cartesiana `newpos_cart` (resultado de transladar algum
átomo da SC por um vetor de rede da PC), encontra a qual átomo da SC (índice
de coluna em `sc_positions_cart`) ela coincide *módulo um vetor de rede da
SC*, e retorna `(jatom, p)`, onde `p` é o vetor inteiro de wrap necessário:

    newpos_cart ≈ sc_positions_cart[:, jatom] + A * p

`Ainv` é a inversa da matriz de rede da SC, `A` (colunas = vetores de rede),
usada para converter para coordenadas fracionárias.

Lança um erro se nenhum átomo corresponder dentro de `tol` (em unidades
fracionárias) — isso indica um mapeamento/`M` errado, ou um caso de
defeito/vacância que precisaria de orbitais placeholder (ainda não
implementado aqui; ver Seç. II C do artigo e o `TODO` em
`translation_operator`).
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
            # p é tal que frac_new = frac_j + p (dentro da tolerância tol).
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

Contabilidade de como os AOs estão organizados por átomo na base da SC.

- `positions`  : (3, Natoms), posições cartesianas dos átomos.
- `norb`       : vetor de tamanho Natoms, número de AOs (funções de base)
                 em cada átomo.
- `offsets`    : vetor de tamanho Natoms, índice (base 0) do primeiro AO de
                 cada átomo dentro da ordenação completa (Nbasis,).
"""
struct AtomBasis
    positions::Matrix{Float64}
    norb::Vector{Int}
    offsets::Vector{Int}
end

"""
    AtomBasis(model::RealSpaceModel)

Constrói o `AtomBasis` do projetor a partir de um `RealSpaceModel`, usando
`reference_positions` (não `positions`) e `norb`. O projetor de unfolding
deve sempre ser construído sobre a topologia ideal/de referência, mesmo
quando a supercélula tem uma geometria física relaxada ou defeituosa — este
construtor existe para que o usuário final não precise lembrar disso.
"""
AtomBasis(model::RealSpaceModel) = AtomBasis(model.reference_positions, model.norb)

function AtomBasis(positions::AbstractMatrix{<:Real}, norb::AbstractVector{<:Integer})
    offsets = cumsum(vcat(0, norb))[1:end-1]
    return AtomBasis(Matrix{Float64}(positions), Vector{Int}(norb), Vector{Int}(offsets))
end

nbasis(ab::AtomBasis) = sum(ab.norb)
natoms(ab::AtomBasis) = length(ab.norb)

"""
    translation_operator(pc_vector, ab::AtomBasis, Ainv, Kfrac; tol=1e-5)

Constrói a matriz unitária (Nbasis × Nbasis) que representa o operador de
translação de Bloch, no espaço-K da SC, para a translação por um vetor de
rede `pc_vector` da PC (cartesiano, comprimento 3), avaliado no ponto K
fracionário `Kfrac` da SC (comprimento 3, na base recíproca B da SC — isto
é, K = Kfrac · B).

Esta função já constrói diretamente o operador transformado por Löwdin,
T'_PC (Eq. 27 do artigo), portanto nenhum overlap é necessário aqui — só o
mapeamento entre átomos e as fases.
"""
function translation_operator(pc_vector::AbstractVector{<:Real},
                               ab::AtomBasis,
                               Ainv::AbstractMatrix{<:Real},
                               Kfrac::AbstractVector{<:Real};
                               tol::Real=1e-5)
    n = nbasis(ab)
    T = zeros(ComplexF64, n, n)
    for iatom in 1:natoms(ab)
        # O artigo define o operador de translação como
        # T|phi_LJ> = |phi_(L-1)J> (Eq. 15): ele move um AO por -a_i.
        # Usar +a_i aqui constrói o operador inverso, o que gera fases de
        # projetor conjugadas (isso já aparece no modelo de brinquedo da Eq. 42).
        newpos = ab.positions[:, iatom] .- pc_vector
        jatom, p = match_atom(newpos, ab.positions, Ainv; tol=tol)
        if ab.norb[jatom] != ab.norb[iatom]
            error("translation_operator: orbital-count mismatch mapping atom " *
                  "$iatom ($(ab.norb[iatom]) AOs) -> atom $jatom " *
                  "($(ab.norb[jatom]) AOs). This happens for defects/dopants " *
                  "with different basis sets; the paper handles this via " *
                  "placeholder orbitals (Sec. II C) which is a TODO here.")
        end
        # Nossas somas de Bloch usam a convenção de fase do artigo. Se o AO
        # mapeado difere pelo vetor de rede da SC, A*p, o elemento de matriz
        # ganha o fator de fase exp(-i K.A*p).
        phase = cispi(-2 * dot(Kfrac, p))
        for q in 1:ab.norb[iatom]
            T[ab.offsets[jatom]+q, ab.offsets[iatom]+q] = phase
        end
    end
    return T
end

"""
    translation_operators(pc_lattice, ab::AtomBasis, sc_lattice, Kfrac; tol=1e-5)

Função de conveniência: constrói os três operadores de translação T_1, T_2,
T_3, um para cada coluna de `pc_lattice` (matriz 3×3, vetores de rede da PC
como colunas), dada a matriz de rede da SC `sc_lattice` (3×3, colunas =
vetores de rede da SC) e o ponto K fracionário da SC, `Kfrac`.
"""
function translation_operators(pc_lattice::AbstractMatrix{<:Real},
                                ab::AtomBasis,
                                sc_lattice::AbstractMatrix{<:Real},
                                Kfrac::AbstractVector{<:Real};
                                tol::Real=1e-5)
    Ainv = inv(Matrix{Float64}(sc_lattice))
    return [translation_operator(pc_lattice[:, i], ab, Ainv, Kfrac; tol=tol) for i in 1:3]
end
