"""
    Diagonalização conjunta de operadores de translação unitários e comutantes.

T_1, T_2, T_3 (um por direção de rede da PC) comutam par a par porque
translações ao longo de vetores de rede diferentes comutam entre si. Cada um
é unitário (permutação de fase pura). Diagonalizamos uma combinação
Hermitiana genérica das partes Hermitiana e anti-Hermitiana de cada T_i.
Seus autovetores são, genericamente (isto é, quase certamente para
coeficientes aleatórios), os autovetores *comuns* de T_1, T_2 e T_3
simultaneamente, enquanto a hermiticidade garante uma base ortonormal
dentro de autoespaços degenerados por orbital. Uma vez obtido um autovetor
v dessa combinação, seus autovalores individuais de T_i são recuperados de
forma exata pelo quociente de Rayleigh λ_i = v† T_i v (exato porque v é de
fato um autovetor de cada T_i).

Esta é uma alternativa numérica à solução analítica por órbitas do artigo
(Eqs. 33-38) e à diagonalização sequencial 3D (Eqs. 46-48). Como os
operadores comutam e são normais, os dois procedimentos constroem os mesmos
autoespaços conjuntos e, portanto, os mesmos projetores. Os testes de
regressão do grafeno em `test/runtests.jl` validam os pesos resultantes e a
regra de soma.
"""

"""
    joint_eigen(Ts::Vector{<:AbstractMatrix{ComplexF64}}; rng=Random.default_rng())

Retorna `(V, lambdas)`:

- `V`       : matriz (n × n) cujas colunas são os autovetores comuns
              normalizados de todas as matrizes em `Ts`.
- `lambdas` : matriz (length(Ts) × n); `lambdas[i, k]` é o autovalor de
              `Ts[i]` para o autovetor `V[:, k]`.
"""
function joint_eigen(Ts::Vector{Matrix{ComplexF64}}; rng::AbstractRNG=Random.default_rng())
    @assert !isempty(Ts)
    n = size(Ts[1], 1)

    # Diagonalizamos uma função *Hermitiana* genérica dos operadores unitários
    # comutantes. Uma combinação complexa genérica é normal em aritmética
    # exata, mas um autosolver geral não é obrigado a devolver uma base
    # ortonormal dentro de autoespaços repetidos (o caso usual quando há
    # vários AOs por átomo). A construção Hermitiana garante ortonormalidade
    # pelo próprio autosolver, enquanto coeficientes reais aleatórios separam
    # tuplas de autovalores conjuntos distintas com probabilidade um.
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
            lambdas[i, k] = dot(vk, T * vk)   # <v|T|v>, quociente de Rayleigh exato
        end
    end
    return V, lambdas
end

"""
    check_joint_eigen(Ts, V, lambdas; atol=1e-8)

Checagem de sanidade: resíduo ‖T_i v_k - λ_{i,k} v_k‖ para cada par
operador/autovetor. Retorna o maior resíduo encontrado — deve ser da ordem
de 1e-10 ou menor quando os autovalores da combinação aleatória estão bem
separados. Um resíduo maior geralmente indica que a combinação aleatória
teve degenerescências acidentais e que vale a pena tentar outra semente
aleatória.
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

Converte autovalores (linhas = direções, colunas = índice do autovetor) em
coordenadas fracionárias de k na PC, `f_k = angle(λ)/(2π) mod 1`,
arredondadas para `digits` casas decimais, de modo que autovetores
degenerados que compartilham o mesmo ponto k (desdobrado) da PC possam ser
agrupados por igualdade exata de chave.
"""
function kfrac_from_lambdas(lambdas::AbstractMatrix{ComplexF64}; digits::Int=6)
    fk = mod.(angle.(lambdas) ./ (2π), 1.0)
    fk = round.(fk; digits=digits)
    # Ruído de ponto flutuante pode levar um ângulo infinitesimalmente abaixo
    # de zero a virar 1.0 depois do mod. As coordenadas fracionárias 1 e 0
    # são idênticas e não podem formar grupos de unfolding separados.
    fk[fk .== 1.0] .= 0.0
    return fk
end

"""
    group_by_kfrac(fk::AbstractMatrix{<:Real})

Agrupa índices de coluna de autovetores pelo mesmo ponto k fracionário
(arredondado). Retorna um `Dict{Vector{Float64}, Vector{Int}}`.
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
