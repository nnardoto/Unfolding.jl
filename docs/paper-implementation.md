# Do artigo ao código

## Referência

Esta implementação segue o formalismo de:

> J. Quan, N. Rybin, M. Scheffler e C. Carbogno, "Efficient band structure unfolding with atom-centered orbitals: General theory and application", *Physical Review B* **113**, 085112 (2026). DOI: [10.1103/7xym-7388](https://doi.org/10.1103/7xym-7388).

O objetivo central do artigo é calcular band unfolding diretamente em uma base LCAO de orbitais atômicos não ortogonais. O método evita representar os estados em ondas planas e não exige calcular explicitamente os autoestados da célula primitiva para construir o projetor.

Este documento descreve quais equações do artigo são utilizadas, onde aparecem no código e quais adaptações numéricas foram feitas.

## Convenções adotadas

As matrizes de rede são armazenadas com os vetores de rede nas **colunas**:

```text
A = [A1 A2 A3]
```

Se `a` é a rede de referência e `A` é a supercélula, a Eq. 1 do artigo é

```math
A = aM,
```

onde `M` é uma matriz inteira e `|det(M)|` é o número de réplicas da célula de referência. No código:

```julia
M = round.(Int, pc_lattice \ sc_lattice)
```

O artigo escreve as coordenadas recíprocas como vetores-linha e obtém `F_K = f_k M` na Eq. 7. O código usa vetores-coluna; portanto, a mesma relação aparece como

```julia
Kfrac = M' * kfrac
```

seguida de uma redução módulo um para levar o ponto à primeira zona de Brillouin da supercélula.

O HDF5 usa a seguinte transformada de Fourier:

```math
H(k)=\sum_R e^{+i2\pi k\cdot R}H(R),\qquad
S(k)=\sum_R e^{+i2\pi k\cdot R}S(R).
```

Aqui `k` é fracionário na base recíproca e `R` é inteiro na base direta. A mesma convenção aparece na Eq. 19 do artigo. A reciprocidade `H(-R)=H(R)^\dagger` e `S(-R)=S(R)^\dagger` é verificada quando o modelo é carregado.

## Fluxo matemático e computacional

Em forma compacta, uma avaliação em um ponto `K` segue esta sequência:

```julia
sc = read_model("supercell.h5")
bands = solve_bands(sc, [Kfrac])

# A rede primitiva é a referência do projetor; os estados vêm da supercélula.
ao = AtomBasis(sc.reference_positions, sc.norb)
W, F, lambdas = unfold_supercell(
    pc_lattice,
    ao,
    sc.lattice,
    Kfrac,
    bands.overlaps[1],
    bands.coefficients[1],
)
```

`Kfrac` está na base recíproca da supercélula. As chaves de `W` são os pontos `k` fracionários na base recíproca da célula de referência; cada valor é um vetor com um peso por banda da supercélula. `AtomBasis(sc.reference_positions, sc.norb)` acima pode ser escrito como `AtomBasis(sc)`.

Esta seção mostra o mapeamento equação-por-equação num único ponto `K`, como referência para quem quer entender ou estender o método. Para uso em um projeto real, ao longo de um caminho de pontos `k`, veja `unfold_bandstructure` no README -- ele encapsula exatamente esta sequência (mais a escolha do `k` desdobrado correto a cada ponto do caminho).

### 1. Obter o Hamiltoniano e o overlap no ponto K

As Eqs. 18–20 do artigo definem o problema generalizado em uma base LCAO não ortogonal:

```math
H_K C_{KN}=E_{KN}S_K C_{KN},
```

com normalização

```math
C_{KM}^{\dagger}S_KC_{KN}=\delta_{MN}.
```

No projeto, `fourier_matrix` transforma as matrizes reais `H(R)` e `S(R)` armazenadas no HDF5. `solve_bands` resolve o problema generalizado e verifica numericamente `C†SC = I`.

Arquivos e funções:

- `src/bands.jl`: `fourier_matrix`, `hamiltonian_at_k`, `overlap_at_k` e `solve_bands`;
- `src/schema.jl`: armazenamento de `H(R)`, `S(R)` e das translações `R`.

### 2. Construir o operador de translação da célula de referência

Na Eq. 15, o artigo define a translação de um orbital para a célula anterior:

```math
\widetilde T|\phi_{LJ}\rangle=|\phi_{(L-1)J}\rangle.
```

`translation_operator` aplica exatamente a translação `-a_i` a cada sítio de referência. O sítio traduzido é identificado módulo a rede da supercélula. Quando a translação atravessa a fronteira periódica, o elemento não nulo da matriz recebe a fase de Bloch correspondente.

Para coordenadas fracionárias `Kfrac` da supercélula e um vetor inteiro de envolvimento periódico `p`, o código usa

```math
e^{-i2\pi K_{\mathrm{frac}}\cdot p}.
```

O resultado é uma matriz unitária de permutação com fases: cada coluna possui um único elemento de módulo um. Essa é a estrutura das Eqs. 29–32 e 41 do artigo.

O mapeamento é construído com `reference_positions`, e não necessariamente com as posições físicas relaxadas. Isso implementa a ideia discutida após a Eq. 38: os projetores devem obedecer à simetria da célula de referência mesmo quando o Hamiltoniano da supercélula quebra essa simetria.

Arquivos e funções:

- `src/geometry.jl`: `match_atom`, `translation_operator` e `translation_operators`;
- `src/schema.jl`: `reference_positions` preserva a topologia ideal usada pelo projetor.

### 3. Eliminar a métrica não ortogonal por Löwdin

A simplificação decisiva do artigo aparece nas Eqs. 24–28. Os coeficientes dos estados da supercélula são transformados por

```math
C'_{KN}=S_K^{1/2}C_{KN}.
```

Ao mesmo tempo, a Eq. 27 mostra que o operador de translação transformado por Löwdin se reduz à matriz de permutação ortogonal:

```math
\widetilde T'_{PC}=S_K^{-1/2}\widetilde T_{PC}S_K^{-1/2}=T_{PC}.
```

É por essa igualdade que `translation_operator` pode ser construído somente com geometria, conectividade dos orbitais e fases de Bloch. O overlap não entra na construção do operador; ele entra uma única vez em `C' = S^(1/2)C`.

No código:

- `hermitian_sqrt` diagonaliza a parte hermitiana de `S_K` e exige autovalores positivos;
- `lowdin_transform` calcula `S_K^(1/2) C_K`;
- `unfold_supercell` passa os coeficientes transformados ao projetor.

### 4. Obter uma base comum dos três operadores de translação

Em três dimensões, o artigo propõe diagonalizar sequencialmente os três operadores com redução dos subespaços, Eqs. 46–48. As Eqs. 33–38 também fornecem uma solução analítica para cada órbita de permutação.

O projeto usa uma variante numérica equivalente. Como os três operadores de translação são unitários, normais e comutam entre si, eles admitem uma base ortonormal comum. `joint_eigen` diagonaliza uma combinação hermitiana genérica:

```math
Q=\sum_i\left[\alpha_i\frac{T_i+T_i^\dagger}{2}
+\beta_i\frac{T_i-T_i^\dagger}{2i}\right].
```

Os coeficientes reais `αᵢ` e `βᵢ` são pseudoaleatórios. Com probabilidade um, essa combinação separa tuplas distintas de autovalores conjuntos. Dentro de um subespaço verdadeiramente degenerado, qualquer base ortonormal é válida para construir o mesmo projetor da Eq. 4.

Depois da diagonalização, os autovalores individuais são recuperados pelos quocientes de Rayleigh

```math
\lambda_{i,n}=F_n^\dagger T_iF_n.
```

`check_joint_eigen` mede os resíduos `||T_i F_n - λ_{i,n}F_n||`. A escolha é uma diferença de implementação em relação às Eqs. 33–38 e 46–48, mas não altera o subespaço projetado nem a Eq. 28.

Arquivo e funções:

- `src/jointdiag.jl`: `joint_eigen`, `check_joint_eigen`, `kfrac_from_lambdas` e `group_by_kfrac`.

### 5. Identificar os pontos k desdobrados

O autovalor de uma translação primitiva é

```math
\lambda_i=e^{i2\pi f_{k,i}}.
```

Assim, `kfrac_from_lambdas` obtém

```math
f_{k,i}=\frac{\arg(\lambda_i)}{2\pi}\pmod 1.
```

Autovetores com a mesma tupla `(f_k1,f_k2,f_k3)` são agrupados. Esses grupos são os subespaços degenerados que entram na soma sobre `n` da Eq. 28. Para uma supercélula com multiplicidade `m=|det(M)|`, devem aparecer `m` pontos `k` associados a cada `K`, de acordo com as Eqs. 5–8.

### 6. Calcular o peso de unfolding

A Eq. 10 define o peso como o valor esperado do projetor da célula de referência:

```math
W^k_{KN}=\langle\Psi_{KN}|P_k|\Psi_{KN}\rangle.
```

Depois da transformação de Löwdin, a expressão usada pelo projeto é exatamente a Eq. 28:

```math
W^k_{KN}=\sum_{n\in k}|F_{kn}^{\dagger}C'_{KN}|^2.
```

`unfold_weights` percorre as colunas `F_kn` pertencentes ao mesmo ponto `k`, calcula os módulos quadrados e soma sobre a degenerescência. `unfold_supercell` reúne a construção dos operadores, a diagonalização conjunta, o agrupamento por `k` e essa projeção.

Para uma supercélula perfeita, os pesos são zero ou um. Quando o Hamiltoniano quebra a simetria de translação da referência, os pesos se tornam fracionários, como discutido após a Eq. 11. O exemplo de grafeno testa os dois regimes:

- supercélula 2×2 perfeita: recupera as bandas primitivas com peso unitário;
- defeito onsite: mantém a mesma topologia de orbitais, mas produz pesos fracionários e caráter espectral espalhado.

### 7. Regras de soma e função espectral

O código verifica a regra de completude

```math
\sum_k W^k_{KN}=1
```

para cada estado `KN` com `check_sum_rule_over_k`. Essa é uma das regras mencionadas após a Eq. 45.

As Eqs. 12 e 52 definem a função espectral:

```math
A(k,E)=\sum_{KN}W^k_{KN}\,\delta(E-E_{KN}).
```

`spectral_function` substitui a delta por uma Lorentziana normalizada de meia largura `broadening`:

```math
\delta(E-E_N)\longrightarrow
\frac{1}{\pi}\frac{\eta}{(E-E_N)^2+\eta^2}.
```

## Papel do HDF5 e do conversor CP2K

O artigo deriva o método para uma representação LCAO e apresenta uma implementação no FHI-aims. O formato HDF5 e o conversor CP2K são decisões de arquitetura deste projeto, não partes do artigo.

O núcleo matemático necessita somente de:

1. rede e posições da supercélula;
2. posições de referência para construir `T_PC`;
3. número e ordenação dos orbitais por átomo;
4. matrizes reais `H(R)` e `S(R)`;
5. relações de translação `R`.

O HDF5 canônico armazena exatamente esses objetos. O conversor em `src/converters/cp2k.jl` lê `KS_CSR_WRITE` e `S_CSR_WRITE`, associa cada arquivo à sua imagem periódica e produz um `RealSpaceModel`. Depois da conversão, nenhuma função de bandas ou unfolding conhece o formato CP2K.

Para uma estrutura relaxada, o conversor deve receber `positions` com a geometria física e `reference_positions` com os sítios ideais que definem a simetria a ser recuperada. O valor padrão `reference_positions=positions` é adequado somente quando os próprios sítios já formam o mapeamento periódico desejado.

Essa separação permite portar outros códigos: cada novo adaptador precisa apenas produzir o mesmo modelo HDF5, preservando a ordem dos AOs e as convenções de fase descritas acima.

## Correspondência rápida entre artigo e código

| Conceito | Equações | Implementação |
|---|---:|---|
| Relação PC–SC e folding de k | 1, 5–8 | `pc_lattice \ sc_lattice`; transformação de `k` no exemplo |
| Problema LCAO generalizado | 18–20 | `fourier_matrix`, `solve_bands` |
| Translação de AOs | 15, 23 | `translation_operator` |
| Estrutura fase-permutação | 29–32, 41 | `match_atom`, `translation_operator` |
| Transformação de Löwdin | 24–27 | `hermitian_sqrt`, `lowdin_transform` |
| Autovetores do projetor | 33–38, 46–48 | `joint_eigen` como alternativa numérica equivalente |
| Peso de unfolding | 10, 22, 28 | `unfold_weights`, `unfold_supercell` |
| Pontos k associados a K | 5–8 | `kfrac_from_lambdas`, `group_by_kfrac` |
| Regra de soma | após 45 | `check_sum_rule_over_k` |
| Função espectral | 12, 52 | `spectral_function` |

## Hipóteses e limitações atuais

### Implementado

- base LCAO real e não ortogonal;
- uma ou mais matrizes de Hamiltoniano para canais de spin colinear;
- supercélulas com transformação inteira, inclusive matrizes `M` não diagonais;
- deslocamentos atômicos quando `reference_positions` preserva o mapeamento ideal;
- defeitos onsite e substituições que preservem o número e a ordenação dos AOs;
- overlap completo e solução do problema generalizado;
- pesos e função espectral com alargamento Lorentziano.

### Ainda não implementado

- orbitais placeholder das Eqs. 43–45 para vacâncias, intersticiais ou espécies com números diferentes de AOs;
- matrizes reais-espaciais complexas, necessárias para alguns casos com acoplamento spin-órbita ou magnetismo não colinear;
- segunda regra de soma sobre todos os estados `N` como verificação pública separada;
- paralelização sobre pontos `K`, discutida na Sec. II D do artigo;
- médias de ensemble em temperatura finita da Eq. 53.

No estado atual, `translation_operator` interrompe explicitamente a execução quando o mapeamento requer números diferentes de orbitais. Isso evita produzir um resultado formalmente incorreto enquanto a extensão por placeholders não estiver disponível.

## Validação implementada

`test/runtests.jl` cobre:

1. ida e volta do modelo pelo HDF5;
2. conversão de arquivos CSR no padrão produzido pelo CP2K;
3. cone de Dirac do grafeno primitivo;
4. normalização `C†SC=I`;
5. regra de soma dos pesos;
6. recuperação das bandas primitivas a partir da supercélula perfeita 2×2.

O gráfico `examples/graphene/output/graphene_unfolding.png` demonstra visualmente a interpretação da Eq. 28: a linha da célula primitiva coincide com os estados de maior peso, enquanto a perturbação onsite transfere parte do caráter espectral para bandas antes escuras.
