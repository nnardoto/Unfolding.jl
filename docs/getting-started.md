# Usando o Unfolding.jl em um projeto real

Este guia mostra o caminho de ponta a ponta para desdobrar bandas de uma
supercélula real (defeito, dopante, relaxação) sobre a célula primitiva,
usando só a API de alto nível. Para o mapeamento equação-por-equação com o
artigo de referência, veja [`paper-implementation.md`](paper-implementation.md);
para o formato dos arquivos, veja [`hdf5-schema.md`](hdf5-schema.md) (modelo
de entrada) e [`unfolded-hdf5-schema.md`](unfolded-hdf5-schema.md) (resultado
exportado). O exemplo executável completo está em `examples/graphene/`.

## 1. Rodar o(s) cálculo(s) de estrutura eletrônica

Você precisa de dois cálculos reais-espaço: um da célula primitiva (ou de
referência) e um da supercélula que quer desdobrar (com o defeito, dopante,
ou relaxação de interesse). Hoje o único conversor pronto é para o CP2K,
pedindo `KS_CSR_WRITE` e `S_CSR_WRITE` com `REAL_SPACE T`, `BINARY F`,
`UPPER_TRIANGULAR F` (veja `examples/graphene/cp2k/README.md` para um
`.inp` de exemplo). O ponto que mais costuma dar problema aqui não é o
Unfolding.jl em si, e sim garantir que você sabe exatamente a ordem dos
átomos e o número de orbitais atômicos (AOs) por átomo que o código usou --
isso tem que bater com o que você passa ao conversor no próximo passo.

Para outro código de estrutura eletrônica, seria necessário escrever um
conversor equivalente ao de `src/converters/cp2k.jl`; a interface exigida
(rede, posições, números atômicos, `norb` por átomo, e as matrizes reais
`H(R)`/`S(R)` indexadas pelas mesmas translações `R`) está descrita em
`docs/hdf5-schema.md`.

## 2. Converter para o HDF5 canônico

```julia
using Unfolding

pc_model = convert_cp2k_to_hdf5("primitive.h5", "pc.out", "pc/ham", "pc/ovl";
    lattice=pc_lattice, positions=pc_positions,
    atomic_numbers=pc_atomic_numbers, norb=pc_norb)

sc_model = convert_cp2k_to_hdf5("supercell.h5", "sc.out", "sc/ham", "sc/ovl";
    lattice=sc_lattice, positions=sc_positions_relaxadas,
    reference_positions=sc_positions_ideais,   # topologia sem a perturbação
    atomic_numbers=sc_atomic_numbers, norb=sc_norb)
```

`reference_positions` só precisa ser diferente de `positions` quando a
supercélula tem uma geometria física diferente da topologia ideal (átomos
deslocados por relaxação). Para um defeito onsite ou substitucional puro,
sem deslocamento atômico, os dois campos podem ser iguais (o valor padrão).

Isso só precisa ser feito uma vez por estrutura; o restante do fluxo lê os
arquivos `.h5` gerados.

## 3. Desdobrar as bandas ao longo de um caminho

```julia
pc = read_model("primitive.h5")
sc = read_model("supercell.h5")

path = [[0.0, 0.0, 0.0], [2/3, 1/3, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]
labels = ["Γ", "K", "M", "Γ"]

result = unfold_bandstructure(pc.lattice, sc, path, 61; tick_labels=labels)
```

Note que aqui não precisamos ter calculado eletronicamente a célula
primitiva -- `unfold_bandstructure` só usa `pc.lattice`, os vetores de rede
(geometria pura). O único lugar onde o cálculo eletrônico da PC entra é no
passo 4 abaixo, e só se você quiser a banda de referência para comparação
visual.

Se você já conhece a matriz de transformação inteira `M` (`A_sc = A_pc*M`,
por exemplo `M = [2 0 0; 0 2 0; 0 0 1]` para uma supercélula 2×2×1) e não
quer manter os vetores de rede da PC à parte, existe uma variante que
recebe `M` diretamente:

```julia
result = unfold_bandstructure([2 0 0; 0 2 0; 0 0 1], sc, path, 61; tick_labels=labels)
```

Os dois métodos dão exatamente o mesmo resultado; `pc_lattice` é
recuperado internamente por `sc.lattice * inv(M)`. A distinção de tipo (`M`
precisa ser uma matriz de inteiros) evita confundir os dois -- e mesmo que
você troque um pelo outro por engano, a checagem interna de que a rede da
supercélula é um múltiplo inteiro da rede recebida barra o erro cedo, em
vez de desdobrar silenciosamente com a física errada.

`unfold_bandstructure` resolve as bandas da supercélula e as desdobra em
cada ponto do caminho, cuidando de todo o mapeamento geométrico entre PC e
SC (veja a docstring da função para os detalhes). Ela lança um erro cedo se
a rede da supercélula não for um múltiplo inteiro da rede primitiva -- o
erro mais comum nessa etapa costuma ser passar a rede errada ou trocar
`positions` por `reference_positions` sem querer.

Hoje isso cobre defeitos onsite, substituições e deslocamentos atômicos que
preservam o número de AOs por átomo. Vacâncias, interstícios ou espécies com
números diferentes de orbitais ainda não são suportados -- veja "Ainda não
implementado" em `paper-implementation.md`.

## 4. Exportar para um arquivo binário

```julia
pcbands = solve_bands(pc, eachcol(result.kpoints_frac))
reference_energies = reduce(hcat, pcbands.energies)

write_unfolded_hdf5("unfolded.h5", result; reference_energies=reference_energies)
```

`reference_energies` é opcional -- inclua se quiser sobrepor a banda exata
da célula primitiva no mesmo arquivo, para comparação visual. O arquivo
resultante é HDF5 puro (não depende de nenhuma biblioteca de plotagem) e
está documentado em `docs/unfolded-hdf5-schema.md`.

## 5. Plotar

O núcleo do pacote não tem opinião sobre como você plota. Dois exemplos
prontos, ambos lendo o mesmo `unfolded.h5`:

```sh
julia --project=examples/graphene/plotting examples/graphene/plotting/plot_unfolding.jl   # Plots.jl
python examples/graphene/plotting/plot_matplotlib.py                                       # Matplotlib
```

Adapte o caminho do arquivo lido nesses scripts para o seu `unfolded.h5`, ou
copie a lógica de leitura (é só HDF5, ver `docs/unfolded-hdf5-schema.md`)
para gnuplot ou qualquer outra ferramenta.

## Resumo

```julia
using Unfolding

pc = read_model("primitive.h5")
sc = read_model("supercell.h5")

result = unfold_bandstructure(pc.lattice, sc, path, n_per_segment; tick_labels=labels)
pcbands = solve_bands(pc, eachcol(result.kpoints_frac))
write_unfolded_hdf5("unfolded.h5", result; reference_energies=reduce(hcat, pcbands.energies))
```

Cinco linhas depois de já ter os dois `RealSpaceModel` -- o resto (geometria
dos operadores de translação, diagonalização conjunta, projeção de Löwdin)
fica encapsulado em `unfold_bandstructure`. As funções de mais baixo nível
(`AtomBasis`, `translation_operators`, `joint_eigen`, `unfold_supercell`,
...) continuam exportadas e disponíveis para quem precisar de um ponto k
avulso, uma malha em vez de um caminho, ou um projetor customizado.
