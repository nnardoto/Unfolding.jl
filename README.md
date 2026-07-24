# Unfolding.jl

Unfolding de bandas em Julia para bases atômicas localizadas e não
ortogonais. A biblioteca lê `H(R)` e `S(R)` de um formato HDF5 independente
do código eletrônico; atualmente há um conversor pronto para as matrizes em
espaço real do CP2K.

Referência: J. Quan, N. Rybin, M. Scheffler e C. Carbogno,
*Phys. Rev. B* **113**, 085112 (2026),
[DOI 10.1103/7xym-7388](https://doi.org/10.1103/7xym-7388).

## Instalação e teste

No diretório da biblioteca:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

Em um script Julia, carregue o pacote com:

```julia
using Unfolding
```

## How-to: CP2K → HDF5 → unfolding

O uso recomendado precisa de apenas dois nomes de entrada:

1. o output textual padrão do CP2K, por exemplo `run.out`;
2. o prefixo comum dos arquivos CSR, por exemplo `outputs/Franqueite`.

Não é necessário fornecer cube, MOLog, rede, posições, espécies ou número de
orbitais manualmente.

### 1. Configure a saída do CP2K

Use `PRINT_LEVEL MEDIUM`, para que o `run.out` contenha célula, coordenadas e
informações das bases:

```text
&GLOBAL
  PROJECT Franqueite
  PRINT_LEVEL MEDIUM
&END GLOBAL
```

No `&DFT / &PRINT`, escreva Hamiltoniano e overlap em espaço real com o mesmo
`FILENAME`:

```text
&PRINT
  &KS_CSR_WRITE ON
    FILENAME ./outputs/Franqueite
    REAL_SPACE T
    BINARY F
    UPPER_TRIANGULAR F
  &END KS_CSR_WRITE

  &S_CSR_WRITE ON
    FILENAME ./outputs/Franqueite
    REAL_SPACE T
    BINARY F
    UPPER_TRIANGULAR F
  &END S_CSR_WRITE
&END PRINT
```

Crie o diretório antes de executar o CP2K:

```sh
mkdir -p outputs
cp2k.psmp -i run.inp -o run.out
```

Ao final devem existir arquivos como:

```text
run.out
outputs/Franqueite-KS_SPIN_1_R_1-1_0.csr
outputs/Franqueite-S_SPIN_1_R_1-1_0.csr
...
```

Se precisar manter `PRINT_LEVEL LOW`, habilite explicitamente `CELL`,
`ATOMIC_COORDINATES` e `KINDS` em `&SUBSYS / &PRINT`; o exemplo completo está
em [`docs/getting-started.md`](docs/getting-started.md).

### 2. Empacote o cálculo uma única vez

```julia
using Unfolding

convert_cp2k_to_hdf5(
    "outputs/Franqueite.h5", # destino canônico
    "run.out",               # output padrão do CP2K
    "outputs/Franqueite",    # prefixo comum de H(R) e S(R)
)
```

O conversor encontra todas as imagens periódicas do prefixo, associa cada
`R_n` à translação impressa no output e grava `H(R)`, `S(R)`, célula, átomos
e bases em `Franqueite.h5`. Depois dessa etapa, os arquivos nativos do CP2K
não participam mais do unfolding.

### 3. Informe a matriz de transformação e faça o unfolding

A convenção é

```text
A_sc = A_pc * M
```

onde as colunas de `A_sc` e `A_pc` são os vetores da supercélula e da célula
primitiva. `M` deve ser uma matriz inteira. Exemplo completo:

```julia
using Unfolding

sc = read_model("outputs/Franqueite.h5")

M = [
    1  1  0
    1 -1  0
    0  0  1
]

path = [
    [0.5, 0.0, 0.0], # X
    [0.0, 0.0, 0.0], # Γ
    [0.5, 0.5, 0.0], # S
    [0.0, 0.5, 0.0], # Y
    [0.0, 0.0, 0.0], # Γ
]

result = unfold_bandstructure(
    M,
    sc,
    path,
    40; # pontos por segmento
    tick_labels=["X", "Γ", "S", "Y", "Γ"],
    progress=true,
)

write_unfolded_hdf5("outputs/Franqueite-unfolded.h5", result)
```

### Paralelização do caminho k

`solve_bands` distribui automaticamente os pontos k entre as threads
disponíveis. Por padrão, `unfold_bandstructure` faz o mesmo. Para usar todos
os núcleos, inicie Julia com múltiplas threads e mantenha o BLAS em uma
thread, evitando duas camadas de paralelismo competindo pelos mesmos núcleos:

```sh
OPENBLAS_NUM_THREADS=1 julia --threads=auto --project=. seu_script.jl
```

A ordem dos pontos no resultado é a mesma do caminho de entrada. Para forçar
o comportamento serial, use `parallel=false` em qualquer uma das duas
funções. `Threads.nthreads()` mostra quantas threads Julia estão disponíveis.
Com `progress=true`, a execução mostra separadamente o avanço da solução das
bandas e do unfolding, incluindo percentual e número de pontos concluídos:

```text
Bandas    [████████████████████████] 100% (157/157)
Unfolding [████████████████████████] 100% (157/157)
```

Para estruturas maiores, a projeção de unfolding pode escalar melhor em
processos Julia independentes:

```julia
result = unfold_bandstructure(
    M,
    sc,
    path,
    40;
    parallel=true,             # bandas entre threads
    unfold_processes=4,        # unfolding em quatro processos
    progress=true,
)
```

`unfold_processes=0` mantém o modo por threads. Um valor positivo reutiliza
workers já presentes e cria os que faltarem com uma thread e BLAS serial.
Os workers criados pela chamada são encerrados ao final. Em uma sequência de
cálculos, `keep_processes=true` evita pagar novamente sua inicialização.

`unfold_batches_per_process=4` é o padrão. Aumente-o se os pontos tiverem
custos muito diferentes; diminua-o para reduzir comunicação. Os dados de
cada ponto são enviados a somente um processo, em blocos.

Processos têm custo de inicialização e serialização. Para modelos pequenos,
threads normalmente continuam mais rápidas. Compare `unfold_processes=0`,
`2`, `4`, ... no caso real, mantendo o produto
`processos × threads do BLAS` dentro do número de núcleos físicos.

Também é possível iniciar workers antes do script:

```sh
OPENBLAS_NUM_THREADS=1 julia -p 4 --threads=auto --project=. seu_script.jl
```

Eles serão reutilizados quando `unfold_processes=4`.

O arquivo final contém pontos k, energias, pesos espectrais e rótulos. Seu
schema está documentado em
[`docs/unfolded-hdf5-schema.md`](docs/unfolded-hdf5-schema.md).

### Script executável do caso real

O exemplo Franckeite faz o empacotamento automaticamente se o HDF5 ainda não
existir; nas execuções seguintes, relê somente o HDF5:

```sh
julia --project=/caminho/para/Unfolding.jl unfold_franckeite.jl
```

Em resumo, os dados específicos do cálculo são:

```text
run.out + prefixo CSR + matriz inteira M + caminho de bandas
```

## Fluxo de dados

```text
CP2K ── conversão única ──> HDF5 canônico ──> unfolding ──> HDF5 de resultado
```

Todos os cálculos posteriores consomem `RealSpaceModel`, portanto um novo
conversor para outro código eletrônico precisa apenas produzir o mesmo
modelo canônico. O schema e a convenção de fases estão em
[`docs/hdf5-schema.md`](docs/hdf5-schema.md), e o mapeamento entre o artigo e
a implementação está em
[`docs/paper-implementation.md`](docs/paper-implementation.md).

Para estudar a teoria com mais contexto — incluindo o gauge periódico do
CP2K, a origem dos pesos espúrios, os testes de validação e a construção do
mapa de calor — veja o capítulo
[`docs/cp2k-unfolding-study.md`](docs/cp2k-unfolding-study.md).

## Exemplo de teste: grafeno

Para gerar os modelos pequenos de grafeno e exportar um unfolding:

```sh
julia --project=. examples/graphene/generate_hdf5.jl
julia --project=. examples/graphene/export_unfolding.jl
```

O guia detalhado de uso em estruturas próprias está em
[`docs/getting-started.md`](docs/getting-started.md).
