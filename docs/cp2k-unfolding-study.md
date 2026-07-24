# Unfolding de bandas em bases LCAO: teoria, gauge do CP2K e validação

## Visão geral

Este capítulo reconstrói, passo a passo, o raciocínio usado pelo
`Unfolding.jl` para desdobrar bandas obtidas com o CP2K. A intenção não é
apenas ensinar a executar o programa, mas fornecer uma base para estudar,
auditar e estender a implementação.

Ao final, o leitor deverá ser capaz de:

1. distinguir um ponto `k` da célula primitiva de seu ponto dobrado `K` na
   zona de Brillouin da supercélula;
2. compreender por que o overlap `S(K)` é indispensável em uma base de
   orbitais atômicos não ortogonais;
3. interpretar o peso de unfolding como o valor esperado de um projetor;
4. explicar por que posições periodicamente equivalentes não podem ser
   trocadas sem ajustar as fases e os índices de imagem das matrizes;
5. reconhecer o erro de gauge que produzia peso espectral fora das bandas
   primitivas;
6. reproduzir o teste de grafeno e avaliar quantitativamente seu resultado;
7. construir um mapa de calor espectral sem confundir suavização visual com
   aumento real da amostragem em `k`.

O formalismo segue J. Quan, N. Rybin, M. Scheffler e C. Carbogno,
“Efficient band structure unfolding with atom-centered orbitals: General
theory and application”, *Physical Review B* **113**, 085112 (2026),
[DOI 10.1103/7xym-7388](https://doi.org/10.1103/7xym-7388).

Para uma correspondência curta entre as equações do artigo e as funções da
biblioteca, consulte [`paper-implementation.md`](paper-implementation.md).

---

## 1. Por que bandas de uma supercélula precisam ser desdobradas?

Uma supercélula contém várias réplicas de uma célula primitiva. Como sua rede
direta é maior, sua zona de Brillouin é menor. Estados que na célula primitiva
aparecem em diferentes pontos `k` são levados ao mesmo ponto `K` da
supercélula. Esse processo é chamado de *folding*.

Se a supercélula contiver `m` células primitivas, cada banda primitiva gera,
em termos gerais, `m` ramos dobrados. Um gráfico direto das bandas da
supercélula fica, portanto, muito mais congestionado que a dispersão
primitiva.

O unfolding responde à pergunta inversa:

> Quanto de um estado da supercélula em `K` possui o caráter translacional de
> um determinado ponto primitivo `k`?

A resposta é um peso espectral

```math
0 \leq W^k_{KN} \leq 1,
```

onde `N` identifica uma banda da supercélula. Em uma supercélula perfeitamente
periódica, os pesos ideais são zero ou um. Defeitos, desordem e relaxações
misturam caracteres de diferentes pontos `k`, produzindo pesos fracionários.

O unfolding não cria novos autovalores. Ele reorganiza os autovalores da
supercélula segundo seu caráter de translação primitiva.

---

## 2. Notação e convenções

As matrizes de rede têm os vetores de rede nas colunas:

```math
A_{\mathrm{pc}}=[a_1\ a_2\ a_3], \qquad
A_{\mathrm{sc}}=[A_1\ A_2\ A_3].
```

| Símbolo | Significado |
|---|---|
| `pc` | célula primitiva ou célula de referência |
| `sc` | supercélula |
| `M` | matriz inteira que relaciona as duas redes |
| `m=|det(M)|` | número de células primitivas na supercélula |
| `k` | ponto na base recíproca da célula primitiva |
| `K` | ponto na base recíproca da supercélula |
| `R` | translação inteira da supercélula no espaço direto |
| `H(R)` | Hamiltoniano entre a célula central e a imagem `R` |
| `S(R)` | overlap entre a célula central e a imagem `R` |
| `C_{KN}` | coeficientes LCAO do estado `N` em `K` |
| `W^k_{KN}` | peso do estado `KN` no subespaço primitivo `k` |

A relação entre as redes é

```math
A_{\mathrm{sc}}=A_{\mathrm{pc}}M.
```

Como o código representa coordenadas fracionárias por vetores-coluna, o
folding recíproco é escrito como

```math
K_{\mathrm{frac}}=M^{\mathsf T}k_{\mathrm{frac}}\pmod 1.
```

O determinante de `M` tem duas interpretações úteis:

- no espaço real, fornece a multiplicidade da supercélula;
- no espaço recíproco, fornece o número de pontos primitivos `k` que dobram
  sobre um mesmo `K`.

### Exemplo: grafeno 2×2

Para uma supercélula 2×2 no plano:

```math
M=
\begin{pmatrix}
2&0&0\\
0&2&0\\
0&0&1
\end{pmatrix},
\qquad |\det M|=4.
```

Quatro pontos da zona primitiva são associados a cada ponto `K` da
supercélula.

---

## 3. Hamiltoniano em espaço real e transformação de Fourier

O CP2K fornece matrizes LCAO em espaço real, indexadas por imagens periódicas.
O `Unfolding.jl` adota

```math
H(k)=\sum_R e^{+i2\pi k\cdot R}H(R),
```

```math
S(k)=\sum_R e^{+i2\pi k\cdot R}S(R).
```

As coordenadas `k` e `R` são fracionárias e inteiras, respectivamente, de
modo que o produto `k·R` é adimensional.

A consistência hermitiana exige

```math
H(-R)=H(R)^\dagger,\qquad
S(-R)=S(R)^\dagger.
```

Essas relações são importantes por dois motivos:

1. garantem que `H(k)` e `S(k)` sejam hermitianos;
2. detectam arquivos incompletos, imagens mal rotuladas ou convenções
   incompatíveis antes do unfolding.

No código, essa etapa corresponde a:

- `fourier_matrix`;
- `hamiltonian_at_k`;
- `overlap_at_k`;
- `solve_bands`.

O formato HDF5 canônico e sua convenção de Fourier estão descritos em
[`hdf5-schema.md`](hdf5-schema.md).

---

## 4. O problema generalizado em uma base não ortogonal

Orbitais atômicos localizados não formam, em geral, uma base ortogonal. As
bandas são obtidas do problema generalizado

```math
H_K C_{KN}=E_{KN}S_KC_{KN}.
```

Os autovetores são normalizados na métrica do overlap:

```math
C_{KM}^{\dagger}S_KC_{KN}=\delta_{MN}.
```

Usar diretamente `C†C=I` seria incorreto. A transformação simétrica de Löwdin
leva os coeficientes para uma representação ortonormal:

```math
C'_{KN}=S_K^{1/2}C_{KN}.
```

Então

```math
{C'}^\dagger C'=I.
```

Na implementação:

- `hermitian_sqrt` constrói `S_K^{1/2}`;
- `lowdin_transform` calcula `C'=S_K^{1/2}C`;
- a positividade de `S_K` e a normalização `C†S_KC≈I` funcionam como
  diagnósticos numéricos.

Essa etapa permite calcular o projetor no espaço ortonormal sem perder a
informação física contida no overlap.

---

## 5. Translações primitivas e o projetor de unfolding

Considere os operadores que traduzem os orbitais por um vetor primitivo
`-a_i`. Dentro de uma supercélula, essa translação funciona como uma
permutação dos orbitais. Quando um orbital atravessa a fronteira periódica,
a permutação recebe a fase de Bloch apropriada.

Para uma translação que envolve uma imagem inteira `p` da supercélula, a fase
usada é

```math
e^{-i2\pi K_{\mathrm{frac}}\cdot p}.
```

Os três operadores de translação:

- são unitários;
- são normais;
- comutam quando a topologia de referência é consistente.

Logo, possuem uma base ortonormal comum `F`. Seus autovalores têm a forma

```math
\lambda_i=e^{i2\pi k_i}.
```

As fases dos autovalores identificam o ponto primitivo:

```math
k_i=\frac{\arg(\lambda_i)}{2\pi}\pmod 1.
```

As colunas de `F` que compartilham o mesmo `k` formam o subespaço usado no
projetor `P_k`. Depois da transformação de Löwdin, o peso é

```math
W^k_{KN}
=
\langle\Psi_{KN}|P_k|\Psi_{KN}\rangle
=
\sum_{n\in k}
\left|F_{kn}^{\dagger}C'_{KN}\right|^2.
```

Na biblioteca, a sequência é:

```text
translation_operators
        ↓
joint_eigen
        ↓
kfrac_from_lambdas + group_by_kfrac
        ↓
lowdin_transform
        ↓
unfold_weights
```

`unfold_supercell` encapsula essa sequência para um ponto;
`unfold_bandstructure` a aplica a todo o caminho.

---

## 6. Posições físicas e posições de referência

O modelo guarda dois conjuntos de coordenadas:

- `positions`: posições físicas usadas no cálculo eletrônico;
- `reference_positions`: topologia ideal usada para construir as translações
  primitivas e o projetor.

Essa separação é essencial quando os átomos relaxam. O Hamiltoniano deve
descrever a estrutura física, mas o projetor deve representar a simetria que
se deseja recuperar.

Para uma supercélula perfeita:

```text
reference_positions = positions
```

Para uma estrutura relaxada:

```text
positions           = geometria relaxada
reference_positions = sítios ideais correspondentes
```

O número, a ordem e a quantidade de orbitais de cada átomo ainda devem
coincidir entre a topologia física e a de referência.

---

## 7. Gauge periódico: posições equivalentes, matrizes diferentes

### 7.1 A sutileza

Duas posições cartesianas

```math
r \quad\text{e}\quad r+A n,
```

com `n` inteiro, representam o mesmo ponto físico sob condições periódicas.
Entretanto, elas não são intercambiáveis dentro de uma representação
`H(R)` sem também transformar os índices de imagem ou as fases.

Ao mover um orbital da célula de referência para uma imagem vizinha, um
elemento antes associado a `R` passa a ser associado a outro vetor `R'`.
No espaço recíproco, essa mudança aparece como uma fase dependente de `k`.

Portanto:

> A invariância física sob translações de rede não significa invariância de
> cada objeto intermediário sob uma mudança isolada de representante.

O Hamiltoniano, o overlap, as posições usadas pelo projetor e a convenção de
Fourier precisam pertencer ao mesmo gauge periódico.

### 7.2 O gauge usado pelo CP2K

Ao construir a lista de vizinhos que origina as matrizes CSR em espaço real,
o CP2K reduz posições periodicamente por uma operação equivalente a centralizar
as coordenadas fracionárias em torno da origem.

Se

```math
s=A^{-1}r
```

é a posição fracionária física, define-se um vetor de imagem

```math
n=\operatorname{round}(s)
```

nas direções periódicas ativas. O representante centralizado é

```math
s_{\mathrm{c}}=s-n,
\qquad
r_{\mathrm{c}}=r-An.
```

Os rótulos `R` dos arquivos CSR referem-se a esse representante centralizado.
Já as coordenadas impressas no output ou em um cube podem estar em
`[0,1)`, ou em outra imagem equivalente.

### 7.3 Como surgiu o erro

Antes da correção, o conversor podia combinar:

1. `H(R)` e `S(R)` rotulados no gauge centralizado do CP2K;
2. `reference_positions` no representante impresso, sem a translação de
   imagem correspondente.

As matrizes eletrônicas continuavam individualmente válidas. O projetor
também parecia geometricamente plausível. Contudo, as fases de fronteira
usadas pelo projetor não correspondiam às fases implícitas nos CSR.

Essa incompatibilidade selecionava parcialmente o subespaço `k` errado. No
gráfico, apareciam pesos em torno de `0,6` longe das bandas independentes da
célula primitiva.

O aspecto traiçoeiro é que uma regra de soma global pode continuar quase
exata: a completude diz que o peso total foi distribuído entre subespaços,
mas não prova que cada subespaço recebeu o rótulo `k` correto.

### 7.4 Correção implementada

O conversor agora calcula os deslocamentos de imagem a partir das posições
**físicas** e aplica os mesmos deslocamentos às posições **de referência**:

```math
s_{\mathrm{phys}}=A^{-1}r_{\mathrm{phys}},
```

```math
n=\operatorname{round}(s_{\mathrm{phys}}),
```

```math
r_{\mathrm{ref}}^{\mathrm{CSR}}
=
r_{\mathrm{ref}}-An.
```

O detalhe “a partir das posições físicas” é importante. Em uma estrutura
relaxada, é a posição física do AO que determina qual imagem o CP2K usou na
lista de vizinhos. Aplicar esse mesmo deslocamento à posição ideal preserva
simultaneamente:

- o gauge eletrônico do CP2K;
- a topologia ideal do projetor.

A implementação está em `_cp2k_reference_in_csr_gauge`, chamada por
`read_cp2k_csr`.

### 7.5 Direções periódicas ativas

Uma direção só é centralizada se aparecer em alguma translação `R` dos CSR.
Isso evita alterar arbitrariamente uma direção não periódica, como o eixo
perpendicular de uma monocamada com vácuo.

No código, a direção `i` é ativa quando:

```julia
any(!iszero, R[i, :])
```

### 7.6 Pontos na fronteira de meia célula

Valores exatamente iguais a `±0,5` são ambíguos: os dois representantes de
fronteira são periodicamente equivalentes. Além disso, componentes de célula
impressos com precisão finita podem transformar `0,5` em `0,500003`.

O conversor estabiliza valores próximos de múltiplos de meia unidade antes
de arredondar. A tolerância padrão é `1e-5` em coordenadas fracionárias. Isso
impede que ruído de impressão altere desnecessariamente a imagem escolhida.

Essa tolerância não corrige uma geometria inconsistente; ela apenas trata a
representação numérica de uma fronteira exata.

---

## 8. Por que um peso alto fora da banda primitiva é suspeito?

Para uma supercélula perfeita, a banda primitiva independente fornece um
teste de referência particularmente forte. Se uma banda da supercélula tem
peso significativo no ponto `k`, sua energia deve coincidir, dentro da
tolerância numérica, com alguma banda da célula primitiva nesse mesmo `k`.

Um peso alto longe de todas as bandas primitivas pode indicar:

- gauge periódico inconsistente;
- sinal incorreto na transformada de Fourier;
- transformação `k↔K` incorreta;
- ordenação de átomos ou AOs incompatível;
- matriz `M` errada;
- comparação em caminhos `k` diferentes;
- alinhamento de energia incorreto;
- arquivos CSR incompletos ou pertencentes a cálculos diferentes.

Em uma estrutura com defeito, algum espalhamento espectral é físico. Em uma
supercélula pristina, ele deve ser praticamente nulo. Por isso o grafeno
perfeito é um bom teste de regressão.

---

## 9. Hierarquia de testes e diagnósticos

Nenhum teste isolado é suficiente. Uma validação robusta deve avançar do
objeto mais básico ao resultado físico.

### 9.1 Integridade dos dados reais

Verifique:

```math
H(-R)=H(R)^\dagger,
\qquad
S(-R)=S(R)^\dagger.
```

Confira ainda:

- mesmo conjunto de imagens para `H` e `S`;
- dimensão matricial igual ao número total de AOs;
- SCF convergido;
- ausência de abort posterior à última convergência;
- mesma ordem atômica e de base em todos os arquivos.

### 9.2 Problema generalizado

Em cada ponto:

```math
C^\dagger S C\approx I.
```

O overlap deve ser positivo definido no subespaço resolvido.

### 9.3 Operadores de translação

Verifique:

```math
T_i^\dagger T_i\approx I,
\qquad
[T_i,T_j]\approx0.
```

Os resíduos dos autovetores conjuntos também devem ser pequenos:

```math
\|T_iF_n-\lambda_{i,n}F_n\|\ll1.
```

### 9.4 Regra de soma sobre todos os pontos dobrados

Para cada estado da supercélula:

```math
\sum_k W^k_{KN}=1.
```

Essa regra testa a completude dos projetores, mas não detecta necessariamente
uma troca sistemática dos rótulos `k`.

### 9.5 Peso total no `k` selecionado

No teste pristino, o script acompanha um único ramo desdobrado em cada ponto
do caminho. O peso total esperado é

```math
\sum_N W^k_{KN}
=
\frac{N_{\mathrm{AO,sc}}}{|\det M|}.
```

Para o grafeno de depuração:

```text
N_AO,sc = 32
|det M| = 4
peso total esperado = 8
```

### 9.6 Comparação com bandas primitivas

Depois de alinhar as referências de energia, para cada banda primitiva
`ε_b(k)` mede-se a menor separação até as bandas da supercélula:

```math
\Delta_b(k)
=
\min_N |E_N(K)-\varepsilon_b(k)|.
```

No cálculo usado durante a correção, a discrepância máxima ficou em
aproximadamente `8,3 meV`.

### 9.7 Vazamento espectral

Escolha uma tolerância energética `δ`, por exemplo `20 meV`. Para cada ponto,
some o peso de bandas que estejam além de `δ` de **todas** as bandas
primitivas:

```math
L(k)
=
\frac{1}{W_{\mathrm{esperado}}}
\sum_{\substack{N\\
\min_b|E_N-\varepsilon_b|>\delta}}
W^k_{KN}.
```

O máximo ao longo do caminho é um teste sensível de inconsistência de fase.
Com 94 pontos no caminho do exemplo, foi observado cerca de `0,30%`. Esse
valor pequeno é compatível com diferenças numéricas entre os dois cálculos
CP2K, mas é suficientemente mensurável para acompanhar regressões.

---

## 10. Exemplo leve e reproduzível: grafeno 2×2

O diretório `examples/graphene/cp2k/` contém:

| Arquivo | Função |
|---|---|
| `graphene_debug_primitive.inp` | cálculo CP2K da célula primitiva |
| `graphene_debug_2x2.inp` | cálculo CP2K da supercélula pristina |
| `debug_unfold.jl` | conversão, unfolding e diagnósticos |
| `plot_debug.py` | mapa de calor e bandas primitivas sobrepostas |

Os inputs usam a base mínima `SZV-MOLOPT-SR-GTH`. A malha da supercélula é
3×3×1 e a malha primitiva equivalente é 6×6×1.

### 10.1 Executar o CP2K

A partir da raiz do repositório:

```sh
mkdir -p examples/graphene/cp2k/run_debug
cd examples/graphene/cp2k/run_debug

OMP_NUM_THREADS=1 cp2k.psmp \
  -i ../graphene_debug_2x2.inp \
  -o graphene_debug.out

OMP_NUM_THREADS=1 cp2k.psmp \
  -i ../graphene_debug_primitive.inp \
  -o graphene_debug_pc.out
```

### 10.2 Converter e desdobrar

De volta à raiz:

```sh
OPENBLAS_NUM_THREADS=1 julia --threads=auto --project=. \
  examples/graphene/cp2k/debug_unfold.jl
```

O padrão é 32 amostras por segmento. Como os vértices compartilhados não são
duplicados, o caminho Γ–K–M–Γ contém:

```math
3\times32-2=94
```

pontos calculados.

Outro valor pode ser informado como segundo argumento:

```sh
julia --threads=auto --project=. \
  examples/graphene/cp2k/debug_unfold.jl \
  examples/graphene/cp2k/run_debug 48
```

Nesse caso, o caminho terá `3×48−2=142` pontos.

### 10.3 Paralelismo

Os pontos do caminho são independentes. `solve_bands` e
`unfold_bandstructure` os distribuem entre as threads Julia, preservando a
ordem no resultado.

Uma configuração adequada em uma máquina com vários núcleos é:

```sh
OPENBLAS_NUM_THREADS=1 julia --threads=auto --project=. ...
```

Isso usa paralelismo externo sobre pontos `k` e evita que cada ponto crie,
simultaneamente, outra equipe grande de threads no BLAS.

Com `progress=true`, linhas persistentes mostram separadamente:

- solução das bandas;
- cálculo dos pesos de unfolding.

Para estruturas grandes, bandas e projeção podem ser transferidas juntas
para processos Julia independentes:

```sh
OPENBLAS_NUM_THREADS=1 julia --threads=auto --project=. \
  examples/graphene/cp2k/debug_unfold.jl \
  examples/graphene/cp2k/run_debug 32 4
```

O terceiro argumento solicita quatro processos para bandas e unfolding. Na
API, a mesma configuração é:

```julia
result = unfold_bandstructure(
    M, model, path, 32;
    unfold_processes=4,
    unfold_batches_per_process=16,
    progress=true,
)
```

Em notebooks, o ciclo de vida pode ser controlado inteiramente por células:

```julia
using Unfolding
using LinearAlgebra

# Célula de configuração
BLAS.set_num_threads(1)
start_unfold_workers(12)
unfold_worker_status()

# Célula de cálculo
result = unfold_bandstructure(
    M, model, path, 32;
    unfold_processes=12,
    unfold_batches_per_process=16,
    progress=true,
)

# Célula de encerramento, quando desejado
stop_unfold_workers()
```

`start_unfold_workers` reutiliza processos existentes e cria apenas os que
faltam. `stop_unfold_workers` encerra somente os processos criados e
registrados pelo pacote. Assim, workers pertencentes a outro cálculo do
notebook não são removidos acidentalmente.

Processos podem ser acrescentados durante a sessão. Threads Julia, por outro
lado, são uma propriedade do kernel e não podem ser aumentadas depois de sua
inicialização. Um kernel com uma thread ainda pode usar o modo multiprocesso
para paralelizar tanto bandas quanto unfolding.

No modo multiprocesso, cada worker resolve as bandas de um ponto e executa
seu unfolding imediatamente. Overlap e coeficientes densos não retornam ao
processo principal; somente energias e pesos são reunidos no final. O modelo
esparso fica em cache uma vez por worker durante a chamada.

Workers que já existem são reutilizados. Se faltarem, a função cria workers
locais com uma thread e BLAS serial. Os workers temporários são removidos ao
final, a menos que `keep_processes=true`.

O número ideal depende do tamanho da base, da quantidade de imagens e do
número de pontos. Processos acrescentam custos de inicialização e
serialização; por isso, o pequeno exemplo de grafeno pode ficar mais lento.
Para um cálculo real:

1. aqueça uma execução curta para compilar as funções;
2. compare `unfold_processes=0`, `2`, `4`, ...;
3. mantenha uma thread BLAS por processo;
4. use `keep_processes=true` ao repetir vários caminhos;
5. use `unfold_batches_per_process=16` ou `32` para manter mais blocos na
   fila e reduzir a cauda com workers ociosos;
6. acompanhe também a memória, pois cada worker mantém as matrizes densas do
   ponto que está processando.

### 10.4 Arquivos produzidos

O script grava:

```text
graphene_debug.h5
graphene_debug_pc.h5
graphene_debug_unfolded.h5
graphene_debug_unfolded.csv
graphene_debug_reference.csv
```

O HDF5 de resultado inclui as bandas primitivas independentes em
`/reference/energies`.

### 10.5 Gerar o gráfico

```sh
python3 examples/graphene/cp2k/plot_debug.py
```

O resultado é `graphene_debug_unfolded.png`.

---

## 11. Do conjunto de deltas ao mapa de calor

A função espectral ideal é

```math
A(k,E)=\sum_N W_N(k)\,\delta(E-E_N(k)).
```

Para visualizar, o script substitui cada delta por uma gaussiana:

```math
\delta(E-E_N)
\longrightarrow
\exp\left[-\frac{(E-E_N)^2}{2\sigma^2}\right].
```

Assim,

```math
A_\sigma(k,E)
=
\sum_N W_N(k)
\exp\left[-\frac{(E-E_N(k))^2}{2\sigma^2}\right].
```

No exemplo:

```text
σ = 0,055 eV
grade de energia = 900 pontos entre −8 e 8 eV
```

Após normalização global, o mapa usa:

- paleta `magma`, perceptualmente ordenada;
- fundo escuro;
- `PowerNorm` com expoente `0,65`, que torna pesos menores visíveis;
- bandas primitivas independentes em ciano;
- linha branca tracejada na referência de energia;
- linhas verticais nos pontos de alta simetria.

### Interpolação visual não é amostragem física

O mapa interpola a intensidade entre colunas vizinhas para evitar uma imagem
pixelada. Essa interpolação:

- melhora apenas a renderização;
- não resolve novos autovalores;
- não acrescenta informação entre os pontos calculados.

Aumentar `n_per_segment` de 16 para 32, por outro lado, refaz realmente o
cálculo e elevou a amostragem de 46 para 94 pontos.

### Escolha do alargamento

Um `σ` muito pequeno produz bandas pontilhadas quando a malha em `k` é
esparsa. Um `σ` muito grande mistura ramos próximos e pode esconder gaps.
Uma estratégia prática é:

1. aumentar primeiro a amostragem em `k`;
2. escolher `σ` comparável à resolução energética desejada;
3. verificar que conclusões físicas não mudam sob uma redução moderada de
   `σ`;
4. manter as bandas primitivas sobrepostas durante a depuração.

---

## 12. Como interpretar o gráfico

Para o grafeno pristino:

- regiões claras do mapa devem acompanhar as linhas ciano;
- o cone de Dirac deve aparecer em `K`;
- bandas com peso quase zero devem permanecer escuras;
- os dois pontos Γ, no início e no fim, devem ser idênticos;
- não devem existir ilhas intensas longe de toda linha ciano.

Em um sistema com defeito:

- o peso pode se distribuir entre vários ramos;
- bandas pouco dispersivas podem indicar estados localizados;
- alargamento em `k` representa quebra de caráter translacional;
- comparação com a célula primitiva continua útil, mas coincidência perfeita
  deixa de ser esperada.

A intensidade do gráfico é normalizada para visualização. Ela não deve ser
interpretada como densidade absoluta de estados sem considerar a
normalização, o alargamento e a amostragem usados.

---

## 13. Roteiro de depuração para um material novo

Siga a ordem abaixo; ela tende a localizar o erro antes das etapas mais caras.

### Dados do CP2K

- [ ] O SCF convergiu?
- [ ] `KS_CSR_WRITE` e `S_CSR_WRITE` usam `REAL_SPACE T`?
- [ ] Ambos usam o mesmo prefixo e o mesmo cálculo?
- [ ] Todas as imagens `R` foram encontradas?
- [ ] A ordem e o número de AOs batem com a matriz?

### Geometria

- [ ] As redes usam vetores nas colunas?
- [ ] `M=A_pc\A_sc` é inteira dentro da tolerância?
- [ ] `|det(M)|` corresponde ao número de réplicas?
- [ ] `reference_positions` descreve a topologia ideal?
- [ ] A ordem dos átomos físicos e de referência é a mesma?
- [ ] O gauge das posições corresponde ao gauge dos CSR?

### Pontos k

- [ ] O caminho está expresso na base recíproca da célula primitiva?
- [ ] O folding usa `K=Mᵀk mod 1`?
- [ ] A comparação primitiva usa exatamente os mesmos pontos `k`?
- [ ] Os vértices repetidos, como Γ, dão o mesmo resultado?

### Álgebra

- [ ] `H(k)` e `S(k)` são hermitianos?
- [ ] `S(k)` é positivo definido?
- [ ] `C†SC≈I`?
- [ ] Os operadores de translação são unitários e comutam?
- [ ] Os resíduos da diagonalização conjunta são pequenos?

### Física

- [ ] Os pesos estão em `[0,1]`?
- [ ] A regra de soma é satisfeita?
- [ ] O peso total selecionado tem o valor esperado?
- [ ] O caso pristino acompanha as bandas primitivas?
- [ ] O vazamento além da tolerância energética é pequeno?
- [ ] O resultado é estável ao aumentar os pontos e variar o alargamento?

---

## 14. Limitações importantes

A implementação atual supõe:

- matrizes reais em espaço real;
- spin colinear;
- correspondência entre os AOs físicos e os AOs da topologia de referência;
- mesma quantidade e ordenação de orbitais nos sítios correspondentes.

Vacâncias, intersticiais e substituições que mudam o número de AOs exigem uma
extensão do espaço com orbitais auxiliares (*placeholders*), ainda não
implementada.

Acoplamento spin–órbita ou magnetismo não colinear pode exigir matrizes
complexas em espaço real e tratamento espinorial.

O gauge automático do CP2K depende das translações presentes nos CSR para
identificar direções ativas. Um sistema com configuração incomum deve ser
validado explicitamente com um caso pristino equivalente.

---

## 15. Exercícios para estudo

### Exercício 1 — multiplicidade

Considere

```math
M=
\begin{pmatrix}
2&1&0\\
0&2&0\\
0&0&1
\end{pmatrix}.
```

Calcule a multiplicidade da supercélula e o número de pontos primitivos que
dobram sobre cada `K`.

**Resposta:** `|det(M)|=4`; portanto, quatro pontos primitivos.

### Exercício 2 — folding

Para `M=diag(2,2,1)` e

```math
k=(2/3,1/3,0),
```

calcule `K=Mᵀk mod 1`.

**Resposta:**

```math
K=(4/3,2/3,0)\pmod1=(1/3,2/3,0).
```

### Exercício 3 — gauge

Em uma rede unidimensional de comprimento `a`, um átomo está em `s=0,8`.
Qual representante centralizado o CP2K usa e qual deslocamento deve ser
aplicado à posição de referência?

**Resposta:** `n=round(0,8)=1`, logo `s_c=-0,2`. Deve-se aplicar
`r_ref^CSR=r_ref-a`.

### Exercício 4 — regra de soma

Explique por que `Σ_k W^k_{KN}=1` pode ser satisfeita mesmo quando o gráfico
atribui peso ao ponto `k` errado.

**Resposta:** a regra testa a completude do conjunto de projetores. Uma
permutação ou rotulação incorreta dos subespaços pode conservar a soma total,
mas associar cada parcela ao `k` errado.

### Exercício 5 — convergência visual

Gere o mapa com 16, 32 e 48 pontos por segmento e com `σ=0,035`, `0,055` e
`0,080 eV`. Separe mudanças devidas à amostragem física das mudanças devidas
apenas ao alargamento.

---

## 16. Resumo conceitual

O método pode ser condensado em seis ideias:

1. a supercélula dobra vários pontos `k` sobre o mesmo `K`;
2. os estados LCAO são resolvidos com a métrica não ortogonal `S(K)`;
3. operadores de translação primitiva identificam os subespaços de caráter
   `k`;
4. a transformação de Löwdin permite projetar em uma representação
   ortonormal;
5. matrizes, posições, imagens e fases precisam usar o mesmo gauge periódico;
6. um caso pristino comparado a bandas primitivas independentes é o teste
   mais informativo da implementação completa.

A correção do CP2K não mudou a física do projetor. Ela garantiu que a
representação geométrica usada pelo projetor fosse a mesma representação
periódica usada pelas matrizes CSR. Essa consistência de gauge é o ponto
central para entender por que pesos espúrios podiam parecer numericamente
plausíveis e, ainda assim, estar associados às bandas erradas.
