"""
Conversor de saída CP2K `KS_CSR_WRITE`/`S_CSR_WRITE` para o HDF5 canônico.

O CP2K escreve um arquivo de texto por imagem periódica em espaço real, mais
uma tabela na saída principal mapeando números de imagem para translações
inteiras. Este arquivo mantém todo o conhecimento desse formato de
nomes/log fora do núcleo do pacote de unfolding.

O importador espera `REAL_SPACE T`, `BINARY F` e `UPPER_TRIANGULAR F`.
"""

function _cp2k_float(token)
    # O CP2K/Fortran pode usar E ou D como caractere de expoente decimal.
    value = tryparse(Float64, replace(token, 'D' => 'E', 'd' => 'e'))
    value === nothing && error("invalid CP2K floating-point token '$token'")
    value
end

function _read_cp2k_csr_matrix(path, n)
    # Apesar do nome da chave de impressão, a saída em texto é uma lista de
    # coordenadas: `linha coluna valor`, indexada a partir de 1. O construtor
    # esparso do Julia converte para CSC e também combina entradas repetidas,
    # caso o CP2K algum dia as emita.
    rows = Int[]
    cols = Int[]
    values = Float64[]
    for (iline, line) in enumerate(eachline(path))
        isempty(strip(line)) && continue
        fields = split(line)
        length(fields) == 3 || error("malformed CSR line $iline in $path")
        row = tryparse(Int, fields[1])
        col = tryparse(Int, fields[2])
        (row === nothing || col === nothing || !(1 <= row <= n) || !(1 <= col <= n)) &&
            error("invalid CSR indices at line $iline in $path")
        push!(rows, row)
        push!(cols, col)
        push!(values, _cp2k_float(fields[3]))
    end
    sparse(rows, cols, values, n, n)
end

function _cp2k_image_mapping(lines, label)
    # Um bloco típico de saída é:
    #   KS CSR write|  21 periodic images
    #        Number    X    Y    Z
    #           1      0    0    0
    # O último bloco é escolhido porque um arquivo pode conter saída de
    # restart/histórico de mais de um cálculo eletrônico.
    pattern = Regex("^\\s*" * label * " CSR write\\|\\s+(\\d+) periodic images\\s*\$")
    starts = findall(line -> match(pattern, line) !== nothing, lines)
    isempty(starts) && error("$label CSR image mapping not found in CP2K output")
    start = last(starts)
    nR = parse(Int, match(pattern, lines[start]).captures[1])
    R = zeros(Int, 3, nR)
    found = 0
    for i in (start+1):length(lines)
        fields = split(strip(lines[i]))
        length(fields) == 4 || continue
        values = tryparse.(Int, fields)
        any(isnothing, values) && continue
        # Exigir números de imagem consecutivos evita que outras linhas de
        # quatro inteiros, mais adiante na saída do CP2K, sejam confundidas
        # com esta tabela.
        values[1] == found + 1 || continue
        found += 1
        R[:, found] = Int[values[2], values[3], values[4]]
        found == nR && break
    end
    found == nR || error("incomplete $label image mapping")
    R
end

_cp2k_regex_escape(text) = replace(text, r"([^A-Za-z0-9_])" => s"\\\1")

function _cp2k_csr_files(prefix, label, spin, nR)
    # Para o prefixo `ham`, o CP2K gera nomes como
    # `ham-KS_SPIN_1_R_3-1_0.csr`. Os sufixos de iteração podem variar, então
    # só o prefixo estável, o rótulo, o spin e o número da imagem são
    # interpretados.
    directory = dirname(abspath(prefix))
    base = basename(prefix)
    pattern = Regex("^" * _cp2k_regex_escape(base) * "-" * label *
                    "_SPIN_" * string(spin) * "_R_(\\d+).*\\.csr\$")
    files = Vector{String}(undef, nR)
    seen = falses(nR)
    for name in readdir(directory)
        matched = match(pattern, name)
        matched === nothing && continue
        index = parse(Int, matched.captures[1])
        1 <= index <= nR || continue
        seen[index] && error("duplicate $label CSR image $index")
        files[index] = joinpath(directory, name)
        seen[index] = true
    end
    all(seen) || error("missing $label CSR images: " * join(findall(.!seen), ","))
    files
end

"""
    read_cp2k_csr(output, h_prefix, s_prefix; lattice, positions,
                  atomic_numbers, norb, reference_positions=positions, spins=[1])

Importa as matrizes em espaço real do CP2K para um `RealSpaceModel`. Esta é
a única camada específica do CP2K; o código a jusante lê apenas o formato
HDF5 canônico.

Os argumentos nomeados obrigatórios descrevem informação que não é
codificada de forma confiável nos arquivos CSR: rede, posições cartesianas,
números atômicos e número de AOs por átomo. O número e a ordem dos AOs
devem corresponder exatamente às linhas/colunas da matriz do CP2K.

Para estruturas relaxadas, passe os sítios ideais separadamente em
`reference_positions`. `spins` contém os números de canal de spin do CP2K a
importar.
"""
function read_cp2k_csr(output::AbstractString,
                       h_prefix::AbstractString,
                       s_prefix::AbstractString;
                       lattice,
                       positions,
                       atomic_numbers,
                       norb,
                       reference_positions=positions,
                       spins=[1],
                       validate=true)
    lines = readlines(output)
    # Não arquivar estruturas eletrônicas parciais ou que falharam. Se um
    # abort ocorreu apenas antes de um restart bem-sucedido posterior, o
    # resultado final ainda é válido.
    converged = findlast(contains("SCF run converged"), lines)
    aborted = findlast(contains("[ABORT]"), lines)
    converged === nothing && error("CP2K output has no converged SCF run")
    (aborted === nothing || aborted < converged) ||
        error("CP2K aborted after its last convergence")
    # H(R) e S(R) precisam ser indexados pelas mesmas translações para que o
    # problema de autovalores generalizado no espaço de Fourier faça sentido.
    RH = _cp2k_image_mapping(lines, "KS")
    RS = _cp2k_image_mapping(lines, "S")
    RH == RS || error("KS and S image mappings differ")
    n = sum(norb)
    nR = size(RH, 2)
    # Preserva os dois níveis de aninhamento H[spin][image]. O overlap do
    # CP2K é comum aos canais de spin, então só S_SPIN_1 é lido.
    H = [[_read_cp2k_csr_matrix(path, n)
          for path in _cp2k_csr_files(h_prefix, "KS", spin, nR)]
         for spin in spins]
    S = [_read_cp2k_csr_matrix(path, n)
         for path in _cp2k_csr_files(s_prefix, "S", 1, nR)]
    RealSpaceModel(lattice, positions, reference_positions, atomic_numbers,
                   norb, RH, H, S;
                   energy_unit="hartree", length_unit="angstrom",
                   source="CP2K", validate=validate)
end

"""
    convert_cp2k_to_hdf5(h5_path, output, h_prefix, s_prefix; kwargs...)

Importa a saída CSR do CP2K, valida o modelo resultante e escreve um único
arquivo HDF5 canônico. Retorna o `RealSpaceModel` em memória para inspeção
imediata.
"""
function convert_cp2k_to_hdf5(h5_path::AbstractString, output, h_prefix, s_prefix; kwargs...)
    model = read_cp2k_csr(output, h_prefix, s_prefix; kwargs...)
    write_model(h5_path, model)
    return model
end
