"""
Conversor de saída CP2K `KS_CSR_WRITE`/`S_CSR_WRITE` para o HDF5 canônico.

O CP2K escreve um arquivo de texto por imagem periódica em espaço real, mais
uma tabela na saída principal mapeando números de imagem para translações
inteiras. Este arquivo mantém todo o conhecimento desse formato de
nomes/log fora do núcleo do pacote de unfolding.

O importador espera `REAL_SPACE T`, `BINARY F` e `UPPER_TRIANGULAR F`.
"""

const _BOHR_TO_ANGSTROM = 0.529177210903

"""
    CP2KFiles(output, csr_prefix, cube, molog; reference_cube=nothing)

Conjunto mínimo de arquivos necessário para importar um cálculo CP2K real:

- `output`: saída textual principal do CP2K, que contém a associação entre
  os índices `R_n` dos CSR e as translações periódicas;
- `csr_prefix`: caminho e prefixo comum aos arquivos
  `*-KS_SPIN_*-R_*.csr` e `*-S_SPIN_*-R_*.csr`;
- `cube`: qualquer `.cube` do mesmo cálculo, usado somente para ler rede,
  posições e números atômicos;
- `molog`: saída de `&PRINT &MO`, usada somente para recuperar a ordem e o
  número de AOs de cada átomo.

Para uma estrutura relaxada, `reference_cube` pode apontar para um `.cube`
com a geometria ideal da mesma supercélula. Se omitido, a própria geometria
de `cube` define a topologia de referência.
"""
struct CP2KFiles
    output::String
    csr_prefix::String
    cube::String
    molog::String
    reference_cube::Union{Nothing,String}
end

function CP2KFiles(output::AbstractString, csr_prefix::AbstractString,
                   cube::AbstractString, molog::AbstractString;
                   reference_cube::Union{Nothing,AbstractString}=nothing)
    CP2KFiles(String(output), String(csr_prefix), String(cube), String(molog),
              reference_cube === nothing ? nothing : String(reference_cube))
end

function _read_cp2k_cube_geometry(path::AbstractString)
    open(path, "r") do io
        eof(io) && error("empty CP2K cube file: $path")
        readline(io)
        eof(io) && error("truncated CP2K cube file: $path")
        readline(io)

        header = split(strip(readline(io)))
        length(header) >= 4 || error("malformed atom-count line in CP2K cube: $path")
        natom = tryparse(Int, header[1])
        natom === nothing && error("invalid atom count in CP2K cube: $path")
        natom = abs(natom)

        lattice_bohr = zeros(Float64, 3, 3)
        for axis in 1:3
            fields = split(strip(readline(io)))
            length(fields) >= 4 || error("malformed grid line $axis in CP2K cube: $path")
            ngrid = tryparse(Int, fields[1])
            ngrid === nothing && error("invalid grid size on line $axis in CP2K cube: $path")
            step = _cp2k_float.(fields[2:4])
            lattice_bohr[:, axis] = abs(ngrid) .* step
        end

        atomic_numbers = Vector{Int}(undef, natom)
        positions_bohr = zeros(Float64, 3, natom)
        for atom in 1:natom
            fields = split(strip(readline(io)))
            length(fields) >= 5 || error("malformed atom line $atom in CP2K cube: $path")
            z = tryparse(Int, fields[1])
            z === nothing && error("invalid atomic number on atom line $atom in CP2K cube: $path")
            atomic_numbers[atom] = z
            positions_bohr[:, atom] = _cp2k_float.(fields[3:5])
        end

        # CP2K escreve seus cubes em unidades atômicas. O formato canônico do
        # pacote usa angstrom para os dados geométricos importados do CP2K.
        return (lattice=lattice_bohr .* _BOHR_TO_ANGSTROM,
                positions=positions_bohr .* _BOHR_TO_ANGSTROM,
                atomic_numbers=atomic_numbers)
    end
end

const _ELEMENT_Z = Dict(
    "H"=>1, "He"=>2, "Li"=>3, "Be"=>4, "B"=>5, "C"=>6, "N"=>7, "O"=>8,
    "F"=>9, "Ne"=>10, "Na"=>11, "Mg"=>12, "Al"=>13, "Si"=>14, "P"=>15,
    "S"=>16, "Cl"=>17, "Ar"=>18, "K"=>19, "Ca"=>20, "Sc"=>21, "Ti"=>22,
    "V"=>23, "Cr"=>24, "Mn"=>25, "Fe"=>26, "Co"=>27, "Ni"=>28, "Cu"=>29,
    "Zn"=>30, "Ga"=>31, "Ge"=>32, "As"=>33, "Se"=>34, "Br"=>35, "Kr"=>36,
    "Rb"=>37, "Sr"=>38, "Y"=>39, "Zr"=>40, "Nb"=>41, "Mo"=>42, "Tc"=>43,
    "Ru"=>44, "Rh"=>45, "Pd"=>46, "Ag"=>47, "Cd"=>48, "In"=>49, "Sn"=>50,
    "Sb"=>51, "Te"=>52, "I"=>53, "Xe"=>54, "Cs"=>55, "Ba"=>56, "La"=>57,
    "Ce"=>58, "Pr"=>59, "Nd"=>60, "Pm"=>61, "Sm"=>62, "Eu"=>63, "Gd"=>64,
    "Tb"=>65, "Dy"=>66, "Ho"=>67, "Er"=>68, "Tm"=>69, "Yb"=>70, "Lu"=>71,
    "Hf"=>72, "Ta"=>73, "W"=>74, "Re"=>75, "Os"=>76, "Ir"=>77, "Pt"=>78,
    "Au"=>79, "Hg"=>80, "Tl"=>81, "Pb"=>82, "Bi"=>83, "Po"=>84, "At"=>85,
    "Rn"=>86)

function _read_cp2k_molog_basis(path::AbstractString)
    # Cada bloco de quatro MOs repete as mesmas linhas de AO. Guardar o mapa
    # por índice torna o parser independente de quantos blocos foram impressos.
    ao_atom = Dict{Int,Int}()
    atom_symbol = Dict{Int,String}()
    pattern = r"^\s*MO\|\s+(\d+)\s+(\d+)\s+([A-Za-z][A-Za-z]?)\s+\S+"
    for line in eachline(path)
        matched = match(pattern, line)
        matched === nothing && continue
        ao = parse(Int, matched.captures[1])
        atom = parse(Int, matched.captures[2])
        symbol = matched.captures[3]
        if haskey(ao_atom, ao) && ao_atom[ao] != atom
            error("MOLog assigns AO $ao to more than one atom in $path")
        end
        if haskey(atom_symbol, atom) && atom_symbol[atom] != symbol
            error("MOLog assigns atom $atom to more than one element in $path")
        end
        ao_atom[ao] = atom
        atom_symbol[atom] = symbol
    end
    isempty(ao_atom) && error("no spherical AO rows found in CP2K MOLog: $path")

    nao = maximum(keys(ao_atom))
    sort!(collect(keys(ao_atom))) == collect(1:nao) ||
        error("CP2K MOLog has a non-contiguous AO index sequence in $path")
    natom = maximum(values(ao_atom))
    sort!(collect(keys(atom_symbol))) == collect(1:natom) ||
        error("CP2K MOLog has a non-contiguous atom index sequence in $path")
    norb = [count(==(atom), values(ao_atom)) for atom in 1:natom]
    atomic_numbers = Vector{Int}(undef, natom)
    for atom in 1:natom
        symbol = atom_symbol[atom]
        haskey(_ELEMENT_Z, symbol) || error("unsupported element '$symbol' in CP2K MOLog")
        atomic_numbers[atom] = _ELEMENT_Z[symbol]
    end
    return (norb=norb, atomic_numbers=atomic_numbers)
end

function _cp2k_file_metadata(files::CP2KFiles)
    for (label, path) in (("output", files.output), ("cube", files.cube),
                          ("MOLog", files.molog))
        isfile(path) || error("CP2K $label file not found: $path")
    end
    geometry = _read_cp2k_cube_geometry(files.cube)
    basis = _read_cp2k_molog_basis(files.molog)
    length(geometry.atomic_numbers) == length(basis.norb) ||
        error("CP2K cube/MOLog atom-count mismatch: $(length(geometry.atomic_numbers)) != $(length(basis.norb))")
    geometry.atomic_numbers == basis.atomic_numbers ||
        error("CP2K cube/MOLog atom order or chemical elements differ")

    reference_positions = geometry.positions
    if files.reference_cube !== nothing
        reference = _read_cp2k_cube_geometry(files.reference_cube)
        reference.atomic_numbers == geometry.atomic_numbers ||
            error("CP2K reference cube has different atom order or chemical elements")
        norm(reference.lattice - geometry.lattice, Inf) <= 1e-8 ||
            error("CP2K reference cube has a different lattice")
        reference_positions = reference.positions
    end
    return (lattice=geometry.lattice, positions=geometry.positions,
            atomic_numbers=geometry.atomic_numbers, norb=basis.norb,
            reference_positions=reference_positions)
end

function _cp2k_float(token)
    # O CP2K/Fortran pode usar E ou D como caractere de expoente decimal.
    value = tryparse(Float64, replace(token, 'D' => 'E', 'd' => 'e'))
    value === nothing && error("invalid CP2K floating-point token '$token'")
    value
end

function _cp2k_length_factor(unit::AbstractString)
    normalized = uppercase(strip(unit))
    normalized in ("ANGSTROM", "ANGSTROMS", "A") && return 1.0
    normalized in ("BOHR", "AU", "A.U.") && return _BOHR_TO_ANGSTROM
    error("unsupported CP2K length unit '$unit' in output metadata")
end

function _cp2k_output_metadata(lines::AbstractVector{<:AbstractString})
    missing_hint = "The CP2K output does not contain the metadata needed by the two-name " *
        "converter. Enable SUBSYS%PRINT%CELL, ATOMIC_COORDINATES and KINDS (or use PRINT_LEVEL MEDIUM)."

    # CELL| Vector a/b/c [angstrom]: ax ay az ...
    lattice = zeros(Float64, 3, 3)
    lengths = zeros(Float64, 3)
    found_axes = falses(3)
    cell_pattern = r"^\s*CELL\|\s+Vector\s+([abc])\s+\[([^\]]+)\]:\s+(\S+)\s+(\S+)\s+(\S+).*\|[abc]\|\s*=\s*(\S+)"
    for line in lines
        matched = match(cell_pattern, line)
        matched === nothing && continue
        axis = findfirst(==(matched.captures[1][1]), ['a', 'b', 'c'])
        factor = _cp2k_length_factor(matched.captures[2])
        lattice[:, axis] = _cp2k_float.(matched.captures[3:5]) .* factor
        lengths[axis] = _cp2k_float(matched.captures[6]) * factor
        found_axes[axis] = true
    end
    all(found_axes) || error(missing_hint * " Missing CELL vectors.")

    # CP2K formats vector components with only three decimals, but prints
    # lengths and angles with six. Reconstruct the precise metric tensor and
    # retain the Cartesian orientation supplied by the approximate vectors.
    angles = Dict{String,Float64}()
    angle_pattern = r"^\s*CELL\|\s+Angle\s+\([^)]*\),\s+(alpha|beta|gamma)\s+\[degree\]:\s+(\S+)"
    for line in lines
        matched = match(angle_pattern, line)
        matched === nothing && continue
        angles[matched.captures[1]] = _cp2k_float(matched.captures[2])
    end
    all(haskey(angles, name) for name in ("alpha", "beta", "gamma")) ||
        error(missing_hint * " Missing CELL angles.")
    a, b, c = lengths
    α, β, γ = deg2rad.((angles["alpha"], angles["beta"], angles["gamma"]))
    metric = [a^2 a*b*cos(γ) a*c*cos(β);
              a*b*cos(γ) b^2 b*c*cos(α);
              a*c*cos(β) b*c*cos(α) c^2]
    approximate_metric = lattice' * lattice
    matrix_sqrt(A) = begin
        values, vectors = eigen(Symmetric(A))
        minimum(values) > 0 || error("invalid non-positive CELL metric in CP2K output")
        vectors * Diagonal(sqrt.(values)) * vectors'
    end
    lattice = lattice * inv(matrix_sqrt(approximate_metric)) * matrix_sqrt(metric)

    # The final complete QUICKSTEP coordinate block wins when outputs from
    # restarts/reruns were concatenated into the same file.
    coord_header = r"^\s*MODULE\s+QUICKSTEP:\s+ATOMIC COORDINATES IN\s+(\S+)"
    coord_starts = [(i, match(coord_header, line)) for (i, line) in enumerate(lines)
                    if match(coord_header, line) !== nothing]
    isempty(coord_starts) && error(missing_hint * " Missing QUICKSTEP atomic coordinates.")
    coord_start, coord_match = last(coord_starts)
    factor = _cp2k_length_factor(coord_match.captures[1])
    atom_pattern = r"^\s*(\d+)\s+(\d+)\s+([A-Za-z]{1,2})\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)"
    kind_indices = Int[]
    atomic_numbers = Int[]
    position_columns = Vector{Float64}[]
    started = false
    for i in (coord_start + 1):length(lines)
        matched = match(atom_pattern, lines[i])
        if matched === nothing
            started && break
            continue
        end
        started = true
        push!(kind_indices, parse(Int, matched.captures[2]))
        push!(atomic_numbers, parse(Int, matched.captures[4]))
        push!(position_columns, _cp2k_float.(matched.captures[5:7]) .* factor)
    end
    isempty(position_columns) && error(missing_hint * " Atomic coordinate table is empty.")
    positions = reduce(hcat, position_columns)

    # KINDS prints the contracted spherical-AO count once per atomic kind.
    kind_starts = findall(contains("ATOMIC KIND INFORMATION"), lines)
    isempty(kind_starts) && error(missing_hint * " Missing ATOMIC KIND INFORMATION.")
    kind_norb = Dict{Int,Int}()
    current_kind = 0
    in_orbital_basis = false
    kind_pattern = r"^\s*(\d+)\.\s+Atomic kind:"
    norb_pattern = r"Number of spherical basis functions:\s+(\d+)"
    for line in @view lines[last(kind_starts):end]
        kind_match = match(kind_pattern, line)
        if kind_match !== nothing
            current_kind = parse(Int, kind_match.captures[1])
            in_orbital_basis = false
            continue
        end
        if occursin("Orbital Basis Set", line)
            in_orbital_basis = true
            continue
        end
        norb_match = match(norb_pattern, line)
        if in_orbital_basis && norb_match !== nothing
            kind_norb[current_kind] = parse(Int, norb_match.captures[1])
            in_orbital_basis = false
        end
    end
    missing_kinds = sort!(unique(k for k in kind_indices if !haskey(kind_norb, k)))
    isempty(missing_kinds) ||
        error(missing_hint * " Missing spherical-AO count for kinds $(join(missing_kinds, ", ")).")
    norb = [kind_norb[k] for k in kind_indices]

    return (lattice=lattice, positions=positions,
            atomic_numbers=atomic_numbers, norb=norb)
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

"""
    _cp2k_reference_in_csr_gauge(lattice, positions, reference_positions, R;
                                 half_tolerance=1e-5)

Move the reference sites to the same periodic-image representatives used by
CP2K's real-space matrices.  CP2K builds the AO neighbor list from
`pbc(position, cell)`, which subtracts the nearest integer from each active
fractional coordinate.  The CSR image labels therefore refer to these centered
positions, rather than necessarily to the coordinates printed in the output or
cube file.

The same image shift obtained from the *physical* positions is applied to the
ideal reference positions.  This is important for relaxed structures: the
electronic matrices follow the physical AO image chosen by CP2K, while the
projector must retain the ideal atom mapping.

CP2K prints cell-vector components with less precision than lengths and angles.
Fractional coordinates reconstructed from that output can consequently turn an
exact boundary value such as `0.5` into `0.500003`.  Values within
`half_tolerance` of a half integer are stabilized before rounding; Julia's
ties-to-even rule then keeps the representative already present in the input at
an ambiguous cell boundary.
"""
function _cp2k_reference_in_csr_gauge(lattice, positions, reference_positions, R;
                                      half_tolerance::Real=1e-5)
    A = Matrix{Float64}(lattice)
    physical = Matrix{Float64}(positions)
    reference = Matrix{Float64}(reference_positions)
    size(reference) == size(physical) ||
        throw(DimensionMismatch("reference_positions/positions shape mismatch"))

    fractional = A \ physical
    image_shifts = zeros(Int, size(fractional))
    for direction in axes(fractional, 1)
        # A direction absent from every CSR image is nonperiodic (or has no AO
        # coupling across its boundary), so changing its representative cannot
        # contribute to the reciprocal-space matrices and is best avoided.
        any(!iszero, @view R[direction, :]) || continue
        for atom in axes(fractional, 2)
            value = fractional[direction, atom]
            nearest_half = round(2value) / 2
            stabilized = abs(value - nearest_half) <= half_tolerance ? nearest_half : value
            image_shifts[direction, atom] = round(Int, stabilized)
        end
    end
    reference - A * image_shifts
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
    csr_reference_positions = _cp2k_reference_in_csr_gauge(
        lattice, positions, reference_positions, RH)
    RealSpaceModel(lattice, positions, csr_reference_positions, atomic_numbers,
                   norb, RH, H, S;
                   energy_unit="hartree", length_unit="angstrom",
                   source="CP2K", validate=validate)
end

"""
    read_cp2k_csr(output, csr_prefix; reference_positions=nothing,
                  spins=[1], validate=true)

Importa um cálculo CP2K usando somente a saída textual padrão e o prefixo
comum aos arquivos KS/S CSR. Rede, coordenadas, espécies, ordem atômica e
número de AOs por átomo são extraídos do próprio `output`; o mapa `R_n`
também vem dele. O input CP2K deve habilitar `SUBSYS%PRINT%CELL`,
`ATOMIC_COORDINATES` e `KINDS`, ou usar `PRINT_LEVEL MEDIUM`.

Para uma estrutura relaxada cuja topologia ideal não coincida com as
coordenadas impressas, passe `reference_positions` explicitamente.
"""
function read_cp2k_csr(output::AbstractString, csr_prefix::AbstractString;
                       reference_positions=nothing, spins=[1], validate=true)
    lines = readlines(output)
    metadata = _cp2k_output_metadata(lines)
    refs = reference_positions === nothing ? metadata.positions : reference_positions
    read_cp2k_csr(output, csr_prefix, csr_prefix;
        lattice=metadata.lattice, positions=metadata.positions,
        atomic_numbers=metadata.atomic_numbers, norb=metadata.norb,
        reference_positions=refs, spins=spins, validate=validate)
end

"""
    read_cp2k_csr(files::CP2KFiles; spins=[1], validate=true)

Importa um cálculo CP2K sem exigir que rede, posições, espécies ou número
de AOs sejam transcritos manualmente. Esses dados são extraídos e validados
entre o `.cube` e o `MOLog` indicados em `files`.
"""
function read_cp2k_csr(files::CP2KFiles; spins=[1], validate=true)
    metadata = _cp2k_file_metadata(files)
    read_cp2k_csr(files.output, files.csr_prefix, files.csr_prefix;
        metadata..., spins=spins, validate=validate)
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

"""
    convert_cp2k_to_hdf5(h5_path, output, csr_prefix; kwargs...)

Empacota `H(R)`, `S(R)` e toda a metainformação em um HDF5 canônico usando
somente dois nomes de entrada: o output CP2K e o prefixo comum dos CSR.
"""
function convert_cp2k_to_hdf5(h5_path::AbstractString,
                              output::AbstractString, csr_prefix::AbstractString; kwargs...)
    model = read_cp2k_csr(output, csr_prefix; kwargs...)
    write_model(h5_path, model)
    return model
end


"""
    convert_cp2k_to_hdf5(h5_path, files::CP2KFiles; kwargs...)

Versão de alto nível que extrai automaticamente toda a metainformação do
conjunto `CP2KFiles`.
"""
function convert_cp2k_to_hdf5(h5_path::AbstractString, files::CP2KFiles; kwargs...)
    model = read_cp2k_csr(files; kwargs...)
    write_model(h5_path, model)
    return model
end
