# Alterar qualquer um dos dois valores abaixo é uma mudança de formato de
# arquivo. Os leitores rejeitam versões desconhecidas em vez de tentar
# interpretar silenciosamente uma convenção de matriz diferente.
const SCHEMA_NAME = "unfolding.realspace"
const SCHEMA_VERSION = (1, 0)

"""
    RealSpaceModel

Modelo LCAO em espaço real, independente do código de estrutura eletrônica
que o gerou. Dimensões dos campos:

- `lattice`: `3 × 3`, vetores de rede diretos como colunas;
- `positions`, `reference_positions`: `3 × natom`, átomos como colunas;
- `R`: `3 × nimages`, translações inteiras da rede direta;
- `H[spin][image]`, `S[image]`: matrizes esparsas `nbasis × nbasis`.

A ordenação dos orbitais atômicos (AOs) é por átomo: as primeiras `norb[1]`
linhas/colunas pertencem ao átomo 1, seguidas por `norb[2]`, e assim por
diante. `reference_positions` guarda a topologia ideal usada para construir
o projetor de unfolding; `positions` pode guardar a geometria física
relaxada ou com defeitos. Vetores de rede e posições usam `length_unit`;
Hamiltonianos usam `energy_unit`.

Para cada imagem, o pacote usa a convenção de Fourier
`A(k) = sum_R exp(+i 2π k⋅R) A(R)`.
"""
struct RealSpaceModel
    lattice::Matrix{Float64}             # Vetores diretos como colunas.
    positions::Matrix{Float64}           # Coordenadas cartesianas físicas.
    reference_positions::Matrix{Float64} # Sítios ideais usados por T_PC.
    atomic_numbers::Vector{Int}           # Mesma ordem de átomos que positions.
    norb::Vector{Int}                     # Nº de AOs consecutivos por átomo.
    R::Matrix{Int}                        # Translações inteiras, uma por imagem.
    H::Vector{Vector{SparseMatrixCSC{Float64,Int}}} # [spin][image].
    S::Vector{SparseMatrixCSC{Float64,Int}}         # [image], igual para todo spin.
    energy_unit::String
    length_unit::String
    source::String
end

function RealSpaceModel(lattice, positions, reference_positions, atomic_numbers,
                        norb, R, H, S; energy_unit="eV", length_unit="angstrom",
                        source="unknown", validate=true)
    # Este construtor externo normaliza os tipos de array na fronteira de
    # entrada; o construtor interno passa a trabalhar com uma única
    # representação previsível.
    model = RealSpaceModel(Matrix{Float64}(lattice), Matrix{Float64}(positions),
        Matrix{Float64}(reference_positions), Vector{Int}(atomic_numbers),
        Vector{Int}(norb), Matrix{Int}(R),
        [SparseMatrixCSC{Float64,Int}.(channel) for channel in H],
        SparseMatrixCSC{Float64,Int}.(S), String(energy_unit),
        String(length_unit), String(source))
    validate && validate_model(model)
    return model
end

nspin(model::RealSpaceModel) = length(model.H)
nbasis(model::RealSpaceModel) = sum(model.norb)
nimages(model::RealSpaceModel) = size(model.R, 2)

function _reciprocity_deviation(matrices, R)
    # Hermiticidade no espaço recíproco para todo k equivale, no espaço real,
    # à relação A(-R) = A(R)†. O resíduo é normalizado pela maior norma
    # encontrada para que a mesma tolerância faça sentido em Hamiltonianos
    # com unidades de energia diferentes.
    lookup = Dict(Tuple(R[:, i]) => i for i in axes(R, 2))
    dev = 0.0
    scale = 1.0
    for i in axes(R, 2)
        j = get(lookup, Tuple(-R[:, i]), 0)
        j == 0 && return Inf
        dev = max(dev, norm(matrices[i] - matrices[j]', Inf))
        scale = max(scale, norm(matrices[i], Inf))
    end
    return dev / scale
end

"""
    validate_model(model; atol=1e-10)

Valida dimensões, valores finitos, unidades, unicidade das translações e as
relações de reciprocidade em espaço real `H(-R)=H(R)†` e `S(-R)=S(R)†`.

Essas checagens são feitas propositalmente antes de qualquer cálculo de
bandas: uma ordenação de AOs errada, ou um mapeamento de imagem errado,
poderia produzir matrizes em espaço recíproco não-Hermitianas — mas ainda
assim de aparência plausível — dificultando o diagnóstico depois.
"""
function validate_model(model::RealSpaceModel; atol=1e-10)
    # Geometria e metadados dos átomos.
    size(model.lattice) == (3, 3) || error("lattice must be 3x3")
    abs(det(model.lattice)) > eps(Float64) || error("lattice must be nonsingular")
    size(model.positions, 1) == 3 || error("positions must have three rows")
    size(model.reference_positions) == size(model.positions) || error("reference_positions shape mismatch")
    natom = size(model.positions, 2)
    length(model.atomic_numbers) == natom == length(model.norb) || error("atom metadata mismatch")
    all(>(0), model.atomic_numbers) || error("atomic numbers must be positive")
    all(>(0), model.norb) || error("norb must be positive for every atom")
    # Contabilidade das imagens periódicas.
    size(model.R, 1) == 3 || error("R must have three rows")
    nR = nimages(model)
    nR > 0 || error("at least one real-space image is required")
    length(Set(Tuple(model.R[:,i]) for i in axes(model.R,2))) == nR || error("duplicate translations in R")
    any(all(iszero, model.R[:,i]) for i in axes(model.R,2)) || error("R must contain the home image (0,0,0)")
    # Todos os blocos de matriz devem seguir a mesma ordenação e dimensão de AOs.
    n = nbasis(model)
    length(model.S) == nR || error("S/R image count mismatch")
    !isempty(model.H) || error("at least one spin channel is required")
    all(length(channel) == nR for channel in model.H) || error("H/R image count mismatch")
    all(size(A) == (n, n) for A in model.S) || error("overlap dimensions mismatch")
    all(size(A) == (n, n) for channel in model.H for A in channel) || error("Hamiltonian dimensions mismatch")
    # Rejeita NaN/Inf já na entrada; uma falha do autosolver mais adiante
    # seria muito mais difícil de diagnosticar.
    all(isfinite, model.lattice) && all(isfinite, model.positions) && all(isfinite, model.reference_positions) ||
        error("geometry contains non-finite values")
    all(all(isfinite, nonzeros(A)) for A in model.S) || error("overlap contains non-finite values")
    all(all(isfinite, nonzeros(A)) for channel in model.H for A in channel) ||
        error("Hamiltonian contains non-finite values")
    !isempty(model.energy_unit) && !isempty(model.length_unit) || error("units must be nonempty")
    # Esta é a checagem física de consistência essencial para a reconstrução
    # de Fourier.
    _reciprocity_deviation(model.S, model.R) <= atol || error("S(-R) != S(R)†")
    for (spin, channel) in enumerate(model.H)
        _reciprocity_deviation(channel, model.R) <= atol || error("H(-R) != H(R)† for spin $spin")
    end
    return true
end

_write_string(group, name, value) = write(group, name, collect(codeunits(value)))
_read_string(group, name) = String(Vector{UInt8}(read(group[name])))

function _write_sparse_set(parent, name, matrices)
    # Guarda os arrays CSC do Julia/SuiteSparse diretamente. Manter a
    # estrutura esparsa evita transformar modelos de orbitais localizados em
    # datasets HDF5 densos e muito maiores do que o necessário.
    group = create_group(parent, name)
    for (i, A) in enumerate(matrices)
        item = create_group(group, string(i))
        write(item, "colptr", A.colptr)
        write(item, "rowval", A.rowval)
        write(item, "nzval", A.nzval)
    end
end

function _read_sparse_set(parent, name, n, nR)
    # As dimensões da matriz vêm de sum(norb); o formato CSC em si só guarda
    # ponteiros de coluna, índices de linha e valores não-nulos.
    group = parent[name]
    [SparseMatrixCSC(n, n, Vector{Int}(read(group[string(i)]["colptr"])),
        Vector{Int}(read(group[string(i)]["rowval"])),
        Vector{Float64}(read(group[string(i)]["nzval"]))) for i in 1:nR]
end

"""
    write_model(path, model)

Escreve a representação HDF5 canônica e independente de código, documentada
em `docs/hdf5-schema.md`. O modelo é validado antes que o arquivo seja
substituído.
"""
function write_model(path::AbstractString, model::RealSpaceModel)
    validate_model(model)
    h5open(path, "w") do file
        # A hierarquia separa dados físicos de metadados de formato, para que
        # futuros conversores não precisem reproduzir nenhum layout de
        # arquivo específico de um código.
        schema = create_group(file, "schema")
        _write_string(schema, "name", SCHEMA_NAME)
        write(schema, "version", collect(SCHEMA_VERSION))
        structure = create_group(file, "structure")
        write(structure, "lattice", model.lattice)
        write(structure, "positions", model.positions)
        write(structure, "reference_positions", model.reference_positions)
        write(structure, "atomic_numbers", model.atomic_numbers)
        basis = create_group(file, "basis")
        write(basis, "norb_per_atom", model.norb)
        translations = create_group(file, "translations")
        write(translations, "R", model.R)
        matrices = create_group(file, "matrices")
        _write_sparse_set(matrices, "overlap", model.S)
        hgroup = create_group(matrices, "hamiltonian")
        for spin in 1:nspin(model)
            _write_sparse_set(hgroup, "spin_$spin", model.H[spin])
        end
        metadata = create_group(file, "metadata")
        _write_string(metadata, "energy_unit", model.energy_unit)
        _write_string(metadata, "length_unit", model.length_unit)
        _write_string(metadata, "source", model.source)
        write(metadata, "nspin", [nspin(model)])
    end
    return path
end

"""
    read_model(path; validate=true)

Lê o formato HDF5 canônico e retorna um `RealSpaceModel`. Todos os
solvers do pacote consomem apenas este tipo; a saída nativa de cada código
de estrutura eletrônica é tratada exclusivamente pelos conversores.
"""
function read_model(path::AbstractString; validate=true)
    h5open(path, "r") do file
        _read_string(file["schema"], "name") == SCHEMA_NAME || error("unsupported HDF5 schema")
        Tuple(read(file["schema/version"])) == SCHEMA_VERSION || error("unsupported schema version")
        lattice = read(file["structure/lattice"])
        positions = read(file["structure/positions"])
        reference = read(file["structure/reference_positions"])
        atomic_numbers = Vector{Int}(read(file["structure/atomic_numbers"]))
        norb = Vector{Int}(read(file["basis/norb_per_atom"]))
        R = Matrix{Int}(read(file["translations/R"]))
        # Os blocos esparsos não repetem suas dimensões nem o número de
        # imagens; ambos são derivados dos metadados de base e translações.
        n = sum(norb); nR = size(R, 2)
        S = _read_sparse_set(file["matrices"], "overlap", n, nR)
        ns = Int(only(read(file["metadata/nspin"])))
        H = [_read_sparse_set(file["matrices/hamiltonian"], "spin_$s", n, nR) for s in 1:ns]
        RealSpaceModel(lattice, positions, reference, atomic_numbers, norb, R, H, S;
            energy_unit=_read_string(file["metadata"], "energy_unit"),
            length_unit=_read_string(file["metadata"], "length_unit"),
            source=_read_string(file["metadata"], "source"), validate=validate)
    end
end

function model_summary(model::RealSpaceModel)
    "$(size(model.positions,2)) atoms, $(nbasis(model)) AOs, $(nimages(model)) images, " *
    "$(nspin(model)) spin channel(s), energy in $(model.energy_unit), source=$(model.source)"
end
