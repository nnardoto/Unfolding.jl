# Changing either value is a file-format change. Readers reject unknown
# versions rather than silently interpreting a different matrix convention.
const SCHEMA_NAME = "unfolding.realspace"
const SCHEMA_VERSION = (1, 0)

"""
    RealSpaceModel

Code-independent real-space LCAO model. Array dimensions are:

- `lattice`: `3 × 3`, with direct-lattice vectors in columns;
- `positions`, `reference_positions`: `3 × natom`, atoms in columns;
- `R`: `3 × nimages`, integer direct-lattice translations;
- `H[spin][image]`, `S[image]`: `nbasis × nbasis` sparse matrices.

The AO ordering is atom-major: the first `norb[1]` rows/columns belong to atom
one, followed by `norb[2]`, and so on. `reference_positions` stores the ideal
topology used to build the unfolding projector; `positions` may store the
relaxed or defective physical geometry. Lattice vectors and positions use
`length_unit`; Hamiltonians use `energy_unit`.

For each image, the package uses
`A(k) = sum_R exp(+i 2π k⋅R) A(R)`.
"""
struct RealSpaceModel
    lattice::Matrix{Float64}             # Direct vectors as columns.
    positions::Matrix{Float64}           # Physical Cartesian coordinates.
    reference_positions::Matrix{Float64} # Ideal sites used by T_PC.
    atomic_numbers::Vector{Int}           # Same atom order as positions.
    norb::Vector{Int}                     # Number of consecutive AOs per atom.
    R::Matrix{Int}                        # Integer translations, one per image.
    H::Vector{Vector{SparseMatrixCSC{Float64,Int}}} # [spin][image].
    S::Vector{SparseMatrixCSC{Float64,Int}}         # [image], spin independent.
    energy_unit::String
    length_unit::String
    source::String
end

function RealSpaceModel(lattice, positions, reference_positions, atomic_numbers,
                        norb, R, H, S; energy_unit="eV", length_unit="angstrom",
                        source="unknown", validate=true)
    # Normalize all user-facing array types at the boundary. The inner
    # constructor then has a single predictable representation.
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
    # Hermiticity in reciprocal space for every k is equivalent to the
    # real-space relation A(-R) = A(R)†. Normalize the residual so the same
    # tolerance is meaningful for Hamiltonians in different energy units.
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

Validate dimensions, finite values, units, unique translations, and the
real-space reciprocity relations `H(-R)=H(R)†` and `S(-R)=S(R)†`.

These checks are deliberately performed before expensive band calculations:
a wrong AO order or image mapping may otherwise produce plausible-looking but
non-Hermitian reciprocal-space matrices.
"""
function validate_model(model::RealSpaceModel; atol=1e-10)
    # Geometry and atom metadata.
    size(model.lattice) == (3, 3) || error("lattice must be 3x3")
    abs(det(model.lattice)) > eps(Float64) || error("lattice must be nonsingular")
    size(model.positions, 1) == 3 || error("positions must have three rows")
    size(model.reference_positions) == size(model.positions) || error("reference_positions shape mismatch")
    natom = size(model.positions, 2)
    length(model.atomic_numbers) == natom == length(model.norb) || error("atom metadata mismatch")
    all(>(0), model.atomic_numbers) || error("atomic numbers must be positive")
    all(>(0), model.norb) || error("norb must be positive for every atom")
    # Periodic-image bookkeeping.
    size(model.R, 1) == 3 || error("R must have three rows")
    nR = nimages(model)
    nR > 0 || error("at least one real-space image is required")
    length(Set(Tuple(model.R[:,i]) for i in axes(model.R,2))) == nR || error("duplicate translations in R")
    any(all(iszero, model.R[:,i]) for i in axes(model.R,2)) || error("R must contain the home image (0,0,0)")
    # All matrix blocks must follow the same AO ordering and dimensions.
    n = nbasis(model)
    length(model.S) == nR || error("S/R image count mismatch")
    !isempty(model.H) || error("at least one spin channel is required")
    all(length(channel) == nR for channel in model.H) || error("H/R image count mismatch")
    all(size(A) == (n, n) for A in model.S) || error("overlap dimensions mismatch")
    all(size(A) == (n, n) for channel in model.H for A in channel) || error("Hamiltonian dimensions mismatch")
    # Reject NaN/Inf at the input boundary; eigensolver failures downstream
    # would be much harder to diagnose.
    all(isfinite, model.lattice) && all(isfinite, model.positions) && all(isfinite, model.reference_positions) ||
        error("geometry contains non-finite values")
    all(all(isfinite, nonzeros(A)) for A in model.S) || error("overlap contains non-finite values")
    all(all(isfinite, nonzeros(A)) for channel in model.H for A in channel) ||
        error("Hamiltonian contains non-finite values")
    !isempty(model.energy_unit) && !isempty(model.length_unit) || error("units must be nonempty")
    # This is the key physical consistency check for Fourier reconstruction.
    _reciprocity_deviation(model.S, model.R) <= atol || error("S(-R) != S(R)†")
    for (spin, channel) in enumerate(model.H)
        _reciprocity_deviation(channel, model.R) <= atol || error("H(-R) != H(R)† for spin $spin")
    end
    return true
end

_write_string(group, name, value) = write(group, name, collect(codeunits(value)))
_read_string(group, name) = String(Vector{UInt8}(read(group[name])))

function _write_sparse_set(parent, name, matrices)
    # Store Julia/SuiteSparse CSC arrays directly. Keeping the sparse structure
    # avoids turning localized-orbital models into large dense HDF5 datasets.
    group = create_group(parent, name)
    for (i, A) in enumerate(matrices)
        item = create_group(group, string(i))
        write(item, "colptr", A.colptr)
        write(item, "rowval", A.rowval)
        write(item, "nzval", A.nzval)
    end
end

function _read_sparse_set(parent, name, n, nR)
    # Matrix dimensions come from sum(norb); CSC itself stores only column
    # pointers, row indices, and nonzero values.
    group = parent[name]
    [SparseMatrixCSC(n, n, Vector{Int}(read(group[string(i)]["colptr"])),
        Vector{Int}(read(group[string(i)]["rowval"])),
        Vector{Float64}(read(group[string(i)]["nzval"]))) for i in 1:nR]
end

"""
    write_model(path, model)

Write the canonical, code-independent HDF5 representation documented in
`docs/hdf5-schema.md`. The model is validated before the file is replaced.
"""
function write_model(path::AbstractString, model::RealSpaceModel)
    validate_model(model)
    h5open(path, "w") do file
        # The hierarchy separates physical data from format metadata so future
        # converters do not need to reproduce any code-specific file layout.
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

Read the canonical HDF5 format into a `RealSpaceModel`. All solvers consume
this type only; native electronic-structure output is handled by converters.
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
        # The sparse blocks do not repeat their dimensions or image count;
        # these are derived from basis and translation metadata.
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
