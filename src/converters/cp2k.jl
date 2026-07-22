"""
CP2K `KS_CSR_WRITE`/`S_CSR_WRITE` to canonical HDF5 converter.

CP2K writes one text file per real-space periodic image plus a table in its
main output that maps image numbers to integer translations. This file keeps
all knowledge of that naming/log format outside the unfolding core.

The importer expects `REAL_SPACE T`, `BINARY F`, and `UPPER_TRIANGULAR F`.
"""

function _cp2k_float(token)
    # CP2K/Fortran may use either E or D for the decimal exponent.
    value = tryparse(Float64, replace(token, 'D' => 'E', 'd' => 'e'))
    value === nothing && error("invalid CP2K floating-point token '$token'")
    value
end

function _read_cp2k_csr_matrix(path, n)
    # Despite the print-key name, the text output is a coordinate list:
    # one-based `row column value`. Julia's sparse constructor converts it to
    # CSC and also combines repeated entries if CP2K ever emits them.
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
    # A typical output block is:
    #   KS CSR write|  21 periodic images
    #        Number    X    Y    Z
    #           1      0    0    0
    # The last block is selected because a file can contain restart/history
    # output from more than one electronic calculation.
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
        # Requiring consecutive image numbers prevents unrelated four-integer
        # lines later in the CP2K output from being mistaken for this table.
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
    # For prefix `ham`, CP2K produces names such as
    # `ham-KS_SPIN_1_R_3-1_0.csr`. Iteration suffixes may vary, so only the
    # stable prefix, label, spin, and image number are interpreted.
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

Import CP2K real-space matrices into a `RealSpaceModel`. This is the only
CP2K-specific layer; downstream code reads the canonical HDF5 format.

Required keyword arguments describe information not encoded reliably in the
CSR files: lattice, Cartesian positions, atomic numbers, and AO count per atom.
The AO counts and ordering must match CP2K's matrix rows/columns exactly.

For relaxed structures, pass ideal sites separately as `reference_positions`.
`spins` contains the CP2K spin-channel numbers to import.
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
    # Do not archive partial or failed electronic structures. If an abort
    # occurred only before a later successful restart, the final result is OK.
    converged = findlast(contains("SCF run converged"), lines)
    aborted = findlast(contains("[ABORT]"), lines)
    converged === nothing && error("CP2K output has no converged SCF run")
    (aborted === nothing || aborted < converged) ||
        error("CP2K aborted after its last convergence")
    # H(R) and S(R) must be indexed by exactly the same translations for the
    # generalized Fourier-space eigenproblem.
    RH = _cp2k_image_mapping(lines, "KS")
    RS = _cp2k_image_mapping(lines, "S")
    RH == RS || error("KS and S image mappings differ")
    n = sum(norb)
    nR = size(RH, 2)
    # Preserve two nesting levels: H[spin][image]. CP2K's overlap is common to
    # spin channels, so only S_SPIN_1 is read.
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

Import CP2K CSR output, validate the resulting model, and write one canonical
HDF5 file. Returns the in-memory `RealSpaceModel` for immediate inspection.
"""
function convert_cp2k_to_hdf5(h5_path::AbstractString, output, h_prefix, s_prefix; kwargs...)
    model = read_cp2k_csr(output, h_prefix, s_prefix; kwargs...)
    write_model(h5_path, model)
    return model
end
