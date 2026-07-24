using Unfolding
using LinearAlgebra
using Random

const HARTREE_TO_EV = 27.211386245988

run_dir = isempty(ARGS) ? joinpath(@__DIR__, "run_debug") : abspath(ARGS[1])
out_path = joinpath(run_dir, "graphene_debug.out")
csr_prefix = joinpath(run_dir, "graphene_debug")
model_path = joinpath(run_dir, "graphene_debug.h5")
pc_out_path = joinpath(run_dir, "graphene_debug_pc.out")
pc_csr_prefix = joinpath(run_dir, "graphene_debug_pc")
pc_model_path = joinpath(run_dir, "graphene_debug_pc.h5")
unfolded_path = joinpath(run_dir, "graphene_debug_unfolded.h5")
csv_path = joinpath(run_dir, "graphene_debug_unfolded.csv")
reference_csv_path = joinpath(run_dir, "graphene_debug_reference.csv")

isfile(out_path) || error("CP2K output not found: $out_path")
isfile(pc_out_path) || error("primitive-cell CP2K output not found: $pc_out_path")

# PRINT_LEVEL MEDIUM lets the converter recover the cell, atoms, elements,
# and number of spherical AOs directly from the CP2K output.
model = convert_cp2k_to_hdf5(model_path, out_path, csr_prefix)
pc_model = convert_cp2k_to_hdf5(pc_model_path, pc_out_path, pc_csr_prefix)
println("Model: ", model_summary(model))
println("Primitive reference: ", model_summary(pc_model))

# A_sc = A_pc*M: the CP2K cell above is a pristine 2x2 graphene supercell.
M = diagm([2, 2, 1])
path = [
    [0.0, 0.0, 0.0],       # Gamma
    [2 / 3, 1 / 3, 0.0],   # K in this reciprocal-basis convention
    [0.5, 0.0, 0.0],       # M
    [0.0, 0.0, 0.0],       # Gamma (repeated continuity check)
]
labels = ["Γ", "K", "M", "Γ"]
n_per_segment = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 32

result = unfold_bandstructure(M, model, path, n_per_segment;
    tick_labels=labels,
    rng=MersenneTwister(2026),
    progress=true)

# Independent primitive-cell calculation on exactly the same requested path.
pc_bands = solve_bands(pc_model, eachcol(result.kpoints_frac); progress=true)
reference_energies = reduce(hcat, pc_bands.energies)
write_unfolded_hdf5(unfolded_path, result; reference_energies=reference_energies)

# For pristine graphene, one selected unfolded k subspace contains exactly
# nbasis(supercell)/det(M) total spectral weight at every path point.
expected_weight = nbasis(model) / abs(det(M))
weight_sums = vec(sum(result.weights; dims=1))
sum_rule_error = maximum(abs.(weight_sums .- expected_weight))
weight_min, weight_max = extrema(result.weights)
gamma_energy_error = maximum(abs.(result.energies[:, 1] .- result.energies[:, end]))
gamma_weight_error = maximum(abs.(result.weights[:, 1] .- result.weights[:, end]))

sum_rule_error < 1e-8 || error("unfolding sum rule failed: deviation=$sum_rule_error")
weight_min >= -1e-10 || error("negative unfolding weight: $weight_min")
weight_max <= 1 + 1e-10 || error("unfolding weight above one: $weight_max")
gamma_energy_error < 1e-10 || error("repeated Γ energies differ: $gamma_energy_error")
gamma_weight_error < 1e-8 || error("repeated Γ weights differ: $gamma_weight_error")

# Carbon contributes four valence electrons. In a spin-restricted run this
# fills two orbitals per atom. Center the debug CSV at the middle of the two
# frontier eigenvalues at K; this is only a plotting reference.
noccupied = 2 * size(model.positions, 2)
k_index = argmin(abs.(result.distance .- result.ticks[2]))
energy_reference = (result.energies[noccupied, k_index] +
                    result.energies[noccupied + 1, k_index]) / 2
pc_noccupied = 2 * size(pc_model.positions, 2)
pc_energy_reference = (reference_energies[pc_noccupied, k_index] +
                       reference_energies[pc_noccupied + 1, k_index]) / 2
shifted_sc = result.energies .- energy_reference
shifted_pc = reference_energies .- pc_energy_reference
reference_error_ev = HARTREE_TO_EV * maximum(
    minimum(abs.(shifted_sc[:, ik] .- shifted_pc[band, ik]))
    for ik in axes(shifted_pc, 2), band in axes(shifted_pc, 1)
)
reference_error_ev < 0.02 ||
    error("primitive/unfolded band mismatch is too large: $reference_error_ev eV")

# Spectral leakage: weight assigned farther than 20 meV from every
# independently calculated primitive band. A pristine supercell should make
# this negligible; a large value exposes a projector/phase inconsistency even
# when the ordinary weight sum rule still passes.
leakage_tolerance_ev = 0.02
leaked_weight = zeros(size(result.energies, 2))
for ik in axes(result.energies, 2), band in axes(result.energies, 1)
    separation_ev = HARTREE_TO_EV * minimum(
        abs.(shifted_pc[:, ik] .- shifted_sc[band, ik]))
    separation_ev > leakage_tolerance_ev &&
        (leaked_weight[ik] += result.weights[band, ik])
end
maximum_leakage_fraction = maximum(leaked_weight) / expected_weight

open(csv_path, "w") do io
    println(io, "distance,k1,k2,k3,band,energy_ev,weight")
    for ik in axes(result.kpoints_frac, 2), band in axes(result.energies, 1)
        values = (result.distance[ik], result.kpoints_frac[:, ik]..., band,
                  (result.energies[band, ik] - energy_reference) * HARTREE_TO_EV,
                  result.weights[band, ik])
        println(io, join(values, ','))
    end
end

open(reference_csv_path, "w") do io
    println(io, "distance,band,energy_ev")
    for ik in axes(reference_energies, 2), band in axes(reference_energies, 1)
        values = (result.distance[ik], band,
                  (reference_energies[band, ik] - pc_energy_reference) * HARTREE_TO_EV)
        println(io, join(values, ','))
    end
end

println("Diagnostics:")
println("  sum-rule maximum error = ", sum_rule_error)
println("  weight range = [", weight_min, ", ", weight_max, "]")
println("  repeated Γ energy error = ", gamma_energy_error, " hartree")
println("  repeated Γ weight error = ", gamma_weight_error)
println("  primitive/unfolded maximum mismatch = ", reference_error_ev, " eV")
println("  maximum spectral leakage (>20 meV from reference) = ",
        100maximum_leakage_fraction, "%")
maximum_leakage_fraction > 1e-3 &&
    println("  WARNING: pristine-cell spectral leakage should be approximately zero")
println("Unfolded HDF5: ", unfolded_path)
println("Plotting CSV:  ", csv_path)
println("Reference CSV: ", reference_csv_path)
