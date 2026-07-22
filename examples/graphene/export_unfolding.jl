using Unfolding
using Random

# Este script mostra o caminho recomendado para o usuário final: uma
# chamada a unfold_bandstructure no lugar do laço manual (interpolar o
# caminho, mapear k->K, resolver bandas, montar o AtomBasis, chamar
# unfold_supercell) que versões anteriores deste exemplo escreviam à mão.
# Só depende de Unfolding -- nenhuma biblioteca de plotagem. A plotagem em
# si fica em examples/graphene/plotting/, que lê o arquivo HDF5 gerado aqui.

root = @__DIR__
models = joinpath(root, "models")
out = joinpath(root, "output")
mkpath(out)

pc = read_model(joinpath(models, "graphene_primitive.h5"))
sc = read_model(joinpath(models, "graphene_2x2_defect.h5"))

path = [[0.0, 0.0, 0.0], [2/3, 1/3, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]
tick_labels = ["Γ", "K", "M", "Γ"]

result = unfold_bandstructure(pc.lattice, sc, path, 61;
    tick_labels=tick_labels, rng=MersenneTwister(2026))

# Bandas exatas da célula primitiva, no mesmo caminho já resolvido acima
# (result.kpoints_frac), só para comparação visual -- não fazem parte do
# unfolding em si.
kpc = [result.kpoints_frac[:, i] for i in axes(result.kpoints_frac, 2)]
pcbands = solve_bands(pc, kpc)
reference_energies = reduce(hcat, pcbands.energies)

path_out = joinpath(out, "graphene_unfolding.h5")
write_unfolded_hdf5(path_out, result; reference_energies=reference_energies)
println("Wrote ", path_out)
