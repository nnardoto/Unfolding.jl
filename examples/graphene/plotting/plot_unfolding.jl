using HDF5
using Plots

# Este script roda em seu próprio ambiente (Project.toml nesta pasta), com
# Plots.jl como dependência. O pacote Unfolding não é necessário aqui: o
# arquivo lido abaixo é HDF5 puro, gravado por
# examples/graphene/export_unfolding.jl, e documentado em
# docs/unfolded-hdf5-schema.md. Qualquer outra ferramenta (Matplotlib,
# gnuplot, ...) pode ler o mesmo arquivo sem depender deste script.

function read_unfolded(path)
    h5open(path, "r") do file
        (distance=read(file["path/distance"]),
         ticks=read(file["path/ticks"]),
         tick_labels=Vector{String}(read(file["path/tick_labels"])),
         energies=read(file["data/energies"]),     # nbands x nk
         weights=read(file["data/weights"]),       # nbands x nk
         reference_energies=haskey(file, "reference") ?
             read(file["reference/energies"]) : nothing)  # nbands_pc x nk, opcional
    end
end

root = @__DIR__
out = joinpath(root, "..", "output")

data = read_unfolded(joinpath(out, "graphene_unfolding.h5"))

p = plot(size=(1050, 680), dpi=180, framestyle=:box, xlabel="Caminho k", ylabel="Energia (eV)",
    title="Grafeno 2×2 com defeito: bandas desdobradas",
    xticks=(data.ticks, data.tick_labels),
    xlims=(first(data.distance), last(data.distance)), ylims=(-8.5, 8.5),
    grid=:y, gridalpha=0.18, legend=:topright)

nbands, nk = size(data.energies)
x = Float64[]; y = Float64[]; w = Float64[]
for i in 1:nk, band in 1:nbands
    push!(x, data.distance[i]); push!(y, data.energies[band, i]); push!(w, data.weights[band, i])
end
# Evita desenhar projeções numericamente nulas, mantendo o caráter fraco
# induzido pelo defeito. Área e cor do marcador codificam o mesmo peso.
visible = findall(>(1e-4), w)
scatter!(p, x[visible], y[visible], marker_z=w[visible], c=:inferno, clims=(0, 1), colorbar=true,
    colorbar_title="Peso", ms=2 .+ 6sqrt.(w[visible]), markerstrokewidth=0, alpha=.82,
    label="Supercélula desdobrada")

if data.reference_energies !== nothing
    nbands_pc = size(data.reference_energies, 1)
    for band in 1:nbands_pc
        plot!(p, data.distance, data.reference_energies[band, :], color=:black, lw=1.5, ls=:dash,
            label=band == 1 ? "Célula primitiva" : "")
    end
end
vline!(p, data.ticks, color=:gray65, lw=.8, label="")

savefig(p, joinpath(out, "graphene_unfolding.png"))
savefig(p, joinpath(out, "graphene_unfolding.pdf"))
println("Wrote ", joinpath(out, "graphene_unfolding.png"))
