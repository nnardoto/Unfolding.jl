using Unfolding
using LinearAlgebra
using Plots
using Random

root=@__DIR__
models=joinpath(root,"models")
pc=read_model(joinpath(models,"graphene_primitive.h5"))
sc=read_model(joinpath(models,"graphene_2x2_defect.h5"))

Γ=[0.0,0.0,0.0]; K=[2/3,1/3,0.0]; M=[0.5,0.0,0.0]
kpc,distance,ticks=interpolate_kpath([Γ,K,M,Γ],61)
pcbands=solve_bands(pc,kpc)

# With lattice vectors in columns, A_sc = A_pc*M. Fractional reciprocal
# coordinates are column vectors here, hence K_sc = M'*k_pc (paper Eq. 7).
transform=round.(Int,pc.lattice\sc.lattice)
# The projector uses the ideal topology, while H contains the symmetry-breaking
# onsite perturbation. This distinction becomes essential for relaxed cells.
ab=AtomBasis(sc.reference_positions,sc.norb)
all_energies=Vector{Float64}[]; all_weights=Vector{Float64}[]
for k in kpc
    Ksc=mod.(transform'*k .+0.5,1.0).-0.5
    bands=solve_bands(sc,[Ksc])
    W,_,_=unfold_supercell(pc.lattice,ab,sc.lattice,Ksc,bands.overlaps[1],bands.coefficients[1];
        rng=MersenneTwister(2026))
    # One SC K point unfolds to det(M) primitive k points. Retain the member
    # that lies on the requested primitive path.
    target=argmin([norm(periodic_frac_distance(key,k)) for key in keys(W)])
    key=collect(keys(W))[target]
    push!(all_energies,bands.energies[1]); push!(all_weights,W[key])
end

p=plot(size=(1050,680),dpi=180,framestyle=:box,xlabel="Caminho k",ylabel="Energia (eV)",
    title="Grafeno 2×2 com defeito: bandas desdobradas",xticks=(ticks,["Γ","K","M","Γ"]),
    xlims=(first(distance),last(distance)),ylims=(-8.5,8.5),grid=:y,gridalpha=0.18,legend=:topright)
x=Float64[]; y=Float64[]; w=Float64[]
for i in eachindex(distance), band in eachindex(all_energies[i])
    push!(x,distance[i]); push!(y,all_energies[i][band]); push!(w,all_weights[i][band])
end
# Avoid drawing numerically zero projectors while retaining weak defect-induced
# character. Marker area and color both encode the same unfolding weight.
visible=findall(>(1e-4),w)
scatter!(p,x[visible],y[visible],marker_z=w[visible],c=:inferno,clims=(0,1),colorbar=true,
    colorbar_title="Peso",ms=2 .+ 6sqrt.(w[visible]),markerstrokewidth=0,alpha=.82,
    label="Supercélula desdobrada")
for band in eachindex(pcbands.energies[1])
    plot!(p,distance,[E[band] for E in pcbands.energies],color=:black,lw=1.5,ls=:dash,
        label=band==1 ? "Célula primitiva" : "")
end
vline!(p,ticks,color=:gray65,lw=.8,label="")

out=joinpath(root,"output"); mkpath(out)
savefig(p,joinpath(out,"graphene_unfolding.png"))
savefig(p,joinpath(out,"graphene_unfolding.pdf"))
println("Wrote ",joinpath(out,"graphene_unfolding.png"))
