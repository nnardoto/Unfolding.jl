using Unfolding
include("graphene_models.jl")

root=@__DIR__
models=joinpath(root,"models")
mkpath(models)

# These files exercise the same public HDF5 writer that external converters
# use. The pristine model checks the exact 0/1-weight limit; the onsite model
# produces fractional spectral weights.
write_model(joinpath(models,"graphene_primitive.h5"),graphene_primitive_model())
write_model(joinpath(models,"graphene_2x2_pristine.h5"),graphene_supercell_model())
write_model(joinpath(models,"graphene_2x2_defect.h5"),graphene_supercell_model(defect_onsite=0.6))

for name in sort(readdir(models))
    model=read_model(joinpath(models,name))
    println(name,": ",model_summary(model))
end
