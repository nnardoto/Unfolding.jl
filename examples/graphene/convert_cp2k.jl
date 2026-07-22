using Unfolding

length(ARGS)==5 || error("usage: convert_cp2k.jl primitive|2x2 CP2K.out H_PREFIX S_PREFIX DEST.h5")
cell,output,hprefix,sprefix,destination=ARGS
include("graphene_models.jl")

if cell=="primitive"
    template=graphene_primitive_model()
elseif cell=="2x2"
    template=graphene_supercell_model()
else
    error("cell must be primitive or 2x2")
end

# DZVP-MOLOPT-SR-GTH carbon: 13 spherical atomic orbitals per atom.
# CP2K orders all AOs of atom 1 first, then atom 2, matching `norb` in the
# canonical schema. Changing the basis requires updating this value.
natom=size(template.positions,2)
mkpath(dirname(abspath(destination)))
model=convert_cp2k_to_hdf5(destination,output,hprefix,sprefix;
    lattice=template.lattice,
    positions=template.positions,
    reference_positions=template.reference_positions,
    atomic_numbers=fill(6,natom),
    norb=fill(13,natom))
println("Wrote ",destination,": ",model_summary(model))
