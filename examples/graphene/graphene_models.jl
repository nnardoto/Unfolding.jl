using Unfolding
using LinearAlgebra
using SparseArrays

function graphene_lattice(; a=2.46, vacuum=15.0)
    # Primitive hexagonal lattice in the xy plane. A finite c vector keeps the
    # example compatible with ordinary 3D periodic electronic-structure codes.
    [a a/2 0.0; 0.0 sqrt(3)*a/2 0.0; 0.0 0.0 vacuum]
end

function graphene_positions(; a=2.46, vacuum=15.0)
    # Two carbon sites. Their separation a/sqrt(3) is the C-C bond length.
    [0.0 0.0; 0.0 a/sqrt(3); vacuum/2 vacuum/2]
end

function _graphene_tb(lattice,positions; hopping=-2.7,onsite=zeros(size(positions,2)),source="graphene TB")
    # One p_z-like orbital per carbon and nearest-neighbor hopping only. The
    # resulting small model tests unfolding algebra without requiring CP2K.
    natom=size(positions,2); bond=1.42
    matrices=Dict{NTuple{3,Int},Matrix{Float64}}()
    # Search the home cell and adjacent in-plane images. This range is enough
    # for nearest neighbors in both the primitive and 2x2 cells.
    for rx in -1:1, ry in -1:1
        R=(rx,ry,0); A=zeros(natom,natom)
        shift=lattice*collect(R)
        for i in 1:natom, j in 1:natom
            d=norm(positions[:,j]+shift-positions[:,i])
            isapprox(d,bond;atol=2e-2) && (A[i,j]=hopping)
        end
        any(x -> !iszero(x),A) && (matrices[R]=A)
    end
    H0=get!(matrices,(0,0,0),zeros(natom,natom))
    H0[diagind(H0)].+=onsite
    # Keep the home image first for readability. The HDF5 schema itself does
    # not require a particular image order because R stores the mapping.
    keys_sorted=sort!(collect(keys(matrices));by=x->(x!=(0,0,0),x...))
    R=hcat(collect.(keys_sorted)...)
    H=[sparse(matrices[key]) for key in keys_sorted]
    # The pedagogical p_z basis is orthonormal: S(0)=I and S(R!=0)=0.
    S=[key==(0,0,0) ? spdiagm(0=>ones(natom)) : spzeros(natom,natom) for key in keys_sorted]
    RealSpaceModel(lattice,positions,positions,fill(6,natom),ones(Int,natom),R,[H],S;
        energy_unit="eV",length_unit="angstrom",source=source)
end

function graphene_primitive_model()
    _graphene_tb(graphene_lattice(),graphene_positions();source="graphene nearest-neighbor TB primitive")
end

function graphene_supercell_model(;repetitions=(2,2),defect_onsite=0.0)
    pc_lattice=graphene_lattice(); base=graphene_positions()
    positions=Vector{Vector{Float64}}()
    for n2 in 0:(repetitions[2]-1), n1 in 0:(repetitions[1]-1), atom in axes(base,2)
        push!(positions,base[:,atom]+pc_lattice*[n1,n2,0])
    end
    X=hcat(positions...)
    M=diagm([repetitions[1],repetitions[2],1])
    # An onsite shift breaks primitive translational symmetry without changing
    # the site/AO topology. It is therefore supported without placeholders.
    onsite=zeros(size(X,2)); onsite[1]=defect_onsite
    label=iszero(defect_onsite) ? "pristine" : "onsite defect $(defect_onsite) eV"
    _graphene_tb(pc_lattice*M,X;onsite=onsite,source="graphene nearest-neighbor TB 2x2 $label")
end
