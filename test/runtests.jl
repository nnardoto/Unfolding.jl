using Test
using LinearAlgebra
using SparseArrays
using Unfolding

include(joinpath(@__DIR__,"..","examples","graphene","graphene_models.jl"))

@testset "Unfolding.jl" begin
    @testset "canonical HDF5 schema" begin
        # A round trip must preserve both metadata and the Fourier-reconstructed
        # Hamiltonian, not merely the shapes of the stored datasets.
        model=graphene_primitive_model()
        mktempdir() do dir
            path=joinpath(dir,"model.h5")
            write_model(path,model)
            restored=read_model(path)
            @test restored.lattice==model.lattice
            @test restored.R==model.R
            @test restored.atomic_numbers==[6,6]
            @test model_summary(restored)==model_summary(model)
            @test hamiltonian_at_k(restored,[.17,.22,0])≈hamiltonian_at_k(model,[.17,.22,0])
        end
    end

    @testset "CP2K CSR converter" begin
        # Synthetic three-image output reproduces CP2K's log and filename
        # conventions while keeping the expected H(k), S(k) analytically clear.
        mktempdir() do dir
            R=[0 -1 1; 0 0 0; 0 0 0]
            H0=sparse([-1.0 0.2; 0.2 0.8])
            Hp=sparse([0.1 -0.03; 0.05 0.02]); Hm=sparse(Matrix(Hp'))
            S0=sparse([1.0 0.04; 0.04 1.0])
            Sp=sparse([0.01 0.002; -0.003 0.005]); Sm=sparse(Matrix(Sp'))
            output=joinpath(dir,"cp2k.out")
            open(output,"w") do io
                println(io," *** SCF run converged in 3 steps ***")
                for label in ("KS","S")
                    println(io," $label CSR write|   3 periodic images")
                    println(io,"      Number    X      Y      Z")
                    println(io,"         1      0      0      0")
                    println(io,"         2     -1      0      0")
                    println(io,"         3      1      0      0")
                end
            end
            function write_csr(path,A)
                open(path,"w") do io
                    for col in axes(A,2), ptr in nzrange(A,col)
                        println(io,A.rowval[ptr]," ",col," ",A.nzval[ptr])
                    end
                end
            end
            for (i,A) in enumerate((H0,Hm,Hp))
                write_csr(joinpath(dir,"ham-KS_SPIN_1_R_$(i)-1_0.csr"),A)
            end
            for (i,A) in enumerate((S0,Sm,Sp))
                write_csr(joinpath(dir,"ovl-S_SPIN_1_R_$(i)-1_0.csr"),A)
            end
            lattice=diagm([2.0,8.0,8.0]); positions=[0.0 1.0; 0.0 0.0; 0.0 0.0]
            h5=joinpath(dir,"converted.h5")
            imported=convert_cp2k_to_hdf5(h5,output,joinpath(dir,"ham"),joinpath(dir,"ovl");
                lattice=lattice,positions=positions,atomic_numbers=[1,1],norb=[1,1])
            restored=read_model(h5); k=[.23,0.0,0.0]
            @test imported.R==R==restored.R
            @test hamiltonian_at_k(restored,k)≈fourier_matrix([H0,Hm,Hp],R,k)
            @test overlap_at_k(restored,k)≈fourier_matrix([S0,Sm,Sp],R,k)
            @test restored.energy_unit=="hartree"
            @test restored.source=="CP2K"
        end
    end

    @testset "graphene primitive Dirac cone" begin
        # Nearest-neighbor graphene provides two useful exact references:
        # zero gap at K and energies ±3|t| at Gamma.
        model=graphene_primitive_model()
        candidates=[[2/3,1/3,0.0],[1/3,2/3,0.0],[1/3,1/3,0.0]]
        gaps=[abs(diff(solve_bands(model,[k]).energies[1])[1]) for k in candidates]
        @test minimum(gaps)<1e-10
        Γbands=solve_bands(model,[[0.0,0.0,0.0]])
        @test Γbands.energies[1]≈[-8.1,8.1] atol=1e-10
        @test norm(Γbands.coefficients[1]'*Γbands.overlaps[1]*Γbands.coefficients[1]-I,Inf)<1e-12
    end

    @testset "2x2 pristine unfolding" begin
        # This is the strongest regression test for the projector: a perfect
        # supercell must obey the sum rule and recover every primitive energy
        # with unit spectral weight at the corresponding unfolded k point.
        pc=graphene_primitive_model(); sc=graphene_supercell_model()
        transform=round.(Int,pc.lattice\sc.lattice)
        ab=AtomBasis(sc.reference_positions,sc.norb)
        for k in ([.07,.11,0.0],[.21,.09,0.0],[.31,.27,0.0])
            K=mod.(transform'*k .+0.5,1.0).-0.5
            bands=solve_bands(sc,[K])
            W,_,_=unfold_supercell(pc.lattice,ab,sc.lattice,K,bands.overlaps[1],bands.coefficients[1])
            @test check_sum_rule_over_k(W)<1e-10
            key=collect(keys(W))[argmin([norm(periodic_frac_distance(q,k)) for q in keys(W)])]
            primitive=solve_bands(pc,[k]).energies[1]
            for energy in primitive
                candidates=findall(>(.99),W[key])
                @test minimum(abs.(bands.energies[1][candidates].-energy))<1e-9
            end
        end
    end
end
