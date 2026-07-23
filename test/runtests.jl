using Test
using LinearAlgebra
using Random
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
                println(io," CELL| Volume [angstrom^3]:                                  128.000000")
                println(io," CELL| Vector a [angstrom]:       2.000     0.000     0.000   |a| =     2.000000")
                println(io," CELL| Vector b [angstrom]:       0.000     8.000     0.000   |b| =     8.000000")
                println(io," CELL| Vector c [angstrom]:       0.000     0.000     8.000   |c| =     8.000000")
                println(io," CELL| Angle (b,c), alpha [degree]:                              90.000000")
                println(io," CELL| Angle (a,c), beta  [degree]:                              90.000000")
                println(io," CELL| Angle (a,b), gamma [degree]:                              90.000000")
                println(io," ATOMIC KIND INFORMATION")
                println(io,"  1. Atomic kind: H                                  Number of atoms:       2")
                println(io,"      Orbital Basis Set                     TEST-BASIS")
                println(io,"        Number of spherical basis functions:              1")
                println(io," MODULE QUICKSTEP: ATOMIC COORDINATES IN ANGSTROM")
                println(io,"    Atom Kind Element             X             Y             Z      Z(eff)      Mass")
                println(io,"      1    1 H   1      0.000000      0.000000      0.000000   1.0000   1.0000")
                println(io,"      2    1 H   1      1.000000      0.000000      0.000000   1.0000   1.0000")
                println(io)
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
            # Layout normal do CP2K: KS e S compartilham o mesmo prefixo.
            for i in 1:3
                cp(joinpath(dir,"ham-KS_SPIN_1_R_$(i)-1_0.csr"),
                   joinpath(dir,"calc-KS_SPIN_1_R_$(i)-1_0.csr"))
                cp(joinpath(dir,"ovl-S_SPIN_1_R_$(i)-1_0.csr"),
                   joinpath(dir,"calc-S_SPIN_1_R_$(i)-1_0.csr"))
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

            # Recommended two-name API: standard output + common CSR prefix.
            imported_two_names=read_cp2k_csr(output,joinpath(dir,"calc"))
            @test imported_two_names.lattice ≈ lattice
            @test imported_two_names.positions ≈ positions
            @test imported_two_names.atomic_numbers == [1,1]
            @test imported_two_names.norb == [1,1]
            @test hamiltonian_at_k(imported_two_names,k) ≈ hamiltonian_at_k(imported,k)

            h5_two_names=joinpath(dir,"two_names.h5")
            @test convert_cp2k_to_hdf5(h5_two_names,output,joinpath(dir,"calc")) isa RealSpaceModel
            @test read_model(h5_two_names).R == R

            # High-level real-case API: geometry comes from a CP2K cube and
            # AO ownership from MOLog, so the caller supplies only files.
            cube=joinpath(dir,"geometry.cube")
            open(cube,"w") do io
                println(io,"-Quickstep-")
                println(io," synthetic test")
                println(io," 2 0.0 0.0 0.0")
                println(io," 2 1.8897261246257702 0.0 0.0")
                println(io," 2 0.0 7.558904498503081 0.0")
                println(io," 2 0.0 0.0 7.558904498503081")
                println(io," 1 0.0 0.0 0.0 0.0")
                println(io," 1 0.0 1.8897261246257702 0.0 0.0")
            end
            molog=joinpath(dir,"MOLog")
            open(molog,"w") do io
                println(io," MO|    1     1 H  1s        1.0")
                println(io," MO|    2     2 H  1s        1.0")
                # Repetition in another MO column block must be harmless.
                println(io," MO|    1     1 H  1s        0.0")
                println(io," MO|    2     2 H  1s        0.0")
            end
            files=CP2KFiles(output,joinpath(dir,"calc"),cube,molog)
            imported_auto=read_cp2k_csr(files)
            @test imported_auto.lattice ≈ lattice
            @test imported_auto.positions ≈ positions
            @test imported_auto.atomic_numbers == [1,1]
            @test imported_auto.norb == [1,1]
            @test hamiltonian_at_k(imported_auto,k) ≈ hamiltonian_at_k(imported,k)

            h5_auto=joinpath(dir,"converted_auto.h5")
            @test convert_cp2k_to_hdf5(h5_auto,files) isa RealSpaceModel
            @test read_model(h5_auto).norb == [1,1]

            path = [[0.0,0.0,0.0], [0.1,0.0,0.0]]
            identity_M = Matrix{Int}(I,3,3)
            result_auto = unfold_bandstructure(identity_M,files,path,2;
                rng=MersenneTwister(11))
            @test result_auto isa UnfoldedBandStructure
            @test size(result_auto.energies) == (2,2)
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

    @testset "unfolded HDF5 round trip" begin
        # write_unfolded_hdf5/read_unfolded_hdf5 replace the old plain-text
        # CSV writer; this checks that the binary format preserves data
        # exactly, including the optional /reference group and the
        # UnfoldedBandStructure-based convenience method.
        kfrac = [0.0 0.5 1.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
        distance = [0.0, 0.5, 1.0]
        energies = [[-1.0, 1.0], [-0.8, 0.6], [-0.5, 0.5]]
        weights = [[1.0, 0.0], [0.7, 0.3], [0.2, 0.8]]
        reference_energies = [-9.0 -8.0 -7.0; 9.0 8.0 7.0]
        mktempdir() do dir
            path = joinpath(dir, "unfolded.h5")
            write_unfolded_hdf5(path, kfrac, distance, energies, weights;
                ticks=[0.0, 1.0], tick_labels=["Γ", "M"], energy_unit="eV",
                reference_energies=reference_energies)
            data = read_unfolded_hdf5(path)
            @test data.kpoints_frac == kfrac
            @test data.distance == distance
            @test data.ticks == [0.0, 1.0]
            @test data.tick_labels == ["Γ", "M"]
            @test data.energies == reduce(hcat, energies)
            @test data.weights == reduce(hcat, weights)
            @test data.energy_unit == "eV"
            @test data.reference_energies == reference_energies

            # write_unfolded_hdf5(path, ::UnfoldedBandStructure) must produce
            # the same file as the low-level call above.
            result = UnfoldedBandStructure(kfrac, distance, [0.0, 1.0], ["Γ", "M"],
                reduce(hcat, energies), reduce(hcat, weights), "eV")
            path2 = joinpath(dir, "unfolded_struct.h5")
            write_unfolded_hdf5(path2, result; reference_energies=reference_energies)
            data2 = read_unfolded_hdf5(path2)
            @test data2.energies == data.energies
            @test data2.weights == data.weights
            @test data2.reference_energies == reference_energies

            # reference_energies is optional; absence must round-trip to nothing.
            path3 = joinpath(dir, "unfolded_noref.h5")
            write_unfolded_hdf5(path3, kfrac, distance, energies, weights)
            @test read_unfolded_hdf5(path3).reference_energies === nothing
        end
    end

    @testset "unfold_bandstructure high-level API" begin
        # unfold_bandstructure must reproduce exactly the manual loop used in
        # "2x2 pristine unfolding" below (interpolate_kpath + solve_bands +
        # AtomBasis + unfold_supercell).
        #
        # unfold_bandstructure keeps only the single unfolded k-group nearest
        # the requested primitive point, out of the det(M)=4 groups a pristine
        # 2x2 supercell K-point folds into. For a pristine supercell that one
        # group receives exactly nbasis(pc) full-weight SC bands (one copy of
        # the primitive spectrum), so summing weights over SC bands at a given
        # path point equals nbasis(pc), not 1 -- the "sum to 1" rule
        # (check_sum_rule_over_k, exercised below) is over all det(M) groups
        # for a fixed SC band, not over bands for a fixed group.
        pc = graphene_primitive_model(); sc = graphene_supercell_model()
        path = [[0.07, 0.11, 0.0], [0.21, 0.09, 0.0], [0.31, 0.27, 0.0]]
        result = unfold_bandstructure(pc.lattice, sc, path, 2; rng=MersenneTwister(7))

        @test size(result.kpoints_frac, 2) == length(result.distance) == length(path)
        @test maximum(abs.(sum(result.weights, dims=1) .- nbasis(pc))) < 1e-10

        ab = AtomBasis(sc)  # AtomBasis(model) convenience constructor
        transform = round.(Int, pc.lattice \ sc.lattice)
        k = path[1]
        K = mod.(transform' * k .+ 0.5, 1.0) .- 0.5
        bands = solve_bands(sc, [K])
        W, _, _ = unfold_supercell(pc.lattice, ab, sc.lattice, K, bands.overlaps[1], bands.coefficients[1];
            rng=MersenneTwister(7))
        key = collect(keys(W))[argmin([norm(periodic_frac_distance(q, k)) for q in keys(W)])]
        @test result.energies[:, 1] ≈ bands.energies[1]
        @test result.weights[:, 1] ≈ W[key]

        serial = unfold_bandstructure(pc.lattice, sc, path, 2;
            rng=MersenneTwister(7), parallel=false)
        threaded = unfold_bandstructure(pc.lattice, sc, path, 2;
            rng=MersenneTwister(7), parallel=true)
        @test threaded.kpoints_frac == serial.kpoints_frac
        @test threaded.distance == serial.distance
        @test threaded.energies ≈ serial.energies
        @test threaded.weights ≈ serial.weights atol=1e-10

        Kpoints = [[0.07, 0.11, 0.0], [0.21, 0.09, 0.0], [0.31, 0.27, 0.0]]
        serial_bands = solve_bands(sc, Kpoints; parallel=false)
        threaded_bands = solve_bands(sc, Kpoints; parallel=true)
        @test threaded_bands.kpoints_frac == serial_bands.kpoints_frac
        @test threaded_bands.energies ≈ serial_bands.energies
        @test threaded_bands.overlaps ≈ serial_bands.overlaps

        progress_output = IOBuffer()
        progress_result = unfold_bandstructure(pc.lattice, sc, path, 2;
            rng=MersenneTwister(7), parallel=true, progress=true,
            progress_io=progress_output)
        progress_text = String(take!(progress_output))
        @test occursin("Bandas", progress_text)
        @test occursin("Unfolding", progress_text)
        @test count(contains("100%"), split(progress_text, '\r')) == 2
        @test progress_result.energies ≈ serial.energies

        # unfold_bandstructure(M, sc, ...) must reproduce exactly the
        # pc_lattice-based call: M = round.(Int, pc.lattice \ sc.lattice) is
        # exactly diagm([2, 2, 1]) for this 2x2 graphene supercell.
        M = round.(Int, pc.lattice \ sc.lattice)
        result_M = unfold_bandstructure(M, sc, path, 2; rng=MersenneTwister(7))
        @test result_M.energies ≈ result.energies
        @test result_M.weights ≈ result.weights
        @test result_M.kpoints_frac == result.kpoints_frac

        @test_throws ErrorException unfold_bandstructure(zeros(Int, 3, 3), sc, path, 2)
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
