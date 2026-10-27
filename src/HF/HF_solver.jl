function HF_solver(Params::Parameters)
    # Make single-particle orbitals - NuHamil ordering ...
    @time Orb = orbitals_make(Params)

    # 1-body kinetic operator ...
    @time T = T1b(Params,Orb)

    # 2-body bare NN interaction & Orbitals ...
    @time V_NN, Orb_NN = V2b_read(Params,Orb)

    # 3-body bare NNN interaction & Orbitals ...
    @time V_NNN, Orb_NNN = V3b_no2b_read(Params,Orb)

    # Solve HF equations ...
    @time h_N, C, Rho, Iteration = HF_solve(Params,Orb,Orb_NN,Orb_NNN,T,V_NN,V_NNN)

    # HF energy calculation ...
    @time E_HF = HF_energy(Params,Rho,Orb,Orb_NN,Orb_NNN,T,V_NN,V_NNN)

    # Calculate the total HF kinetic energy ...
    @time T_HF = T1b_energy(Params,Rho,Orb,T)

    # Calculation summary ...
    @time HF_summary(Params,E_HF,T_HF,Iteration)

    # Calculate & export radial HF densities & radii ...
    Summary_File = "io/" * Params.Calc.Path * "/HF/HF_summary.dat"
    Densities_File = "io/" * Params.Calc.Path * "/HF/densities/HF_radial_densities.dat"
    @time OBDM_export(Params,Orb,Summary_File,Densities_File,Rho,C)

    # Single-Particle States export ...
    @time HF_SPS_summary(Params,h_N,Orb)

    # Export of single-particle HF Hamiltonian & HF single-particle states ... in matrix C ...
    @time HF_export(Params,C,O1B(C.p' * Rho.p * C.p, C.n' * Rho.n * C.n),h_N,Orb)

    # Perform Beyond-mean-field HF-MBPT calculations & export the residual 2-body NN interation ...
    if Params.Calc.HF.BMF == true
        # Make density-dependent residual NN interaction ... NO2B approximation ...
        @time V_NN_den = V2b_residual_no2b(Params,Orb,Orb_NN,Orb_NNN,Rho,V_NN,V_NNN)

        # Transform the density-dependent NN interaction to the target HF basis ...
        @time V_NN_res = O2b_transformation(Params,Orb,Orb_NN,V_NN_den,C)

        # Export residual 2-body interaction in the binary format ...
        V_NN_export_path = "io/" * Params.Calc.Path * "/HF/bin/H2b.bin"
        @time O2b_export(Params,Orb_NN,V_NN_res,V_NN_export_path)

        if HF_MBPT_occupation_check(Params,O1B(C.p' * Rho.p * C.p, C.n' * Rho.n * C.n))
            # Perform HF-MBPT calculation ...
            @time HF_MBPT(Params,Orb,Orb_NN,h_N,V_NN_res)

            # If set to true, calculate the correlated ground state OBDM & TBCM from HF-MBPT & replace HF binaries ...
            if Params.Calc.HF.MBPT.Export == true
                @time HF_MBPT_export(Params,Orb,Orb_NN,Orb_NNN,C,T,h_N,V_NN,V_NN_res,V_NNN)
            end
        end

        # Deallocate the residual NN interaction ...
        V_NN_den = nothing
        V_NN_res = nothing

        # Perform the Garbage Collection ...
        GC.gc()
    end

    # Deallocate the NN & NNN interaction ...
    V_NN, V_NNN = nothing, nothing

    # Perform the Garbage Collection ...
    GC.gc()

    return
end

function HF_solve(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,Orb_NNN::Orb3B,T::O1B,V_NN::O2B,V_NNN::Array{Vector{Vector{Float32}},4})
    # Read calculation parameters ...
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1) * (N_max + 2), 2)
    epsilon = Params.Calc.HF.Tol
    iteration_max = Params.Calc.HF.Imax

    # Initialize local iteration parameters ...
    dE, iteration = 1.0, 0

    # Preallocate arrays ...
        # Vectors for single-particle energies ...
    SPE = pnVector(zeros(Float64,a_max),zeros(Float64,a_max))
    SPE_old = pnVector(zeros(Float64,a_max),zeros(Float64,a_max))

        # HF single-particle Hamiltonian matrix ...
    h = O1B(zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max))

        # Initial guess on single-particle states & 1-body density matrix ... LHO orbitals ...
    C = O1B(diagm(ones(Float64,a_max)), diagm(ones(Float64,a_max)))
    Rho = HF_density_operator(C,Orb)

    println("\nInitializing the iteration of HF mean-field equations ...")

    # Iterate spherical HF equations ...
    @time while (iteration < iteration_max) && (dE > epsilon)

        # Perform iteration of HF equations ... fills the HF Hamiltonian ...
        h = HF_allocate(Params,Rho,Orb,Orb_NN,Orb_NNN,T,V_NN,V_NNN)

        # Diagonalize the HF Hamiltonians ...
        pSPE, pC = eigen(Symmetric(h.p),sortby=+)
        nSPE, nC = eigen(Symmetric(h.n),sortby=+)

        # Allocate single-particle energies & HF orbitals ...
        SPE = pnVector(pSPE,nSPE)
        C = O1B(pC,nC)

        Orb, SPE, C = HF_orbitals_make(Params,SPE,C,Orb)

        # Generate new HF density matrix ...
        Rho = HF_density_operator(C,Orb)

        # Check on convergence of HF SPEs ...
        dE = (sum(abs.(SPE.p .- SPE_old.p )) + sum(abs.(SPE.n .- SPE_old.n))) / Float64(2 * a_max)

        SPE_old  = pnVector(deepcopy(SPE.p), deepcopy(SPE.n))
        iteration += 1

        @printf("\tIteration number: %4d   dE = %12.9f MeV\n", iteration, dE)
    end

    println("\nIteration of HF equations with NO2B NN+NNN interaction has finished ...")

    return O1B(diagm(SPE.p),diagm(SPE.n)), C, Rho, iteration
end