function BCS_solver(Params::Parameters,Params_Ref::Parameters)
    # Make single-particle orbitals - NuHamil ordering ...
    Orb = orbitals_make(Params)

    # 1-body kinetic operator ...
    T = T1b(Params,Orb)

    # 2-body NN interaction & Orbitals ...
        # Temporarily, CMS is set to reference core nucleus ... no longer true
        # Currently, the CMS is set to the target nucleus ...
    @time V_NN, Orb_NN = V2b_read(Params,Orb)

    # 3-body NNN interaction & Orbitals ...
    @time V_NNN, Orb_NNN = V3b_no2b_read(Params,Orb)

    # Solve HF-BCS equations ...
    @time Lambda, SPE, SQE, C, U, V, Rho, Kappa, Delta, Convergence, Iteration = HF_BCS_solve(Params,Params_Ref,Orb,Orb_NN,Orb_NNN,T,V_NN,V_NNN)

    # Calculate the total mean-field + BCS pairing ground-state energy ...
        # Temporarily, CMS is set to reference core nucleus ...
    @time E_MF, E_BCS = BCS_energy(Params,Rho,Kappa,Orb,Orb_NN,Orb_NNN,T,V_NN,V_NNN)

    # Calculate the total BCS ground-state kinetic energy ...
    @time T_BCS = T1b_energy(Params,Rho,Orb,T)

    # Particle number fluctuation calculation ...
    @time dA = BCS_particle_number_dispersion(Params,U,V,Orb)

    # Calculation summary ...
    @time BCS_summary(Params,Params_Ref,E_MF,E_BCS,T_BCS,Lambda,dA,Convergence,Iteration)

    # Evaluate BCS charge radii & radial densities ...
    Summary_File = "io/" * Params.Calc.Path * "/BCS/BCS_summary.dat"
    Densities_File = "io/" * Params.Calc.Path * "/BCS/densities/BCS_radial_densities.dat"
    @time OBDM_export(Params_Ref,Orb,Summary_File,Densities_File,Rho,C)

    # Export of single-quasiparticle energies, amplitudes U & V & also possibly radial densities ...
    @time BCS_summary_SQS(Params_Ref,SPE,SQE,U,V)

    if Params.Calc.BCS.BMF == true
        # Allocate H1B ... 1-body BCS Hamiltonian in the canonical & quasiparticle basis ...
        @time H_N = BCS_allocate_H1b(Params,SQE)

        # Make density dependent residual 2-body interaction in the LHO basis ...
        @time V_NN = V2b_residual_no2b(Params,Orb,Orb_NN,Orb_NNN,Rho,V_NN,V_NNN)

        # Perform transformation of U and V to the canonical basis ...
        U_m, V_m = O1B(diagm(U.p),diagm(U.n)), O1B(diagm(V.p),diagm(V.n))

        # Allocate H_NN ... residual interaction 2-body Hamiltonian in the quasiparticle basis ...
        @time H_NN = qpO2b(Params,Orb,Orb_NN,V_NN,C,U_m,V_m)

        # Perform final export of BCS solution into binary files ...
        @time BCS_export(Params,Orb,Orb_NN,C,U_m,V_m,H_N,H_NN)

        # Perform the HF-BCS-BMBPT(2) calculation of correlation energy ...
        @time HF_BCS_BMBPT_energy(Params,Orb,Orb_NN,H_N,H_NN)

        # Deallocate 2-body quasiparticle Hamiltonian H_NN ...
        H_NN = nothing
    end

    # Deallocate V_NN 2-body & V_NNN 3-body interaction ...
    V_NN = nothing
    V_NNN = nothing

    # Perform the Garbage Collection ...
    GC.gc()

    return
end

function HF_BCS_solve(Params::Parameters,Params_Ref::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,Orb_NNN::Orb3B,T::O1B,V_NN::O2B,V_NNN::Array{Vector{Vector{Float32}},4})
    # Read calculation parameters ...
    Z_target, N_target = Params.Calc.Z, Params.Calc.A - Params.Calc.Z
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1) * (N_max + 2), 2)
    epsilon = Params.Calc.BCS.Tol

    # Setup local iteration variables ...
    Iteration, Iteration_max, Convergence = 0, Params.Calc.BCS.Imax, false
    
    # Preallocate some arrays ...
        # BCS amplitudes U & V vectors ...
    V = pnVector(zeros(Float64,a_max), zeros(Float64,a_max))
    U = pnVector(zeros(Float64,a_max), zeros(Float64,a_max))
        # BCS SQEs ...
    SQE = pnVector(zeros(Float64,a_max), zeros(Float64,a_max))
        # The pairing gap vector ...
    Delta = pnVector(Params.Calc.BCS.pD0 .* ones(Float64,a_max), Params.Calc.BCS.nD0 .* ones(Float64,a_max))

    # Solve the HF-BCS approximation  ...
    println("\nStarting iteration of HF-BCS with NO2B NN+NNN interaction ...\n")

    # Define local function for reference HF calculation ...
    function HF_BCS_HF_solve()
        # Setup single-particle orbitals for HF - NuHamil ordering ...
        Orb_HF = orbitals_make(Params_Ref)

        # Read 2-body NN interaction with CM correction for reference nucleus ...
            # Note that further self-consistent HF-BCS iterations are considered
            # with respect to the target nucleus ... the target values of A ...
        #V_NN_Ref, Orb_NN_Ref = V2b_read(Params_Ref,Orb_HF)
        V_NN_Ref, Orb_NN_Ref = V2b_read(Params,Orb_HF)

        # Call the HF Solver for reference closed-shell nucleus ...
        println("\nSolving the HF equations for reference closed-shell system ...")
        println("Reference nucleus:     A = " * string(Params_Ref.Calc.A) * ",     Z = " * string(Params_Ref.Calc.Z) * "\n")
        #@time h, C, Rho, Iteration_HF = HF_solve(Params_Ref,Orb_HF,Orb_NN_Ref,Orb_NNN,T,V_NN_Ref,V_NNN)
        @time h, C, Rho, Iteration_HF = HF_solve(Params,Orb_HF,Orb_NN,Orb_NNN,T,V_NN,V_NNN)

        # Extract the HF SPEs ...
        SPE = pnVector(diag(h.p),diag(h.n))

        # Calculate the HF mean-field energy ... for comparison
        println("\nCalculating the HF mean-field ground-state energy ... sanity check ...")
        #@time E_HF = HF_energy(Params_Ref,Rho,Orb_HF,Orb_NN_Ref,Orb_NNN,T,V_NN_Ref,V_NNN)
        @time E_HF = HF_energy(Params,Rho,Orb_HF,Orb_NN,Orb_NNN,T,V_NN,V_NNN)

        # Make residual density-dependent NN interaction in canonical HF basis... J = 0 - s-wave only ...
        println("\nMaking density-depenent residual NN interaction ... s-wave channel (J = 0) ...")
        #@time V_NN_Res, Orb_NN_Res = BCS_V2b(Params_Ref,Orb,Orb_NN_Ref,Orb_NNN,V_NN_Ref,V_NNN,C,Rho)
        @time V_NN_Res, Orb_NN_Res = BCS_V2b(Params,Orb,Orb_NN,Orb_NNN,V_NN,V_NNN,C,Rho)

        # Drop V_NN_Ref and force Garbage Collection ...
        V_NN_Ref = nothing
        GC.gc()

        # Determine the initial value of chemical potential from the HF calculation ...
        Lambda = HF_BCS_initialize_chemical_potential(Params_Ref,SPE,Orb_HF)

        return SPE, C, Rho, h, Lambda, V_NN_Res, Orb_NN_Res
    end

    # Solve the HF equations for the reference closed-shell nucleus ...
    @time SPE, C, Rho, h, Lambda, V_NN_Res, Orb_NN_Res = HF_BCS_HF_solve()

    # Solve BCS equations ...
    println("\nInitializing the HF-BCS approximation ...")

    # Setup particle numbers Z & N ... exact from the HF iteration ...
    Z, N = Z_target, N_target

    # Setup particle number differences dZ & dN ...
    dZ, dN = abs(Z_target - Z) + 0.1, abs(N_target - N) + 0.1

    # Iteratively solve the BCS equations ...
    println("\nStarting iteration of BCS equations ...\n")
    while ((abs(dZ) > epsilon) || (abs(dN) > epsilon)) && (Iteration < Iteration_max)

        # Initial BCS iteration ... based on the initial pairing gap Delta ...
        if Iteration == 0
            # Evaluate the BCS SQEs ...
            SQE = BCS_allocate_SQE(Params,SPE,Lambda,Delta)

            # Evaluate the BCS amplitudes U & V ...
            U, V = BCS_allocate_amplitudes(Params,Lambda,SPE,Delta,Orb)

            # Evaluate <Z> & <N> from BCS amplitudes V ...
            Z, N = BCS_particle_number(Params,V,Orb)

            # Evaluate the particle number deviations dZ & dN ...
            dZ, dN = Z_target - Z, N_target - N

            # Evaluate the chemical potential Lambda ...
            Lambda = BCS_chemical_potential(Params,SPE,Delta,Lambda,pnFloat(dZ,dN),Orb)

        # Regular BCS iteration ... starting with gap equation & input values of V & U amplitudes ...
        else
            # Calculate the pairing gap Delta ...
            Delta = BCS_allocate_Delta(Params,U,V,Orb,Orb_NN_Res,V_NN_Res)

            # Evaluate the BCS amplitudes U & V ...
            U, V = BCS_allocate_amplitudes(Params,Lambda,SPE,Delta,Orb)

            # Evaluate <Z> & <N> from BCS amplitudes V ...
            Z, N = BCS_particle_number(Params,V,Orb)

            # Evaluate the chemical potential Lambda ...
            Lambda = BCS_chemical_potential(Params,SPE,Delta,Lambda,pnFloat(dZ,dN),Orb)
        
        end

        # Evaluate the BCS iteration ...
        Iteration += 1
        dZ, dN = Z_target - Z, N_target - N
        println("\tBCS iteration number:   " * string(Iteration) * "   dZ = " * string(round(dZ, sigdigits=8))* "   &   dN = " * string(round(dN, sigdigits=8)))
        println("\t\tCurrent particle numbers     ...     Z = " * string(Z) * ", N = " * string(N))
        println("\t\tCurrent chemical potentials        ...     pLambda = " * string(round(Lambda.p, digits = 6)) * " MeV, nLambda = " * string(round(Lambda.n, digits = 6)) * " MeV")

        # If BCS convergence is reached, terminate the loop ...
        if (abs(dZ) < epsilon) && (abs(dN) < epsilon)
            if Params.Calc.BCS.ScBCS == false
                Convergence = true
                println("\nBCS iteration with residual NN interaction has converged ...")
            else
                println("\nInitial ScBCS iteration with residual NN interaction has converged ...")
            end
            break
        end

        # If BCS convergence is not reached, terminate the loop ...
        if Iteration == Iteration_max
            Convergence = false
            println("\nBCS iteration with residual NN interaction has terminated ...")
            println("\n\t (!!!) CONVERGENCE WAS NOT REACHED (!!!) \n")
            break
        end

    end

    # Start self-consistent iteration of BCS equations ...
    if Params.Calc.BCS.ScBCS == true
        println("\nStarting self-consistent iteration of BCS equations ...\n")

        # Initialize mean-field iteration variables ...
        Iteration, dE = 0, 1.0

        # Initialize vector for old SPEs ...
        SPE_Old = pnVector(zeros(Float64,a_max), zeros(Float64,a_max))

        # Start the HF-BCS self-consistent loop iteration ...
        while (dE > epsilon) && (Iteration < Iteration_max)

            # Allocate density operators Rho & Kappa ...
            Rho, Kappa = BCS_density_operator(Params,U,V)

            # Transform Rho & Kappa to the reference LHO basis ...
            Rho = O1B(C.p * Rho.p * C.p', C.n * Rho.n * C.n')
            Kappa = O1B(C.p * Kappa.p * C.p', C.n * Kappa.n * C.n')

            # Evaluate new HF mean-field Hamiltonian h ...
            h = HF_BCS_allocate(Params,Rho,Kappa,Orb,Orb_NN,Orb_NNN,T,V_NN,V_NNN)

            # Diagonalize the HF Hamiltonians ...
            pSPE, pC = eigen(Symmetric(h.p), sortby=+)
            nSPE, nC = eigen(Symmetric(h.n), sortby=+)

            # Reorder HF orbitals & SPEs ...
            #C, SPE = HF_orbital_ordering(Orb,a_max,O1B(pC,nC),pnVector(pSPE,nSPE))
            Orb, SPE, C = HF_orbitals_make(Params,pnVector(pSPE,nSPE),O1B(pC,nC),Orb)

            # Make residual density-dependent NN interaction in canonical HF basis... J = 0 - s-wave only ...
                # To avoid spam, Terminal Output is supressed for this call ...
            Out = "/dev/null"
            if Sys.iswindows()
                Out = "NUL"
            end
            open(Out, "w") do devnull_io
                redirect_stdout(devnull_io) do
                    redirect_stderr(devnull_io) do
                        V_NN_Res, Orb_NN_Res = BCS_V2b(Params,Orb,Orb_NN,Orb_NNN,V_NN,V_NNN,C,Rho)
                        return V_NN_Res, Orb_NN_Res
                    end
                end
            end

            # Iterate the BCS equations ...
            Iteration_BCS = 0
            dZ, dN = 0.1, 0.1

            println("\tStarting iteration of BCS equations ...\n")
            while ((dZ > epsilon) || (dN > epsilon)) && (Iteration_BCS < Iteration_max)
                
                # Initial BCS iteration ... twofold for 2 initial guesses on Chemical Potentials ...
                if Iteration_BCS == 0
                    # Evaluate the BCS amplitudes U & V ...
                    U, V = BCS_allocate_amplitudes(Params,Lambda,SPE,Delta,Orb)

                    # Evaluate <Z> & <N> from BCS amplitudes V ...
                    Z, N = BCS_particle_number(Params,V,Orb)

                    # Evaluate the particle number deviations dZ & dN ...
                    dZ, dN = abs(Z_target - Z), abs(N_target - N)

                    # Evaluate the chemical potential Lambda ...
                    Lambda = BCS_chemical_potential(Params,SPE,Delta,Lambda,pnFloat(dZ,dN),Orb)

                # Regular BCS iteration ... starting with gap equation & input values of V & U amplitudes ...
                elseif Iteration_BCS > 0
                    # Calculate the pairing gap Delta ...
                    Delta = BCS_allocate_Delta(Params,U,V,Orb,Orb_NN_Res,V_NN_Res)

                    # Evaluate the BCS amplitudes U & V ...
                    U, V = BCS_allocate_amplitudes(Params,Lambda,SPE,Delta,Orb)

                    # Evaluate <Z> & <N> from BCS amplitudes V ...
                    Z, N = BCS_particle_number(Params,V,Orb)

                    # Evaluate the chemical potential Lambda ...
                    Lambda = BCS_chemical_potential(Params,SPE,Delta,Lambda,pnFloat(dZ,dN),Orb)

                end

                # Evaluate the BCS iteration ...
                Iteration_BCS += 1
                dZ, dN = abs(Params.Calc.Z - Z), abs(Params.Calc.A - Params.Calc.Z - N)

            end

            # Evaluate mean-field iteration ...
            Iteration += 1
            dE = (sum(abs.(SPE.p .- SPE_Old.p )) + sum(abs.(SPE.n .- SPE_Old.n))) / Float64(2 * a_max)
            println("\tScBCS teration number:   " * string(Iteration) * "   dE = " * string(round(dE, sigdigits=8)) * " MeV" * ",   dZ = " * string(round(dZ, sigdigits=8))* "   &   dN = " * string(round(dN, sigdigits=8)))
            println("\t\tCurrent particle numbers     ...     Z = " * string(Z) * ", N = " * string(N))
            println("\t\tCurrent chemical potentials        ...     pLambda = " * string(round(Lambda.p, digits = 6)) * " MeV, nLambda = " * string(round(Lambda.n, digits = 6)) * " MeV")

            # Save current SPEs ...
            SPE_Old = pnVector(deepcopy(SPE.p),deepcopy(SPE.n))

            # If HF-ScBCS convergence is reached, terminate the loop ...
            if (abs(dZ) < epsilon) && (abs(dN) < epsilon) && dE < epsilon
                Convergence = true
                println("\nScBCS iteration with residual NN interaction has converged ...")
                break
            end

            # If HF-ScBCS convergence is not reached, terminate the loop ...
            if Iteration == Iteration_max
                Convergence = false
                println("\nScBCS iteration with residual NN interaction has terminated ...")
                println("\n\t (!!!) CONVERGENCE WAS NOT REACHED (!!!) \n")
                break
            end
        
        end

    end

    # Determine the BCS SQEs ...
    SQE = BCS_allocate_SQE(Params,SPE,Lambda,Delta)

    # Determine the BCS amplitudes U & V ...
    U, V = BCS_allocate_amplitudes(Params,Lambda,SPE,Delta,Orb)

    # Determine the resulting density operators Rho & Kappa ...
    Rho, Kappa = BCS_density_operator(Params,U,V)

    # Perform transformation of 1-body densities Rho & Kappa to the reference LHO basis ...
    Rho = O1B(C.p * Rho.p * C.p', C.n * Rho.n * C.n')
    Kappa = O1B(C.p * Kappa.p * C.p', C.n * Kappa.n * C.n')

    return Lambda, SPE, SQE, C, U, V, Rho, Kappa, Delta, Convergence, Iteration
end