function HFB_solver(Params::Parameters)
    # Make single-particle orbitals - NuHamil ordering ...
    Orb = orbitals_make(Params)

    # 1-body kinetic operator ...
    T = T1b(Params,Orb)

    # 2-body NN interaction & Orbitals ...
    @time V_NN, Orb_NN = V2b_read(Params,Orb)

    # 3-body NNN interaction & Orbitals ...
    @time V_NNN, Orb_NNN = V3b_no2b_read(Params,Orb)

    # Solve HFB equations ...
    @time Lambda, SQE, U, V, C, Rho, Kappa, H, Delta, Convergence, Iteration = HFB_solve(Params,Orb,Orb_NN,Orb_NNN,T,V_NN,V_NNN)

    # Calculation of the total HFB mean-field ground-state energy ...
    @time E_HFB = HFB_energy(Params,Rho,Kappa,Orb,Orb_NN,Orb_NNN,T,V_NN,V_NNN)

    # Calculate the total HFB ground-state kinetic energy ...
    @time T_HFB = T1b_energy(Params,Rho,Orb,T)

    # Particle number fluctuation calculation ...
    @time dA = HFB_particle_number_dispersion(Params,Rho,Orb)

    # Calculation summary ...
    @time HFB_summary(Params,E_HFB,T_HFB,Lambda,dA,Convergence,Iteration)

    # Evaluate HFB charge radii & radial densities ...
    Summary_File = "io/" * Params.Calc.Path * "/HFB/HFB_summary.dat"
    Densities_File = "io/" * Params.Calc.Path * "/HFB/densities/HFB_radial_densities.dat"
    @time OBDM_export(Params,Orb,Summary_File,Densities_File,Rho,C)

    # Export of single-quasiparticle energies, amplitudes U & V & also possibly radial densities ...
    @time HFB_summary_SQS(Params,SQE,O1B(C.p' * Rho.p * C.p,C.n' * Rho.n * C.n),Orb)

    if Params.Calc.HFB.BMF == true
        # Allocate H1B ... 1-body HFB Hamiltonian in the quasiparticle basis ...
        @time H_N = HFB_allocate_H1b(Params,SQE)

        # Make density dependent residual 2-body interaction in the LHO basis ...
        @time V_NN = V2b_residual_no2b(Params,Orb,Orb_NN,Orb_NNN,Rho,V_NN,V_NNN)

        # Include the N^2 contribution to the residual 2-body interaction in the LHO basis ... if LNT is enabled ...
        if Params.Calc.HFB.LNT == true && Params.Calc.HFB.LNRes == true
            @time V_NN = HFB_Lipkin_Nogami_V2b_residual_no2b(Params,Orb,Orb_NN,V_NN)
        end

        # Perform transformation of U and V to the canonical basis ...
        U, V = O1B(C.p' * U.p, C.n' * U.n), O1B(C.p' * V.p, C.n' * V.n)

        # Allocate H_NN ... residual interaction 2-body Hamiltonian in the quasiparticle basis ...
        @time H_NN = qpO2b(Params,Orb,Orb_NN,V_NN,C,U,V)

        # Perform final export of HFB solution into binary files ...
        @time HFB_export(Params,Orb,Orb_NN,C,U,V,
                         O1B(C.p' * Rho.p * C.p,C.n' * Rho.n * C.n),
                         O1B(C.p' * Kappa.p * C.p,C.n' * Kappa.n * C.n),H_N,H_NN)

        # Perform the HFB-BMBPT(2) calculation of correlation energy ...
        @time HFB_BMBPT_energy(Params,Orb,Orb_NN,H_N,H_NN)

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

function HFB_solve(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,Orb_NNN::Orb3B,T::O1B,V_NN::O2B,V_NNN::Array{Vector{Vector{Float32}},4})
    # Read calculation parameters ...
    Z_target, N_target = Float64(Params.Calc.Z), Float64(Params.Calc.A - Params.Calc.Z)
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1) * (N_max + 2), 2)
    epsilon = Params.Calc.HFB.Tol
    Pairing = Params.Calc.HFB.Pairing

    # Setup local iteration variables ...
    Iteration, Iteration_max, Convergence, Degeneracy = 0, Params.Calc.HFB.Imax, false, false
    dE, d2E, dZ, dN = 0.1, 0.1, 0.1, 0.1

    # Preallocate arrays ...
        # Matrices for U & V HFB ...
    V = O1B(zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max))
    U = O1B(zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max))
         # Matrices for densities Rho & Kappa ...
    Rho = O1B(zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max))
    Kappa = O1B(zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max))
        # Matrices for the single-particle field H & pairing field Delta ...
    H = O1B(zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max))
    Delta = O1B(zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max))
        # Vectors for single-(quasi)particle energies ...
    SQE = pnVector(zeros(Float64,a_max),zeros(Float64,a_max))
        # Identity matrix ...
    Eye = diagm(ones(Float64,a_max))
        # Arrays for degenerate solutions ...
        E_HFB_1, E_HFB_2 = 0.0, 0.0
        Lambda_1, Lambda_2 = pnFloat(0.0,0.0), pnFloat(0.0,0.0)
        U_1, U_2 = O1B(zeros(Float64,a_max,a_max),zeros(Float64,a_max,a_max)), O1B(zeros(Float64,a_max,a_max),zeros(Float64,a_max,a_max))
        V_1, V_2 = O1B(zeros(Float64,a_max,a_max),zeros(Float64,a_max,a_max)), O1B(zeros(Float64,a_max,a_max),zeros(Float64,a_max,a_max))
        Rho_1, Rho_2 = O1B(zeros(Float64,a_max,a_max),zeros(Float64,a_max,a_max)), O1B(zeros(Float64,a_max,a_max),zeros(Float64,a_max,a_max))
        Kappa_1, Kappa_2 = O1B(zeros(Float64,a_max,a_max),zeros(Float64,a_max,a_max)), O1B(zeros(Float64,a_max,a_max),zeros(Float64,a_max,a_max))
        H_1, H_2 = O1B(zeros(Float64,a_max,a_max),zeros(Float64,a_max,a_max)), O1B(zeros(Float64,a_max,a_max),zeros(Float64,a_max,a_max))
        Delta_1, Delta_2 = O1B(zeros(Float64,a_max,a_max),zeros(Float64,a_max,a_max)), O1B(zeros(Float64,a_max,a_max),zeros(Float64,a_max,a_max))
        SQE_1, SQE_2 = O1B(zeros(Float64,a_max,a_max),zeros(Float64,a_max,a_max)), O1B(zeros(Float64,a_max,a_max),zeros(Float64,a_max,a_max))


    # Setup particle numbers ...
    Z, N = 0.1, 0.1

    # Initial guess on chemical potentials lambda ...
    Lambda = pnFloat(Params.Calc.HFB.pL0, Params.Calc.HFB.nL0)

    # Initial guess on densities Rho & Kappa ...
    Rho, Kappa = HFB_density_operator_initialize(Params,Orb)

    # Initialite the Broyden's method ...
    Broyden, BroyVec = HFB_Broyden_initialize(Params,Rho,Kappa,Orb)

    # Preallocate SQE_old ...
    SQE_old = pnVector(SQE.p,SQE.n)

    # Initialize particle numbers & chemical potentials for secant method ...
    Z_1, Z_2 = 0.0, 0.0
    N_1, N_2 = 0.0, 0.0
    Lambda_1, Lambda_2 = pnFloat(0.75 * Lambda.p, 0.75 * Lambda.n), pnFloat(Lambda.p, Lambda.n)

    # Solve the spherical HFB equations ... by the means of self-consistent iteration ...
    println("\nStarting iteration of HFB equations ...")

    while ((dE > epsilon) || (dZ > epsilon) || (dN > epsilon)) && (Iteration < Iteration_max)
        # Perform several secant iterations for the chemical potentil Lambda ...
        @inbounds for L in 1:Iteration_max
            # Initialization of the secant method ...
            if (L == 1) && (Iteration == 0)
                # Allocate the single-particle fields H_1, H_2 and pairing fields Delta_1, Delta_2 ...
                H_1, Delta_1 = HFB_allocate(Params,Lambda_1,Rho,Kappa,Orb,Orb_NN,Orb_NNN,T,V_NN,V_NNN)
                H_2, Delta_2 = HFB_allocate(Params,Lambda_2,Rho,Kappa,Orb,Orb_NN,Orb_NNN,T,V_NN,V_NNN)

                # Diagonalize the HFB equations ... basis is reordered as needed ...
                if Pairing == "Full"
                    SQE_1, U_1, V_1 = HFB_diagonalize(Params,H_1,Delta_1,Orb)
                    SQE_2, U_2, V_2 = HFB_diagonalize(Params,H_2,Delta_2,Orb)
                elseif Pairing == "MCA"
                     SQE_1, U_1, V_1 = HFB_diagonalize_MCA(Params,H_1,Delta_1,Orb)
                     SQE_2, U_2, V_2 = HFB_diagonalize_MCA(Params,H_2,Delta_2,Orb)
                elseif Pairing == "BCS"
                     SQE_1, U_1, V_1 = HFB_diagonalize_BCS(Params,H_1,Delta_1,Orb)
                     SQE_2, U_2, V_2 = HFB_diagonalize_BCS(Params,H_2,Delta_2,Orb)
                end
                
                # Generate temporary densities Rho & Kappa ...
                Rho_1, Kappa_1 = HFB_density_operator(Params,U_1,V_1,Orb)
                Rho_2, Kappa_2 = HFB_density_operator(Params,U_2,V_2,Orb)

                # Determine new average particle numbers ...
                Z_1, N_1 = HFB_particle_number(Params,Rho_1,Orb)
                Z_2, N_2 = HFB_particle_number(Params,Rho_2,Orb)

                # Update average particle numbers ...
                Z, N = Z_2, N_2

                # Perform secant iteration to determine new optimal value of Lambda ...
                Lambda = HFB_chemical_potential_secant(Params,Lambda_1,pnFloat(Z_target - Z_1, N_target - N_1),Lambda_2,pnFloat(Z_target - Z_2, N_target - N_2))

                # Update chemical potentials ...
                Lambda_1 = pnFloat(Lambda_2.p,Lambda_2.n)
                Lambda_2 = pnFloat(Lambda.p,Lambda.n)

                # Update single-quasiparticle energies ...
                SQE = pnVector(SQE_2.p, SQE_2.n)

                # Update amplitudes U & V ...
                U, V = O1B(U_2.p,U_2.n), O1B(V_2.p,V_2.n)

                # Update fields H & Delta ...
                H, Delta = O1B(H_2.p,H_2.n), O1B(Delta_2.p,Delta_2.n)

            end

            # Allocate the single-particle field H and the pairing field Delta ...
                # Smart re-allocation of H ... only Lambda is tweaked, Delta remains the same ...
            H = O1B(H.p .+ (Lambda_1.p - Lambda.p ) .* Eye, H.n .+ (Lambda_1.n - Lambda.n ) .* Eye)

            # Diagonalize the HFB equations ... basis is reordered as needed ...
            if Pairing == "Full"
                SQE, U, V = HFB_diagonalize(Params,H,Delta,Orb)
            elseif Pairing == "MCA"
                SQE, U, V = HFB_diagonalize_MCA(Params,H,Delta,Orb)
            elseif Pairing == "BCS"
                SQE, U, V = HFB_diagonalize_BCS(Params,H,Delta,Orb)
            end
            
            # Generate temporary densities Rho & Kappa ...
            Rho_Temp, Kappa_Temp = HFB_density_operator(Params,U,V,Orb)

            # Determine new average particle numbers ...
            Z, N = HFB_particle_number(Params,Rho_Temp,Orb)

            # Update average particle numbers ...
            Z_1, N_1 = Z_2, N_2
            Z_2, N_2 = Z, N

            # Determine new chemical potential Lambda ...
            Lambda = HFB_chemical_potential_secant(Params,Lambda_1,pnFloat(Z_target - Z_1, N_target - N_1),Lambda_2,pnFloat(Z_target - Z_2, N_target - N_2))

            # Update chemical potentials ...
            Lambda_1 = pnFloat(Lambda_2.p,Lambda_2.n)
            Lambda_2 = pnFloat(Lambda.p,Lambda.n)

            # Evaluate iteration of the chemical potential Lambda ...
            dZ, dN = abs(Z_target - Z), abs(N_target - N)

            # Check finite differences for particle numbers ....
            if dZ < epsilon && dN < epsilon
                break
            end
        end

        # Generate new densities Rho & Kappa ...
        Rho, Kappa = HFB_density_operator(Params,U,V,Orb)

        # Perform update of densities ...
        Rho, Kappa, BroyVec = HFB_Broyden_update(Params,Iteration,Rho,Kappa,Broyden,BroyVec,Orb)

        # Allocate the single-particle field H and the pairing field Delta ...
        H, Delta = HFB_allocate(Params,Lambda,Rho,Kappa,Orb,Orb_NN,Orb_NNN,T,V_NN,V_NNN)

        # Include the Lipkin-Nogami correction to fields H & Delta ...
        if Params.Calc.HFB.LNT == true && Iteration > 15
            H, Delta = HFB_Lipkin_Nogami(Params,Orb,Orb_NN,Orb_NNN,Rho,Kappa,H,Delta,V_NN,V_NNN)
        end

        # Diagonalize the HFB equations ... basis is reordered as needed ...
        if Pairing == "Full"
            SQE, U, V = HFB_diagonalize(Params,H,Delta,Orb)
        elseif Pairing == "MCA"
            SQE, U, V = HFB_diagonalize_MCA(Params,H,Delta,Orb)
        elseif Pairing == "BCS"
            SQE, U, V = HFB_diagonalize_BCS(Params,H,Delta,Orb)
        end

        # Evaluate particle numbers ...
        Z, N = HFB_particle_number(Params,Rho,Orb)

        # Evaluate iteration of the single-quasiparticle energies ...
        d2E = abs(dE - (sum(abs.(SQE.p .- SQE_old.p )) + sum(abs.(SQE.n .- SQE_old.n))) / Float64(2 * a_max))
        dE = (sum(abs.(SQE.p .- SQE_old.p )) + sum(abs.(SQE.n .- SQE_old.n))) / Float64(2 * a_max)
        dZ, dN = abs(Z_target - Z), abs(N_target - N)

        if Degeneracy == false
            Iteration += 1
            @printf("\tHFB iteration number: %4d   dE = %12.9f MeV,   dZ = %12.9f,   dN = %12.9f\n", Iteration, dE, dZ, dN)
        end

        # Store old values of SQE ...
        SQE_old = pnVector(SQE.p,SQE.n)

        # Resolve degenerate solutions ...
        if Degeneracy == true
            H_2, Delta_2, Lambda_2 = O1B(H.p,H.n), O1B(Delta.p,Delta.n), pnFloat(Lambda.p,Lambda.n)
            if Pairing == "Full"
                SQE_2, U_2, V_2 = HFB_diagonalize(Params,H_2,Delta_2,Orb)
            elseif Pairing == "MCA"
                SQE_2, U_2, V_2 = HFB_diagonalize_MCA(Params,H_2,Delta_2,Orb)
            elseif Pairing == "BCS"
                SQE_2, U_2, V_2 = HFB_diagonalize_BCS(Params,H_2,Delta_2,Orb)
            end
            Rho_2, Kappa_2 = HFB_density_operator(Params,U_2,V_2,Orb)
            Out = "/dev/null"
            if Sys.iswindows()
                Out = "NUL"
            end
            open(Out, "w") do devnull_io
                redirect_stdout(devnull_io) do
                    redirect_stderr(devnull_io) do
                        E_HFB_2 = HFB_energy(Params,Rho_2,Kappa_2,Orb,Orb_NN,Orb_NNN,T,V_NN,V_NNN)
                        return E_HFB_2
                    end
                end
            end
            if sum(E_HFB_1) < sum(E_HFB_2)
                U, V, Lambda = O1B(U_1.p,U_1.n), O1B(V_1.p,V_1.n), pnFloat(Lambda_1.p,Lambda_1.n)
                H, Delta, SQE = O1B(H_1.p,H_1.n), O1B(Delta_1.p,Delta_1.n), pnVector(SQE_1.p,SQE_1.n)
                Rho, Kappa = O1B(Rho_1.p,Rho_1.n), O1B(Kappa_1.p,Kappa_1.n)
            else
                U, V, Lambda = O1B(U_2.p,U_2.n), O1B(V_2.p,V_2.n), pnFloat(Lambda_2.p,Lambda_2.n)
                H, Delta, SQE = O1B(H_2.p,H_2.n), O1B(Delta_2.p,Delta_2.n), pnVector(SQE_2.p,SQE_2.n)
                Rho, Kappa = O1B(Rho_2.p,Rho_2.n), O1B(Kappa_2.p,Kappa_2.n)
            end
            println("\nThe degenerate solution has been chosen ... HFB iteration terminates\n")
            Convergence = true
            break
        end

        # If convergence is reached & terminate the loop ...
        if dE < epsilon && dZ < epsilon && dN < epsilon
            println("\nHFB iteration with residual NN interaction has converged ...")
            Convergence = true
            break
        end

        # Check for degenerate solutions ...
        if d2E < 1e-3 * dE && dZ < 1e-3 && dN < 1e-3 && Iteration > 100
            println("\nHFB iteration stucked at a degenerate solution ... Degenerate solutions will be analyzed ...")
            Degeneracy = true
            H_1, Delta_1, Lambda_1 = O1B(H.p,H.n), O1B(Delta.p,Delta.n), pnFloat(Lambda.p,Lambda.n)
            if Pairing == "Full"
                SQE_1, U_1, V_1 = HFB_diagonalize(Params,H_1,Delta_1,Orb)
            elseif Pairing == "MCA"
                SQE_1, U_1, V_1 = HFB_diagonalize_MCA(Params,H_1,Delta_1,Orb)
            elseif Pairing == "BCS"
                SQE_1, U_1, V_1 = HFB_diagonalize_BCS(Params,H_1,Delta_1,Orb)
            end
            Out = "/dev/null"
            if Sys.iswindows()
                Out = "NUL"
            end
            open(Out, "w") do devnull_io
                redirect_stdout(devnull_io) do
                    redirect_stderr(devnull_io) do
                        E_HFB_1 = HFB_energy(Params,Rho_1,Kappa_1,Orb,Orb_NN,Orb_NNN,T,V_NN,V_NNN)
                        return E_HFB_1
                    end
                end
            end
        end

        # Terminate the loop if maximum number of iterations is reached ...
        if Iteration == Iteration_max
            println("\nMaximum number of HFB iterations reached ... Iteration terminates & convergence was not reached ...\n")
            Convergence = false
            break
        end

    end

    # Construct the canonical basis & evaluate approximate (BCS-like) amplitudes
    # & single-quasiparticle energies u_C, v_C & SQE_C ...
    #   U, V, Rho, Kappa, H, Delta ... remain expressed in the reference LHO basis ...
    if Params.Calc.HFB.Pairing == "BCS" || Params.Calc.HFB.Pairing == "MCA"
        C = HFB_canonical_basis_BCS(Params,H,Orb)
    else
        C = HFB_canonical_basis(Params,Rho,Orb)
    end

    # Final re-ordering of the HFB solution ... ascending in U ...
    SQE, U, V = HFB_orbital_ordering(Params,SQE,U,V,Orb,Final_Ordering=true,C=C)

    return Lambda, SQE, U, V, C, Rho, Kappa, H, Delta, Convergence, Iteration
end

function HFB_diagonalize_BCS(Params::Parameters,H::O1B,Delta::O1B,Orb::Vector{Orb1B})
    # Read parameters ...
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1) * (N_max + 2), 2)

    # Preallocate arrays for solutions ...
    pU, pV, pSQE = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max), zeros(Float64,a_max)
    nU, nV, nSQE = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max), zeros(Float64,a_max)

    # Find the canonical basis by diagonalizing H ...
    pSPE, pC = eigen(Symmetric(H.p),sortby=+)
    nSPE, nC = eigen(Symmetric(H.n),sortby=+)

    # Reorder HF orbitals & SPEs ...
    C, SPE = HF_orbital_ordering(Orb,a_max,O1B(pC,nC),pnVector(pSPE,nSPE))
    pC .= C.p
    nC .= C.n
    pSPE .= SPE.p
    nSPE .= SPE.n

    # Transform Delta to the canonical basis ...
    pDelta = pC' * Delta.p * pC
    nDelta = nC' * Delta.n * nC

    # Remove off-diagonal elements of Delta in the canonical basis ...
    @inbounds Threads.@threads for a in 1:a_max
        @inbounds for b in 1:a_max
            if a != b
                pDelta[a,b] = 0.0
                nDelta[a,b] = 0.0
            end
        end
    end

    # Allocate U and V amplitudes in the canonical basis ...
    @inbounds for a in 1:a_max
        l_a = Orb[a].l
        Phase = Float64((-1)^(l_a))

        # Protons ...
        pE_a, pD_a = pSPE[a], pDelta[a,a]
        pU[a,a] = Phase * sqrt(0.5 * (1.0 + pE_a / sqrt(pE_a^2 + pD_a^2)))
        pV[a,a] = sqrt(0.5 * (1.0 - pE_a / sqrt(pE_a^2 + pD_a^2)))
        pSQE[a] = sqrt(pE_a^2 + pD_a^2)

        # Neutrons ...
        nE_a, nD_a = nSPE[a], nDelta[a,a]
        nU[a,a] = Phase * sqrt(0.5 * (1.0 + nE_a / sqrt(nE_a^2 + nD_a^2)))
        nV[a,a] = sqrt(0.5 * (1.0 - nE_a / sqrt(nE_a^2 + nD_a^2)))
        nSQE[a] = sqrt(nE_a^2 + nD_a^2)
    end

    # Transform U & V back to the reference basis ...
    pU .= pC * pU
    pV .= pC * pV
    nU .= nC * nU
    nV .= nC * nV

    return pnVector(pSQE,nSQE), O1B(pU,nU), O1B(pV,nV)
end

function HFB_diagonalize_MCA(Params::Parameters,H::O1B,Delta::O1B,Orb::Vector{Orb1B})
    # Read & define parameters ...
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1) * (N_max + 2), 2)
    HFB_dim = 2 * a_max
    Tol = Params.Calc.HFB.Tol
    nTol = 1e-14

    # Preallocate arrays for solutions ...
    pU = Matrix{Float64}(undef,a_max,a_max)
    pV = Matrix{Float64}(undef,a_max,a_max)
    pSQE = Vector{Float64}(undef,a_max)
    nU = Matrix{Float64}(undef,a_max,a_max)
    nV = Matrix{Float64}(undef,a_max,a_max)
    nSQE = Vector{Float64}(undef,a_max)

    # Initialite the arrays for HFB eigenvalue system ...
    pHFB = Matrix{Float64}(undef,HFB_dim,HFB_dim)
    nHFB = Matrix{Float64}(undef,HFB_dim,HFB_dim)

    # Allocate and fill the HFB eigenvalue systems explicitly ...
    @inbounds Threads.@threads for a in 1:a_max
        @inbounds for b in 1:a_max
            pH_ab = H.p[a,b]
            pD_ab = Delta.p[a,b]
            nH_ab = H.n[a,b]
            nD_ab = Delta.n[a,b]

            pHFB[a,b] = pH_ab
            pHFB[a,b+a_max] = pD_ab
            pHFB[a+a_max,b] = pD_ab
            pHFB[a+a_max,b+a_max] = -pH_ab

            nHFB[a,b] = nH_ab
            nHFB[a,b+a_max] = nD_ab
            nHFB[a+a_max,b] = nD_ab
            nHFB[a+a_max,b+a_max] = -nH_ab
        end
    end

    # Solve the HFB equations ... (in-place eigensolvers to avoid extra copies)
    pE, pW = eigen!(Symmetric(pHFB),sortby=+)
    nE, nW = eigen!(Symmetric(nHFB),sortby=+)

    # Only positive energy solutions are extracted ...
    @inbounds for a in 1:a_max
        pSQE[a], nSQE[a] = pE[a+a_max], nE[a+a_max]
        @views pU[:,a] .= pW[1:a_max,a+a_max]
        @views pV[:,a] .= pW[a_max+1:2*a_max,a+a_max]
        @views nU[:,a] .= nW[1:a_max,a+a_max]
        @views nV[:,a] .= nW[a_max+1:2*a_max,a+a_max]
    end

    # Perform reordering - to match quantum numbers j & l ...
    SQE, U, V = HFB_orbital_ordering(Params,pnVector(pSQE,nSQE),O1B(pU,nU),O1B(pV,nV),Orb)

    # Extract U & V ...
    pU, pV = U.p, V.p
    nU, nV = U.n, V.n

    # Calculate the normal density operator Rho ...
    pRho = Matrix{Float64}(undef,a_max,a_max)
    nRho = Matrix{Float64}(undef,a_max,a_max)
    @inbounds Threads.@threads for a in 1:a_max
        @inbounds for b in 1:a_max
            if (Orb[a].j == Orb[b].j) && (Orb[a].l == Orb[b].l)
                pRhoSum = 0.0
                nRhoSum = 0.0
                @inbounds for c in 1:a_max
                    if (Orb[a].j == Orb[c].j) && (Orb[a].l == Orb[c].l)
                        pMERho = pV[a,c] * pV[b,c]
                        nMERho = nV[a,c] * nV[b,c]
                        pRhoSum = pRhoSum + pMERho
                        nRhoSum = nRhoSum + nMERho
                    end
                end
                pRho[a,b] = pRhoSum
                nRho[a,b] = nRhoSum
            else
                pRho[a,b] = 0.0
                nRho[a,b] = 0.0
            end
        end
    end

    # Symmetrize the density matrix Rho ...
    pRho .= 0.5 .* (pRho .+ pRho')
    nRho .= 0.5 .* (nRho .+ nRho')

    # Regularize the density matrix Rho ...
    @inbounds for a in 1:a_max
        pRho[a,a] += 1e-11 * Float64(a_max - a + 1)
        nRho[a,a] += 1e-11 * Float64(a_max - a + 1)
    end

    # Determine the canonical basis & occupation numbers by diagonalizing Rho ...
    pOcc, pC = eigen(Symmetric(pRho),sortby=-)
    nOcc, nC = eigen(Symmetric(nRho),sortby=-)

    # Reorder the canonical basis & occupation numbers ...
    pC, nC, pOcc, nOcc = HFB_canonical_orbital_ordering(Params,pnVector(pOcc,nOcc),O1B(pC,nC),Orb)

    # Clean numerical noise in the occupation numbers ...
    @inbounds for a in 1:a_max
        po, no = pOcc[a], nOcc[a]
        # Case of protons ...
        if po < Tol
            pOcc[a] = 0.0
        elseif (1.0 - po) < Tol
            pOcc[a] = 1.0
        end
        # Case of neutrons ...
        if no < Tol
            nOcc[a] = 0.0
        elseif (1.0 - no) < Tol
            nOcc[a] = 1.0
        end
    end

    # Transform Delta to the canonical basis ...
    pDelta = pC' * Delta.p * pC
    nDelta = nC' * Delta.n * nC

    # Remove off-diagonal elements of Delta in the canonical basis ...
    @inbounds for a in 1:a_max
        @inbounds for b in 1:a_max
            if a != b
                pDelta[a,b] = 0.0
                nDelta[a,b] = 0.0
            end
        end
    end

    # Transform Delta back to the reference LHO basis ...
    pDelta .= pC * pDelta * pC'
    nDelta .= nC * nDelta * nC'

    # Allocate and fill the HFB eigenvalue systems explicitly ...
    @inbounds Threads.@threads for a in 1:a_max
        @inbounds for b in 1:a_max
            pH_ab = H.p[a,b]
            pD_ab = Delta.p[a,b]
            nH_ab = H.n[a,b]
            nD_ab = Delta.n[a,b]

            pHFB[a,b] = pH_ab
            pHFB[a,b+a_max] = pD_ab
            pHFB[a+a_max,b] = pD_ab
            pHFB[a+a_max,b+a_max] = -pH_ab

            nHFB[a,b] = nH_ab
            nHFB[a,b+a_max] = nD_ab
            nHFB[a+a_max,b] = nD_ab
            nHFB[a+a_max,b+a_max] = -nH_ab
        end
    end

    # Solve the HFB equations ... (in-place eigensolvers to avoid extra copies)
    pE, pW = eigen!(Symmetric(pHFB),sortby=+)
    nE, nW = eigen!(Symmetric(nHFB),sortby=+)

    # Only positive energy solutions are extracted ...
    @inbounds for a in 1:a_max
        pSQE[a], nSQE[a] = pE[a+a_max], nE[a+a_max]
        @views pU[:,a] .= pW[1:a_max,a+a_max]
        @views pV[:,a] .= pW[a_max+1:2*a_max,a+a_max]
        @views nU[:,a] .= nW[1:a_max,a+a_max]
        @views nV[:,a] .= nW[a_max+1:2*a_max,a+a_max]
    end

    # Remove numerical noise ...
    @inbounds Threads.@threads for a in 1:a_max
        @inbounds for b in 1:a_max
            pu, pv = pU[a,b], pV[a,b]
            nu, nv = nU[a,b], nV[a,b]
            if abs(pu) < nTol
                pU[a,b] = 0.0
            end
            if abs(pv) < nTol
                pV[a,b] = 0.0
            end
            if abs(nu) < nTol
                nU[a,b] = 0.0
            end
            if abs(nv) < nTol
                nV[a,b] = 0.0
            end
        end
    end

    # Perform reordering - to match quantum numbers j & l ...
    SQE, U, V = HFB_orbital_ordering(Params,pnVector(pSQE,nSQE),O1B(pU,nU),O1B(pV,nV),Orb)

    # Extract U & V ...
    pU .= U.p
    pV .= V.p
    nU .= U.n
    nV .= V.n

    # Calculate the normal density operator Rho ...
    pRho = Matrix{Float64}(undef,a_max,a_max)
    nRho = Matrix{Float64}(undef,a_max,a_max)
    @inbounds Threads.@threads for a in 1:a_max
        @inbounds for b in 1:a_max
            if (Orb[a].j == Orb[b].j) && (Orb[a].l == Orb[b].l)
                pRhoSum = 0.0
                nRhoSum = 0.0
                @inbounds for c in 1:a_max
                    if (Orb[a].j == Orb[c].j) && (Orb[a].l == Orb[c].l)
                        pMERho = pV[a,c] * pV[b,c]
                        nMERho = nV[a,c] * nV[b,c]
                        pRhoSum = pRhoSum + pMERho
                        nRhoSum = nRhoSum + nMERho
                    end
                end
                pRho[a,b] = pRhoSum
                nRho[a,b] = nRhoSum
            else
                pRho[a,b] = 0.0
                nRho[a,b] = 0.0
            end
        end
    end

    # Symmetrize the density matrix Rho ...
    pRho .= 0.5 .* (pRho .+ pRho')
    nRho .= 0.5 .* (nRho .+ nRho')

    # Regularize the density matrix Rho ...
    @inbounds for a in 1:a_max
        pRho[a,a] += 1e-11 * Float64(a_max - a + 1)
        nRho[a,a] += 1e-11 * Float64(a_max - a + 1)
    end

    # Determine the canonical basis & occupation numbers by diagonalizing Rho ...
    pOcc, pC = eigen(Symmetric(pRho),sortby=-)
    nOcc, nC = eigen(Symmetric(nRho),sortby=-)

    # Reorder the canonical basis & occupation numbers ...
    pC, nC, pOcc, nOcc = HFB_canonical_orbital_ordering(Params,pnVector(pOcc,nOcc),O1B(pC,nC),Orb)

    # Clean numerical noise in the occupation numbers ...
    @inbounds for a in 1:a_max
        po, no = pOcc[a], nOcc[a]
        # Case of protons ...
        if po < Tol
            pOcc[a] = 0.0
        elseif (1.0 - po) < Tol
            pOcc[a] = 1.0
        end
        # Case of neutrons ...
        if no < Tol
            nOcc[a] = 0.0
        elseif (1.0 - no) < Tol
            nOcc[a] = 1.0
        end
    end

    # Initialize U & V in the canonical BMZT decomposed basis ...
    pU_BMZ = zeros(Float64,a_max,a_max)
    pV_BMZ = zeros(Float64,a_max,a_max)
    nU_BMZ = zeros(Float64,a_max,a_max)
    nV_BMZ = zeros(Float64,a_max,a_max)

    # Calculate U & V amplitudes in the BMZT decomposed basis ...
    @inbounds for a in 1:a_max
        Phase = Float64((-1)^(Orb[a].l))
        pv2, nv2 = pOcc[a], nOcc[a]

        pu = Phase * sqrt(max(0.0, 1.0 - pv2))
        pv = sqrt(min(1.0,pv2))
        nu = Phase * sqrt(max(0.0, 1.0 - nv2))
        nv = sqrt(min(1.0,nv2))

        pU_BMZ[a,a] = pu
        pV_BMZ[a,a] = pv
        nU_BMZ[a,a] = nu
        nV_BMZ[a,a] = nv
    end

    # Precalculate the matrix products C' * U & C' * V ...
    pCU, pCV = pC' * pU, pC' * pV
    nCU, nCV = nC' * nU, nC' * nV

    # Initialite quasiparticle transformation matrices D as identity matrices...
    pD = Matrix{Float64}(I,a_max,a_max)
    nD = Matrix{Float64}(I,a_max,a_max)

    # Calculate the quasiparticle transformation matrices D ...
    # Note we use the bigger of U & V amplitudes in the BMZT decomposed
    # basis to avoid numerical instabilities ... division by numerical 0 ...
    #
    # Note that D is actually D^T ...
    @inbounds Threads.@threads for a in 1:a_max
        l_a, j_a = Orb[a].l, Orb[a].j
        @inbounds for b in 1:a_max
            l_b, j_b = Orb[b].l, Orb[b].j
            if l_a == l_b && j_a == j_b
                pu, pv = pU_BMZ[a,a], pV_BMZ[a,a]
                nu, nv = nU_BMZ[a,a], nV_BMZ[a,a]

                # Case of protons ...
                if abs(pu) > abs(pv)
                    pD[a,b] = pCU[a,b] / pu
                else
                    pD[a,b] = pCV[a,b] / pv
                end

                # Case of neutrons ...
                if abs(nu) > abs(nv)
                    nD[a,b] = nCU[a,b] / nu
                else
                    nD[a,b] = nCV[a,b] / nv
                end
            end
        end
    end

    # Now transform U & V from the BMZT decomposed basis to the LHO basis ...
    pU .= pC * pU_BMZ * pD
    pV .= pC * pV_BMZ * pD

    nU .= nC * nU_BMZ * nD
    nV .= nC * nV_BMZ * nD

    # Remove numerical noise ...
    @inbounds Threads.@threads for a in 1:a_max
        @inbounds for b in 1:a_max
            pu, pv = pU[a,b], pV[a,b]
            nu, nv = nU[a,b], nV[a,b]
            if abs(pu) < nTol
                pU[a,b] = 0.0
            end
            if abs(pv) < nTol
                pV[a,b] = 0.0
            end
            if abs(nu) < nTol
                nU[a,b] = 0.0
            end
            if abs(nv) < nTol
                nV[a,b] = 0.0
            end
        end
    end

    # Allocate U & V ...
    U = O1B(pU,nU)
    V = O1B(pV,nV)

    return SQE, U, V
end

function HFB_diagonalize(Params::Parameters,H::O1B,Delta::O1B,Orb::Vector{Orb1B})
    # Read & define parameters ...
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1) * (N_max + 2), 2)
    HFB_dim = 2 * a_max
    Tol = 1e-11
    nTol = 1e-13

    # Preallocate arrays for solutions ...
    pU = Matrix{Float64}(undef,a_max,a_max)
    pV = Matrix{Float64}(undef,a_max,a_max)
    pSQE = Vector{Float64}(undef,a_max)
    nU = Matrix{Float64}(undef,a_max,a_max)
    nV = Matrix{Float64}(undef,a_max,a_max)
    nSQE = Vector{Float64}(undef,a_max)

    # Initialite the arrays for HFB eigenvalue system ...
    pHFB = Matrix{Float64}(undef,HFB_dim,HFB_dim)
    nHFB = Matrix{Float64}(undef,HFB_dim,HFB_dim)

    # Allocate and fill the HFB eigenvalue systems explicitly ...
    @inbounds Threads.@threads for a in 1:a_max
        @inbounds for b in 1:a_max
            pH_ab = H.p[a,b]
            pD_ab = Delta.p[a,b]
            nH_ab = H.n[a,b]
            nD_ab = Delta.n[a,b]

            pHFB[a,b] = pH_ab
            pHFB[a,b+a_max] = pD_ab
            pHFB[a+a_max,b] = pD_ab
            pHFB[a+a_max,b+a_max] = -pH_ab

            nHFB[a,b] = nH_ab
            nHFB[a,b+a_max] = nD_ab
            nHFB[a+a_max,b] = nD_ab
            nHFB[a+a_max,b+a_max] = -nH_ab
        end
    end

    # Solve the HFB equations ... (in-place eigensolvers to avoid extra copies)
    pE, pW = eigen!(Symmetric(pHFB),sortby=+)
    nE, nW = eigen!(Symmetric(nHFB),sortby=+)

    # Only positive energy solutions are extracted ...
    @inbounds for a in 1:a_max
        pSQE[a], nSQE[a] = pE[a+a_max], nE[a+a_max]
        @views pU[:,a] .= pW[1:a_max,a+a_max]
        @views pV[:,a] .= pW[a_max+1:2*a_max,a+a_max]
        @views nU[:,a] .= nW[1:a_max,a+a_max]
        @views nV[:,a] .= nW[a_max+1:2*a_max,a+a_max]
    end

    # Remove numerical noise ...
    @inbounds Threads.@threads for a in 1:a_max
        @inbounds for b in 1:a_max
            pu, pv = pU[a,b], pV[a,b]
            nu, nv = nU[a,b], nV[a,b]
            if abs(pu) < nTol
                pU[a,b] = 0.0
            end
            if abs(pv) < nTol
                pV[a,b] = 0.0
            end
            if abs(nu) < nTol
                nU[a,b] = 0.0
            end
            if abs(nv) < nTol
                nV[a,b] = 0.0
            end
        end
    end

    # Perform reordering - to match quantum numbers j & l ...
    SQE, U, V = HFB_orbital_ordering(Params,pnVector(pSQE,nSQE),O1B(pU,nU),O1B(pV,nV),Orb)

    # Extract U & V ...
    pU .= U.p
    pV .= V.p
    nU .= U.n
    nV .= V.n

    # Calculate the normal density operator Rho ...
    pRho = Matrix{Float64}(undef,a_max,a_max)
    nRho = Matrix{Float64}(undef,a_max,a_max)
    @inbounds Threads.@threads for a in 1:a_max
        @inbounds for b in 1:a_max
            if (Orb[a].j == Orb[b].j) && (Orb[a].l == Orb[b].l)
                pRhoSum = 0.0
                nRhoSum = 0.0
                @inbounds for c in 1:a_max
                    if (Orb[a].j == Orb[c].j) && (Orb[a].l == Orb[c].l)
                        pMERho = pV[a,c] * pV[b,c]
                        nMERho = nV[a,c] * nV[b,c]
                        pRhoSum = pRhoSum + pMERho
                        nRhoSum = nRhoSum + nMERho
                    end
                end
                pRho[a,b] = pRhoSum
                nRho[a,b] = nRhoSum
            else
                pRho[a,b] = 0.0
                nRho[a,b] = 0.0
            end
        end
    end

    # Symmetrize the density matrix Rho ...
    pRho .= 0.5 .* (pRho .+ pRho')
    nRho .= 0.5 .* (nRho .+ nRho')

    # Clean numerical noise & regularize the density matrix Rho ...
    @inbounds for a in 1:a_max
        l_a, j_a = Orb[a].l, Orb[a].j
        @inbounds for b in 1:a_max
            l_b, j_b = Orb[b].l, Orb[b].j
            if l_a != l_b || j_a != j_b
                pRho[a,b] = 0.0
                nRho[a,b] = 0.0
            end
        end
        R = 10.0 * Float64(l_a*(N_max+1) + div(j_a+1,2))
        pRho[a,a] += R
        nRho[a,a] += R
    end

    # Determine the canonical basis & occupation numbers by diagonalizing Rho ...
    pOcc, pC = eigen(Symmetric(pRho),sortby=-)
    nOcc, nC = eigen(Symmetric(nRho),sortby=-)

    #pOcc, pC, nOcc, nC = HFB_canonical_basis_diagonalize(Params,O1B(pRho,nRho),Orb)

    # Reorder the canonical basis & occupation numbers ...
    pC, nC, pOcc, nOcc = HFB_canonical_orbital_ordering(Params,pnVector(pOcc,nOcc),O1B(pC,nC),Orb)

    # Adjust the occupation numbers ... regularization factor is removed ...

    @inbounds for a in 1:a_max
        l_a, j_a = Orb[a].l, Orb[a].j
        R = 10.0 * Float64(l_a*(N_max+1) + div(j_a+1,2))
        pOcc[a] -= R
        nOcc[a] -= R
    end


    # Remove numerical noise ...
    @inbounds Threads.@threads for a in 1:a_max
        po, no = pOcc[a], nOcc[a]
        # Case of proton occupation numbers ...
        if po < Tol
            pOcc[a] = 0.0
        elseif (1.0 - po) < Tol
            pOcc[a] = 1.0
        end

        # Case of neutron occupation numbers ...
        if no < Tol
            nOcc[a] = 0.0
        elseif (1.0 - no) < Tol
            nOcc[a] = 1.0
        end
    end

    # Initialize U & V in the canonical BMZT decomposed basis ...
    pU_BMZ = zeros(Float64,a_max,a_max)
    pV_BMZ = zeros(Float64,a_max,a_max)
    nU_BMZ = zeros(Float64,a_max,a_max)
    nV_BMZ = zeros(Float64,a_max,a_max)

    # Calculate U & V amplitudes in the BMZT decomposed basis ...
    @inbounds Threads.@threads for a in 1:a_max
        Phase = Float64((-1)^(Orb[a].l))
        pv2, nv2 = pOcc[a], nOcc[a]

        pv = sqrt(min(1.0,pv2))
        nv = sqrt(min(1.0,nv2))

        pu = Phase * sqrt(abs(1.0 - pv^2))
        nu = Phase * sqrt(abs(1.0 - nv^2))

        pU_BMZ[a,a] = pu
        pV_BMZ[a,a] = pv
        nU_BMZ[a,a] = nu
        nV_BMZ[a,a] = nv
    end

    # Precalculate the matrix products C' * U & C' * V ...
    pCU, pCV = pC' * pU, pC' * pV
    nCU, nCV = nC' * nU, nC' * nV

    # Initialite quasiparticle transformation matrices D as identity matrices...
    pD = Matrix{Float64}(I,a_max,a_max)
    nD = Matrix{Float64}(I,a_max,a_max)

    # Calculate the quasiparticle transformation matrices D ...
    # Note we use the bigger of U & V amplitudes in the BMZT decomposed
    # basis to avoid numerical instabilities ... division by numerical 0 ...
    #
    # Note that D is actually D^T ...
    @inbounds Threads.@threads for a in 1:a_max
        l_a, j_a = Orb[a].l, Orb[a].j
        @inbounds for b in 1:a_max
            l_b, j_b = Orb[b].l, Orb[b].j
            if l_a == l_b && j_a == j_b
                pu, pv = pU_BMZ[a,a], pV_BMZ[a,a]
                nu, nv = nU_BMZ[a,a], nV_BMZ[a,a]

                # Case of protons ...
                if abs(pu) > abs(pv)
                    pD[a,b] = pCU[a,b] / pu
                else
                    pD[a,b] = pCV[a,b] / pv
                end

                # Case of neutrons ...
                if abs(nu) > abs(nv)
                    nD[a,b] = nCU[a,b] / nu
                else
                    nD[a,b] = nCV[a,b] / nv
                end
            end
        end
    end

    # Now transform U & V from the BMZT decomposed basis to the LHO basis ...
    pU .= pC * pU_BMZ * pD
    pV .= pC * pV_BMZ * pD

    nU .= nC * nU_BMZ * nD
    nV .= nC * nV_BMZ * nD

    # Remove numerical noise ...
    @inbounds Threads.@threads for a in 1:a_max
        @inbounds for b in 1:a_max
            pu, pv = pU[a,b], pV[a,b]
            nu, nv = nU[a,b], nV[a,b]
            if abs(pu) < nTol
                pU[a,b] = 0.0
            end
            if abs(pv) < nTol
                pV[a,b] = 0.0
            end
            if abs(nu) < nTol
                nU[a,b] = 0.0
            end
            if abs(nv) < nTol
                nV[a,b] = 0.0
            end
        end
    end

    # Allocate U & V ...
    U = O1B(pU,nU)
    V = O1B(pV,nV)
 
    return SQE, U, V
end