function RPA_solver(Params::Parameters)
    # Read parameters ...
    IO = "io/" * Params.Calc.Path * "/" * Params.Calc.RPA.Reference

    # Import all data & construct needed structures ...
        # Read reference single-particle orbitals ...
        Orb = orbitals_import(Params,IO * "/bin/orb.bin")
        
        @time C = O1b_import(Params,Orb,IO * "/bin/C.bin")

        # Import reference 1-body density matrix (OBDM) ...
        @time Rho = O1b_import(Params,Orb,IO * "/bin/rho.bin")

        # Import reference 1-body Hamiltonian ...
        @time h_N = O1b_import(Params,Orb,IO * "/bin/H1b.bin")

        # Import reference 2-body interaction ...
        println("\nImporting reference 2-body interaction ...")

        @time V_NN, Orb_NN = O2b_import(Params,Orb,IO * "/bin/H2b.bin",Make_Orb_NN = true)

        # If available, import 2-body correlation function ... (then it is included in A & B)
        Sigma = O2b_initialize_empty()
        TBCF_allocate = false
        if isfile(IO * "/bin/sigma.bin") && (Params.Calc.RPA.Reference_TBCF == true)
            println("\nImporting reference 2-body correlation function ...")
            @time Sigma = O2b_import(Params,Orb,IO * "/bin/sigma.bin")
            TBCF_allocate = true
        end

        # Construct 2-body 1p-1h orbitals ...
        @time Orb_1p1h = orbitals_1p1h_make(Params,Orb;Tol_par = Params.Calc.RPA.Tol_par,Tol_hol = Params.Calc.RPA.Tol_hol)

        # Construct 1-body transition operators ...
        @time TrOp = Tr1b_initialize(Params,Orb,1.0)

        @time TrOp = Tr1b_transformation(Params,Orb,C,TrOp)

    # Solve the RPA equations ...
        # Allocate matrices A & B ...
        @time A, B, N = RPA_allocate(Params,Orb,Orb_1p1h,Orb_NN,h_N,V_NN,Rho,Sigma,TBCF_allocate)

        # Solve the RPA eigenvalue problem ...
        @time E, X, Y, Stability = RPA_diagonalize(Params,Orb,Orb_1p1h,A,B,N,TrOp)

    # Process the RPA solutions ...
        # Calculate RPA 1-Body Density Matrix ...
        @time Rho = RPA_OBDM_oY2(Params,Orb,Orb_1p1h,Y)

        # Calculate RPA 2-body correlation function ...
        if Params.Calc.RPA.TBCF == true
            @time Sigma = RPA_sigma(Params,Orb,Orb_1p1h,Orb_NN,X,Y)
        else
            @time Sigma = O2b_initialize_empty()
        end

        # Evaluate the RPA charge radius chR ...
        chR2 = rho_chR2(Params,Orb,C,Rho)

        # Recompute the 1-body transition operators ...
        @time TrOp = Tr1b_initialize(Params,Orb,chR2)

        @time TrOp = Tr1b_transformation(Params,Orb,C,TrOp)

        # Calculate RPA correlation energy ...
        @time E_corr = RPA_energy(Params,Orb,Orb_1p1h,E,X,Y)

        # Electromagnetic reduced multipole operators ...
        @time rM = RPA_rM(Params,Orb,Orb_1p1h,Orb_NN,TrOp,Rho,Sigma,X,Y)

        # Electromagnetic reduced transition intensities ...
        @time rB = RPA_rB(Params,Orb_1p1h,rM,real.(E))

        # Transform RPA OBDM to the LHO basis ...
        Rho = O1B(C.p * Rho.p * C.p', C.n * Rho.n * C.n')

        # Export of RPA & TDA solutions ...
        @time RPA_export(Params,Orb,Orb_1p1h,Orb_NN,Stability,E_corr,E,X,Y,rB,C,Rho,Sigma,h_N)
        
    return
end

function HF_ScRPA_solver(Params::Parameters)
    # Read parameters ...
    IO = "io/" * Params.Calc.Path * "/" * Params.Calc.RPA.Reference
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1)*(N_max + 2),2)

    epsilon = Params.Calc.RPA.Tol
    Iteration_max = Params.Calc.RPA.Imax

    alpha_TBCF = Params.Calc.RPA.TBCF_alpha     # linear mixing in the TBCF iteration ...
    beta_TBCF = Params.Calc.RPA.TBCF_beta           # length of the ramp (damping) of the 2-body correlation function Sigma in the outer loop ...
    TBCF = Params.Calc.RPA.TBCF             # 2-body correlation function Sigma in the outer loop ...

    # Define local functions for the iterative procedure ...
        # Local function for initial quenching of 2-body correlation function Sigma ...
    function HF_ScRPA_solver_beta(i::Int64)
        beta_i = tanh((Float64(i) / beta_TBCF)^2)
        if beta_i > 0.95
            beta_i = 1.0
        end
        return beta_i
    end
        # Local function for output summary log file when iteration terminates due to convergence issues ...
    function ScRPA_terminate_summary(N_iterations::Int64,E_QBA_list::Vector{Float64},Stability_list::Vector{Bool},beta::Float64)
        # Read parameters ...
        A = Params.Calc.A
        Z = Params.Calc.Z
        hw = Params.Calc.hw
        CMS = Params.Calc.CMS
        RPA_Params = Params.Calc.RPA

        Output_Path = "io/" * Params.Calc.Path * "/" * RPA_Params.Name * "/RPA_summary.dat"

        println("\nWriting HF-ScRPA summary file (non-converged iteration) ...")

        Summary = open(Output_Path, "w")
            println(Summary, "Spherical self-consistent Random-Phase Approximation (HF-ScRPA) review:")
            println(Summary, "\n!!! The HF-ScRPA iteration has NOT converged ... no RPA results were exported !!!")
            println(Summary, "\nNuclide data:    A = " * string(A) * ", Z = " * string(Z))
            println(Summary, "\nCalculation data:    hw = " * string(hw) * " , N_max = " * string(N_max) *
                    ", s.p. J-basis size = " * string(a_max) * ", s.p. M-basis size = " * string(div((N_max+1)*(N_max+2)*(N_max+3),6)))
            println(Summary, "\t\t\tCenter-of-Mass System correction is set to:   CMS = ''" * string(CMS) * "''")
            if RPA_Params.Ortho == true
                println(Summary, "\nOrthogonalization of spurious Goldstone levels (1-) is included ...")
            else
                println(Summary, "\nOrthogonalization of spurious Goldstone levels (1-) is NOT included ...")
            end

            println(Summary, "\nHF-ScRPA parameters:")
            println(Summary, "\tReference = ''" * RPA_Params.Reference * "'',   Reference_TBCF = " * string(RPA_Params.Reference_TBCF) * ",   ScH = " * string(RPA_Params.ScH))
            println(Summary, "\tTol = " * string(RPA_Params.Tol) * ",   Imax = " * string(RPA_Params.Imax) * ",   Tol_hol = " * string(RPA_Params.Tol_hol) * ",   Tol_par = " * string(RPA_Params.Tol_par))
            println(Summary, "\tOBDM_alpha = " * string(RPA_Params.OBDM_alpha) * ",   OBDM_diagonal = " * string(RPA_Params.OBDM_diagonal))
            println(Summary, "\tTBCF = " * string(TBCF) * ",   TBCF_alpha = " * string(alpha_TBCF) * ",   TBCF_beta = " * string(beta_TBCF))

            println(Summary, "\nConvergence information:")
            println(Summary, "\tIterations performed = " * string(N_iterations) * " (Imax = " * string(Iteration_max) * ")")
            if TBCF
                println(Summary, "\tCriterion ... relative change of E_QBA below Tol in two consecutive iterations, with beta = 1 ...")
                println(Summary, "\tLast beta (ramp of Sigma) = " * string(beta))
            else
                println(Summary, "\tCriterion ... relative change of E_QBA below Tol in two consecutive iterations ...")
            end
            println(Summary, "\tUnstable iterations (complex RPA eigenvalues) = " * string(count(==(false),Stability_list)) * " of " * string(length(Stability_list)))

            N_E = length(E_QBA_list)
            if N_E >= 2
                Delta_E = E_QBA_list[N_E] - E_QBA_list[N_E-1]
                println(Summary, "\tLast E0 = " * string(E_QBA_list[N_E]) * " MeV,   last change = " * string(abs(Delta_E)) * " MeV (relative " * string(abs(Delta_E) / max(abs(E_QBA_list[N_E]),1e-12)) * ")")
            end

            println(Summary, "\nLast iterations ... (Iteration, E_QBA [MeV], relative change, stable)")
            @inbounds for i in max(2,N_E-9):N_E
                dE_rel = abs(E_QBA_list[i] - E_QBA_list[i-1]) / max(abs(E_QBA_list[i]),1e-12)
                println(Summary, string(i - 1) * "\t" * string(E_QBA_list[i]) * "\t" * string(dE_rel) * "\t" * string(Stability_list[i-1]))
            end
        close(Summary)

        println("\tHF-ScRPA summary file exported ... ''" * Output_Path * "''")

        return
    end

    # Import all data & construct needed structures ...
        # Import reference single-particle orbitals ...
        Orb_ref = orbitals_import(Params,IO * "/bin/orb.bin")

        # Read the bare nuclear Hamiltonian (LHO basis) ...
            # 1-body kinetic operator ...
        @time T = T1b(Params,Orb_ref)
            # 2-body bare NN interaction ...
        @time V_NN_bare, Orb_NN = V2b_read(Params)
            # 3-body bare NNN interaction ...
        @time V_NNN_bare, Orb_NNN = V3b_no2b_read(Params)

        # Import reference basis ... transformation LHO -> HF ...
        @time C_LHO_HF = O1b_import(Params,Orb_ref,IO * "/bin/C.bin")

        # Import reference 1-body density matrix (OBDM) ...
        @time Rho_HF = O1b_import(Params,Orb_ref,IO * "/bin/rho.bin")

        # Import reference 1-body Hamiltonian ...
        @time h_N = O1b_import(Params,Orb_ref,IO * "/bin/H1b.bin")

        # Import reference 2-body interaction ...
        println("\nImporting reference 2-body interaction ...")

        @time V_NN = O2b_import(Params,Orb_ref,IO * "/bin/H2b.bin")

        # No 2-body correlation function in the initial RPA ...
        Sigma = O2b_initialize_empty()

        # Construct 2-body 1p-1h orbitals ...
        @time Orb_1p1h = orbitals_1p1h_make(Params,Orb_ref;Tol_par = Params.Calc.RPA.Tol_par,Tol_hol = Params.Calc.RPA.Tol_hol)

        # Construct 1-body particle & hole orbitals for the OBDM ...
        @time Orb_ph = RPA_OBDM_orbitals_make(Params,Orb_ref)

        # Construct 1-body transition operators in the LHO basis ...
        @time TrOp_LHO = Tr1b_initialize(Params,Orb_ref,1.0)

        @time TrOp = Tr1b_transformation(Params,Orb_ref,C_LHO_HF,TrOp_LHO)

    # Solve the initial RPA equations ... standard RPA on top of the reference solution ...
        println("\nSolving the initial RPA equations ...")

        # Allocate matrices A & B ...
        @time A, B, N = RPA_allocate(Params,Orb_ref,Orb_1p1h,Orb_NN,h_N,V_NN,Rho_HF,Sigma,false)

        # Solve the RPA eigenvalue problem ...
        @time E, X, Y, Stability = RPA_diagonalize(Params,Orb_ref,Orb_1p1h,A,B,N,TrOp)

        if Stability == false
            println("\n\tWarning! The initial RPA solution is unstable ...")
        end

        # Quasi-boson correlation energy of the initial RPA solution ...
        E_QBA_old = RPA_energy_QBA(Params,Orb_ref,Orb_1p1h,E,X,Y)

        # History of E_QBA (initial RPA first) & of the stability of the iterations ...
        E_QBA_list = [E_QBA_old]
        Stability_list = Bool[]

    # Solve the self-consistent RPA equations ...
        # Current orbitals, OBDM & basis transformations ... start from the reference (HF) solution ...
        Orb = Orb_ref
        Rho = O1b_deepcopy(Rho_HF)

        C_HF_NAT = O1B(diagm(ones(Float64,a_max)),diagm(ones(Float64,a_max)))
        C_LHO_NAT = O1b_deepcopy(C_LHO_HF)

        # Iteration control ...
        Iteration = 1
        Converged = false
        E_QBA = E_QBA_old
        dE_QBA = 1.0
        N_converged = 0

        # Damping (ramp) of the 2-body correlation function ... first iteration ...
        beta = HF_ScRPA_solver_beta(1)

        # Initialize 2-body correlation functions ... 
        #   Sigma     ... used in h_N, V_NN, A & B, current NAT basis ...
        #   Sigma_old ... previous Sigma, current NAT basis, only for the mixing ...
        Sigma_old = O2b_initialize_empty()

        println("\nStarting HF-ScRPA iteration ...")

        # At most Imax iterations ...
        while Iteration <= Iteration_max
            # Old basis ... for the rotation of Sigma ...
            C_HF_NAT_old = O1b_deepcopy(C_HF_NAT)

            # 1) 2-body correlation function from the current RPA amplitudes ... current NAT basis ...
            #    (evaluated before the OBDM & orbital updates, so that X, Y, Orb & Orb_1p1h are consistent)
            if TBCF
                # Sigma_new (damped by beta) ... a new object, Sigma_old still holds the previous Sigma ...
                @time Sigma = RPA_sigma(Params,Orb,Orb_1p1h,Orb_NN,X,Y;beta = beta,Report = false)

                # Linear mixing with the previous correlation function ... both in the current NAT basis ...
                if Iteration > 1
                    RPA_sigma_mix!(Sigma,Sigma_old,alpha_TBCF)
                end
            end

            # 2) Self-consistent OBDM from the current RPA amplitudes ... updates C_HF_NAT in place ...
            @time Rho, C_LHO_NAT, Orb = HF_ScRPA_OBDM_solve(Params,Orb,Orb_ph,Orb_1p1h,X,Y,Rho,C_HF_NAT)

            # Rotate Sigma into the new NAT basis ... C_step = old NAT -> new NAT ...
            if TBCF
                C_step = O1B(C_HF_NAT_old.p' * C_HF_NAT.p, C_HF_NAT_old.n' * C_HF_NAT.n)
                @time Sigma = O2b_transformation(Params,Orb,Orb_NN,Sigma,C_step,Report=false)

                # Sigma_old in the new NAT basis as well ...
                Sigma_old = deepcopy(Sigma)
            end

            # 4) Hamiltonian for the new OBDM in the new NAT basis ...
            @time h_N, V_NN = RPA_Hamiltonian(Params,Orb,Orb_NN,Orb_NNN,C_LHO_NAT,Rho,Sigma,T,V_NN_bare,V_NNN_bare)

            # Transition operators in the new NAT basis ...
            @time TrOp = Tr1b_transformation(Params,Orb,C_LHO_NAT,TrOp_LHO)

            # 5) RPA equations in the new NAT basis with the new occupations ...
            @time A, B, N = RPA_allocate(Params,Orb,Orb_1p1h,Orb_NN,h_N,V_NN,Rho,Sigma,TBCF)

            @time E, X, Y, Stability = RPA_diagonalize(Params,Orb,Orb_1p1h,A,B,N,TrOp)

            # 6) Convergence measure ... relative change of the quasi-boson correlation energy E_QBA between two iterations ...
            E_QBA = RPA_energy_QBA(Params,Orb,Orb_1p1h,E,X,Y)

            if abs(E_QBA) > 1e-12
                dE_QBA = abs(E_QBA - E_QBA_old) / abs(E_QBA)
            else
                dE_QBA = abs(E_QBA - E_QBA_old)
            end

            E_QBA_old = E_QBA

            push!(E_QBA_list,E_QBA)
            push!(Stability_list,Stability)

            @printf("\tHF-ScRPA iteration:   Iteration number = %4d   E0 = %12.9f MeV   dE0_rel = %12.9f\n", Iteration, E_QBA, dE_QBA)

            # Check convergence critterion ...
            if dE_QBA < epsilon && (!TBCF || beta == 1.0)
                N_converged += 1
            else
                N_converged = 0
            end

            if N_converged >= 2 && (!TBCF || beta == 1.0)
                Converged = true
                break
            end

            # Damping (ramp) of the 2-body correlation function for the next iteration ...
            beta = HF_ScRPA_solver_beta(Iteration + 1)

            Iteration += 1
        end

        # Number of performed iterations ...
        if Converged
            N_iterations = Iteration
        else
            N_iterations = Iteration - 1
        end

        @printf("\nHF-ScRPA iteration finished after %4d iterations ...   E0 = %12.9f MeV   dE0_rel = %12.9f\n", N_iterations, E_QBA, dE_QBA)

        if Converged
            println("\nHF-ScRPA iteration has converged ...")
        else
            println("\nHF-ScRPA iteration has NOT converged ... Run terminated ...")
            ScRPA_terminate_summary(N_iterations,E_QBA_list,Stability_list,HF_ScRPA_solver_beta(N_iterations))
            return
        end

    # Final RPA solution for the final OBDM (& 2-body correlation function), final NAT basis ...
        println("\nSolving the final HF-ScRPA equations ...")

        # 2-body correlation function from the last RPA amplitudes ...
        if TBCF == true
            # Hamiltonian for the final OBDM & correlation function ...
            @time h_N, V_NN = RPA_Hamiltonian(Params,Orb,Orb_NN,Orb_NNN,C_LHO_NAT,Rho,Sigma,T,V_NN_bare,V_NNN_bare)
        end

        # Sigma_old is no longer needed ...
        Sigma_old = nothing

        # Allocate final matrices A, B & N ...
        @time A, B, N = RPA_allocate(Params,Orb,Orb_1p1h,Orb_NN,h_N,V_NN,Rho,Sigma,TBCF)

        # Final diagonalization ...
        @time E, X, Y, Stability = RPA_diagonalize(Params,Orb,Orb_1p1h,A,B,N,TrOp)

        if Stability == false
            println("\nWarning! The final HF-ScRPA solution is unstable ...")
        end

    # Process the RPA solutions ...
        # Calculate RPA 2-body correlation function from the final amplitudes ...
        @time Sigma = RPA_sigma(Params,Orb,Orb_1p1h,Orb_NN,X,Y)

        # Evaluate the RPA charge radius chR ...
        chR2 = rho_chR2(Params,Orb,C_LHO_NAT,Rho)

        # Recompute the 1-body transition operators ...
        @time TrOp = Tr1b_initialize(Params,Orb,chR2)

        @time TrOp = Tr1b_transformation(Params,Orb,C_LHO_NAT,TrOp)

        # Update the occupation numbers of the single-particle orbitals from the converged OBDM ...
        Orb = orbitals_update(Orb,Rho)

        # Calculate RPA correlation energy ... (Rho - Rho_HF in the final NAT basis)
        @time E_corr = RPA_energy(Params,Orb,Orb_1p1h,Orb_NN,Orb_NNN,E,X,Y,V_NN,T,V_NN_bare,V_NNN_bare,Rho,Sigma,C_LHO_HF,Rho_HF,C_LHO_NAT)

        # Electromagnetic reduced multipole operators ...
        @time rM = RPA_rM(Params,Orb,Orb_1p1h,Orb_NN,TrOp,Rho,Sigma,X,Y)

        # Electromagnetic reduced transition intensities ...
        @time rB = RPA_rB(Params,Orb_1p1h,rM,real.(E))

        # Transform the OBDM to the LHO basis ...
        Rho_LHO = O1b_transformation(Rho,O1b_transpose(C_LHO_NAT))

        # Export of ScRPA solutions ...
        @time RPA_export(Params,Orb,Orb_1p1h,Orb_NN,Stability,E_corr,E,X,Y,rB,C_LHO_NAT,Rho_LHO,Sigma,h_N)

    return
end