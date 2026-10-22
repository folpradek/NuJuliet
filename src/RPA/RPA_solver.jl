function RPA_solver(Params::Parameters)
    # Read parameters ...
    IO = "io/" * Params.Calc.Path * "/" * Params.Calc.RPA.Reference

    # Import all data & construct needed structures ...
        # Read reference single-particle orbitals ...
        Orb = orbitals_import(Params,IO * "/bin/orb.bin")

        @time C = O1b_import(Params,Orb,IO * "/bin/C.bin")

        # Import reference 1-body density matrix (OBDM) ...
        @time Rho = O1b_import(Params,Orb,IO * "/bin/rho.bin")

        # Import reference 1-body Hamiltonian in the LHO basis ...
        @time h_N = O1b_import(Params,Orb,IO * "/bin/H1b.bin")

        # Import reference 2-body interaction ...
        println("\nImporting reference 2-body interaction ...")

        @time V_NN, Orb_NN = O2b_import(Params,Orb,IO * "/bin/H2b.bin",Make_Orb_NN = true)

        # If available, import 2-body correlation function ...
        Sigma = O2b_initialize_empty()
        if isfile(IO * "/bin/sigma.bin") && (Params.Calc.RPA.TBCF == true)
            println("\nImporting reference 2-body correlation function ...")
            @time Sigma = O2b_import(Params,Orb,IO * "/bin/sigma.bin")
        end

        # Construct 2-body 1p-1h orbitals ...
        @time Orb_1p1h = orbitals_1p1h_make(Params,Orb)

        # Construct 1-body transition operators ...
        @time TrOp = Tr1b_initialize(Params,Orb,1.0)

        @time TrOp = Tr1b_transformation(Params,Orb,C,TrOp)

    # Solve the RPA equations ...
        # Allocate matrices A & B ...
        @time A, B, N = RPA_allocate(Params,Orb,Orb_1p1h,Orb_NN,h_N,V_NN,Rho,Sigma)

        # Solve the RPA eigenvalue problem ...
        @time E, X, Y, Stability = RPA_diagonalize(Params,Orb_1p1h,A,B,N,TrOp)

    # Process the RPA solutions ...
        # Calculate RPA One-Body Density Matrix (OBDM) ...
        @time Rho_RPA = RPA_OBDM(Params,Orb,Orb_1p1h,Y)

        # Evaluate the RPA charge radius chR ...
        chR2 = OBDM_chR2(Params,Orb,C,Rho_RPA)

        # Recompute the 1-body transition operators ...
        @time TrOp = Tr1b_initialize(Params,Orb,chR2)

        @time TrOp = Tr1b_transformation(Params,Orb,C,TrOp)

        # Calculate RPA 2-body correlation function ...
        #Sigma_RPA = O2b_initialize_empty()
        @time Sigma = RPA_sigma(Params,Orb,Orb_1p1h,Orb_NN,Rho,X,Y)

        # Calculate RPA correlation energy ...
        @time E_corr = RPA_energy(Params,Orb,Orb_1p1h,Orb_NN,E,Y,Rho_RPA,h_N,Sigma,V_NN)

        # Electromagnetic reduced multipole operators ...
            # Placeholder if test ...
        if Params.Calc.RPA.Transitions == "Exact"
            @time rM = RPA_rM(Params,Orb,Orb_1p1h,Orb_NN,TrOp,Rho,Sigma,X,Y)
        elseif Params.Calc.RPA.Transitions == "Linear"
            @time rM = RPA_rM_linearized(Params,Orb_1p1h,TrOp,Rho,X,Y)
        end

        # Electromagnetic reduced transition intensities ...
        @time rB = RPA_rB(Params,Orb_1p1h,rM,real.(E))

        # Transform RPA OBDM to the LHO basis ...
        Rho = O1B(C.p * Rho_RPA.p * C.p', C.n * Rho_RPA.n * C.n')

        # Export of RPA & TDA solutions ...
        @time RPA_export(Params,Orb,Orb_1p1h,Stability,E_corr,E,X,Y,rB,C,Rho,Sigma)

    return
end