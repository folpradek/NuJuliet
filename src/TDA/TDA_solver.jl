function TDA_solver(Params::Parameters)
    # Import the reference solution ... orbitals, basis, OBDM, Hamiltonian & (Reference_TBCF = true) the 2-body correlation function ...
    Orb, Orb_1p1h, C, Rho, h_N, V_NN, Orb_NN, Sigma = TDA_solver_import(Params)

    # Construct 1-body transition operators ... reference basis ...
        # Charge radius of the reference ground state ... (only needed by the E1 vortical/toroidal/compression operators, not exported)
        chR2 = rho_chR2(Params,Orb,C,Rho)

        @time TrOp = Tr1b_initialize(Params,Orb,chR2)

        @time TrOp = Tr1b_transformation(Params,Orb,C,TrOp)

    # Solve the TDA equations ...
        # Allocate matrices A & N ...
        @time A, N = TDA_allocate(Params,Orb,Orb_1p1h,Orb_NN,h_N,V_NN,Sigma)

        # Solve the generalized TDA eigenvalue problem A X = E N X ...
        @time E, X = TDA_diagonalize(Params,Orb,Orb_1p1h,A,N,TrOp)

    # Process the TDA solutions ...
        # Electromagnetic reduced multipole operators ... exact transition matrix elements w.r.t. the reference ground state ...
        @time rM = TDA_rM(Params,Orb,Orb_1p1h,Orb_NN,TrOp,Rho,Sigma,X)

        # Electromagnetic reduced transition intensities ...
        @time rB = TDA_rB(Params,Orb_1p1h,rM,E)

        # Export of TDA solutions ...
        @time TDA_export(Params,Orb,Orb_1p1h,E,X,rB,C,Rho)

    return
end

function TDA_solver_import(Params::Parameters)
    # Import path for the reference solution ...
    IO = "io/" * Params.Calc.Path * "/" * Params.Calc.TDA.Reference

    # Read reference single-particle orbitals ...
    Orb = orbitals_import(Params,IO * "/bin/orb.bin")

    # Import reference basis ... transformation LHO -> reference basis ...
    @time C = O1b_import(Params,Orb,IO * "/bin/C.bin")

    # Import reference 1-body density matrix (OBDM) ... reference basis ...
    @time Rho = O1b_import(Params,Orb,IO * "/bin/rho.bin")

    # Construct 2-body 1p-1h orbitals ...
    @time Orb_1p1h = orbitals_1p1h_make(Params,Orb;Tol_par = Params.Calc.TDA.Tol_par,Tol_hol = Params.Calc.TDA.Tol_hol)

    # Import reference 1-body Hamiltonian & 2-body interaction ...
    @time h_N = O1b_import(Params,Orb,IO * "/bin/H1b.bin")

    println("\nImporting reference 2-body interaction ...")
    @time V_NN, Orb_NN = O2b_import(Params,Orb,IO * "/bin/H2b.bin",Make_Orb_NN = true)

    # Import reference 2-body correlation function ... (Reference_TBCF = true)
    Sigma = O2b_initialize_empty()
    if Params.Calc.TDA.Reference_TBCF == true
        if !isfile(IO * "/bin/sigma.bin")
            error("Reference_TBCF = true, but no reference 2-body correlation function ''" * IO * "/bin/sigma.bin'' is available ...")
        end
        println("\nImporting reference 2-body correlation function ...")

        @time Sigma = O2b_import(Params,Orb,IO * "/bin/sigma.bin")
    end

    return Orb, Orb_1p1h, C, Rho, h_N, V_NN, Orb_NN, Sigma
end