function RPA_solver(Params::Parameters)
    # Make s.p. orbitals ...
    @time Orb = orbitals_make(Params)

    # Import LHO -> reference HF transformation matrices ...
    @time C_HF = O1b_import(Params,Orb,"IO/" * Params.Calc.Path * "/Bin/C_HF.bin")

    # Calculate the inverse transformation matrix to C_HF ... HF -> LHO ...
    C_HF_LHO = O1B(C_HF.p',C_HF.n')

    # Import reference HF 1-body Hamiltonian ... in the HF basis ...
    @time h_N_HF = O1b_import(Params,Orb,"IO/" * Params.Calc.Path * "/Bin/h_HF.bin")

    # Import Residual 2-body NN interaction ...
    println("\nImporting HF Residual 2-body interaction ...")
    @time V_NN, Orb_NN = O2b_import(Params,Orb,"IO/" * Params.Calc.Path * "/Bin/V2B_HF.bin",Make_Orb_NN=true)

        # Perform perturbative calculation of 1-body density matrix (OBDM) ... construct basis of NOs ...
        C, Rho = HF_MBPT_OBDM_NOB(Params,Orb,Orb_NN,V_NN)

        # Evaluate the MBPT 2-body correlation function matrix (TBCM) ... in the LHO basis ...
        Sigma_NN_LHO = HF_MBPT_TBDM(Params,C_HF_LHO,h_N_HF,V_NN,Orb,Orb_NN)

        # Perform transformation of the 2-body correlation function into the target basis ...
        @time Sigma_NN = O2b_transformation(Params,Orb,Orb_NN,Sigma_NN_LHO,C)

        # Deallocate the HF transformation matrix & residual interaction to free memory ...
        C_HF, C_HF_LHO = nothing, nothing
        h_N_HF = nothing
        V_NN = nothing
        GC.gc()

        # Import kinetic operator & bare NN & NNN interactions ...
            # 1-body kinetic operator ...
            @time T = T1b(Params,Orb)

            # 2-body NN interaction & Orbitals ...
            @time V_NN_bare, Orb_NN_bare = V2b_read(Params,Orb)

            # 3-body NNN interaction & Orbitals ...
            @time V_NNN_bare, Orb_NNN_bare = V3b_no2b_read(Params,Orb)

        # Construct new 1-body mean-field Hamiltonian ...
        @time h_N = h1b(Params,Orb,Orb_NN_bare,Orb_NNN_bare,T,V_NN_bare,V_NNN_bare,O1B(C.p * Rho.p * C.p', C.n * Rho.n * C.n'),Sigma_NN_LHO)

        # Deallocate & perform garbage collection ...
        Sigma_NN_LHO, Orb_NN = nothing, nothing
        GC.gc()

        # Transform the 1-body mean-field Hamiltonian h_N to the target ... NAT basis ...
        h_N = O1B(C.p' * h_N.p * C.p, C.n' * h_N.n * C.n)

        # Construct new residual 2-body interaction ...
        @time V_NN, Orb_NN = V2b_no2b(Params,Orb,Orb_NN_bare,Orb_NNN_bare,V_NN_bare,V_NNN_bare,O1B(C.p * Rho.p * C.p', C.n * Rho.n * C.n'),C)

            #E_corr = V2b_correlation_energy(Params,Orb,Orb_NN,Sigma_NN,V_NN)
            #println("E_corr = $(E_corr) MeV")
            #throw("Stop here ... 1-body correlated Hamiltonian evaluated ...")

            #@time E_NAT = HF_energy(Params,O1B(C.p * Rho.p * C.p', C.n * Rho.n * C.n'),Orb,Orb_NN_bare,Orb_NNN_bare,T,V_NN_bare,V_NNN_bare)
            #println("Mean-field NAT energy: E_NAT = $(E_NAT) MeV")

        T = nothing
        V_NN_bare, Orb_NN_bare = nothing, nothing
        V_NNN_bare, Orb_NNN_bare = nothing, nothing
        GC.gc()

    # Prepare Particle & Hole orbitals ...
    @time N_Particle, Particle, N_Hole, Hole = orbitals_ph_make(Params,Orb)

    # Prepare 1p-1h phonon states ...
    @time N_Phonon, Phonon = orbitals_one_phonon_make(N_Particle,Particle,N_Hole,Hole)

    # Initialize transition operators ...
    @time TrOp = Tr1b_initialize(Params,Orb,1.0)

    @time TrOp = Tr1b_transformation(Params,Orb,C,TrOp)

    # Count & pre-index all phonon states in JP subspaces ...
    @time N_nu, Orb_Phonon = RPA_phonon_count(Params,N_Phonon,Phonon)

    # Allocate matrices A & B ...
    @time A, B, N = RPA_allocate(Params,N_nu,Orb_Phonon,Phonon,Particle,Hole,Orb,Orb_NN,h_N,V_NN,Rho,Sigma_NN)

    # Solve TDA eigenvalue problem and RPA generalized-eigenvalue problem ...
    @time E_RPA, X_RPA, Y_RPA, Stability = RPA_diagonalize(Params,A,B,N,N_nu,Orb_Phonon,Phonon,Particle,Hole,TrOp)

    # Calculate RPA One-Body Density Matrix (OBDM) ...
    @time Rho_RPA = RPA_OBDM(Params,Orb,Orb_Phonon,Phonon,Particle,Hole,N_nu,Y_RPA)

    # Evaluate the RPA charge radius chR ...
    chR2 = OBDM_chR2(Params,Orb,C,Rho_RPA)

    # Reinitialize the 1-body transition operators ...
    @time TrOp = Tr1b_initialize(Params,Orb,chR2)

    @time TrOp = Tr1b_transformation(Params,Orb,C,TrOp)

    # Transform RPA OBDM to the LHO basis ...
    Rho_RPA = O1B(C.p * Rho_RPA.p * C.p', C.n * Rho_RPA.n * C.n')


        # Test evaluation of the RPA correlation energy ...
        Sigma_RPA = RPA_sigma2b_allocate(Params,N_nu,Orb_Phonon,Phonon,N_Particle,Particle,N_Hole,Hole,Orb,Orb_NN,X_RPA,Y_RPA)


    # Calculate RPA correlation energy ...
    @time E_corr = RPA_energy(Params,N_nu,E_RPA,Y_RPA,N)

        # Test calculation of the correlation energy
        E_corr = V2b_correlation_energy(Params,Orb,Orb_NN,Sigma_RPA,V_NN)

        # Run PN test
        RPA_rho2b_allocate_test(Params,N_nu,Orb_Phonon,Phonon,N_Particle,Particle,N_Hole,Hole,Orb,Orb_NN,X_RPA,Y_RPA)

    # Electromagnetic reduced multipole operators ...
    @time rM_RPA = RPA_rM(Params,N_nu,Orb_Phonon,Phonon,Particle,Hole,X_RPA,Y_RPA,TrOp)

    # Electromagnetic reduced transition intensities ...
    @time rB_RPA = RPA_rB(Params,N_nu,rM_RPA,real.(E_RPA))

    # Export of RPA & TDA solutions ...
    @time RPA_export(Params,Orb,N_nu,E_corr,E_RPA,X_RPA,Y_RPA,rB_RPA,C,Rho_RPA,Stability)

    return
end