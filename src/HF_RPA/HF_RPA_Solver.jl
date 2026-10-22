function HF_RPA_solver(Params::Parameters)
    # Make s.p. orbitals - NuHamil ordering convention ...
    @time Orb = orbitals_make(Params)

    # Import LHO -> reference OBDM transformation matrices ...
    @time C = O1b_import(Params,Orb,"IO/" * Params.Calc.Path * "/Bin/C_HF.bin")

    # Import Residual 2-body NN interaction ...
    println("\nImporting Residual 2-body interaction ...")
    @time V_NN, Orb_NN = O2b_import(Params,Orb,"IO/" * Params.Calc.Path * "/Bin/V2B_HF.bin",Make_Orb_NN=true)

    # Prepare Particle & Hole orbitals ...
    @time N_Particle, Particle, N_Hole, Hole = orbitals_ph_make(Params,Orb)

    # Prepare 1p-1h phonon states ...
    @time N_Phonon, Phonon = orbitals_one_phonon_make(N_Particle,Particle,N_Hole,Hole)

    # Initialize transition operators ...
    @time TrOp = Tr1b_initialize(Params,Orb,1.0)

    @time TrOp = Tr1b_transformation(Params,Orb,C,TrOp)

    # Count & pre-index all phonon states in JP subspaces ...
    @time N_nu, Orb_Phonon = HF_RPA_phonon_count(Params,N_Phonon,Phonon)

    # Allocate matrices A & B ...
    @time A, B = HF_RPA_allocate(Params,N_nu,Orb_Phonon,Phonon,Particle,Hole,Orb,Orb_NN,V_NN)

    # Solve TDA eigenvalue problem and RPA generalized-eigenvalue problem ...
    @time E_TDA, X_TDA, E_RPA, X_RPA, Y_RPA, Stability = HF_RPA_diagonalize(Params,A,B,N_nu,Orb_Phonon,Phonon,Particle,Hole,TrOp)

    # Calculate RPA One-Body Density Matrix (OBDM) ...
    @time Rho_RPA = HF_RPA_OBDM(Params,Orb,Orb_Phonon,Phonon,Particle,Hole,N_nu,Y_RPA)

        # Evaluation of the RPA correlation energy ... for testing purposes!!!
        #HF_RPA_energy_density(Params,N_nu,N_Particle,Particle,N_Hole,Hole,Orb,Orb_NN,V_NN,E_RPA,Y_RPA,Rho_RPA)

        #HF_RPA_energy_bosonic(Params,N_nu,E_RPA,Y_RPA)

    # Evaluate the RPA charge radius chR ...
    chR2 = OBDM_chR2(Params,Orb,C,Rho_RPA)

    # Reinitialize the 1-body transition operators ...
    @time TrOp = Tr1b_initialize(Params,Orb,chR2)

    @time TrOp = Tr1b_transformation(Params,Orb,C,TrOp)

    # Transform RPA OBDM to the LHO basis ...
    Rho_RPA = O1B(C.p * Rho_RPA.p * C.p', C.n * Rho_RPA.n * C.n')

    # Calculate RPA correlation energy ...
    @time E_corr = HF_RPA_energy(Params,N_nu,E_RPA,Y_RPA)

    # Electromagnetic reduced multipole operators ...
    @time rM_TDA, rM_RPA = HF_RPA_rM(Params,N_nu,Orb_Phonon,Phonon,Particle,Hole,X_TDA,X_RPA,Y_RPA,TrOp)

    # Electromagnetic reduced transition intensities ...
    @time rB_TDA = HF_RPA_rB(Params,N_nu,rM_TDA,E_TDA)
    @time rB_RPA = HF_RPA_rB(Params,N_nu,rM_RPA,real.(E_RPA))

    # Export of RPA & TDA solutions ...
    @time HF_RPA_export(Params,Orb,N_nu,E_corr,E_TDA,E_RPA,X_TDA,X_RPA,Y_RPA,rB_TDA,rB_RPA,C,Rho_RPA,Stability)

        # RPA transition radial densities export ...
        # Requires manual control in corresponding function in HF_RPA_Transitions.jl file
        #   !!! One has to choose the phonons to export !!!
        #@time HF_RPA_Transition_Densities_Export(Params,Orb,N_nu,Orb_Phonon,Phonon,Particle,Hole,X_RPA,Y_RPA,TrOp)

    return
end