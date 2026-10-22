function QRPA_solver(Params::Parameters)
    # Make s.p. orbitals - NuHamil ordering convention ...
    @time Orb = orbitals_make(Params)

    # Make 2qp orbitals  ...
    Orb_2qp = orbitals_2qp_make(Params,Orb)

    # Import LHO -> (HFB) quasiparticle mean-field transformation matrices ...
    @time C = O1b_import(Params,Orb,"io/" * Params.Calc.Path * "/" * Params.Calc.QRPA.Reference * "/bin/C.bin")

    # Import U & V quasiparticle amplitudes ...
    @time U = O1b_import(Params,Orb,"io/" * Params.Calc.Path * "/" * Params.Calc.QRPA.Reference * "/bin/U.bin")
    @time V = O1b_import(Params,Orb,"io/" * Params.Calc.Path * "/" * Params.Calc.QRPA.Reference * "/bin/V.bin")

    # Import H_N ... 1-body quasiparticle Hamiltonian ...
    @time H_N = qpO1b_import(Params,Orb,"io/" * Params.Calc.Path * "/" * Params.Calc.QRPA.Reference * "/bin/qpH1b.bin")

    # Import residual 2-body NN interaction quasiparticle Hamiltonian ...
    @time H_NN, Orb_NN = qpO2b_import(Params,Orb,"io/" * Params.Calc.Path * "/" * Params.Calc.QRPA.Reference * "/bin/qpH2b.bin")

    # Initialize the quasiparticle transition operators ...
    @time qpTrOp = qpTr1b_initialize(Params,Orb,1.0,C,U,V)

    # Initialize the 1-body particle number operator in quasiparticle representation ...
    qpN = qpN1b_make(Params,U,V)

    # Allocate the QRPA matrices A & B ...
    @time A, B = QRPA_allocate(Params,Orb,Orb_NN,Orb_2qp,H_N,H_NN)

    # Solve QRPA eigenvalue problem ...
    @time E_QRPA, X_QRPA, Y_QRPA, Stability = QRPA_diagonalize(Params,Orb,Orb_2qp,A,B,qpN,qpTrOp)

    # Determine the A-body 1-phonon QRPA levels ...
    @time A_phonon = QRPA_A_phonon_particle_number(Params,Orb,Orb_2qp,U,V,X_QRPA,Y_QRPA)

    # Calculate the QRPA density operator Rho ...
    @time Rho = QRPA_OBDM(Params,Orb,Orb_2qp,U,V,Y_QRPA,A_phonon)

    # Evaluate the reference charge radius chR ...
    chR2 = OBDM_chR2(Params,Orb,C,Rho)

    # Initialize the quasiparticle transition operators ...
    @time qpTrOp = qpTr1b_initialize(Params,Orb,chR2,C,U,V)

    # Calculate the QRPA reduced electromagnetic transition amplitudes M ...
    @time rM_QRPA = QRPA_rM(Params,Orb_2qp,X_QRPA,Y_QRPA,qpTrOp)

    # Calculate the QRPA reduced electromagnetic transition intensities B ...
    @time rB_QRPA = QRPA_rB(Params,Orb_2qp,rM_QRPA,E_QRPA)

    # Calculate the QRPA correlation energy ...
    @time E_corr = QRPA_energy(Params,Orb_2qp,E_QRPA,Y_QRPA,A_phonon)

    # Export of QRPA solutions ...
    @time QRPA_export(Params,Orb,Orb_2qp,Stability,E_corr,Rho,C,E_QRPA,X_QRPA,Y_QRPA,A_phonon,rB_QRPA)

    println("\nQRPA solver executed properly ...\n")

    return
end