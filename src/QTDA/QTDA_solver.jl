function QTDA_solver(Params::Parameters)
    # Make s.p. orbitals - NuHamil ordering convention ...
    @time Orb = orbitals_make(Params)

    # Make 2qp orbitals  ...
    Orb_2qp = orbitals_2qp_make(Params,Orb)

    # Import LHO -> (HFB) quasiparticle mean-field transformation matrices ...
    @time C = O1b_import(Params,Orb,"io/" * Params.Calc.Path * "/" * Params.Calc.QTDA.Reference * "/bin/C.bin")

    # Import U & V quasiparticle amplitudes ...
    @time U = O1b_import(Params,Orb,"io/" * Params.Calc.Path * "/" * Params.Calc.QTDA.Reference * "/bin/U.bin")
    @time V = O1b_import(Params,Orb,"io/" * Params.Calc.Path * "/" * Params.Calc.QTDA.Reference * "/bin/V.bin")

    # Construct the 1-body density operators Rho & Kappa ...
    Rho, Kappa = HFB_density_operator(Params,U,V,Orb)

    # Import H_N ... 1-body quasiparticle Hamiltonian ...
    @time H_N = qpO1b_import(Params,Orb,"io/" * Params.Calc.Path * "/" * Params.Calc.QTDA.Reference * "/bin/qpH1b.bin")

    # Import residual 2-body NN interaction quasiparticle Hamiltonian ...
    @time H_NN, Orb_NN = qpO2b_import(Params,Orb,"io/" * Params.Calc.Path * "/" * Params.Calc.QTDA.Reference * "/bin/qpH2b.bin")

    # Evaluate the reference charge radius chR ...
    chR2 = OBDM_chR2(Params,Orb,C,Rho)

    # Initialize the quasiparticle transition operators ...
    @time qpTrOp = qpTr1b_initialize(Params,Orb,chR2,C,U,V)

    # Initialize the 1-body particle number operator in quasiparticle representation ...
    qpN = qpN1b_make(Params,U,V)

    # Allocate the QTDA matrix A ...
    @time A = QTDA_allocate(Params,Orb,Orb_NN,Orb_2qp,H_N,H_NN)

    # Solve QTDA eigenvalue problem ...
    @time E_QTDA, X_QTDA= QTDA_diagonalize(Params,Orb,Orb_2qp,A,qpN,qpTrOp)

    # Calculate the QTDA reduced electromagnetic transition amplitudes  M ...
    @time rM_QTDA = QTDA_rM(Params,Orb_2qp,X_QTDA,qpTrOp)

    # Calculate the QTDA reduced electromagnetic transition intensities  B ...
    @time rB_QTDA = QTDA_rB(Params,Orb_2qp,rM_QTDA,E_QTDA)

    # Export of QTDA solutions ...
    @time QTDA_export(Params,Orb,Orb_2qp,Rho,E_QTDA,X_QTDA,rB_QTDA)

    println("\tQTDA solver executed properly ...\n")

    return
end