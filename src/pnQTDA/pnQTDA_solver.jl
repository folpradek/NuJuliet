function pnQTDA_solver(Params::Parameters)
    # Make s.p. orbitals - NuHamil ordering convention ...
    @time Orb = orbitals_make(Params)

    # Make proton-neutron 2qp orbitals  ...
    Orb_pn2qp = orbitals_pn2qp_make(Params,Orb)

    # Import H_N ... 1-body quasiparticle Hamiltonian ...
    @time H_N = qpO1b_import(Params,Orb,"io/" * Params.Calc.Path * "/" * Params.Calc.pnQTDA.Reference * "/bin/qpH1b.bin")

    # Import residual 2-body NN interaction quasiparticle Hamiltonian ...
    @time H_NN, Orb_NN = qpO2b_import(Params,Orb,"io/" * Params.Calc.Path * "/" * Params.Calc.pnQTDA.Reference * "/bin/qpH2b.bin")

    # Allocate the pnQTDA matrix A ...
    @time A = pnQTDA_allocate(Params,Orb_NN,Orb_pn2qp,H_N,H_NN)

    # Solve pnQTDA eigenvalue problem ...
    @time E_pnQTDA, X_pnQTDA = pnQTDA_diagonalize(Params,A)

    # Export of pnQTDA solutions ...
    @time pnQTDA_export(Params,Orb_pn2qp,E_pnQTDA,X_pnQTDA)

    println("\tpnQTDA solver executed properly ...\n")

    return
end
