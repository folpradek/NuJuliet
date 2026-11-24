function HFB_summary(Params::Parameters,E_HFB::Vector{Float64},T_HFB::Float64,Lambda::pnFloat,dA::pnFloat,Convergence::Bool,Iteration::Int64)
    # Read parameters ...
    hw = Params.Calc.hw
    A, Z = Params.Calc.A, Params.Calc.Z
    dZ, dN = dA.p, dA.n
    N_max = Params.Calc.Nmax
    N_2max = Params.Calc.N2max
    N_3max = Params.Calc.N3max
    CMS = Params.Calc.CMS
    Output_File = Params.Calc.Path
    epsilon = Params.Calc.HFB.Tol

    # Export the HF calculation summary ...
    println("\nExporting HFB calculation summary ...")
    Summary =  open(string("io/", Output_File, "/HFB/HFB_summary.dat"), "a")
        println(Summary, "Spherical Hartree-Fock-Bogoliubov mean-field calculation summary")
        println(Summary, "\nTarget nuclid:       A = " * string(A) * ", Z = " * string(Z))
        println(Summary, "\nCalculation data:    hw = " * string(hw) * " , N_max = " * string(N_max) *
                " , N_2max = " * string(N_2max) * " , N_3max = " * string(N_3max) * ", J-basis size = " *
                string(div((N_max+1)*(N_max+2),2)) * ", M-basis size = " * string(div((N_max+1)*(N_max+2)*(N_max+3),6)))
        println(Summary, "\t\t\tCenter-of-Mass System correction is set to:   CMS = ''" * string(CMS) * "''")
        if CMS == "CMS1+2B"
            println(Summary, "\n\t\t\tCombined 1-body + 2-body Center-of-Mass System (CMS) motion correction is set ...")
        elseif CMS == "CMS2B"
            println(Summary, "\n\t\t\tOnly pure 2-body Center-of-Mass System (CMS) motion correction is set ...")
        else
            println(Summary, "\n\t\t\tNo Center-of-Mass System (CMS) motion correction is set ...")
        end
        println(Summary, "\nPairing interaction parameters:")
        println(Summary, "cP2N     = " * string(Params.Int.cP2N))
        println(Summary, "cP3N     = " * string(Params.Int.cP3N))
        if Convergence == true
            println(Summary, "\nHFB equations converged in " * string(Iteration) * " iterations with precision epsilon = " * string(epsilon) * " ...")
        else
            println(Summary, "\n\nHFB equations did NOT converge in " * string(Iteration) * " iterations with precision epsilon = " * string(epsilon) * " ...")
        end

        println(Summary, "\nSpherical HFB solution review:")

        @printf(Summary, "\nNumber of HFB Iterations = %5d, Convergence precision =  %.3e (dE <-> MeV, dN <-> Nucleons)\n", Iteration, epsilon)
        @printf(Summary, "\nE_HFB      = %15.8f\t MeV \t\t ... \t HFB mean-field ground-state energy\n", sum(E_HFB))
        @printf(Summary, "E_MF       = %15.8f\t MeV \t\t ... \t mean-field ground-state energy\n", E_HFB[1])
        @printf(Summary, "E_Par      = %15.8f\t MeV \t\t ... \t pairing ground-state energy\n", E_HFB[2] + E_HFB[3])
        @printf(Summary, "E_Par (2N) = %15.8f\t MeV \t\t ... \t 2-body pairing ground-state energy\n", E_HFB[2])
        @printf(Summary, "E_Par (3N) = %15.8f\t MeV \t\t ... \t 3-body pairing ground-state energy\n", E_HFB[3])
        if Params.Calc.HFB.LNT == true
            @printf(Summary, "E_LN       = %15.8f\t MeV \t\t ... \t HFB mean-field Lipkin-Nogami ground-state energy correction\n", E_HFB[4])
         end
        @printf(Summary, "T_HFB      = %15.8f\t MeV\t\t ... \t HFB mean-field ground-state kinetic energy\n", T_HFB)
        @printf(Summary, "\nE_HFB / A      = %15.8f\t MeV / Nucleon \t\t ... \t HFB mean-field ground-state energy per nucleon\n", sum(E_HFB) / A)
        @printf(Summary, "E_MF  / A      = %15.8f\t MeV / Nucleon \t\t ... \t mean-field ground-state energy per nucleon\n", E_HFB[1] / A)
        @printf(Summary, "E_Par / A      = %15.8f\t MeV / Nucleon \t\t ... \t pairing ground-state energy per nucleon\n", (E_HFB[2] + E_HFB[3]) / A)
        @printf(Summary, "E_Par / A (2N) = %15.8f\t MeV / Nucleon \t\t ... \t 2-body pairing ground-state energy per nucleon\n", E_HFB[2] / A)
        @printf(Summary, "E_Par / A (3N) = %15.8f\t MeV / Nucleon \t\t ... \t 3-body pairing ground-state energy per nucleon\n", E_HFB[3] / A)
        @printf(Summary, "T_HFB / A      = %15.8f\t MeV / Nucleon\t\t ... \t HFB mean-field ground-state kinetic energy per nucleon\n", T_HFB / A)
        @printf(Summary, "\n\nValues of HFB chemical potentials:\n")
        @printf(Summary, "\npLambda =  %15.8f MeV\n", Lambda.p)
        @printf(Summary, "nLambda =  %15.8f MeV\n", Lambda.n)
        if Params.Calc.HFB.LNT == true
            Lambda_2 = HFB_Lipkin_Nogami_Lambda_2_import(Params)
            @printf(Summary, "\n\nValues of LN-HFB coefficients:\n")
            @printf(Summary, "\npLambda_2 =  %15.8f MeV\n", Lambda_2.p)
            @printf(Summary, "nLambda_2 =  %15.8f MeV\n", Lambda_2.n)
        end
        @printf(Summary, "\nDispersion of HFB particle numbers dZ & dN:\n")
        @printf(Summary, "\ndZ = %12.6f\n", dZ)
        @printf(Summary, "dN = %12.6f\n", dN)
        @printf(Summary, "\ndZ / Z = %12.6f\n", dZ / Float64(Z))
        @printf(Summary, "dN / N = %12.6f\n", dN / Float64(A - Z))

    close(Summary)

    return
end

function HFB_summary_SQS(Params::Parameters,SQE::pnVector,Rho::O1B,Orb::Vector{Orb1B})
    # Read parameters
    N_max = Params.Calc.Nmax
    N_2max = Params.Calc.N2max
    Output_File = Params.Calc.Path
    a_max = div((N_max + 1)*(N_max + 2),2)
    J_max = N_2max + 1

    # Initialize temporary counters for radial numbers ...
    pn, nn = zeros(Int64,N_max+1,J_max+1), zeros(Int64,N_max+1,J_max+1)

    # Export orbitals ...
    println("\nExporting information on single-quasiparticle orbits ...")
    Summary =  open(string("io/", Output_File, "/HFB/HFB_summary.dat"), "a")
        println(Summary,"\nProton single-quasiparticle states")
        println(Summary,"_______________________________________________________________________________________________________")
        @printf(Summary, "%4s %4s %4s %10s %14s", "n", "l", "2j", "O_a", "E\n")
        @inbounds for a in 1:a_max
            l_a = Orb[a].l
            j_a = Orb[a].j
            n_a = pn[l_a+1,j_a+1]
            @printf(Summary, "%4d %4d %4d %10.3f %14.5f\n", n_a, l_a, j_a,Rho.p[a,a],SQE.p[a])
            pn[l_a+1,j_a+1] += 1
        end
        println(Summary,"_______________________________________________________________________________________________________")

        println(Summary,"\nNeutron single-quasiparticle states")
        println(Summary,"_______________________________________________________________________________________________________")
        @printf(Summary, "%4s %4s %4s %10s %14s", "n", "l", "2j", "O_a", "E\n")
        @inbounds for a in 1:a_max
            l_a = Orb[a].l
            j_a = Orb[a].j
            n_a = nn[l_a+1,j_a+1]
            @printf(Summary, "%4d %4d %4d %10.3f %14.5f\n", n_a, l_a, j_a,Rho.n[a,a],SQE.n[a])
            nn[l_a+1,j_a+1] += 1
        end
        println(Summary,"_______________________________________________________________________________________________________")

    close(Summary)

    return
end

function HFB_export(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,C::O1B,U::O1B,V::O1B,Rho::O1B,Kappa::O1B,H_N::qpO1B,H_NN::qpO2B)
    # Read parameters
    Output_File = Params.Calc.Path

    # Export HFB canonical basis matrix C ...
    C_Export_Path = "io/" * Output_File * "/HFB/bin/C.bin"
    O1b_export(Params,Orb,C,C_Export_Path)

    # Export HFB amplitudes U & V ... in the canonical basis ...
        # Case of U ...
    U_Export_Path = "io/" * Output_File * "/HFB/bin/U.bin"
    O1b_export(Params,Orb,U,U_Export_Path)

        # Case of V ...
    V_Export_Path = "io/" * Output_File * "/HFB/bin/V.bin"
    O1b_export(Params,Orb,V,V_Export_Path)

    # Export HFB canonical basis density operator Rho ...
    Rho_Export_Path = "io/" * Output_File * "/HFB/bin/Rho.bin"
    O1b_export(Params,Orb,Rho,Rho_Export_Path)

    # Export HFB canonical basis pairing tensor Kappa ...
    Kappa_Export_Path = "io/" * Output_File * "/HFB/bin/Kappa.bin"
    O1b_export(Params,Orb,Kappa,Kappa_Export_Path)

    # Export HFB 1-body Hamiltonian H_N ...
    H1B_Export_Path = "io/" * Output_File * "/HFB/bin/qpH1b.bin"
    qpO1b_export(Params,Orb,H_N,H1B_Export_Path)

    # Export HFB 2-body Hamiltonian H_NN ..
    H2B_Export_Path = "io/" * Output_File * "/HFB/bin/qpH2b.bin"
    qpO2b_export(Params,Orb_NN,H_NN,H2B_Export_Path)

    return
end