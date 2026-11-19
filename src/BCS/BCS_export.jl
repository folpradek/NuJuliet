function BCS_summary(Params::Parameters,Params_Ref::Parameters,E_MF::Float64,E_BCS::Vector{Float64},T_BCS::Float64,Lambda::pnFloat,dA::pnFloat,Convergence::Bool,Iteration::Int64)
    # Read parameters ...
    hw = Params.Calc.hw
    A, Z = Params.Calc.A, Params.Calc.Z
    A_ref, Z_ref = Params_Ref.Calc.A, Params_Ref.Calc.Z
    dZ, dN = dA.p, dA.n
    N_max = Params.Calc.Nmax
    N_2max = Params.Calc.N2max
    N_3max = Params.Calc.N3max
    CMS = Params.Calc.CMS
    Output_File = Params.Calc.Path
    epsilon = Params.Calc.BCS.Tol

    # Export the HF calculation summary ...
    println("\nExporting BCS calculation summary ...")
    Summary =  open(string("io/", Output_File, "/BCS/BCS_summary.dat"), "a")
        println(Summary, "Target nuclid:       A = " * string(A) * ", Z = " * string(Z))
        println(Summary, "Reference nuclid:    A = " * string(A_ref) * ", Z = " * string(Z_ref))
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
        println(Summary, "\t\t\tCMS correction is calculated with respect to the reference (closed-subshell) nucleus! ...")
        if Convergence == true
            println(Summary, "\n\nBCS equations converged in " * string(Iteration) * " iterations with precision epsilon = " * string(epsilon) * " ...")
        else
            println(Summary, "\n\nBCS equations did NOT converge in " * string(Iteration) * " iterations with precision epsilon = " * string(epsilon) * " ...")
        end
        println(Summary, "\n\nBCS solution review:")

        @printf(Summary, "\nE_tot      = %15.8f\t MeV \t\t ... \t Total BCS ground-state energy\n", E_MF + sum(E_BCS))
        @printf(Summary, "E_MF       = %15.8f\t MeV \t\t ... \t Mean-field ground-state energy\n", E_MF)
        @printf(Summary, "E_BCS      = %15.8f\t MeV \t\t ... \t BCS pairing ground-state energy\n", sum(E_BCS))
        @printf(Summary, "E_BCS (2N) = %15.8f\t MeV \t\t ... \t BCS pairing NN ground-state energy\n", E_BCS[1])
        @printf(Summary, "E_BCS (3N) = %15.8f\t MeV \t\t ... \t BCS pairing NNN ground-state energy\n", E_BCS[2])
        @printf(Summary, "T_BCS      = %15.8f\t MeV \t\t ... \t BCS mean-field ground-state kinetic energy\n", T_BCS)
        @printf(Summary, "\nE_tot / A      = %15.8f\t MeV / Nucleon \t\t ... \t Total BCS ground-state energy per nucleon\n", (E_MF + sum(E_BCS)) / A)
        @printf(Summary, "E_MF / A       = %15.8f\t MeV / Nucleon \t\t ... \t Mean-field ground-state energy per nucleon\n", E_MF / A)
        @printf(Summary, "E_BCS / A      = %15.8f\t MeV / Nucleon \t\t ... \t BCS pairing ground-state energy per nucleon\n", sum(E_BCS) / A)
        @printf(Summary, "E_BCS / A (2N) = %15.8f\t MeV / Nucleon \t\t ... \t BCS pairing NN ground-state energy per nucleon\n", E_BCS[1] / A)
        @printf(Summary, "E_BCS / A (3N) = %15.8f\t MeV / Nucleon \t\t ... \t BCS pairing NNN ground-state energy per nucleon\n", E_BCS[2] / A)
        @printf(Summary, "T_BCS / A      = %15.8f\t MeV / Nucleon \t\t ... \t BCS mean-field ground-state kinetic energy per nucleon\n", T_BCS / A)
        @printf(Summary, "\n\nValues of BCS chemical potentials:\n")
        @printf(Summary, "\npLambda =  %15.8f MeV\n", Lambda.p)
        @printf(Summary, "nLambda =  %15.8f MeV\n", Lambda.n)
        println(Summary, "\nBCS particle number dispersions dZ & dN:")
        println(Summary, "\ndZ = " * string(round(dZ, sigdigits=6)))
        println(Summary, "dN = " * string(round(dN, sigdigits=6)))
        println(Summary, "\ndZ / Z = " * string(round(dA.p / Float64(Z), sigdigits=6)))
        println(Summary, "dN / N = " * string(round(dA.n / Float64(A - Z), sigdigits=6)))
    close(Summary)

    return
end

function BCS_summary_SQS(Params::Parameters,SPE::pnVector,SQE::pnVector,U::pnVector,V::pnVector)
    # Read parameters
    N_max = Params.Calc.Nmax
    N_2max = Params.Calc.N2max
    a_max = div((N_max + 1)*(N_max + 2),2)
    J_max = N_2max + 1
    Output_File = Params.Calc.Path

    # Setup single-particle orbitals, these are used for reference HF - NuHamil ordering ...
    Orb = orbitals_make(Params)

    # Initialite occupation counters ...
    pn, nn = zeros(Int64,N_max+1,J_max+1), zeros(Int64,N_max+1,J_max+1)

    # Export orbitals ...
    println("\nExporting information on single-(quasi)particle orbits ...")
    Summary =  open(string("io/" * Output_File, "/BCS/BCS_summary.dat"), "a")
        println(Summary,"\nProton single-(quasi)particle states")
        println(Summary,"____________________________________________________________________")
        @printf(Summary, "%4s %4s %4s %8s %14s %14s %14s %14s", "n", "l", "2j", "O_a", "e", "U2", "V2", "E\n")
        for a in 1:a_max
            l_a = Orb[a].l
            j_a = Orb[a].j
            n_a = pn[l_a+1,j_a+1]
            if Orb[a].pO == 1
                O_a = 1.0
            else
                O_a = 0.0
            end
            @printf(Summary, "%4s %4s %4s %8.3f %14.5f %14.3f %14.3f %14.5f\n",n_a, l_a, j_a, O_a, SPE.p[a], U.p[a]^2, V.p[a]^2, SQE.p[a])
            pn[l_a+1,j_a+1] += 1
        end
        println(Summary,"____________________________________________________________________")

        println(Summary,"\nNeutron single-(quasi)particle states")
        println(Summary,"____________________________________________________________________")
        @printf(Summary, "%4s %4s %4s %8s %14s %14s %14s %14s", "n", "l", "2j", "O_a", "e", "U2", "V2", "E\n")
        for a in 1:a_max
            l_a = Orb[a].l
            j_a = Orb[a].j
            n_a = nn[l_a+1,j_a+1]
            if Orb[a].nO == 1
                O_a = 1.0
            else
                O_a = 0.0
            end
            @printf(Summary, "%4s %4s %4s %8.3f %14.5f %14.3f %14.3f %14.5f\n",n_a, l_a, j_a, O_a, SPE.n[a], U.n[a]^2, V.n[a]^2, SQE.n[a])
            nn[l_a+1,j_a+1] += 1
        end
        println(Summary,"____________________________________________________________________")

    close(Summary)

    return
end

function BCS_export(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,C::O1B,U::O1B,V::O1B,H_N::qpO1B,H_NN::qpO2B)
    # Read parameters
    Output_File = Params.Calc.Path

    # Export HFB canonical basis matrix C ...
    C_Export_Path = "io/" * Output_File * "/BCS/bin/C.bin"
    O1b_export(Params,Orb,C,C_Export_Path)

    # Export HFB amplitudes U & V ... in the canonical basis ...
        # Case of U ...
    U_Export_Path = "io/" * Output_File * "/BCS/bin/U.bin"
    O1b_export(Params,Orb,U,U_Export_Path)

        # Case of V ...
    V_Export_Path = "io/" * Output_File * "/BCS/bin/V.bin"
    O1b_export(Params,Orb,V,V_Export_Path)

    # Export HFB 1-body Hamiltonian H_N ...
    H1B_Export_Path = "io/" * Output_File * "/BCS/bin/qpH1b.bin"
    qpO1b_export(Params,Orb,H_N,H1B_Export_Path)

    # Export HFB 2-body Hamiltonian H_NN ..
    H2B_Export_Path = "io/" * Output_File * "/BCS/bin/qpH2b.bin"
    qpO2b_export(Params,Orb_NN,H_NN,H2B_Export_Path)

    return
end