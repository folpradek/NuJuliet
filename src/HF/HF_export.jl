function HF_summary(Params::Parameters,E_HF::Float64,T_HF::Float64,iteration::Int64)
    # Read parameters ...
    hw = Params.Calc.hw
    A = Params.Calc.A
    Z = Params.Calc.Z
    N_max = Params.Calc.Nmax
    N_2max = Params.Calc.N2max
    N_3max = Params.Calc.N3max
    CMS = Params.Calc.CMS
    Output_File = Params.Calc.Path
    epsilon = Params.Calc.HF.Tol

    # Export the HF calculation summary ...
    println("\nExporting HF calculation summary ...")
    Summary =  open(string("io/", Output_File, "/HF/HF_summary.dat"), "a")
        println(Summary, "Nuclid data:    A = " * string(A) * ", Z = " * string(Z))
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
        println(Summary, "\nSpherical Hartree-Fock solution review:")

        @printf(Summary, "\nNumber of Iterations = %5d, Precision = %.3e \t MeV\n", iteration, epsilon)
        @printf(Summary, "\nE_HF = %15.8f\t MeV \t\t ... \t Mean-field ground state energy\n", E_HF)
        @printf(Summary, "T_HF = %15.8f\t MeV \t\t ... \t Mean-field kinetic energy\n", T_HF)
        @printf(Summary, "\nE_HF / A = %15.8f\t MeV / Nucleon \t\t ... \t Mean-field ground state energy per nucleon\n", E_HF / A)
        @printf(Summary, "T_HF / A = %15.8f\t MeV / Nucleon \t\t ... \t Mean-field total kinetic energy per nucleon\n", T_HF / A)
    close(Summary)

    return
end

function HF_SPS_summary(Params::Parameters,h::O1B,Orb::Vector{Orb1B})
    # Read parameters
    N_max = Params.Calc.Nmax
    N_2max = Params.Calc.N2max
    Output_File = Params.Calc.Path
    a_max = div((N_max + 1)*(N_max + 2),2)
    J_max = N_2max + 1

    # Read the diagonal part of h ...
    pSPE, nSPE = diag(h.p), diag(h.n)

    # Initialize array for radial number n ...
    pn, nn = zeros(Int64,N_max+1,J_max+1), zeros(Int64,N_max+1,J_max+1)

    # Precompute energy ordered s.p. states for print ...
    Sort_pOrbs = sortperm(pSPE, by = x -> real(x))
    Sort_nOrbs = sortperm(nSPE, by = x -> real(x))

    # Export orbitals ...
    println("\nExporting HF calculation summary ...")
    Summary =  open(string("io/", Output_File, "/HF/HF_summary.dat"), "a")
        println(Summary,"\nProton single-particle states")
        println(Summary,"____________________________________________________________________")
        @printf(Summary, "%4s %4s %4s %8s %14s", "n", "l", "2j", "O_a", "E\n")
        @inbounds for Ind_a in 1:a_max
            a = Sort_pOrbs[Ind_a]
            l_a = Orb[a].l
            j_a = Orb[a].j
            n_a = pn[l_a+1,j_a+1]
            O_a = Orb[a].pO
            E_a = pSPE[a]
            @printf(Summary, "%4d %4d %4d %8.3f %14.5f\n",n_a, l_a, j_a, O_a, E_a)
            pn[l_a+1,j_a+1] += 1
        end
        println(Summary,"____________________________________________________________________")

        println(Summary,"\nNeutron single-particle states")
        println(Summary,"____________________________________________________________________")
        @printf(Summary, "%4s %4s %4s %8s %14s", "n", "l", "2j", "O_a", "E\n")
        for Ind_a in 1:a_max
            a = Sort_nOrbs[Ind_a]
            l_a = Orb[a].l
            j_a = Orb[a].j
            n_a = nn[l_a+1,j_a+1]
            O_a = Orb[a].nO
            E_a = nSPE[a]
        @printf(Summary, "%4d %4d %4d %8.3f %14.5f\n",n_a, l_a, j_a, O_a, E_a)
            nn[l_a+1,j_a+1] += 1
        end
        println(Summary,"____________________________________________________________________")

    close(Summary)

    return
end

function HF_export(Params::Parameters,C::O1B,Rho::O1B,h::O1B,Orb::Vector{Orb1B})
    # Read parameters
    Output_file = Params.Calc.Path

    # Export HF singe-particle orbitals ...
    Orb_export_path = "io/" * Output_file * "/HF/bin/orb.bin"
    orbitals_export(Params,Orb,Orb_export_path)

    # Export HF orbitals matrix C ...
    C_export_path = "io/" * Output_file * "/HF/bin/C.bin"
    O1b_export(Params,Orb,C,C_export_path)

    # Export HF density matrix Rho ...
    Rho_export_path = "io/" * Output_file * "/HF/bin/rho.bin"
    O1b_export(Params,Orb,Rho,Rho_export_path)

    # Export HF single-particle Hamiltonian h ...
    h_export_path = "io/" * Output_file * "/HF/bin/H1b.bin"
    O1b_export(Params,Orb,h,h_export_path)

    return
end