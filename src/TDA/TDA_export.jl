function TDA_export(Params::Parameters,Orb::Vector{Orb1B},Orb_1p1h::Orb2B_1p1h,
                    E::Matrix{Vector{Float64}},X::Matrix{Matrix{Float64}},rB::ReducedTransition,C::O1B,Rho::O1B)

    # Export of TDA summary ... (no ground-state information, i.e. no energies & radii)
    @time TDA_summary(Params,Orb_1p1h,E)

    # Export of TDA energies & amplitudes in binary format ...
    @time TDA_binary_export(Params,E,X)

    # Export of TDA spectra ...
    @time TDA_spectrum_export(Params,Orb_1p1h,E)

    # Export of TDA plot-ready spectra ...
    @time TDA_plot_spectrum_export(Params,Orb_1p1h,E)

    # Export of TDA amplitudes ...
    @time TDA_amplitudes_export(Params,Orb,Orb_1p1h,E,X)

    # Export of TDA electric transitions ...
    @time TDA_transitions_export(Params,Orb,Orb_1p1h,C,Rho,E,rB)

    return
end

function TDA_summary(Params::Parameters,Orb_1p1h::Orb2B_1p1h,E::Matrix{Vector{Float64}})
    TDA_Params = Params.Calc.TDA
    # Read parameters ...
    A = Params.Calc.A
    Z = Params.Calc.Z
    hw = Params.Calc.hw
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1
    CMS = Params.Calc.CMS
    Orthogon = Params.Calc.TDA.Ortho

    # Set the export paths for summary files ...
    Output_Path_TDA = "io/" * Params.Calc.Path * "/" * Params.Calc.TDA.Name * "/TDA_summary.dat"

    println("\nWriting TDA summary files ...")

    # Count the negative TDA eigenvalues & find the lowest one ... (spurious 1- level excluded for Ortho = true)
    N_neg = 0
    E_min, J_min, P_min = Inf, -1, 0
    @inbounds for J in 0:J_max
        @inbounds for P in 1:2
            @inbounds for (i_nu,E_nu) in enumerate(E[P,J+1])
                if Orthogon == true && J == 1 && P == 2 && i_nu == 1
                    continue
                end
                if E_nu < -1e-8
                    N_neg += 1
                end
                if E_nu < E_min
                    E_min, J_min, P_min = E_nu, J, P
                end
            end
        end
    end

    # Export of TDA summary file ...
    Summary =  open(Output_Path_TDA, "w")
        println(Summary, "Spherical Tamm-Dancoff Approximation review:")
        println(Summary, "\nNuclide data:    A = " * string(A) * ", Z = " * string(Z))
        println(Summary, "\nCalculation data:    hw = " * string(hw) * " , N_max = " * string(N_max) *
                ", s.p. J-basis size = " * string(div((N_max+1)*(N_max+2),2)) * ", s.p. M-basis size = " * string(div((N_max+1)*(N_max+2)*(N_max+3),6)))
        println(Summary, "\t\t\tCenter-of-Mass System correction is set to:   CMS = ''" * string(CMS) * "''")
        if CMS == "CMS1+2B"
            println(Summary, "\n\t\t\tCombined 1-body + 2-body Center-of-Mass System (CMS) motion correction is set ...")
        elseif CMS == "CMS2B"
            println(Summary, "\n\t\t\tOnly pure 2-body Center-of-Mass System (CMS) motion correction is set ...")
        else
            println(Summary, "\n\t\t\tNo Center-of-Mass System (CMS) motion correction is set ...")
        end

        println(Summary, "\nTDA parameters:")
        println(Summary, "\tName = ''" * Params.Calc.TDA.Name * "'',   Reference = ''" * TDA_Params.Reference * "'',   Reference_TBCF = " * string(TDA_Params.Reference_TBCF) * ",   Ortho = " * string(Orthogon))
        println(Summary, "\t1p-1h space tolerances:   Tol_hol = " * string(TDA_Params.Tol_hol) * ",   Tol_par = " * string(TDA_Params.Tol_par))
        if TDA_Params.Reference_TBCF == true
            println(Summary, "\tReference 2-body correlation function is INCLUDED in the matrix A & in the (exact) transition matrix elements ...")
        else
            println(Summary, "\tReference 2-body correlation function is NOT included ...")
        end
        println(Summary, "\tTransition matrix elements ... ''Exact'' with respect to the reference ground state ...")

        if Orthogon == true
            println(Summary, "\nOrthogonalization of spurious center-of-mass levels (1-) is included ...")
        else
            println(Summary, "\nOrthogonalization of spurious center-of-mass levels (1-) is NOT included ...")
        end

        if N_neg == 0
            println(Summary, "\tTDA spectrum is STABLE ... no negative eigenvalues detected ...")
        else
            println(Summary, "\tTDA spectrum is UNSTABLE ... " * string(N_neg) * " negative eigenvalue(s) detected ...")
        end
        if isfinite(E_min)
            @printf(Summary, "\tLowest TDA excitation energy:   E = %12.6f MeV   (J = %d, P = %s)\n", E_min, J_min, P_min == 1 ? "+" : "-")
        end

        println(Summary, "\nInformation on dimensions of 1-phonon 1p-1h TDA subspaces ...")

        N_M_tot = 0
        @inbounds for J in 0:J_max
            @inbounds for P in 1:2
                N_nu = Orb_1p1h.N[P,J+1]
                n = N_nu * (2*J + 1)
                N_M_tot += n
            end
        end

        println(Summary, "\nTotal number of 1 phonon states in J-scheme = " * string(sum(Orb_1p1h.N)))
        println(Summary, "Total number of 1 phonon states in M-scheme = " * string(N_M_tot))

        println(Summary, "\nJ\tN_phonon")
        @inbounds for J in 0:J_max
            N_nu_Jp, N_nu_Jm = Orb_1p1h.N[1,J+1], Orb_1p1h.N[2,J+1]
            Row = string(J) * "\t" * string(N_nu_Jp + N_nu_Jm)
            println(Summary, Row)
        end
        println(Summary, "\nSubspace\tP = +")
        @inbounds for J in 0:J_max
            N_nu = Orb_1p1h.N[1,J+1]
            Row = string(J) * "\t" * string(N_nu)
            println(Summary, Row)
        end
        println(Summary, "\nSubspace\tP = -")
        @inbounds for J in 0:J_max
            N_nu = Orb_1p1h.N[2,J+1]
            Row = string(J) * "\t" * string(N_nu)
            println(Summary, Row)
        end
    close(Summary)

    println("\tTDA summary file exported ...")

    return
end

function TDA_binary_export(Params::Parameters,E::Matrix{Vector{Float64}},X::Matrix{Matrix{Float64}})
    # Read parameters ...
    N_max = Params.Calc.Nmax
    J_max = 2*N_max + 1

    println("\nPerforming binary export of TDA energies E & amplitudes X ...")

    # Set the output file paths ...
    Output_path_X = "io/" * Params.Calc.Path * "/" * Params.Calc.TDA.Name * "/bin/TDA_X.bin"
    Output_path_E = "io/" * Params.Calc.Path * "/" * Params.Calc.TDA.Name * "/bin/TDA_E.bin"

    # Binary export of TDA solutions ... (Float64, same ordering as RPA_X.bin & RPA_E.bin)
        # Amplitudes X ...
    open(Output_path_X, "w") do Export_File
        @inbounds for J in 0:J_max
            @inbounds for P in 1:2
                write(Export_File,permutedims(X[P,J+1]))
            end
        end
    end
        # Energies E ...
    open(Output_path_E, "w") do Export_File
        @inbounds for J in 0:J_max
            @inbounds for P in 1:2
                write(Export_File,E[P,J+1])
            end
        end
    end

    println("\tBinary export of TDA energies E & amplitudes X completed ...")

    return
end

function TDA_spectrum_export(Params::Parameters,Orb_1p1h::Orb2B_1p1h,E::Matrix{Vector{Float64}})
    # Read parameters ...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1

    println("\nPreparing export of TDA spectra ...")

    # Set the output file paths ...
    Output_path = "io/" * Params.Calc.Path * "/" * Params.Calc.TDA.Name * "/spectra/TDA.dat"

    # TDA spectrum export ...
    open(Output_path, "w") do Write_File
        @printf(Write_File, "%-5s %-5s %-20s\n", "J", "P", "E")
        @inbounds for J in 0:J_max
            @inbounds for P in 1:2
                N_nu = Orb_1p1h.N[P,J+1]
                @inbounds for i_nu in 1:N_nu
                    @printf(Write_File, "%-5d %-5s %-20.6f\n", J, P == 1 ? "+" : "-", E[P,J+1][i_nu])
                end
                println(Write_File,"\n")
            end
        end
    end

    println("\tExport of TDA spectra successfully finished ...")

    return
end

function TDA_plot_spectrum_export(Params::Parameters,Orb_1p1h::Orb2B_1p1h,E::Matrix{Vector{Float64}})
    # Read parameters ...
    N_max = Params.Calc.Nmax
    J_max = 2*N_max + 1

    # Prepare the TDA plot-ready spectra for export ...
    println("\nPreparing plot-ready export of TDA solutions ...")

    # Set export paths ...
    Output_path = "io/" * Params.Calc.Path * "/" * Params.Calc.TDA.Name * "/spectra/TDA_plot.dat"

    # Spectra label gap parameter ...
    Delta = 1.0

    # Count the total number of 1-phonon excitations ...
    N_ph = Int64(sum(Orb_1p1h.N))

    # Initialize arrays for solution export ...
    Levels = Matrix{Float64}(undef,4,N_ph+1)

    # 0+ ground state addition ...
    Levels[1,1] = 0.0
    Levels[2,1] = 1.0
    Levels[3,1] = 0.0
    Levels[4,1] = 0.0

    # Allocate TDA spectrum export array ...
    nu_count = 1
    @inbounds for J in 0:J_max
        @inbounds for P in 1:2
            N_nu = Orb_1p1h.N[P,J+1]
            @inbounds for i_nu in 1:N_nu
                nu_count += 1
                Levels[1,nu_count] = Float64(J)
                Levels[2,nu_count] = Float64(P)
                Levels[3,nu_count] = E[P,J+1][i_nu]
                Levels[4,nu_count] = E[P,J+1][i_nu]
            end
        end
    end

    # Reorder spectrum ascending in energy ...
    Sort_TDA = sortperm(Levels[3,:])
    Levels .= Levels[:,Sort_TDA]

    # Adjust the positions of J & P labels ...
    @inbounds for i_nu in 2:(N_ph+1)
        e_1 = Levels[4,i_nu-1]
        e_2 = Levels[4,i_nu]
        if e_2 < e_1
            Levels[4,i_nu] = e_1 + Delta
        elseif abs(e_2 - e_1) < Delta
            Levels[4,i_nu] = e_1 + Delta
        end
    end

    # TDA spectrum plot data export ...
    println("\tPerforming the export of plot-ready TDA spectra ...")
    open(Output_path, "w") do Write_File
        @printf(Write_File, "%-5s %-5s %-12s %-12s\n", "J", "P", "E", "label")
        @inbounds for i_nu in 1:(N_ph+1)
            J = Int64(round(Levels[1,i_nu]))
            P = abs(Levels[2,i_nu] - 1.0) < 1e-3 ? "+" : "-"
            @printf(Write_File, "%-5d %-5s %-12.6f %-12.6f\n", J, P, Levels[3,i_nu], Levels[4,i_nu])
        end
    end
    println("\t\tTDA plot-ready spectra successfully exported ...")

    return
end

function TDA_amplitudes_export(Params::Parameters,Orb::Vector{Orb1B},Orb_1p1h::Orb2B_1p1h,E::Matrix{Vector{Float64}},X::Matrix{Matrix{Float64}})
    # Read parameters ...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1

    println("\nPreparing export of the TDA amplitudes X ...")

    # Count the total number of 1-phonon levels ...
    N_lev = sum(Orb_1p1h.N)

    # Levels ... (E, |X|^2, J, P, dominant 1p-1h component: a_p, a_h, T, weight)
    Levels = Vector{Tuple{Float64,Float64,Int64,Int64,Int64,Int64,Int64,Float64}}(undef,N_lev)

    nu_count = 0
    @inbounds for J in 0:J_max
        @inbounds for P in 1:2
            N_nu = Orb_1p1h.N[P,J+1]
            Orb_ph_list = Orb_1p1h.Ind[P,J+1]
            E_JP, X_JP = E[P,J+1], X[P,J+1]

            # Occupation weights n_h - n_p of 1p-1h orbitals, i.e. the metric of TDA amplitudes: sum_ph (n_h - n_p) * |X|^2 = 1 ...
            dn_JP = Vector{Float64}(undef,N_nu)
            @inbounds for (i_ph,Orb_ph) in enumerate(Orb_ph_list)
                a_p, a_h, T_ph = Orb_ph.a, Orb_ph.b, Orb_ph.T
                if T_ph == -1
                    dn_JP[i_ph] = Orb[a_h].pO - Orb[a_p].pO
                elseif T_ph == 1
                    dn_JP[i_ph] = Orb[a_h].nO - Orb[a_p].nO
                end
            end

            @inbounds for i_nu in 1:N_nu
                nu_count += 1

                x2Sum, w_max, i_max = 0.0, -1.0, 0
                @inbounds for i_ph in 1:N_nu
                    w = dn_JP[i_ph] * abs2(X_JP[i_ph,i_nu])
                    x2Sum += w
                    if w > w_max
                        w_max, i_max = w, i_ph
                    end
                end

                Orb_ph = Orb_ph_list[i_max]
                Levels[nu_count] = (E_JP[i_nu],x2Sum,J,P,Orb_ph.a,Orb_ph.b,Orb_ph.T,w_max)
            end
        end
    end

    # Write out the TDA energies, norms & dominant components ... ordered according to J & P ...
    Output_Path = "io/" * Params.Calc.Path * "/" * Params.Calc.TDA.Name * "/amplitudes/TDA_amplitudes_ordered.dat"
    TDA_amplitudes_write(Output_Path,Levels,Orb)

    # Write out the TDA energies, norms & dominant components ... ascending in energy ...
    Output_Path = "io/" * Params.Calc.Path * "/" * Params.Calc.TDA.Name * "/amplitudes/TDA_amplitudes.dat"
    TDA_amplitudes_write(Output_Path,Levels[sortperm([Lev[1] for Lev in Levels])],Orb)

    println("\tExport of TDA amplitudes X completed ...")

    return
end

function TDA_amplitudes_write(Output_Path::String,Levels::Vector{Tuple{Float64,Float64,Int64,Int64,Int64,Int64,Int64,Float64}},Orb::Vector{Orb1B})
    open(Output_Path, "w") do Write_File
        @printf(Write_File, "%-8s %-14s %-4s %-4s %-12s %-6s %-14s %-14s %-12s\n", "Ind", "E", "J", "P", "|X|^2", "t", "p (n,l,2j)", "h (n,l,2j)", "max weight")
        @inbounds for (i_nu,Lev) in enumerate(Levels)
            (E_nu, x2, J, P, a_p, a_h, T, w) = Lev
            p_str = "(" * string(Orb[a_p].n) * "," * string(Orb[a_p].l) * "," * string(Orb[a_p].j) * ")"
            h_str = "(" * string(Orb[a_h].n) * "," * string(Orb[a_h].l) * "," * string(Orb[a_h].j) * ")"
            @printf(Write_File, "%-8d %-14.6f %-4d %-4s %-12.6f %-6s %-14s %-14s %-12.6f\n", i_nu, E_nu, J, P == 1 ? "+" : "-", x2, T == -1 ? "p" : "n", p_str, h_str, w)
        end
    end

    return
end

function TDA_transitions_export(Params::Parameters,Orb::Vector{Orb1B},Orb_1p1h::Orb2B_1p1h,C::O1B,Rho::O1B,E::Matrix{Vector{Float64}},rB::ReducedTransition)
    # Read parameters ...
    Z = Params.Calc.Z
    Output_File = "io/" * Params.Calc.Path

    # Define basic constants ...
    hc, pmc2 = 197.326980, 938.272013

    # Calculate proton radii moments rN ...
        # Rho is the reference OBDM in the reference basis, as expected by rho_rN ... (radii are only used for the TRK values, not exported)
    pR2 = rho_rN(Params,2,Rho,C,Orb)
    pR4 = rho_rN(Params,4,Rho,C,Orb)

    println("\nPreparing export of information on TDA 1-phonon electromagnetic transitions ...")

    # Set the path for "TDA_summary.dat" output file ...
    Output_path = Output_File * "/" * Params.Calc.TDA.Name * "/TDA_summary.dat"

    # Initialize TDA electromagnetic moments ...
        # Physical moments ...
    m_n1_phE0, m_0_phE0, m_1_phE0, m_2_phE0, m_3_phE0 = 0.0, 0.0, 0.0, 0.0, 0.0
    m_n1_phE1, m_0_phE1, m_1_phE1, m_2_phE1, m_3_phE1 = 0.0, 0.0, 0.0, 0.0, 0.0
    m_n1_phE2, m_0_phE2, m_1_phE2, m_2_phE2, m_3_phE2 = 0.0, 0.0, 0.0, 0.0, 0.0
    m_n1_phE3, m_0_phE3, m_1_phE3, m_2_phE3, m_3_phE3 = 0.0, 0.0, 0.0, 0.0, 0.0
        # Isoscalar moments ...
    m_n1_isE0, m_0_isE0, m_1_isE0, m_2_isE0, m_3_isE0 = 0.0, 0.0, 0.0, 0.0, 0.0
    m_n1_isE1, m_0_isE1, m_1_isE1, m_2_isE1, m_3_isE1 = 0.0, 0.0, 0.0, 0.0, 0.0
    m_n1_isE2, m_0_isE2, m_1_isE2, m_2_isE2, m_3_isE2 = 0.0, 0.0, 0.0, 0.0, 0.0
    m_n1_isE3, m_0_isE3, m_1_isE3, m_2_isE3, m_3_isE3 = 0.0, 0.0, 0.0, 0.0, 0.0
        # Isovector moments ...
    m_n1_ivE0, m_0_ivE0, m_1_ivE0, m_2_ivE0, m_3_ivE0 = 0.0, 0.0, 0.0, 0.0, 0.0
    m_n1_ivE1, m_0_ivE1, m_1_ivE1, m_2_ivE1, m_3_ivE1 = 0.0, 0.0, 0.0, 0.0, 0.0
    m_n1_ivE2, m_0_ivE2, m_1_ivE2, m_2_ivE2, m_3_ivE2 = 0.0, 0.0, 0.0, 0.0, 0.0
    m_n1_ivE3, m_0_ivE3, m_1_ivE3, m_2_ivE3, m_3_ivE3 = 0.0, 0.0, 0.0, 0.0, 0.0

    # Calculate the physical Thomas-Reiche-Kuhn sum rules ...
        # Reference values ... (exactly fulfilled by RPA for a HF reference & local interaction, not by TDA) ...
    TRK_E0 = hc^2 / pmc2 / (2.0 * pi) * Float64(Z) * pR2
    TRK_E1 = hc^2 / pmc2 * 9.0 / (8.0 * pi) * Float64(Z)
    TRK_E2 = hc^2 / pmc2 * 50.0 / (8.0 * pi) * Float64(Z) * pR2
    TRK_E3 = hc^2 / pmc2 * 147.0 / (8.0 * pi) * Float64(Z) * pR4

    # Calculate the electromagnetic moments m_k ...
        # E0 moments ...
        J, P = 0, 1
        N_nu = Orb_1p1h.N[P,J+1]
        @inbounds for i_nu = 1:N_nu
            E_nu, phB, isB, ivB = real(E[P,J+1][i_nu]), rB.E0.ph[i_nu], rB.E0.is[i_nu], rB.E0.iv[i_nu]

            # Case of m_-1 ...
            if abs(E_nu) > 1e-3
                phm, ism, ivm = phB / E_nu, isB / E_nu, ivB / E_nu
                m_n1_phE0 += phm
                m_n1_isE0 += ism
                m_n1_ivE0 += ivm
            end

            # Case of m_0 ...
            phm, ism, ivm = phB, isB, ivB
            m_0_phE0 += phm
            m_0_isE0 += ism
            m_0_ivE0 += ivm

            # Case of m_1 ...
            phm, ism, ivm = E_nu * phB, E_nu * isB, E_nu * ivB
            m_1_phE0 += phm
            m_1_isE0 += ism
            m_1_ivE0 += ivm

            # Case of m_2 ...
            phm, ism, ivm = E_nu^2 * phB, E_nu^2 * isB, E_nu^2 * ivB
            m_2_phE0 += phm
            m_2_isE0 += ism
            m_2_ivE0 += ivm

            # Case of m_3 ...
            phm, ism, ivm = E_nu^3 * phB, E_nu^3 * isB, E_nu^3 * ivB
            m_3_phE0 += phm
            m_3_isE0 += ism
            m_3_ivE0 += ivm

        end

        # E1 moments ...
        J, P = 1, 2
        N_nu = Orb_1p1h.N[P,J+1]
        @inbounds for i_nu = 1:N_nu
            # Skip the spurious 1- Goldstone level, stored as the 1st solution in the case of orthogonalization ...
            if Params.Calc.TDA.Ortho == true && i_nu == 1
                continue
            end

            E_nu, phB, isB, ivB = real(E[P,J+1][i_nu]), rB.E1.ph[i_nu], rB.E1.is[i_nu], rB.E1.iv[i_nu]

            # Case of m_-1 ...
            if abs(E_nu) > 1e-3
                phm, ism, ivm = phB / E_nu, isB / E_nu, ivB / E_nu
                m_n1_phE1 += phm
                m_n1_isE1 += ism
                m_n1_ivE1 += ivm
            end

            # Case of m_0 ...
            phm, ism, ivm = phB, isB, ivB
            m_0_phE1 += phm
            m_0_isE1 += ism
            m_0_ivE1 += ivm

            # Case of m_1 ...
            phm, ism, ivm = E_nu * phB, E_nu * isB, E_nu * ivB
            m_1_phE1 += phm
            m_1_isE1 += ism
            m_1_ivE1 += ivm

            # Case of m_2 ...
            phm, ism, ivm = E_nu^2 * phB, E_nu^2 * isB, E_nu^2 * ivB
            m_2_phE1 += phm
            m_2_isE1 += ism
            m_2_ivE1 += ivm

            # Case of m_3 ...
            phm, ism, ivm = E_nu^3 * phB, E_nu^3 * isB, E_nu^3 * ivB
            m_3_phE1 += phm
            m_3_isE1 += ism
            m_3_ivE1 += ivm

        end

        # E2 moments ...
        J, P = 2, 1
        N_nu = Orb_1p1h.N[P,J+1]
        @inbounds for i_nu = 1:N_nu
            E_nu, phB, isB, ivB = real(E[P,J+1][i_nu]), rB.E2.ph[i_nu], rB.E2.is[i_nu], rB.E2.iv[i_nu]

            # Case of m_-1 ...
            if abs(E_nu) > 1e-3
                phm, ism, ivm = phB / E_nu, isB / E_nu, ivB / E_nu
                m_n1_phE2 += phm
                m_n1_isE2 += ism
                m_n1_ivE2 += ivm
            end

            # Case of m_0 ...
            phm, ism, ivm = phB, isB, ivB
            m_0_phE2 += phm
            m_0_isE2 += ism
            m_0_ivE2 += ivm

            # Case of m_1 ...
            phm, ism, ivm = E_nu * phB, E_nu * isB, E_nu * ivB
            m_1_phE2 += phm
            m_1_isE2 += ism
            m_1_ivE2 += ivm

            # Case of m_2 ...
            phm, ism, ivm = E_nu^2 * phB, E_nu^2 * isB, E_nu^2 * ivB
            m_2_phE2 += phm
            m_2_isE2 += ism
            m_2_ivE2 += ivm

            # Case of m_3 ...
            phm, ism, ivm = E_nu^3 * phB, E_nu^3 * isB, E_nu^3 * ivB
            m_3_phE2 += phm
            m_3_isE2 += ism
            m_3_ivE2 += ivm
        end

        # E3 moments ...
        J, P = 3, 2
        N_nu = Orb_1p1h.N[P,J+1]
        @inbounds for i_nu = 1:N_nu
            E_nu, phB, isB, ivB = real(E[P,J+1][i_nu]), rB.E3.ph[i_nu], rB.E3.is[i_nu], rB.E3.iv[i_nu]

            # Case of m_-1 ...
            if abs(E_nu) > 1e-3
                phm, ism, ivm = phB / E_nu, isB / E_nu, ivB / E_nu
                m_n1_phE3 += phm
                m_n1_isE3 += ism
                m_n1_ivE3 += ivm
            end

            # Case of m_0 ...
            phm, ism, ivm = phB, isB, ivB
            m_0_phE3 += phm
            m_0_isE3 += ism
            m_0_ivE3 += ivm

            # Case of m_1 ...
            phm, ism, ivm = E_nu * phB, E_nu * isB, E_nu * ivB
            m_1_phE3 += phm
            m_1_isE3 += ism
            m_1_ivE3 += ivm

            # Case of m_2 ...
            phm, ism, ivm = E_nu^2 * phB, E_nu^2 * isB, E_nu^2 * ivB
            m_2_phE3 += phm
            m_2_isE3 += ism
            m_2_ivE3 += ivm

            # Case of m_3 ...
            phm, ism, ivm = E_nu^3 * phB, E_nu^3 * isB, E_nu^3 * ivB
            m_3_phE3 += phm
            m_3_isE3 += ism
            m_3_ivE3 += ivm

        end

    # Export the electromagnetic moments & TRK sum-rule values to the summary file ...
    Summary =  open(Output_path, "a")
        println(Summary, "\nReview of TDA electromagnetic transition moments m_k ...")

        println(Summary, "\n\tE0 physical:")
        @printf(Summary, "m_-1  = %20.8f\te^2 fm^4 / MeV\n", m_n1_phE0)
        @printf(Summary, "m_0   = %20.8f\te^2 fm^4\n", m_0_phE0)
        @printf(Summary, "m_1   = %20.8f\te^2 fm^4 MeV\n", m_1_phE0)
        @printf(Summary, "m_2   = %20.8f\te^2 fm^4 MeV^2\n", m_2_phE0)
        @printf(Summary, "m_3   = %20.8f\te^2 fm^4 MeV^3\n", m_3_phE0)

        println(Summary, "\n\tE1 physical:")
        @printf(Summary, "m_-1  = %20.8f\te^2 fm^2 / MeV\n", m_n1_phE1)
        @printf(Summary, "m_0   = %20.8f\te^2 fm^2\n", m_0_phE1)
        @printf(Summary, "m_1   = %20.8f\te^2 fm^2 MeV\n", m_1_phE1)
        @printf(Summary, "m_2   = %20.8f\te^2 fm^2 MeV^2\n", m_2_phE1)
        @printf(Summary, "m_3   = %20.8f\te^2 fm^2 MeV^3\n", m_3_phE1)

        println(Summary, "\n\tE2 physical:")
        @printf(Summary, "m_-1  = %20.8f\te^2 fm^4 / MeV\n", m_n1_phE2)
        @printf(Summary, "m_0   = %20.8f\te^2 fm^4\n", m_0_phE2)
        @printf(Summary, "m_1   = %20.8f\te^2 fm^4 MeV\n", m_1_phE2)
        @printf(Summary, "m_2   = %20.8f\te^2 fm^4 MeV^2\n", m_2_phE2)
        @printf(Summary, "m_3   = %20.8f\te^2 fm^4 MeV^3\n", m_3_phE2)

        println(Summary, "\n\tE3 physical:")
        @printf(Summary, "m_-1  = %20.8f\te^2 fm^6 / MeV\n", m_n1_phE3)
        @printf(Summary, "m_0   = %20.8f\te^2 fm^6\n", m_0_phE3)
        @printf(Summary, "m_1   = %20.8f\te^2 fm^6 MeV\n", m_1_phE3)
        @printf(Summary, "m_2   = %20.8f\te^2 fm^6 MeV^2\n", m_2_phE3)
        @printf(Summary, "m_3   = %20.8f\te^2 fm^6 MeV^3\n", m_3_phE3)



        println(Summary, "\n\tE0 isoscalar:")
        @printf(Summary, "m_-1  = %20.8f\te^2 fm^4 / MeV\n", m_n1_isE0)
        @printf(Summary, "m_0   = %20.8f\te^2 fm^4\n", m_0_isE0)
        @printf(Summary, "m_1   = %20.8f\te^2 fm^4 MeV\n", m_1_isE0)
        @printf(Summary, "m_2   = %20.8f\te^2 fm^4 MeV^2\n", m_2_isE0)
        @printf(Summary, "m_3   = %20.8f\te^2 fm^4 MeV^3\n", m_3_isE0)

        println(Summary, "\n\tE1 isoscalar:")
        @printf(Summary, "m_-1  = %20.8f\te^2 fm^2 / MeV\n", m_n1_isE1)
        @printf(Summary, "m_0   = %20.8f\te^2 fm^2\n", m_0_isE1)
        @printf(Summary, "m_1   = %20.8f\te^2 fm^2 MeV\n", m_1_isE1)
        @printf(Summary, "m_2   = %20.8f\te^2 fm^2 MeV^2\n", m_2_isE1)
        @printf(Summary, "m_3   = %20.8f\te^2 fm^2 MeV^3\n", m_3_isE1)

        println(Summary, "\n\tE2 isoscalar:")
        @printf(Summary, "m_-1  = %20.8f\te^2 fm^4 / MeV\n", m_n1_isE2)
        @printf(Summary, "m_0   = %20.8f\te^2 fm^4\n", m_0_isE2)
        @printf(Summary, "m_1   = %20.8f\te^2 fm^4 MeV\n", m_1_isE2)
        @printf(Summary, "m_2   = %20.8f\te^2 fm^4 MeV^2\n", m_2_isE2)
        @printf(Summary, "m_3   = %20.8f\te^2 fm^4 MeV^3\n", m_3_isE2)

        println(Summary, "\n\tE3 isoscalar:")
        @printf(Summary, "m_-1  = %20.8f\te^2 fm^6 / MeV\n", m_n1_isE3)
        @printf(Summary, "m_0   = %20.8f\te^2 fm^6\n", m_0_isE3)
        @printf(Summary, "m_1   = %20.8f\te^2 fm^6 MeV\n", m_1_isE3)
        @printf(Summary, "m_2   = %20.8f\te^2 fm^6 MeV^2\n", m_2_isE3)
        @printf(Summary, "m_3   = %20.8f\te^2 fm^6 MeV^3\n", m_3_isE3)



        println(Summary, "\n\tE0 isovector:")
        @printf(Summary, "m_-1  = %20.8f\te^2 fm^4 / MeV\n", m_n1_ivE0)
        @printf(Summary, "m_0   = %20.8f\te^2 fm^4\n", m_0_ivE0)
        @printf(Summary, "m_1   = %20.8f\te^2 fm^4 MeV\n", m_1_ivE0)
        @printf(Summary, "m_2   = %20.8f\te^2 fm^4 MeV^2\n", m_2_ivE0)
        @printf(Summary, "m_3   = %20.8f\te^2 fm^4 MeV^3\n", m_3_ivE0)

        println(Summary, "\n\tE1 isovector:")
        @printf(Summary, "m_-1  = %20.8f\te^2 fm^2 / MeV\n", m_n1_ivE1)
        @printf(Summary, "m_0   = %20.8f\te^2 fm^2\n", m_0_ivE1)
        @printf(Summary, "m_1   = %20.8f\te^2 fm^2 MeV\n", m_1_ivE1)
        @printf(Summary, "m_2   = %20.8f\te^2 fm^2 MeV^2\n", m_2_ivE1)
        @printf(Summary, "m_3   = %20.8f\te^2 fm^2 MeV^3\n", m_3_ivE1)

        println(Summary, "\n\tE2 isovector:")
        @printf(Summary, "m_-1  = %20.8f\te^2 fm^4 / MeV\n", m_n1_ivE2)
        @printf(Summary, "m_0   = %20.8f\te^2 fm^4\n", m_0_ivE2)
        @printf(Summary, "m_1   = %20.8f\te^2 fm^4 MeV\n", m_1_ivE2)
        @printf(Summary, "m_2   = %20.8f\te^2 fm^4 MeV^2\n", m_2_ivE2)
        @printf(Summary, "m_3   = %20.8f\te^2 fm^4 MeV^3\n", m_3_ivE2)

        println(Summary, "\n\tE3 isovector:")
        @printf(Summary, "m_-1  = %20.8f\te^2 fm^6 / MeV\n", m_n1_ivE3)
        @printf(Summary, "m_0   = %20.8f\te^2 fm^6\n", m_0_ivE3)
        @printf(Summary, "m_1   = %20.8f\te^2 fm^6 MeV\n", m_1_ivE3)
        @printf(Summary, "m_2   = %20.8f\te^2 fm^6 MeV^2\n", m_2_ivE3)
        @printf(Summary, "m_3   = %20.8f\te^2 fm^6 MeV^3\n", m_3_ivE3)



        println(Summary, "\nPhysical Thomas-Reiche-Kuhn (TRK) energy-weighted sum-rule values (reference values, not fulfilled exactly by TDA) ...\n")
        @printf(Summary, "S_E0 = %20.8f\te^2 fm^4 MeV\n", TRK_E0)
        @printf(Summary, "S_E1 = %20.8f\te^2 fm^2 MeV\n", TRK_E1)
        @printf(Summary, "S_E2 = %20.8f\te^2 fm^4 MeV\n", TRK_E2)
        @printf(Summary, "S_E3 = %20.8f\te^2 fm^6 MeV\n", TRK_E3)

    close(Summary)

    # Export the reduced transition intensities B ... (Output_File = "io/" * Params.Calc.Path)
    println("\nPreparing export of TDA transition intensities ...")
    # TDA E0 export ...
        open(Output_File * "/" * Params.Calc.TDA.Name * "/transitions/E0/TDA_E0.dat", "w") do Write_File
            J, P = 0, 1
            N_nu = Orb_1p1h.N[P,J+1]
            E_JP = E[P,J+1]
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for i_nu = 1:N_nu
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_JP[i_nu]), rB.E0.ph[i_nu], rB.E0.is[i_nu], rB.E0.iv[i_nu])
            end
        end

    # TDA E1 export ...
            # Standard E1 mode ...
        open(Output_File * "/" * Params.Calc.TDA.Name * "/transitions/E1/TDA_E1.dat", "w") do Write_File
            J, P = 1, 2
            N_nu = Orb_1p1h.N[P,J+1]
            E_JP = E[P,J+1]
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for i_nu in 1:N_nu
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_JP[i_nu]), rB.E1.ph[i_nu], rB.E1.is[i_nu], rB.E1.iv[i_nu])
            end
        end
            # Full vortical mode ...
        open(Output_File * "/" * Params.Calc.TDA.Name * "/transitions/E1/TDA_E1_V.dat", "w") do Write_File
            J, P = 1, 2
            N_nu = Orb_1p1h.N[P,J+1]
            E_JP = E[P,J+1]
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for i_nu in 1:N_nu
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_JP[i_nu]), rB.E1_V.ph[i_nu], rB.E1_V.is[i_nu], rB.E1_V.iv[i_nu])
            end
        end
            # Convective vortical mode ...
        open(Output_File * "/" * Params.Calc.TDA.Name * "/transitions/E1/TDA_E1_V_conv.dat", "w") do Write_File
            J, P = 1, 2
            N_nu = Orb_1p1h.N[P,J+1]
            E_JP = E[P,J+1]
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for i_nu in 1:N_nu
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_JP[i_nu]), rB.E1_VC.ph[i_nu], rB.E1_VC.is[i_nu], rB.E1_VC.iv[i_nu])
            end
        end
            # Spin vortical mode ...
        open(Output_File * "/" * Params.Calc.TDA.Name * "/transitions/E1/TDA_E1_V_spin.dat", "w") do Write_File
            J, P = 1, 2
            N_nu = Orb_1p1h.N[P,J+1]
            E_JP = E[P,J+1]
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for i_nu in 1:N_nu
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_JP[i_nu]), rB.E1_VS.ph[i_nu], rB.E1_VS.is[i_nu], rB.E1_VS.iv[i_nu])
            end
        end
            # Full toroidal mode ...
        open(Output_File * "/" * Params.Calc.TDA.Name * "/transitions/E1/TDA_E1_T.dat", "w") do Write_File
            J, P = 1, 2
            N_nu = Orb_1p1h.N[P,J+1]
            E_JP = E[P,J+1]
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for i_nu in 1:N_nu
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_JP[i_nu]), rB.E1_T.ph[i_nu], rB.E1_T.is[i_nu], rB.E1_T.iv[i_nu])
            end
        end
            # Convective toroidal mode ...
        open(Output_File * "/" * Params.Calc.TDA.Name * "/transitions/E1/TDA_E1_T_conv.dat", "w") do Write_File
            J, P = 1, 2
            N_nu = Orb_1p1h.N[P,J+1]
            E_JP = E[P,J+1]
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for i_nu in 1:N_nu
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_JP[i_nu]), rB.E1_TC.ph[i_nu], rB.E1_TC.is[i_nu], rB.E1_TC.iv[i_nu])
            end
        end
            # Spin toroidal mode ...
        open(Output_File * "/" * Params.Calc.TDA.Name * "/transitions/E1/TDA_E1_T_spin.dat", "w") do Write_File
            J, P = 1, 2
            N_nu = Orb_1p1h.N[P,J+1]
            E_JP = E[P,J+1]
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for i_nu in 1:N_nu
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_JP[i_nu]), rB.E1_TS.ph[i_nu], rB.E1_TS.is[i_nu], rB.E1_TS.iv[i_nu])
            end
        end
            # Full compression mode ...
        open(Output_File * "/" * Params.Calc.TDA.Name * "/transitions/E1/TDA_E1_C.dat", "w") do Write_File
            J, P = 1, 2
            N_nu = Orb_1p1h.N[P,J+1]
            E_JP = E[P,J+1]
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for i_nu in 1:N_nu
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_JP[i_nu]), rB.E1_C.ph[i_nu], rB.E1_C.is[i_nu], rB.E1_C.iv[i_nu])
            end
        end
            # Convective compression mode ...
        open(Output_File * "/" * Params.Calc.TDA.Name * "/transitions/E1/TDA_E1_C_conv.dat", "w") do Write_File
            J, P = 1, 2
            N_nu = Orb_1p1h.N[P,J+1]
            E_JP = E[P,J+1]
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for i_nu in 1:N_nu
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_JP[i_nu]), rB.E1_CC.ph[i_nu], rB.E1_CC.is[i_nu], rB.E1_CC.iv[i_nu])
            end
        end
            # Spin compression mode ...
        open(Output_File * "/" * Params.Calc.TDA.Name * "/transitions/E1/TDA_E1_C_spin.dat", "w") do Write_File
            J, P = 1, 2
            N_nu = Orb_1p1h.N[P,J+1]
            E_JP = E[P,J+1]
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for i_nu in 1:N_nu
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_JP[i_nu]), rB.E1_CS.ph[i_nu], rB.E1_CS.is[i_nu], rB.E1_CS.iv[i_nu])
            end
        end
            # Continuity equation electric dipole compression mode ...
        open(Output_File * "/" * Params.Calc.TDA.Name * "/transitions/E1/TDA_E1_c.dat", "w") do Write_File
            J, P = 1, 2
            N_nu = Orb_1p1h.N[P,J+1]
            E_JP = E[P,J+1]
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for i_nu in 1:N_nu
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_JP[i_nu]), rB.E1_c.ph[i_nu], rB.E1_c.is[i_nu], rB.E1_c.iv[i_nu])
            end
        end
            # NLO LWA electric dipole mode ...
        open(Output_File * "/" * Params.Calc.TDA.Name * "/transitions/E1/TDA_E1_NLO_LWA.dat", "w") do Write_File
            J, P = 1, 2
            N_nu = Orb_1p1h.N[P,J+1]
            E_JP = E[P,J+1]
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for i_nu in 1:N_nu
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_JP[i_nu]), rB.E1_NLO_LWA.ph[i_nu], rB.E1_NLO_LWA.is[i_nu], rB.E1_NLO_LWA.iv[i_nu])
            end
        end

    # TDA E2 export ...
        open(Output_File * "/" * Params.Calc.TDA.Name * "/transitions/E2/TDA_E2.dat", "w") do Write_File
            J, P = 2, 1
            N_nu = Orb_1p1h.N[P,J+1]
            E_JP = E[P,J+1]
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for i_nu in 1:N_nu
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_JP[i_nu]), rB.E2.ph[i_nu], rB.E2.is[i_nu], rB.E2.iv[i_nu])
            end
        end

    # TDA E3 export ...
        open(Output_File * "/" * Params.Calc.TDA.Name * "/transitions/E3/TDA_E3.dat", "w") do Write_File
            J, P = 3, 2
            N_nu = Orb_1p1h.N[P,J+1]
            E_JP = E[P,J+1]
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for i_nu in 1:N_nu
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_JP[i_nu]), rB.E3.ph[i_nu], rB.E3.is[i_nu], rB.E3.iv[i_nu])
            end
        end

    println("\tExport of information on TDA 1-phonon electromagnetic transitions completed ...")
    return
end