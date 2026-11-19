function RPA_export(Params::Parameters,Orb::Vector{Orb1B},Orb_1p1h::Orb2B_1p1h,Orb_NN::Orb2B,Stability::Bool,E_corr::NamedTuple,E::Matrix{Vector{ComplexF64}},
                    X::Matrix{Matrix{ComplexF64}},Y::Matrix{Matrix{ComplexF64}},rB::ReducedTransition,C::O1B,Rho::O1B,Sigma::O2B,h::O1B)

    # Export of RPA summary ...
    @time RPA_summary(Params,Orb_1p1h,E_corr,Stability)

    # Export of RPA energies & amplitudes in binary format ...
    @time RPA_binary_export(Params,Orb_1p1h,E,X,Y)

    # Export of RPA spectra ...
    @time RPA_spectrum_export(Params,Orb_1p1h,E)

    # Export of RPA plot-ready spectra ...
    @time RPA_plot_spectrum_export(Params,Orb_1p1h,E)

    # Export of RPA |X|^2 & |Y|^2 amplitudes ...
    @time RPA_amplitudes_export(Params,Orb,Orb_1p1h,E,X,Y)

    # Calculate & export radial RPA densities & radii ...
    Summary_File = "io/" * Params.Calc.Path * "/" * Params.Calc.RPA.Name * "/RPA_summary.dat"
    Densities_File = "io/" * Params.Calc.Path * "/" * Params.Calc.RPA.Name * "/densities/" * Params.Calc.RPA.Name * "_radial_densities.dat"
    @time rho_export(Params,Orb,Summary_File,Densities_File,Rho,C)

    # Export RPA single-particle orbitals ...
    @time C_NAT = RPA_orbitals_export(Params,Orb,C,Rho,h)

    # Export RPA 2-body correlation function Sigma ... RPA basis, or the natural-orbital basis for RPA_Orb = true ...
    if Params.Calc.RPA.TBCF == true
        Sigma_export_path = "io/" * Params.Calc.Path * "/" * Params.Calc.RPA.Name * "/bin/sigma.bin"
        @time O2b_export(Params,Orb_NN,Sigma,Sigma_export_path)

        # rm
        #=
            # Sigma is given in the RPA basis ... it is already the natural-orbital basis if C_NAT = 1 (HF-ScRPA, or a diagonal OBDM such as RPA_OBDM_oY2),
            # otherwise (off-diagonal OBDM, e.g. RPA_OBDM_sigma, or sign changes of the natural orbitals) it is transformed to match C.bin ...
            C_NAT_identity = (maximum(abs.(C_NAT.p - I)) < 1e-12) && (maximum(abs.(C_NAT.n - I)) < 1e-12)

            if Params.Calc.RPA.RPA_Orb == true && !C_NAT_identity
                # Transformation RPA basis -> natural orbitals ... (O2b_transformation works in place, hence the copy)
                @time Sigma_NAT = O2b_transformation(Params,Orb,Orb_NN,deepcopy(Sigma),C_NAT,Report = false)
                @time O2b_export(Params,Orb_NN,Sigma_NAT,Sigma_export_path)
            else
                @time O2b_export(Params,Orb_NN,Sigma,Sigma_export_path)
            end
        =#
    end

    # Export of RPA electric transitions ...
    @time RPA_transitions_export(Params,Orb,Orb_1p1h,C,Rho,E,rB)

    return
end

function RPA_orbitals_export(Params::Parameters,Orb::Vector{Orb1B},C::O1B,Rho::O1B,h::O1B)
    # Read parameters ...
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1)*(N_max + 2),2)

    println("\nPreparing export of RPA single-particle orbitals ...")

    # Set the export paths ...
    Summary_File = "io/" * Params.Calc.Path * "/" * Params.Calc.RPA.Name * "/RPA_summary.dat"
    Occupation_File = "io/" * Params.Calc.Path * "/" * Params.Calc.RPA.Name * "/densities/RPA_occupation.dat"

    # OBDM in the RPA basis ... Rho_RPA = C' * Rho * C ...
    Rho_RPA = O1b_transformation(Rho,C)

    # Natural orbitals of the OBDM ... transformation RPA basis -> natural orbitals ...
    _, C_NAT, _ = OBDM_natural_orbitals(Params,Orb,Rho_RPA)

    # Occupations & mean-field energy centroids in the natural orbitals ...
    Rho_NAT = O1b_transformation(Rho_RPA,C_NAT)
    h_NAT = O1b_transformation(h,C_NAT)

    pO, nO = diag(Rho_NAT.p), diag(Rho_NAT.n)
    pE, nE = diag(h_NAT.p), diag(h_NAT.n)

    # Reference orbitals & occupations ... (Orb carries the ScRPA occupations in HF_ScRPA_solver)
    Orb_ref = orbitals_import(Params,"io/" * Params.Calc.Path * "/" * Params.Calc.RPA.Reference * "/bin/orb.bin")

    # Export into the summary file ... proton & neutron single-particle states separately, energy ordered (as in HF_SPS_summary) ...
    Summary = open(Summary_File, "a")
        println(Summary, "\nRPA single-particle orbitals ... natural orbitals of the RPA OBDM, E = <a|h|a> centroids of the mean-field Hamiltonian h ...")

        for (Label, O_t, E_t) in (("Proton", pO, pE), ("Neutron", nO, nE))
            # Radial number n counted in the energy order within each (l,j) block ...
            n_count = zeros(Int64,N_max+1,2*N_max+2)

            println(Summary, "\n" * Label * " single-particle states")
            println(Summary, "____________________________________________________________________")
            @printf(Summary, "%4s %4s %4s %10s %14s\n", "n", "l", "2j", "O_a", "E")
            @inbounds for a in sortperm(E_t)
                l_a, j_a = Orb[a].l, Orb[a].j
                @printf(Summary, "%4d %4d %4d %10.6f %14.6f\n", n_count[l_a+1,j_a+1], l_a, j_a, O_t[a], E_t[a])
                n_count[l_a+1,j_a+1] += 1
            end
            println(Summary, "____________________________________________________________________")
            @printf(Summary, "%-14s %15.6f\n", "Particle number ...", sum(hat(Orb[a].j) * O_t[a] for a in 1:a_max))
        end
    close(Summary)

    # Export of occupations & depletions with respect to the reference ...
    open(Occupation_File, "w") do Write_File
        @printf(Write_File, "%-6s %-6s %-6s %-6s %-12s %-12s %-12s %-12s %-12s %-12s\n", "a", "n", "l", "2j", "pO_ref", "pO_RPA", "pO_dep", "nO_ref", "nO_RPA", "nO_dep")
        @inbounds for a in 1:a_max
            @printf(Write_File, "%-6d %-6d %-6d %-6d %-12.6f %-12.6f %-12.6f %-12.6f %-12.6f %-12.6f\n", a, Orb[a].n, Orb[a].l, Orb[a].j,
                    Orb_ref[a].pO, pO[a], pO[a] - Orb_ref[a].pO, Orb_ref[a].nO, nO[a], nO[a] - Orb_ref[a].nO)
        end
    end

    # Binary export of the RPA orbitals, basis & OBDM ...
    if Params.Calc.RPA.RPA_Orb == true
        # RPA natural orbitals as a new reference basis ... orb.bin, C.bin, rho.bin & H1b.bin in the natural-orbital basis ...
        RPA_natural_orbitals_export(Params,Orb,C,Rho_RPA,C_NAT,h)
    else
        # Export RPA single-particle orbitals ... (occupations as carried by Orb)
        Orb_export_path = "io/" * Params.Calc.Path * "/" * Params.Calc.RPA.Name * "/bin/orb.bin"
        orbitals_export(Params,Orb,Orb_export_path)

        # Export RPA single-particle basis orbitals ... LHO -> RPA basis ...
        C_export_path = "io/" * Params.Calc.Path * "/" * Params.Calc.RPA.Name * "/bin/C.bin"
        O1b_export(Params,Orb,C,C_export_path)

        # Export RPA 1-body density matrix ... RPA basis ...
        Rho_export_path = "io/" * Params.Calc.Path * "/" * Params.Calc.RPA.Name * "/bin/rho.bin"
        O1b_export(Params,Orb,Rho_RPA,Rho_export_path)
    end

    println("\tExport of RPA single-particle orbitals finished ...")

    return C_NAT
end

function RPA_natural_orbitals_export(Params::Parameters,Orb::Vector{Orb1B},C::O1B,Rho_RPA::O1B,C_NAT::O1B,h::O1B)
    # Set the export path ...
    Path = "io/" * Params.Calc.Path * "/" * Params.Calc.RPA.Name * "/bin/"

    # OBDM, occupations & mean-field Hamiltonian in the natural-orbital basis ...
    Rho_NAT = O1b_diagonal(O1b_transformation(Rho_RPA,C_NAT))
    h_NAT = O1b_transformation(h,C_NAT)
    Orb_NAT = orbitals_update(deepcopy(Orb),Rho_NAT)

    # Transformation LHO -> RPA natural orbitals ...
    C_LHO_NAT = O1B(C.p * C_NAT.p, C.n * C_NAT.n)

    # Export ...
    orbitals_export(Params,Orb_NAT,Path * "orb.bin")
    O1b_export(Params,Orb_NAT,C_LHO_NAT,Path * "C.bin")
    O1b_export(Params,Orb_NAT,Rho_NAT,Path * "rho.bin")
    O1b_export(Params,Orb_NAT,h_NAT,Path * "H1b.bin")

    return
end

function RPA_summary(Params::Parameters,Orb_1p1h::Orb2B_1p1h,E_corr::NamedTuple,Stability::Bool)
    # Read parameters ...
    A = Params.Calc.A
    Z = Params.Calc.Z
    hw = Params.Calc.hw
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1
    CMS = Params.Calc.CMS
    Orthogon = Params.Calc.RPA.Ortho
    Output_File = Params.Calc.Path

    # Set the export paths for summary files ...
    Output_Path_RPA = "io/" * Output_File * "/" * Params.Calc.RPA.Name * "/RPA_summary.dat"

    println("\nWriting RPA summary files ...")

    # Export of RPA summary file ...
    Summary =  open(Output_Path_RPA, "w")
        println(Summary, "Spherical Random-Phase Approximation review:")
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
        if Orthogon == true
            println(Summary, "\nOrthogonalization of spurious Goldstone levels (1-) is included ...")
        else
            println(Summary, "\nOrthogonalization of spurious Goldstone levels (1-) is NOT included ...")
        end

        if Stability == true
            println(Summary, "\tRPA system is STABLE ... no complex eigenvalues detected ...")
        else
            println(Summary, "\tRPA system is UNSTABLE ... complex eigenvalues detected ...")
        end

        println(Summary, "\nInformation on dimensions of 1-phonon 1p-1h RPA subspaces ...")

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

        # RPA correlation energy ... the two total values (components see RPA_energy) ...
        println(Summary, "\n\nRPA ground-state correlation energy ...")
        @printf(Summary, "\nE_corr     = %12.6f \t MeV ... RPA ground-state correlation energy (quasi-boson)", E_corr.E_QBA)
        if !isnan(E_corr.E_dens)
            @printf(Summary, "\nE_corr     = %12.6f \t MeV ... RPA ground-state correlation energy (density)", E_corr.E_dens)
        end
        println(Summary)
    close(Summary)

    println("\tRPA summary file exported ...")

    return
end

function RPA_binary_export(Params::Parameters,Orb_1p1h::Orb2B_1p1h,E::Matrix{Vector{ComplexF64}},X::Matrix{Matrix{ComplexF64}},Y::Matrix{Matrix{ComplexF64}})
    # Read parameters ...
    N_max = Params.Calc.Nmax
    J_max = 2*N_max + 1
    Output_File = Params.Calc.Path

    println("\nPerforming binary export of RPA energies E & amplitudes X & Y ...")

    # Set the output file paths ...
    Output_path_X = "io/" * Output_File * "/" * Params.Calc.RPA.Name * "/bin/RPA_X.bin"
    Output_path_Y = "io/" * Output_File * "/" * Params.Calc.RPA.Name * "/bin/RPA_Y.bin"
    Output_path_E = "io/" * Output_File * "/" * Params.Calc.RPA.Name * "/bin/RPA_E.bin"

    # Binary export of RPA solutions ...
        # Amplitudes X ...
    open(Output_path_X, "w") do Export_File
        @inbounds for J in 0:J_max
            @inbounds for P in 1:2
                write(Export_File,permutedims(X[P,J+1]))
            end
        end
    end
        # Amplitudes Y ...
    open(Output_path_Y, "w") do Export_File
        @inbounds for J in 0:J_max
            @inbounds for P in 1:2
                write(Export_File,permutedims(Y[P,J+1]))
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

    println("\tBinary export of RPA energies E, amplitudes X & Y completed ...")

    return
end

function RPA_spectrum_export(Params::Parameters,Orb_1p1h::Orb2B_1p1h,E::Matrix{Vector{ComplexF64}})
    # Read parameters ...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1
    Output_File = Params.Calc.Path

    println("\nPreparing export of RPA spectra ...")

    # Set the output file paths ...
    Output_path = "io/" * Output_File * "/" * Params.Calc.RPA.Name * "/spectra/RPA.dat"

    # RPA spectrum export ...
    open(Output_path, "w") do Write_File
        @printf(Write_File, "%-5s %-5s %-20s %-20s\n", "J", "P", "Re E", "Im E")
        @inbounds for J in 0:J_max
            @inbounds for P in 1:2
                N_nu = Orb_1p1h.N[P,J+1]
                @inbounds for i_nu in 1:N_nu
                    if P == 1
                        @printf(Write_File, "%-5d %-5s %-20.6f %-20.6f\n", J, "+", real(E[P,J+1][i_nu]), imag(E[P,J+1][i_nu]))
                    else
                        @printf(Write_File, "%-5d %-5s %-20.6f %-20.6f\n", J, "-", real(E[P,J+1][i_nu]), imag(E[P,J+1][i_nu]))
                    end
                end
                println(Write_File,"\n")
            end
        end
    end

    println("\tExport of RPA spectra successfully finished ...")

    return
end

function RPA_plot_spectrum_export(Params::Parameters,Orb_1p1h::Orb2B_1p1h,E::Matrix{Vector{ComplexF64}})
    # Read parameters ...
    N_max = Params.Calc.Nmax
    J_max = 2*N_max + 1
    Output_File = Params.Calc.Path

    # Prepare the RPA plot-ready spectra for export ...
    println("\nPreparing plot-ready export of RPA solutions ...")

    # Set export paths ...
    Output_path = "io/" * Output_File * "/" * Params.Calc.RPA.Name * "/spectra/RPA_plot.dat"

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

    # Allocate RPA spectrum export array ...
    nu_count = 1
    @inbounds for J in 0:J_max
        @inbounds for P in 1:2
            N_nu = Orb_1p1h.N[P,J+1]
            @inbounds for i_nu in 1:N_nu
                nu_count += 1
                Levels[1,nu_count] = Float64(J)
                Levels[2,nu_count] = Float64(P)
                Levels[3,nu_count] = real(E[P,J+1][i_nu])
                Levels[4,nu_count] = real(E[P,J+1][i_nu])
            end
        end
    end

    # Retrieve RPA energies for export ...
    E_RPA_full = @views Levels[3,:]

    # Reorder spectrum ascending in real part of energy ...
    Sort_RPA = sortperm(E_RPA_full)
        # Perform the reordering ...
    Levels .= @views Levels[:,Sort_RPA]

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

    # RPA spectrum plot data export ...
    println("\tPerforming the export of plot-ready RPA spectra ...")
    open(Output_path, "w") do Write_File
        @printf(Write_File, "%-5s %-5s %-12s %-12s\n", "J", "P", "E", "label")
        @inbounds for i_nu in 1:(N_ph+1)
            J = Int64(round(Levels[1,i_nu]))
            P = "P"
            if abs(Levels[2,i_nu] - 1.0) < 1e-3
                P = "+"
            else
                P = "-"
            end
            E_nu = Levels[3,i_nu]
            E_m = Levels[4,i_nu]
            @printf(Write_File, "%-5d %-5s %-12.6f %-12.6f\n", J, P, E_nu, E_m)
        end
    end
    println("\t\tRPA plot-ready spectra successfully exported ...")

    return
end

function RPA_amplitudes_export(Params::Parameters,Orb::Vector{Orb1B},Orb_1p1h::Orb2B_1p1h,E::Matrix{Vector{ComplexF64}},X::Matrix{Matrix{ComplexF64}},Y::Matrix{Matrix{ComplexF64}})
    # Read parameters ...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1
    Output_File = Params.Calc.Path

    # Set export paths ...
    Output_Path = "io/" * Output_File * "/" * Params.Calc.RPA.Name * "/amplitudes/RPA_amplitudes_ordered.dat"

    println("\nPreparing export norms of the RPA amplitudes X & Y ...")

    # First a systematic export according to J & P numbers ...
    open(Output_Path, "w") do Write_File
        @printf(Write_File, "%-12s %-12s %-12s\n", "E", "|X|^2", "|Y|^2")
        @inbounds for J in 0:J_max
            @inbounds for P in 1:2
                if P == 1
                    println(Write_File, "\nJ = " * string(J) * ",\tP = +")
                else
                    println(Write_File, "\nJ = " * string(J) * ",\tP = -")
                end
                N_nu = Orb_1p1h.N[P,J+1]
                E_JP, X_JP, Y_JP = E[P,J+1], X[P,J+1], Y[P,J+1]

                # Occupation weights n_h - n_p of 1p-1h orbitals, i.e. the metric of RPA amplitudes: sum_ph (n_h - n_p) * (|X|^2 - |Y|^2) = 1 ...
                dn_JP = Vector{Float64}(undef,N_nu)
                @inbounds for (i_ph,Orb_ph) in enumerate(Orb_1p1h.Ind[P,J+1])
                    a_p, a_h, T_ph = Orb_ph.a, Orb_ph.b, Orb_ph.T
                    if T_ph == -1
                        dn_JP[i_ph] = Orb[a_h].pO - Orb[a_p].pO
                    elseif T_ph == 1
                        dn_JP[i_ph] = Orb[a_h].nO - Orb[a_p].nO
                    end
                end

                @inbounds for i_nu in 1:N_nu
                    x = @views sum(dn_JP .* abs2.(X_JP[:,i_nu]))
                    y = @views sum(dn_JP .* abs2.(Y_JP[:,i_nu]))
                    @printf(Write_File, "%-12.8f %-12.8f %-12.8f\n", real(E_JP[i_nu]), x, y)
                end
            end
        end
    end

    # Second, perform an energy ordered export suitable for analysis ...

    # Set export path ...
    Output_Path = "io/" * Output_File * "/" * Params.Calc.RPA.Name * "/amplitudes/RPA_amplitudes.dat"

    # Count the total number of 1-phonon levels ...
    N_lev = sum(Orb_1p1h.N)

    # Initialize vectors for export ...
    Levels = Vector{Tuple{Float64,Float64,Float64,Int64,Int64}}(undef,N_lev)
    Levels_E = Vector{Float64}(undef,N_lev)

    # Allocate the RPA 1-phonon levels ...
    nu_count = 0
    @inbounds for J in 0:J_max
        @inbounds for P in 1:2
            N_nu = Orb_1p1h.N[P,J+1]
            E_JP, X_JP, Y_JP = E[P,J+1], X[P,J+1], Y[P,J+1]

            # Occupation weights n_h - n_p of 1p-1h orbitals, i.e. the metric of RPA amplitudes: sum_ph (n_h - n_p) * (|X|^2 - |Y|^2) = 1 ...
            dn_JP = Vector{Float64}(undef,N_nu)
            @inbounds for (i_ph,Orb_ph) in enumerate(Orb_1p1h.Ind[P,J+1])
                a_p, a_h, T_ph = Orb_ph.a, Orb_ph.b, Orb_ph.T
                if T_ph == -1
                    dn_JP[i_ph] = Orb[a_h].pO - Orb[a_p].pO
                elseif T_ph == 1
                    dn_JP[i_ph] = Orb[a_h].nO - Orb[a_p].nO
                end
            end

            @inbounds for i_nu in 1:N_nu
                nu_count += 1

                E_nu = real(E_JP[i_nu])

                x2Sum = 0.0
                y2Sum = 0.0
                @inbounds for i_ph in 1:N_nu
                    x2 = dn_JP[i_ph] * abs2(X_JP[i_ph,i_nu])
                    y2 = dn_JP[i_ph] * abs2(Y_JP[i_ph,i_nu])
                    y2Sum += y2
                    x2Sum += x2
                end
                Levels[nu_count] = (E_nu,x2Sum,y2Sum,J,P)
                Levels_E[nu_count] = E_nu
            end
        end
    end

    # Determine the reordering of RPA indices ... ascending in energy ...
    sort_ind = sortperm(Levels_E)

    # Perform the reordering of RPA energies & amplitudes ...
    Levels = Levels[sort_ind]

    # Write out the ordered RPA energies & norms of amplitudes ...
    open(Output_Path, "w") do Write_File
        @printf(Write_File, "%-12s %-12s %-12s %-12s %-12s %-12s\n", "Ind", "E", "J", "P", "|X|^2", "|Y|^2")
        @inbounds for i_nu in 1:N_lev
            (E_nu, x2, y2, J, P) = Levels[i_nu]
            if P == 1
                @printf(Write_File, "%-12d %-12.6f %-12d %-12s %-12.6f %-12.6f\n", i_nu, E_nu, J, "+", x2, y2)
            elseif P == 2
                @printf(Write_File, "%-12d %-12.6f %-12d %-12s %-12.6f %-12.6f\n", i_nu, E_nu, J, "-", x2, y2)
            end
        end
    end

    println("\tExport of RPA norms of X & Y completed ...")

    return
end

function RPA_transitions_export(Params::Parameters,Orb::Vector{Orb1B},Orb_1p1h::Orb2B_1p1h,C::O1B,Rho::O1B,E::Matrix{Vector{ComplexF64}},rB::ReducedTransition)
    # Read parameters ...
    Z = Params.Calc.Z
    Output_File = "io/" * Params.Calc.Path

    # Define basic constants ...
    hc, pmc2 = 197.326980, 938.272013

    # Calculate proton radii moments rN ...
        # rho_rN expects Rho in the RPA basis, while the given Rho is in the LHO basis ... Rho_RPA = C' * Rho * C ...
    Rho_RPA = O1b_transformation(Rho,C)
    pR2 = rho_rN(Params,2,Rho_RPA,C,Orb)
    pR4 = rho_rN(Params,4,Rho_RPA,C,Orb)

    println("\nPreparing export of information on RPA 1-phonon electromagnetic transitions ...")

    # Set the path for "RPA_Summary.dat" output file ...
    Output_path = Output_File * "/" * Params.Calc.RPA.Name * "/RPA_summary.dat"

    # Initialize RPA electromagnetic moments ...
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
        # Case of RPA ...
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
            if Params.Calc.RPA.Ortho == true && i_nu == 1
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
        println(Summary, "\nReview of RPA electromagnetic transition moments m_k ...")

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



        println(Summary, "\nReview of RPA physical Thomas-Reiche-Kuhn (TRK) energy-weighted sum-rule values ...\n")
        @printf(Summary, "S_E0 = %20.8f\te^2 fm^4 MeV\n", TRK_E0)
        @printf(Summary, "S_E1 = %20.8f\te^2 fm^2 MeV\n", TRK_E1)
        @printf(Summary, "S_E2 = %20.8f\te^2 fm^4 MeV\n", TRK_E2)
        @printf(Summary, "S_E3 = %20.8f\te^2 fm^6 MeV\n", TRK_E3)

    close(Summary)

    # Export the reduced transition intensities B ... (Output_File = "io/" * Params.Calc.Path)
    println("\nPreparing export of RPA transition intensities ...")
    # RPA E0 export ...
        open(Output_File * "/" * Params.Calc.RPA.Name * "/transitions/E0/RPA_E0.dat", "w") do Write_File
            J, P = 0, 1
            N_nu = Orb_1p1h.N[P,J+1]
            E_JP = E[P,J+1]
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for i_nu = 1:N_nu
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_JP[i_nu]), rB.E0.ph[i_nu], rB.E0.is[i_nu], rB.E0.iv[i_nu])
            end
        end

    # RPA E1 export ...
            # Standard E1 mode ...
        open(Output_File * "/" * Params.Calc.RPA.Name * "/transitions/E1/RPA_E1.dat", "w") do Write_File
            J, P = 1, 2
            N_nu = Orb_1p1h.N[P,J+1]
            E_JP = E[P,J+1]
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for i_nu in 1:N_nu
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_JP[i_nu]), rB.E1.ph[i_nu], rB.E1.is[i_nu], rB.E1.iv[i_nu])
            end
        end
            # Full vortical mode ...
        open(Output_File * "/" * Params.Calc.RPA.Name * "/transitions/E1/RPA_E1_V.dat", "w") do Write_File
            J, P = 1, 2
            N_nu = Orb_1p1h.N[P,J+1]
            E_JP = E[P,J+1]
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for i_nu in 1:N_nu
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_JP[i_nu]), rB.E1_V.ph[i_nu], rB.E1_V.is[i_nu], rB.E1_V.iv[i_nu])
            end
        end
            # Convective vortical mode ...
        open(Output_File * "/" * Params.Calc.RPA.Name * "/transitions/E1/RPA_E1_V_conv.dat", "w") do Write_File
            J, P = 1, 2
            N_nu = Orb_1p1h.N[P,J+1]
            E_JP = E[P,J+1]
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for i_nu in 1:N_nu
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_JP[i_nu]), rB.E1_VC.ph[i_nu], rB.E1_VC.is[i_nu], rB.E1_VC.iv[i_nu])
            end
        end
            # Spin vortical mode ...
        open(Output_File * "/" * Params.Calc.RPA.Name * "/transitions/E1/RPA_E1_V_spin.dat", "w") do Write_File
            J, P = 1, 2
            N_nu = Orb_1p1h.N[P,J+1]
            E_JP = E[P,J+1]
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for i_nu in 1:N_nu
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_JP[i_nu]), rB.E1_VS.ph[i_nu], rB.E1_VS.is[i_nu], rB.E1_VS.iv[i_nu])
            end
        end
            # Full toroidal mode ...
        open(Output_File * "/" * Params.Calc.RPA.Name * "/transitions/E1/RPA_E1_T.dat", "w") do Write_File
            J, P = 1, 2
            N_nu = Orb_1p1h.N[P,J+1]
            E_JP = E[P,J+1]
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for i_nu in 1:N_nu
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_JP[i_nu]), rB.E1_T.ph[i_nu], rB.E1_T.is[i_nu], rB.E1_T.iv[i_nu])
            end
        end
            # Convective toroidal mode ...
        open(Output_File * "/" * Params.Calc.RPA.Name * "/transitions/E1/RPA_E1_T_conv.dat", "w") do Write_File
            J, P = 1, 2
            N_nu = Orb_1p1h.N[P,J+1]
            E_JP = E[P,J+1]
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for i_nu in 1:N_nu
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_JP[i_nu]), rB.E1_TC.ph[i_nu], rB.E1_TC.is[i_nu], rB.E1_TC.iv[i_nu])
            end
        end
            # Spin toroidal mode ...
        open(Output_File * "/" * Params.Calc.RPA.Name * "/transitions/E1/RPA_E1_T_spin.dat", "w") do Write_File
            J, P = 1, 2
            N_nu = Orb_1p1h.N[P,J+1]
            E_JP = E[P,J+1]
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for i_nu in 1:N_nu
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_JP[i_nu]), rB.E1_TS.ph[i_nu], rB.E1_TS.is[i_nu], rB.E1_TS.iv[i_nu])
            end
        end
            # Full compression mode ...
        open(Output_File * "/" * Params.Calc.RPA.Name * "/transitions/E1/RPA_E1_C.dat", "w") do Write_File
            J, P = 1, 2
            N_nu = Orb_1p1h.N[P,J+1]
            E_JP = E[P,J+1]
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for i_nu in 1:N_nu
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_JP[i_nu]), rB.E1_C.ph[i_nu], rB.E1_C.is[i_nu], rB.E1_C.iv[i_nu])
            end
        end
            # Convective compression mode ...
        open(Output_File * "/" * Params.Calc.RPA.Name * "/transitions/E1/RPA_E1_C_conv.dat", "w") do Write_File
            J, P = 1, 2
            N_nu = Orb_1p1h.N[P,J+1]
            E_JP = E[P,J+1]
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for i_nu in 1:N_nu
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_JP[i_nu]), rB.E1_CC.ph[i_nu], rB.E1_CC.is[i_nu], rB.E1_CC.iv[i_nu])
            end
        end
            # Spin compression mode ...
        open(Output_File * "/" * Params.Calc.RPA.Name * "/transitions/E1/RPA_E1_C_spin.dat", "w") do Write_File
            J, P = 1, 2
            N_nu = Orb_1p1h.N[P,J+1]
            E_JP = E[P,J+1]
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for i_nu in 1:N_nu
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_JP[i_nu]), rB.E1_CS.ph[i_nu], rB.E1_CS.is[i_nu], rB.E1_CS.iv[i_nu])
            end
        end
            # Continuity equation electric dipole compression mode ...
        open(Output_File * "/" * Params.Calc.RPA.Name * "/transitions/E1/RPA_E1_c.dat", "w") do Write_File
            J, P = 1, 2
            N_nu = Orb_1p1h.N[P,J+1]
            E_JP = E[P,J+1]
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for i_nu in 1:N_nu
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_JP[i_nu]), rB.E1_c.ph[i_nu], rB.E1_c.is[i_nu], rB.E1_c.iv[i_nu])
            end
        end
            # NLO LWA electric dipole mode ...
        open(Output_File * "/" * Params.Calc.RPA.Name * "/transitions/E1/RPA_E1_NLO_LWA.dat", "w") do Write_File
            J, P = 1, 2
            N_nu = Orb_1p1h.N[P,J+1]
            E_JP = E[P,J+1]
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for i_nu in 1:N_nu
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_JP[i_nu]), rB.E1_NLO_LWA.ph[i_nu], rB.E1_NLO_LWA.is[i_nu], rB.E1_NLO_LWA.iv[i_nu])
            end
        end

    # RPA E2 export ...
        open(Output_File * "/" * Params.Calc.RPA.Name * "/transitions/E2/RPA_E2.dat", "w") do Write_File
            J, P = 2, 1
            N_nu = Orb_1p1h.N[P,J+1]
            E_JP = E[P,J+1]
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for i_nu in 1:N_nu
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_JP[i_nu]), rB.E2.ph[i_nu], rB.E2.is[i_nu], rB.E2.iv[i_nu])
            end
        end

    # RPA E3 export ...
        open(Output_File * "/" * Params.Calc.RPA.Name * "/transitions/E3/RPA_E3.dat", "w") do Write_File
            J, P = 3, 2
            N_nu = Orb_1p1h.N[P,J+1]
            E_JP = E[P,J+1]
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for i_nu in 1:N_nu
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_JP[i_nu]), rB.E3.ph[i_nu], rB.E3.is[i_nu], rB.E3.iv[i_nu])
            end
        end

    println("\tExport of information on RPA 1-phonon electromagnetic transitions completed ...")
    return
end