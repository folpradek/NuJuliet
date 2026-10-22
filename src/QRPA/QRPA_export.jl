function QRPA_export(Params::Parameters,Orb::Vector{Orb1B},Orb_2qp::qpOrb2B,Stability::Bool,E_corr::Float64,Rho::O1B,C::O1B,E_QRPA::Matrix{Vector{ComplexF64}},X_QRPA::Matrix{Matrix{ComplexF64}},Y_QRPA::Matrix{Matrix{ComplexF64}},A_phonon::Matrix{Vector{Bool}},rB_QRPA::ReducedTransition)
    # Export of QRPA summary ...
    @time QRPA_summary(Params,Orb_2qp,Stability,E_corr)

    # Export of QRPA energies & amplitudes in binary format ...
    @time QRPA_binary_export(Params,Orb_2qp,E_QRPA,X_QRPA,Y_QRPA)

    # Export of QRPA spectra ...
    @time QRPA_spectrum_export(Params,Orb_2qp,E_QRPA,A_phonon)

    # Export of QRPA plot-ready spectra ...
    @time QRPA_spectrum_plot_export(Params,Orb_2qp,E_QRPA)

    # Export of QRPA norms of X & Y in human-readable format ...
    @time QRPA_amplitudes_export(Params,Orb_2qp,E_QRPA,X_QRPA,Y_QRPA)

    # Export QRPA particle number values ...
    @time QRPA_particle_number_export(Params,Orb,Rho)

    # Calculate & export radial HF-RPA densities & radii ...
    Summary_File = "io/" * Params.Calc.Path * "/QRPA/QRPA_summary.dat"
    Densities_File = "io/" * Params.Calc.Path * "/QRPA/densities/QRPA_radial_densities.dat"
    @time OBDM_export(Params,Orb,Summary_File,Densities_File,Rho,C)

    # Export QRPA electric transitions ...
    @time QRPA_transitions_export(Params,Orb,Orb_2qp,Rho,E_QRPA,rB_QRPA)
    
    return
end

function QRPA_summary(Params::Parameters,Orb_2qp::qpOrb2B,Stability::Bool,E_corr::Float64)
    # Read parameters ...
    A = Params.Calc.A
    Z = Params.Calc.Z
    hw = Params.Calc.hw
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1
    CMS = Params.Calc.CMS
    Orthogon = Params.Calc.QRPA.Ortho
    Output_File = Params.Calc.Path

    # Set the path for "QRPA_Summary.dat" output file ...
    Output_Path = "io/" * Output_File * "/QRPA/QRPA_summary.dat"

    # Write the summary of QRPA calculation ...
    println("\nPreparing QRPA summary file ...")
    Summary =  open(Output_Path, "w")
        println(Summary, "Spherical Quasiparticle Random-Phase Approximation review:")
        println(Summary, "\nNuclid data:    A = " * string(A) * ", Z = " * string(Z))
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
            println(Summary, "\nOrthogonalization of spurious Goldstone levels (0+, 1-) is included ...")
        else
            println(Summary, "\nOrthogonalization of spurious Goldstone levels (0+, 1-) is NOT included ...")
        end
        println(Summary, "\nStability of the QRPA system:")

        if Stability == true
            println(Summary, "\tQRPA system is STABLE ... no complex eigenvalues detected ...")
        else
            println(Summary, "\tQRPA system is UNSTABLE ... complex eigenvalues detected ...")
        end

        println(Summary, "\nInformation on dimensions of QRPA 1-phonon 2qp subspaces ...")

        N_qp = 0
        @inbounds for J in 0:J_max
            @inbounds for P in 1:2
                n_qp = Orb_2qp.N[P,J+1] * (2*J + 1)
                N_qp += n_qp
            end
        end

        println(Summary, "\nTotal number of 1-phonon quasiparticle states in J-scheme = " * string(sum(Orb_2qp.N)))
        println(Summary, "Total number of 1-phonon quasiparticle states in M-scheme = " * string(N_qp))

        println(Summary, "\nJ\tN_phonon")
        @inbounds for J in 0:J_max
            N_qp_Jp, N_qp_Jm = Orb_2qp.N[1,J+1], Orb_2qp.N[2,J+1]
            Row = string(string(J) * "\t" * string(N_qp_Jp + N_qp_Jm))
            println(Summary, Row)
        end
        println(Summary, "\nSubspace\tP = +")
        @inbounds for J in 0:J_max
            N_qp = Orb_2qp.N[1,J+1]
            Row = string(string(J) * "\t" *string(N_qp))
            println(Summary, Row)
        end
        println(Summary, "\nSubspace\tP = -")
        @inbounds for J in 0:J_max
            N_qp = Orb_2qp.N[2,J+1]
            Row = string(string(J) * "\t" *string(N_qp))
            println(Summary, Row)
        end

        @printf(Summary, "\nE_corr     = %12.6f \t MeV ... QRPA ground-state correlation energy", E_corr)
        @printf(Summary, "\nE_corr / A = %12.6f \t MeV ... QRPA ground-state correlation energy per nuclon\n", E_corr / Float64(A))

    close(Summary)

    println("\nQRPA summary exported ...")

    return
end

function QRPA_binary_export(Params::Parameters,Orb_2qp::qpOrb2B,E_QRPA::Matrix{Vector{ComplexF64}},X_QRPA::Matrix{Matrix{ComplexF64}},Y_QRPA::Matrix{Matrix{ComplexF64}})
    # Read parameters ...
    N_max = Params.Calc.Nmax
    J_max = 2*N_max + 1
    Output_File = Params.Calc.Path

    # Set-up export buffer  ...
    Buffer_size, Buffer_count = 10000000, 0
    Buffer = Vector{ComplexF64}(undef,Buffer_size)

    println("\nPerforming binary export of QRPA energies E & amplitudes X & Y ...")

    # Set the out-put file paths ...
    Output_Path_X = "io/" * Output_File * "/QRPA/bin/QRPA_X.bin"
    Output_Path_Y = "io/" * Output_File * "/QRPA/bin/QRPA_Y.bin"
    Output_Path_E = "io/" * Output_File * "/QRPA/bin/QRPA_E.bin"

    # Binary export of QRPA solutions ...
        # Amplitudes X ...
    open(Output_Path_X, "w") do Export_File
        @inbounds for J in 0:J_max
            @inbounds for P in 1:2
                N_qp = Orb_2qp.N[P,J+1]
                @inbounds for nu in 1:N_qp
                    @inbounds for qp in 1:N_qp
                        X = X_QRPA[P,J+1][qp,nu]
                        Buffer_count += 1
                        Buffer[Buffer_count] = X
                        if Buffer_count == Buffer_size
                            write(Export_File,Buffer)
                            Buffer_count = 0
                        end
                    end
                end
                if Buffer_count > 0
                    write(Export_File,Buffer[1:Buffer_count])
                    Buffer_count = 0
                end
            end
        end
    end
        # Reset the export buffer  ...
    Buffer_count = 0
    Buffer = Vector{ComplexF64}(undef,Buffer_size)
        # Amplitudes Y ...
    open(Output_Path_Y, "w") do Export_File
        @inbounds for J in 0:J_max
            @inbounds for P in 1:2
                N_qp = Orb_2qp.N[P,J+1]
                @inbounds for nu in 1:N_qp
                    @inbounds for qp in 1:N_qp
                        Y = Y_QRPA[P,J+1][qp,nu]
                        Buffer_count += 1
                        Buffer[Buffer_count] = Y
                        if Buffer_count == Buffer_size
                            write(Export_File,Buffer)
                            Buffer_count = 0
                        end
                    end
                end
                if Buffer_count > 0
                    write(Export_File,Buffer[1:Buffer_count])
                    Buffer_count = 0
                end
            end
        end
    end
        # Reset the export buffer  ...
    Buffer_count = 0
    Buffer = Vector{ComplexF64}(undef,Buffer_size)
        # Energies E ...
    open(Output_Path_E, "w") do Export_File
        @inbounds for J in 0:J_max
            @inbounds for P in 1:2
                N_qp = Orb_2qp.N[P,J+1]
                @inbounds for nu in 1:N_qp
                    E = E_QRPA[P,J+1][nu]
                    Buffer_count += 1
                    Buffer[Buffer_count] = E
                    if Buffer_count == Buffer_size
                        write(Export_File,Buffer)
                        Buffer_count = 0
                    end
                end
                if Buffer_count > 0
                    write(Export_File,Buffer[1:Buffer_count])
                    Buffer_count = 0
                end
            end
        end
    end

    println("\tBinary export of QRPA energies E amplitudes X & Y completed ...")

    return
end

function QRPA_spectrum_export(Params::Parameters,Orb_2qp::qpOrb2B,E_QRPA::Matrix{Vector{ComplexF64}},A_phonon::Matrix{Vector{Bool}})
    # Read parameters ...
    N_max = Params.Calc.Nmax
    J_max = 2*N_max + 1
    Output_File = Params.Calc.Path

    # Set the out-put file paths ...
    Output_Path = "io/" * Output_File * "/QRPA/spectra/QRPA.dat"

    println("\nPreparing export of QRPA spectra ...")

    # QRPA spectrum export ...
    open(Output_Path, "w") do Write_File
        @printf(Write_File, "%-5s %-5s %-20s %-20s %-12s\n", "J", "P", "Re E", "Im E", "A-body")
        @inbounds for J in 0:J_max
            @inbounds for P in 1:2
                N_qp = Orb_2qp.N[P,J+1]
                @inbounds for nu in 1:N_qp
                    A_body = "true"

                    if A_phonon[P,J+1][nu] == false
                        A_body = "false"
                    end

                    if P == 1
                        @printf(Write_File, "%-5d %-5s %-20.6f %-20.6f %-12s\n", J, "+", real(E_QRPA[P,J+1][nu]), imag(E_QRPA[P,J+1][nu]), A_body)
                    else
                        @printf(Write_File, "%-5d %-5s %-20.6f %-20.6f %-12s\n", J, "-", real(E_QRPA[P,J+1][nu]), imag(E_QRPA[P,J+1][nu]), A_body)
                    end
                end
                println(Write_File,"\n")
            end
        end
    end

    println("\tExport of QRPA spectra succesfully finished ...")

    return
end

function QRPA_spectrum_plot_export(Params::Parameters,Orb_2qp::qpOrb2B,E_QRPA::Matrix{Vector{ComplexF64}})
    # Read parameters ...
    N_max = Params.Calc.Nmax
    J_max = 2*N_max + 1
    IO = Params.Calc.Path

    # Set the out-put file paths ...
    Output_Path = "io/" * IO * "/QRPA/spectra/QRPA_Plot.dat"

    # Spectra label gap parameter ...
    Delta = 1.0

    # Prepare the QRPA plot-ready spectra for export ...
    println("\nPreparing plot-ready export of QRPA spectra ...")

    # Count the total number of 1-phonon excitations ...
    N_qp = Int64(sum(Orb_2qp.N))

    # Initialite array for solution export ...
    QRPA_export = Matrix{Float64}(undef,4,N_qp+1)

    # 0+ ground state addition ...
    QRPA_export[1,1] = 0.0
    QRPA_export[2,1] = 1.0
    QRPA_export[3,1] = 0.0
    QRPA_export[4,1] = 0.0

    # Allocate QRPA spectrum export array ...
    qp_count = 1
    @inbounds for J in 0:J_max
        @inbounds for P in 1:2
            N_qg = Orb_2qp.N[P,J+1]
            @inbounds for nu in 1:N_qg
                qp_count += 1
                QRPA_export[1,qp_count] = Float64(J)
                QRPA_export[2,qp_count] = Float64(P)
                QRPA_export[3,qp_count] = real(E_QRPA[P,J+1][nu])
                QRPA_export[4,qp_count] = real(E_QRPA[P,J+1][nu])
            end
        end
    end

    # Retrieve QRPA energies for export ...
    E_QRPA_export = @views QRPA_export[3,:]

    # Reorder spectrum ascending in real part of energy ...
    Sort = sortperm(E_QRPA_export, by = x -> real(x))
        # Perform the reordering ...
    QRPA_export .= @views QRPA_export[:,Sort]

    # Adjust the positions of J & P labels ...
    @inbounds for nu in 2:(N_qp+1)
        E_1 = QRPA_export[4,nu-1]
        E_2 = QRPA_export[4,nu]
        if E_2 < E_1
            QRPA_export[4,nu] = E_1 + 1.0
        elseif abs(E_2 - E_1) < Delta
            QRPA_export[4,nu] = E_1 + 1.0
        end
    end

    # QRPA spectrum plot data export ...
    println("\nPerforming the export of plot-ready QRPA spectra ...")
    open(Output_Path, "w") do Write_File
        @printf(Write_File, "%-5s %-5s %-12s %-12s\n", "J", "P", "E", "label")
        @inbounds for nu in 1:N_qp
            J = Int64(round(QRPA_export[1,nu]))
            P = "P"
            if abs(QRPA_export[2,nu] - 1.0) < 1e-3
                P = "+"
            else
                P = "-"
            end
            E = QRPA_export[3,nu]
            E_m = QRPA_export[4,nu]
            @printf(Write_File, "%-5s %-5s %-12.6f %-12.6f\n", J, P, E, E_m)
        end
    end

    println("\nQRPA plot-ready spectra successfully exported ...")

    return
end

function QRPA_amplitudes_export(Params::Parameters,Orb_2qp::qpOrb2B,E_QRPA::Matrix{Vector{ComplexF64}},X_QRPA::Matrix{Matrix{ComplexF64}},Y_QRPA::Matrix{Matrix{ComplexF64}})
    # Read parameters ...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1
    Output_File = Params.Calc.Path

    # Set export paths ...
    Output_Path = "io/" * Output_File * "/QRPA/amplitudes/QRPA_amplitudes_ordered.dat"

    println("\nPreparing export of the norms of the QRPA amplitudes X & Y ...")

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
                N_qp = Orb_2qp.N[P,J+1]
                @inbounds for nu in 1:N_qp
                    x = @views norm(X_QRPA[P,J+1][:,nu])^2
                    y = @views norm(Y_QRPA[P,J+1][:,nu])^2
                    @printf(Write_File, "%-12.8f %-12.8f %-12.8f\n", real(E_QRPA[P,J+1][nu]), x, y)
                end
            end
        end
    end

    # Second, perform an energy ordered export suitable for analysis ...

    # Set export path ...
    Output_Path = "io/" * Output_File * "/QRPA/amplitudes/QRPA_amplitudes.dat"

    # Count the total number of 1-phonon levels ...
    N_qp = sum(Orb_2qp.N)

    # Initialize vectors for export ...
    E_QRPA_ord = Vector{Float64}(undef,N_qp)
    x_QRPA = Vector{Float64}(undef,N_qp)
    y_QRPA = Vector{Float64}(undef,N_qp)
    J_QRPA = Vector{Int64}(undef,N_qp)
    P_QRPA = Vector{Int64}(undef,N_qp)

    # Allocate the QRPA 1-phonon levels ...
    qp_count = 0
    @inbounds for J in 0:J_max
        @inbounds for P in 1:2
            N_qp = Orb_2qp.N[P,J+1]
            @inbounds for nu in 1:N_qp
                qp_count += 1
                E_QRPA_ord[qp_count] = real(E_QRPA[P,J+1][nu])
                xSum = 0.0
                ySum = 0.0
                @inbounds for qp in 1:N_qp
                    x = abs2(X_QRPA[P,J+1][qp,nu])
                    y = abs2(Y_QRPA[P,J+1][qp,nu])
                    ySum += y
                    xSum += x
                end
                x_QRPA[qp_count] = xSum
                y_QRPA[qp_count] = ySum
                J_QRPA[qp_count] = J + 1 
                P_QRPA[qp_count] = P
            end
        end
    end

    # Determine the reordering of QRPA indices ... ascending in energy ...
    Sort_Ind = sortperm(E_QRPA_ord)

    # Perform the reordering of QRPA energies & amplitudes ...
    E_QRPA_ord = E_QRPA_ord[Sort_Ind]
    x_QRPA = x_QRPA[Sort_Ind]
    y_QRPA = y_QRPA[Sort_Ind]
    J_QRPA = J_QRPA[Sort_Ind]
    P_QRPA = P_QRPA[Sort_Ind]

    # Write out the ordered QRPA energies & norms of amplitudes ...
    open(Output_Path, "w") do Write_File
        N_qp = sum(Orb_2qp.N)
        @printf(Write_File, "%-12s %-12s %-12s %-12s %-12s %-12s\n", "Ind", "E", "J", "P", "|X|^2", "|Y|^2")
        @inbounds for nu in 1:N_qp
            P, J = P_QRPA[nu], J_QRPA[nu] - 1
            if P == 1
                @printf(Write_File, "%-12d %-12.6f %-12d %-12s %-12.6f %-12.6f\n", nu, E_QRPA_ord[nu], J, "+", x_QRPA[nu], y_QRPA[nu])
            elseif P == 2
                @printf(Write_File, "%-12d %-12.6f %-12d %-12s %-12.6f %-12.6f\n", nu, E_QRPA_ord[nu], J, "-", x_QRPA[nu], y_QRPA[nu])
            end
        end
    end

    println("\tExport of QRPA norms of X & Y completed ...")

    return
end

function QRPA_particle_number_export(Params::Parameters,Orb::Vector{Orb1B},Rho::O1B)
    # Read parameters ...
    Output_File = Params.Calc.Path
    
    println("\nCalculating the QRPA corrections to particle numbers ...")

    # Evaluate the QRPA corrected particle numbers ...
    Z, N = HFB_particle_number(Params,Rho,Orb)

    println("\tValues of QRPA corrected particle numbers are ...")
    @printf("\t\tdZ = %12.6f\n", Z)
    @printf("\t\tdN = %12.6f\n", N)

    # Set the path for "QRPA_Summary.dat" output file ...
    Output_Path = "io/" * Output_File * "/QRPA/QRPA_summary.dat"

    # Write the particle number vlues into summary file for QRPA calculation ...
    println("\n\tExporting the corrected QRPA particle numbers ...")
    Summary =  open(Output_Path, "a")
        @printf(Summary, "\nQRPA particle number values Z & N:\n")
        @printf(Summary, "\nZ = %12.6f\n", Z)
        @printf(Summary, "N = %12.6f\n", N)
    close(Summary)

    return
end

function QRPA_transitions_export(Params::Parameters,Orb::Vector{Orb1B},Orb_2qp::qpOrb2B,Rho::O1B,E_QRPA::Matrix{Vector{ComplexF64}},rB_QRPA::ReducedTransition)
    # Read parameters ...
    Z = Params.Calc.Z
    Output_File = Params.Calc.Path

    # Define basic constants ...
    hc, pmc2 = 197.326980, 938.272013

    # Import the LHO to reference basis transformation matrix ...
    @time C = O1b_import(Params,Orb,"io/" * Params.Calc.Path * "/" * Params.Calc.QRPA.Reference * "/bin/C.bin")

    # Calculate proton radii moments rN ...
        # Case of QRPA ...
    pR2 = OBDM_rN(Params,2,Rho,C,Orb) 
    pR4 = OBDM_rN(Params,4,Rho,C,Orb) 

    println("\nPreparing export of information on QRPA 1-phonon electromagnetic transitions ...")

    # Set the path for "QRPA_Summary.dat" output file ...
    Output_Path = "io/" * Output_File * "/QRPA/QRPA_summary.dat"

    # Initialize QRPA electromagnetic moments ...
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

    # Calculate the physical Thomas-Reiche-Kuhn sum rule ...
    TRK_E0 = hc^2 / pmc2 / (2.0 * pi) * Float64(Z) * pR2
    TRK_E1 = hc^2 / pmc2 * 9.0 / (8.0 * pi) * Float64(Z)
    TRK_E2 = hc^2 / pmc2 * 50.0 / (8.0 * pi) * Float64(Z) * pR2
    TRK_E3 = hc^2 / pmc2 * 147.0 / (8.0 * pi) * Float64(Z) * pR4

    # Calculate the electromagnetic moments m_k ...
        # E0 moments ...
        J, P = 0, 1
        @inbounds for nu = 1:Orb_2qp.N[P,J+1]
            E, phB, isB, ivB = real(E_QRPA[P,J+1][nu]), rB_QRPA.E0.ph[nu], rB_QRPA.E0.is[nu], rB_QRPA.E0.iv[nu]

            # Case of m_-1 & m_0 ...
            if abs(E) > 1e-3
                phm, ism, ivm = phB / E, isB / E, ivB / E
                m_n1_phE0 += phm
                m_n1_isE0 += ism
                m_n1_ivE0 += ivm

                phm, ism, ivm = phB, isB, ivB
                m_0_phE0 += phm
                m_0_isE0 += ism
                m_0_ivE0 += ivm
            end

            # Case of m_1 ...
            phm, ism, ivm = E * phB, E * isB, E * ivB
            m_1_phE0 += phm
            m_1_isE0 += ism
            m_1_ivE0 += ivm

            # Case of m_2 ...
            phm, ism, ivm = E^2 * phB, E^2 * isB, E^2 * ivB
            m_2_phE0 += phm
            m_2_isE0 += ism
            m_2_ivE0 += ivm

            # Case of m_3 ...
            phm, ism, ivm = E^3 * phB, E^3 * isB, E^3 * ivB
            m_3_phE0 += phm
            m_3_isE0 += ism
            m_3_ivE0 += ivm

        end
        # E1 moments ...
        J, P = 1, 2
        @inbounds for nu = 1:Orb_2qp.N[P,J+1]
            E, phB, isB, ivB = real(E_QRPA[P,J+1][nu]), rB_QRPA.E1.ph[nu], rB_QRPA.E1.is[nu], rB_QRPA.E1.iv[nu]

            # Case of m_-1 & m_0  ...
            if abs(E) > 1e-3
                phm, ism, ivm = phB / E, isB / E, ivB / E
                m_n1_phE1 += phm
                m_n1_isE1 += ism
                m_n1_ivE1 += ivm

                phm, ism, ivm = phB, isB, ivB
                m_0_phE1 += phm
                m_0_isE1 += ism
                m_0_ivE1 += ivm
            end

            # Case of m_1 ...
            phm, ism, ivm = E * phB, E * isB, E * ivB
            m_1_phE1 += phm
            m_1_isE1 += ism
            m_1_ivE1 += ivm

            # Case of m_2 ...
            phm, ism, ivm = E^2 * phB, E^2 * isB, E^2 * ivB
            m_2_phE1 += phm
            m_2_isE1 += ism
            m_2_ivE1 += ivm

            # Case of m_3 ...
            phm, ism, ivm = E^3 * phB, E^3 * isB, E^3 * ivB
            m_3_phE1 += phm
            m_3_isE1 += ism
            m_3_ivE1 += ivm

        end
        # E2 moments ...
        J, P = 2, 1
        @inbounds for nu = 1:Orb_2qp.N[P,J+1]
            E, phB, isB, ivB = real(E_QRPA[P,J+1][nu]), rB_QRPA.E2.ph[nu], rB_QRPA.E2.is[nu], rB_QRPA.E2.iv[nu]

            # Case of m_-1 ...
            if abs(E) > 1e-3
                phm, ism, ivm = phB / E, isB / E, ivB / E
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
            phm, ism, ivm = E * phB, E * isB, E * ivB
            m_1_phE2 += phm
            m_1_isE2 += ism
            m_1_ivE2 += ivm

            # Case of m_2 ...
            phm, ism, ivm = E^2 * phB, E^2 * isB, E^2 * ivB
            m_2_phE2 += phm
            m_2_isE2 += ism
            m_2_ivE2 += ivm

            # Case of m_3 ...
            phm, ism, ivm = E^3 * phB, E^3 * isB, E^3 * ivB
            m_3_phE2 += phm
            m_3_isE2 += ism
            m_3_ivE2 += ivm
        end
        # E3 moments ...
        J, P = 3, 2
        @inbounds for nu = 1:Orb_2qp.N[P,J+1]
            E, phB, isB, ivB = real(E_QRPA[P,J+1][nu]), rB_QRPA.E3.ph[nu], rB_QRPA.E3.is[nu], rB_QRPA.E3.iv[nu]

            # Case of m_-1 ...
            if abs(E) > 1e-3
                phm, ism, ivm = phB / E, isB / E, ivB / E
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
            phm, ism, ivm = E * phB, E * isB, E * ivB
            m_1_phE3 += phm
            m_1_isE3 += ism
            m_1_ivE3 += ivm

            # Case of m_2 ...
            phm, ism, ivm = E^2 * phB, E^2 * isB, E^2 * ivB
            m_2_phE3 += phm
            m_2_isE3 += ism
            m_2_ivE3 += ivm

            # Case of m_3 ...
            phm, ism, ivm = E^3 * phB, E^3 * isB, E^3 * ivB
            m_3_phE3 += phm
            m_3_isE3 += ism
            m_3_ivE3 += ivm
        end

    # Export the moments & TRK sum-rule values to the summary file ...
    Summary =  open(Output_Path, "a")
        println(Summary, "\nReview of QRPA electrogmagnetic transition moments m_k ...")

        println(Summary, "\n\tE0 physical:")
        @printf(Summary, "m _-1 = %20.8f\te^2 fm^4 / MeV\n", m_n1_phE0)
        @printf(Summary, "m _0  = %20.8f\te^2 fm^4\n", m_0_phE0)
        @printf(Summary, "m _1  = %20.8f\te^2 fm^4 MeV\n", m_1_phE0)
        @printf(Summary, "m _2  = %20.8f\te^2 fm^4 MeV^2\n", m_2_phE0)
        @printf(Summary, "m _3  = %20.8f\te^2 fm^4 MeV^3\n", m_3_phE0)

        println(Summary, "\n\tE1 physical:")
        @printf(Summary, "m _-1 = %20.8f\te^2 fm^2 / MeV\n", m_n1_phE1)
        @printf(Summary, "m _0  = %20.8f\te^2 fm^2\n", m_0_phE1)
        @printf(Summary, "m _1  = %20.8f\te^2 fm^2 MeV\n", m_1_phE1)
        @printf(Summary, "m _2  = %20.8f\te^2 fm^2 MeV^2\n", m_2_phE1)
        @printf(Summary, "m _3  = %20.8f\te^2 fm^2 MeV^3\n", m_3_phE1)

        println(Summary, "\n\tE2 physical:")
        @printf(Summary, "m _-1 = %20.8f\te^2 fm^4 / MeV\n", m_n1_phE2)
        @printf(Summary, "m _0  = %20.8f\te^2 fm^4\n", m_0_phE2)
        @printf(Summary, "m _1  = %20.8f\te^2 fm^4 MeV\n", m_1_phE2)
        @printf(Summary, "m _2  = %20.8f\te^2 fm^4 MeV^2\n", m_2_phE2)
        @printf(Summary, "m _3  = %20.8f\te^2 fm^4 MeV^3\n", m_3_phE2)

        println(Summary, "\n\tE3 physical:")
        @printf(Summary, "m _-1 = %20.8f\te^2 fm^6 / MeV\n", m_n1_phE3)
        @printf(Summary, "m _0  = %20.8f\te^2 fm^6\n", m_0_phE3)
        @printf(Summary, "m _1  = %20.8f\te^2 fm^6 MeV\n", m_1_phE3)
        @printf(Summary, "m _2  = %20.8f\te^2 fm^6 MeV^2\n", m_2_phE3)
        @printf(Summary, "m _3  = %20.8f\te^2 fm^6 MeV^3\n", m_3_phE3)



        println(Summary, "\n\tE0 isoscalar:")
        @printf(Summary, "m _-1 = %20.8f\te^2 fm^4 / MeV\n", m_n1_isE0)
        @printf(Summary, "m _0  = %20.8f\te^2 fm^4\n", m_0_isE0)
        @printf(Summary, "m _1  = %20.8f\te^2 fm^4 MeV\n", m_1_isE0)
        @printf(Summary, "m _2  = %20.8f\te^2 fm^4 MeV^2\n", m_2_isE0)
        @printf(Summary, "m _3  = %20.8f\te^2 fm^4 MeV^3\n", m_3_isE0)

        println(Summary, "\n\tE1 isoscalar:")
        @printf(Summary, "m _-1 = %20.8f\te^2 fm^2 / MeV\n", m_n1_isE1)
        @printf(Summary, "m _0  = %20.8f\te^2 fm^2\n", m_0_isE1)
        @printf(Summary, "m _1  = %20.8f\te^2 fm^2 MeV\n", m_1_isE1)
        @printf(Summary, "m _2  = %20.8f\te^2 fm^2 MeV^2\n", m_2_isE1)
        @printf(Summary, "m _3  = %20.8f\te^2 fm^2 MeV^3\n", m_3_isE1)

        println(Summary, "\n\tE2 isoscalar:")
        @printf(Summary, "m _-1 = %20.8f\te^2 fm^4 / MeV\n", m_n1_isE2)
        @printf(Summary, "m _0  = %20.8f\te^2 fm^4\n", m_0_isE2)
        @printf(Summary, "m _1  = %20.8f\te^2 fm^4 MeV\n", m_1_isE2)
        @printf(Summary, "m _2  = %20.8f\te^2 fm^4 MeV^2\n", m_2_isE2)
        @printf(Summary, "m _3  = %20.8f\te^2 fm^4 MeV^3\n", m_3_isE2)

        println(Summary, "\n\tE3 isoscalar:")
        @printf(Summary, "m _-1 = %20.8f\te^2 fm^6 / MeV\n", m_n1_isE3)
        @printf(Summary, "m _0  = %20.8f\te^2 fm^6\n", m_0_isE3)
        @printf(Summary, "m _1  = %20.8f\te^2 fm^6 MeV\n", m_1_isE3)
        @printf(Summary, "m _2  = %20.8f\te^2 fm^6 MeV^2\n", m_2_isE3)
        @printf(Summary, "m _3  = %20.8f\te^2 fm^6 MeV^3\n", m_3_isE3)



        println(Summary, "\n\tE0 isovector:")
        @printf(Summary, "m _-1 = %20.8f\te^2 fm^4 / MeV\n", m_n1_ivE0)
        @printf(Summary, "m _0  = %20.8f\te^2 fm^4\n", m_0_ivE0)
        @printf(Summary, "m _1  = %20.8f\te^2 fm^4 MeV\n", m_1_ivE0)
        @printf(Summary, "m _2  = %20.8f\te^2 fm^4 MeV^2\n", m_2_ivE0)
        @printf(Summary, "m _3  = %20.8f\te^2 fm^4 MeV^3\n", m_3_ivE0)

        println(Summary, "\n\tE1 isovector:")
        @printf(Summary, "m _-1 = %20.8f\te^2 fm^2 / MeV\n", m_n1_ivE1)
        @printf(Summary, "m _0  = %20.8f\te^2 fm^2\n", m_0_ivE1)
        @printf(Summary, "m _1  = %20.8f\te^2 fm^2 MeV\n", m_1_ivE1)
        @printf(Summary, "m _2  = %20.8f\te^2 fm^2 MeV^2\n", m_2_ivE1)
        @printf(Summary, "m _3  = %20.8f\te^2 fm^2 MeV^3\n", m_3_ivE1)

        println(Summary, "\n\tE2 isovector:")
        @printf(Summary, "m _-1 = %20.8f\te^2 fm^4 / MeV\n", m_n1_ivE2)
        @printf(Summary, "m _0  = %20.8f\te^2 fm^4\n", m_0_ivE2)
        @printf(Summary, "m _1  = %20.8f\te^2 fm^4 MeV\n", m_1_ivE2)
        @printf(Summary, "m _2  = %20.8f\te^2 fm^4 MeV^2\n", m_2_ivE2)
        @printf(Summary, "m _3  = %20.8f\te^2 fm^4 MeV^3\n", m_3_ivE2)

        println(Summary, "\n\tE3 isovector:")
        @printf(Summary, "m _-1 = %20.8f\te^2 fm^6 / MeV\n", m_n1_ivE3)
        @printf(Summary, "m _0  = %20.8f\te^2 fm^6\n", m_0_ivE3)
        @printf(Summary, "m _1  = %20.8f\te^2 fm^6 MeV\n", m_1_ivE3)
        @printf(Summary, "m _2  = %20.8f\te^2 fm^6 MeV^2\n", m_2_ivE3)
        @printf(Summary, "m _3  = %20.8f\te^2 fm^6 MeV^3\n", m_3_ivE3)



        println(Summary, "\nReview of QRPA physical Thomas-Reiche-Kuhn (TRK) energy-weighted sum-rule values ...\n")
        @printf(Summary, "S_E0 = %20.8f\te^2 fm^4 Mev\n", TRK_E0)
        @printf(Summary, "S_E1 = %20.8f\te^2 fm^2 Mev\n", TRK_E1)
        @printf(Summary, "S_E2 = %20.8f\te^2 fm^4 Mev\n", TRK_E2)
        @printf(Summary, "S_E3 = %20.8f\te^2 fm^6 Mev\n", TRK_E3)

    close(Summary)

    println("\tPreparing export of QRPA 1-phonon electromagnetic transition intensities ...")

    # Set the path for export of reduced transition intensities B ...
    Output_File = "io/" * Params.Calc.Path

    # QRPA E0 export ...
            # Standard E0 mode ...
        open(Output_File * "/QRPA/transitions/E0/QRPA_E0.dat", "w") do Write_File
            J, P = 0, 1
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for nu = 1:Orb_2qp.N[P,J+1]
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_QRPA[P,J+1][nu]), rB_QRPA.E0.ph[nu], rB_QRPA.E0.is[nu], rB_QRPA.E0.iv[nu])
            end
        end

    # QRPA E1 export ...
            # Standard E1 mode ...
        open(Output_File * "/QRPA/transitions/E1/QRPA_E1.dat", "w") do Write_File
            J, P = 1, 2
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for nu = 1:Orb_2qp.N[P,J+1]
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_QRPA[P,J+1][nu]), rB_QRPA.E1.ph[nu], rB_QRPA.E1.is[nu], rB_QRPA.E1.iv[nu])
            end
        end
            # Full vortical mode ...
        open(Output_File * "/QRPA/transitions/E1/QRPA_E1_V.dat", "w") do Write_File
            J, P = 1, 2
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for nu in 1:Orb_2qp.N[P,J+1]
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_QRPA[P,J+1][nu]), rB_QRPA.E1_V.ph[nu], rB_QRPA.E1_V.is[nu], rB_QRPA.E1_V.iv[nu])
            end
        end
            # Convective vortical mode ...
        open(Output_File * "/QRPA/transitions/E1/QRPA_E1_V_conv.dat", "w") do Write_File
            J, P = 1, 2
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for nu in 1:Orb_2qp.N[P,J+1]
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_QRPA[P,J+1][nu]), rB_QRPA.E1_VC.ph[nu], rB_QRPA.E1_VC.is[nu], rB_QRPA.E1_VC.iv[nu])
            end
        end
            # Spin vortical mode ...
        open(Output_File * "/QRPA/transitions/E1/QRPA_E1_V_spin.dat", "w") do Write_File
            J, P = 1, 2
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for nu in 1:Orb_2qp.N[P,J+1]
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_QRPA[P,J+1][nu]), rB_QRPA.E1_VS.ph[nu], rB_QRPA.E1_VS.is[nu], rB_QRPA.E1_VS.iv[nu])
            end
        end
            # Full toroidal mode ...
        open(Output_File * "/QRPA/transitions/E1/QRPA_E1_T.dat", "w") do Write_File
            J, P = 1, 2
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for nu in 1:Orb_2qp.N[P,J+1]
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_QRPA[P,J+1][nu]), rB_QRPA.E1_T.ph[nu], rB_QRPA.E1_T.is[nu], rB_QRPA.E1_T.iv[nu])
            end
        end
            # Convective toroidal mode ...
        open(Output_File * "/QRPA/transitions/E1/QRPA_E1_T_conv.dat", "w") do Write_File
            J, P = 1, 2
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for nu in 1:Orb_2qp.N[P,J+1]
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_QRPA[P,J+1][nu]), rB_QRPA.E1_TC.ph[nu], rB_QRPA.E1_TC.is[nu], rB_QRPA.E1_TC.iv[nu])
            end
        end
            # Spin toroidal mode ...
        open(Output_File * "/QRPA/transitions/E1/QRPA_E1_T_spin.dat", "w") do Write_File
            J, P = 1, 2
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for nu in 1:Orb_2qp.N[P,J+1]
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_QRPA[P,J+1][nu]), rB_QRPA.E1_TS.ph[nu], rB_QRPA.E1_TS.is[nu], rB_QRPA.E1_TS.iv[nu])
            end
        end
            # Full compression mode ...
        open(Output_File * "/QRPA/transitions/E1/QRPA_E1_C.dat", "w") do Write_File
            J, P = 1, 2
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for nu in 1:Orb_2qp.N[P,J+1]
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_QRPA[P,J+1][nu]), rB_QRPA.E1_C.ph[nu], rB_QRPA.E1_C.is[nu], rB_QRPA.E1_C.iv[nu])
            end
        end
            # Convective compression mode ...
        open(Output_File * "/QRPA/transitions/E1/QRPA_E1_C_conv.dat", "w") do Write_File
            J, P = 1, 2
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for nu in 1:Orb_2qp.N[P,J+1]
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_QRPA[P,J+1][nu]), rB_QRPA.E1_CC.ph[nu], rB_QRPA.E1_CC.is[nu], rB_QRPA.E1_CC.iv[nu])
            end
        end
            # Spin compression mode ...
        open(Output_File * "/QRPA/transitions/E1/QRPA_E1_C_spin.dat", "w") do Write_File
            J, P = 1, 2
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for nu in 1:Orb_2qp.N[P,J+1]
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_QRPA[P,J+1][nu]), rB_QRPA.E1_CS.ph[nu], rB_QRPA.E1_CS.is[nu], rB_QRPA.E1_CS.iv[nu])
            end
        end
            # Continuity equation electric dipole compression mode ...
        open(Output_File * "/QRPA/transitions/E1/QRPA_E1_c.dat", "w") do Write_File
            J, P = 1, 2
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for nu in 1:Orb_2qp.N[P,J+1]
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_QRPA[P,J+1][nu]), rB_QRPA.E1_c.ph[nu], rB_QRPA.E1_c.is[nu], rB_QRPA.E1_c.iv[nu])
            end
        end
            # NLO LWA electric dipole mode ...
        open(Output_File * "/QRPA/transitions/E1/QRPA_E1_NLO_LWA.dat", "w") do Write_File
            J, P = 1, 2
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for nu in 1:Orb_2qp.N[P,J+1]
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_QRPA[P,J+1][nu]), rB_QRPA.E1_NLO_LWA.ph[nu], rB_QRPA.E1_NLO_LWA.is[nu], rB_QRPA.E1_NLO_LWA.iv[nu])
            end
        end


    # QRPA E2 export ...
            # Standard E2 mode ...
        open(Output_File * "/QRPA/transitions/E2/QRPA_E2.dat", "w") do Write_File
            J, P = 2, 1
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for nu = 1:Orb_2qp.N[P,J+1]
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_QRPA[P,J+1][nu]), rB_QRPA.E2.ph[nu], rB_QRPA.E2.is[nu], rB_QRPA.E2.iv[nu])
            end
        end

    # QRPA E3 export ...
            # Standard E3 mode ...
        open(Output_File * "/QRPA/transitions/E3/QRPAE3.dat", "w") do Write_File
            J, P = 3, 2
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for nu = 1:Orb_2qp.N[P,J+1]
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_QRPA[P,J+1][nu]), rB_QRPA.E3.ph[nu], rB_QRPA.E3.is[nu], rB_QRPA.E3.iv[nu])
            end
        end

    println("\t\tExport of information on QRPA 1-phonon electromagnetic transitions completed ...")

    return
end