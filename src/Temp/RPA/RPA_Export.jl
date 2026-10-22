function RPA_export(Params::Parameters,Orb::Vector{Orb1B},N_nu::Matrix{Int64},E_corr::Float64,E_RPA::Matrix{Vector{ComplexF64}},
                    X_RPA::Matrix{Matrix{ComplexF64}},Y_RPA::Matrix{Matrix{ComplexF64}},rB_RPA::ReducedTransition,C::O1B,Rho_RPA::O1B,Stability::Bool)

    # Export of RPA summary ...
    @time RPA_summary(Params,N_nu,E_corr,Stability)

    # Export of RPA energies & amplitudes in binary format ...
    @time RPA_binary_export(Params,N_nu,E_RPA,X_RPA,Y_RPA)

    # Export of RPA spectra ...
    @time RPA_spectrum_export(Params,N_nu,E_RPA)

    # Export of RPA plot-ready spectra ...
    @time RPA_plot_spectrum_export(Params,N_nu,E_RPA)

    # Export of RPA |X|^2 & |Y|^2 amplitudes, not plot ready ...
    @time RPA_amplitudes_export(Params,N_nu,E_RPA,X_RPA,Y_RPA)

    # Calculate & export radial RPA densities & radii ...
    Summary_File = "IO/" * Params.Calc.Path * "/RPA/RPA_Summary.dat"
    Densities_File = "IO/" * Params.Calc.Path * "/RPA/Densities/RPA_Radial_Densities.dat"
    @time OBDM_export(Params,Orb,Summary_File,Densities_File,Rho_RPA,C)

    # Export of RPA electric transitions ...
    @time RPA_transitions_export(Params,Orb,N_nu,C,Rho_RPA,E_RPA,rB_RPA)

    return
end

function RPA_summary(Params::Parameters,N_nu::Matrix{Int64},E_corr::Float64,Stability::Bool)
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
    Output_Path_RPA = "IO/" * Output_File * "/RPA/RPA_Summary.dat"

    println("\nWritting RPA summary files ...")

    # Export of RPA summary file ...
    Summary =  open(Output_Path_RPA, "w")
        println(Summary, "Spherical Random-Phase Approximation review:")
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
                n = N_nu[J+1,P] * (2*J + 1)
                N_M_tot += n
            end
        end

        println(Summary, "\nTotal number of 1 phonon states in J-scheme = " * string(sum(N_nu)))
        println(Summary, "Total number of 1 phonon states in M-scheme = " * string(N_M_tot))

        println(Summary, "\nJ\tN_phonon")
        @inbounds for J in 0:J_max
            N_qp_Jp, N_qp_Jm = N_nu[J+1,1], N_nu[J+1,2]
            Row = string(string(J) * "\t" * string(N_qp_Jp + N_qp_Jm))
            println(Summary, Row)
        end
        println(Summary, "\nSubspace\tP = +")
        @inbounds for J in 0:J_max
            N_qp = N_nu[J+1,1]
            Row = string(string(J) * "\t" *string(N_qp))
            println(Summary, Row)
        end
        println(Summary, "\nSubspace\tP = -")
        @inbounds for J in 0:J_max
            N_qp = N_nu[J+1,2]
            Row = string(string(J) * "\t" *string(N_qp))
            println(Summary, Row)
        end

        @printf(Summary, "\nE_corr     = %12.6f \t MeV ... RPA ground-state correlation energy", E_corr)
        @printf(Summary, "\nE_corr / A = %12.6f \t MeV ... RPA ground-state correlation energy per nuclon\n", E_corr / Float64(A))
        
    close(Summary)

    println("\tRPA summary file exported ...")

    return
end

function RPA_binary_export(Params::Parameters,N_nu::Matrix{Int64},E_RPA::Matrix{Vector{ComplexF64}},X_RPA::Matrix{Matrix{ComplexF64}},Y_RPA::Matrix{Matrix{ComplexF64}})
    # Read parameters ...
    N_max = Params.Calc.Nmax
    J_max = 2*N_max + 1
    Output_File = Params.Calc.Path

    # Set-up export buffer  ...
    Buffer_size, Buffer_count = 10000000, 0
    Buffer_RPA = Vector{ComplexF64}(undef,Buffer_size)

    println("\nPerforming binary export of RPA energies E & amplitudes X & Y ...")

    # Set the out-put file paths ...
        # Case of RPA ...
    Output_Path_RPA_X = "IO/" * Output_File * "/Bin/RPA_X.bin"
    Output_Path_RPA_Y = "IO/" * Output_File * "/Bin/RPA_Y.bin"
    Output_Path_RPA_E = "IO/" * Output_File * "/Bin/RPA_E.bin"

    # Binary export of RPA solutions ...
        # Amplitudes X ...
    open(Output_Path_RPA_X, "w") do Export_File
        @inbounds for J in 0:J_max
            @inbounds for P in 1:2
                N_ph = N_nu[J+1,P]
                @inbounds for nu in 1:N_ph
                    @inbounds for ph in 1:N_ph
                        X = X_RPA[J+1,P][ph,nu]
                        Buffer_count += 1
                        Buffer_RPA[Buffer_count] = X
                        if Buffer_count == Buffer_size
                            write(Export_File,Buffer_RPA)
                            Buffer_count = 0
                        end
                    end
                end
                if Buffer_count > 0
                    write(Export_File,Buffer_RPA[1:Buffer_count])
                    Buffer_count = 0
                end
            end
        end
    end
        # Reset the RPA export buffer ...
    Buffer_count = 0
    Buffer_RPA = Vector{ComplexF64}(undef,Buffer_size)
        # Amplitudes Y ...
    open(Output_Path_RPA_Y, "w") do Export_File
        @inbounds for J in 0:J_max
            @inbounds for P in 1:2
                N_ph = N_nu[J+1,P]
                @inbounds for nu in 1:N_ph
                    @inbounds for ph in 1:N_ph
                        Y = Y_RPA[J+1,P][ph,nu]
                        Buffer_count += 1
                        Buffer_RPA[Buffer_count] = Y
                        if Buffer_count == Buffer_size
                            write(Export_File,Buffer_RPA)
                            Buffer_count = 0
                        end
                    end
                end
                if Buffer_count > 0
                    write(Export_File,Buffer_RPA[1:Buffer_count])
                    Buffer_count = 0
                end
            end
        end
    end
        # Reset the RPA export buffer ...
    Buffer_count = 0
    Buffer_RPA = Vector{ComplexF64}(undef,Buffer_size)
        # Energies E ...
    open(Output_Path_RPA_E, "w") do Export_File
        @inbounds for J in 0:J_max
            @inbounds for P in 1:2
                N_ph = N_nu[J+1,P]
                @inbounds for nu in 1:N_ph
                    E = E_RPA[J+1,P][nu]
                    Buffer_count += 1
                    Buffer_RPA[Buffer_count] = E
                    if Buffer_count == Buffer_size
                        write(Export_File,Buffer_RPA)
                        Buffer_count = 0
                    end
                end
                if Buffer_count > 0
                    write(Export_File,Buffer_RPA[1:Buffer_count])
                    Buffer_count = 0
                end
            end
        end
    end

    println("\tBinary export of RPA energies E amplitudes X & Y completed ...")

    return
end

function RPA_spectrum_export(Params::Parameters,N_nu::Matrix{Int64},E_RPA::Matrix{Vector{ComplexF64}})
    # Read parameters ...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1
    Output_File = Params.Calc.Path

    println("\nPreparing export of RPA spectra ...")

    # Set the out-put file paths ...
    Output_Path_RPA = "IO/" * Output_File * "/RPA/Spectra/RPA.dat"

    # RPA spectrum export ...
    open(Output_Path_RPA, "w") do Write_File
        @printf(Write_File, "%-5s %-5s %-20s %-20s\n", "J", "P", "Re E", "Im E")
        @inbounds for J in 0:J_max
            @inbounds for P in 1:2
                N_ph = N_nu[J+1,P]
                @inbounds for nu in 1:N_ph
                    if P == 1
                        @printf(Write_File, "%-5d %-5s %-20.6f %-20.6f\n", J, "+", real(E_RPA[J+1,P][nu]), imag(E_RPA[J+1,P][nu]))
                    else
                        @printf(Write_File, "%-5d %-5s %-20.6f %-20.6f\n", J, "-", real(E_RPA[J+1,P][nu]), imag(E_RPA[J+1,P][nu]))
                    end
                end
                println(Write_File,"\n")
            end
        end
    end

    println("\tExport of RPA spectra succesfully finished ...")

    return
end

function RPA_plot_spectrum_export(Params::Parameters,N_nu::Matrix{Int64},E_RPA::Matrix{Vector{ComplexF64}})
    # Read parameters ...
    N_max = Params.Calc.Nmax
    J_max = 2*N_max + 1
    Output_File = Params.Calc.Path

    # Prepare the RPA plot-ready spectra for export ...
    println("\nPreparing plot-ready export of RPA solutions ...")

    # Set export paths ...
    Output_Path_RPA = "IO/" * Output_File * "/RPA/Spectra/RPA_Plot.dat"

    # Spectra label gap parameter ...
    Delta = 1.0

    # Count the total number of 1-phonon excitations ...
    N_ph = Int64(sum(N_nu))

    # Initialite arrays for solution export ...
    RPA_Solution = Matrix{Float64}(undef,4,N_ph+1)

    # 0+ ground state addition ...
    RPA_Solution[1,1] = 0.0
    RPA_Solution[2,1] = 1.0
    RPA_Solution[3,1] = 0.0
    RPA_Solution[4,1] = 0.0

    # Allocate RPA spectrum export array ...
    nu_count = 1
    @inbounds for J in 0:J_max
        @inbounds for P in 1:2
            N_ph_JP = N_nu[J+1,P]
            @inbounds for nu in 1:N_ph_JP
                nu_count += 1
                RPA_Solution[1,nu_count] = Float64(J)
                RPA_Solution[2,nu_count] = Float64(P)
                RPA_Solution[3,nu_count] = real(E_RPA[J+1,P][nu])
                RPA_Solution[4,nu_count] = real(E_RPA[J+1,P][nu])
            end
        end
    end

    # Retrieve RPA energies for export ...
    E_RPA_full = @views RPA_Solution[3,:]

    # Reorder spectrum ascending in real part of energy ...
    Sort_RPA = sortperm(E_RPA_full, by = x -> real(x))
        # Perform the reordering ...
    RPA_Solution .= @views RPA_Solution[:,Sort_RPA]

    # Adjust the positions of J & P labels ...
    @inbounds for nu in 2:(N_ph+1)
        e_RPA_1 = RPA_Solution[4,nu-1]
        e_RPA_2 = RPA_Solution[4,nu]
        if e_RPA_2 < e_RPA_1
            RPA_Solution[4,nu] = e_RPA_1 + 1.0
        elseif abs(e_RPA_2 - e_RPA_1) < Delta
            RPA_Solution[4,nu] = e_RPA_1 + 1.0
        end
    end

    # RPA spectrum plot data export ...
    println("\tPerforming the export of plot-ready RPA spectra ...")
    open(Output_Path_RPA, "w") do Write_File
        @printf(Write_File, "%-5s %-5s %-12s %-12s\n", "J", "P", "E", "label")
        @inbounds for nu in 1:N_ph
            J = Int64(round(RPA_Solution[1,nu]))
            P = "P"
            if abs(RPA_Solution[2,nu] - 1.0) < 1e-3
                P = "+"
            else
                P = "-"
            end
            E = RPA_Solution[3,nu]
            E_m = RPA_Solution[4,nu]
            @printf(Write_File, "%-5d %-5s %-12.6f %-12.6f\n", J, P, E, E_m)
        end
    end
    println("\t\tRPA plot-ready spectra successfully exported ...")

    return
end

function RPA_amplitudes_export(Params::Parameters,N_nu::Matrix{Int64},E_RPA::Matrix{Vector{ComplexF64}},X_RPA::Matrix{Matrix{ComplexF64}},Y_RPA::Matrix{Matrix{ComplexF64}})
    # Read parameters ...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1
    Output_File = Params.Calc.Path

    # Set export paths ...
    Output_Path = "IO/" * Output_File * "/RPA/Amplitudes/RPA_Amplitudes_Ordered.dat"

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
                N_ph = N_nu[J+1,P]
                @inbounds for nu in 1:N_ph
                    x = @views norm(X_RPA[J+1,P][:,nu])^2
                    y = @views norm(Y_RPA[J+1,P][:,nu])^2
                    @printf(Write_File, "%-12.8f %-12.8f %-12.8f\n", real(E_RPA[J+1,P][nu]), x, y)
                end
            end
        end
    end

    # Second, perform an energy ordered export suitable for analysis ...

    # Set export path ...
    Output_Path = "IO/" * Output_File * "/RPA/Amplitudes/RPA_Amplitudes.dat"

    # Count the total number of 1-phonon levels ...
    N_ph = sum(N_nu)

    # Initialize vectors for export ...
    E_RPA_ord = Vector{Float64}(undef,N_ph)
    x_RPA = Vector{Float64}(undef,N_ph)
    y_RPA = Vector{Float64}(undef,N_ph)
    J_RPA = Vector{Int64}(undef,N_ph)
    P_RPA = Vector{Int64}(undef,N_ph)

    # Allocate the RPA 1-phonon levels ...
    ph_count = 0
    @inbounds for J in 0:J_max
        @inbounds for P in 1:2
            N_ph = N_nu[J+1,P]
            @inbounds for nu in 1:N_ph
                ph_count += 1
                E_RPA_ord[ph_count] = real(E_RPA[J+1,P][nu])
                xSum = 0.0
                ySum = 0.0
                @inbounds for ph in 1:N_ph
                    x = abs2(X_RPA[J+1,P][ph,nu])
                    y = abs2(Y_RPA[J+1,P][ph,nu])
                    ySum += y
                    xSum += x
                end
                x_RPA[ph_count] = xSum
                y_RPA[ph_count] = ySum
                J_RPA[ph_count] = J + 1 
                P_RPA[ph_count] = P
            end
        end
    end

    # Determine the reordering of RPA indices ... ascending in energy ...
    Sort_Ind = sortperm(E_RPA_ord)

    # Perform the reordering of RPA energies & amplitudes ...
    E_RPA_ord = E_RPA_ord[Sort_Ind]
    x_RPA = x_RPA[Sort_Ind]
    y_RPA = y_RPA[Sort_Ind]
    J_RPA = J_RPA[Sort_Ind]
    P_RPA = P_RPA[Sort_Ind]

    # Write out the ordered RPA energies & norms of amplitudes ...
    open(Output_Path, "w") do Write_File
        N_ph = sum(N_nu)
        @printf(Write_File, "%-12s %-12s %-12s %-12s %-12s %-12s\n", "Ind", "E", "J", "P", "|X|^2", "|Y|^2")
        @inbounds for nu in 1:N_ph
            P, J = P_RPA[nu], J_RPA[nu] - 1
            if P == 1
                @printf(Write_File, "%-12d %-12.6f %-12d %-12s %-12.6f %-12.6f\n", nu, E_RPA_ord[nu], J, "+", x_RPA[nu], y_RPA[nu])
            elseif P == 2
                @printf(Write_File, "%-12d %-12.6f %-12d %-12s %-12.6f %-12.6f\n", nu, E_RPA_ord[nu], J, "-", x_RPA[nu], y_RPA[nu])
            end
        end
    end

    println("\tExport of RPA norms of X & Y completed ...")

    return
end

function RPA_transitions_export(Params::Parameters,Orb::Vector{Orb1B},N_nu::Matrix{Int64},C::O1B,Rho_RPA::O1B,E_RPA::Matrix{Vector{ComplexF64}},rB_RPA::ReducedTransition)
    # Read parameters ...
    Z = Params.Calc.Z
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1)*(N_max + 2),2)
    Output_File = Params.Calc.Path

    # Define basic constants ...
    hc, pmc2 = 197.326980, 938.272013

    # Calculate proton radii moments rN ...
        # Case of RPA ...
    pR2_RPA = OBDM_rN(Params,2,Rho_RPA,C,Orb) 
    pR4_RPA = OBDM_rN(Params,4,Rho_RPA,C,Orb)

    println("\nPreparing export of information on RPA 1-phonon electromagnetic transitions ...")


    # Set the path for "RPA_Summary.dat" output file ...
    Output_Path_RPA = "IO/" * Output_File * "/RPA/RPA_Summary.dat"

    # Initialize RPA electromagnetic moments ...
        # Physical moments ...
    m_n1_phE0_RPA, m_0_phE0_RPA, m_1_phE0_RPA, m_2_phE0_RPA, m_3_phE0_RPA = 0.0, 0.0, 0.0, 0.0, 0.0
    m_n1_phE1_RPA, m_0_phE1_RPA, m_1_phE1_RPA, m_2_phE1_RPA, m_3_phE1_RPA = 0.0, 0.0, 0.0, 0.0, 0.0
    m_n1_phE2_RPA, m_0_phE2_RPA, m_1_phE2_RPA, m_2_phE2_RPA, m_3_phE2_RPA = 0.0, 0.0, 0.0, 0.0, 0.0
    m_n1_phE3_RPA, m_0_phE3_RPA, m_1_phE3_RPA, m_2_phE3_RPA, m_3_phE3_RPA = 0.0, 0.0, 0.0, 0.0, 0.0
        # Isoscalar moments ...
    m_n1_isE0_RPA, m_0_isE0_RPA, m_1_isE0_RPA, m_2_isE0_RPA, m_3_isE0_RPA = 0.0, 0.0, 0.0, 0.0, 0.0
    m_n1_isE1_RPA, m_0_isE1_RPA, m_1_isE1_RPA, m_2_isE1_RPA, m_3_isE1_RPA = 0.0, 0.0, 0.0, 0.0, 0.0
    m_n1_isE2_RPA, m_0_isE2_RPA, m_1_isE2_RPA, m_2_isE2_RPA, m_3_isE2_RPA = 0.0, 0.0, 0.0, 0.0, 0.0
    m_n1_isE3_RPA, m_0_isE3_RPA, m_1_isE3_RPA, m_2_isE3_RPA, m_3_isE3_RPA = 0.0, 0.0, 0.0, 0.0, 0.0
        # Isovector moments ...
    m_n1_ivE0_RPA, m_0_ivE0_RPA, m_1_ivE0_RPA, m_2_ivE0_RPA, m_3_ivE0_RPA = 0.0, 0.0, 0.0, 0.0, 0.0
    m_n1_ivE1_RPA, m_0_ivE1_RPA, m_1_ivE1_RPA, m_2_ivE1_RPA, m_3_ivE1_RPA = 0.0, 0.0, 0.0, 0.0, 0.0
    m_n1_ivE2_RPA, m_0_ivE2_RPA, m_1_ivE2_RPA, m_2_ivE2_RPA, m_3_ivE2_RPA = 0.0, 0.0, 0.0, 0.0, 0.0
    m_n1_ivE3_RPA, m_0_ivE3_RPA, m_1_ivE3_RPA, m_2_ivE3_RPA, m_3_ivE3_RPA = 0.0, 0.0, 0.0, 0.0, 0.0

    # Calculate the physical Thomas-Reiche-Kuhn sum rules ...
        # Case of RPA ...
    TRK_E0_RPA = hc^2 / pmc2 / (2.0 * pi) * Float64(Z) * pR2_RPA
    TRK_E1_RPA = hc^2 / pmc2 * 9.0 / (8.0 * pi) * Float64(Z)
    TRK_E2_RPA = hc^2 / pmc2 * 50.0 / (8.0 * pi) * Float64(Z) * pR2_RPA
    TRK_E3_RPA = hc^2 / pmc2 * 147.0 / (8.0 * pi) * Float64(Z) * pR4_RPA

    # Calculate the electromagnetic moments m_k ...
        # E0 moments ...
        J, P = 0, 1
        @inbounds for nu = 1:N_nu[J+1,P]
            E, phB, isB, ivB = real(E_RPA[J+1,P][nu]), rB_RPA.E0.ph[nu], rB_RPA.E0.is[nu], rB_RPA.E0.iv[nu]

            # Case of m_-1 ...
            if abs(E) > 1e-3
                phm, ism, ivm = phB / E, isB / E, ivB / E
                m_n1_phE0_RPA += phm
                m_n1_isE0_RPA += ism
                m_n1_ivE0_RPA += ivm
            end

            # Case of m_0 ...
            phm, ism, ivm = phB, isB, ivB
            m_0_phE0_RPA += phm
            m_0_isE0_RPA += ism
            m_0_ivE0_RPA += ivm

            # Case of m_1 ...
            phm, ism, ivm = E * phB, E * isB, E * ivB
            m_1_phE0_RPA += phm
            m_1_isE0_RPA += ism
            m_1_ivE0_RPA += ivm

            # Case of m_2 ...
            phm, ism, ivm = E^2 * phB, E^2 * isB, E^2 * ivB
            m_2_phE0_RPA += phm
            m_2_isE0_RPA += ism
            m_2_ivE0_RPA += ivm

            # Case of m_3 ...
            phm, ism, ivm = E^3 * phB, E^3 * isB, E^3 * ivB
            m_3_phE0_RPA += phm
            m_3_isE0_RPA += ism
            m_3_ivE0_RPA += ivm

        end
        # E1 moments ...
        J, P = 1, 2
        @inbounds for nu = 1:N_nu[J+1,P]
            E, phB, isB, ivB = real(E_RPA[J+1,P][nu]), rB_RPA.E1.ph[nu], rB_RPA.E1.is[nu], rB_RPA.E1.iv[nu]

            # Case of m_-1 ...
            if abs(E) > 1e-3
                phm, ism, ivm = phB / E, isB / E, ivB / E
                m_n1_phE1_RPA += phm
                m_n1_isE1_RPA += ism
                m_n1_ivE1_RPA += ivm
            end

            # Case of m_0 ...
            phm, ism, ivm = phB, isB, ivB
            m_0_phE1_RPA += phm
            m_0_isE1_RPA += ism
            m_0_ivE1_RPA += ivm

            # Case of m_1 ...
            phm, ism, ivm = E * phB, E * isB, E * ivB
            m_1_phE1_RPA += phm
            m_1_isE1_RPA += ism
            m_1_ivE1_RPA += ivm

            # Case of m_2 ...
            phm, ism, ivm = E^2 * phB, E^2 * isB, E^2 * ivB
            m_2_phE1_RPA += phm
            m_2_isE1_RPA += ism
            m_2_ivE1_RPA += ivm

            # Case of m_3 ...
            phm, ism, ivm = E^3 * phB, E^3 * isB, E^3 * ivB
            m_3_phE1_RPA += phm
            m_3_isE1_RPA += ism
            m_3_ivE1_RPA += ivm

        end
        # E2 moments ...
        J, P = 2, 1
        @inbounds for nu = 1:N_nu[J+1,P]
            E, phB, isB, ivB = real(E_RPA[J+1,P][nu]), rB_RPA.E2.ph[nu], rB_RPA.E2.is[nu], rB_RPA.E2.iv[nu]

            # Case of m_-1 ...
            if abs(E) > 1e-3
                phm, ism, ivm = phB / E, isB / E, ivB / E
                m_n1_phE2_RPA += phm
                m_n1_isE2_RPA += ism
                m_n1_ivE2_RPA += ivm
            end

            # Case of m_0 ...
            phm, ism, ivm = phB, isB, ivB
            m_0_phE2_RPA += phm
            m_0_isE2_RPA += ism
            m_0_ivE2_RPA += ivm

            # Case of m_1 ...
            phm, ism, ivm = E * phB, E * isB, E * ivB
            m_1_phE2_RPA += phm
            m_1_isE2_RPA += ism
            m_1_ivE2_RPA += ivm

            # Case of m_2 ...
            phm, ism, ivm = E^2 * phB, E^2 * isB, E^2 * ivB
            m_2_phE2_RPA += phm
            m_2_isE2_RPA += ism
            m_2_ivE2_RPA += ivm

            # Case of m_3 ...
            phm, ism, ivm = E^3 * phB, E^3 * isB, E^3 * ivB
            m_3_phE2_RPA += phm
            m_3_isE2_RPA += ism
            m_3_ivE2_RPA += ivm
        end

        # E3 moments ...
        J, P = 3, 2
        @inbounds for nu = 1:N_nu[J+1,P]
            E, phB, isB, ivB = real(E_RPA[J+1,P][nu]), rB_RPA.E3.ph[nu], rB_RPA.E3.is[nu], rB_RPA.E3.iv[nu]

            # Case of m_-1 ...
            if abs(E) > 1e-3
                phm, ism, ivm = phB / E, isB / E, ivB / E
                m_n1_phE3_RPA += phm
                m_n1_isE3_RPA += ism
                m_n1_ivE3_RPA += ivm
            end

            # Case of m_0 ...
            phm, ism, ivm = phB, isB, ivB
            m_0_phE3_RPA += phm
            m_0_isE3_RPA += ism
            m_0_ivE3_RPA += ivm

            # Case of m_1 ...
            phm, ism, ivm = E * phB, E * isB, E * ivB
            m_1_phE3_RPA += phm
            m_1_isE3_RPA += ism
            m_1_ivE3_RPA += ivm

            # Case of m_2 ...
            phm, ism, ivm = E^2 * phB, E^2 * isB, E^2 * ivB
            m_2_phE3_RPA += phm
            m_2_isE3_RPA += ism
            m_2_ivE3_RPA += ivm

            # Case of m_3 ...
            phm, ism, ivm = E^3 * phB, E^3 * isB, E^3 * ivB
            m_3_phE3_RPA += phm
            m_3_isE3_RPA += ism
            m_3_ivE3_RPA += ivm

        end

    # Export the electromagnetic moments & TRK sum-rule values to the summary file ...
    Summary =  open(Output_Path_RPA, "a")
        println(Summary, "\nReview of RPA electromagnetic transition moments m_k ...")

        println(Summary, "\n\tE0 physical:")
        @printf(Summary, "m _-1 = %20.8f\te^2 fm^4 / MeV\n", m_n1_phE0_RPA)
        @printf(Summary, "m _0  = %20.8f\te^2 fm^4\n", m_0_phE0_RPA)
        @printf(Summary, "m _1  = %20.8f\te^2 fm^4 MeV\n", m_1_phE0_RPA)
        @printf(Summary, "m _2  = %20.8f\te^2 fm^4 MeV^2\n", m_2_phE0_RPA)
        @printf(Summary, "m _3  = %20.8f\te^2 fm^4 MeV^3\n", m_3_phE0_RPA)

        println(Summary, "\n\tE1 physical:")
        @printf(Summary, "m _-1 = %20.8f\te^2 fm^2 / MeV\n", m_n1_phE1_RPA)
        @printf(Summary, "m _0  = %20.8f\te^2 fm^2\n", m_0_phE1_RPA)
        @printf(Summary, "m _1  = %20.8f\te^2 fm^2 MeV\n", m_1_phE1_RPA)
        @printf(Summary, "m _2  = %20.8f\te^2 fm^2 MeV^2\n", m_2_phE1_RPA)
        @printf(Summary, "m _3  = %20.8f\te^2 fm^2 MeV^3\n", m_3_phE1_RPA)

        println(Summary, "\n\tE2 physical:")
        @printf(Summary, "m _-1 = %20.8f\te^2 fm^4 / MeV\n", m_n1_phE2_RPA)
        @printf(Summary, "m _0  = %20.8f\te^2 fm^4\n", m_0_phE2_RPA)
        @printf(Summary, "m _1  = %20.8f\te^2 fm^4 MeV\n", m_1_phE2_RPA)
        @printf(Summary, "m _2  = %20.8f\te^2 fm^4 MeV^2\n", m_2_phE2_RPA)
        @printf(Summary, "m _3  = %20.8f\te^2 fm^4 MeV^3\n", m_3_phE2_RPA)

        println(Summary, "\n\tE3 physical:")
        @printf(Summary, "m _-1 = %20.8f\te^2 fm^6 / MeV\n", m_n1_phE3_RPA)
        @printf(Summary, "m _0  = %20.8f\te^2 fm^6\n", m_0_phE3_RPA)
        @printf(Summary, "m _1  = %20.8f\te^2 fm^6 MeV\n", m_1_phE3_RPA)
        @printf(Summary, "m _2  = %20.8f\te^2 fm^6 MeV^2\n", m_2_phE3_RPA)
        @printf(Summary, "m _3  = %20.8f\te^2 fm^6 MeV^3\n", m_3_phE3_RPA)



        println(Summary, "\n\tE0 isoscalar:")
        @printf(Summary, "m _-1 = %20.8f\te^2 fm^4 / MeV\n", m_n1_isE0_RPA)
        @printf(Summary, "m _0  = %20.8f\te^2 fm^4\n", m_0_isE0_RPA)
        @printf(Summary, "m _1  = %20.8f\te^2 fm^4 MeV\n", m_1_isE0_RPA)
        @printf(Summary, "m _2  = %20.8f\te^2 fm^4 MeV^2\n", m_2_isE0_RPA)
        @printf(Summary, "m _3  = %20.8f\te^2 fm^4 MeV^3\n", m_3_isE0_RPA)

        println(Summary, "\n\tE1 isoscalar:")
        @printf(Summary, "m _-1 = %20.8f\te^2 fm^2 / MeV\n", m_n1_isE1_RPA)
        @printf(Summary, "m _0  = %20.8f\te^2 fm^2\n", m_0_isE1_RPA)
        @printf(Summary, "m _1  = %20.8f\te^2 fm^2 MeV\n", m_1_isE1_RPA)
        @printf(Summary, "m _2  = %20.8f\te^2 fm^2 MeV^2\n", m_2_isE1_RPA)
        @printf(Summary, "m _3  = %20.8f\te^2 fm^2 MeV^3\n", m_3_isE1_RPA)

        println(Summary, "\n\tE2 isoscalar:")
        @printf(Summary, "m _-1 = %20.8f\te^2 fm^4 / MeV\n", m_n1_isE2_RPA)
        @printf(Summary, "m _0  = %20.8f\te^2 fm^4\n", m_0_isE2_RPA)
        @printf(Summary, "m _1  = %20.8f\te^2 fm^4 MeV\n", m_1_isE2_RPA)
        @printf(Summary, "m _2  = %20.8f\te^2 fm^4 MeV^2\n", m_2_isE2_RPA)
        @printf(Summary, "m _3  = %20.8f\te^2 fm^4 MeV^3\n", m_3_isE2_RPA)

        println(Summary, "\n\tE3 isoscalar:")
        @printf(Summary, "m _-1 = %20.8f\te^2 fm^6 / MeV\n", m_n1_isE3_RPA)
        @printf(Summary, "m _0  = %20.8f\te^2 fm^6\n", m_0_isE3_RPA)
        @printf(Summary, "m _1  = %20.8f\te^2 fm^6 MeV\n", m_1_isE3_RPA)
        @printf(Summary, "m _2  = %20.8f\te^2 fm^6 MeV^2\n", m_2_isE3_RPA)
        @printf(Summary, "m _3  = %20.8f\te^2 fm^6 MeV^3\n", m_3_isE3_RPA)



        println(Summary, "\n\tE0 isovector:")
        @printf(Summary, "m _-1 = %20.8f\te^2 fm^4 / MeV\n", m_n1_ivE0_RPA)
        @printf(Summary, "m _0  = %20.8f\te^2 fm^4\n", m_0_ivE0_RPA)
        @printf(Summary, "m _1  = %20.8f\te^2 fm^4 MeV\n", m_1_ivE0_RPA)
        @printf(Summary, "m _2  = %20.8f\te^2 fm^4 MeV^2\n", m_2_ivE0_RPA)
        @printf(Summary, "m _3  = %20.8f\te^2 fm^4 MeV^3\n", m_3_ivE0_RPA)

        println(Summary, "\n\tE1 isovector:")
        @printf(Summary, "m _-1 = %20.8f\te^2 fm^2 / MeV\n", m_n1_ivE1_RPA)
        @printf(Summary, "m _0  = %20.8f\te^2 fm^2\n", m_0_ivE1_RPA)
        @printf(Summary, "m _1  = %20.8f\te^2 fm^2 MeV\n", m_1_ivE1_RPA)
        @printf(Summary, "m _2  = %20.8f\te^2 fm^2 MeV^2\n", m_2_ivE1_RPA)
        @printf(Summary, "m _3  = %20.8f\te^2 fm^2 MeV^3\n", m_3_ivE1_RPA)

        println(Summary, "\n\tE2 isovector:")
        @printf(Summary, "m _-1 = %20.8f\te^2 fm^4 / MeV\n", m_n1_ivE2_RPA)
        @printf(Summary, "m _0  = %20.8f\te^2 fm^4\n", m_0_ivE2_RPA)
        @printf(Summary, "m _1  = %20.8f\te^2 fm^4 MeV\n", m_1_ivE2_RPA)
        @printf(Summary, "m _2  = %20.8f\te^2 fm^4 MeV^2\n", m_2_ivE2_RPA)
        @printf(Summary, "m _3  = %20.8f\te^2 fm^4 MeV^3\n", m_3_ivE2_RPA)

        println(Summary, "\n\tE3 isovector:")
        @printf(Summary, "m _-1 = %20.8f\te^2 fm^6 / MeV\n", m_n1_ivE3_RPA)
        @printf(Summary, "m _0  = %20.8f\te^2 fm^6\n", m_0_ivE3_RPA)
        @printf(Summary, "m _1  = %20.8f\te^2 fm^6 MeV\n", m_1_ivE3_RPA)
        @printf(Summary, "m _2  = %20.8f\te^2 fm^6 MeV^2\n", m_2_ivE3_RPA)
        @printf(Summary, "m _3  = %20.8f\te^2 fm^6 MeV^3\n", m_3_ivE3_RPA)



        println(Summary, "\nReview of RPA physical Thomas-Reiche-Kuhn (TRK) energy-weighted sum-rule values ...\n")
        @printf(Summary, "S_E0 = %20.8f\te^2 fm^4\n", TRK_E0_RPA)
        @printf(Summary, "S_E1 = %20.8f\te^2 fm^2\n", TRK_E1_RPA)
        @printf(Summary, "S_E2 = %20.8f\te^2 fm^4\n", TRK_E2_RPA)
        @printf(Summary, "S_E3 = %20.8f\te^2 fm^6\n", TRK_E3_RPA)

    close(Summary)

    # Set the path for export of reduced transition intensities B ...
    Output_File = "IO/" * Params.Calc.Path

    println("\nPreparing export of RPA transition intensities ...")
    # RPA E0 export ...
        open(Output_File * "/RPA/Transitions/E0/RPA_E0.dat", "w") do Write_File
            J, P = 0, 1
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for nu = 1:N_nu[J+1,P]
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_RPA[J+1,P][nu]), rB_RPA.E0.ph[nu], rB_RPA.E0.is[nu], rB_RPA.E0.iv[nu])
            end
        end

    # RPA E1 export ...
            # Standard E1 mode ...
        open(Output_File * "/RPA/Transitions/E1/RPA_E1.dat", "w") do Write_File
            J, P = 1, 2
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for nu = 1:N_nu[J+1,P]
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_RPA[J+1,P][nu]), rB_RPA.E1.ph[nu], rB_RPA.E1.is[nu], rB_RPA.E1.iv[nu])
            end
        end
            # Full vortical mode ...
        open(Output_File * "/RPA/Transitions/E1/RPA_E1_V.dat", "w") do Write_File
            J, P = 1, 2
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for nu in 1:N_nu[J+1,P]
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_RPA[J+1,P][nu]), rB_RPA.E1_V.ph[nu], rB_RPA.E1_V.is[nu], rB_RPA.E1_V.iv[nu])
            end
        end
            # Convective vortical mode ...
        open(Output_File * "/RPA/Transitions/E1/RPA_E1_V_conv.dat", "w") do Write_File
            J, P = 1, 2
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for nu in 1:N_nu[J+1,P]
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_RPA[J+1,P][nu]), rB_RPA.E1_VC.ph[nu], rB_RPA.E1_VC.is[nu], rB_RPA.E1_VC.iv[nu])
            end
        end
            # Spin vortical mode ...
        open(Output_File * "/RPA/Transitions/E1/RPA_E1_V_spin.dat", "w") do Write_File
            J, P = 1, 2
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for nu in 1:N_nu[J+1,P]
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_RPA[J+1,P][nu]), rB_RPA.E1_VS.ph[nu], rB_RPA.E1_VS.is[nu], rB_RPA.E1_VS.iv[nu])
            end
        end

            # Full toroidal mode ...
        open(Output_File * "/RPA/Transitions/E1/RPA_E1_T.dat", "w") do Write_File
            J, P = 1, 2
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for nu in 1:N_nu[J+1,P]
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_RPA[J+1,P][nu]), rB_RPA.E1_T.ph[nu], rB_RPA.E1_T.is[nu], rB_RPA.E1_T.iv[nu])
            end
        end
            # Convective toroidal mode ...
        open(Output_File * "/RPA/Transitions/E1/RPA_E1_T_conv.dat", "w") do Write_File
            J, P = 1, 2
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for nu in 1:N_nu[J+1,P]
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_RPA[J+1,P][nu]), rB_RPA.E1_TC.ph[nu], rB_RPA.E1_TC.is[nu], rB_RPA.E1_TC.iv[nu])
            end
        end
            # Spin toroidal mode ...
        open(Output_File * "/RPA/Transitions/E1/RPA_E1_T_spin.dat", "w") do Write_File
            J, P = 1, 2
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for nu in 1:N_nu[J+1,P]
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_RPA[J+1,P][nu]), rB_RPA.E1_TS.ph[nu], rB_RPA.E1_TS.is[nu], rB_RPA.E1_TS.iv[nu])
            end
        end

            # Isoscalar electric dipole compression mode ...
        open(Output_File * "/RPA/Transitions/E1/RPA_E1_C.dat", "w") do Write_File
            J, P = 1, 2
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for nu in 1:N_nu[J+1,P]
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_RPA[J+1,P][nu]), rB_RPA.E1_C.ph[nu], rB_RPA.E1_C.is[nu], rB_RPA.E1_C.iv[nu])
            end
        end

            # NLO LWA electric dipole mode ...
        open(Output_File * "/RPA/Transitions/E1/RPA_E1_NLO_LWA.dat", "w") do Write_File
            J, P = 1, 2
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for nu in 1:N_nu[J+1,P]
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_RPA[J+1,P][nu]), rB_RPA.E1_NLO_LWA.ph[nu], rB_RPA.E1_NLO_LWA.is[nu], rB_RPA.E1_NLO_LWA.iv[nu])
            end
        end

    # RPA E2 export ...
        open(Output_File * "/RPA/Transitions/E2/RPA_E2.dat", "w") do Write_File
            J, P = 2, 1
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for nu = 1:N_nu[J+1,P]
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_RPA[J+1,P][nu]), rB_RPA.E2.ph[nu], rB_RPA.E2.is[nu], rB_RPA.E2.iv[nu])
            end
        end

    # RPA E3 export ...
        open(Output_File * "/RPA/Transitions/E3/RPA_E3.dat", "w") do Write_File
            J, P = 3, 2
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for nu = 1:N_nu[J+1,P]
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_RPA[J+1,P][nu]), rB_RPA.E3.ph[nu], rB_RPA.E3.is[nu], rB_RPA.E3.iv[nu])
            end
        end

    println("\tExport of information on RPA 1-phonon electromagnetic transitions completed ...")

    return
end