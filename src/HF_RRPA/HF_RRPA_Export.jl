function HF_RRPA_export(Params::Parameters,Orb::Vector{Orb1B},N_nu::Matrix{Int64},E_corr::Float64,C::O1B,Rho::O1B,h::O1B,E_RPA::Matrix{Vector{ComplexF64}},X_RPA::Matrix{Matrix{ComplexF64}},Y_RPA::Matrix{Matrix{ComplexF64}},rB_RPA::ReducedTransition)
    # Read parameters
    Output_File = Params.Calc.Path

    # Export RRPA summary file ...
    HF_RRPA_export_summary(Params,N_nu,E_corr)

    # Export RRPA natural orbitals ... matrix C ...
    C_Export_Path = "IO/" * Output_File * "/Bin/C_RRPA.bin"
    O1b_export(Params,Orb,C,C_Export_Path)

    # Export RRPA OBDM Rho ...
    Rho_Export_Path = "IO/" * Output_File * "/Bin/Rho_RRPA.bin"
    O1b_export(Params,Orb,Rho,Rho_Export_Path)

    # Export RRPA effective single-particle Hamiltonian h ...
    h_Export_Path = "IO/" * Output_File * "/Bin/h_RRPA.bin"
    O1b_export(Params,Orb,h,h_Export_Path)

    # Export of RRPA energies & amplitudes in binary format ...
    @time HF_RRPA_export_binary(Params,N_nu,E_RPA,X_RPA,Y_RPA)

    # Export of RRPA |X|^2 & |Y|^2 amplitudes, not plot ready ...
    HF_RRPA_export_amplitudes(Params,N_nu,E_RPA,X_RPA,Y_RPA)

    # Export of RRPA spectra ...
    HF_RRPA_export_spectrum(Params,N_nu,E_RPA)

    # Export of RRPA plot-ready spectra ...
    HF_RRPA_export_spectrum_plot(Params,N_nu,E_RPA)

    # Export single-particle level occupations ...
    HF_RRPA_export_occupation(Params,Orb,Rho)

    # Calculate & export radial RRPA densities & radii ...
    Summary_File = "IO/" * Params.Calc.Path * "/RRPA/RRPA_Summary.dat"
    Densities_File = "IO/" * Params.Calc.Path * "/RRPA/Densities/RRPA_Radial_Densities.dat"
    @time OBDM_export(Params,Orb,Summary_File,Densities_File,O1B(C.p * Rho.p * C.p', C.n * Rho.n * C.n'),C)

    # Determine & export RRPA effective single-particle energies ...
    @time HF_RRPA_export_single_particle_spectrum(Params,Orb,h)

    # Export of RRPA electromagnetic transitions ...
    HF_RRPA_export_transitions(Params,Orb,N_nu,C,Rho,E_RPA,rB_RPA)

    return
end

function HF_RRPA_export_summary(Params::Parameters,N_nu::Matrix{Int64},E_corr::Float64)
    # Read parameters ...
    A = Params.Calc.A
    Z = Params.Calc.Z
    hw = Params.Calc.hw
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1
    CMS = Params.Calc.CMS
    Orthogon = Params.Calc.RRPA.Ortho
    Output_File = Params.Calc.Path

    # Set the path for "RRPA_Summary.dat" output file ...
    Output_Path = "IO/" * Output_File * "/RRPA/RRPA_Summary.dat"
    
    println("\nPreparing RRPA summary ...")
    Summary =  open(Output_Path, "w")
        println(Summary, "Spherical Renormalized-Random-Phase Approximation review:")
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

        println(Summary, "\nInformation on dimensions of 1-phonon 1p-1h RRPA subspaces ...")

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
        println(Summary, "\n\tTo get the total correct ground-state energy add the mean-field energy ...")

    close(Summary)

    println("\nRRPA summary exported ...")

    return

end

function HF_RRPA_export_binary(Params::Parameters,N_nu::Matrix{Int64},E_RPA::Matrix{Vector{ComplexF64}},X_RPA::Matrix{Matrix{ComplexF64}},Y_RPA::Matrix{Matrix{ComplexF64}})
    # Read parameters ...
    N_max = Params.Calc.Nmax
    J_max = 2*N_max + 1
    Output_File = Params.Calc.Path

    # Set-up export buffer  ...
    Buffer_size, Buffer_count = 10000000, 0
    Buffer = Vector{ComplexF64}(undef,Buffer_size)

    println("\nPerforming binary export of RRPA energies E & amplitudes X & Y ...")

    # Set the RRPA out-put file paths ...
    Output_Path_RPA_X = "IO/" * Output_File * "/Bin/RRPA_X.bin"
    Output_Path_RPA_Y = "IO/" * Output_File * "/Bin/RRPA_Y.bin"
    Output_Path_RPA_E = "IO/" * Output_File * "/Bin/RRPA_E.bin"

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
        # Reset the RPA export buffer ...
    Buffer_count = 0
    Buffer = Vector{ComplexF64}(undef,Buffer_size)
        # Amplitudes Y ...
    open(Output_Path_RPA_Y, "w") do Export_File
        @inbounds for J in 0:J_max
            @inbounds for P in 1:2
                N_ph = N_nu[J+1,P]
                @inbounds for nu in 1:N_ph
                    @inbounds for ph in 1:N_ph
                        Y = Y_RPA[J+1,P][ph,nu]
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
        # Reset the RPA export buffer ...
    Buffer_count = 0
    Buffer = Vector{ComplexF64}(undef,Buffer_size)
        # Energies E ...
    open(Output_Path_RPA_E, "w") do Export_File
        @inbounds for J in 0:J_max
            @inbounds for P in 1:2
                N_ph = N_nu[J+1,P]
                @inbounds for nu in 1:N_ph
                    E = E_RPA[J+1,P][nu]
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

    println("\tBinary export of RRPA energies E amplitudes X & Y completed ...")

    return
end

function HF_RRPA_export_spectrum(Params::Parameters,N_nu::Matrix{Int64},E_RPA::Matrix{Vector{ComplexF64}})
    # Read calculation parameters ...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1
    Output_File = Params.Calc.Path

    println("\nPreparing export of RRPA spectra ...")

    # Set the out-put file path ...
    Output_Path = "IO/" * Output_File * "/RRPA/Spectra/RRPA.dat"

    # RRPA spectrum export ...
    open(Output_Path, "w") do Write_File
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

    println("\tExport of RRPA spectra succesfully finished ...")

    return
end

function HF_RRPA_export_spectrum_plot(Params::Parameters,N_nu::Matrix{Int64},E_RPA::Matrix{Vector{ComplexF64}})
    # Read calculation parameters ...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1
    Output_File = Params.Calc.Path

    # Prepare the TDA & RPA plot-ready spectra for export ...
    println("\nPreparing plot-ready export of RRPA solutions ...")

    # Set export path ...
    Output_Path = "IO/" * Output_File * "/RRPA/Spectra/RRPA_Plot.dat"

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

    # Allocate RRPA spectrum export array ...
    nu_count = 1
    @inbounds for J in 0:J_max
        @inbounds for P in 1:2
            N_qg = N_nu[J+1,P]
            @inbounds for nu in 1:N_qg
                nu_count += 1
                RPA_Solution[1,nu_count] = Float64(J)
                RPA_Solution[2,nu_count] = Float64(P)
                RPA_Solution[3,nu_count] = real(E_RPA[J+1,P][nu])
                RPA_Solution[4,nu_count] = real(E_RPA[J+1,P][nu])
            end
        end
    end

    # Retrieve RRPA energies for export ...
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

    # RRPA spectrum plot data export ...
    println("\tPerforming the export of plot-ready RPA spectra ...")
    open(Output_Path, "w") do Write_File
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
            @printf(Write_File, "%-5s %-5s %-12.6f %-12.6f\n", J, P, E, E_m)
        end
    end

    println("\t\tRRPA plot-ready spectra successfully exported ...")

    return
end

function HF_RRPA_export_occupation(Params::Parameters,Orb::Vector{Orb1B},Rho::O1B)
    # Read parameters ...
    N_max = Params.Calc.Nmax
    a_max = div((N_max+1)*(N_max+2),2)
    Output_File = Params.Calc.Path

    # Read the 1-body density matrix Rho ...
    pRho, nRho = Rho.p, Rho.n

    # Calculate the reference 1-body density matrix Rho ...
    pRho_HF, nRho_HF = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)
    @inbounds for a in 1:a_max
        pRho_HF[a,a] = Orb[a].pO
        nRho_HF[a,a] = Orb[a].nO
    end

    # Set the export path ...
    Output_Path = "IO/" * Output_File * "/RRPA/Densities/RRPA_Occupation.dat"
    
    # Initialize the vectors for occupation numbers & orbital numbers ...
    pO_HF = Vector{Float64}(undef,a_max)
    nO_HF = Vector{Float64}(undef,a_max)
    pO_RRPA = Vector{Float64}(undef,a_max)
    nO_RRPA = Vector{Float64}(undef,a_max)
    pa = Vector{Int64}(undef,a_max)
    na = Vector{Int64}(undef,a_max)

    # Evaluate the occupation numbers & orbital numbers ...
    @inbounds for a in 1:a_max
        pO_HF[a] = pRho_HF[a,a]
        nO_HF[a] = nRho_HF[a,a]
        pO_RRPA[a] = round(pRho[a,a], digits = 4)
        nO_RRPA[a] = round(nRho[a,a], digits = 4)
        pa[a] = a
        na[a] = a
    end

    # Determine reordering of proton & neutron orbitals ...
    Sort_pInd = sortperm(pO_RRPA, rev = true)
    Sort_nInd = sortperm(nO_RRPA, rev = true)
        # Perform reordering of proton orbitals ...
    pO_HF = pO_HF[Sort_pInd]
    pO_RRPA = pO_RRPA[Sort_pInd]
    pa = pa[Sort_pInd]
        # Perform reordering of neutron orbitals ...
    nO_HF = nO_HF[Sort_nInd]
    nO_RRPA = nO_RRPA[Sort_nInd]
    na = na[Sort_nInd]

    # Export RRPA occupation numbers ...
    open(Output_Path, "w") do Write_File
        @printf(Write_File, "%-6s %-6s %-12s %-12s %-12s %-6s %-12s %-12s %-12s\n", "a", "a_p", "pN_HF", "pN_RRPA", "pN_dep", "a_n", "nN_HF", "nN_RRPA", "nN_dep")
        @inbounds for a in 1:a_max
            @printf(Write_File, "%-6d %-6d %-12.6f %-12.6f %-12.6f %-6s %-12.6f %-12.6f %-12.6f\n", a, pa[a], pO_HF[a], pO_RRPA[a], pO_RRPA[a] - pO_HF[a], na[a], nO_HF[a], nO_RRPA[a], nO_RRPA[a] - nO_HF[a])
        end
    end

    return
end

function HF_RRPA_export_single_particle_spectrum(Params::Parameters,Orb::Vector{Orb1B},h::O1B)
    # Read parameters ...
    N_max = Params.Calc.Nmax
    a_max = div((N_max+1)*(N_max+2),2)
    j_max = 2*N_max + 1
    Output_File = Params.Calc.Path

    # Prepare the summary file export path ...
    Summary_File = "IO/" * Output_File * "/RRPA/RRPA_Summary.dat"

    println("\nPreparing export of effective RRPA single-particle energies ...")

    # Diagonalize the effective RRPA single-particle Hamiltonian h ...
    pE, pC = eigen(h.p, sortby = e -> real(e))
    nE, nC = eigen(h.n, sortby = e -> real(e))

    # Initialite list of quantum numbers for single-particle energies ...
    pn, nn = zeros(Int64,N_max+1,j_max), zeros(Int64,N_max+1,j_max)
    pl, nl = Vector{Int64}(undef,a_max), Vector{Int64}(undef,a_max)
    pj, nj = Vector{Int64}(undef,a_max), Vector{Int64}(undef,a_max)

    # Calculate the quantum numbers l & j for single-particle energies ...
    @inbounds for a in 1:a_max
        plSum, pjSum = 0.0, 0.0
        nlSum, njSum = 0.0, 0.0
        @inbounds for b in 1:a_max
            l_b, j_b = Orb[b].l, Orb[b].j
            pN, nN = pC[b,a]^2, nC[b,a]^2
            plME, pjME = l_b * pN, j_b * pN
            nlME, njME = l_b * nN, j_b * nN
            plSum += plME
            pjSum += pjME
            nlSum += nlME
            njSum += njME
        end
        pl[a], pj[a] = Int64(round(plSum)), Int64(round(pjSum))
        nl[a], nj[a] = Int64(round(nlSum)), Int64(round(njSum))
    end

    # Perform the export of RRPA single-particle energies into summary file ...
    Summary =  open(Summary_File, "a")
        println(Summary,"\nProton RRPA single-particle states ...")
        println(Summary,"____________________________________________________________________")
        @printf(Summary, "%4s %4s %4s %14s", "n", "l", "2j", "E\n")
        @inbounds for a in 1:a_max
            l_a, j_a = pl[a], pj[a]
            n_a = pn[l_a+1,j_a]
            E_a = pE[a]
            @printf(Summary, "%4d %4d %4d %14.6f\n",n_a, l_a, j_a, E_a)
            pn[l_a+1,j_a] += 1
        end
        println(Summary,"____________________________________________________________________")

        println(Summary,"\nNeutron RRPA single-particle states ...")
        println(Summary,"____________________________________________________________________")
        @printf(Summary, "%4s %4s %4s %14s", "n", "l", "2j", "E\n")
        @inbounds for a in 1:a_max
            l_a, j_a = nl[a], nj[a]
            n_a = nn[l_a+1,j_a]
            E_a = nE[a]
            @printf(Summary, "%4d %4d %4d %14.6f\n",n_a, l_a, j_a, E_a)
            nn[l_a+1,j_a] += 1
        end
        println(Summary,"____________________________________________________________________")

    close(Summary)

    println("\tExport of RRPA effective single-particle energies finished ...")

    return
end

function HF_RRPA_export_amplitudes(Params::Parameters,N_nu::Matrix{Int64},E_RPA::Matrix{Vector{ComplexF64}},X_RPA::Matrix{Matrix{ComplexF64}},Y_RPA::Matrix{Matrix{ComplexF64}})
    # Read parameters ...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1
    Output_File = Params.Calc.Path

    # Set export paths ...
    Output_Path = "IO/" * Output_File * "/RRPA/Amplitudes/RRPA_Amplitudes_Ordered.dat"

    println("\nPreparing export norms of the RRPA amplitudes X & Y ...")

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
    Output_Path = "IO/" * Output_File * "/RRPA/Amplitudes/RRPA_Amplitudes.dat"

    # Count the total number of 1-phonon levels ...
    N_ph = sum(N_nu)

    # Initialize vectors for export ...
    E_RPA_ord = Vector{Float64}(undef,N_ph)
    x_RPA = Vector{Float64}(undef,N_ph)
    y_RPA = Vector{Float64}(undef,N_ph)
    J_RPA = Vector{Int64}(undef,N_ph)
    P_RPA = Vector{Int64}(undef,N_ph)

    # Allocate the RRPA 1-phonon levels ...
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

    # Determine the reordering of RRPA indices ... ascending in energy ...
    Sort_Ind = sortperm(E_RPA_ord)

    # Perform the reordering of RRPA energies & amplitudes ...
    E_RPA_ord = E_RPA_ord[Sort_Ind]
    x_RPA = x_RPA[Sort_Ind]
    y_RPA = y_RPA[Sort_Ind]
    J_RPA = J_RPA[Sort_Ind]
    P_RPA = P_RPA[Sort_Ind]

    # Write out the ordered RRPA energies & norms of amplitudes ...
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

    println("\tExport of RRPA norms of X & Y completed ...")

    return
end

function HF_RRPA_export_transitions(Params::Parameters,Orb::Vector{Orb1B},N_nu::Matrix{Int64},C::O1B,Rho::O1B,E_RPA::Matrix{Vector{ComplexF64}},rB_RPA::ReducedTransition)
    # Read parameters ...
    Z = Params.Calc.Z
    Output_File = Params.Calc.Path

    # Define basic constants ...
    hc, pmc2 = 197.326980, 938.272013

    # Calculate proton radii moments rN ...
    pR2 = OBDM_rN(Params,2,Rho,C,Orb) 
    pR4 = OBDM_rN(Params,4,Rho,C,Orb)

    println("\nPreparing export of information on RRPA 1-phonon electromagnetic transitions ...")

    # Set the path for "RRPA_Summary.dat" output file ...
    Output_Path_RPA = "IO/" * Output_File * "/RRPA/RRPA_Summary.dat"

    # Initialize RRPA electromagnetic moments ...
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

    # Calculate the physical Thomas-Reiche-Kuhn RRPA sum rules ...
    TRK_E0 = hc^2 / pmc2 / (2.0 * pi) * Float64(Z) * pR2
    TRK_E1 = hc^2 / pmc2 * 9.0 / (8.0 * pi) * Float64(Z)
    TRK_E2 = hc^2 / pmc2 * 50.0 / (8.0 * pi) * Float64(Z) * pR2
    TRK_E3 = hc^2 / pmc2 * 147.0 / (8.0 * pi) * Float64(Z) * pR4

    # Calculate the electromagnetic moments m_k ...
        # E0 moments ...
        J, P = 0, 1
        @inbounds for nu = 1:N_nu[J+1,P]
            E, phB, isB, ivB = real(E_RPA[J+1,P][nu]), rB_RPA.E0.ph[nu], rB_RPA.E0.is[nu], rB_RPA.E0.iv[nu]

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
        @inbounds for nu = 1:N_nu[J+1,P]
            E, phB, isB, ivB = real(E_RPA[J+1,P][nu]), rB_RPA.E1.ph[nu], rB_RPA.E1.is[nu], rB_RPA.E1.iv[nu]

            # Case of m_-1 ...
            if abs(E) > 1e-3
                phm, ism, ivm = phB / E, isB / E, ivB / E
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
        @inbounds for nu = 1:1:N_nu[J+1,P]
            E, phB, isB, ivB = real(E_RPA[J+1,P][nu]), rB_RPA.E2.ph[nu], rB_RPA.E2.is[nu], rB_RPA.E2.iv[nu]

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
        @inbounds for nu = 1:N_nu[J+1,P]
            E, phB, isB, ivB = real(E_RPA[J+1,P][nu]), rB_RPA.E3.ph[nu], rB_RPA.E3.is[nu], rB_RPA.E3.iv[nu]

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

    # Export the RRPA electromagnetic moments & TRK sum-rule values to the summary file ...
    Summary =  open(Output_Path_RPA, "a")
        println(Summary, "\nReview of RRPA electrogmagnetic transition moments m_k ...")

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



        println(Summary, "\nReview of RRPA physical Thomas-Reiche-Kuhn (TRK) energy-weighted sum-rule values ...\n")
        @printf(Summary, "S_E0 = %20.8f\te^2 fm^4\n", TRK_E0)
        @printf(Summary, "S_E1 = %20.8f\te^2 fm^2\n", TRK_E1)
        @printf(Summary, "S_E2 = %20.8f\te^2 fm^4\n", TRK_E2)
        @printf(Summary, "S_E3 = %20.8f\te^2 fm^6\n", TRK_E3)

    close(Summary)

    # Set the export path ...
    Output_File = "IO/" * Params.Calc.Path

    println("\nPreparing export of RRPA transition intensities ...")

    # RRPA E0 export ...
        open(Output_File * "/RRPA/Transitions/E0/RRPA_E0.dat", "w") do Write_File
            J, P = 0, 1
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for nu = 1:N_nu[J+1,P]
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_RPA[J+1,P][nu]), rB_RPA.E0.ph[nu], rB_RPA.E0.is[nu], rB_RPA.E0.iv[nu])
            end
        end

    # RRPA E1 export ...
            # Standard E1 mode ...
        open(Output_File * "/RRPA/Transitions/E1/RRPA_E1.dat", "w") do Write_File
            J, P = 1, 2
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for nu = 1:N_nu[J+1,P]
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_RPA[J+1,P][nu]), rB_RPA.E1.ph[nu], rB_RPA.E1.is[nu], rB_RPA.E1.iv[nu])
            end
        end
            # Full vortical mode ...
        open(Output_File * "/RRPA/Transitions/E1/RRPA_E1_V.dat", "w") do Write_File
            J, P = 1, 2
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for nu in 1:N_nu[J+1,P]
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_RPA[J+1,P][nu]), rB_RPA.E1_V.ph[nu], rB_RPA.E1_V.is[nu], rB_RPA.E1_V.iv[nu])
            end
        end
            # Convective vortical mode ...
        open(Output_File * "/RRPA/Transitions/E1/RRPA_E1_V_conv.dat", "w") do Write_File
            J, P = 1, 2
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for nu in 1:N_nu[J+1,P]
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_RPA[J+1,P][nu]), rB_RPA.E1_VC.ph[nu], rB_RPA.E1_VC.is[nu], rB_RPA.E1_VC.iv[nu])
            end
        end
            # Spin vortical mode ...
        open(Output_File * "/RRPA/Transitions/E1/RRPA_E1_V_spin.dat", "w") do Write_File
            J, P = 1, 2
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for nu in 1:N_nu[J+1,P]
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_RPA[J+1,P][nu]), rB_RPA.E1_VS.ph[nu], rB_RPA.E1_VS.is[nu], rB_RPA.E1_VS.iv[nu])
            end
        end

            # Full toroidal mode ...
        open(Output_File * "/RRPA/Transitions/E1/RRPA_E1_T.dat", "w") do Write_File
            J, P = 1, 2
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for nu in 1:N_nu[J+1,P]
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_RPA[J+1,P][nu]), rB_RPA.E1_T.ph[nu], rB_RPA.E1_T.is[nu], rB_RPA.E1_T.iv[nu])
            end
        end
            # Convective toroidal mode ...
        open(Output_File * "/RRPA/Transitions/E1/RRPA_E1_T_conv.dat", "w") do Write_File
            J, P = 1, 2
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for nu in 1:N_nu[J+1,P]
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_RPA[J+1,P][nu]), rB_RPA.E1_TC.ph[nu], rB_RPA.E1_TC.is[nu], rB_RPA.E1_TC.iv[nu])
            end
        end
            # Spin toroidal mode ...
        open(Output_File * "/RRPA/Transitions/E1/RRPA_E1_T_spin.dat", "w") do Write_File
            J, P = 1, 2
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for nu in 1:N_nu[J+1,P]
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_RPA[J+1,P][nu]), rB_RPA.E1_TS.ph[nu], rB_RPA.E1_TS.is[nu], rB_RPA.E1_TS.iv[nu])
            end
        end

            # Full compression mode ...
        open(Output_File * "/RRPA/Transitions/E1/RRPA_E1_C.dat", "w") do Write_File
            J, P = 1, 2
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for nu in 1:N_nu[J+1,P]
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_RPA[J+1,P][nu]), rB_RPA.E1_C.ph[nu], rB_RPA.E1_C.is[nu], rB_RPA.E1_C.iv[nu])
            end
        end
            # Convective compression mode ...
        open(Output_File * "/RRPA/Transitions/E1/RRPA_E1_C_conv.dat", "w") do Write_File
            J, P = 1, 2
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for nu in 1:N_nu[J+1,P]
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_RPA[J+1,P][nu]), rB_RPA.E1_CC.ph[nu], rB_RPA.E1_CC.is[nu], rB_RPA.E1_CC.iv[nu])
            end
        end
            # Spin compression mode ...
        open(Output_File * "/RRPA/Transitions/E1/RRPA_E1_C_spin.dat", "w") do Write_File
            J, P = 1, 2
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for nu in 1:N_nu[J+1,P]
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_RPA[J+1,P][nu]), rB_RPA.E1_CS.ph[nu], rB_RPA.E1_CS.is[nu], rB_RPA.E1_CS.iv[nu])
            end
        end

            # Continuity equation electric dipole compression mode ...
        open(Output_File * "/RRPA/Transitions/E1/RRPA_E1_c.dat", "w") do Write_File
            J, P = 1, 2
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for nu in 1:N_nu[J+1,P]
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_RPA[J+1,P][nu]), rB_RPA.E1_c.ph[nu], rB_RPA.E1_c.is[nu], rB_RPA.E1_c.iv[nu])
            end
        end

            # NLO LWA electric dipole mode ...
        open(Output_File * "/RRPA/Transitions/E1/RRPA_E1_NLO_LWA.dat", "w") do Write_File
            J, P = 1, 2
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for nu in 1:N_nu[J+1,P]
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_RPA[J+1,P][nu]), rB_RPA.E1_NLO_LWA.ph[nu], rB_RPA.E1_NLO_LWA.is[nu], rB_RPA.E1_NLO_LWA.iv[nu])
            end
        end

    # RRPA E2 export ...
        open(Output_File * "/RRPA/Transitions/E2/RRPA_E2.dat", "w") do Write_File
            J, P = 2, 1
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for nu = 1:N_nu[J+1,P]
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_RPA[J+1,P][nu]), rB_RPA.E2.ph[nu], rB_RPA.E2.is[nu], rB_RPA.E2.iv[nu])
            end
        end

    # RRPA E3 export ...
        open(Output_File * "/RRPA/Transitions/E3/RRPA_E3.dat", "w") do Write_File
            J, P = 3, 2
            @printf(Write_File, "%-20s %-20s %-20s %-20s\n", "E", "B_ph", "B_is", "B_iv")
            @inbounds for nu = 1:N_nu[J+1,P]
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f\n", real(E_RPA[J+1,P][nu]), rB_RPA.E3.ph[nu], rB_RPA.E3.is[nu], rB_RPA.E3.iv[nu])
            end
        end

    println("\tExport of information on RRPA 1-phonon electromagnetic transitions completed ...")

    return
end