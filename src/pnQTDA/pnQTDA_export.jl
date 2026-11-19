function pnQTDA_export(Params::Parameters,Orb_pn2qp::Orb2B_2qp,E_pnQTDA::Matrix{Vector{Float64}},X_pnQTDA::Matrix{Matrix{Float64}})
    # Export of pnQTDA summary ...
    @time pnQTDA_summary(Params,Orb_pn2qp)

    # Export of pnQTDA energies & amplitudes in binary format ...
    @time pnQTDA_binary_export(Params,E_pnQTDA,X_pnQTDA)

    # Export of pnQTDA spectra ...
    @time pnQTDA_spectrum_export(Params,Orb_pn2qp,E_pnQTDA)

    # Export of pnQTDA plot-ready spectra ...
    @time pnQTDA_spectrum_plot_export(Params,Orb_pn2qp,E_pnQTDA)

    return
end

function pnQTDA_summary(Params::Parameters,Orb_pn2qp::Orb2B_2qp)
    # Read parameters ...
    A = Params.Calc.A
    Z = Params.Calc.Z
    hw = Params.Calc.hw
    N_max = Params.Calc.Nmax
    J_max = 2*N_max + 1
    CMS = Params.Calc.CMS
    Reference = Params.Calc.pnQTDA.Reference
    Output_File = Params.Calc.Path

    # Set the path for "pnQTDA_Summary.dat" output file ...
    Output_Path = "io/" * Output_File * "/pnQTDA/pnQTDA_summary.dat"

    println("\nPreparing pnQTDA summary file ...")
    Summary =  open(Output_Path, "w")
        println(Summary, "Spherical proton-neutron Quasiparticle Tamm-Dancoff Approximation review:")
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
        println(Summary, "\nReference quasiparticle mean-field solution:   Reference = ''" * Reference * "''")

        N_qp = 0
        @inbounds for J in 0:J_max
            @inbounds for P in 1:2
                n_qp = Orb_pn2qp.N[P,J+1] * (2*J + 1)
                N_qp += n_qp
            end
        end

        println(Summary, "\nInformation on dimensions of pnQTDA 1-phonon subspaces ...")

        println(Summary, "\nTotal number of 1-phonon pn quasiparticle states in J-scheme = " * string(sum(Orb_pn2qp.N)))
        println(Summary, "Total number of 1-phonon pn quasiparticle states in M-scheme = " * string(N_qp))

        println(Summary, "\nJ\tN_phonon")
        @inbounds for J in 0:J_max
            N_qp_Jp, N_qp_Jm = Orb_pn2qp.N[1,J+1], Orb_pn2qp.N[2,J+1]
            Row = string(string(J) * "\t" * string(N_qp_Jp + N_qp_Jm))
            println(Summary, Row)
        end
        println(Summary, "\nSubspace\tP = +")
        @inbounds for J in 0:J_max
            N_qp = Orb_pn2qp.N[1,J+1]
            Row = string(string(J) * "\t" *string(N_qp))
            println(Summary, Row)
        end
        println(Summary, "\nSubspace\tP = -")
        @inbounds for J in 0:J_max
            N_qp = Orb_pn2qp.N[2,J+1]
            Row = string(string(J) * "\t" *string(N_qp))
            println(Summary, Row)
        end

    close(Summary)

    println("\npnQTDA summary exported ...")

    return
end

function pnQTDA_binary_export(Params::Parameters,E_pnQTDA::Matrix{Vector{Float64}},X_pnQTDA::Matrix{Matrix{Float64}})
    # Read parameters ...
    N_max = Params.Calc.Nmax
    J_max = 2*N_max + 1
    Output_File = Params.Calc.Path

    println("\nPerforming binary export of pnQTDA energies E & amplitudes X ...")

    # Set the out-put file paths ...
    Output_Path_X = "io/" * Output_File * "/pnQTDA/bin/pnQTDA_X.bin"
    Output_Path_E = "io/" * Output_File * "/pnQTDA/bin/pnQTDA_E.bin"

    # Binary export of pnQTDA solutions ... J & P blocks in the same order as read in pnQTDA_import_binary ...
        # Amplitudes X ... column-major, i.e. X[qp,nu] with qp running fastest ...
    open(Output_Path_X, "w") do Export_File
        @inbounds for J in 0:J_max
            @inbounds for P in 1:2
                write(Export_File,X_pnQTDA[P,J+1])
            end
        end
    end
        # Energies E ...
    open(Output_Path_E, "w") do Export_File
        @inbounds for J in 0:J_max
            @inbounds for P in 1:2
                write(Export_File,E_pnQTDA[P,J+1])
            end
        end
    end

    println("\tBinary export of pnQTDA energies E amplitudes X completed ...")

    return
end

function pnQTDA_spectrum_export(Params::Parameters,Orb_pn2qp::Orb2B_2qp,E_pnQTDA::Matrix{Vector{Float64}})
    # Read parameters ...
    N_max = Params.Calc.Nmax
    J_max = 2*N_max + 1
    Output_File = Params.Calc.Path

    # Set the out-put file paths ...
    Output_Path = "io/" * Output_File * "/pnQTDA/spectra/pnQTDA.dat"

    println("\nPreparing export of pnQTDA spectra ...")

    # pnQTDA spectrum export
    open(Output_Path, "w") do Write_File
        @printf(Write_File, "%-5s %-5s %-10s\n", "J", "P", "E")
        @inbounds for J in 0:J_max
            @inbounds for P in 1:2
                N_qp = Orb_pn2qp.N[P,J+1]
                @inbounds for nu in 1:N_qp
                    if P == 1
                        @printf(Write_File, "%-5d %-5s %-10.6f\n", J, "+", E_pnQTDA[P,J+1][nu])
                    else
                        @printf(Write_File, "%-5d %-5s %-10.6f\n", J, "-", E_pnQTDA[P,J+1][nu])
                    end
                end
                println(Write_File,"\n")
            end
        end
    end

    println("\tExport of pnQTDA spectra succesfully finished ...")

    return
end

function pnQTDA_spectrum_plot_export(Params::Parameters,Orb_pn2qp::Orb2B_2qp,E_pnQTDA::Matrix{Vector{Float64}})
    # Read parameters ...
    N_max = Params.Calc.Nmax
    J_max = 2*N_max + 1
    IO = Params.Calc.Path

    # Set the out-put file paths ...
    Output_Path = "io/" * IO * "/pnQTDA/spectra/pnQTDA_plot.dat"

    # Spectra label gap parameter ...
    Delta = 1.0

    # Prepare the pnQTDA plot-ready spectra for export ...
    println("\nPreparing plot-ready export of pnQTDA spectra ...")

    # Count the total number of 1-phonon excitations ... no 0+ ground state, pn phonons live in the neighbouring odd-odd nuclei ...
    N_qp = Int64(sum(Orb_pn2qp.N))

    # Initialite array for solution export ...
    pnQTDA_export = Matrix{Float64}(undef,4,N_qp)

    # Allocate pnQTDA spectrum export array ...
    qp_count = 0
    @inbounds for J in 0:J_max
        @inbounds for P in 1:2
            N_ab = Orb_pn2qp.N[P,J+1]
            @inbounds for nu in 1:N_ab
                qp_count += 1
                pnQTDA_export[1,qp_count] = Float64(J)
                pnQTDA_export[2,qp_count] = Float64(P)
                pnQTDA_export[3,qp_count] = E_pnQTDA[P,J+1][nu]
                pnQTDA_export[4,qp_count] = E_pnQTDA[P,J+1][nu]
            end
        end
    end

    # Reorder spectrum ascending in energy ...
    Sort = sortperm(pnQTDA_export[3,:])
        # Perform the reordering ...
    pnQTDA_export .= @views pnQTDA_export[:,Sort]

    # Adjust the positions of J & P labels ...
    @inbounds for nu in 2:N_qp
        E_1 = pnQTDA_export[4,nu-1]
        E_2 = pnQTDA_export[4,nu]
        if E_2 < E_1
            pnQTDA_export[4,nu] = E_1 + 1.0
        elseif abs(E_2 - E_1) < Delta
            pnQTDA_export[4,nu] = E_1 + 1.0
        end
    end

    # pnQTDA spectrum plot data export ...
    println("\nPerforming the export of plot-ready pnQTDA spectra ...")
    open(Output_Path, "w") do Write_File
        @printf(Write_File, "%-5s %-5s %-12s %-12s\n", "J", "P", "E", "label")
        @inbounds for nu in 1:N_qp
            J = Int64(round(pnQTDA_export[1,nu]))
            P = "P"
            if abs(pnQTDA_export[2,nu] - 1.0) < 1e-3
                P = "+"
            else
                P = "-"
            end
            E = pnQTDA_export[3,nu]
            E_m = pnQTDA_export[4,nu]
            @printf(Write_File, "%-5s %-5s %-12.6f %-12.6f\n", J, P, E, E_m)
        end
    end

    println("\tpnQTDA plot-ready spectra successfully exported ...")

    return
end
