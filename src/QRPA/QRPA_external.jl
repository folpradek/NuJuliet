function QRPA_import_binary(Params::Parameters)
    # Parameters / dimensions
    N_max = Params.Calc.Nmax
    J_max = 2*N_max + 1
    bin_path = "io/" * Params.Calc.Path * "/QRPA/bin/"

    # Initialite the list of values of J & P for iteration ...
    JP_list = JP_initialize(J_max)

    # Make s.p. orbitals - NuHamil ordering convention (same as in QRPA_solver) ...
    Orb = orbitals_make(Params)

    # Construct 2-body 2qp orbitals ...
    Orb_2qp = orbitals_2qp_make(Params,Orb)

    println("\nImporting QRPA solutions from binary format ...")

    # Initialize amplitudes & energies ...
    X_QRPA = Matrix{Matrix{ComplexF64}}(undef,2,J_max+1)
    Y_QRPA = Matrix{Matrix{ComplexF64}}(undef,2,J_max+1)
    E_QRPA = Matrix{Vector{ComplexF64}}(undef,2,J_max+1)

    # Initialite each J & P block & count the expected number of MEs ...
    N_X, N_E = 0, 0
    @inbounds for JP in JP_list
        J, P = JP[1], JP[2]
        N_2qp = Orb_2qp.N[P,J+1]
        X_QRPA[P,J+1] = Matrix{ComplexF64}(undef,N_2qp,N_2qp)
        Y_QRPA[P,J+1] = Matrix{ComplexF64}(undef,N_2qp,N_2qp)
        E_QRPA[P,J+1] = Vector{ComplexF64}(undef,N_2qp)
        N_X += N_2qp^2
        N_E += N_2qp
    end

    # Set the import file paths...
    import_path_X = bin_path * "QRPA_X.bin"
    import_path_Y = bin_path * "QRPA_Y.bin"
    import_path_E = bin_path * "QRPA_E.bin"

    # Read QRPA amplitudes X ...
    open(import_path_X, "r") do Import_File
        @inbounds for JP in JP_list
            J, P = JP[1], JP[2]
            read!(Import_File,X_QRPA[P,J+1])
        end
    end

    # Read QRPA amplitudes Y ...
    open(import_path_Y, "r") do Import_File
        @inbounds for JP in JP_list
            J, P = JP[1], JP[2]
            read!(Import_File,Y_QRPA[P,J+1])
        end
    end

    # Read QRPA energies E ...
    open(import_path_E, "r") do Import_File
        @inbounds for JP in JP_list
            J, P = JP[1], JP[2]
            read!(Import_File,E_QRPA[P,J+1])
        end
    end

    println("\tQRPA solution binaries have been successfully imported ...")

    return X_QRPA, Y_QRPA, E_QRPA
end