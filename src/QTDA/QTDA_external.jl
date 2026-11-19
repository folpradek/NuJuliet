function QTDA_import_binary(Params::Parameters)
    # Parameters / dimensions
    N_max = Params.Calc.Nmax
    J_max = 2*N_max + 1
    bin_path = "io/" * Params.Calc.Path * "/QTDA/bin/"

    # Initialite the list of values of J & P for iteration ...
    JP_list = JP_initialize(J_max)

    # Make s.p. orbitals - NuHamil ordering convention (same as in QTDA_solver) ...
    Orb = orbitals_make(Params)

    # Construct 2-body 2qp orbitals ...
    Orb_2qp = orbitals_2qp_make(Params,Orb)

    println("\nImporting QTDA solutions from binary format ...")

    # Initialize amplitudes & energies ...
    X_QTDA = Matrix{Matrix{Float64}}(undef,2,J_max+1)
    E_QTDA = Matrix{Vector{Float64}}(undef,2,J_max+1)

    # Initialite each J & P block & count the expected number of MEs ...
    N_X, N_E = 0, 0
    @inbounds for JP in JP_list
        J, P = JP[1], JP[2]
        N_2qp = Orb_2qp.N[P,J+1]
        X_QTDA[P,J+1] = Matrix{Float64}(undef,N_2qp,N_2qp)
        E_QTDA[P,J+1] = Vector{Float64}(undef,N_2qp)
        N_X += N_2qp^2
        N_E += N_2qp
    end

    # Set the import file paths ...
    import_path_X = bin_path * "QTDA_X.bin"
    import_path_E = bin_path * "QTDA_E.bin"

    # Read QTDA amplitudes X ...
    open(import_path_X, "r") do Import_File
        @inbounds for JP in JP_list
            J, P = JP[1], JP[2]
            read!(Import_File,X_QTDA[P,J+1])
        end
    end

    # Read QTDA energies E ...
    open(import_path_E, "r") do Import_File
        @inbounds for JP in JP_list
            J, P = JP[1], JP[2]
            read!(Import_File,E_QTDA[P,J+1])
        end
    end

    println("\tQTDA solution binaries have been successfully imported ...")

    return X_QTDA, E_QTDA
end
