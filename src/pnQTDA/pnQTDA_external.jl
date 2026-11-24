function pnQTDA_import_binary(Params::Parameters)
    # Parameters / dimensions
    N_max = Params.Calc.Nmax
    J_max = 2*N_max + 1
    bin_path = "io/" * Params.Calc.Path * "/pnQTDA/bin/"

    # Initialite the list of values of J & P for iteration ...
    JP_list = JP_initialize(J_max)

    # Make s.p. orbitals - NuHamil ordering convention (same as in pnQTDA_solver) ...
    Orb = orbitals_make(Params)

    # Construct 2-body pn 2qp orbitals ...
    Orb_pn2qp = orbitals_pn2qp_make(Params,Orb)

    println("\nImporting pnQTDA solutions from binary format ...")

    # Initialize amplitudes & energies ...
    X_pnQTDA = Matrix{Matrix{Float64}}(undef,2,J_max+1)
    E_pnQTDA = Matrix{Vector{Float64}}(undef,2,J_max+1)

    # Initialite each J & P block ...
    @inbounds for JP in JP_list
        J, P = JP[1], JP[2]
        N_2qp = Orb_pn2qp.N[P,J+1]
        X_pnQTDA[P,J+1] = Matrix{Float64}(undef,N_2qp,N_2qp)
        E_pnQTDA[P,J+1] = Vector{Float64}(undef,N_2qp)
    end

    # Set the import file paths ...
    import_path_X = bin_path * "pnQTDA_X.bin"
    import_path_E = bin_path * "pnQTDA_E.bin"

    # Read pnQTDA amplitudes X ...
    open(import_path_X, "r") do Import_File
        @inbounds for JP in JP_list
            J, P = JP[1], JP[2]
            read!(Import_File,X_pnQTDA[P,J+1])
        end
    end

    # Read pnQTDA energies E ...
    open(import_path_E, "r") do Import_File
        @inbounds for JP in JP_list
            J, P = JP[1], JP[2]
            read!(Import_File,E_pnQTDA[P,J+1])
        end
    end

    println("\tpnQTDA solution binaries have been successfully imported ...")

    return X_pnQTDA, E_pnQTDA
end
