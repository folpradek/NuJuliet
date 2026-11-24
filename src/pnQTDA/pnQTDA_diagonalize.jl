function pnQTDA_diagonalize(Params::Parameters,A::Matrix{Matrix{Float64}})
    # Read calculation parameters ...
    N_max = Params.Calc.Nmax
    J_max = 2*N_max + 1

    # Make the list of values of J & P for iteration ...
    JP_list = JP_initialize(J_max)

    # Initialite the storage for pnQTDA solutions ...
    E_pnQTDA = Matrix{Vector{Float64}}(undef,2,J_max+1)
    X_pnQTDA = Matrix{Matrix{Float64}}(undef,2,J_max+1)

    # Perform the diagonalization of the pnQTDA matrix A ... no spurious Goldstone levels in the pn channel ...
    println("\nDiagonalizing the pnQTDA matrix A ...")

    @inbounds for JP in JP_list
        J, P = JP[1], JP[2]

        E_pnQTDA_JP, X_pnQTDA_JP = eigen!(A[P,J+1], sortby=+)

        # Store the pnQTDA solutions ...
        E_pnQTDA[P,J+1] = E_pnQTDA_JP
        X_pnQTDA[P,J+1] = X_pnQTDA_JP
    end

    println("\nThe pnQTDA matrix A succesfully diagonalized ...")

    return E_pnQTDA, X_pnQTDA
end
