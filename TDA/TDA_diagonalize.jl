function TDA_diagonalize(Params::Parameters,Orb::Vector{Orb1B},Orb_1p1h::Orb2B_1p1h,A::Matrix{Matrix{Float64}},N::Matrix{Matrix{Float64}},TrOp::Tr1B)    # Read calculation parameters ...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1
    Orthogon = Params.Calc.TDA.Ortho

    # Initialize the list of values of J & P for iteration ...
    JP_list = JP_initialize(J_max)

    println("\nStarting diagonalization of the TDA system of equations ...")

    # Initialize the storage for TDA solutions ...
    E = Matrix{Vector{Float64}}(undef,2,J_max+1)
    X = Matrix{Matrix{Float64}}(undef,2,J_max+1)

    # Diagonalization of TDA eigenvalue problem ...
    @inbounds for JP in JP_list
        J, P = JP[1], JP[2]
        N_ph = Orb_1p1h.N[P,J+1]

        A_JP = A[P,J+1]
        N_JP_diag = diag(N[P,J+1])

        println("\t\tDiagonalizing:\tJ = $J, P = $(P == 1 ? "+" : "-")")

        if N_ph == 0
            E[P,J+1] = Vector{Float64}(undef,0)
            X[P,J+1] = Matrix{Float64}(undef,0,0)
            continue
        end

        X_JP = Matrix{Float64}(undef,N_ph,N_ph)
        E_JP = Vector{Float64}(undef,N_ph)

        if Orthogon == false || !(J == 1 && P == 2) || N_ph < 2
            # Evaluate inv(sqrt(N)) as diagonal ...
            inv_sqrtN = 1.0 ./ sqrt.(N_JP_diag)

            # Renormalized TDA matrix H = N^(-1/2) * A * N^(-1/2) ...
            H = inv_sqrtN .* A_JP .* transpose(inv_sqrtN)

            # Solve the real symmetric eigenvalue problem ...
            TDA_system = eigen!(Symmetric(H))

            E_JP .= TDA_system.values

            # TDA amplitudes X = N^(-1/2) * R ... (N-normalized)
            X_JP .= inv_sqrtN .* TDA_system.vectors

        elseif Orthogon == true && (J == 1 && P == 2)
            # Define physical subspace dimension ...
            N_ort = N_ph - 1

            # Initialize the vector for orthogonalization of spurious 1- level ...
            Spur = TDA_spurious_ini(Orb,Orb_1p1h,TrOp)

            # Perform the orthogonalization of spurious 1- level ...
            U = Matrix{Float64}(undef,N_ph,N_ph)

            Spur_ind = TDA_spurious_ortho!(U,Spur)

            # Remove the spurious column from 1p-1h basis ... U_sub is the physical subspace basis ...
            @inbounds for a in (Spur_ind+1):N_ph
                @views U[:,a-1] .= U[:,a]
            end

            @views U_sub = U[:,1:N_ort]

            # Projected matrices A_ort = U_sub' * A * U_sub & N_ort = U_sub' * N * U_sub ...
            AU = A_JP * U_sub
            A_ort = transpose(U_sub) * AU

            AU .= N_JP_diag .* U_sub
            N_ort = transpose(U_sub) * AU

            # Solve the generalized symmetric-definite eigenvalue problem A_ort * X_ort = E * N_ort * X_ort ...
            TDA_system = eigen!(Symmetric(A_ort),Symmetric(N_ort))

            @views E_JP[2:N_ph] .= TDA_system.values

            # TDA amplitudes in the full 1p-1h space X = U_sub * X_ort ...
            @views mul!(X_JP[:,2:N_ph],U_sub,TDA_system.vectors)

            # Solve the TDA eigenvalue problem in the spurious subspace ... 1x1 eigenvalue problem ...
            A_spur = dot(Spur,A_JP,Spur)
            N_spur = dot(Spur,N_JP_diag .* Spur)

            E_spur = A_spur / N_spur

            println("\t\t\tSpurious 1- TDA level   ...   E = ($(round(E_spur, digits=7))) MeV")

            E_JP[1] = E_spur
            @views X_JP[:,1] .= Spur ./ sqrt(N_spur)
        end

        # Check the stability of the reference ... (negative TDA energies)
        if any(E_nu -> E_nu < -1e-8, E_JP)
            println("\t\t\tNegative TDA eigenvalue detected in channel:\tJ = $J, P = $(P == 1 ? "+" : "-")")
        end

        E[P,J+1] = E_JP
        X[P,J+1] = X_JP
    end

    println("\tFinished diagonalization of the TDA system of equations ...")

    return E, X
end
