function QTDA_diagonalize(Params::Parameters,Orb::Vector{Orb1B},Orb_2qp::qpOrb2B,A::Matrix{Matrix{Float64}},qpN::qpO1B,qpTrOp::qpTr1B)
    # Read calculation parameters ...
    N_max = Params.Calc.Nmax
    J_max = 2*N_max + 1
    Orthogon = Params.Calc.QTDA.Ortho

    # Make the list of values of J & P for iteration ...
    JP_list = JP_initialize(J_max)

    # Determine the largest 2qp space dimension ...
    N_qp_max = max(Orb_2qp.N[1,1],Orb_2qp.N[2,2])

    # Initialite the storage for QTDA solutions ...
    E_QTDA = Matrix{Vector{Float64}}(undef,2,J_max+1)
    X_QTDA = Matrix{Matrix{Float64}}(undef,2,J_max+1)

    # Prepare buffers for solutions ...
    if Orthogon == true
        M1_temp = zeros(Float64,N_qp_max,N_qp_max)
        M2_temp = zeros(Float64,N_qp_max,N_qp_max)
        M3_temp = zeros(Float64,N_qp_max,N_qp_max)
        M4_temp = zeros(Float64,N_qp_max,N_qp_max)
    end

    # Perform the diagonalization of the QTDA matrix A ...
    println("\nDiagonalizing the QTDA matrix A ...")

    @inbounds for JP in JP_list
        J, P = JP[1], JP[2]

        A_JP = A[P,J+1]

        if Orthogon == false || !((J == 0 && P == 1) || (J == 1 && P == 2))
            E_QTDA_JP, X_QTDA_JP = eigen!(A_JP, sortby=+)

        elseif Orthogon == true && J == 0 && P == 1
            # Read the dimension of 0+ 2-qp space ...
            N_qp = Orb_2qp.N[1,1]

            @views U = M1_temp[1:N_qp,1:N_qp]
            @views V = M2_temp[1:N_qp,1:N_qp]
            @views I = M3_temp[1:N_qp,1:N_qp]
            @views Pm = M4_temp[1:N_qp,1:N_qp-1]

            U .= 0.0
            V .= 0.0
            Pm .= 0.0

            # Initialize the vector for orthogonalization of spurious 0+ level ...
            Spur_PN = QTDA_spurious_PN_initialize(Orb,Orb_2qp,qpN)

            # Perform the orthogonalization of spurious 0+ level ...
            V, Spur_ind = QTDA_spurious_orthogonalize(N_qp,Spur_PN,V,I)

            @inbounds for a in 1:(Spur_ind-1)
                @views U[:,a] = V[:,a]
            end

            @inbounds for a in (Spur_ind+1):N_qp
                @views U[:,a-1] = V[:,a]
            end

            @views A_JP_ort = M2_temp[1:N_qp-1,1:N_qp-1]
            A_JP_ort .= 0.0

            @views U_sub = U[:,1:N_qp-1]

            mul!(Pm, A_JP, U_sub)
            mul!(A_JP_ort, transpose(U_sub), Pm)

            E_QTDA_JP_ort, X_QTDA_JP_ort = eigen!(A_JP_ort, sortby=+)

            Pm .= 0.0

            mul!(Pm, U_sub, X_QTDA_JP_ort)

            E_QTDA_JP = Vector{Float64}(undef,N_qp)
            E_QTDA_JP[1] = 0.0
            @inbounds for a in 2:N_qp
                E_QTDA_JP[a] = E_QTDA_JP_ort[a-1]
            end

            X_QTDA_JP = Matrix{Float64}(undef,N_qp,N_qp)
            @views X_QTDA_JP[:,1] = Spur_PN
            @inbounds for a in 2:N_qp
                @views X_QTDA_JP[:,a] = Pm[:,a-1]
            end
            
        elseif Orthogon == true && J == 1 && P == 2
            # Read the dimension of 1- 2-qp space ...
            N_qp = Orb_2qp.N[2,2]

            @views U = M1_temp[1:N_qp,1:N_qp]
            @views V = M2_temp[1:N_qp,1:N_qp]
            @views I = M3_temp[1:N_qp,1:N_qp]
            @views Pm = M4_temp[1:N_qp,1:N_qp-1]

            U .= 0.0
            V .= 0.0
            Pm .= 0.0

            # Initialize the vector for orthogonalization of spurious 1- level ...
            Spur_CM = QTDA_spurious_CM_initialize(Orb_2qp,qpTrOp)

            V, Spur_ind = QTDA_spurious_orthogonalize(N_qp,Spur_CM,V,I)

            @inbounds for a in 1:(Spur_ind-1)
                @views U[:,a] = V[:,a]
            end

            @inbounds for a in (Spur_ind+1):N_qp
                @views U[:,a-1] = V[:,a]
            end

            @views A_JP_ort = M2_temp[1:N_qp-1,1:N_qp-1]
            A_JP_ort .= 0.0

            @views U_sub = U[:,1:N_qp-1]

            mul!(Pm, A_JP, U_sub)
            mul!(A_JP_ort, transpose(U_sub), Pm) 

            E_QTDA_JP_ort, X_QTDA_JP_ort = eigen!(A_JP_ort, sortby=+)
            
            Pm .= 0.0

            mul!(Pm, U_sub, X_QTDA_JP_ort)

            E_QTDA_JP = Vector{Float64}(undef,N_qp)
            E_QTDA_JP[1] = 0.0
            @inbounds for a in 2:N_qp
                E_QTDA_JP[a] = E_QTDA_JP_ort[a-1]
            end

            X_QTDA_JP = Matrix{Float64}(undef,N_qp,N_qp)
            @views X_QTDA_JP[:,1] = Spur_CM
            @inbounds for a in 2:N_qp
                @views X_QTDA_JP[:,a] = Pm[:,a-1]
            end
        end

        # Store the QTDA solutions ...
        E_QTDA[P,J+1] = E_QTDA_JP
        X_QTDA[P,J+1] = X_QTDA_JP
    end

    println("\nThe QTDA matrix A succesfully diagonalized ...")

    return E_QTDA, X_QTDA
end