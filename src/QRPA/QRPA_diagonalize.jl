function QRPA_diagonalize(Params::Parameters,Orb::Vector{Orb1B},Orb_2qp::qpOrb2B,A::Matrix{Matrix{Float64}},B::Matrix{Matrix{Float64}},qpN::qpO1B,qpTrOp::qpTr1B)
    # Read parameters ...
    N_max = Params.Calc.Nmax
    J_max = 2*N_max + 1
    Orthogon = Params.Calc.QRPA.Ortho

    # Initialize QRPA stability condition ...
    Stability = true

    # Make the list of values of J & P for iteration ...
    JP_list = JP_initialize(J_max)

    # Determine the largest 2qp space dimension ...
    N_qp_max = maximum(Orb_2qp.N)

    # Initialite the storage for QRPA solutions ...
    E_QRPA = Matrix{Vector{ComplexF64}}(undef,2,J_max+1)
    X_QRPA = Matrix{Matrix{ComplexF64}}(undef,2,J_max+1)
    Y_QRPA = Matrix{Matrix{ComplexF64}}(undef,2,J_max+1)

    # Initialize temporary working arrays ...
    M1_temp = Matrix{Float64}(undef,N_qp_max,N_qp_max)
    M2_temp = Matrix{Float64}(undef,N_qp_max,N_qp_max)
    M3_temp = Matrix{Float64}(undef,N_qp_max,N_qp_max)
    M4_temp = Matrix{ComplexF64}(undef,N_qp_max,N_qp_max)


    if Orthogon == true
        N_qp_temp = max(Orb_2qp.N[1,1],Orb_2qp.N[2,2])
        M6_temp = Matrix{Float64}(undef,N_qp_temp,N_qp_temp)
        M7_temp = Matrix{Float64}(undef,N_qp_temp,N_qp_temp)
        M8_temp = Matrix{Float64}(undef,N_qp_temp,N_qp_temp)
        M9_temp = Matrix{Float64}(undef,N_qp_temp,N_qp_temp)
        M10_temp = Matrix{Float64}(undef,N_qp_temp,N_qp_temp)
        M11_temp = Matrix{ComplexF64}(undef,N_qp_temp,N_qp_temp)
        M12_temp = Matrix{ComplexF64}(undef,N_qp_temp,N_qp_temp)
    end


    # Perform the diagonalization of the QRPA system ...
    println("\nDiagonalizing the QRPA system ...")
    @inbounds for JP in JP_list
        J, P = JP[1], JP[2]
        N_qp = Orb_2qp.N[P,J+1]

        A_JP = A[P,J+1]
        B_JP = B[P,J+1]

        if P == 1
            println("\t\tDiagonalizing the block:\tJ = $J, P = +")
        else
            println("\t\tDiagonalizing the block:\tJ = $J, P = -")
        end

        X_QRPA_JP = Matrix{ComplexF64}(undef,N_qp,N_qp)
        Y_QRPA_JP = Matrix{ComplexF64}(undef,N_qp,N_qp)
        E_QRPA_JP = Vector{ComplexF64}(undef,N_qp)

        if Orthogon == false || !((J == 0 && P == 1) || (J == 1 && P == 2))

            @views N_m = M1_temp[1:N_qp,1:N_qp]
            @views N_p = M2_temp[1:N_qp,1:N_qp]

            @views N_p .= A_JP .+ B_JP
            @views N_m .= A_JP .- B_JP

            @views H = M3_temp[1:N_qp,1:N_qp]

            H .= N_m * N_p

            # Solve general NxN non-symmetric real eigenvalue problem (A - B) (A + B) ...
            d = eigen!(H)

            @views R = M3_temp[1:N_qp,1:N_qp]

            E2 = d.values
            R .= d.vectors

            @inbounds for a in 1:N_qp
                E = sqrt(ComplexF64(E2[a]))
                E_QRPA_JP[a] = E
            end
            
            @views N_p_R = M4_temp[1:N_qp,1:N_qp]

            N_p_R .= N_p * R

            @inbounds for nu in 1:N_qp
                E_nu = E_QRPA_JP[nu]
                sqrtE = sqrt(E_nu)
                sE = 0.5 * sqrtE
                iE = 0.5 / sqrtE

                @views rcol  = R[:,nu]
                @views npcol = N_p_R[:,nu]
                @views xcol  = X_QRPA_JP[:,nu]
                @views ycol  = Y_QRPA_JP[:,nu]

                @. xcol = sE * rcol + iE * npcol
                @. ycol = sE * rcol - iE * npcol
            end

        elseif Orthogon == true && J == 0 && P == 1

            X_QRPA_JP_ort = Matrix{ComplexF64}(undef,N_qp-1,N_qp-1)
            Y_QRPA_JP_ort = Matrix{ComplexF64}(undef,N_qp-1,N_qp-1)
            E_QRPA_JP_ort = Vector{ComplexF64}(undef,N_qp-1)

            @views U = M6_temp[1:N_qp,1:N_qp]
            @views V = M7_temp[1:N_qp,1:N_qp]
            @views I = M8_temp[1:N_qp,1:N_qp]

            U .= 0.0
            V .= 0.0
            I .= 0.0

            # Initialize the vector for orthogonalization of spurious 0+ level ...
            Spur_PN = QRPA_spurious_PN_initialize(Orb,Orb_2qp,qpN)

            # Perform the orthogonalization of spurious 0+ level ...
            V, Spur_ind = QRPA_spurious_orthogonalize(N_qp,Spur_PN,V,I)

            @inbounds for a in 1:(Spur_ind-1)
                @views U[:,a] = V[:,a]
            end

            @inbounds for a in (Spur_ind+1):N_qp
                @views U[:,a-1] = V[:,a]
            end

            @views A_JP_ort = M7_temp[1:N_qp-1,1:N_qp-1]
            @views B_JP_ort = M8_temp[1:N_qp-1,1:N_qp-1]

            A_JP_ort .= 0.0
            B_JP_ort .= 0.0

            @views U_sub = U[:,1:N_qp-1]

            @views Pm = M9_temp[1:N_qp,1:N_qp-1]
            @views Qm = M10_temp[1:N_qp,1:N_qp-1]

            Pm .= A_JP * U_sub
            Qm .= B_JP * U_sub

            A_JP_ort .= U_sub' * Pm
            B_JP_ort .= U_sub' * Qm

            #mul!(Pm, A_JP, U_sub)
            #mul!(Qm, B_JP, U_sub)

            #mul!(A_JP_ort, transpose(U_sub), Pm)
            #mul!(B_JP_ort, transpose(U_sub), Qm)

            # Solve the generalized QRPA eigenvalue problem in the orthogonalized subspace ...
            @views N_m = M1_temp[1:N_qp-1,1:N_qp-1]
            @views N_p = M2_temp[1:N_qp-1,1:N_qp-1]

            @views N_p .= A_JP_ort .+ B_JP_ort
            @views N_m .= A_JP_ort .- B_JP_ort

            @views H = M3_temp[1:N_qp-1,1:N_qp-1]

            H .= N_m * N_p

            # Solve general NxN non-symmetric real eigenvalue problem (A - B) (A + B) ...
            d = eigen!(H)

            @views R = M3_temp[1:N_qp-1,1:N_qp-1]

            E2 = d.values
            R .= d.vectors

            @inbounds for a in 1:(N_qp-1)
                E = sqrt(ComplexF64(E2[a]))
                E_QRPA_JP_ort[a] = E
            end
            
            @views N_p_R = M4_temp[1:N_qp-1,1:N_qp-1]

            N_p_R .= N_p * R

            @inbounds for nu in 1:(N_qp-1)
                E_nu = E_QRPA_JP_ort[nu]
                sqrtE = sqrt(E_nu)
                sE = 0.5 * sqrtE
                iE = 0.5 / sqrtE

                @views rcol  = R[:,nu]
                @views npcol = N_p_R[:,nu]
                @views xcol  = X_QRPA_JP_ort[:,nu]
                @views ycol  = Y_QRPA_JP_ort[:,nu]

                @. xcol = sE * rcol + iE * npcol
                @. ycol = sE * rcol - iE * npcol
            end

            # Finish the orthogonalization of QRPA amplitudes in the full space ...

            @views Pm = M11_temp[1:N_qp,1:N_qp-1]
            @views Qm = M12_temp[1:N_qp,1:N_qp-1]

            Pm .= 0.0
            Qm .= 0.0

            Pm .= U_sub * X_QRPA_JP_ort
            Qm .= U_sub * Y_QRPA_JP_ort

            #mul!(Pm, U_sub, X_QRPA_JP_ort)
            #mul!(Qm, U_sub, Y_QRPA_JP_ort)

            E_QRPA_JP = Vector{ComplexF64}(undef,N_qp)
            E_QRPA_JP[1] = ComplexF64(0.0)
            @inbounds for a in 2:N_qp
                E_QRPA_JP[a] = E_QRPA_JP_ort[a-1]
            end

            X_QRPA_JP = Matrix{ComplexF64}(undef,N_qp,N_qp)
            Y_QRPA_JP = Matrix{ComplexF64}(undef,N_qp,N_qp)
            @views X_QRPA_JP[:,1] = Spur_PN ./ sqrt(2)
            @views Y_QRPA_JP[:,1] = -Spur_PN ./ sqrt(2)
            @inbounds for a in 2:N_qp
                @views X_QRPA_JP[:,a] = Pm[:,a-1]
                @views Y_QRPA_JP[:,a] = Qm[:,a-1]
            end
            
        elseif Orthogon == true && J == 1 && P == 2

            X_QRPA_JP_ort = Matrix{ComplexF64}(undef,N_qp-1,N_qp-1)
            Y_QRPA_JP_ort = Matrix{ComplexF64}(undef,N_qp-1,N_qp-1)
            E_QRPA_JP_ort = Vector{ComplexF64}(undef,N_qp-1)

            @views U = M6_temp[1:N_qp,1:N_qp]
            @views V = M7_temp[1:N_qp,1:N_qp]
            @views I = M8_temp[1:N_qp,1:N_qp]

            U .= 0.0
            V .= 0.0
            I .= 0.0

            # Initialize the vector for orthogonalization of spurious 0+ level ...
            Spur_CM = QRPA_spurious_CM_initialize(Orb_2qp,qpTrOp)

            # Perform the orthogonalization of spurious 0+ level ...
            V, Spur_ind = QRPA_spurious_orthogonalize(N_qp,Spur_CM,V,I)

            @inbounds for a in 1:(Spur_ind-1)
                @views U[:,a] = V[:,a]
            end

            @inbounds for a in (Spur_ind+1):N_qp
                @views U[:,a-1] = V[:,a]
            end

            @views A_JP_ort = M7_temp[1:N_qp-1,1:N_qp-1]
            @views B_JP_ort = M8_temp[1:N_qp-1,1:N_qp-1]

            A_JP_ort .= 0.0
            B_JP_ort .= 0.0

            @views U_sub = U[:,1:N_qp-1]

            @views Pm = M9_temp[1:N_qp,1:N_qp-1]
            @views Qm = M10_temp[1:N_qp,1:N_qp-1]

            Pm .= A_JP * U_sub
            Qm .= B_JP * U_sub

            A_JP_ort .= U_sub' * Pm
            B_JP_ort .= U_sub' * Qm

            #mul!(Pm, A_JP, U_sub)
            #mul!(Qm, B_JP, U_sub)

            #mul!(A_JP_ort, transpose(U_sub), Pm)
            #mul!(B_JP_ort, transpose(U_sub), Qm)

            # Solve the generalized QRPA eigenvalue problem in the orthogonalized subspace ...
            @views N_m = M1_temp[1:N_qp-1,1:N_qp-1]
            @views N_p = M2_temp[1:N_qp-1,1:N_qp-1]

            @views N_p .= A_JP_ort .+ B_JP_ort
            @views N_m .= A_JP_ort .- B_JP_ort

            @views H = M3_temp[1:N_qp-1,1:N_qp-1]

            H .= N_m * N_p

            # Solve general NxN non-symmetric real eigenvalue problem (A - B) (A + B) ...
            d = eigen!(H)

            @views R = M3_temp[1:N_qp-1,1:N_qp-1]

            E2 = d.values
            R .= d.vectors

            @inbounds for a in 1:(N_qp-1)
                E = sqrt(ComplexF64(E2[a]))
                E_QRPA_JP_ort[a] = E
            end
            
            @views N_p_R = M4_temp[1:N_qp-1,1:N_qp-1]

            N_p_R .= N_p * R

            @inbounds for nu in 1:(N_qp-1)
                E_nu = E_QRPA_JP_ort[nu]
                sqrtE = sqrt(E_nu)
                sE = 0.5 * sqrtE
                iE = 0.5 / sqrtE

                @views rcol  = R[:,nu]
                @views npcol = N_p_R[:,nu]
                @views xcol  = X_QRPA_JP_ort[:,nu]
                @views ycol  = Y_QRPA_JP_ort[:,nu]

                @. xcol = sE * rcol + iE * npcol
                @. ycol = sE * rcol - iE * npcol
            end

            # Finish the orthogonalization of QRPA amplitudes in the full space ...

            @views Pm = M11_temp[1:N_qp,1:N_qp-1]
            @views Qm = M12_temp[1:N_qp,1:N_qp-1]

            Pm .= 0.0
            Qm .= 0.0

            Pm .= U_sub * X_QRPA_JP_ort
            Qm .= U_sub * Y_QRPA_JP_ort

            #mul!(Pm, U_sub, X_QRPA_JP_ort)
            #mul!(Qm, U_sub, Y_QRPA_JP_ort)

            E_QRPA_JP = Vector{ComplexF64}(undef,N_qp)
            E_QRPA_JP[1] = ComplexF64(0.0)
            @inbounds for a in 2:N_qp
                E_QRPA_JP[a] = E_QRPA_JP_ort[a-1]
            end

            X_QRPA_JP = Matrix{ComplexF64}(undef,N_qp,N_qp)
            Y_QRPA_JP = Matrix{ComplexF64}(undef,N_qp,N_qp)
            @views X_QRPA_JP[:,1] = Spur_CM ./ sqrt(2)
            @views Y_QRPA_JP[:,1] = -Spur_CM ./ sqrt(2)
            @inbounds for a in 2:N_qp
                @views X_QRPA_JP[:,a] = Pm[:,a-1]
                @views Y_QRPA_JP[:,a] = Qm[:,a-1]
            end

        end

        # Renormalization of X and Y amplitudes ...
        @inbounds for nu in 1:N_qp
            X_norm = 0.0
            Y_norm = 0.0
            @inbounds for qp in 1:N_qp
                X_norm += abs2(X_QRPA_JP[qp,nu])
                Y_norm += abs2(Y_QRPA_JP[qp,nu])
            end
            if (X_norm - Y_norm) > 1e-8
                QRPA_norm = ComplexF64(1.0 / sqrt(abs(X_norm - Y_norm)))
                @inbounds for qp in 1:N_qp
                    X_QRPA_JP[qp,nu] = X_QRPA_JP[qp,nu] * QRPA_norm
                    Y_QRPA_JP[qp,nu] = Y_QRPA_JP[qp,nu] * QRPA_norm
                end
            elseif (X_norm - Y_norm) < -1e-8
                QRPA_norm = (0.0 - 1.0im) *  ComplexF64(1.0 / sqrt(abs(X_norm - Y_norm)))
                @inbounds for qp in 1:N_qp
                    X_QRPA_JP[qp,nu] = X_QRPA_JP[qp,nu] * QRPA_norm
                    Y_QRPA_JP[qp,nu] = Y_QRPA_JP[qp,nu] * QRPA_norm
                end
            else
                QRPA_norm = 0.0
                @inbounds for qp in 1:N_qp
                    QRPA_norm += abs2(X_QRPA_JP[qp,nu]) + abs2(Y_QRPA_JP[qp,nu])
                end
                QRPA_norm = ComplexF64(1.0 / sqrt(abs(QRPA_norm)))
                @inbounds for qp in 1:N_qp
                    X_QRPA_JP[qp,nu] = X_QRPA_JP[qp,nu] * QRPA_norm
                    Y_QRPA_JP[qp,nu] = Y_QRPA_JP[qp,nu] * QRPA_norm
                end
            end
        end

        # Determine stability of QRPA system ...
        @inbounds for nu in 1:N_qp
            if abs(imag(E_QRPA_JP[nu])) > 1e-8
                if P == 1
                    println("\t\t\tQRPA instability detected in channel:\tJ = $J, P = +")
                elseif P == 2
                    println("\t\t\tQRPA instability detected in channel:\tJ = $J, P = -")
                end
                Stability = false
            end
        end

        # Retrieve the QRPA solutions ...
        E_QRPA[P,J+1] = E_QRPA_JP
        X_QRPA[P,J+1] = X_QRPA_JP
        Y_QRPA[P,J+1] = Y_QRPA_JP

    end

    println("\tThe system of QRPA equations has been succesfully solved ...")

    return E_QRPA, X_QRPA, Y_QRPA, Stability
end