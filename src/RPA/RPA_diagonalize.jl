function RPA_diagonalize(Params::Parameters,Orb_1p1h::Orb2B_1p1h,A::Matrix{Matrix{Float64}},B::Matrix{Matrix{Float64}},N::Matrix{Matrix{Float64}},TrOp::Tr1B)
    # Read calculation parameters ...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1
    Orthogon = Params.Calc.RPA.Ortho

    # Initialize the Stability boolean ...
    Stability = true

    # Initialize the list of values of J & P for iteration ...
    JP_list = JP_initialize(J_max)

    # Determine the largest 2qp space dimension ...
    N_ph_max = maximum(Orb_1p1h.N)

    println("\nStarting diagonalization of the RPA system of equations ...")
    
    # Initialize the storage for RPA solutions ...
    E = Matrix{Vector{ComplexF64}}(undef,2,J_max+1)
    X = Matrix{Matrix{ComplexF64}}(undef,2,J_max+1)
    Y = Matrix{Matrix{ComplexF64}}(undef,2,J_max+1)

    # Initialize temporary working arrays ...
    M1_temp = Matrix{Float64}(undef,N_ph_max,N_ph_max)
    M2_temp = Matrix{Float64}(undef,N_ph_max,N_ph_max)
    M3_temp = Matrix{Float64}(undef,N_ph_max,N_ph_max)
    M4_temp = Matrix{ComplexF64}(undef,N_ph_max,N_ph_max)

    if Orthogon == true
        N_ph_temp = Orb_1p1h.N[2,2]
        M6_temp = Matrix{Float64}(undef,N_ph_temp,N_ph_temp)
        M7_temp = Matrix{Float64}(undef,N_ph_temp,N_ph_temp)
        M8_temp = Matrix{Float64}(undef,N_ph_temp,N_ph_temp)
        M9_temp = Matrix{Float64}(undef,N_ph_temp,N_ph_temp)
        M10_temp = Matrix{Float64}(undef,N_ph_temp,N_ph_temp)
        M11_temp = Matrix{ComplexF64}(undef,N_ph_temp,N_ph_temp)
        M12_temp = Matrix{ComplexF64}(undef,N_ph_temp,N_ph_temp)

        M13_temp = Matrix{Float64}(undef,N_ph_temp,N_ph_temp)
        M14_temp = Matrix{Float64}(undef,N_ph_temp,N_ph_temp)
    end

    # Diagonalization of RPA eigenvalue problem ...
    @inbounds for JP in JP_list
        J, P = JP[1], JP[2]
        N_ph = Orb_1p1h.N[P,J+1]

        A_JP = A[P,J+1]
        B_JP = B[P,J+1]
        N_JP = N[P,J+1]

        X_JP = Matrix{ComplexF64}(undef,N_ph,N_ph)
        Y_JP = Matrix{ComplexF64}(undef,N_ph,N_ph)
        E_JP = Vector{ComplexF64}(undef,N_ph)

        println("\t\tDiagonalizing the block:\tJ = $J, P = $(P == 1 ? "+" : "-")")

        if Orthogon == false || !(J == 1 && P == 2)

            # Renormalize A & B and evaluate inv(sqrt(N)) ...
            sqrtN_JP = sqrt.(N_JP)
            inv_sqrtN_JP = inv(sqrtN_JP)

            A_JP .= inv_sqrtN_JP * A_JP * inv_sqrtN_JP
            B_JP .= inv_sqrtN_JP * B_JP * inv_sqrtN_JP

            @views N_m = M1_temp[1:N_ph,1:N_ph]
            @views N_p = M2_temp[1:N_ph,1:N_ph]

            @views N_p .= A_JP .+ B_JP
            @views N_m .= A_JP .- B_JP

            @views H = M3_temp[1:N_ph,1:N_ph]

            H .= N_m * N_p

            # Solve general NxN non-symmetric real eigenvalue problem (A - B) (A + B) ...
            d = eigen!(H)

            @views R = M3_temp[1:N_ph,1:N_ph]

            E2 = d.values
            R .= d.vectors

            @inbounds for i_nu in 1:N_ph
                E_nu = sqrt(ComplexF64(E2[i_nu]))
                E_JP[i_nu] = E_nu
            end
            
            @views N_p_R = M4_temp[1:N_ph,1:N_ph]

            N_p_R .= N_p * R

            @inbounds for i_nu in 1:N_ph
                E_nu = E_JP[i_nu]
                sqrtE = sqrt(E_nu)
                sE = 0.5 * sqrtE
                iE = 0.5 / sqrtE

                @views rcol  = R[:,i_nu]
                @views npcol = N_p_R[:,i_nu]
                @views xcol  = X_JP[:,i_nu]
                @views ycol  = Y_JP[:,i_nu]

                @. xcol = sE * rcol + iE * npcol
                @. ycol = sE * rcol - iE * npcol
            end

            # Normalization of X and Y amplitudes ...
            @inbounds for i_nu in 1:N_ph
                X_nu, Y_nu = @views X_JP[:,i_nu], @views Y_JP[:,i_nu]
                N_nu = 1.0 / sqrt(abs(X_nu' * N_JP * X_nu - Y_nu' * N_JP * Y_nu))

                # Non-zero norm states ...
                if N_nu > 1e-8
                    @views X_JP[:,i_nu] .= N_nu .* X_nu
                    @views Y_JP[:,i_nu] .= N_nu .* Y_nu
                end

                # Zero-norm states ...
                if N_nu < 1e-8
                    throw("Missing implementation of the zero-norm states ...")
                end
            end

        elseif Orthogon == true && (J == 1 && P == 2)

            X_JP_ort = Matrix{ComplexF64}(undef,N_ph-1,N_ph-1)
            Y_JP_ort = Matrix{ComplexF64}(undef,N_ph-1,N_ph-1)
            E_JP_ort = Vector{ComplexF64}(undef,N_ph-1)

            @views U = M6_temp[1:N_ph,1:N_ph]
            @views V = M7_temp[1:N_ph,1:N_ph]
            @views I = M8_temp[1:N_ph,1:N_ph]

            U .= 0.0
            V .= 0.0
            I .= 0.0

            # Initialize the vector for orthogonalization of spurious 1- level ...
            spurious_level = RPA_spurious_ini(Orb_1p1h,TrOp)

            # Perform the orthogonalization of spurious 1- level ...
            V, spurious_ind = RPA_spurious_ortho(N_ph,spurious_level)

            @inbounds for a in 1:(spurious_ind-1)
                @views U[:,a] = V[:,a]
            end

            @inbounds for a in (spurious_ind+1):N_ph
                @views U[:,a-1] = V[:,a]
            end

            @views A_JP_ort = M7_temp[1:N_ph-1,1:N_ph-1]
            @views B_JP_ort = M8_temp[1:N_ph-1,1:N_ph-1]
            @views N_JP_ort = M13_temp[1:N_ph-1,1:N_ph-1]

            A_JP_ort .= 0.0
            B_JP_ort .= 0.0
            N_JP_ort .= 0.0

            @views U_sub = U[:,1:N_ph-1]

            @views Pm = M9_temp[1:N_ph,1:N_ph-1]
            @views Qm = M10_temp[1:N_ph,1:N_ph-1]
            @views Rm = M14_temp[1:N_ph,1:N_ph-1]

            Pm .= A_JP * U_sub
            Qm .= B_JP * U_sub
            Rm .= N_JP * U_sub

            A_JP_ort .= U_sub' * Pm
            B_JP_ort .= U_sub' * Qm
            N_JP_ort .= U_sub' * Rm

            # Multiply by the overlap matrix ...
            sqrtN_JP_ort = sqrt(N_JP_ort)
            inv_sqrtN_JP_ort = inv(sqrtN_JP_ort)

            A_JP_ort .= inv_sqrtN_JP_ort * A_JP_ort * inv_sqrtN_JP_ort
            B_JP_ort .= inv_sqrtN_JP_ort * B_JP_ort * inv_sqrtN_JP_ort

            # Solve the generalized RPA eigenvalue problem in the orthogonalized subspace ...
            @views N_m = M1_temp[1:N_ph-1,1:N_ph-1]
            @views N_p = M2_temp[1:N_ph-1,1:N_ph-1]

            @views N_p .= A_JP_ort .+ B_JP_ort
            @views N_m .= A_JP_ort .- B_JP_ort

            @views H = M3_temp[1:N_ph-1,1:N_ph-1]

            H .= N_m * N_p

            # Solve general NxN non-symmetric real eigenvalue problem (A - B) (A + B) ...
            d = eigen!(H)

            @views R = M3_temp[1:N_ph-1,1:N_ph-1]

            E2 = d.values
            R .= d.vectors

            @inbounds for i_nu in 1:(N_ph-1)
                E_nu = sqrt(ComplexF64(E2[i_nu]))
                E_JP_ort[i_nu] = E_nu
            end

            @views N_p_R = M4_temp[1:N_ph-1,1:N_ph-1]

            N_p_R .= N_p * R

            @inbounds for i_nu in 1:(N_ph-1)
                E_nu = E_JP_ort[i_nu]
                sqrtE = sqrt(E_nu)
                sE = 0.5 * sqrtE
                iE = 0.5 / sqrtE

                @views rcol  = R[:,i_nu]
                @views npcol = N_p_R[:,i_nu]
                @views xcol  = X_JP_ort[:,i_nu]
                @views ycol  = Y_JP_ort[:,i_nu]

                @. xcol = sE * rcol + iE * npcol
                @. ycol = sE * rcol - iE * npcol
            end

            # Normalization of X and Y amplitudes ...
            @inbounds for i_nu in 1:(N_ph-1)
                X_norm = 0.0
                Y_norm = 0.0
                @inbounds for i_ph in 1:(N_ph-1)
                    X_norm += abs2(X_JP_ort[i_ph,i_nu])
                    Y_norm += abs2(Y_JP_ort[i_ph,i_nu])
                end
                if (X_norm - Y_norm) > 1e-8
                    RPA_norm = ComplexF64(1.0 / sqrt(abs(X_norm - Y_norm)))
                    @inbounds for i_ph in 1:(N_ph-1)
                        X_JP_ort[i_ph,i_nu] = X_JP_ort[i_ph,i_nu] * RPA_norm
                        Y_JP_ort[i_ph,i_nu] = Y_JP_ort[i_ph,i_nu] * RPA_norm
                    end
                elseif (X_norm - Y_norm) < -1e-8
                    RPA_norm = (0.0 - 1.0im) *  ComplexF64(1.0 / sqrt(abs(X_norm - Y_norm)))
                    @inbounds for i_ph in 1:(N_ph-1)
                        X_JP_ort[i_ph,i_nu] = X_JP_ort[i_ph,i_nu] * RPA_norm
                        Y_JP_ort[i_ph,i_nu] = Y_JP_ort[i_ph,i_nu] * RPA_norm
                    end
                else
                    RPA_norm = 0.0
                    @inbounds for i_ph in 1:(N_ph-1)
                        RPA_norm += abs2(X_JP_ort[i_ph,i_nu]) + abs2(Y_JP_ort[i_ph,i_nu])
                    end
                    RPA_norm = ComplexF64(1.0 / sqrt(abs(RPA_norm)))
                    @inbounds for i_ph in 1:(N_ph-1)
                        X_JP_ort[i_ph,i_nu] = X_JP_ort[i_ph,i_nu] * RPA_norm
                        Y_JP_ort[i_ph,i_nu] = Y_JP_ort[i_ph,i_nu] * RPA_norm
                    end
                end
            end

            # Perform renormalization due to the overlap matrix N ...
            @inbounds for i_nu in 1:(N_ph-1)
                @views X_JP_ort[:,i_nu] = inv_sqrtN_JP_ort * X_JP_ort[:,i_nu]
                @views Y_JP_ort[:,i_nu] = inv_sqrtN_JP_ort * Y_JP_ort[:,i_nu]
            end

            # Finish the orthogonalization of RPA amplitudes in the full space ...
            @views Pm = M11_temp[1:N_ph,1:N_ph-1]
            @views Qm = M12_temp[1:N_ph,1:N_ph-1]

            Pm .= 0.0
            Qm .= 0.0

            Pm .= U_sub * X_JP_ort
            Qm .= U_sub * Y_JP_ort

            E_JP = Vector{ComplexF64}(undef,N_ph)
            E_JP[1] = ComplexF64(0.0)
            @inbounds for a in 2:N_ph
                E_JP[a] = E_JP_ort[a-1]
            end

            X_JP = Matrix{ComplexF64}(undef,N_ph,N_ph)
            Y_JP = Matrix{ComplexF64}(undef,N_ph,N_ph)
            @views X_JP[:,1] = spurious_level ./ sqrt(2)
            @views Y_JP[:,1] = -spurious_level ./ sqrt(2)
            @inbounds for a in 2:N_ph
                @views X_JP[:,a] = Pm[:,a-1]
                @views Y_JP[:,a] = Qm[:,a-1]
            end

        end

        # Determine Stability of RPA system ...
        @inbounds for i_nu in 1:N_ph
            if abs(imag(E_JP[i_nu])) > 1e-8
                if P == 2
                    println("\t\t\tRPA INSTABILITY DETECTED in channel:\tJ = $J, P = -")
                else
                    println("\t\t\tRPA INSTABILITY DETECTED in channel:\tJ = $J, P = +")
                end
                Stability = false
            end
        end
        
        E[P,J+1] = E_JP
        X[P,J+1] = X_JP
        Y[P,J+1] = Y_JP
    end

    println("\tFinished diagonalization of the RPA system of equations ...")

    return E, X, Y, Stability
end