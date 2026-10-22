function RPA_diagonalize(Params::Parameters,A::Matrix{Matrix{Float64}},B::Matrix{Matrix{Float64}},N::Matrix{Matrix{Float64}},N_nu::Matrix{Int64},Orb_Phonon::Matrix{Vector{Int64}},Phonon::Vector{PhState},Particle::pnSVector,Hole::pnSVector,TrOp::Tr1B)
    # Read calculation parameters ...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1
    Orthogon = Params.Calc.RPA.Ortho

    # Initialize the stability boolean ...
    Stability = true

    # Initialite the list of values of J & P for iteration ...
    JP_list = JP_initialize(J_max)

    # Determine the largest 2qp space dimension ...
    N_ph_max = maximum(N_nu)

    println("\nDiagonalizing the RPA system ...")
    
    # Initialite the storage for RPA solutions ...
    E_RPA = Matrix{Vector{ComplexF64}}(undef,J_max+1,2)
    X_RPA = Matrix{Matrix{ComplexF64}}(undef,J_max+1,2)
    Y_RPA = Matrix{Matrix{ComplexF64}}(undef,J_max+1,2)

    # Initialize temporary working arrays ...
    M1_temp = Matrix{Float64}(undef,N_ph_max,N_ph_max)
    M2_temp = Matrix{Float64}(undef,N_ph_max,N_ph_max)
    M3_temp = Matrix{Float64}(undef,N_ph_max,N_ph_max)
    M4_temp = Matrix{ComplexF64}(undef,N_ph_max,N_ph_max)


    if Orthogon == true
        N_ph_temp = N_nu[2,2]
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
        N_ph = N_nu[J+1,P]

        A_JP = A[J+1,P]
        B_JP = B[J+1,P]
        N_JP = N[J+1,P]

        X_RPA_JP = Matrix{ComplexF64}(undef,N_ph,N_ph)
        Y_RPA_JP = Matrix{ComplexF64}(undef,N_ph,N_ph)
        E_RPA_JP = Vector{ComplexF64}(undef,N_ph)

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

            @inbounds for a in 1:N_ph
                E = sqrt(ComplexF64(E2[a]))
                E_RPA_JP[a] = E
            end
            
            @views N_p_R = M4_temp[1:N_ph,1:N_ph]

            N_p_R .= N_p * R

            @inbounds for nu in 1:N_ph
                E_nu = E_RPA_JP[nu]
                sqrtE = sqrt(E_nu)
                sE = 0.5 * sqrtE
                iE = 0.5 / sqrtE

                @views rcol  = R[:,nu]
                @views npcol = N_p_R[:,nu]
                @views xcol  = X_RPA_JP[:,nu]
                @views ycol  = Y_RPA_JP[:,nu]

                @. xcol = sE * rcol + iE * npcol
                @. ycol = sE * rcol - iE * npcol
            end

            # Normalization of X and Y amplitudes ...
            @inbounds for nu in 1:N_ph
                X_norm = 0.0
                Y_norm = 0.0
                @inbounds for ph in 1:N_ph
                    X_norm += abs2(X_RPA_JP[ph,nu])
                    Y_norm += abs2(Y_RPA_JP[ph,nu])
                end
                if (X_norm - Y_norm) > 1e-8
                    RPA_norm = ComplexF64(1.0 / sqrt(abs(X_norm - Y_norm)))
                    @inbounds for ph in 1:N_ph
                        X_RPA_JP[ph,nu] = X_RPA_JP[ph,nu] * RPA_norm
                        Y_RPA_JP[ph,nu] = Y_RPA_JP[ph,nu] * RPA_norm
                    end
                elseif (X_norm - Y_norm) < -1e-8
                    RPA_norm = (0.0 - 1.0im) *  ComplexF64(1.0 / sqrt(abs(X_norm - Y_norm)))
                    @inbounds for ph in 1:N_ph
                        X_RPA_JP[ph,nu] = X_RPA_JP[ph,nu] * RPA_norm
                        Y_RPA_JP[ph,nu] = Y_RPA_JP[ph,nu] * RPA_norm
                    end
                else
                    RPA_norm = 0.0
                    @inbounds for ph in 1:N_ph
                        RPA_norm += abs2(X_RPA_JP[ph,nu]) + abs2(Y_RPA_JP[ph,nu])
                    end
                    RPA_norm = ComplexF64(1.0 / sqrt(abs(RPA_norm)))
                    @inbounds for ph in 1:N_ph
                        X_RPA_JP[ph,nu] = X_RPA_JP[ph,nu] * RPA_norm
                        Y_RPA_JP[ph,nu] = Y_RPA_JP[ph,nu] * RPA_norm
                    end
                end
            end

            # Perform renormalization due to the overlap matrix N ...
            @inbounds for nu in 1:N_ph
                @inbounds for ph in 1:N_ph
                    inv_dn_ph = 1.0 / sqrt(N_JP[ph,ph])
                    X_RPA_JP[ph,nu] = X_RPA_JP[ph,nu] * inv_dn_ph
                    Y_RPA_JP[ph,nu] = Y_RPA_JP[ph,nu] * inv_dn_ph
                end
            end

        elseif Orthogon == true && (J == 1 && P == 2)

            X_RPA_JP_ort = Matrix{ComplexF64}(undef,N_ph-1,N_ph-1)
            Y_RPA_JP_ort = Matrix{ComplexF64}(undef,N_ph-1,N_ph-1)
            E_RPA_JP_ort = Vector{ComplexF64}(undef,N_ph-1)

            @views U = M6_temp[1:N_ph,1:N_ph]
            @views V = M7_temp[1:N_ph,1:N_ph]
            @views I = M8_temp[1:N_ph,1:N_ph]

            U .= 0.0
            V .= 0.0
            I .= 0.0

            # Initialize the vector for orthogonalization of spurious 0+ level ...
            Spur_State = RPA_Spur_Ini(N_ph,Orb_Phonon[2,2],Phonon,Particle,Hole,TrOp)

            # Perform the orthogonalization of spurious 0+ level ...
            V, Spur_ind = RPA_Spur_Ortho(N_ph,Spur_State)

            @inbounds for a in 1:(Spur_ind-1)
                @views U[:,a] = V[:,a]
            end

            @inbounds for a in (Spur_ind+1):N_ph
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

            #display(N_JP_ort)
            #display(sqrtN_JP_ort)
            #display(inv_sqrtN_JP_ort)
            #throw("stop here")

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

            @inbounds for a in 1:(N_ph-1)
                E = sqrt(ComplexF64(E2[a]))
                E_RPA_JP_ort[a] = E
            end

            @views N_p_R = M4_temp[1:N_ph-1,1:N_ph-1]

            N_p_R .= N_p * R

            @inbounds for nu in 1:(N_ph-1)
                E_nu = E_RPA_JP_ort[nu]
                sqrtE = sqrt(E_nu)
                sE = 0.5 * sqrtE
                iE = 0.5 / sqrtE

                @views rcol  = R[:,nu]
                @views npcol = N_p_R[:,nu]
                @views xcol  = X_RPA_JP_ort[:,nu]
                @views ycol  = Y_RPA_JP_ort[:,nu]

                @. xcol = sE * rcol + iE * npcol
                @. ycol = sE * rcol - iE * npcol
            end



                        # Normalization of X and Y amplitudes ...
                        @inbounds for nu in 1:(N_ph-1)
                            X_norm = 0.0
                            Y_norm = 0.0
                            @inbounds for ph in 1:(N_ph-1)
                                X_norm += abs2(X_RPA_JP_ort[ph,nu])
                                Y_norm += abs2(Y_RPA_JP_ort[ph,nu])
                            end
                            if (X_norm - Y_norm) > 1e-8
                                RPA_norm = ComplexF64(1.0 / sqrt(abs(X_norm - Y_norm)))
                                @inbounds for ph in 1:(N_ph-1)
                                    X_RPA_JP_ort[ph,nu] = X_RPA_JP_ort[ph,nu] * RPA_norm
                                    Y_RPA_JP_ort[ph,nu] = Y_RPA_JP_ort[ph,nu] * RPA_norm
                                end
                            elseif (X_norm - Y_norm) < -1e-8
                                RPA_norm = (0.0 - 1.0im) *  ComplexF64(1.0 / sqrt(abs(X_norm - Y_norm)))
                                @inbounds for ph in 1:(N_ph-1)
                                    X_RPA_JP_ort[ph,nu] = X_RPA_JP_ort[ph,nu] * RPA_norm
                                    Y_RPA_JP_ort[ph,nu] = Y_RPA_JP_ort[ph,nu] * RPA_norm
                                end
                            else
                                RPA_norm = 0.0
                                @inbounds for ph in 1:(N_ph-1)
                                    RPA_norm += abs2(X_RPA_JP_ort[ph,nu]) + abs2(Y_RPA_JP_ort[ph,nu])
                                end
                                RPA_norm = ComplexF64(1.0 / sqrt(abs(RPA_norm)))
                                @inbounds for ph in 1:(N_ph-1)
                                    X_RPA_JP_ort[ph,nu] = X_RPA_JP_ort[ph,nu] * RPA_norm
                                    Y_RPA_JP_ort[ph,nu] = Y_RPA_JP_ort[ph,nu] * RPA_norm
                                end
                            end
                        end

                        # Perform renormalization due to the overlap matrix N ...
                        @inbounds for nu in 1:(N_ph-1)
                            @views X_RPA_JP_ort[:,nu] = inv_sqrtN_JP_ort * X_RPA_JP_ort[:,nu]
                            @views Y_RPA_JP_ort[:,nu] = inv_sqrtN_JP_ort * Y_RPA_JP_ort[:,nu]
                        end




            # Finish the orthogonalization of RPA amplitudes in the full space ...
            @views Pm = M11_temp[1:N_ph,1:N_ph-1]
            @views Qm = M12_temp[1:N_ph,1:N_ph-1]

            Pm .= 0.0
            Qm .= 0.0

            Pm .= U_sub * X_RPA_JP_ort
            Qm .= U_sub * Y_RPA_JP_ort

            E_RPA_JP = Vector{ComplexF64}(undef,N_ph)
            E_RPA_JP[1] = ComplexF64(0.0)
            @inbounds for a in 2:N_ph
                E_RPA_JP[a] = E_RPA_JP_ort[a-1]
            end

            X_RPA_JP = Matrix{ComplexF64}(undef,N_ph,N_ph)
            Y_RPA_JP = Matrix{ComplexF64}(undef,N_ph,N_ph)
            @views X_RPA_JP[:,1] = Spur_State ./ sqrt(2)
            @views Y_RPA_JP[:,1] = -Spur_State ./ sqrt(2)
            @inbounds for a in 2:N_ph
                @views X_RPA_JP[:,a] = Pm[:,a-1]
                @views Y_RPA_JP[:,a] = Qm[:,a-1]
            end



        end

        # Determine stability of RPA system ...
        @inbounds for nu in 1:N_ph
            if abs(imag(E_RPA_JP[nu])) > 1e-8
                println("\t\t\tRPA INSTABILITY DETECTED in channel:\tJ = $J, P = +")
                Stability = false
            end
        end
        
        E_RPA[J+1,P] = E_RPA_JP
        X_RPA[J+1,P] = X_RPA_JP
        Y_RPA[J+1,P] = Y_RPA_JP
    end

    println("\nDiagonalization done ...")

    return E_RPA, X_RPA, Y_RPA, Stability
end

function RPA_diagonalize_Old(Params::Parameters,A::Matrix{Matrix{Float64}},B::Matrix{Matrix{Float64}},N::Matrix{Matrix{Float64}},N_nu::Matrix{Int64},Orb_Phonon::Matrix{Vector{Int64}},Phonon::Vector{PhState},Particle::pnSVector,Hole::pnSVector,TrOp::Tr1B)
    # Read calculation parameters ...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1
    Orthogon = Params.Calc.RPA.Ortho

    # Initialize the stability boolean ...
    Stability = true

    # Initialite the list of values of J & P for iteration ...
    JP_list = JP_initialize(J_max)

    # Determine the largest 2qp space dimension ...
    N_ph_max = maximum(N_nu)

    println("\nDiagonalizing the RPA system ...")
    
    # Initialite the storage for RPA solutions ...
    E_RPA = Matrix{Vector{ComplexF64}}(undef,2,J_max+1)
    X_RPA = Matrix{Matrix{ComplexF64}}(undef,2,J_max+1)
    Y_RPA = Matrix{Matrix{ComplexF64}}(undef,2,J_max+1)

    # Initialize temporary working arrays ...
    M1_temp = Matrix{Float64}(undef,N_ph_max,N_ph_max)
    M2_temp = Matrix{Float64}(undef,N_ph_max,N_ph_max)
    M3_temp = Matrix{Float64}(undef,N_ph_max,N_ph_max)
    M4_temp = Matrix{ComplexF64}(undef,N_ph_max,N_ph_max)


    if Orthogon == true
        N_ph_temp = N_nu[2,2]
        M6_temp = Matrix{Float64}(undef,N_ph_temp,N_ph_temp)
        M7_temp = Matrix{Float64}(undef,N_ph_temp,N_ph_temp)
        M8_temp = Matrix{Float64}(undef,N_ph_temp,N_ph_temp)
        M9_temp = Matrix{Float64}(undef,N_ph_temp,N_ph_temp)
        M10_temp = Matrix{Float64}(undef,N_ph_temp,N_ph_temp)
        M11_temp = Matrix{ComplexF64}(undef,N_ph_temp,N_ph_temp)
        M12_temp = Matrix{ComplexF64}(undef,N_ph_temp,N_ph_temp)
    end

    # Diagonalization of RPA eigenvalue problem ...
    @inbounds Threads.@threads for JP in JP_list
        J, P = JP[1], JP[2]
        N_ph = N_nu[J+1,P]

        A_JP = A[J+1,P]
        B_JP = B[J+1,P]
        N_JP = N[J+1,P]

        println("\t\tDiagonalizing the block:\tJ = $J, P = $(P == 1 ? "+" : "-")")

        if (Orthogon == true) && (J == 1) && (P == 2)
            Spur_State = RPA_Spur_Ini(N_ph,Orb_Phonon[J+1,P],Phonon,Particle,Hole,TrOp)
            V, Spur_ind = RPA_Spur_Ortho(N_ph,Spur_State)
            U = zeros(Float64,N_ph,N_ph)

            @inbounds for a in 1:(Spur_ind-1)
                @views U[:,a] = V[:,a]
            end

            @inbounds for a in (Spur_ind+1):N_ph
                @views U[:,a-1] = V[:,a]
            end

            A_JP_ort = zeros(Float64,N_ph - 1,N_ph - 1)
            B_JP_ort = zeros(Float64,N_ph - 1,N_ph - 1)

            Pm = zeros(Float64,N_ph,N_ph-1)
            Qm = zeros(Float64,N_ph,N_ph-1)

            @inbounds for a in 1:N_ph
                @inbounds for b in 1:(N_ph-1)
                    @inbounds for c in 1:N_ph
                        Pm[a,b] = Pm[a,b] + A_JP[a,c] * U[c,b]
                        Qm[a,b] = Qm[a,b] + B_JP[a,c] * U[c,b]
                    end
                end
            end

            @inbounds for a in 1:(N_ph-1)
                @inbounds for b in 1:(N_ph-1)
                    @inbounds for c in 1:N_ph
                        A_JP_ort[a,b] = A_JP_ort[a,b] + U[c,a] * Pm[c,b]
                        B_JP_ort[a,b] = B_JP_ort[a,b] + U[c,a] * Qm[c,b]
                    end
                end
            end

            A_JP = A_JP_ort
            B_JP = B_JP_ort

            # Solve RPA eigen-value problem ...
            X_RPA_JP = Matrix{ComplexF64}(undef,N_ph-1,N_ph-1)
            Y_RPA_JP = Matrix{ComplexF64}(undef,N_ph-1,N_ph-1)

            N_p = @views A_JP .+ B_JP
            N_m = @views A_JP .- B_JP

            # (1) Case of A - B positive-semidefinite ....
            if isposdef(N_m)
                D_m, T = eigen(N_m)
                sqrtD_m = diagm(sqrt.(D_m))
                invsqrtD_m = diagm(1.0 ./ sqrt.(D_m))
                D_m = diagm(D_m)
                
                M_p = sqrtD_m * T' * N_p * T * sqrtD_m

                E_RPA_JP, R = eigen(M_p)

                sort_indices = sortperm(E_RPA_JP, by = x -> real(x))
                @views E_RPA_JP = Complex.(E_RPA_JP[sort_indices])
                R = @views R[:, sort_indices]
    
                @views E_RPA_JP = sqrt.(Complex.(E_RPA_JP))
    
                @inbounds for ph in 1:(N_ph-1)
                    @views X_RPA_JP[:,ph] .= 0.5 .* T * (sqrtD_m ./ E_RPA_JP[ph] .+ invsqrtD_m) * R[:,ph]
                    @views Y_RPA_JP[:,ph] .= 0.5 .* T * (sqrtD_m ./ E_RPA_JP[ph] .- invsqrtD_m) * R[:,ph]
                end
            # (2) Case of A + B positive-semidefinite ...  
            elseif isposdef(N_p)
                D_p, T = eigen(N_p)
                sqrtD_p = diagm(sqrt.(D_p))
                invsqrtD_p = diagm(1.0 ./ sqrt.(D_p))
                D_p = diagm(D_p)
            
                M_m = sqrtD_p * T' * N_m * T * sqrtD_p

                E_RPA_JP, R = eigen(M_m)

                sort_indices = sortperm(E_RPA_JP, by = x -> real(x))
                @views E_RPA_JP = Complex.(E_RPA_JP[sort_indices])
                R = @views R[:, sort_indices]
    
                @views E_RPA_JP = sqrt.(Complex.(E_RPA_JP))
    
                @inbounds for ph in 1:(N_ph-1)
                    @views X_RPA_JP[:,ph] .= 0.5 .* T * (sqrtD_p ./ E_RPA_JP[ph] .+ invsqrtD_p) * R[:,ph]
                    @views Y_RPA_JP[:,ph] .= 0.5 .* T * (sqrtD_p ./ E_RPA_JP[ph] .- invsqrtD_p) * R[:,ph]
                end
            # (3) General case with complex matrices ...
            else
                D_m, T = eigen(N_m)
                sqrtD_m = diagm(sqrt.(Complex.(D_m)))
                invsqrtD_m = diagm(1.0 ./ sqrt.(Complex.(D_m)))
                D_m = diagm(D_m)
                
                M_p = sqrtD_m * T' * N_p * T * sqrtD_m
                
                E_RPA_JP, R = eigen(M_p)

                sort_indices = sortperm(E_RPA_JP, by = x -> real(x))
                @views E_RPA_JP = Complex.(E_RPA_JP[sort_indices])
                R = @views R[:, sort_indices]
    
                @views E_RPA_JP = sqrt.(Complex.(E_RPA_JP))
    
                @inbounds for ph in 1:(N_ph-1)
                    @views X_RPA_JP[:,ph] .= 0.5 .* T * (sqrtD_m ./ E_RPA_JP[ph] .+ invsqrtD_m) * R[:,ph]
                    @views Y_RPA_JP[:,ph] .= 0.5 .* T * (sqrtD_m ./ E_RPA_JP[ph] .- invsqrtD_m) * R[:,ph]
                end
            end

            Pm = zeros(ComplexF64,N_ph,N_ph-1)
            Qm = zeros(ComplexF64,N_ph,N_ph-1)

            @inbounds for a in 1:N_ph
                @inbounds for b in 1:(N_ph -1)
                    @inbounds for c in 1:(N_ph-1)
                        Pm[a,b] += ComplexF64(U[a,c]) * X_RPA_JP[c,b]
                        Qm[a,b] += ComplexF64(U[a,c]) * Y_RPA_JP[c,b]
                    end
                end
            end

            E_RPA_JP_new = zeros(ComplexF64,N_ph)
            @inbounds for a in 2:N_ph
                E_RPA_JP_new[a] = E_RPA_JP[a-1]
            end

            X_RPA_JP_new = zeros(ComplexF64, N_ph, N_ph)
            Y_RPA_JP_new = zeros(ComplexF64, N_ph, N_ph)

            @views X_RPA_JP_new[:,1] = Spur_State / sqrt(2)
            @views Y_RPA_JP_new[:,1] = -1.0 *  Spur_State / sqrt(2)

            @inbounds for a in 2:N_ph
                @views X_RPA_JP_new[:,a] = Pm[:,a-1]
                @views Y_RPA_JP_new[:,a] = Qm[:,a-1]
            end

            X_RPA_JP = X_RPA_JP_new
            Y_RPA_JP = Y_RPA_JP_new
            E_RPA_JP = E_RPA_JP_new

        else
            # Solve RPA eigen-value problem ...

            X_RPA_JP = Matrix{ComplexF64}(undef,N_ph,N_ph)
            Y_RPA_JP = Matrix{ComplexF64}(undef,N_ph,N_ph)

            N_p = A_JP .+ B_JP
            N_m = A_JP .- B_JP

            # (1) Case of A - B positive-semidefinite ....
            if isposdef(N_m)
                D_m, T = eigen(N_m)
                sqrtD_m = diagm(sqrt.(D_m))
                invsqrtD_m = diagm(1.0 ./ sqrt.(D_m))
                D_m = diagm(D_m)
                
                M_p = sqrtD_m * T' * N_p * T * sqrtD_m

                E_RPA_JP, R = eigen(M_p)

                sort_indices = sortperm(E_RPA_JP, by = x -> real(x))
                @views E_RPA_JP = Complex.(E_RPA_JP[sort_indices])
                R = @views R[:, sort_indices]
    
                @views E_RPA_JP = sqrt.(Complex.(E_RPA_JP))
    
                @inbounds for ph in 1:N_ph
                    @views X_RPA_JP[:,ph] = 0.5 .* T * (sqrtD_m ./ E_RPA_JP[ph] .+ invsqrtD_m) * R[:,ph]
                    @views Y_RPA_JP[:,ph] = 0.5 .* T * (sqrtD_m ./ E_RPA_JP[ph] .- invsqrtD_m) * R[:,ph]
                end
            # (2) Case of A + B positive-semidefinite ...  
            elseif isposdef(N_p)
                D_p, T = eigen(N_p)
                sqrtD_p = diagm(sqrt.(D_p))
                invsqrtD_p = diagm(1.0 ./ sqrt.(D_p))
                D_p = diagm(D_p)
            
                M_m = sqrtD_p * T' * N_m * T * sqrtD_p

                E_RPA_JP, R = eigen(M_m)

                sort_indices = sortperm(E_RPA_JP, by = x -> real(x))
                @views E_RPA_JP = Complex.(E_RPA_JP[sort_indices])
                R = @views R[:, sort_indices]
    
                @views E_RPA_JP = sqrt.(Complex.(E_RPA_JP))
    
                @inbounds for ph in 1:N_ph
                    @views X_RPA_JP[:,ph] .= 0.5 .* T * (sqrtD_p ./ E_RPA_JP[ph] .+ invsqrtD_p) * R[:,ph]
                    @views Y_RPA_JP[:,ph] .= 0.5 .* T * (sqrtD_p ./ E_RPA_JP[ph] .- invsqrtD_p) * R[:,ph]
                end
            # (3) General case with complex matrices ...
            else
                D_m, T = eigen(N_m)
                sqrtD_m = diagm(sqrt.(Complex.(D_m)))
                invsqrtD_m = diagm(1.0 ./ sqrt.(Complex.(D_m)))
                D_m = diagm(D_m)
                
                M_p = sqrtD_m * T' * N_p * T * sqrtD_m

                E_RPA_JP, R = eigen(M_p)

                sort_indices = sortperm(E_RPA_JP, by = x -> real(x))
                @views E_RPA_JP = Complex.(E_RPA_JP[sort_indices])
                R = @views R[:, sort_indices]
    
                @views E_RPA_JP = sqrt.(Complex.(E_RPA_JP))
    
                @inbounds for ph in 1:N_ph
                    @views X_RPA_JP[:,ph] .= 0.5 .* T * (sqrtD_m ./ E_RPA_JP[ph] .+ invsqrtD_m) * R[:,ph]
                    @views Y_RPA_JP[:,ph] .= 0.5 .* T * (sqrtD_m ./ E_RPA_JP[ph] .- invsqrtD_m) * R[:,ph]
                end
            end

        end

        # Renormalization of X and Y amplitudes ...
        @inbounds for nu in 1:N_ph
            X_norm = 0.0
            Y_norm = 0.0
            @inbounds for ph in 1:N_ph
                X_norm += abs2(X_RPA_JP[ph,nu])
                Y_norm += abs2(Y_RPA_JP[ph,nu])
            end
            if (X_norm - Y_norm) > 1e-8
                RPA_norm = ComplexF64(1.0 / sqrt(abs(X_norm - Y_norm)))
                @inbounds for ph in 1:N_ph
                    X_RPA_JP[ph,nu] = X_RPA_JP[ph,nu] * RPA_norm
                    Y_RPA_JP[ph,nu] = Y_RPA_JP[ph,nu] * RPA_norm
                end
            elseif (X_norm - Y_norm) < -1e-8
                RPA_norm = (0.0 - 1.0im) *  ComplexF64(1.0 / sqrt(abs(X_norm - Y_norm)))
                @inbounds for ph in 1:N_ph
                    X_RPA_JP[ph,nu] = X_RPA_JP[ph,nu] * RPA_norm
                    Y_RPA_JP[ph,nu] = Y_RPA_JP[ph,nu] * RPA_norm
                end
            else
                RPA_norm = 0.0
                @inbounds for ph in 1:N_ph
                    RPA_norm += abs2(X_RPA_JP[ph,nu]) + abs2(Y_RPA_JP[ph,nu])
                end
                RPA_norm = ComplexF64(1.0 / sqrt(abs(RPA_norm)))
                @inbounds for ph in 1:N_ph
                    X_RPA_JP[ph,nu] = X_RPA_JP[ph,nu] * RPA_norm
                    Y_RPA_JP[ph,nu] = Y_RPA_JP[ph,nu] * RPA_norm
                end
            end
        end

        # Determine stability of QRPA system ...
        @inbounds for nu in 1:N_ph
            if abs(imag(E_RPA_JP[nu])) > 1e-8
                println("\t\t\tRPA INSTABILITY DETECTED in channel:\tJ = $J, P = +")
                Stability = false
            end
        end
        
        E_RPA[J+1,P] = E_RPA_JP
        X_RPA[J+1,P] = X_RPA_JP
        Y_RPA[J+1,P] = Y_RPA_JP
    end

    println("\nDiagonalization done ...")

    return E_RPA, X_RPA, Y_RPA, Stability
end