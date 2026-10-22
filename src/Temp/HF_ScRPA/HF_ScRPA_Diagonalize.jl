function HF_ScRPA_diagonalize(Params::Parameters,A::Matrix{Matrix{Float64}},B::Matrix{Matrix{Float64}},N::Matrix{Matrix{Float64}},M::Matrix{Matrix{Float64}},N_nu::Matrix{Int64},Orb_Phonon::Matrix{Vector{Int64}},Phonon::Vector{PhState},Particle::pnSVector,Hole::pnSVector,TrOp::Tr1B)
    # Read calculation parameters ...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1
    Orthogon = Params.Calc.RPA.Ortho

    JP_List = JP_initialize(J_max)

    println("\nDiagonalizing RPA & TDA matrices ...")
    
    # Preallocate arrays for solutions
    X_TDA = Matrix{Matrix{Float64}}(undef,J_max+1,2)
    E_TDA = Matrix{Vector{Float64}}(undef,J_max+1,2)

    X_RPA = Matrix{Matrix{ComplexF64}}(undef,J_max+1,2)
    Y_RPA = Matrix{Matrix{ComplexF64}}(undef,J_max+1,2)
    E_RPA = Matrix{Vector{ComplexF64}}(undef,J_max+1,2)

    #BLAS.set_num_threads(Threads.nthreads())

    # Diagonalization of TDA & RPA matrices with orthogonalization of spurious 1- states ...
    @inbounds Threads.@threads for JP in JP_List
        J = JP[1]
        P = JP[2]
        N_ph = N_nu[J+1,P]

        # TDA method ...
        A_JP = A[J+1,P]

        if (Orthogon == true) && (J == 1) && (P == 2)
            Spur_State = Spur_Ini(N_ph,Orb_Phonon[J+1,P],Phonon,Particle,Hole,TrOp)
            V, Spur_Ind = Spur_Ortho(N_ph,Spur_State)
            U = zeros(Float64,N_ph,N_ph)

            @inbounds for a in 1:(Spur_Ind-1)
                @views U[:,a] = V[:,a]
            end

            @inbounds for a in (Spur_Ind+1):N_ph
                @views  U[:,a-1] = V[:,a]
            end

            A_JP_ort = zeros(Float64,N_ph - 1, N_ph - 1)

            Pm = zeros(Float64,N_ph,N_ph-1)
            @inbounds for a in 1:N_ph
                @inbounds for b in 1:(N_ph-1)
                    @inbounds for c in 1:N_ph
                        Pm[a,b] = Pm[a,b] + A_JP[a,c] * U[c,b]
                    end
                end
            end

            @inbounds for a in 1:(N_ph-1)
                @inbounds for b in 1:(N_ph-1)
                    @inbounds for c in 1:N_ph
                        A_JP_ort[a,b] = A_JP_ort[a,b] + U[c,a] * Pm[c,b]
                    end
                end
            end

            E_TDA_JP, X_TDA_JP = eigen(A_JP_ort, sortby=+)
            sort_inds_TDA = sortperm(E_TDA_JP, by = x -> real(x))
            E_TDA_JP, X_TDA_JP = @views E_TDA_JP[sort_inds_TDA], X_TDA_JP[:,sort_inds_TDA]

            Pm = zeros(Float64,N_ph,N_ph-1)
            @inbounds for a in 1:N_ph
                @inbounds for b in 1:(N_ph -1)
                    @inbounds for c in 1:(N_ph-1)
                        ME = U[a,c] * X_TDA_JP[c,b]
                        Pm[a,b] += ME
                    end
                end
            end

            E_TDA_JP_new = zeros(Float64,N_ph)
            @inbounds for a in 2:N_ph
                E_TDA_JP_new[a] = E_TDA_JP[a-1]
            end

            X_TDA_JP_new = zeros(Float64, N_ph, N_ph)
            @views X_TDA_JP_new[:,1] = Spur_State
            @inbounds for a in 2:N_ph
                @views X_TDA_JP_new[:,a] = Pm[:,a-1]

            end

            X_TDA_JP = X_TDA_JP_new
            E_TDA_JP = E_TDA_JP_new
        else
            E_TDA_JP, X_TDA_JP = eigen(A_JP, sortby=+)
            sort_inds_TDA = sortperm(E_TDA_JP, by = x -> real(x))
            E_TDA_JP, X_TDA_JP = E_TDA_JP[sort_inds_TDA], @views X_TDA_JP[:,sort_inds_TDA]
        end

        E_TDA[J+1,P] = E_TDA_JP
        X_TDA[J+1,P] = X_TDA_JP



        # RPA diagonalization ...
        A_JP = A[J+1,P]
        B_JP = B[J+1,P]
        N_JP = N[J+1,P]
        M_JP = M[J+1,P]

        if (Orthogon == true) && (J == 1) && (P == 2)
            Spur_State = Spur_Ini(N_ph,Orb_Phonon[J+1,P],Phonon,Particle,Hole,TrOp)
            V, Spur_ind = Spur_Ortho(N_ph,Spur_State)
            U = zeros(Float64,N_ph,N_ph)

            @inbounds for a in 1:(Spur_ind-1)
                @views U[:,a] = V[:,a]
            end

            @inbounds for a in (Spur_ind+1):N_ph
                @views U[:,a-1] = V[:,a]
            end

            A_JP_ort = zeros(Float64,N_ph - 1,N_ph - 1)
            B_JP_ort = zeros(Float64,N_ph - 1,N_ph - 1)
            N_JP_ort = zeros(Float64,N_ph - 1,N_ph - 1)
            M_JP_ort = zeros(Float64,N_ph - 1,N_ph - 1)
            Pm = zeros(Float64,N_ph,N_ph-1)
            Qm = zeros(Float64,N_ph,N_ph-1)
            Rm = zeros(Float64,N_ph,N_ph-1)
            Sm = zeros(Float64,N_ph,N_ph-1)
            @inbounds for a in 1:N_ph
                @inbounds for b in 1:(N_ph-1)
                    @inbounds for c in 1:N_ph
                        Pm[a,b] = Pm[a,b] + A_JP[a,c] * U[c,b]
                        Qm[a,b] = Qm[a,b] + B_JP[a,c] * U[c,b]
                        Rm[a,b] = Rm[a,b] + N_JP[a,c] * U[c,b]
                        Sm[a,b] = Sm[a,b] + M_JP[a,c] * U[c,b]
                    end
                end
            end

            @inbounds for a in 1:(N_ph-1)
                @inbounds for b in 1:(N_ph-1)
                    @inbounds for c in 1:N_ph
                        A_JP_ort[a,b] = A_JP_ort[a,b] + U[c,a] * Pm[c,b]
                        B_JP_ort[a,b] = B_JP_ort[a,b] + U[c,a] * Qm[c,b]
                        N_JP_ort[a,b] = N_JP_ort[a,b] + U[c,a] * Rm[c,b]
                        M_JP_ort[a,b] = M_JP_ort[a,b] + U[c,a] * Sm[c,b]
                    end
                end
            end

            A_JP = A_JP_ort
            B_JP = B_JP_ort
            N_JP = N_JP_ort
            M_JP = M_JP_ort

            # Solve RPA eigen-value problem ...

            X_RPA_JP = Matrix{ComplexF64}(undef,N_ph-1,N_ph-1)
            Y_RPA_JP = Matrix{ComplexF64}(undef,N_ph-1,N_ph-1)

            # Old diagonalization ...
            #=
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
            =#

            # New diagonalization ... including overlap matrix ...
            if !isapprox(det(BigFloat.(A_JP - B_JP)), 0.0, atol = 1e-7)
                inv_AmB_NpM = inv(A_JP - B_JP) * (N_JP + M_JP)
                NmM = N_JP - M_JP
                E_squared, X_prime = eigen(A_JP + B_JP, NmM * inv_AmB_NpM)
                DO = diagm(ones(Float64,N_ph-1))

                @views E_RPA_JP = sqrt.(Complex.(E_squared))

                @inbounds for nu in 1:(N_ph-1)
                    @views X_RPA_JP[:,nu] .= 1.0 / sqrt(2.0) * (DO + E_RPA_JP[nu] * inv_AmB_NpM) * X_prime[:,nu]
                    @views Y_RPA_JP[:,nu] .= 1.0 / sqrt(2.0) * (DO - E_RPA_JP[nu] * inv_AmB_NpM) * X_prime[:,nu]
                end

            elseif !isapprox(det(BigFloat.(A_JP + B_JP)), 0.0, atol = 1e-7)
                inv_ApB_NmM = inv(A_JP + B_JP) * (N_JP - M_JP)
                NpM = N_JP + M_JP
                E_squared, X_prime = eigen(A_JP - B_JP, NpM * inv_ApB_NmM)
                DO = diagm(ones(Float64,N_ph-1))

                @views E_RPA_JP = sqrt.(Complex.(E_squared))

                @inbounds for nu in 1:(N_ph-1)
                    @views X_RPA_JP[:,nu] .= 1.0 / sqrt(2.0) * (DO + E_RPA_JP[nu] * inv_ApB_NmM) * X_prime[:,nu]
                    @views Y_RPA_JP[:,nu] .= 1.0 / sqrt(2.0) * (DO - E_RPA_JP[nu] * inv_ApB_NmM) * X_prime[:,nu]
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

            # Old diagonalization ...
            #=
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
            =#

            # New diagonalization ... including overlap matrix ...
            if !isapprox(det(BigFloat.(A_JP - B_JP)), 0.0, atol = 1e-7)
                inv_AmB_NpM = inv(A_JP - B_JP) * (N_JP + M_JP)
                NmM = N_JP - M_JP
                E_squared, X_prime = eigen(A_JP + B_JP, NmM * inv_AmB_NpM)
                DO = diagm(ones(Float64,N_ph))

                @views E_RPA_JP = sqrt.(Complex.(E_squared))

                @inbounds for nu in 1:N_ph
                    @views X_RPA_JP[:,nu] .= 1.0 / sqrt(2.0) * (DO + E_RPA_JP[nu] * inv_AmB_NpM) * X_prime[:,nu]
                    @views Y_RPA_JP[:,nu] .= 1.0 / sqrt(2.0) * (DO - E_RPA_JP[nu] * inv_AmB_NpM) * X_prime[:,nu]
                end

            elseif !isapprox(det(BigFloat.(A_JP + B_JP)), 0.0, atol = 1e-7)
                inv_ApB_NmM = inv(A_JP + B_JP) * (N_JP - M_JP)
                NpM = N_JP + M_JP
                E_squared, X_prime = eigen(A_JP - B_JP, NpM * inv_ApB_NmM)
                DO = diagm(ones(Float64,N_ph))

                @views E_RPA_JP = sqrt.(Complex.(E_squared))

                @inbounds for nu in 1:N_ph
                    @views X_RPA_JP[:,nu] .= 1.0 / sqrt(2.0) * (DO + E_RPA_JP[nu] * inv_ApB_NmM) * X_prime[:,nu]
                    @views Y_RPA_JP[:,nu] .= 1.0 / sqrt(2.0) * (DO - E_RPA_JP[nu] * inv_ApB_NmM) * X_prime[:,nu]
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
        
        E_RPA[J+1,P] = E_RPA_JP
        X_RPA[J+1,P] = X_RPA_JP
        Y_RPA[J+1,P] = Y_RPA_JP

    end

    println("\nDiagonalization done ...")

    return E_TDA, X_TDA, E_RPA, X_RPA, Y_RPA
end