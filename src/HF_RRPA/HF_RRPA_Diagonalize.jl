function HF_RRPA0_diagonalize(Params::Parameters,A::Matrix{Matrix{Float64}},B::Matrix{Matrix{Float64}},N_nu::Matrix{Int64})
    # Read calculation parameters ...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1

    JP_List = JP_initialize(J_max)
    
    # Preallocate arrays for solutions
    X_RPA = Matrix{Matrix{ComplexF64}}(undef,J_max+1,2)
    Y_RPA = Matrix{Matrix{ComplexF64}}(undef,J_max+1,2)
    E_RPA = Matrix{Vector{ComplexF64}}(undef,J_max+1,2)

    # Diagonalization of TDA & RPA matrices without orthogonalization of spurious 1- states ...
    @inbounds Threads.@threads for JP in JP_List
        J = JP[1]
        P = JP[2]
        N_ph = N_nu[J+1,P]

        # RPA diagonalization ...
        A_JP = A[J+1,P]
        B_JP = B[J+1,P]

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
                @inbounds for ph in 1:N_ph
                    X_RPA_JP[ph,nu] = ComplexF64(0.0)
                    Y_RPA_JP[ph,nu] = ComplexF64(0.0)
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

        # Removal of complex amplitudes for OBDM iteration ...
        @inbounds for nu in 1:N_ph
            Y2Sum = 0.0
            @inbounds for ph in 1:N_ph
                Y2Sum += abs(Y_RPA_JP[ph,nu])^2
            end
            if Y2Sum > 0.501
                @inbounds for ph in 1:N_ph
                    X_RPA_JP[ph,nu], Y_RPA_JP[ph,nu] = ComplexF64(0.0), ComplexF64(0.0)
                end
            end
        end

        @inbounds for nu in 1:N_ph
            E_nu = E_RPA_JP[nu]
            if abs(imag(E_nu)) > 1e-3
                @inbounds for ph in 1:N_ph
                    X_RPA_JP[ph,nu], Y_RPA_JP[ph,nu] = ComplexF64(0.0), ComplexF64(0.0)
                end
            end
        end

        E_RPA[J+1,P] = E_RPA_JP
        X_RPA[J+1,P] = X_RPA_JP
        Y_RPA[J+1,P] = Y_RPA_JP

    end

    return E_RPA, X_RPA, Y_RPA
end

function HF_RRPA_iteration_diagonalize(Params::Parameters,A::Matrix{Matrix{Float64}},B::Matrix{Matrix{Float64}},N_nu::Matrix{Int64})
    # Read calculation parameters ...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1

    JP_List = JP_initialize(J_max)
    
    # Preallocate arrays for solutions
    X_RPA = Matrix{Matrix{ComplexF64}}(undef,J_max+1,2)
    Y_RPA = Matrix{Matrix{ComplexF64}}(undef,J_max+1,2)
    E_RPA = Matrix{Vector{ComplexF64}}(undef,J_max+1,2)

    # Diagonalization of TDA & RPA matrices without orthogonalization of spurious 1- states ...
    @inbounds for JP in JP_List
        J = JP[1]
        P = JP[2]
        N_ph = N_nu[J+1,P]

        # RPA diagonalization ...
        A_JP = A[J+1,P]
        B_JP = B[J+1,P]

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
                @inbounds for ph in 1:N_ph
                    X_RPA_JP[ph,nu] = ComplexF64(0.0)
                    Y_RPA_JP[ph,nu] = ComplexF64(0.0)
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

        # Removal of complex amplitudes for OBDM iteration ...
        @inbounds for nu in 1:N_ph
            Y2Sum = 0.0
            @inbounds for ph in 1:N_ph
                Y2Sum += abs(Y_RPA_JP[ph,nu])^2
            end
            if Y2Sum > 0.501
                @inbounds for ph in 1:N_ph
                    X_RPA_JP[ph,nu], Y_RPA_JP[ph,nu] = ComplexF64(0.0), ComplexF64(0.0)
                end
            end
        end

        @inbounds for nu in 1:N_ph
            E_nu = E_RPA_JP[nu]
            if abs(imag(E_nu)) > 1e-3
                @inbounds for ph in 1:N_ph
                    X_RPA_JP[ph,nu], Y_RPA_JP[ph,nu] = ComplexF64(0.0), ComplexF64(0.0)
                end
            end
        end

        E_RPA[J+1,P] = E_RPA_JP
        X_RPA[J+1,P] = X_RPA_JP
        Y_RPA[J+1,P] = Y_RPA_JP

    end

    return E_RPA, X_RPA, Y_RPA
end

function HF_RRPA_diagonalize(Params::Parameters,A::Matrix{Matrix{Float64}},B::Matrix{Matrix{Float64}},N_nu::Matrix{Int64},Orb_Phonon::Matrix{Vector{Int64}},Phonon::Vector{PhState},Particle::pnSVector,Hole::pnSVector,TrOp::Tr1B,Rho::O1B,X_RPA_0::Matrix{Matrix{ComplexF64}},Y_RPA_0::Matrix{Matrix{ComplexF64}})
    # Read calculation parameters ...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1
    Orthogon = Params.Calc.RRPA.Ortho

    JP_List = JP_initialize(J_max)

    # Preallocate arrays for solutions
    X_RPA = Matrix{Matrix{ComplexF64}}(undef,J_max+1,2)
    Y_RPA = Matrix{Matrix{ComplexF64}}(undef,J_max+1,2)
    E_RPA = Matrix{Vector{ComplexF64}}(undef,J_max+1,2)

    # Diagonalization of RPA matrices with orthogonalization of spurious 1- states ...
    @inbounds for JP in JP_List
        J = JP[1]
        P = JP[2]
        N_ph = N_nu[J+1,P]

        # RPA diagonalization ...
        A_JP = A[J+1,P]
        B_JP = B[J+1,P]

        if (Orthogon == true) && (J == 1) && (P == 2)
            Spur_State = HF_RRPA_spurious_initialize(N_ph,Orb_Phonon[J+1,P],Phonon,Particle,Hole,TrOp,Rho,X_RPA_0,Y_RPA_0)
            V, Spur_ind = HF_RRPA_spurious_orthogonalize(N_ph,Spur_State)
            U = zeros(ComplexF64,N_ph,N_ph)

            @inbounds for a in 1:(Spur_ind-1)
                @views U[:,a] = V[:,a]
            end

            @inbounds for a in (Spur_ind+1):N_ph
                @views U[:,a-1] = V[:,a]
            end

            A_JP_ort = zeros(ComplexF64,N_ph - 1,N_ph - 1)
            B_JP_ort = zeros(ComplexF64,N_ph - 1,N_ph - 1)
            Pm = zeros(ComplexF64,N_ph,N_ph-1)
            Qm = zeros(ComplexF64,N_ph,N_ph-1)
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
            @views X_RPA_JP_new[:,1] = Spur_State
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

        E_RPA[J+1,P] = E_RPA_JP
        X_RPA[J+1,P] = X_RPA_JP
        Y_RPA[J+1,P] = Y_RPA_JP

    end

    return E_RPA, X_RPA, Y_RPA
end