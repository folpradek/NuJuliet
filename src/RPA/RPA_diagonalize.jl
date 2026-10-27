function RPA_diagonalize(Params::Parameters,Orb::Vector{Orb1B},Orb_1p1h::Orb2B_1p1h,A::Matrix{Matrix{Float64}},B::Matrix{Matrix{Float64}},N::Matrix{Matrix{Float64}},M::Matrix{Matrix{Float64}},TrOp::Tr1B)
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
    M5_temp = Matrix{ComplexF64}(undef,N_ph_max,N_ph_max)

    # Additional working arrays for the orthogonalized 1- block ...
    if Orthogon == true
        N_ph_temp = Orb_1p1h.N[2,2]
        M6_temp = Matrix{Float64}(undef,N_ph_temp,N_ph_temp)
        M7_temp = Matrix{Float64}(undef,N_ph_temp,N_ph_temp)
        M8_temp = Matrix{Float64}(undef,N_ph_temp,N_ph_temp)
        M9_temp = Matrix{Float64}(undef,N_ph_temp,N_ph_temp)
        M10_temp = Matrix{Float64}(undef,N_ph_temp,N_ph_temp)
        M11_temp = Matrix{Float64}(undef,N_ph_temp,N_ph_temp)
    end

    # Diagonalization of RPA eigenvalue problem ...
    @inbounds for JP in JP_list
        J, P = JP[1], JP[2]
        N_ph = Orb_1p1h.N[P,J+1]

        A_JP = A[P,J+1]
        B_JP = B[P,J+1]
        N_JP = N[P,J+1]
        M_JP = M[P,J+1]

        X_JP = Matrix{ComplexF64}(undef,N_ph,N_ph)
        Y_JP = Matrix{ComplexF64}(undef,N_ph,N_ph)
        E_JP = Vector{ComplexF64}(undef,N_ph)

        println("\t\tDiagonalizing:\tJ = $J, P = $(P == 1 ? "+" : "-")")

        if Orthogon == false || !(J == 1 && P == 2)
            N_JP_diag = diag(N_JP)
            M_JP_diag = diag(M_JP)

            # Evaluate inv(sqrt(N + M)) & inv(sqrt(N - M)) as diagonals ...
            inv_sqrtN_p = 1.0 ./ sqrt.(N_JP_diag .+ M_JP_diag)
            inv_sqrtN_m = 1.0 ./ sqrt.(N_JP_diag .- M_JP_diag)

            # Renormalize A & B ...
            @views H_m = M1_temp[1:N_ph,1:N_ph]
            @views H_p = M2_temp[1:N_ph,1:N_ph]

            H_p .= inv_sqrtN_m .* (A_JP .+ B_JP) .* transpose(inv_sqrtN_p)
            H_m .= inv_sqrtN_p .* (A_JP .- B_JP) .* transpose(inv_sqrtN_m)

            @views H = M3_temp[1:N_ph,1:N_ph]

            mul!(H,H_p,H_m)

            # Solve general NxN non-symmetric real eigenvalue problem ...
            RPA_system = eigen!(H)

            # Eigenvalues & eigenvectors are complex only in the case of instability ...
            E2_nu = RPA_system.values

            @views r_m = M4_temp[1:N_ph,1:N_ph]
            @views p_m = M5_temp[1:N_ph,1:N_ph]

            r_m .= RPA_system.vectors

            @inbounds for i_nu in 1:N_ph
                E_nu = sqrt(ComplexF64(E2_nu[i_nu]))
                E_JP[i_nu] = E_nu
            end

            # Evaluate p_m = H_m * r_m separately for real & imaginary parts (BLAS real-real products) ...
            @views r_temp = M2_temp[1:N_ph,1:N_ph]
            @views p_temp = M3_temp[1:N_ph,1:N_ph]

            r_temp .= real.(r_m)
            mul!(p_temp,H_m,r_temp)
            p_m .= p_temp

            if eltype(RPA_system.vectors) <: Complex
                r_temp .= imag.(r_m)
                mul!(p_temp,H_m,r_temp)
                p_m .+= im .* p_temp
            end

            @inbounds for i_nu in 1:N_ph
                E_nu = E_JP[i_nu]
                iE = 1.0 / E_nu

                @inbounds for i_ph in 1:N_ph
                    r = inv_sqrtN_m[i_ph] * r_m[i_ph,i_nu]
                    p = inv_sqrtN_p[i_ph] * p_m[i_ph,i_nu]

                    X_JP[i_ph,i_nu] = 0.5 * iE * p + 0.5 * r
                    Y_JP[i_ph,i_nu] = 0.5 * iE * p - 0.5 * r
                end
            end

            # Normalization of X and Y amplitudes ...
            @inbounds for i_nu in 1:N_ph
                X_nu, Y_nu = @views X_JP[:,i_nu], @views Y_JP[:,i_nu]
                RPA_norm = ComplexF64(0.0)

                @inbounds for i_ph in 1:N_ph
                    x, y = X_nu[i_ph], Y_nu[i_ph]
                    RPA_norm += N_JP_diag[i_ph] * (abs2(x) - abs2(y)) + M_JP_diag[i_ph] * (conj(x) * y - conj(y) * x)
                end

                # Non-zero norm states ...
                if abs(RPA_norm) > 1e-8
                    norm_nu = 1.0 / sqrt(abs(RPA_norm))
                    @views X_JP[:,i_nu] .= norm_nu .* X_nu
                    @views Y_JP[:,i_nu] .= norm_nu .* Y_nu
                end

                # Zero-norm states (e.g. complex modes of an unstable system) ...
                if abs(RPA_norm) < 1e-8
                    throw("Missing implementation of the zero-norm states ...")
                end
            end

        elseif Orthogon == true && (J == 1 && P == 2)
                # Comment on optimization by Claude ...
                # Orthogonalization of the spurious center-of-mass 1- level ...
                #
                # The physical RPA solutions are restricted to the subspace orthogonal to the spurious state Spur,
                #
                #     X = U_sub * X_ort ,      Y = U_sub * Y_ort ,
                #
                # where the N_ph - 1 columns of U_sub form an orthonormal basis of the complement of Spur. In this basis
                # the overlap matrices N_ort +- M_ort = U_sub' * (N +- M) * U_sub are no longer diagonal & the RPA system
                # is renormalized by their symmetric inverse square roots inv_sqrtN_pm_ort. Defining the rectangular
                # N_ph x (N_ph - 1) matrices
                #
                #     W_p = U_sub * inv_sqrtN_p_ort ,      W_m = U_sub * inv_sqrtN_m_ort ,
                #
                # the RPA system in the orthogonalized subspace reads (cf. the non-orthogonalized branch)
                #
                #     H_p = W_m' * (A + B) * W_p ,      H_m = W_p' * (A - B) * W_m ,      H_p * H_m * r_m = E^2 * r_m ,
                #
                # & the amplitudes are obtained directly in the full 1p-1h space,
                #
                #     (X - Y) = W_m * r_m ,      E * (X + Y) = W_p * H_m * r_m .
                #
                # Since U_sub is orthonormal, the RPA norm X_ort' * N_ort * X_ort - ... equals X' * N * X - ..., hence the
                # amplitudes are normalized in the full space with the diagonal overlap matrices N & M. The spurious state
                # itself is obtained from the 2x2 RPA problem in the subspace spanned by Spur & stored as the first solution ...
            N_ort = N_ph - 1

            N_JP_diag = diag(N_JP)
            M_JP_diag = diag(M_JP)

            # Initialize the vector for orthogonalization of spurious 1- level ...
            Spur = RPA_spurious_ini(Orb,Orb_1p1h,TrOp)

            # Perform the orthogonalization of spurious 1- level ...
            @views U = M6_temp[1:N_ph,1:N_ph]

            Spur_ind = RPA_spurious_ortho!(U,Spur)

            # Remove the spurious column, the complement of Spur is stored in U_sub = U[:,1:N_ph-1] ...
            @inbounds for a in (Spur_ind+1):N_ph
                @views U[:,a-1] .= U[:,a]
            end

            @views U_sub = U[:,1:N_ort]

            # Evaluate inverse square roots of the overlap matrices N_ort +- M_ort = U_sub' * (N +- M) * U_sub ...
            # Using the spectral decomposition N_ort +- M_ort = V_N * diagm(lambda) * V_N', the symmetric inverse
            # square root reads inv_sqrtN_+-_ort = (V_N * diagm(lambda.^(-1/4))) * (V_N * diagm(lambda.^(-1/4)))' ...
            @views T = M7_temp[1:N_ph,1:N_ort]
            @views N_pm_ort = M8_temp[1:N_ort,1:N_ort]
            @views inv_sqrtN_p_ort = M10_temp[1:N_ort,1:N_ort]
            @views inv_sqrtN_m_ort = M11_temp[1:N_ort,1:N_ort]

            T .= (N_JP_diag .+ M_JP_diag) .* U_sub
            mul!(N_pm_ort,transpose(U_sub),T)

            N_system = eigen!(Symmetric(N_pm_ort))
            V_N = N_system.vectors
            V_N .*= transpose(N_system.values .^ (-0.25))

            mul!(inv_sqrtN_p_ort,V_N,transpose(V_N))

            # Both overlap matrices coincide for M = 0 ...
            if iszero(M_JP_diag)
                inv_sqrtN_m_ort .= inv_sqrtN_p_ort
            else
                T .= (N_JP_diag .- M_JP_diag) .* U_sub
                mul!(N_pm_ort,transpose(U_sub),T)

                N_system = eigen!(Symmetric(N_pm_ort))
                V_N = N_system.vectors
                V_N .*= transpose(N_system.values .^ (-0.25))

                mul!(inv_sqrtN_m_ort,V_N,transpose(V_N))
            end

            # Define W_+- = U_sub * inv_sqrtN_+-_ort ...
            @views W_m = M8_temp[1:N_ph,1:N_ort]
            @views W_p = M9_temp[1:N_ph,1:N_ort]

            mul!(W_m,U_sub,inv_sqrtN_m_ort)
            mul!(W_p,U_sub,inv_sqrtN_p_ort)

            # Define H_+- in the orthogonalized subspace ...
            @views AB = M3_temp[1:N_ph,1:N_ph]
            @views H_m = M1_temp[1:N_ort,1:N_ort]
            @views H_p = M2_temp[1:N_ort,1:N_ort]

            AB .= A_JP .+ B_JP
            mul!(T,AB,W_p)
            mul!(H_p,transpose(W_m),T)

            AB .= A_JP .- B_JP
            mul!(T,AB,W_m)
            mul!(H_m,transpose(W_p),T)

            @views H = M3_temp[1:N_ort,1:N_ort]

            mul!(H,H_p,H_m)

            # Solve general NxN non-symmetric real eigenvalue problem ...
            RPA_system = eigen!(H)

            # Eigenvalues & eigenvectors are complex only in the case of instability ...
            E2_nu = RPA_system.values
            r_ort = RPA_system.vectors

            @inbounds for i_nu in 1:N_ort
                E_nu = sqrt(ComplexF64(E2_nu[i_nu]))
                E_JP[i_nu+1] = E_nu
            end

            # Evaluate (X - Y) = W_m * r_ort & E (X + Y) = W_p * H_m * r_ort separately for real & imaginary parts (BLAS real-real products) ...
            @views r_full = M4_temp[1:N_ph,1:N_ort]
            @views p_full = M5_temp[1:N_ph,1:N_ort]

            @views r_temp = M2_temp[1:N_ort,1:N_ort]
            @views p_temp = M3_temp[1:N_ort,1:N_ort]

            r_temp .= real.(r_ort)
            mul!(p_temp,H_m,r_temp)
            mul!(T,W_m,r_temp)
            r_full .= T
            mul!(T,W_p,p_temp)
            p_full .= T

            if eltype(r_ort) <: Complex
                r_temp .= imag.(r_ort)
                mul!(p_temp,H_m,r_temp)
                mul!(T,W_m,r_temp)
                r_full .+= im .* T
                mul!(T,W_p,p_temp)
                p_full .+= im .* T
            end

            @inbounds for i_nu in 1:N_ort
                E_nu = E_JP[i_nu+1]
                iE = 1.0 / E_nu

                @inbounds for i_ph in 1:N_ph
                    r = r_full[i_ph,i_nu]
                    p = p_full[i_ph,i_nu]

                    X_JP[i_ph,i_nu+1] = 0.5 * iE * p + 0.5 * r
                    Y_JP[i_ph,i_nu+1] = 0.5 * iE * p - 0.5 * r
                end
            end

            # Normalization of X and Y amplitudes ...
            @inbounds for i_nu in 2:N_ph
                X_nu, Y_nu = @views X_JP[:,i_nu], @views Y_JP[:,i_nu]
                RPA_norm = ComplexF64(0.0)

                @inbounds for i_ph in 1:N_ph
                    x, y = X_nu[i_ph], Y_nu[i_ph]
                    RPA_norm += N_JP_diag[i_ph] * (abs2(x) - abs2(y)) + M_JP_diag[i_ph] * (conj(x) * y - conj(y) * x)
                end

                # Non-zero norm states ...
                if abs(RPA_norm) > 1e-8
                    norm_nu = 1.0 / sqrt(abs(RPA_norm))
                    @views X_JP[:,i_nu] .= norm_nu .* X_nu
                    @views Y_JP[:,i_nu] .= norm_nu .* Y_nu
                end

                # Zero-norm states (e.g. complex modes of an unstable system) ...
                if abs(RPA_norm) < 1e-8
                    throw("Missing implementation of the zero-norm states ...")
                end
            end

            # Solve the 2x2 RPA eigenvalue problem in the spurious subspace, X = x * Spur & Y = y * Spur ...
            A_spur, B_spur = dot(Spur,A_JP,Spur), dot(Spur,B_JP,Spur)
            N_spur, M_spur = dot(Spur,N_JP_diag .* Spur), dot(Spur,M_JP_diag .* Spur)

            inv_sqrtN_p_spur, inv_sqrtN_m_spur = 1.0 / sqrt(N_spur + M_spur), 1.0 / sqrt(N_spur - M_spur)

            H_p_spur = inv_sqrtN_m_spur * (A_spur + B_spur) * inv_sqrtN_p_spur
            H_m_spur = inv_sqrtN_p_spur * (A_spur - B_spur) * inv_sqrtN_m_spur

            E_spur = sqrt(ComplexF64(H_p_spur * H_m_spur))

            r_spur = inv_sqrtN_m_spur
            p_spur = inv_sqrtN_p_spur * H_m_spur

            x_spur = 0.5 * p_spur / E_spur + 0.5 * r_spur
            y_spur = 0.5 * p_spur / E_spur - 0.5 * r_spur

            norm_spur = N_spur * (abs2(x_spur) - abs2(y_spur)) + M_spur * (conj(x_spur) * y_spur - conj(y_spur) * x_spur)

            println("\t\t\tSpurious 1- level from the RPA problem at\tE = $E_spur")

            # Non-zero norm spurious state ...
            if abs(E_spur) > 1e-8 && abs(norm_spur) > 1e-8
                norm_nu = 1.0 / sqrt(abs(norm_spur))
                E_JP[1] = E_spur
                @views X_JP[:,1] .= (norm_nu * x_spur) .* Spur
                @views Y_JP[:,1] .= (norm_nu * y_spur) .* Spur
            end

            # Zero-energy or zero-norm spurious state (Goldstone mode or imaginary E_spur) ...
            if abs(E_spur) < 1e-8 || abs(norm_spur) < 1e-8
                println("\t\t\tZero-norm spurious 1- level, setting E = 0 & X = -Y = Spur / sqrt(2) ...")
                E_JP[1] = ComplexF64(0.0)
                @views X_JP[:,1] .= Spur ./ sqrt(2)
                @views Y_JP[:,1] .= -Spur ./ sqrt(2)
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
