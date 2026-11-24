function RPA_diagonalize(Params::Parameters,Orb::Vector{Orb1B},Orb_1p1h::Orb2B_1p1h,A::Matrix{Matrix{Float64}},B::Matrix{Matrix{Float64}},N::Matrix{Matrix{Float64}},TrOp::Tr1B)
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

        println("\t\tDiagonalizing:\tJ = $J, P = $(P == 1 ? "+" : "-")")

        if Orthogon == false || !(J == 1 && P == 2)
            # Comment on memory optimization by Claude ...
                # Only two real N_ph x N_ph working arrays are used. Since
                #
                #     H = H_p * H_m = inv_sqrtN * (A + B) * K ,      K = inv_sqrtN^2 * (A - B) * inv_sqrtN = inv_sqrtN * H_m ,
                #
                # the matrix H is evaluated as inv_sqrtN * (A * K + B * K) without storing A + B, H_p & H_m. The amplitudes
                # follow from the eigenvectors R of H as
                #
                #     (X - Y) = inv_sqrtN * R ,      E * (X + Y) = inv_sqrtN * H_m * R = K * R ,
                #
                # where the output arrays X_JP & Y_JP are used to temporarily store (X - Y) & E * (X + Y) ...

            N_JP_diag = diag(N_JP)

            # Evaluate inv(sqrt(N)) as diagonal ...
            inv_sqrtN = 1.0 ./ sqrt.(N_JP_diag)

            # Renormalize A - B ... invN_H_m = inv_sqrtN^2 * (A - B) * inv_sqrtN ...
            @views invN_H_m = M1_temp[1:N_ph,1:N_ph]

            invN_H_m .= inv_sqrtN.^2 .* (A_JP .- B_JP) .* transpose(inv_sqrtN)

            # Evaluate H = inv_sqrtN * (A * invN_H_m + B * invN_H_m) ...
            @views H = M2_temp[1:N_ph,1:N_ph]

            mul!(H,A_JP,invN_H_m)
            mul!(H,B_JP,invN_H_m,1.0,1.0)
            H .*= inv_sqrtN

            # Solve general NxN non-symmetric real eigenvalue problem ...
            RPA_system = eigen!(H)

            # Eigenvalues & eigenvectors are complex only in the case of instability ...
            E2_nu = RPA_system.values
            R = RPA_system.vectors

            @inbounds for i_nu in 1:N_ph
                E_nu = sqrt(ComplexF64(E2_nu[i_nu]))
                E_JP[i_nu] = E_nu
            end

            # Store (X - Y) = inv_sqrtN * R in X_JP ...
            X_JP .= inv_sqrtN .* R

            # Store E * (X + Y) = invN_H_m * R in Y_JP, real & imaginary parts separately (BLAS real-real products) ...
            @views p_temp = M2_temp[1:N_ph,1:N_ph]

            R_re = eltype(R) <: Complex ? real.(R) : R

            mul!(p_temp,invN_H_m,R_re)
            Y_JP .= p_temp

            if eltype(R) <: Complex
                mul!(p_temp,invN_H_m,imag.(R))
                Y_JP .+= im .* p_temp
            end

            # Calculate RPA amplitudes X & Y ...
            @inbounds for i_nu in 1:N_ph
                E_nu = E_JP[i_nu]
                iE = 1.0 / E_nu

                @inbounds for i_ph in 1:N_ph
                    r = X_JP[i_ph,i_nu]
                    p = Y_JP[i_ph,i_nu]

                    X_JP[i_ph,i_nu] = 0.5 * iE * p + 0.5 * r
                    Y_JP[i_ph,i_nu] = 0.5 * iE * p - 0.5 * r
                end
            end

            # Normalization of X and Y amplitudes ...
            @inbounds for i_nu in 1:N_ph
                @views X_nu, Y_nu = X_JP[:,i_nu], Y_JP[:,i_nu]
                RPA_norm = ComplexF64(0.0)

                @inbounds for i_ph in 1:N_ph
                    x, y = X_nu[i_ph], Y_nu[i_ph]
                    RPA_norm += N_JP_diag[i_ph] * (abs2(x) - abs2(y))
                end

                # Non-zero norm states ...
                if abs(RPA_norm) > 1e-12
                    norm_nu = 1.0 / sqrt(abs(RPA_norm))
                    X_nu .*= norm_nu
                    Y_nu .*= norm_nu
                
                # Zero-norm states ...
                else
                    X_nu .*= 0.0
                    Y_nu .*= 0.0
                    println("\t\t\tZero-norm RPA state detected, setting X = Y = 0 ...")
                end
            end

        else
            # Comment on optimization by Claude ...
                # Orthogonalization of the spurious center-of-mass 1- level ...
                #
                # The physical RPA solutions are restricted to the subspace orthogonal to the spurious state Spur,
                #
                #     X = U_sub * X_ort ,      Y = U_sub * Y_ort ,
                #
                # where the N_ph - 1 columns of U_sub form an orthonormal basis of the complement of Spur. In this basis
                # the overlap matrix N_ort = U_sub' * N * U_sub is no longer diagonal & the RPA system is renormalized
                # by its symmetric inverse square root inv_sqrtN_ort. Defining the rectangular N_ph x (N_ph - 1) matrix
                #
                #     W = U_sub * inv_sqrtN_ort ,
                #
                # the RPA system in the orthogonalized subspace reads (cf. the non-orthogonalized branch)
                #
                #     H_p = W' * (A + B) * W ,      H_m = W' * (A - B) * W ,      H_p * H_m * R = E^2 * R ,
                #
                # & the amplitudes are obtained directly in the full 1p-1h space,
                #
                #     (X - Y) = W * R ,      E * (X + Y) = W * H_m * R .
                #
                # Since U_sub is orthonormal, the RPA norm X_ort' * N_ort * X_ort - ... equals X' * N * X - ..., hence the
                # amplitudes are normalized in the full space with the diagonal overlap matrix N. The spurious state
                # itself is obtained from the 2x2 RPA problem in the subspace spanned by Spur & stored as the first solution ...
                #
                # Comment on memory optimization by Claude ...
                # In the code W is denoted U_invN. Besides M1_temp & M2_temp only two additional real N_ph x N_ph working
                # arrays are needed. A + B & A - B are never stored, H_+- = W' * A * W +- W' * B * W is evaluated from the two
                # products W' * A * W & W' * B * W, & the output arrays X_JP & Y_JP temporarily store (X - Y) & E * (X + Y) ...

            # Define physical subspace dimension ...
            N_ort = N_ph - 1

            N_JP_diag = diag(N_JP)

            # Define additional working arrays for the orthogonalized 1- block ...
            M3_temp = Matrix{Float64}(undef,N_ph,N_ph)
            M4_temp = Matrix{Float64}(undef,N_ph,N_ph)

            # Initialize the vector for orthogonalization of spurious 1- level ...
            Spur = RPA_spurious_ini(Orb,Orb_1p1h,TrOp)

            # Perform the orthogonalization of spurious 1- level ...
            @views U = M3_temp[1:N_ph,1:N_ph]

            # Calculate the spurious dipole state ...
            Spur_ind = RPA_spurious_ortho!(U,Spur)

            # Remove the spurious column from 1p-1h basis ... U_sub is the physical subspace basis ...
            @inbounds for a in (Spur_ind+1):N_ph
                @views U[:,a-1] .= U[:,a]
            end

            # Define the physical subspace ...
            @views U_sub = U[:,1:N_ort]

            # Evaluate inverse square root of the overlap matrix N_ort_JP = U_sub' * N * U_sub ...
            # Using the spectral decomposition N_ort_JP = V_N * diagm(lambda) * V_N', the symmetric inverse
            # square root reads inv_sqrtN_ort = (V_N * diagm(lambda.^(-1/4))) * (V_N * diagm(lambda.^(-1/4)))' ...
            @views N_U_temp = M2_temp[1:N_ph,1:N_ort]
            @views N_ort_JP = M1_temp[1:N_ort,1:N_ort]
            @views inv_sqrtN_ort = M4_temp[1:N_ort,1:N_ort]

            N_U_temp .= N_JP_diag .* U_sub
            mul!(N_ort_JP,transpose(U_sub),N_U_temp)

            N_system = eigen!(Symmetric(N_ort_JP))
            V_N = N_system.vectors
            V_N .*= transpose(N_system.values .^ (-0.25))

            mul!(inv_sqrtN_ort,V_N,transpose(V_N))

            # Define U_invN = U_sub * inv_sqrtN_ort ... (U_sub & inv_sqrtN_ort are no longer needed afterwards)
            @views U_invN = M2_temp[1:N_ph,1:N_ort]

            mul!(U_invN,U_sub,inv_sqrtN_ort)

            # Define H_+- in the orthogonalized physical subspace ...
            # H_p = U_invN' * A * U_invN + U_invN' * B * U_invN & H_m = U_invN' * A * U_invN - U_invN' * B * U_invN ...
            @views AB_U_temp = M3_temp[1:N_ph,1:N_ort]
            @views H_p = M4_temp[1:N_ort,1:N_ort]
            @views H_m = M1_temp[1:N_ort,1:N_ort]

            mul!(AB_U_temp,A_JP,U_invN)
            mul!(H_p,transpose(U_invN),AB_U_temp)

            mul!(AB_U_temp,B_JP,U_invN)
            mul!(H_m,transpose(U_invN),AB_U_temp)

            H_p .+= H_m
            H_m .= H_p .- 2.0 .* H_m

            @views H = M3_temp[1:N_ort,1:N_ort]

            mul!(H,H_p,H_m)

            # Solve general NxN non-symmetric real eigenvalue problem ...
            RPA_system = eigen!(H)

            # Eigenvalues & eigenvectors are complex only in the case of instability ...
            E2_nu = RPA_system.values
            R_ort = RPA_system.vectors

            @inbounds for i_nu in 1:N_ort
                E_nu = sqrt(ComplexF64(E2_nu[i_nu]))
                E_JP[i_nu+1] = E_nu
            end

            # Store (X - Y) = U_invN * R_ort & E (X + Y) = U_invN * H_m * R_ort in X_JP & Y_JP ...
            # Real & imaginary parts separately (BLAS real-real products) ...
            @views XmY = X_JP[:,2:N_ph]
            @views XpY = Y_JP[:,2:N_ph]

            @views Hr_temp = M4_temp[1:N_ort,1:N_ort]
            @views U_Hr_temp = M3_temp[1:N_ph,1:N_ort]

            R_re = eltype(R_ort) <: Complex ? real.(R_ort) : R_ort

            mul!(U_Hr_temp,U_invN,R_re)
            XmY .= U_Hr_temp
            mul!(Hr_temp,H_m,R_re)
            mul!(U_Hr_temp,U_invN,Hr_temp)
            XpY .= U_Hr_temp

            if eltype(R_ort) <: Complex
                R_im = imag.(R_ort)

                mul!(U_Hr_temp,U_invN,R_im)
                XmY .+= im .* U_Hr_temp
                mul!(Hr_temp,H_m,R_im)
                mul!(U_Hr_temp,U_invN,Hr_temp)
                XpY .+= im .* U_Hr_temp
            end

            # Calculate RPA amplitudes X & Y ...
            @inbounds for i_nu in 2:N_ph
                E_nu = E_JP[i_nu]
                iE = 1.0 / E_nu

                @inbounds for i_ph in 1:N_ph
                    r = X_JP[i_ph,i_nu]
                    p = Y_JP[i_ph,i_nu]

                    X_JP[i_ph,i_nu] = 0.5 * iE * p + 0.5 * r
                    Y_JP[i_ph,i_nu] = 0.5 * iE * p - 0.5 * r
                end
            end

            # Normalization of X and Y amplitudes ...
            @inbounds for i_nu in 2:N_ph
                @views X_nu, Y_nu = X_JP[:,i_nu], Y_JP[:,i_nu]
                RPA_norm = ComplexF64(0.0)

                @inbounds for i_ph in 1:N_ph
                    x, y = X_nu[i_ph], Y_nu[i_ph]
                    RPA_norm += N_JP_diag[i_ph] * (abs2(x) - abs2(y))
                end

                # Non-zero norm states ...
                if abs(RPA_norm) > 1e-12
                    norm_nu = 1.0 / sqrt(abs(RPA_norm))
                    X_nu .*= norm_nu
                    Y_nu .*= norm_nu
                
                # Zero-norm states ...
                else
                    X_nu .*= 0.0
                    Y_nu .*= 0.0
                    println("\t\t\tZero-norm RPA state detected, setting X = Y = 0 ...")
                end
            end

            # Solve the RPA eigenvalue problem in the spurious Goldstone boson subspace ... 2x2 eigenvalue problem ...

            # Define the spurious subspace matrices ...
                # Define spurious matrices A & B ... scalars ...
            A_spur, B_spur = dot(Spur,A_JP,Spur), dot(Spur,B_JP,Spur)
            N_spur = dot(Spur,N_JP_diag .* Spur)
                # Define inverted square roots of the norm kernel N ...
            inv_sqrtN_spur = 1.0 / sqrt(N_spur)
                # Define the spurious RPA Hamiltonian in the 2x2 subspace of Spur ...
            H_p_spur = inv_sqrtN_spur * (A_spur + B_spur) * inv_sqrtN_spur
            H_m_spur = inv_sqrtN_spur * (A_spur - B_spur) * inv_sqrtN_spur

            # Solve for the spurious RPA eigenvalue energy ...
            E_spur = sqrt(ComplexF64(H_p_spur * H_m_spur))

            # Recover the amplitudes P & R ...
            r_spur = inv_sqrtN_spur
            p_spur = inv_sqrtN_spur * H_m_spur

            # Calculate RPA amplitudes X & Y ...
            x_spur = 0.5 * p_spur / E_spur + 0.5 * r_spur
            y_spur = 0.5 * p_spur / E_spur - 0.5 * r_spur

            # Evaluate the norm of the spurious RPA level ...
            norm_spur = N_spur * (abs2(x_spur) - abs2(y_spur))

            println("\t\t\tSpurious 1- RPA Goldstone boson   ...   E = ($(round(E_spur, digits=7))) MeV")

            # Non-zero norm spurious state ...
            if abs(E_spur) > 1e-8 && abs(norm_spur) > 1e-8
                norm_nu = 1.0 / sqrt(abs(norm_spur))
                E_JP[1] = E_spur
                @views X_JP[:,1] .= (norm_nu * x_spur) .* Spur
                @views Y_JP[:,1] .= (norm_nu * y_spur) .* Spur
            end

            # Zero-energy or zero-norm spurious state (Goldstone mode or imaginary E_spur) ...
            #   X = Y = 0 as for the other zero-norm states ... (X = -Y = Spur / sqrt(2) is not normalizable & added O(10 MeV)
            #   spurious contributions to the OBDM, Sigma & the correlation energies)
            if abs(E_spur) < 1e-8 || abs(norm_spur) < 1e-8
                println("\t\t\tZero-norm spurious 1- level, setting E = 0 & X = Y = 0 ...")
                E_JP[1] = ComplexF64(0.0)
                @views X_JP[:,1] .= 0.0
                @views Y_JP[:,1] .= 0.0
            end
        end

        # Determine Stability of RPA system ... physical solutions only (the spurious 1- Goldstone level i_nu = 1 is excluded) ...
        nu_start = RPA_spurious(Orthogon,J,P) ? 2 : 1
        if any(E_nu -> abs(imag(E_nu)) > 1e-8, @view E_JP[nu_start:N_ph])
            println("\t\t\tRPA INSTABILITY DETECTED in channel:\tJ = $J, P = $(P == 1 ? "+" : "-")")
            Stability = false
        end

        E[P,J+1] = E_JP
        X[P,J+1] = X_JP
        Y[P,J+1] = Y_JP
    end

    println("\tFinished diagonalization of the RPA system of equations ...")

    return E, X, Y, Stability
end
