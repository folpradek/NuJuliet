function RPA_energy(Params::Parameters,Orb::Vector{Orb1B},Orb_1p1h::Orb2B_1p1h,E::Matrix{Vector{ComplexF64}},X::Matrix{Matrix{ComplexF64}},Y::Matrix{Matrix{ComplexF64}})
    println("\nCalculating RPA correlation energy ...")

    # Quasi-boson formula energy ...
    E_QBA = RPA_energy_QBA(Params,Orb,Orb_1p1h,E,X,Y)

    # Density energy ...
    E_rho = NaN64

    E_corr = (QBA = E_QBA, rho = E_rho)

    println("\tRPA correlation energy (QBA)   ...   E_QBA = " * string(round(E_QBA,digits=6)) * " MeV")

    return E_corr
end

function RPA_energy(Params::Parameters,Orb::Vector{Orb1B},Orb_1p1h::Orb2B_1p1h,Orb_NN::Orb2B,Orb_NNN::Orb3B,
                    E::Matrix{Vector{ComplexF64}},X::Matrix{Matrix{ComplexF64}},Y::Matrix{Matrix{ComplexF64}},
                    V_NN::O2B,T_bare::O1B,V_NN_bare::O2B,V_NNN_bare::Array{Vector{Vector{Float32}},4},Rho::O1B,Sigma::O2B,C_HF::O1B,Rho_HF::O1B,C_NAT::O1B)
    println("\nCalculating RPA correlation energy ...")

    # Calculate the analytic correlation energy formula ...
    E_QBA = RPA_energy_QBA(Params,Orb,Orb_1p1h,E,X,Y)

    # Density energy contributions ...
    E_rho, E_sig, E_sub = RPA_energy_rho(Params,Orb,Orb_1p1h,Orb_NN,Orb_NNN,E,X,Y,V_NN,T_bare,V_NN_bare,V_NNN_bare,Rho,Sigma,C_HF,Rho_HF,C_NAT)

    # Density energy ... Pauli-projected 2-body correlation function Sigma ...
    E_den = E_rho + E_sig + E_sub

    # Print the resulting values for correlation energies ...
    println("\tRPA correlation energy (QBA) ...                         E_QBA = " * string(round(E_QBA,digits=6)) * " MeV")
    println("\tMean-field energy shift E_MF[rho] - E_MF[rho_HF] ...     E_rho = " * string(round(E_rho,digits=6)) * " MeV")
    println("\t2-body correlation energy 1/4 V.Sigma ...                E_sig = " * string(round(E_sig,digits=6)) * " MeV")
    println("\tSubtraction of the Sigma terms of A & B ...              E_sub = " * string(round(E_sub,digits=6)) * " MeV")
    println("\tRPA correlation energy (Density) ...                     E_den = " * string(round(E_den,digits=6)) * " MeV")

    # Define the output correlation energy tuple ...
    E_corr = (QBA = E_QBA, rho = E_den)

    return E_corr
end

function RPA_energy_QBA(Params::Parameters,Orb::Vector{Orb1B},Orb_1p1h::Orb2B_1p1h,E::Matrix{Vector{ComplexF64}},X::Matrix{Matrix{ComplexF64}},Y::Matrix{Matrix{ComplexF64}})
    # Read parameters ...
    N_max = Params.Calc.Nmax
    J_max = 2*N_max + 1

    # Initialize the list of values of J & P for iteration ...
    JP_list = JP_initialize(J_max)
    N_JP = length(JP_list)

    # Results of individual J & P blocks ...
    E_QBA_JP = zeros(Float64,N_JP)
    N_drop_JP = zeros(Int64,N_JP)

    @inbounds Threads.@threads for i_JP in 1:N_JP
        J, P = JP_list[i_JP][1], JP_list[i_JP][2]

        N_ph = Orb_1p1h.N[P,J+1]
        if N_ph == 0
            continue
        end

        Orb_1p1h_JP = Orb_1p1h.Ind[P,J+1]
        J_hat = Float64(2*J + 1)

        E_JP, X_JP, Y_JP = E[P,J+1], X[P,J+1], Y[P,J+1]

        # Skip the spurious 1- Goldstone level (i_nu = 1 of the orthogonalized 1- block) ...
        nu_start = RPA_spurious(Params.Calc.RPA.Ortho,J,P) ? 2 : 1

        # Occupation differences D_ph = n_h - n_p ... (RPA metric N)
        D_JP = zeros(Float64,N_ph)
        @inbounds for i_ph in 1:N_ph
            a_p, a_h, T_ph = Orb_1p1h_JP[i_ph].a, Orb_1p1h_JP[i_ph].b, Orb_1p1h_JP[i_ph].T
            if T_ph == -1
                D_JP[i_ph] = Orb[a_h].pO - Orb[a_p].pO
            else
                D_JP[i_ph] = Orb[a_h].nO - Orb[a_p].nO
            end
        end

        SumE = 0.0
        @inbounds for i_nu in nu_start:N_ph
            X_nu, Y_nu = view(X_JP,:,i_nu), view(Y_JP,:,i_nu)

            # Physical solutions only ... non-zero norm & real energy ...
            RPA_norm = 0.0
            @inbounds for i_ph in 1:N_ph
                RPA_norm += D_JP[i_ph] * (abs2(X_nu[i_ph]) - abs2(Y_nu[i_ph]))
            end

            if abs(RPA_norm) < 1e-8
                continue
            end

            if abs(imag(E_JP[i_nu])) > 1e-8
                N_drop_JP[i_JP] += 1
                continue
            end

            # Quasi-boson energy ... -w_nu Y_nu^+ N Y_nu ...
            @inbounds for i_ph in 1:N_ph
                SumE -= J_hat * real(E_JP[i_nu]) * D_JP[i_ph] * abs2(Y_nu[i_ph])
            end
        end

        E_QBA_JP[i_JP] = SumE
    end

    if sum(N_drop_JP) > 0
        println("\tWarning! " * string(sum(N_drop_JP)) * " RPA solutions with complex energies (instability) are excluded from the quasi-boson energy ...")
        println("\t         (they are still contained in the 2-body correlation function Sigma ...)")
    end

    return sum(E_QBA_JP)
end

function RPA_energy_rho(Params::Parameters,Orb::Vector{Orb1B},Orb_1p1h::Orb2B_1p1h,Orb_NN::Orb2B,Orb_NNN::Orb3B,
                        E::Matrix{Vector{ComplexF64}},X::Matrix{Matrix{ComplexF64}},Y::Matrix{Matrix{ComplexF64}},
                        V_NN::O2B,T_bare::O1B,V_NN_bare::O2B,V_NNN_bare::Array{Vector{Vector{Float32}},4},Rho::O1B,Sigma::O2B,C_HF::O1B,Rho_HF::O1B,C_NAT::O1B)
    # 1-body part ... exact mean-field energy shift E_MF[rho] - E_MF[rho_HF] with the bare Hamiltonian (LHO basis) ...
    #   (Tr(h rho) is not the mean-field energy, it counts the 2-body part twice & the 3-body part three times)
    #   Note: H1b_energy evaluates the full sum over all orbitals (only zero Rho_ad are skipped), it does not depend on the occupations of Orb ...
    Rho_LHO = O1b_transformation(Rho,O1b_transpose(C_NAT))
    Rho_HF_LHO = O1b_transformation(Rho_HF,O1b_transpose(C_HF))

    E_MF = H1b_energy(Params,Orb,Orb_NN,Orb_NNN,T_bare,V_NN_bare,V_NNN_bare,Rho_LHO)
    E_HF = H1b_energy(Params,Orb,Orb_NN,Orb_NNN,T_bare,V_NN_bare,V_NNN_bare,Rho_HF_LHO)

    E_rho = E_MF - E_HF

    # Calculate the 2-body correlation energy ... (V_NN density dependent, contains the 1/4 W rho Sigma 3-body part)
    E_sig = H2b_energy(Params,Orb,Orb_NN,V_NN,Sigma)

    # Subtraction of the Sigma terms of A & B ... only if Sigma enters A & B ...
    E_sub = 0.0
    if Params.Calc.RPA.TBCF
        E_sub = RPA_energy_sub(Params,Orb,Orb_1p1h,Orb_NN,E,X,Y,V_NN,Sigma)
    end

    return E_rho, E_sig, E_sub
end

function RPA_energy_sub(Params::Parameters,Orb::Vector{Orb1B},Orb_1p1h::Orb2B_1p1h,Orb_NN::Orb2B,
                        E::Matrix{Vector{ComplexF64}},X::Matrix{Matrix{ComplexF64}},Y::Matrix{Matrix{ComplexF64}},V_NN::O2B,Sigma::O2B)
    # Read parameters ...
    N_max = Params.Calc.Nmax
    J_max = 2*N_max + 1

    # Initialize the list of values of J & P for iteration ...
    JP_list = JP_initialize(J_max)
    N_JP = length(JP_list)

    # Calculate subtraction energy of the 2-body correlation function terms of A & B ...
    Zero = [zeros(Float64,Orb_1p1h.N[P,J+1],Orb_1p1h.N[P,J+1]) for P in 1:2, J in 0:J_max]
    @time dA, dB = RPA_allocate_TBCF(Params,Orb,Orb_1p1h,Orb_NN,V_NN,Sigma,Zero,deepcopy(Zero))

    # Results of individual J & P blocks ...
    E_sub_JP = zeros(Float64,N_JP)

    @inbounds Threads.@threads for i_JP in 1:N_JP
        J, P = JP_list[i_JP][1], JP_list[i_JP][2]

        N_ph = Orb_1p1h.N[P,J+1]
        if N_ph == 0
            continue
        end

        Orb_1p1h_JP = Orb_1p1h.Ind[P,J+1]
        J_hat = Float64(2*J + 1)

        E_JP, X_JP, Y_JP = E[P,J+1], X[P,J+1], Y[P,J+1]

        # Skip the spurious 1- Goldstone level (i_nu = 1 of the orthogonalized 1- block) ...
        nu_start = RPA_spurious(Params.Calc.RPA.Ortho,J,P) ? 2 : 1
        dA_JP, dB_JP = dA[P,J+1], dB[P,J+1]

        # Occupation differences D_ph = n_h - n_p ... (RPA metric N)
        D_JP = zeros(Float64,N_ph)
        @inbounds for i_ph in 1:N_ph
            a_p, a_h, T_ph = Orb_1p1h_JP[i_ph].a, Orb_1p1h_JP[i_ph].b, Orb_1p1h_JP[i_ph].T
            if T_ph == -1
                D_JP[i_ph] = Orb[a_h].pO - Orb[a_p].pO
            else
                D_JP[i_ph] = Orb[a_h].nO - Orb[a_p].nO
            end
        end

        E_subSum = 0.0
        @inbounds for i_nu in nu_start:N_ph
            X_nu, Y_nu = view(X_JP,:,i_nu), view(Y_JP,:,i_nu)

            # Physical solutions only ... non-zero norm & real energy ...
            RPA_norm = 0.0
            @inbounds for i_ph in 1:N_ph
                RPA_norm += D_JP[i_ph] * (abs2(X_nu[i_ph]) - abs2(Y_nu[i_ph]))
            end

            if abs(RPA_norm) < 1e-8
                continue
            end

            if abs(imag(E_JP[i_nu])) > 1e-8
                continue
            end

            # Energy of the Sigma terms of A & B ... Y_nu^+ dA Y_nu + X_nu^+ dB Y_nu ...
            E_subSum += J_hat * real(dot(Y_nu,dA_JP * Y_nu) + dot(X_nu,dB_JP * Y_nu))
        end

        E_sub_JP[i_JP] = E_subSum
    end

    return sum(E_sub_JP)
end