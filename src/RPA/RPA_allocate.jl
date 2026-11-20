function RPA_allocate(Params::Parameters,Orb::Vector{Orb1B},Orb_1p1h::Orb2B_1p1h,Orb_NN::Orb2B,h_N::O1B,V_NN::O2B,Rho::O1B,Sigma::O2B,TBCF::Bool)
    # Read calculation parameters ...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1

    JP_list = JP_initialize(J_max)
    
    println("\nAllocating RPA matrices A, B & N ...")

    # Initialize matrices A & B ...
    A = Matrix{Matrix{Float64}}(undef,2,J_max+1)
    B = Matrix{Matrix{Float64}}(undef,2,J_max+1)
    N = Matrix{Matrix{Float64}}(undef,2,J_max+1)

    # Allocate matrices A & B ... no 2-body correlation function yet ...
    @time @inbounds Threads.@threads for JP in JP_list
        J, P = JP[1], JP[2]
        N_ph = Orb_1p1h.N[P,J+1]
        Orb_1p1h_JP = Orb_1p1h.Ind[P,J+1]

        A_JP = zeros(Float64,N_ph,N_ph)
        B_JP = zeros(Float64,N_ph,N_ph)
        N_JP = zeros(Float64,N_ph,N_ph)

        @inbounds for i_ph in 1:N_ph
            Orb_ph = Orb_1p1h_JP[i_ph]
            a_p, a_h, T_ph = Orb_ph.a, Orb_ph.b, Orb_ph.T
            n_p, n_h = 0.0, 0.0

            l_p, j_p = Orb[a_p].l, Orb[a_p].j
            l_h, j_h = Orb[a_h].l, Orb[a_h].j

            if T_ph == -1
                n_p = Orb[a_p].pO
                n_h = Orb[a_h].pO
            elseif T_ph == 1
                n_p = Orb[a_p].nO
                n_h = Orb[a_h].nO
            end

            dn_ph = n_h - n_p

            @inbounds for i_qg in 1:i_ph
                Orb_qg = Orb_1p1h_JP[i_qg]
                a_q, a_g, T_qg = Orb_qg.a, Orb_qg.b, Orb_qg.T
                n_q, n_g = 0.0, 0.0

                l_q, j_q = Orb[a_q].l, Orb[a_q].j
                l_g, j_g = Orb[a_g].l, Orb[a_g].j

                if T_qg == -1
                    n_q = Orb[a_q].pO
                    n_g = Orb[a_g].pO
                elseif T_qg == 1
                    n_q = Orb[a_q].nO
                    n_g = Orb[a_g].nO
                end

                dn_qg = n_g - n_q

                delta_pq = Float64(kronecker_delta(a_p,a_q))
                delta_hg = Float64(kronecker_delta(a_h,a_g))
                delta_pg = Float64(kronecker_delta(a_p,a_g))
                delta_hq = Float64(kronecker_delta(a_h,a_q))

                P_pg = rem(l_p + l_g,2) + 1
                P_pq = rem(l_p + l_q,2) + 1
    
                ASum = 0.0
                BSum = 0.0
                NSum = 0.0

                # 1-body mean-field term h & the overlap matrix N ...
                if T_ph == T_qg
                    if T_ph == -1
                        h_pq, h_hg = h_N.p[a_p,a_q], h_N.p[a_h,a_g]

                        AME = 0.5 * (dn_ph + dn_qg) * (h_pq * delta_hg - h_hg * delta_pq)
                        ASum += AME

                        h_pg, h_hq = h_N.p[a_p,a_g], h_N.p[a_h,a_q]

                        BME = 0.5 * phase(J + div(j_p + j_h,2)) * (dn_ph - dn_qg) * (h_pg * delta_hq - h_hq * delta_pg)
                        BSum += BME

                        NME = delta_pq * delta_hg * dn_ph
                        NSum += NME

                    elseif T_ph == 1
                        h_pq, h_hg = h_N.n[a_p,a_q], h_N.n[a_h,a_g]

                        AME = 0.5 * (dn_ph + dn_qg) * (h_pq * delta_hg - h_hg * delta_pq)
                        ASum += AME

                        h_pg, h_hq = h_N.n[a_p,a_g], h_N.n[a_h,a_q]

                        BME = 0.5 * phase(J + div(j_p + j_h,2)) * (dn_ph - dn_qg) * (h_pg * delta_hq - h_hq * delta_pg)
                        BSum += BME

                        NME = delta_pq * delta_hg * dn_ph
                        NSum += NME
                    end
                end

                # 2-body scattering TDA matrix A ...
                if rem(l_p + l_g, 2) == rem(l_q + l_h, 2)
                    @inbounds for I in div(max(abs(j_p - j_g),abs(j_q - j_h)),2):div(min(j_p + j_g,j_q + j_h),2)
                        I_hat = Float64(2*I + 1)
                        if T_ph == -1 && T_qg == -1
                            Amp = dn_ph * dn_qg * phase(div(j_h + j_q,2) + I) * I_hat * w6j(j_p,j_h,2*J,j_q,j_g,2*I)
                            ME = Amp * O2b_pp(a_p,a_g,a_h,a_q,I,P_pg,V_NN,Orb,Orb_NN)
                            ASum +=  ME
                        elseif T_ph == -1 && T_qg == 1
                            Amp = dn_ph * dn_qg * phase(div(j_h + j_q,2) + I) * I_hat * w6j(j_p,j_h,2*J,j_q,j_g,2*I)
                            ME = Amp * O2b_pn(a_p,a_g,a_h,a_q,I,P_pg,V_NN,Orb_NN)
                            ASum += ME
                        elseif T_ph == 1 && T_qg == -1
                            Amp = dn_ph * dn_qg * phase(div(j_g + j_p,2) + I) * I_hat * w6j(j_q,j_g,2*J,j_p,j_h,2*I)
                            ME = Amp * O2b_pn(a_q,a_h,a_g,a_p,I,P_pg,V_NN,Orb_NN)
                            ASum += ME
                        elseif T_ph == 1 && T_qg == 1
                            Amp = dn_ph * dn_qg * phase(div(j_h + j_q,2) + I) * I_hat * w6j(j_p,j_h,2*J,j_q,j_g,2*I)
                            ME = Amp * O2b_nn(a_p,a_g,a_h,a_q,I,P_pg,V_NN,Orb,Orb_NN)
                            ASum +=  ME
                        end
                    end
                end
                
                # 2-body correlation matrix B ...
                if rem(l_p + l_q, 2) == rem(l_h + l_g, 2)
                    @inbounds for I in div(max(abs(j_p - j_q),abs(j_h - j_g)),2):div(min(j_p + j_q,j_h + j_g),2)
                        I_hat = Float64(2*I + 1)
                        if T_ph == -1 && T_qg == -1
                            Amp = dn_ph * dn_qg * phase(div(j_h + j_q,2) + J + I) * I_hat * w6j(j_p,j_h,2*J,j_g,j_q,2*I) 
                            ME = Amp * O2b_pp(a_p,a_q,a_h,a_g,I,P_pq,V_NN,Orb,Orb_NN)
                            BSum += ME
                        elseif T_ph == -1 && T_qg == 1
                            Amp = dn_ph * dn_qg * phase(div(j_h + j_q,2) + J + I) * I_hat * w6j(j_p,j_h,2*J,j_g,j_q,2*I)
                            ME = Amp * O2b_pn(a_p,a_q,a_h,a_g,I,P_pq,V_NN,Orb_NN)
                            BSum += ME
                        elseif T_ph == 1 && T_qg == -1
                            Amp = dn_ph * dn_qg * phase(div(j_g + j_p,2) + J + I) * I_hat * w6j(j_q,j_g,2*J,j_h,j_p,2*I)
                            ME = Amp * O2b_pn(a_q,a_p,a_g,a_h,I,P_pq,V_NN,Orb_NN)
                            BSum += ME
                        elseif T_ph == 1 && T_qg == 1
                            Amp = dn_ph * dn_qg * phase(div(j_h + j_q,2) + J + I) * I_hat * w6j(j_p,j_h,2*J,j_g,j_q,2*I)
                            ME = Amp * O2b_nn(a_p,a_q,a_h,a_g,I,P_pq,V_NN,Orb,Orb_NN)
                            BSum += ME
                        end
                    end
                end

                A_JP[i_ph,i_qg] = ASum
                B_JP[i_ph,i_qg] = BSum
                N_JP[i_ph,i_qg] = NSum

                if i_ph != i_qg
                    A_JP[i_qg,i_ph] = ASum
                    B_JP[i_qg,i_ph] = BSum
                    N_JP[i_qg,i_ph] = NSum
                end
            end
        end

        # For given J & P allocate A & B ...
        A[P,J+1] = A_JP
        B[P,J+1] = B_JP
        N[P,J+1] = N_JP
    end

    # If available & set to true, include 2-body correlation function effects ...
    if TBCF == true
        @time A, B = RPA_allocate_TBCF(Params,Orb,Orb_1p1h,Orb_NN,V_NN,Sigma,A,B)

        if Params.Calc.RPA.DEV
            # 1p-1h to 2p-2h self-energy coupling of A (compensates the contact terms O(V sigma) of A from RPA_allocate_TBCF) ...
            @time A = RPA_allocate_SE(Params,Orb,Orb_1p1h,Orb_NN,h_N,V_NN,A, Mode = "OnShell")
        end
    end

    println("\n\tRPA matrices A, B & N allocated ...")

    return A, B, N
end

function RPA_allocate_TBCF(Params::Parameters,Orb::Vector{Orb1B},Orb_1p1h::Orb2B_1p1h,Orb_NN::Orb2B,V_NN::O2B,Sigma::O2B,A::Matrix{Matrix{Float64}},B::Matrix{Matrix{Float64}})
    # Read calculation parameters ...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1
    a_max = div((N_max + 1)*(N_max + 2),2)

    JP_list = JP_initialize(J_max)

    println("\t\tComputing 2-body correlation function contributions to the RPA matrices A & B ...")

    # Precompute box diagram 1 ...
    W1_NN, Orb_NN_temp = O2b_temp_initialize(Params,Orb,Make_Orb_NN = true)

    @inbounds for JP in JP_list
        J, P = JP[1], JP[2]
        N = Orb_NN_temp.N[P,J+1]

        # Cross-coupled MEs X[Bra,Col] = sum_J_1 (2J_1 + 1) {j_k j_l J; j_d j_b J_1} <kb;J_1|O|ld;J_1> ... Bra = (k,l) & Col = (b,d) ...
            # pp, pn & nn ... <kb|O|ld> with phase (-1)^J_1 ...
        ppXV, ppXS = zeros(Float64,N,N), zeros(Float64,N,N)
        pnXV, pnXS = zeros(Float64,N,N), zeros(Float64,N,N)
        nnXV, nnXS = zeros(Float64,N,N), zeros(Float64,N,N)
            # np ... <bk|O|dl> with phase (-1)^J_1 * (-1)^((j_k + j_l)/2) ...
        npXV, npXS = zeros(Float64,N,N), zeros(Float64,N,N)
            # Mixed pn ... <kb|O|dl> without phase ...
        mpnXV, mpnXS = zeros(Float64,N,N), zeros(Float64,N,N)

        # Each Bra is spawned as a separate task for dynamic load balancing ...
        @sync for Bra in 1:N
            Threads.@spawn @inbounds begin
                a_k = Orb_NN_temp.Ind[P,J+1][Bra][1]
                a_l = Orb_NN_temp.Ind[P,J+1][Bra][2]

                l_k, j_k = Orb[a_k].l, Orb[a_k].j
                l_l, j_l = Orb[a_l].l, Orb[a_l].j

                Phase_kl = phase(div(j_k + j_l,2))

                @inbounds for Col in 1:N
                    a_b = Orb_NN_temp.Ind[P,J+1][Col][1]
                    a_d = Orb_NN_temp.Ind[P,J+1][Col][2]

                    l_b, j_b = Orb[a_b].l, Orb[a_b].j
                    l_d, j_d = Orb[a_d].l, Orb[a_d].j

                    P_kb = parity(l_k + l_b)

                    ppV, ppS = 0.0, 0.0
                    pnV, pnS = 0.0, 0.0
                    nnV, nnS = 0.0, 0.0
                    npV, npS = 0.0, 0.0
                    mpnV, mpnS = 0.0, 0.0

                    J_1_min, J_1_max = div(max(abs(j_k - j_b), abs(j_l - j_d)),2), div(min(j_k + j_b, j_l + j_d),2)
                    @inbounds for J_1 in J_1_min:J_1_max
                        Amp_J_1 = hat(2*J_1) * w6j(j_k,j_l,2*J,j_d,j_b,2*J_1)
                        Amp = phase(J_1) * Amp_J_1

                        ppV += Amp * O2b_pp(a_k,a_b,a_l,a_d,J_1,P_kb,V_NN,Orb,Orb_NN)
                        ppS += Amp * O2b_pp(a_k,a_b,a_l,a_d,J_1,P_kb,Sigma,Orb,Orb_NN)

                        pnV += Amp * O2b_pn(a_k,a_b,a_l,a_d,J_1,P_kb,V_NN,Orb_NN)
                        pnS += Amp * O2b_pn(a_k,a_b,a_l,a_d,J_1,P_kb,Sigma,Orb_NN)

                        nnV += Amp * O2b_nn(a_k,a_b,a_l,a_d,J_1,P_kb,V_NN,Orb,Orb_NN)
                        nnS += Amp * O2b_nn(a_k,a_b,a_l,a_d,J_1,P_kb,Sigma,Orb,Orb_NN)

                        npV += Phase_kl * Amp * O2b_pn(a_b,a_k,a_d,a_l,J_1,P_kb,V_NN,Orb_NN)
                        npS += Phase_kl * Amp * O2b_pn(a_b,a_k,a_d,a_l,J_1,P_kb,Sigma,Orb_NN)

                        mpnV += Amp_J_1 * O2b_pn(a_k,a_b,a_d,a_l,J_1,P_kb,V_NN,Orb_NN)
                        mpnS += Amp_J_1 * O2b_pn(a_k,a_b,a_d,a_l,J_1,P_kb,Sigma,Orb_NN)
                    end

                    ppXV[Bra,Col], ppXS[Bra,Col] = ppV, ppS
                    pnXV[Bra,Col], pnXS[Bra,Col] = pnV, pnS
                    nnXV[Bra,Col], nnXS[Bra,Col] = nnV, nnS
                    npXV[Bra,Col], npXS[Bra,Col] = npV, npS
                    mpnXV[Bra,Col], mpnXS[Bra,Col] = mpnV, mpnS
                end
            end
        end

        # W1[Bra,Ket] = sum_Col X_V[Bra,Col] * X_S[Ket,Col] + X_S[Bra,Col] * X_V[Ket,Col] ...
            # Proton-proton & proton-neutron part ...
        mul!(W1_NN.pp[P,J+1],ppXV,ppXS')
        mul!(W1_NN.pp[P,J+1],ppXS,ppXV',1.0,1.0)
        mul!(W1_NN.pp[P,J+1],pnXV,pnXS',1.0,1.0)
        mul!(W1_NN.pp[P,J+1],pnXS,pnXV',1.0,1.0)
            # Proton-neutron part ...
        mul!(W1_NN.pn[P,J+1],mpnXV,mpnXS')
        mul!(W1_NN.pn[P,J+1],mpnXS,mpnXV',1.0,1.0)
            # Neutron-neutron & neutron-proton part ...
        mul!(W1_NN.nn[P,J+1],nnXV,nnXS')
        mul!(W1_NN.nn[P,J+1],nnXS,nnXV',1.0,1.0)
        mul!(W1_NN.nn[P,J+1],npXV,npXS',1.0,1.0)
        mul!(W1_NN.nn[P,J+1],npXS,npXV',1.0,1.0)
    end

    # Precompute box diagram 2 ...
    W2_NN = O2b_temp_initialize(Params,Orb)

    @inbounds for JP in JP_list
        J, P = JP[1], JP[2]
        N = Orb_NN_temp.N[P,J+1]

        # Direct J-coupled MEs O[Bra,Ket] = <kl;J|O|mn;J> ... Bra = (k,l) & Ket = (m,n) ...
        ppV, ppS = zeros(Float64,N,N), zeros(Float64,N,N)
        pnV, pnS = zeros(Float64,N,N), zeros(Float64,N,N)
        nnV, nnS = zeros(Float64,N,N), zeros(Float64,N,N)

        @inbounds Threads.@threads for Bra in 1:N
            a_k = Orb_NN_temp.Ind[P,J+1][Bra][1]
            a_l = Orb_NN_temp.Ind[P,J+1][Bra][2]

            @inbounds for Ket in 1:N
                a_m = Orb_NN_temp.Ind[P,J+1][Ket][1]
                a_n = Orb_NN_temp.Ind[P,J+1][Ket][2]

                ppV[Bra,Ket] = O2b_pp(a_k,a_l,a_m,a_n,J,P,V_NN,Orb,Orb_NN)
                ppS[Bra,Ket] = O2b_pp(a_k,a_l,a_m,a_n,J,P,Sigma,Orb,Orb_NN)

                pnV[Bra,Ket] = O2b_pn(a_k,a_l,a_m,a_n,J,P,V_NN,Orb_NN)
                pnS[Bra,Ket] = O2b_pn(a_k,a_l,a_m,a_n,J,P,Sigma,Orb_NN)

                nnV[Bra,Ket] = O2b_nn(a_k,a_l,a_m,a_n,J,P,V_NN,Orb,Orb_NN)
                nnS[Bra,Ket] = O2b_nn(a_k,a_l,a_m,a_n,J,P,Sigma,Orb,Orb_NN)
            end
        end

        # W2[Bra,Ket] = sum_(c,d) <kl|V|cd> <mn|Sigma|cd> + <kl|Sigma|cd> <mn|V|cd> ...
        mul!(W2_NN.pp[P,J+1],ppV,ppS')
        mul!(W2_NN.pp[P,J+1],ppS,ppV',1.0,1.0)
        mul!(W2_NN.pn[P,J+1],pnV,pnS')
        mul!(W2_NN.pn[P,J+1],pnS,pnV',1.0,1.0)
        mul!(W2_NN.nn[P,J+1],nnV,nnS')
        mul!(W2_NN.nn[P,J+1],nnS,nnV',1.0,1.0)
    end

    if Params.Calc.RPA.DEV
        # Precompute the 1-body polarization diagrams (contact terms O(V sigma) of A, cf. RPA_allocate_SE) ... sum_(c,d) <lb|Sigma|cd> <kb|V|cd> + <kb|Sigma|cd> <lb|V|cd> = W2[(k,b),(l,b)] ...
        pf, nf = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)

        # Calculate 1-body polarization potential ...
        List_kl = [(Orb_a,Orb_b) for Orb_a in Orb for Orb_b in Orb if Orb_a.a <= Orb_b.a]

        @inbounds Threads.@threads for Orb_kl in List_kl
            (Orb_k,Orb_l) = Orb_kl
            a_k, l_k, j_k = Orb_k.a, Orb_k.l, Orb_k.j
            a_l, l_l, j_l = Orb_l.a, Orb_l.l, Orb_l.j

            if j_k != j_l
                continue
            end

            pfSum, nfSum = 0.0, 0.0

            @inbounds for Orb_b in Orb
                a_b, l_b, j_b = Orb_b.a, Orb_b.l, Orb_b.j

                P_kb = parity(l_k + l_b)

                if P_kb != parity(l_l + l_b)
                    continue
                end

                I_min, I_max = div(abs(j_k - j_b),2), div(j_k + j_b,2)
                @inbounds for I in I_min:I_max
                    Amp = 0.5 * hat(2*I) / hat(j_k)

                    # Proton-proton polarization potential ...
                    ppME = 0.5 * Amp * O2b_temp(a_k,a_b,a_l,a_b,I,P_kb,-1,W2_NN,Orb_NN_temp)

                    pnME = Amp * O2b_temp(a_k,a_b,a_l,a_b,I,P_kb,0,W2_NN,Orb_NN_temp)

                    pfSum += ppME + pnME

                    # Neutron-neutron polarization potential ...
                    nnME = 0.5 * Amp * O2b_temp(a_k,a_b,a_l,a_b,I,P_kb,1,W2_NN,Orb_NN_temp)

                    npME = phase(1 + div(j_k + j_l,2)) * Amp * O2b_temp(a_b,a_k,a_b,a_l,I,P_kb,0,W2_NN,Orb_NN_temp)

                    nfSum += nnME + npME
                end
            end

            pf[a_k,a_l] = pfSum
            nf[a_k,a_l] = nfSum

            if a_k != a_l
                pf[a_l,a_k] = pfSum
                nf[a_l,a_k] = nfSum
            end
        end
    end

    # Allocate 2-body correlation function contributions to the RPA matrices A & B ...
    @inbounds for JP in JP_list
        J, P = JP[1], JP[2]
        N_ph = Orb_1p1h.N[P,J+1]
        Orb_1p1h_JP = Orb_1p1h.Ind[P,J+1]

        A_JP = zeros(Float64,N_ph,N_ph)
        B_JP = zeros(Float64,N_ph,N_ph)

        @inbounds Threads.@threads for i_ph in 1:N_ph
            Orb_ph = Orb_1p1h_JP[i_ph]
            a_p, a_h, T_ph = Orb_ph.a, Orb_ph.b, Orb_ph.T

            l_p, j_p = Orb[a_p].l, Orb[a_p].j
            l_h, j_h = Orb[a_h].l, Orb[a_h].j

            @inbounds for i_qg in 1:i_ph
                Orb_qg = Orb_1p1h_JP[i_qg]
                a_q, a_g, T_qg = Orb_qg.a, Orb_qg.b, Orb_qg.T

                l_q, j_q = Orb[a_q].l, Orb[a_q].j
                l_g, j_g = Orb[a_g].l, Orb[a_g].j

                P_pg = parity(l_p + l_g)
                P_pq = parity(l_p + l_q)

                ASum = 0.0
                BSum = 0.0

                if Params.Calc.RPA.DEV
                    # 1-body polarization diagrams ... contact terms O(V sigma) of A (backward branches of the 2nd order self-energy) ...
                    if T_ph == T_qg
                        Amp_A = phase(1)

                        if T_ph == -1
                            # Hole-hole & particle-particle polarization diagrams ...
                            if a_p == a_q
                                ASum += Amp_A * pf[a_h,a_g]
                            end

                            if a_h == a_g
                                ASum += Amp_A * pf[a_p,a_q]
                            end

                        elseif T_ph == 1
                            # Hole-hole & particle-particle polarization diagrams ...
                            if a_p == a_q
                                ASum += Amp_A * nf[a_h,a_g]
                            end

                            if a_h == a_g
                                ASum += Amp_A * nf[a_p,a_q]
                            end
                        end
                    end

                    # 1-body polarization diagrams of B ... (not included)
                    if T_ph == T_qg
                        Amp_B = phase(1 - J + div(j_p + j_h,2))

                        if T_ph == -1
                            # Hole-particle & particle-hole polarization diagrams ...
                            if a_p == a_g
                                BSum += Amp_B * pf[a_h,a_q]
                            end

                            if a_h == a_q
                                BSum += Amp_B * pf[a_p,a_g]
                            end

                        elseif T_ph == 1
                            # Hole-particle & particle-hole polarization diagrams ...
                            if a_p == a_g
                                BSum += Amp_B * nf[a_h,a_q]
                            end

                            if a_h == a_q
                                BSum += Amp_B * nf[a_p,a_g]
                            end
                        end
                    end
                end

                # 2-body scattering TDA matrix A ...
                    # Box diagram 1 ...
                I_min, I_max = div(max(abs(j_p - j_q), abs(j_h - j_g)),2), div(min(j_p + j_q, j_h + j_g),2)
                @inbounds for I in I_min:I_max
                    if T_ph == -1 && T_qg == -1
                        Amp = phase(2 + J + I + div(j_h + j_g,2)) * hat(2*I) * w6j(j_p,j_q,2*I,j_g,j_h,2*J)
                        ASum += Amp * O2b_temp(a_p,a_q,a_h,a_g,I,P_pq,-1,W1_NN,Orb_NN_temp)

                    elseif T_ph == -1 && T_qg == 1
                        Amp = phase(2 + J + I + div(j_h + j_q,2)) * hat(2*I) * w6j(j_p,j_q,2*I,j_g,j_h,2*J)
                        ASum += Amp * O2b_temp(a_p,a_q,a_h,a_g,I,P_pq,0,W1_NN,Orb_NN_temp)

                    elseif T_ph == 1 && T_qg == -1
                        Amp = phase(2 + J + I + div(j_g + j_p,2)) * hat(2*I) * w6j(j_q,j_p,2*I,j_h,j_g,2*J)
                        ASum += Amp * O2b_temp(a_q,a_p,a_g,a_h,I,P_pq,0,W1_NN,Orb_NN_temp)

                    elseif T_ph == 1 && T_qg == 1
                        Amp = phase(2 + J + I + div(j_h + j_g,2)) * hat(2*I) * w6j(j_p,j_q,2*I,j_g,j_h,2*J)
                        ASum += Amp * O2b_temp(a_p,a_q,a_h,a_g,I,P_pq,1,W1_NN,Orb_NN_temp)
                    end
                end
                    # Box diagram 2 ...
                I_min, I_max = div(max(abs(j_p - j_g), abs(j_q - j_h)),2), div(min(j_p + j_g, j_q + j_h),2)
                @inbounds for I in I_min:I_max
                    Amp = phase(1 + I + div(j_h + j_q,2)) * hat(2*I) * w6j(j_p,j_h,2*J,j_q,j_g,2*I)

                    if T_ph == -1 && T_qg == -1
                        ASum += 0.5 * Amp * O2b_temp(a_p,a_g,a_h,a_q,I,P_pg,-1,W2_NN,Orb_NN_temp)

                    elseif T_ph == -1 && T_qg == 1
                        ASum += Amp * O2b_temp(a_p,a_g,a_h,a_q,I,P_pg,0,W2_NN,Orb_NN_temp)

                    elseif T_ph == 1 && T_qg == -1
                        ASum += phase(div(j_p + j_h + j_q + j_g,2)) * Amp * O2b_temp(a_g,a_p,a_q,a_h,I,P_pg,0,W2_NN,Orb_NN_temp)

                    elseif T_ph == 1 && T_qg == 1
                        ASum += 0.5 * Amp * O2b_temp(a_p,a_g,a_h,a_q,I,P_pg,1,W2_NN,Orb_NN_temp)
                    end
                end

                # 2-body correlation matrix B ...
                    # Box diagram 1 ...
                I_min, I_max = div(max(abs(j_p - j_g), abs(j_h - j_q)),2), div(min(j_p + j_g, j_h + j_q),2)
                @inbounds for I in I_min:I_max
                    if T_ph == -1 && T_qg == -1
                        Amp = phase(1 + I + div(j_h + j_g,2)) * hat(2*I) * w6j(j_p,j_g,2*I,j_q,j_h,2*J)
                        BSum += Amp * O2b_temp(a_p,a_g,a_h,a_q,I,P_pg,-1,W1_NN,Orb_NN_temp)

                    elseif T_ph == -1 && T_qg == 1
                        Amp = phase(1 + I + div(j_h + j_q,2)) * hat(2*I) * w6j(j_p,j_g,2*I,j_q,j_h,2*J)
                        BSum += Amp * O2b_temp(a_p,a_g,a_h,a_q,I,P_pg,0,W1_NN,Orb_NN_temp)

                    elseif T_ph == 1 && T_qg == -1
                        Amp = phase(1 + I + div(j_g + j_p,2)) * hat(2*I) * w6j(j_q,j_h,2*I,j_p,j_g,2*J)
                        BSum += Amp * O2b_temp(a_q,a_h,a_g,a_p,I,P_pg,0,W1_NN,Orb_NN_temp)

                    elseif T_ph == 1 && T_qg == 1
                        Amp = phase(1 + I + div(j_h + j_g,2)) * hat(2*I) * w6j(j_p,j_g,2*I,j_q,j_h,2*J)
                        BSum += Amp * O2b_temp(a_p,a_g,a_h,a_q,I,P_pg,1,W1_NN,Orb_NN_temp)
                    end
                end
                    # Box diagram 2 ...
                I_min, I_max = div(max(abs(j_p - j_q), abs(j_h - j_g)),2), div(min(j_p + j_q, j_h + j_g),2)
                @inbounds for I in I_min:I_max
                    Amp = phase(J + I + div(j_h + j_q,2)) * hat(2*I) * w6j(j_p,j_h,2*J,j_g,j_q,2*I)

                    if T_ph == -1 && T_qg == -1
                        BSum += 0.5 * Amp * O2b_temp(a_p,a_q,a_h,a_g,I,P_pq,-1,W2_NN,Orb_NN_temp)

                    elseif T_ph == -1 && T_qg == 1
                        BSum += Amp * O2b_temp(a_p,a_q,a_h,a_g,I,P_pq,0,W2_NN,Orb_NN_temp)

                    elseif T_ph == 1 && T_qg == -1
                        BSum += phase(div(j_p + j_h + j_q + j_g,2)) * Amp * O2b_temp(a_q,a_p,a_g,a_h,I,P_pq,0,W2_NN,Orb_NN_temp)

                    elseif T_ph == 1 && T_qg == 1
                        BSum += 0.5 * Amp * O2b_temp(a_p,a_q,a_h,a_g,I,P_pq,1,W2_NN,Orb_NN_temp)
                    end
                end

                A_JP[i_ph,i_qg] = ASum
                B_JP[i_ph,i_qg] = BSum

                if i_ph != i_qg
                    A_JP[i_qg,i_ph] = ASum
                    B_JP[i_qg,i_ph] = BSum
                end
            end
        end

        # For given J & P allocate A & B ...
        A[P,J+1] .+= A_JP
        B[P,J+1] .+= B_JP
    end

    println("\t\t\tFinished on evaluation of 2-body correlation function contributions to the RPA matrices A & B ...")

    return A, B
end

# For testing purposes ...
function RPA_allocate_SE_Old(Params::Parameters,Orb::Vector{Orb1B},Orb_1p1h::Orb2B_1p1h,Orb_NN::Orb2B,h_N::O1B,V_NN::O2B,A::Matrix{Matrix{Float64}};
                         Mode::String = "QP",Regularization::String = "DSRG",Gamma::Float64 = 2.0,Omega::Float64 = 0.0,dE_bin::Float64 = Gamma / 20.0)
    #=
        Energy-dependent 1p-1h to 2p-2h self-energy coupling of the TDA matrix A ... (general correlated reference in the NAT basis)
            A(E) = A^(s) + Sigma(E) ,   Sigma(E) = A_12 G(E) A_21 + B_12 G(-E) B_21 ,   G_c(E) = (E N_22 - A_22)^-1 = 1 / (N_c (E - Delta_c)) ,
            Delta_c = e_p1p1 + e_p2p2 - e_h1h1 - e_h2h2 ,   N_c = n_h1 n_h2 nb_p1 nb_p2 - nb_h1 nb_h2 n_p1 n_p2 ,   nb = 1 - n ...
            The static part A^(s) is the input A (incl. the contact terms O(V sigma) of A from RPA_allocate_TBCF, if used) ...

        2p-2h configurations c = (p1 p2 h1 h2) ... strictly ordered by the occupation numbers, min(n_h1,n_h2) > max(n_p1,n_p2) (N_c > 0),
            with p1,p2 (h1,h2) from the particle (hole) orbitals of the 1p-1h space (Tol_par, Tol_hol) ... valid also when all orbitals are included ...
            Forward coupling  A_12 ... alpha & beta vertices of the notes ...
            Backward coupling B_12(ph,c) = A_12(hp,c) ... the same vertices for the reversed pair (particle h, hole p) ... non-zero only if p
                acts as a hole or h as a particle of c, i.e. B_12 = 0 for disjoint particle & hole sets (then Sigma(E) = A_12 G(E) A_21 as in the notes) ...
            (checked exactly against the double commutators in the quasi-free reference state, i.e. sigma & 3-body cumulant dropped)
            The B self-energy (A_12 G(E) B_21 + B_12 G(-E) A_21) is neglected ...

        Spectral representation ... every matrix element is computed only once as
            Sigma(E)_ij = sum_b W_ij[b] f(Delta_b - E) + sum_b Wb_ij[b] f(Delta_b + E) ,   f(x) = 1/x (regularized) ,
            where the pole energies Delta_c are collected on a grid Delta_b with spacing dE_bin (linear splitting between 2 neighbouring bins) ...
            Any energy & any regularization is then a cheap contraction of W & Wb (forward & backward part) ...

        Regularization of the 2p-2h denominators ...
            "DSRG"      ... f(x) = (1 - exp(-(x/Gamma)^2)) / x ... (default) unchanged for |x| >> Gamma, smooth & bounded (|f| <= 0.64/Gamma) at the poles ...
            "Imaginary" ... f(x) = x / (x^2 + Gamma^2) ... real part of the imaginary level shift ...
            "None"      ... f(x) = 1/x ... singular at the poles (only for Mode = "OnShell" or "Fixed") ...

        Static approximation of A(E) ... Mode =
            "QP"      ... (default) quasiparticle solution of the energy-dependent problem ... in the N-orthonormal basis H(E) = N^-1/2 A(E) N^-1/2 ...
                          1) for every branch k (k-th eigenvalue lambda_k(E) of H(E)) all roots of lambda_k(E) = E are found (scan & bisection) ...
                             with a regularized (bounded & continuous) Sigma at least one root exists for every branch ...
                          2) for every branch the root with the largest 1p-1h weight Z = 1/(1 - <u|dH/dE|u>) is selected ...
                          3) static Hermitian matrix with exactly these eigenvalues & the closest orthonormal eigenvectors (symmetric
                             orthogonalization of the root eigenvectors, des Cloizeaux) ... A_eff = N^1/2 X_s E X_s^T N^1/2 ...
            "OnShell" ... dA_{ph,p'h'} = 1/2 * ( Sigma(e_pp - e_hh)_{ph,p'h'} + Sigma(e_p'p' - e_h'h')_{ph,p'h'} ) ...
            "Fixed"   ... dA = Sigma(Omega) ...

        J-coupled form ... |ph;JM> = sum_(m_p,m_h) (-1)^(j_h - m_h) <j_p m_p j_h -m_h|J M> a+_p a_h |0> (same as in RPA_allocate) ...
            <ab;K|V|cd;K> ... non-normalized antisymmetrized J-coupled MEs (O2b_pp, O2b_pn & O2b_nn) ...
            Orbitals are labeled by o = (a,T) with T = -1 (proton) & T = 1 (neutron) ... all isospin combinations of the
            intermediate 2p-2h configurations (pp, pn, np & nn) are summed explicitly ...

        Returns A_eff (the input A is overwritten channel by channel) ...
    =#

    # Read calculation parameters ...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1

    Tol_par = Params.Calc.RPA.Tol_par
    Tol_hol = Params.Calc.RPA.Tol_hol

    Tol_ord = 1e-8      # occupation ordering of the 2p-2h configurations (as for the 1p-1h pairs in orbitals_1p1h_make) ...
    Tol_N = 1e-8        # 2p-2h configurations with |N_c| < Tol_N are excluded (pseudo-inverse) ...
    Tol_dup = 1e-8      # QP ... a root eigenvector (numerically) identical to the already selected ones is replaced (fallback) ...

    if !(Mode in ("QP","OnShell","Fixed"))
        throw(ArgumentError("RPA_allocate_SE ... unknown Mode = ''" * Mode * "'' (use ''QP'', ''OnShell'' or ''Fixed'')"))
    end
    if !(Regularization in ("DSRG","Imaginary","None"))
        throw(ArgumentError("RPA_allocate_SE ... unknown Regularization = ''" * Regularization * "'' (use ''DSRG'', ''Imaginary'' or ''None'')"))
    end
    if Mode == "QP" && Regularization == "None"
        throw(ArgumentError("RPA_allocate_SE ... Mode = ''QP'' requires a regularized self-energy (Regularization = ''DSRG'' or ''Imaginary'')"))
    end

    JP_list = JP_initialize(J_max)

    println("\t\tComputing 1p-1h to 2p-2h self-energy coupling of 1p-1h A matrix ...")
    println("\t\t\tMode = " * Mode * (Mode == "Fixed" ? " (Omega = " * string(Omega) * " MeV)" : "") *
            ", Regularization = " * Regularization * ", Gamma = " * string(Gamma) * " MeV, dE_bin = " * string(dE_bin) * " MeV")

    # Particle & hole orbitals o = (a,T) of the intermediate 2p-2h configurations ... same criteria as for the 1p-1h space ...
    #   (the sets may overlap, e.g. Tol_hol < Tol_par ... the roles in a 2p-2h configuration are fixed by the occupation ordering)
    Par = Vector{Tuple{Int64,Int64}}()
    Hol = Vector{Tuple{Int64,Int64}}()

    @inbounds for Orb_a in Orb
        a = Orb_a.a
        if Orb_a.pO < Tol_par
            push!(Par,(a,-1))
        end
        if Orb_a.pO > Tol_hol
            push!(Hol,(a,-1))
        end
        if Orb_a.nO < Tol_par
            push!(Par,(a,1))
        end
        if Orb_a.nO > Tol_hol
            push!(Hol,(a,1))
        end
    end

    # Overlapping particle & hole sets ... only then the backward coupling B_12 can be non-zero ...
    Overlap = !isempty(intersect(Par,Hol))
    if Overlap
        println("\t\t\tOverlapping particle & hole orbitals ... backward 2p-2h coupling B_12 G(-E) B_21 is included in A ...")
    end

    # Define local functions for the self-energy ...
        # Occupation numbers n_a & nb_a = 1 - n_a of the orbital o = (a,T) ...
    function n(o::Tuple{Int64,Int64})
        a, T = o
        if T == -1
            return Orb[a].pO
        else
            return Orb[a].nO
        end
    end

    nb(o::Tuple{Int64,Int64}) = 1.0 - n(o)

        # Kronecker delta of two orbitals ...
    delta(o_1::Tuple{Int64,Int64},o_2::Tuple{Int64,Int64}) = (o_1 == o_2) ? 1.0 : 0.0

        # Generalized mean field e_ab = h_N[a,b] ... non-zero only for equal isospin, l & j ...
    function e(o_1::Tuple{Int64,Int64},o_2::Tuple{Int64,Int64})
        a, T_a = o_1
        b, T_b = o_2
        if (T_a != T_b) || (Orb[a].l != Orb[b].l) || (Orb[a].j != Orb[b].j)
            return 0.0
        end
        if T_a == -1
            return h_N.p[a,b]
        else
            return h_N.n[a,b]
        end
    end

        # J-coupled antisymmetrized ME <o_1 o_2;K|O|o_3 o_4;K> of a 2-body operator O for all isospin combinations (pp, pn, np & nn) ...
    function O2b_ME(O::O2B,o_1::Tuple{Int64,Int64},o_2::Tuple{Int64,Int64},o_3::Tuple{Int64,Int64},o_4::Tuple{Int64,Int64},K::Int64)
        a, T_a = o_1
        b, T_b = o_2
        c, T_c = o_3
        d, T_d = o_4

        l_a, j_a = Orb[a].l, Orb[a].j
        l_b, j_b = Orb[b].l, Orb[b].j
        l_c, j_c = Orb[c].l, Orb[c].j
        l_d, j_d = Orb[d].l, Orb[d].j

        # Charge, parity & angular momentum conservation ...
        if (T_a + T_b) != (T_c + T_d)
            return 0.0
        end
        if parity(l_a + l_b) != parity(l_c + l_d)
            return 0.0
        end
        if !triangle(j_a,j_b,2*K) || !triangle(j_c,j_d,2*K) || (K + 1) > size(Orb_NN.Map,3)
            return 0.0
        end

        P = parity(l_a + l_b)

        if T_a == T_b
            # Proton-proton & neutron-neutron MEs ... (only pairs inside the N2max truncation are available)
            if Orb_NN.Map[max(a,b),min(a,b),K+1,2] == 0 || Orb_NN.Map[max(c,d),min(c,d),K+1,2] == 0
                return 0.0
            end
            if T_a == -1
                return O2b_pp(a,b,c,d,K,P,O,Orb,Orb_NN)
            else
                return O2b_nn(a,b,c,d,K,P,O,Orb,Orb_NN)
            end
        elseif T_a == -1
            if T_c == -1
                # Proton-neutron ME <pn|O|pn> ...
                if Orb_NN.Map[a,b,K+1,1] == 0 || Orb_NN.Map[c,d,K+1,1] == 0
                    return 0.0
                end
                return O2b_pn(a,b,c,d,K,P,O,Orb_NN)
            else
                # <pn|O|np> = -(-1)^((j_c + j_d)/2 - K) <pn|O|pn> ...
                return -phase(div(j_c + j_d,2) - K) * O2b_ME(O,o_1,o_2,o_4,o_3,K)
            end
        else
            # <np|O|..> = -(-1)^((j_a + j_b)/2 - K) <pn|O|..> ...
            return -phase(div(j_a + j_b,2) - K) * O2b_ME(O,o_2,o_1,o_3,o_4,K)
        end
    end

        # Interaction V ...
    V(o_1,o_2,o_3,o_4,K::Int64) = O2b_ME(V_NN,o_1,o_2,o_3,o_4,K)

        # Occupation factors F^h_(hh1;pp1p2), F^p_(pp1;hh1h2) & G_(hh1;pp1) ...
    function Fh(h,h_1,p,p_1,p_2)
        F = 0.5 * (n(h) - n(p)) * (n(h_1) * nb(p_1) * nb(p_2) + nb(h_1) * n(p_1) * n(p_2))
        F += 0.5 * (n(h) * n(h_1) * nb(p_1) * nb(p_2) - nb(h) * nb(h_1) * n(p_1) * n(p_2))
        return F
    end

    function Fp(p,p_1,h,h_1,h_2)
        F = 0.5 * (n(h) - n(p)) * (nb(p_1) * n(h_1) * n(h_2) + n(p_1) * nb(h_1) * nb(h_2))
        F += 0.5 * (nb(p) * nb(p_1) * n(h_1) * n(h_2) - n(p) * n(p_1) * nb(h_1) * nb(h_2))
        return F
    end

    G(h,h_1,p,p_1) = (n(h) - n(p)) * (n(h_1) - n(p_1))

        # Regularized f(x) ~ 1/x & its derivative df/dx ...
    function f_reg(x::Float64)
        if Regularization == "DSRG"
            return (x == 0.0) ? 0.0 : -expm1(-(x / Gamma)^2) / x
        elseif Regularization == "Imaginary"
            return x / (x^2 + Gamma^2)
        else
            return (x == 0.0) ? 0.0 : 1.0 / x
        end
    end

    function df_reg(x::Float64)
        if Regularization == "DSRG"
            return (x == 0.0) ? 1.0 / Gamma^2 : (2.0 * (x / Gamma)^2 * exp(-(x / Gamma)^2) + expm1(-(x / Gamma)^2)) / x^2
        elseif Regularization == "Imaginary"
            return (Gamma^2 - x^2) / (x^2 + Gamma^2)^2
        else
            return (x == 0.0) ? 0.0 : -1.0 / x^2
        end
    end

        # Grid of the 2p-2h pole energies Delta_b ... covers all possible Delta_c = e_p1p1 + e_p2p2 - e_h1h1 - e_h2h2 ...
    e_par = [e(o,o) for o in Par]
    e_hol = [e(o,o) for o in Hol]
    Delta_min = 2.0 * minimum(e_par) - 2.0 * maximum(e_hol) - 2.0 * dE_bin
    Delta_max = 2.0 * maximum(e_par) - 2.0 * minimum(e_hol) + 2.0 * dE_bin
    N_bin = ceil(Int64,(Delta_max - Delta_min) / dE_bin) + 2
    Delta_bin = [Delta_min + (b - 1) * dE_bin for b in 1:N_bin]

        # Membership in the particle & hole sets ... (the external orbitals of the reversed pairs may enter any slot of c)
    is_par = Set(Par)
    is_hol = Set(Hol)

        # 2p-2h pole of the configuration c = (p1 p2 h1 h2) ... energy Delta_c & weight 1/N_c ...
        #   (0 if c is not in the 2p-2h space, i.e. p1,p2 not particles, h1,h2 not holes or not ordered by the occupation
        #    numbers min(n_h1,n_h2) > max(n_p1,n_p2), or if |N_c| < Tol_N)
    function pole(p_1,p_2,h_1,h_2)
        if !(p_1 in is_par) || !(p_2 in is_par) || !(h_1 in is_hol) || !(h_2 in is_hol)
            return 0.0, 0.0
        end
        if (min(n(h_1),n(h_2)) - max(n(p_1),n(p_2))) <= Tol_ord
            return 0.0, 0.0
        end
        N_c = n(h_1) * n(h_2) * nb(p_1) * nb(p_2) - nb(h_1) * nb(h_2) * n(p_1) * n(p_2)
        if abs(N_c) < Tol_N
            return 0.0, 0.0
        end
        return e(p_1,p_1) + e(p_2,p_2) - e(h_1,h_1) - e(h_2,h_2), 1.0 / N_c
    end

        # Add the term Weight * f(Delta - E) to the spectral representation W_ij ... (linear splitting between the 2 neighbouring bins)
    function add_pole!(W_ij,Delta::Float64,Weight::Float64)
        x = (Delta - Delta_min) / dE_bin
        b = floor(Int64,x) + 1
        t = x - (b - 1)
        W_ij[b] += (1.0 - t) * Weight
        W_ij[b+1] += t * Weight
        return
    end

        # J-coupled vertex alpha^(ph)_(h1;p1p2) = <h1 p;K|alpha|p1 p2;K> ... the hole h is a spectator ...
    function alpha(h,h_1,p,p_1,p_2,K::Int64)
        j_p, j_p2 = Orb[p[1]].j, Orb[p_2[1]].j
        ME = Fh(h,h_1,p,p_1,p_2) * V(h_1,p,p_1,p_2,K)
        ME += 0.5 * delta(p,p_2) * e(h_1,p_1) * G(h,h_1,p,p_1)
        ME -= 0.5 * phase(div(j_p + j_p2,2) - K) * delta(p,p_1) * e(h_1,p_2) * G(h,h_1,p,p_2)
        return ME
    end

        # J-coupled vertex beta^(ph)_(p1;h1h2) = <h p1;K|beta|h1 h2;K> ... the particle p is a spectator ...
    beta(p,p_1,h,h_1,h_2,K::Int64) = Fp(p,p_1,h,h_1,h_2) * V(h,p_1,h_1,h_2,K)

        # Particle-hole exchange term -sum_(p1,h1) alpha^(ph)_(h1;p1p') beta^(p'h')_(p1;h1h) / D_(p1p'h1h) ...
        #   recoupled through the (p p'^-1)_L & (h h'^-1)_L channels ... (p,h) = (p,h) & (q,g) = (p',h')
    function SE_exchange!(W_ij,p,h,q,g,J::Int64)
        j_p, j_h, j_q, j_g = Orb[p[1]].j, Orb[h[1]].j, Orb[q[1]].j, Orb[g[1]].j

        @inbounds for p_1 in Par
            j_p1 = Orb[p_1[1]].j
            @inbounds for h_1 in Hol
                j_h1 = Orb[h_1[1]].j

                if (h_1[2] + p[2]) != (p_1[2] + q[2])
                    continue
                end

                Delta_c, N_c_inv = pole(p_1,q,h_1,h)
                if N_c_inv == 0.0
                    continue
                end

                F_p = Fp(q,p_1,g,h_1,h)

                Sum = 0.0
                L_min = div(max(abs(j_p - j_q),abs(j_h - j_g),abs(j_p1 - j_h1)),2)
                L_max = div(min(j_p + j_q,j_h + j_g,j_p1 + j_h1),2)
                @inbounds for L in L_min:L_max
                    # Recoupling (p h^-1)_J (p' h'^-1)_J <- (p p'^-1)_L (h h'^-1)_L ...
                    C_JL = hat(2*L) * phase(div(j_h + j_q,2) + L + J) * w6j(j_p,j_h,2*J,j_g,j_q,2*L)

                    # Pandya transformation of alpha^(ph)_(h1;p1p') to the (p p'^-1)_L (p1 h1^-1)_L channel ...
                    Y_L = 0.0
                    K_min = div(max(abs(j_h1 - j_p),abs(j_p1 - j_q)),2)
                    K_max = div(min(j_h1 + j_p,j_p1 + j_q),2)
                    @inbounds for K in K_min:K_max
                        Amp = hat(2*K) * phase(div(j_q + j_p1,2) + K) * w6j(j_p,j_q,2*L,j_p1,j_h1,2*K)
                        Y_L += Amp * alpha(h,h_1,p,p_1,q,K)
                    end
                    Y_L *= phase(div(j_p + j_h1 + j_q + j_p1,2))

                    # Pandya transformation of beta^(p'h')_(p1;h1h) to the (p1 h1^-1)_L (h h'^-1)_L channel ...
                    Z_L = 0.0
                    K_min = div(max(abs(j_p1 - j_g),abs(j_h1 - j_h)),2)
                    K_max = div(min(j_p1 + j_g,j_h1 + j_h),2)
                    @inbounds for K in K_min:K_max
                        Amp = hat(2*K) * phase(div(j_h1 + j_h,2) + K) * w6j(j_p1,j_h1,2*L,j_h,j_g,2*K)
                        Z_L += Amp * V(p_1,g,h_1,h,K)
                    end
                    Z_L *= - F_p

                    Sum += C_JL * Y_L * Z_L
                end

                add_pole!(W_ij,Delta_c,- Sum * N_c_inv)
            end
        end

        return
    end

        # Spectral representation W_ij of the J-coupled self-energy Sigma(E)_{ph,p'h'} ... (p,h) = (p,h) & (q,g) = (p',h')
    function SE_element!(W_ij,p,h,q,g,J::Int64)
        l_p, j_p = Orb[p[1]].l, Orb[p[1]].j
        l_h, j_h = Orb[h[1]].l, Orb[h[1]].j
        l_q, j_q = Orb[q[1]].l, Orb[q[1]].j
        l_g, j_g = Orb[g[1]].l, Orb[g[1]].j

        Phase_phqg = phase(div(j_p + j_h + j_q + j_g,2))

        # Term 1 ... particle self-energy -1/2 delta_hh' sum_(h1,p1,p2) alpha^(ph)_(h1;p1p2) alpha^(p'h)_(h1;p1p2) / D_(p1p2hh1) ...
        if (h == g) && (l_p == l_q) && (j_p == j_q)
            @inbounds for h_1 in Hol, p_1 in Par, p_2 in Par
                if (h_1[2] + p[2]) != (p_1[2] + p_2[2])
                    continue
                end

                Delta_c, N_c_inv = pole(p_1,p_2,h,h_1)
                if N_c_inv == 0.0
                    continue
                end

                j_h1, j_p1, j_p2 = Orb[h_1[1]].j, Orb[p_1[1]].j, Orb[p_2[1]].j

                Sum = 0.0
                K_min = div(max(abs(j_h1 - j_p),abs(j_p1 - j_p2)),2)
                K_max = div(min(j_h1 + j_p,j_p1 + j_p2),2)
                @inbounds for K in K_min:K_max
                    Sum += hat(2*K) * alpha(h,h_1,p,p_1,p_2,K) * alpha(h,h_1,q,p_1,p_2,K)
                end

                add_pole!(W_ij,Delta_c,- 0.5 * Sum / hat(j_p) * N_c_inv)
            end
        end

        # Term 2 ... +1/2 sum_(p1,p2) alpha^(ph)_(h';p1p2) alpha^(p'h')_(h;p1p2) / D_(p1p2hh') ...
        @inbounds for p_1 in Par, p_2 in Par
            if (g[2] + p[2]) != (p_1[2] + p_2[2])
                continue
            end

            Delta_c, N_c_inv = pole(p_1,p_2,h,g)
            if N_c_inv == 0.0
                continue
            end

            j_p1, j_p2 = Orb[p_1[1]].j, Orb[p_2[1]].j

            Sum = 0.0
            K_min = div(max(abs(j_g - j_p),abs(j_h - j_q),abs(j_p1 - j_p2)),2)
            K_max = div(min(j_g + j_p,j_h + j_q,j_p1 + j_p2),2)
            @inbounds for K in K_min:K_max
                Amp = hat(2*K) * Phase_phqg * w6j(j_p,j_h,2*J,j_q,j_g,2*K)
                Sum += Amp * alpha(h,g,p,p_1,p_2,K) * alpha(g,h,q,p_1,p_2,K)
            end

            add_pole!(W_ij,Delta_c,0.5 * Sum * N_c_inv)
        end

        # Term 3 ... hole self-energy -1/2 delta_pp' sum_(p1,h1,h2) beta^(ph)_(p1;h1h2) beta^(ph')_(p1;h1h2) / D_(pp1h1h2) ...
        if (p == q) && (l_h == l_g) && (j_h == j_g)
            @inbounds for p_1 in Par, h_1 in Hol, h_2 in Hol
                if (h[2] + p_1[2]) != (h_1[2] + h_2[2])
                    continue
                end

                Delta_c, N_c_inv = pole(p,p_1,h_1,h_2)
                if N_c_inv == 0.0
                    continue
                end

                j_p1, j_h1, j_h2 = Orb[p_1[1]].j, Orb[h_1[1]].j, Orb[h_2[1]].j

                Sum = 0.0
                K_min = div(max(abs(j_h - j_p1),abs(j_h1 - j_h2)),2)
                K_max = div(min(j_h + j_p1,j_h1 + j_h2),2)
                @inbounds for K in K_min:K_max
                    Sum += hat(2*K) * beta(p,p_1,h,h_1,h_2,K) * beta(p,p_1,g,h_1,h_2,K)
                end

                add_pole!(W_ij,Delta_c,- 0.5 * Sum / hat(j_h) * N_c_inv)
            end
        end

        # Term 4 ... +1/2 sum_(h1,h2) beta^(ph)_(p';h1h2) beta^(p'h')_(p;h1h2) / D_(pp'h1h2) ...
        @inbounds for h_1 in Hol, h_2 in Hol
            if (h[2] + q[2]) != (h_1[2] + h_2[2])
                continue
            end

            Delta_c, N_c_inv = pole(p,q,h_1,h_2)
            if N_c_inv == 0.0
                continue
            end

            j_h1, j_h2 = Orb[h_1[1]].j, Orb[h_2[1]].j

            Sum = 0.0
            K_min = div(max(abs(j_h - j_q),abs(j_g - j_p),abs(j_h1 - j_h2)),2)
            K_max = div(min(j_h + j_q,j_g + j_p,j_h1 + j_h2),2)
            @inbounds for K in K_min:K_max
                Amp = hat(2*K) * Phase_phqg * w6j(j_p,j_h,2*J,j_q,j_g,2*K)
                Sum += Amp * beta(p,q,h,h_1,h_2,K) * beta(q,p,g,h_1,h_2,K)
            end

            add_pole!(W_ij,Delta_c,0.5 * Sum * N_c_inv)
        end

        # Term 5 ... -sum_(p1,h1) alpha^(ph)_(h1;p1p') beta^(p'h')_(p1;h1h) / D_(p1p'h1h) ...
        SE_exchange!(W_ij,p,h,q,g,J)

        # Term 6 ... -sum_(p1,h1) beta^(ph)_(p1;h1h') alpha^(p'h')_(h1;p1p) / D_(p1ph1h') ... (term 5 with ph <-> p'h')
        SE_exchange!(W_ij,q,g,p,h,J)

        return
    end

        # Self-energy matrix Sigma(E)_ij = sum_b W[b,i,j] f(Delta_b - E) + sum_b Wb[b,i,j] f(Delta_b + E) & its derivative dSigma/dE
        #   from the spectral representations W (forward, A_12 G(E) A_21) & Wb (backward, B_12 G(-E) B_21) ...
    function Sigma_E(W::Array{Float64,3},Wb::Array{Float64,3},E::Float64)
        f_E = [f_reg(Delta - E) for Delta in Delta_bin]
        Sigma = reshape(transpose(reshape(W,N_bin,:)) * f_E,size(W,2),size(W,3))
        if Overlap
            fb_E = [f_reg(Delta + E) for Delta in Delta_bin]
            Sigma += reshape(transpose(reshape(Wb,N_bin,:)) * fb_E,size(Wb,2),size(Wb,3))
        end
        return Sigma
    end

    function dSigma_E(W::Array{Float64,3},Wb::Array{Float64,3},E::Float64)
        df_E = [- df_reg(Delta - E) for Delta in Delta_bin]
        dSigma = reshape(transpose(reshape(W,N_bin,:)) * df_E,size(W,2),size(W,3))
        if Overlap
            dfb_E = [df_reg(Delta + E) for Delta in Delta_bin]
            dSigma += reshape(transpose(reshape(Wb,N_bin,:)) * dfb_E,size(Wb,2),size(Wb,3))
        end
        return dSigma
    end

        # Quasiparticle (QP) static matrix A_eff for one (J,P) channel ... A_s = A^(s), N_d = diagonal of the metric N ...
    function SE_QP(A_s::Matrix{Float64},N_d::Vector{Float64},W::Array{Float64,3},Wb::Array{Float64,3})
        N_ph = size(A_s,1)
        L = 1.0 ./ sqrt.(N_d)                                       # N^-1/2 ...

        H_0 = L .* A_s .* L'
        H(E) = Symmetric(H_0 + L .* Sigma_E(W,Wb,E) .* L')
        dH(E) = L .* dSigma_E(W,Wb,E) .* L'

        # 1-p1h weight of a root ... Z = 1/(1 - s) for the physical slope s = <u|dH/dE|u> <= 0, positive slopes (regularization) are penalized ...
        function weight(u::Vector{Float64},E::Float64)
            s = dot(u,dH(E) * u)
            return (s <= 0.0) ? 1.0 / (1.0 - s) : 1.0 / (1.0 + s)
        end

        # Energy window [E_lo,E_hi] ... every branch satisfies lambda_k(E_lo) > E_lo & lambda_k(E_hi) < E_hi (Sigma is bounded) ...
        e_0 = eigvals(Symmetric(H_0))
        E_lo, E_hi = e_0[1] - Gamma, e_0[N_ph] + Gamma
        while !(all(eigvals(H(E_lo)) .> E_lo))
            E_lo -= 5.0 * Gamma
        end
        while !(all(eigvals(H(E_hi)) .< E_hi))
            E_hi += 5.0 * Gamma
        end

        # Scan of g_k(E) = lambda_k(E) - E on a grid with step Gamma/4 ...
        E_grid = collect(range(E_lo,E_hi,length = ceil(Int64,(E_hi - E_lo) / (0.25 * Gamma)) + 1))
        g_grid = [eigvals(H(E)) .- E for E in E_grid]

        E_root = zeros(Float64,N_ph)
        Z_root = zeros(Float64,N_ph)
        X = zeros(Float64,N_ph,N_ph)

        @inbounds for k in 1:N_ph
            Z_best = -1.0
            @inbounds for i in 1:(length(E_grid) - 1)
                g_a, g_b = g_grid[i][k], g_grid[i+1][k]
                if (g_a > 0.0) == (g_b > 0.0)
                    continue
                end

                # Bisection of the sign change of g_k ...
                E_a, E_b = E_grid[i], E_grid[i+1]
                @inbounds for Iteration in 1:100
                    E_m = 0.5 * (E_a + E_b)
                    g_m = eigvals(H(E_m))[k] - E_m
                    if (g_m > 0.0) == (g_a > 0.0)
                        E_a = E_m
                    else
                        E_b = E_m
                    end
                    if (E_b - E_a) < 1e-10
                        break
                    end
                end

                E_k = 0.5 * (E_a + E_b)
                u_k = eigen(H(E_k)).vectors[:,k]
                Z_k = weight(u_k,E_k)

                # Root with the largest 1p-1h weight ...
                if Z_k > Z_best
                    Z_best = Z_k
                    E_root[k], Z_root[k] = E_k, Z_k
                    X[:,k] = u_k
                end
            end
        end

        # Fallback for (numerically) identical root eigenvectors ... direction orthogonal to the selected ones & its own
        # scalar self-consistency E = <v|H(E)|v> (root with the largest weight) ...
        N_fallback = 0
        Q = zeros(Float64,N_ph,0)
        @inbounds for k in sortperm(Z_root,rev = true)
            r = X[:,k] - Q * (Q' * X[:,k])
            if dot(r,r) < Tol_dup
                N_fallback += 1
                if norm(r) < 1e-12
                    r = svd(Q,full = true).U[:,size(Q,2)+1]
                end
                v = r / norm(r)
                q_grid = [dot(v,H(E) * v) - E for E in E_grid]
                Z_best = -1.0
                @inbounds for i in 1:(length(E_grid) - 1)
                    if (q_grid[i] > 0.0) == (q_grid[i+1] > 0.0)
                        continue
                    end
                    E_a, E_b = E_grid[i], E_grid[i+1]
                    @inbounds for Iteration in 1:100
                        E_m = 0.5 * (E_a + E_b)
                        if (dot(v,H(E_m) * v) - E_m > 0.0) == (q_grid[i] > 0.0)
                            E_a = E_m
                        else
                            E_b = E_m
                        end
                        if (E_b - E_a) < 1e-10
                            break
                        end
                    end
                    E_v = 0.5 * (E_a + E_b)
                    Z_v = weight(v,E_v)
                    if Z_v > Z_best
                        Z_best = Z_v
                        E_root[k], Z_root[k] = E_v, Z_v
                    end
                end
                X[:,k] = v
                r = v - Q * (Q' * v)
            end
            Q = hcat(Q,r / norm(r))
        end

        # Symmetric (Lowdin) orthogonalization of the root eigenvectors & the static Hermitian matrix ...
        S_X = Symmetric(X' * X)
        X_s = X * inv(sqrt(S_X))
        H_eff = X_s * Diagonal(E_root) * X_s'
        H_eff = 0.5 * (H_eff + H_eff')

        A_eff = H_eff ./ L ./ L'

        S_X_eig = eigvals(S_X)
        return A_eff, minimum(Z_root), S_X_eig[N_ph] / S_X_eig[1], N_fallback
    end

    # Allocate the self-energy contributions & the static matrices A for all (J,P) channels ...
    Z_min, Cond_max, N_fallback_total = 1.0, 1.0, 0

    @inbounds for JP in JP_list
        J, P = JP[1], JP[2]
        N_ph = Orb_1p1h.N[P,J+1]
        Orb_1p1h_JP = Orb_1p1h.Ind[P,J+1]

        if N_ph == 0
            continue
        end

        # Spectral representation of Sigma(E) ... forward W_JP & backward Wb_JP part ...
        W_JP = zeros(Float64,N_bin,N_ph,N_ph)
        Wb_JP = zeros(Float64,Overlap ? N_bin : 0,N_ph,N_ph)

        @inbounds Threads.@threads for i_ph in 1:N_ph
            Orb_ph = Orb_1p1h_JP[i_ph]
            p, h = (Orb_ph.a,Orb_ph.T), (Orb_ph.b,Orb_ph.T)
            j_p, j_h = Orb[p[1]].j, Orb[h[1]].j

            @inbounds for i_qg in 1:i_ph
                Orb_qg = Orb_1p1h_JP[i_qg]
                q, g = (Orb_qg.a,Orb_qg.T), (Orb_qg.b,Orb_qg.T)
                j_q, j_g = Orb[q[1]].j, Orb[g[1]].j

                # Forward part A_12 G(E) A_21 ...
                SE_element!(view(W_JP,:,i_ph,i_qg),p,h,q,g,J)

                # Backward part B_12 G(-E) B_21 ... the same element for the reversed pairs (h,p) & (h',p') at -E,
                #   (-1)^(j_h - j_p + j_h' - j_p') from the recoupling of the reversed pairs ... (only for overlapping particle & hole sets)
                if Overlap
                    SE_element!(view(Wb_JP,:,i_ph,i_qg),h,p,g,q,J)
                    Wb_JP[:,i_ph,i_qg] .*= phase(div(j_h - j_p + j_g - j_q,2))
                end

                if i_ph != i_qg
                    W_JP[:,i_qg,i_ph] .= W_JP[:,i_ph,i_qg]
                    Wb_JP[:,i_qg,i_ph] .= Wb_JP[:,i_ph,i_qg]
                end
            end
        end

        # Static matrix A^(s) ... the input A ...
        A_s = A[P,J+1]

        # Static approximation of A(E) = A^(s) + Sigma(E) ...
        if Mode == "Fixed"
            A[P,J+1] = A_s + Sigma_E(W_JP,Wb_JP,Omega)

        elseif Mode == "OnShell"
            w = [e((o.a,o.T),(o.a,o.T)) - e((o.b,o.T),(o.b,o.T)) for o in Orb_1p1h_JP]
            Sigma_w = [Sigma_E(W_JP,Wb_JP,w[i]) for i in 1:N_ph]
            A[P,J+1] = A_s + [0.5 * (Sigma_w[i][i,j] + Sigma_w[j][i,j]) for i in 1:N_ph, j in 1:N_ph]

        elseif Mode == "QP"
            N_d = [n((o.b,o.T)) - n((o.a,o.T)) for o in Orb_1p1h_JP]
            A_eff, Z_JP, Cond_JP, N_fallback_JP = SE_QP(A_s,N_d,W_JP,Wb_JP)
            A[P,J+1] = A_eff
            Z_min, Cond_max, N_fallback_total = min(Z_min,Z_JP), max(Cond_max,Cond_JP), N_fallback_total + N_fallback_JP
        end
    end

    if Mode == "QP"
        @printf("\t\t\tQP solution ... min 1p-1h weight Z = %.3f, max condition number of the root eigenvectors = %.2e, fallbacks = %d\n",
                Z_min, Cond_max, N_fallback_total)
    end

    println("\t\t\tFinished on evaluation of 1p-1h to 2p-2h self-energy contributions to the matrix ...")

    return A
end

function RPA_allocate_SE(Params::Parameters,Orb::Vector{Orb1B},Orb_1p1h::Orb2B_1p1h,Orb_NN::Orb2B,h_N::O1B,V_NN::O2B,A::Matrix{Matrix{Float64}};
                         Mode::String = "QP",Regularization::String = "DSRG",Gamma::Float64 = 2.0,Omega::Float64 = 0.0,dE_bin::Float64 = Gamma / 20.0,
                         Chunk::Int64 = 32768)
    #=
        Energy-dependent 1p-1h to 2p-2h self-energy coupling of the TDA matrix A ... (general correlated reference in the NAT basis)
            A(E) = A^(s) + Sigma(E) ,   Sigma(E) = A_12 G(E) A_21 + B_12 G(-E) B_21 ,   G_c(E) = (E N_22 - A_22)^-1 = 1 / (N_c (E - Delta_c)) ,
            Delta_c = e_p1p1 + e_p2p2 - e_h1h1 - e_h2h2 ,   N_c = n_h1 n_h2 nb_p1 nb_p2 - nb_h1 nb_h2 n_p1 n_p2 ,   nb = 1 - n ...
            The static part A^(s) is the input A (incl. the contact terms O(V sigma) of A from RPA_allocate_TBCF, if used) ...

        2p-2h configurations ... J-coupled & orthonormal, c = [(p1 p2)_J12 (h1 h2)_J34]_J with p1 <= p2 & h1 <= h2 (1/sqrt(2) for equal orbitals),
            strictly ordered by the occupation numbers, min(n_h1,n_h2) > max(n_p1,n_p2) (N_c > 0), p1,p2 (h1,h2) from the particle (hole)
            orbitals of the 1p-1h space (Tol_par, Tol_hol) ... valid also when all orbitals are included ...
            Forward coupling  A_12(ph,c) ... alpha & beta vertices of the notes, recoupled to the 2p-2h basis by a single 6j symbol ...
            Backward coupling B_12(ph,c) = A_12(hp,c) ... the same vertices for the reversed pair (particle h, hole p) ... non-zero only if p
                acts as a hole or h as a particle of c, i.e. B_12 = 0 for disjoint particle & hole sets (then Sigma(E) = A_12 G(E) A_21 as in the notes) ...
            The B self-energy (A_12 G(E) B_21 + B_12 G(-E) A_21) is neglected ...

        Algorithm for every (J,P) channel ...
            1) list of the 2p-2h configurations, sorted by their pole energy Delta_c ...
            2) coupling matrices A_12 (& B_12) built column by column (threads over configurations), in blocks of Chunk configurations ...
            3) spectral representation Sigma(E)_ij = sum_b W[i,j,b] f(Delta_b - E) + sum_b Wb[i,j,b] f(Delta_b + E) ,   f(x) = 1/x (regularized) ,
               the pole energies are collected on a grid Delta_b with spacing dE_bin (linear splitting between 2 neighbouring bins) ...
               W[:,:,b] = sum_(c in bin) A_12[:,c] (w_c / N_c) A_12[:,c]^T ... matrix products (threads over bins) ...
            4) static approximation ... any energy is then one matrix-vector product of W ...
            Everything is rebuilt from the current occupations, h_N & V_NN in every call (self-consistent RPA loop) ...
            Memory ... N_ph^2 * N_bin (W & Wb) + N_ph * Chunk (A_12 & B_12) Float64 numbers per channel ...

        Regularization of the 2p-2h denominators ...
            "DSRG"      ... f(x) = (1 - exp(-(x/Gamma)^2)) / x ... (default) unchanged for |x| >> Gamma, smooth & bounded (|f| <= 0.64/Gamma) at the poles ...
            "Imaginary" ... f(x) = x / (x^2 + Gamma^2) ... real part of the imaginary level shift ...
            "None"      ... f(x) = 1/x ... singular at the poles (only for Mode = "OnShell" or "Fixed") ...

        Static approximation of A(E) ... Mode =
            "QP"      ... (default) quasiparticle solution of the energy-dependent problem ... in the N-orthonormal basis H(E) = N^-1/2 A(E) N^-1/2 ...
                          1) for every branch k (k-th eigenvalue lambda_k(E) of H(E)) all roots of lambda_k(E) = E are found (scan & Illinois
                             regula falsi, threads over branches) ... with a regularized (bounded & continuous) Sigma at least one root exists ...
                          2) for every branch the root with the largest 1p-1h weight Z = 1/(1 - <u|dH/dE|u>) is selected ...
                          3) static Hermitian matrix with exactly these eigenvalues & the closest orthonormal eigenvectors (symmetric
                             orthogonalization of the root eigenvectors, des Cloizeaux) ... A_eff = N^1/2 X_s E X_s^T N^1/2 ...
            "OnShell" ... dA_{ph,p'h'} = 1/2 * ( Sigma(e_pp - e_hh)_{ph,p'h'} + Sigma(e_p'p' - e_h'h')_{ph,p'h'} ) ...
            "Fixed"   ... dA = Sigma(Omega) ...

        J-coupled form ... |ph;JM> = sum_(m_p,m_h) (-1)^(j_h - m_h) <j_p m_p j_h -m_h|J M> a+_p a_h |0> (same as in RPA_allocate) ...
            <ab;K|V|cd;K> ... non-normalized antisymmetrized J-coupled MEs (O2b_pp, O2b_pn & O2b_nn) ...
            Orbitals are labeled by o = (a,T) with T = -1 (proton) & T = 1 (neutron) ... all isospin combinations are included ...

        Threads ... Julia threads are used for the loops, BLAS runs single-threaded inside them (restored at the end) ...

        Returns A_eff (the input A is overwritten channel by channel) ...
    =#

    # Read calculation parameters ...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1

    Tol_par = Params.Calc.RPA.Tol_par
    Tol_hol = Params.Calc.RPA.Tol_hol

    Tol_ord = 1e-8      # occupation ordering of the 2p-2h configurations (as for the 1p-1h pairs in orbitals_1p1h_make) ...
    Tol_N = 1e-8        # 2p-2h configurations with |N_c| < Tol_N are excluded (pseudo-inverse) ...
    Tol_dup = 1e-8      # QP ... a root eigenvector (numerically) identical to the already selected ones is replaced (fallback) ...
    Tol_root = 1e-10    # QP ... accuracy of the roots in MeV ...

    if !(Mode in ("QP","OnShell","Fixed"))
        throw(ArgumentError("RPA_allocate_SE ... unknown Mode = ''" * Mode * "'' (use ''QP'', ''OnShell'' or ''Fixed'')"))
    end
    if !(Regularization in ("DSRG","Imaginary","None"))
        throw(ArgumentError("RPA_allocate_SE ... unknown Regularization = ''" * Regularization * "'' (use ''DSRG'', ''Imaginary'' or ''None'')"))
    end
    if Mode == "QP" && Regularization == "None"
        throw(ArgumentError("RPA_allocate_SE ... Mode = ''QP'' requires a regularized self-energy (Regularization = ''DSRG'' or ''Imaginary'')"))
    end

    JP_list = JP_initialize(J_max)

    println("\t\tComputing 1p-1h to 2p-2h self-energy coupling of 1p-1h A matrix ...")
    println("\t\t\tMode = " * Mode * (Mode == "Fixed" ? " (Omega = " * string(Omega) * " MeV)" : "") *
            ", Regularization = " * Regularization * ", Gamma = " * string(Gamma) * " MeV, dE_bin = " * string(dE_bin) * " MeV")

    # Particle & hole orbitals o = (a,T) of the intermediate 2p-2h configurations ... same criteria as for the 1p-1h space ...
    #   (the sets may overlap, e.g. Tol_hol < Tol_par ... the roles in a 2p-2h configuration are fixed by the occupation ordering)
    Par = Vector{Tuple{Int64,Int64}}()
    Hol = Vector{Tuple{Int64,Int64}}()

    @inbounds for Orb_a in Orb
        a = Orb_a.a
        if Orb_a.pO < Tol_par
            push!(Par,(a,-1))
        end
        if Orb_a.pO > Tol_hol
            push!(Hol,(a,-1))
        end
        if Orb_a.nO < Tol_par
            push!(Par,(a,1))
        end
        if Orb_a.nO > Tol_hol
            push!(Hol,(a,1))
        end
    end

    # Overlapping particle & hole sets ... only then the backward coupling B_12 can be non-zero ...
    Overlap = !isempty(intersect(Par,Hol))
    if Overlap
        println("\t\t\tOverlapping particle & hole orbitals ... backward 2p-2h coupling B_12 G(-E) B_21 is included in A ...")
    end

    # Define local functions for the self-energy ...
        # Occupation numbers n_a & nb_a = 1 - n_a, angular momentum 2*j_a & orbital momentum l_a of the orbital o = (a,T) ...
    function n(o::Tuple{Int64,Int64})
        a, T = o
        if T == -1
            return Orb[a].pO
        else
            return Orb[a].nO
        end
    end

    nb(o::Tuple{Int64,Int64}) = 1.0 - n(o)
    j(o::Tuple{Int64,Int64}) = Orb[o[1]].j
    l(o::Tuple{Int64,Int64}) = Orb[o[1]].l

        # Kronecker delta of two orbitals ...
    delta(o_1::Tuple{Int64,Int64},o_2::Tuple{Int64,Int64}) = (o_1 == o_2) ? 1.0 : 0.0

        # Generalized mean field e_ab = h_N[a,b] ... non-zero only for equal isospin, l & j ...
    function e(o_1::Tuple{Int64,Int64},o_2::Tuple{Int64,Int64})
        a, T_a = o_1
        b, T_b = o_2
        if (T_a != T_b) || (Orb[a].l != Orb[b].l) || (Orb[a].j != Orb[b].j)
            return 0.0
        end
        if T_a == -1
            return h_N.p[a,b]
        else
            return h_N.n[a,b]
        end
    end

        # J-coupled antisymmetrized ME <o_1 o_2;K|V|o_3 o_4;K> for all isospin combinations (pp, pn, np & nn) ...
    function V(o_1::Tuple{Int64,Int64},o_2::Tuple{Int64,Int64},o_3::Tuple{Int64,Int64},o_4::Tuple{Int64,Int64},K::Int64)
        a, T_a = o_1
        b, T_b = o_2
        c, T_c = o_3
        d, T_d = o_4

        l_a, j_a = Orb[a].l, Orb[a].j
        l_b, j_b = Orb[b].l, Orb[b].j
        l_c, j_c = Orb[c].l, Orb[c].j
        l_d, j_d = Orb[d].l, Orb[d].j

        # Charge, parity & angular momentum conservation ...
        if (T_a + T_b) != (T_c + T_d)
            return 0.0
        end
        if parity(l_a + l_b) != parity(l_c + l_d)
            return 0.0
        end
        if !triangle(j_a,j_b,2*K) || !triangle(j_c,j_d,2*K) || (K + 1) > size(Orb_NN.Map,3)
            return 0.0
        end

        P = parity(l_a + l_b)

        if T_a == T_b
            # Proton-proton & neutron-neutron MEs ... (only pairs inside the N2max truncation are available)
            if Orb_NN.Map[max(a,b),min(a,b),K+1,2] == 0 || Orb_NN.Map[max(c,d),min(c,d),K+1,2] == 0
                return 0.0
            end
            if T_a == -1
                return O2b_pp(a,b,c,d,K,P,V_NN,Orb,Orb_NN)
            else
                return O2b_nn(a,b,c,d,K,P,V_NN,Orb,Orb_NN)
            end
        elseif T_a == -1
            if T_c == -1
                # Proton-neutron ME <pn|V|pn> ...
                if Orb_NN.Map[a,b,K+1,1] == 0 || Orb_NN.Map[c,d,K+1,1] == 0
                    return 0.0
                end
                return O2b_pn(a,b,c,d,K,P,V_NN,Orb_NN)
            else
                # <pn|V|np> = -(-1)^((j_c + j_d)/2 - K) <pn|V|pn> ...
                return -phase(div(j_c + j_d,2) - K) * V(o_1,o_2,o_4,o_3,K)
            end
        else
            # <np|V|..> = -(-1)^((j_a + j_b)/2 - K) <pn|V|..> ...
            return -phase(div(j_a + j_b,2) - K) * V(o_2,o_1,o_3,o_4,K)
        end
    end

        # Occupation factors F^h_(hh1;pp1p2), F^p_(pp1;hh1h2) & G_(hh1;pp1) ...
    function Fh(h,h_1,p,p_1,p_2)
        F = 0.5 * (n(h) - n(p)) * (n(h_1) * nb(p_1) * nb(p_2) + nb(h_1) * n(p_1) * n(p_2))
        F += 0.5 * (n(h) * n(h_1) * nb(p_1) * nb(p_2) - nb(h) * nb(h_1) * n(p_1) * n(p_2))
        return F
    end

    function Fp(p,p_1,h,h_1,h_2)
        F = 0.5 * (n(h) - n(p)) * (nb(p_1) * n(h_1) * n(h_2) + n(p_1) * nb(h_1) * nb(h_2))
        F += 0.5 * (nb(p) * nb(p_1) * n(h_1) * n(h_2) - n(p) * n(p_1) * nb(h_1) * nb(h_2))
        return F
    end

    G(h,h_1,p,p_1) = (n(h) - n(p)) * (n(h_1) - n(p_1))

        # J-coupled vertex alpha^(ph)_(h1;p1p2) = <h1 p;K|alpha|p1 p2;K> ... the hole h is a spectator ...
    function alpha(h,h_1,p,p_1,p_2,K::Int64)
        ME = Fh(h,h_1,p,p_1,p_2) * V(h_1,p,p_1,p_2,K)
        ME += 0.5 * delta(p,p_2) * e(h_1,p_1) * G(h,h_1,p,p_1)
        ME -= 0.5 * phase(div(j(p) + j(p_2),2) - K) * delta(p,p_1) * e(h_1,p_2) * G(h,h_1,p,p_2)
        return ME
    end

        # J-coupled vertex beta^(ph)_(p1;h1h2) = <h p1;K|beta|h1 h2;K> ... the particle p is a spectator ...
    beta(p,p_1,h,h_1,h_2,K::Int64) = Fp(p,p_1,h,h_1,h_2) * V(h,p_1,h_1,h_2,K)

        # Recoupling <(p h^-1)_J| ... |[(p1 p2)_J12 (h1 h2)_J34]_J> of the vertices (all angular momenta doubled except J, J12 & J34) ...
        #   alpha ... spectator hole h = h2, vertex hole j_x = j_h1 ... beta ... spectator particle p = p2, vertex particle j_x = j_p1 ...
    function C_alpha(j_p::Int64,j_h::Int64,j_x::Int64,J::Int64,J12::Int64,J34::Int64)
        Amp = sqrt(hat(2*J12) * hat(2*J34)) * w6j(j_p,j_h,2*J,2*J34,2*J12,j_x)
        return phase(div(j_p + j_h,2) + div(j_p + j_x,2) + J + J12) * Amp
    end

    function C_beta(j_p::Int64,j_h::Int64,j_x::Int64,J::Int64,J12::Int64,J34::Int64)
        Amp = sqrt(hat(2*J12) * hat(2*J34)) * w6j(j_p,j_h,2*J,2*J34,2*J12,j_x)
        return phase(div(j_p + j_h,2) + div(j_p + j_x,2) + div(j_h + j_x,2) + J + J12 + J34) * Amp
    end

        # Regularized f(x) ~ 1/x & its derivative df/dx ...
    function f_reg(x::Float64)
        if Regularization == "DSRG"
            return (x == 0.0) ? 0.0 : -expm1(-(x / Gamma)^2) / x
        elseif Regularization == "Imaginary"
            return x / (x^2 + Gamma^2)
        else
            return (x == 0.0) ? 0.0 : 1.0 / x
        end
    end

    function df_reg(x::Float64)
        if Regularization == "DSRG"
            return (x == 0.0) ? 1.0 / Gamma^2 : (2.0 * (x / Gamma)^2 * exp(-(x / Gamma)^2) + expm1(-(x / Gamma)^2)) / x^2
        elseif Regularization == "Imaginary"
            return (Gamma^2 - x^2) / (x^2 + Gamma^2)^2
        else
            return (x == 0.0) ? 0.0 : -1.0 / x^2
        end
    end

    # Grid of the 2p-2h pole energies Delta_b ... covers all possible Delta_c = e_p1p1 + e_p2p2 - e_h1h1 - e_h2h2 ...
    e_par = [e(o,o) for o in Par]
    e_hol = [e(o,o) for o in Hol]
    Delta_min = 2.0 * minimum(e_par) - 2.0 * maximum(e_hol) - 2.0 * dE_bin
    Delta_max = 2.0 * maximum(e_par) - 2.0 * minimum(e_hol) + 2.0 * dE_bin
    N_bin = ceil(Int64,(Delta_max - Delta_min) / dE_bin) + 2
    Delta_bin = [Delta_min + (b - 1) * dE_bin for b in 1:N_bin]

    # Particle-particle & hole-hole pairs (o_1 <= o_2 in the order of Par & Hol) grouped by (total isospin, parity, J_12) ...
    function pairs_make(Orb_list::Vector{Tuple{Int64,Int64}})
        Pairs = Dict{Tuple{Int64,Int64,Int64},Vector{Tuple{Tuple{Int64,Int64},Tuple{Int64,Int64}}}}()
        @inbounds for i_1 in 1:length(Orb_list), i_2 in i_1:length(Orb_list)
            o_1, o_2 = Orb_list[i_1], Orb_list[i_2]
            @inbounds for K in div(abs(j(o_1) - j(o_2)),2):div(j(o_1) + j(o_2),2)
                # Equal orbitals ... only even K (antisymmetry) ...
                if (o_1 == o_2) && isodd(K)
                    continue
                end
                Key = (o_1[2] + o_2[2],rem(l(o_1) + l(o_2),2),K)
                push!(get!(Pairs,Key,Vector{Tuple{Tuple{Int64,Int64},Tuple{Int64,Int64}}}()),(o_1,o_2))
            end
        end
        return Pairs
    end

    PP_pairs = pairs_make(Par)
    HH_pairs = pairs_make(Hol)

    # J-coupled 2p-2h configurations of the (J,P) channel sorted by Delta_c ... (p1,p2,h1,h2,J12,J34,Delta_c,1/N_c)
    function configurations_make(J::Int64,P::Int64)
        Configurations = Vector{Tuple{Tuple{Int64,Int64},Tuple{Int64,Int64},Tuple{Int64,Int64},Tuple{Int64,Int64},Int64,Int64,Float64,Float64}}()

        @inbounds for ((T_12,L_12,J12),PP_list) in PP_pairs
            @inbounds for ((T_34,L_34,J34),HH_list) in HH_pairs
                if (T_12 != T_34) || (parity(L_12 + L_34) != P) || !triangle(2*J12,2*J34,2*J)
                    continue
                end

                @inbounds for (p_1,p_2) in PP_list, (h_1,h_2) in HH_list
                    # Occupation ordering ... min(n_h1,n_h2) > max(n_p1,n_p2) ...
                    if (min(n(h_1),n(h_2)) - max(n(p_1),n(p_2))) <= Tol_ord
                        continue
                    end

                    N_c = n(h_1) * n(h_2) * nb(p_1) * nb(p_2) - nb(h_1) * nb(h_2) * n(p_1) * n(p_2)
                    if abs(N_c) < Tol_N
                        continue
                    end

                    Delta_c = e(p_1,p_1) + e(p_2,p_2) - e(h_1,h_1) - e(h_2,h_2)

                    push!(Configurations,(p_1,p_2,h_1,h_2,J12,J34,Delta_c,1.0 / N_c))
                end
            end
        end

        sort!(Configurations,by = c -> c[7])

        return Configurations
    end

        # Column A_c = A_12[:,c] (or B_12[:,c] for the reversed pairs) of the configuration c ...
        #   Rows_H[x] ... rows i whose 1p-1h hole is the orbital x (with their particle) ...
        #   Rows_P[x] ... rows i whose 1p-1h particle is the orbital x (with their hole) ...
        #   (for B_12 the particle & hole of every row are exchanged & the row gets (-1)^(j_h - j_p))
    function A12_column!(A_c,Rows_H,Rows_P,Configuration,J::Int64,Reversed::Bool)
        p_1, p_2, h_1, h_2, J12, J34 = Configuration[1], Configuration[2], Configuration[3], Configuration[4], Configuration[5], Configuration[6]

        Norm = 1.0 / sqrt((1.0 + delta(p_1,p_2)) * (1.0 + delta(h_1,h_2)))

        # Hole h2 is the spectator ... alpha^(ph)_(h1;p1p2) ...
        @inbounds for (i,p) in get(Rows_H,h_2,Tuple{Int64,Tuple{Int64,Int64}}[])
            Amp = Norm * (Reversed ? phase(div(j(p) - j(h_2),2)) : 1.0)
            A_c[i] += Amp * C_alpha(j(p),j(h_2),j(h_1),J,J12,J34) * alpha(h_2,h_1,p,p_1,p_2,J12)
        end

        # Hole h1 is the spectator ... - alpha^(ph)_(h2;p1p2) with the exchange of the hole pair ...
        @inbounds for (i,p) in get(Rows_H,h_1,Tuple{Int64,Tuple{Int64,Int64}}[])
            Amp = - Norm * (Reversed ? phase(div(j(p) - j(h_1),2)) : 1.0) * phase(div(j(h_1) + j(h_2),2) - J34)
            A_c[i] += Amp * C_alpha(j(p),j(h_1),j(h_2),J,J12,J34) * alpha(h_1,h_2,p,p_1,p_2,J12)
        end

        # Particle p2 is the spectator ... beta^(ph)_(p1;h1h2) ...
        @inbounds for (i,h) in get(Rows_P,p_2,Tuple{Int64,Tuple{Int64,Int64}}[])
            Amp = Norm * (Reversed ? phase(div(j(p_2) - j(h),2)) : 1.0)
            A_c[i] += Amp * C_beta(j(p_2),j(h),j(p_1),J,J12,J34) * beta(p_2,p_1,h,h_1,h_2,J34)
        end

        # Particle p1 is the spectator ... - beta^(ph)_(p2;h1h2) with the exchange of the particle pair ...
        @inbounds for (i,h) in get(Rows_P,p_1,Tuple{Int64,Tuple{Int64,Int64}}[])
            Amp = - Norm * (Reversed ? phase(div(j(p_1) - j(h),2)) : 1.0) * phase(div(j(p_1) + j(p_2),2) - J12)
            A_c[i] += Amp * C_beta(j(p_1),j(h),j(p_2),J,J12,J34) * beta(p_1,p_2,h,h_1,h_2,J34)
        end

        return
    end

        # Add the configurations of one block to the spectral representation W ... W[:,:,b] += A_12[:,c] (w_c / N_c) A_12[:,c]^T ...
        #   (linear splitting of Delta_c between the bins b_c & b_c + 1 ... threads over the bins, each bin is owned by one thread)
    function W_add!(W::Array{Float64,3},A_12::Matrix{Float64},Bin::Vector{Int64},Weight_lo::Vector{Float64},Weight_hi::Vector{Float64})
        Bin_list = unique(vcat(Bin,Bin .+ 1))
        @inbounds Threads.@threads for b in Bin_list
            # Configurations with b_c = b (weight 1 - t) & b_c = b - 1 (weight t) ... contiguous, the configurations are sorted ...
            Range_lo = searchsortedfirst(Bin,b):searchsortedlast(Bin,b)
            Range_hi = searchsortedfirst(Bin,b - 1):searchsortedlast(Bin,b - 1)

            W_b = view(W,:,:,b)
            if !isempty(Range_lo)
                A_w = A_12[:,Range_lo] .* transpose(Weight_lo[Range_lo])
                mul!(W_b,A_w,transpose(view(A_12,:,Range_lo)),1.0,1.0)
            end
            if !isempty(Range_hi)
                A_w = A_12[:,Range_hi] .* transpose(Weight_hi[Range_hi])
                mul!(W_b,A_w,transpose(view(A_12,:,Range_hi)),1.0,1.0)
            end
        end
        return
    end

        # Self-energy matrix Sigma(E)_ij = sum_b W[i,j,b] f(Delta_b - E) + sum_b Wb[i,j,b] f(Delta_b + E) & its derivative dSigma/dE ...
    function Sigma_E(W::Array{Float64,3},Wb::Array{Float64,3},E::Float64)
        N = size(W,1)
        f_E = [f_reg(Delta - E) for Delta in Delta_bin]
        Sigma = reshape(reshape(W,N^2,N_bin) * f_E,N,N)
        if Overlap
            fb_E = [f_reg(Delta + E) for Delta in Delta_bin]
            Sigma += reshape(reshape(Wb,N^2,N_bin) * fb_E,N,N)
        end
        return Sigma
    end

    function dSigma_E(W::Array{Float64,3},Wb::Array{Float64,3},E::Float64)
        N = size(W,1)
        df_E = [- df_reg(Delta - E) for Delta in Delta_bin]
        dSigma = reshape(reshape(W,N^2,N_bin) * df_E,N,N)
        if Overlap
            dfb_E = [df_reg(Delta + E) for Delta in Delta_bin]
            dSigma += reshape(reshape(Wb,N^2,N_bin) * dfb_E,N,N)
        end
        return dSigma
    end

        # Quasiparticle (QP) static matrix A_eff for one (J,P) channel ... A_s = A^(s), N_d = diagonal of the metric N ...
    function SE_QP(A_s::Matrix{Float64},N_d::Vector{Float64},W::Array{Float64,3},Wb::Array{Float64,3})
        N_ph = size(A_s,1)
        L = 1.0 ./ sqrt.(N_d)                                       # N^-1/2 ...

        H_0 = L .* A_s .* L'
        H(E) = Symmetric(H_0 + L .* Sigma_E(W,Wb,E) .* L')
        dH(E) = L .* dSigma_E(W,Wb,E) .* L'
        g(E,k) = eigvals(H(E))[k] - E

        # 1p-1h weight of a root ... Z = 1/(1 - s) for the slope s = <u|dH/dE|u> <= 0, positive slopes (regularization) are penalized ...
        function weight(u::Vector{Float64},E::Float64)
            s = dot(u,dH(E) * u)
            return (s <= 0.0) ? 1.0 / (1.0 - s) : 1.0 / (1.0 + s)
        end

        # Energy window [E_lo,E_hi] ... every branch satisfies lambda_k(E_lo) > E_lo & lambda_k(E_hi) < E_hi (Sigma is bounded) ...
        e_0 = eigvals(Symmetric(H_0))
        E_lo, E_hi = e_0[1] - Gamma, e_0[N_ph] + Gamma
        while !(all(eigvals(H(E_lo)) .> E_lo))
            E_lo -= 5.0 * Gamma
        end
        while !(all(eigvals(H(E_hi)) .< E_hi))
            E_hi += 5.0 * Gamma
        end

        # Scan of g_k(E) = lambda_k(E) - E on a grid with step Gamma/4 ... (threads over the grid points)
        E_grid = collect(range(E_lo,E_hi,length = ceil(Int64,(E_hi - E_lo) / (0.25 * Gamma)) + 1))
        g_grid = zeros(Float64,N_ph,length(E_grid))
        @inbounds Threads.@threads for i in 1:length(E_grid)
            g_grid[:,i] = eigvals(H(E_grid[i])) .- E_grid[i]
        end

        E_root = zeros(Float64,N_ph)
        Z_root = fill(-1.0,N_ph)
        X = zeros(Float64,N_ph,N_ph)

        # Roots of every branch ... Illinois regula falsi in every sign change, the root with the largest weight Z ... (threads over the branches)
        @inbounds Threads.@threads for k in 1:N_ph
            @inbounds for i in 1:(length(E_grid) - 1)
                E_a, E_b = E_grid[i], E_grid[i+1]
                g_a, g_b = g_grid[k,i], g_grid[k,i+1]
                if (g_a > 0.0) == (g_b > 0.0)
                    continue
                end

                E_m, Side = E_a, 0
                @inbounds for Iteration in 1:200
                    E_m = (E_a * g_b - E_b * g_a) / (g_b - g_a)
                    g_m = g(E_m,k)
                    if (abs(g_m) < Tol_root) || ((E_b - E_a) < Tol_root)
                        break
                    end
                    if (g_m > 0.0) == (g_a > 0.0)
                        E_a, g_a = E_m, g_m
                        if Side == -1
                            g_b *= 0.5
                        end
                        Side = -1
                    else
                        E_b, g_b = E_m, g_m
                        if Side == 1
                            g_a *= 0.5
                        end
                        Side = 1
                    end
                end

                u_k = eigen(H(E_m)).vectors[:,k]
                Z_k = weight(u_k,E_m)

                # Root with the largest 1p-1h weight ...
                if Z_k > Z_root[k]
                    E_root[k], Z_root[k] = E_m, Z_k
                    X[:,k] = u_k
                end
            end
        end

        # Fallback for (numerically) identical root eigenvectors ... direction orthogonal to the selected ones & its own
        # scalar self-consistency E = <v|H(E)|v> (root with the largest weight) ...
        N_fallback = 0
        Q = zeros(Float64,N_ph,0)
        @inbounds for k in sortperm(Z_root,rev = true)
            r = X[:,k] - Q * (Q' * X[:,k])
            if dot(r,r) < Tol_dup
                N_fallback += 1
                if norm(r) < 1e-12
                    r = svd(Q,full = true).U[:,size(Q,2)+1]
                end
                v = r / norm(r)
                q_grid = [dot(v,H(E) * v) - E for E in E_grid]
                Z_best = -1.0
                @inbounds for i in 1:(length(E_grid) - 1)
                    if (q_grid[i] > 0.0) == (q_grid[i+1] > 0.0)
                        continue
                    end
                    E_a, E_b = E_grid[i], E_grid[i+1]
                    @inbounds for Iteration in 1:100
                        E_m = 0.5 * (E_a + E_b)
                        if (dot(v,H(E_m) * v) - E_m > 0.0) == (q_grid[i] > 0.0)
                            E_a = E_m
                        else
                            E_b = E_m
                        end
                        if (E_b - E_a) < Tol_root
                            break
                        end
                    end
                    E_v = 0.5 * (E_a + E_b)
                    Z_v = weight(v,E_v)
                    if Z_v > Z_best
                        Z_best = Z_v
                        E_root[k], Z_root[k] = E_v, Z_v
                    end
                end
                X[:,k] = v
                r = v - Q * (Q' * v)
            end
            Q = hcat(Q,r / norm(r))
        end

        # Symmetric (Lowdin) orthogonalization of the root eigenvectors & the static Hermitian matrix ...
        S_X = Symmetric(X' * X)
        X_s = X * inv(sqrt(S_X))
        H_eff = X_s * Diagonal(E_root) * X_s'
        H_eff = 0.5 * (H_eff + H_eff')

        A_eff = H_eff ./ L ./ L'
        A_eff = 0.5 * (A_eff + A_eff')

        S_X_eig = eigvals(S_X)
        return A_eff, minimum(Z_root), S_X_eig[N_ph] / S_X_eig[1], N_fallback
    end

    # BLAS single-threaded inside the threaded loops ... (restored at the end)
    N_BLAS = BLAS.get_num_threads()
    BLAS.set_num_threads(1)

    # Allocate the self-energy contributions & the static matrices A for all (J,P) channels ...
    Z_min, Cond_max, N_fallback_total, N_configurations_max = 1.0, 1.0, 0, 0

    try
        @inbounds for JP in JP_list
            J, P = JP[1], JP[2]
            N_ph = Orb_1p1h.N[P,J+1]
            Orb_1p1h_JP = Orb_1p1h.Ind[P,J+1]

            if N_ph == 0
                continue
            end

            # Rows of the 1p-1h states by their hole & particle orbitals ... (reversed pairs for B_12: particle h & hole p)
            Rows_H = Dict{Tuple{Int64,Int64},Vector{Tuple{Int64,Tuple{Int64,Int64}}}}()
            Rows_P = Dict{Tuple{Int64,Int64},Vector{Tuple{Int64,Tuple{Int64,Int64}}}}()
            Rows_H_rev = Dict{Tuple{Int64,Int64},Vector{Tuple{Int64,Tuple{Int64,Int64}}}}()
            Rows_P_rev = Dict{Tuple{Int64,Int64},Vector{Tuple{Int64,Tuple{Int64,Int64}}}}()
            @inbounds for i in 1:N_ph
                p, h = (Orb_1p1h_JP[i].a,Orb_1p1h_JP[i].T), (Orb_1p1h_JP[i].b,Orb_1p1h_JP[i].T)
                push!(get!(Rows_H,h,Vector{Tuple{Int64,Tuple{Int64,Int64}}}()),(i,p))
                push!(get!(Rows_P,p,Vector{Tuple{Int64,Tuple{Int64,Int64}}}()),(i,h))
                push!(get!(Rows_H_rev,p,Vector{Tuple{Int64,Tuple{Int64,Int64}}}()),(i,h))
                push!(get!(Rows_P_rev,h,Vector{Tuple{Int64,Tuple{Int64,Int64}}}()),(i,p))
            end

            # 2p-2h configurations sorted by Delta_c, their bins & weights ...
            Configurations = configurations_make(J,P)
            N_configurations = length(Configurations)
            N_configurations_max = max(N_configurations_max,N_configurations)

            Bin = zeros(Int64,N_configurations)
            Weight_lo = zeros(Float64,N_configurations)
            Weight_hi = zeros(Float64,N_configurations)
            @inbounds for c in 1:N_configurations
                x = (Configurations[c][7] - Delta_min) / dE_bin
                Bin[c] = floor(Int64,x) + 1
                t = x - (Bin[c] - 1)
                Weight_lo[c] = (1.0 - t) * Configurations[c][8]
                Weight_hi[c] = t * Configurations[c][8]
            end

            # Spectral representation of Sigma(E) ... forward W_JP & backward Wb_JP part, configurations in blocks of Chunk ...
            W_JP = zeros(Float64,N_ph,N_ph,N_bin)
            Wb_JP = zeros(Float64,N_ph,N_ph,Overlap ? N_bin : 0)

            @inbounds for c_start in 1:Chunk:N_configurations
                Block = c_start:min(c_start + Chunk - 1,N_configurations)

                A_12 = zeros(Float64,N_ph,length(Block))
                B_12 = zeros(Float64,N_ph,Overlap ? length(Block) : 0)

                @inbounds Threads.@threads for c in 1:length(Block)
                    A12_column!(view(A_12,:,c),Rows_H,Rows_P,Configurations[Block[c]],J,false)
                    if Overlap
                        A12_column!(view(B_12,:,c),Rows_H_rev,Rows_P_rev,Configurations[Block[c]],J,true)
                    end
                end

                W_add!(W_JP,A_12,Bin[Block],Weight_lo[Block],Weight_hi[Block])
                if Overlap
                    W_add!(Wb_JP,B_12,Bin[Block],Weight_lo[Block],Weight_hi[Block])
                end
            end

            # Sigma(E) = - sum_c A_12 f(Delta_c - E)/N_c A_21 - ... (sign of G_c(E) = - f(Delta_c - E)/N_c) ...
            W_JP .*= -1.0
            Wb_JP .*= -1.0

            # Static matrix A^(s) ... the input A ...
            A_s = A[P,J+1]

            # Static approximation of A(E) = A^(s) + Sigma(E) ...
            if Mode == "Fixed"
                Sigma_Omega = Sigma_E(W_JP,Wb_JP,Omega)
                A[P,J+1] = A_s + 0.5 * (Sigma_Omega + Sigma_Omega')

            elseif Mode == "OnShell"
                w = [e((o.a,o.T),(o.a,o.T)) - e((o.b,o.T),(o.b,o.T)) for o in Orb_1p1h_JP]
                Sigma_w = Vector{Matrix{Float64}}(undef,N_ph)
                @inbounds Threads.@threads for i in 1:N_ph
                    Sigma_w[i] = Sigma_E(W_JP,Wb_JP,w[i])
                end
                Sigma_on = [0.5 * (Sigma_w[i][i,k] + Sigma_w[k][i,k]) for i in 1:N_ph, k in 1:N_ph]
                A[P,J+1] = A_s + 0.5 * (Sigma_on + Sigma_on')

            elseif Mode == "QP"
                N_d = [n((o.b,o.T)) - n((o.a,o.T)) for o in Orb_1p1h_JP]
                A_eff, Z_JP, Cond_JP, N_fallback_JP = SE_QP(A_s,N_d,W_JP,Wb_JP)
                A[P,J+1] = A_eff
                Z_min, Cond_max, N_fallback_total = min(Z_min,Z_JP), max(Cond_max,Cond_JP), N_fallback_total + N_fallback_JP
            end
        end
    finally
        BLAS.set_num_threads(N_BLAS)
    end

    @printf("\t\t\tLargest channel ... %d J-coupled 2p-2h configurations, %d pole bins\n", N_configurations_max, N_bin)
    if Mode == "QP"
        @printf("\t\t\tQP solution ... min 1p-1h weight Z = %.3f, max condition number of the root eigenvectors = %.2e, fallbacks = %d\n",
                Z_min, Cond_max, N_fallback_total)
    end

    println("\t\t\tFinished on evaluation of 1p-1h to 2p-2h self-energy contributions to the matrix ...")

    return A
end