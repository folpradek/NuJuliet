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

    #=
        # Precompute the 1-body polarization diagrams ... sum_(c,d) <lb|Sigma|cd> <kb|V|cd> + <kb|Sigma|cd> <lb|V|cd> = W2[(k,b),(l,b)] ...
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
    =#

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

                #=
                    # 1-body polarization diagrams ...
                    if T_ph == T_qg
                        Amp_A = phase(1)
                        Amp_B = phase(1 - J + div(j_p + j_h,2))

                        if T_ph == -1
                            # Hole-hole & particle-particle polarization diagrams ...
                            if a_p == a_q
                                ASum += Amp_A * pf[a_h,a_g]
                            end

                            if a_h == a_g
                                ASum += Amp_A * pf[a_p,a_q]
                            end

                            # Hole-particle & particle-hole polarization diagrams ...
                            if a_p == a_g
                                BSum += Amp_B * pf[a_h,a_q]
                            end

                            if a_h == a_q
                                BSum += Amp_B * pf[a_p,a_g]
                            end

                        elseif T_ph == 1
                            # Hole-hole & particle-particle polarization diagrams ...
                            if a_p == a_q
                                ASum += Amp_A * nf[a_h,a_g]
                            end

                            if a_h == a_g
                                ASum += Amp_A * nf[a_p,a_q]
                            end

                            # Hole-particle & particle-hole polarization diagrams ...
                            if a_p == a_g
                                BSum += Amp_B * nf[a_h,a_q]
                            end

                            if a_h == a_q
                                BSum += Amp_B * nf[a_p,a_g]
                            end
                        end
                    end
                =#

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