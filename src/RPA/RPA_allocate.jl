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
            #S_A = RPA_allocate_HF_S(Params,Orb,Orb_1p1h,Orb_NN,h_N,V_NN;gamma = 5.0)
            #A .+= S_A
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

    # Define local functions for the matrix elements ...
    function O2b_ppnn_VS(a::Int64,b::Int64,c::Int64,d::Int64,J::Int64,P::Int64,V::O2B,S::O2B,Orb::Vector{Orb1B},Orb_NN::Orb2B)
        # Same as O2b_pp & O2b_nn (see O2b.jl), but V_pp, S_pp, V_nn & S_nn are read with a single index lookup ...
        if a < b
            a_s, b_s = b, a
            exp_ab = J + 1 + div(Orb[a].j + Orb[b].j, 2)
            Sign_ab = isodd(exp_ab) ? -1.0 : 1.0
        else
            a_s, b_s = a, b
            Sign_ab = 1.0
        end
        if c < d
            c_s, d_s = d, c
            exp_cd = J + 1 + div(Orb[c].j + Orb[d].j, 2)
            Sign_cd = isodd(exp_cd) ? -1.0 : 1.0
        else
            c_s, d_s = c, d
            Sign_cd = 1.0
        end
        Amp = Sign_ab * Sign_cd
        if (a < b) && (c < d)
            exp = div(Orb[a].j + Orb[b].j + Orb[c].j + Orb[d].j, 2)
            Amp = isodd(exp) ? -1.0 : 1.0
        end
        @inbounds Ind = O2b_index(a_s,b_s,c_s,d_s,J,P,1,Orb_NN)
        @inbounds return Amp * V.pp[P,J+1][Ind], Amp * S.pp[P,J+1][Ind], Amp * V.nn[P,J+1][Ind], Amp * S.nn[P,J+1][Ind]
    end

    function O2b_pn_VS(a::Int64,b::Int64,c::Int64,d::Int64,J::Int64,P::Int64,V::O2B,S::O2B,Orb_NN::Orb2B)
        # Same as O2b_pn (see O2b.jl), but V_pn & S_pn are read with a single index lookup ...
        @inbounds Ind = O2b_index(a,b,c,d,J,P,0,Orb_NN)
        @inbounds return V.pn[P,J+1][Ind], S.pn[P,J+1][Ind]
    end

    println("\t\tComputing 2-body correlation function contributions to the RPA matrices A & B ...")

    # Performance notes ...
        # A & B below read the box diagrams W1 & W2 only as W[Bra,Ket] with Ket = (h,x), i.e., the first orbital of Ket is a hole ...
        # W1 & W2 are symmetric, hence only the columns Ket = (h,x) are evaluated, which reduces the O(N^3) products to O(N^2 N_h) ...
        # the remaining entries of W1_NN & W2_NN are never touched (zero), so they also do not occupy physical memory ...

    # List of hole orbitals present in the 1p-1h basis ...
    Is_hole = falses(a_max)
    @inbounds for Orb_1p1h_JP in Orb_1p1h.Ind
        @inbounds for Orb_ph in Orb_1p1h_JP
            Is_hole[Orb_ph.b] = true
        end
    end

    # Precompute box diagram 1 ...
    W1_NN, Orb_NN_temp = O2b_temp_initialize(Params,Orb,Make_Orb_NN = true)

    @inbounds for JP in JP_list
        J, P = JP[1], JP[2]
        N = Orb_NN_temp.N[P,J+1]
        Orb_NN_JP = Orb_NN_temp.Ind[P,J+1]

        # NN orbitals (k,l) with hole k & phases (-1)^((j_k + j_l)/2) ...
        Kets = [Ket for Ket in 1:N if Is_hole[Orb_NN_JP[Ket][1]]]
        Phase = [phase(div(Orb[Orb_NN_JP[Bra][1]].j + Orb[Orb_NN_JP[Bra][2]].j,2)) for Bra in 1:N]

        if N == 0 || isempty(Kets)
            continue
        end

        # Precompute Amp_J_1 = (2J_1 + 1) {j_k j_l J; j_d j_b J_1} for all j's as SixJ[J_1+1,i_b,i_d,i_l,i_k] with i = (j + 1)/2 ...
        SixJ = zeros(Float64,J_max+1,N_max+1,N_max+1,N_max+1,N_max+1)

        @inbounds Threads.@threads for i_kl in 1:(N_max+1)^2
            i_k, i_l = div(i_kl - 1,N_max + 1) + 1, rem(i_kl - 1,N_max + 1) + 1
            j_k, j_l = 2*i_k - 1, 2*i_l - 1

            if triangle(j_k,j_l,2*J) == false
                continue
            end

            @inbounds for i_b in 1:(N_max+1), i_d in 1:(N_max+1)
                j_b, j_d = 2*i_b - 1, 2*i_d - 1
                @inbounds for J_1 in div(max(abs(j_k - j_b), abs(j_l - j_d)),2):div(min(j_k + j_b, j_l + j_d),2)
                    SixJ[J_1+1,i_b,i_d,i_l,i_k] = hat(2*J_1) * w6j(j_k,j_l,2*J,j_d,j_b,2*J_1)
                end
            end
        end

        # Cross-coupled MEs X[Bra,Col] = sum_J_1 (2J_1 + 1) {j_k j_l J; j_d j_b J_1} <kb;J_1|O|ld;J_1> ... Bra = (k,l) & Col = (b,d) ...
            # pp, pn & nn ... <kb|O|ld> with phase (-1)^J_1 ...
            # pp & nn ... X[Col,Bra] = (-1)^((j_k + j_l + j_b + j_d)/2) X[Bra,Col], so only Col <= Bra is evaluated ...
        ppXV, ppXS = zeros(Float64,N,N), zeros(Float64,N,N)
        pnXV, pnXS = zeros(Float64,N,N), zeros(Float64,N,N)
        nnXV, nnXS = zeros(Float64,N,N), zeros(Float64,N,N)
            # Mixed pn ... <kb|O|dl> without phase ...
        mpnXV, mpnXS = zeros(Float64,N,N), zeros(Float64,N,N)

        # Each Bra is spawned as a separate task for dynamic load balancing ...
        @sync for Bra in 1:N
            Threads.@spawn @inbounds begin
                a_k = Orb_NN_JP[Bra][1]
                a_l = Orb_NN_JP[Bra][2]

                l_k, j_k = Orb[a_k].l, Orb[a_k].j
                l_l, j_l = Orb[a_l].l, Orb[a_l].j

                i_k, i_l = div(j_k + 1,2), div(j_l + 1,2)

                @inbounds for Col in 1:N
                    a_b = Orb_NN_JP[Col][1]
                    a_d = Orb_NN_JP[Col][2]

                    l_b, j_b = Orb[a_b].l, Orb[a_b].j
                    l_d, j_d = Orb[a_d].l, Orb[a_d].j

                    P_kb = parity(l_k + l_b)

                    Lower = Col <= Bra

                    ppV, ppS = 0.0, 0.0
                    pnV, pnS = 0.0, 0.0
                    nnV, nnS = 0.0, 0.0
                    mpnV, mpnS = 0.0, 0.0

                    i_b, i_d = div(j_b + 1,2), div(j_d + 1,2)

                    J_1_min, J_1_max = div(max(abs(j_k - j_b), abs(j_l - j_d)),2), div(min(j_k + j_b, j_l + j_d),2)
                    @inbounds for J_1 in J_1_min:J_1_max
                        Amp_J_1 = SixJ[J_1+1,i_b,i_d,i_l,i_k]
                        Amp = phase(J_1) * Amp_J_1

                        if Lower
                            ppV_J_1, ppS_J_1, nnV_J_1, nnS_J_1 = O2b_ppnn_VS(a_k,a_b,a_l,a_d,J_1,P_kb,V_NN,Sigma,Orb,Orb_NN)
                            ppV += Amp * ppV_J_1
                            ppS += Amp * ppS_J_1

                            nnV += Amp * nnV_J_1
                            nnS += Amp * nnS_J_1
                        end

                        pnV_J_1, pnS_J_1 = O2b_pn_VS(a_k,a_b,a_l,a_d,J_1,P_kb,V_NN,Sigma,Orb_NN)
                        pnV += Amp * pnV_J_1
                        pnS += Amp * pnS_J_1

                        mpnV_J_1, mpnS_J_1 = O2b_pn_VS(a_k,a_b,a_d,a_l,J_1,P_kb,V_NN,Sigma,Orb_NN)
                        mpnV += Amp_J_1 * mpnV_J_1
                        mpnS += Amp_J_1 * mpnS_J_1
                    end

                    if Lower
                        Phase_kldb = Phase[Bra] * Phase[Col]
                        ppXV[Bra,Col], ppXS[Bra,Col] = ppV, ppS
                        nnXV[Bra,Col], nnXS[Bra,Col] = nnV, nnS
                        ppXV[Col,Bra], ppXS[Col,Bra] = Phase_kldb * ppV, Phase_kldb * ppS
                        nnXV[Col,Bra], nnXS[Col,Bra] = Phase_kldb * nnV, Phase_kldb * nnS
                    end

                    pnXV[Bra,Col], pnXS[Bra,Col] = pnV, pnS
                    mpnXV[Bra,Col], mpnXS[Bra,Col] = mpnV, mpnS
                end
            end
        end

        # np ... <bk|O|dl> with phase (-1)^J_1 * (-1)^((j_k + j_l)/2) ... X_np[Bra,Col] = (-1)^((j_k + j_l)/2) X_pn[Col,Bra] ...
        npXV, npXS = Phase .* transpose(pnXV), Phase .* transpose(pnXS)

        # W1[Bra,Ket] = sum_Col X_V[Bra,Col] * X_S[Ket,Col] + X_S[Bra,Col] * X_V[Ket,Col] ... evaluated only for Ket = (h,x) ...
        W1 = zeros(Float64,N,length(Kets))
            # Proton-proton & proton-neutron part ...
        mul!(W1,ppXV,transpose(ppXS[Kets,:]))
        mul!(W1,ppXS,transpose(ppXV[Kets,:]),1.0,1.0)
        mul!(W1,pnXV,transpose(pnXS[Kets,:]),1.0,1.0)
        mul!(W1,pnXS,transpose(pnXV[Kets,:]),1.0,1.0)
        W1_NN.pp[P,J+1][:,Kets] .= W1
            # Proton-neutron part ...
        mul!(W1,mpnXV,transpose(mpnXS[Kets,:]))
        mul!(W1,mpnXS,transpose(mpnXV[Kets,:]),1.0,1.0)
        W1_NN.pn[P,J+1][:,Kets] .= W1
            # Neutron-neutron & neutron-proton part ...
        mul!(W1,nnXV,transpose(nnXS[Kets,:]))
        mul!(W1,nnXS,transpose(nnXV[Kets,:]),1.0,1.0)
        mul!(W1,npXV,transpose(npXS[Kets,:]),1.0,1.0)
        mul!(W1,npXS,transpose(npXV[Kets,:]),1.0,1.0)
        W1_NN.nn[P,J+1][:,Kets] .= W1
    end

    # Precompute box diagram 2 ...
    W2_NN = O2b_temp_initialize(Params,Orb)

    # 1-body polarization potential (DEV) ... accumulated channel by channel in the loop below ...
    pf, nf = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)
    List_kl = [(Orb_a,Orb_b) for Orb_a in Orb for Orb_b in Orb if Orb_a.a <= Orb_b.a && Orb_a.j == Orb_b.j]

    @inbounds for JP in JP_list
        J, P = JP[1], JP[2]
        N = Orb_NN_temp.N[P,J+1]
        Orb_NN_JP = Orb_NN_temp.Ind[P,J+1]

        # NN orbitals (k,l) with hole k ...
        Kets = [Ket for Ket in 1:N if Is_hole[Orb_NN_JP[Ket][1]]]

        if N == 0 || (isempty(Kets) && !Params.Calc.RPA.DEV)
            continue
        end

        # Direct J-coupled MEs O[Bra,Ket] = <kl;J|O|mn;J> ... Bra = (k,l) & Ket = (m,n) ...
        ppV, ppS = zeros(Float64,N,N), zeros(Float64,N,N)
        pnV, pnS = zeros(Float64,N,N), zeros(Float64,N,N)
        nnV, nnS = zeros(Float64,N,N), zeros(Float64,N,N)

        # Each Bra is spawned as a separate task for dynamic load balancing (triangular loop) ...
        @sync for Bra in 1:N
            Threads.@spawn @inbounds begin
                a_k = Orb_NN_JP[Bra][1]
                a_l = Orb_NN_JP[Bra][2]

                # O is symmetric, O[Ket,Bra] = O[Bra,Ket], hence only Ket <= Bra is evaluated ...
                @inbounds for Ket in 1:Bra
                    a_m = Orb_NN_JP[Ket][1]
                    a_n = Orb_NN_JP[Ket][2]

                    ppV[Bra,Ket], ppS[Bra,Ket], nnV[Bra,Ket], nnS[Bra,Ket] = O2b_ppnn_VS(a_k,a_l,a_m,a_n,J,P,V_NN,Sigma,Orb,Orb_NN)

                    pnV[Bra,Ket], pnS[Bra,Ket] = O2b_pn_VS(a_k,a_l,a_m,a_n,J,P,V_NN,Sigma,Orb_NN)

                    ppV[Ket,Bra], ppS[Ket,Bra] = ppV[Bra,Ket], ppS[Bra,Ket]
                    pnV[Ket,Bra], pnS[Ket,Bra] = pnV[Bra,Ket], pnS[Bra,Ket]
                    nnV[Ket,Bra], nnS[Ket,Bra] = nnV[Bra,Ket], nnS[Bra,Ket]
                end
            end
        end

        # W2[Bra,Ket] = sum_(c,d) <kl|V|cd> <mn|Sigma|cd> + <kl|Sigma|cd> <mn|V|cd> ... evaluated only for Ket = (h,x) ...
        if !isempty(Kets)
            W2 = zeros(Float64,N,length(Kets))
            mul!(W2,ppV,transpose(ppS[Kets,:]))
            mul!(W2,ppS,transpose(ppV[Kets,:]),1.0,1.0)
            W2_NN.pp[P,J+1][:,Kets] .= W2
            mul!(W2,pnV,transpose(pnS[Kets,:]))
            mul!(W2,pnS,transpose(pnV[Kets,:]),1.0,1.0)
            W2_NN.pn[P,J+1][:,Kets] .= W2
            mul!(W2,nnV,transpose(nnS[Kets,:]))
            mul!(W2,nnS,transpose(nnV[Kets,:]),1.0,1.0)
            W2_NN.nn[P,J+1][:,Kets] .= W2
        end

        ###=
        if Params.Calc.RPA.DEV
            # 1-body polarization diagrams (contact terms O(V sigma) of A, cf. RPA_allocate_SE) ... sum_(c,d) <lb|Sigma|cd> <kb|V|cd> + <kb|Sigma|cd> <lb|V|cd> = W2[(k,b),(l,b)] ...
                # W2[(k,b),(l,b)] is needed for all k,l, hence it is evaluated directly as a column dot product (O is symmetric, O[Bra,:] = O[:,Bra]) ...
            I = J

            # Calculate 1-body polarization potential ... contribution of the channel I = J ...
            @inbounds Threads.@threads for Orb_kl in List_kl
                (Orb_k,Orb_l) = Orb_kl
                a_k, l_k, j_k = Orb_k.a, Orb_k.l, Orb_k.j
                a_l, l_l, j_l = Orb_l.a, Orb_l.l, Orb_l.j

                pfSum, nfSum = 0.0, 0.0

                @inbounds for Orb_b in Orb
                    a_b, l_b, j_b = Orb_b.a, Orb_b.l, Orb_b.j

                    P_kb = parity(l_k + l_b)

                    if P_kb != P || P_kb != parity(l_l + l_b) || triangle(j_k,j_b,2*I) == false
                        continue
                    end

                    Amp = 0.5 * hat(2*I) / hat(j_k)

                    kb = O2b_temp_index(a_k,a_b,I,P_kb,Orb_NN_temp)
                    lb = O2b_temp_index(a_l,a_b,I,P_kb,Orb_NN_temp)
                    bk = O2b_temp_index(a_b,a_k,I,P_kb,Orb_NN_temp)
                    bl = O2b_temp_index(a_b,a_l,I,P_kb,Orb_NN_temp)

                    # Proton-proton polarization potential ...
                    ppME = 0.5 * Amp * @views (dot(ppV[:,kb],ppS[:,lb]) + dot(ppS[:,kb],ppV[:,lb]))

                    pnME = Amp * @views (dot(pnV[:,kb],pnS[:,lb]) + dot(pnS[:,kb],pnV[:,lb]))

                    pfSum += ppME + pnME

                    # Neutron-neutron polarization potential ...
                    nnME = 0.5 * Amp * @views (dot(nnV[:,kb],nnS[:,lb]) + dot(nnS[:,kb],nnV[:,lb]))

                    npME = phase(1 + div(j_k + j_l,2)) * Amp * @views (dot(pnV[:,bk],pnS[:,bl]) + dot(pnS[:,bk],pnV[:,bl]))

                    nfSum += nnME + npME
                end

                pf[a_k,a_l] += pfSum
                nf[a_k,a_l] += nfSum
            end
        end
        ##=#
    end

    if Params.Calc.RPA.DEV
        # Symmetrize the 1-body polarization potential ...
        @inbounds for (Orb_k,Orb_l) in List_kl
            a_k, a_l = Orb_k.a, Orb_l.a
            pf[a_l,a_k] = pf[a_k,a_l]
            nf[a_l,a_k] = nf[a_k,a_l]
        end
    end

    # Allocate 2-body correlation function contributions to the RPA matrices A & B ...
    @inbounds for JP in JP_list
        J, P = JP[1], JP[2]
        N_ph = Orb_1p1h.N[P,J+1]
        Orb_1p1h_JP = Orb_1p1h.Ind[P,J+1]

        A_JP = zeros(Float64,N_ph,N_ph)
        B_JP = zeros(Float64,N_ph,N_ph)

        # Each row i_ph is spawned as a separate task for dynamic load balancing (triangular loop) ...
        @sync for i_ph in 1:N_ph
            Threads.@spawn @inbounds begin
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
                            # W2 is symmetric ... W2[(g,p),(q,h)] = W2[(q,h),(g,p)] with the hole g in Ket ...
                            ASum += phase(div(j_p + j_h + j_q + j_g,2)) * Amp * O2b_temp(a_q,a_h,a_g,a_p,I,P_pg,0,W2_NN,Orb_NN_temp)

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

function RPA_allocate_corr_SE(Params::Parameters,w::Union{Float64,AbstractVector{Float64}},Orb::Vector{Orb1B},Orb_1p1h::Orb2B_1p1h,Orb_NN::Orb2B,
                              h_N::O1B,V_NN::O2B,Sigma::Union{O2B,Nothing};
                              HF::Bool = false,Resolvent::String = "Diagonal",Metric_12::Bool = true,Eta::Float64 = 0.0,Tol_N::Float64 = 1e-8,
                              Return_blocks::Bool = false,Quantity::String = "Sigma")
    #=
        Energy-dependent 1p-1h to 2p-2h self-energy of the TDA matrix A for a general correlated reference (NAT basis) ...
            Sigma(w) = (A_12 - w N_12) (w N_22 - A_22)^-1 (A_21 - w N_21) ,
            i.e. the exact downfolding of the 2p-2h block of the full SRPA-TDA problem [A_11 A_12; A_21 A_22] = w [N_11 N_12; N_21 N_22]
            onto the 1p-1h space ... A_11 + Sigma(w) is to be used with the metric N_11 (the A & N of RPA_allocate) ...

        Input ... notes A12_correlated_SRPA, A22_correlated_SRPA & SRPA_metric_correlated (Sec. 3, three- & four-body cumulants neglected) ...
            Natural orbitals ... rho_ab = n_a d_ab (n_a = Orb[a].pO/nO), e_ab = h_N (generalized mean field, full matrix in every (T,l,j) block),
            sigma = Sigma (2-body cumulant, J-coupled like V_NN) ...

        Leading order ... rho = O(1), h & e = O(h), V & sigma = O(V) ... only the terms O(V), O(h) & O(h V) are kept (in N_22 & N_12 the terms O(1)
        & O(V)), i.e. sigma V, sigma sigma V, rho sigma V & rho rho sigma V are dropped ... In the NAT basis the remaining terms take the form
            (all orbitals of one 2p-2h configuration are different, min(n_h1,n_h2) > max(n_p1,n_p2))
            N_12 = - P_p1p2[d_pp2 s_hp1h1h2] - P_h1h2[d_hh2 s_h1pp1p2] ,
            A_12 = P_h1h2[d_hh2 (Fa v_h1pp1p2 + Wa_h1pp1p2) + e_hh2 s_h1pp1p2] + P_p1p2[d_pp2 (Fb v_hp1h1h2 + Wb_hp1h1h2) - e_pp2 s_hp1h1h2]
                   + 1/2 P_p1p2 P_h1h2[d_hh2 d_pp2 e_h1p1 (n_h - n_p)(n_h1 - n_p1)] ,
                Fa, Fb ... occupation factors (Fh & Fp below, the same as in RPA_allocate_SE) ,
                Wa = 1/2 (e_h1a s_apP - e_pa s_h1aP - s_h1pap2 e_ap1 - s_h1pp1a e_ap2) , Wb = 1/2 (e_ha s_ap1H - e_p1a s_haH + s_hp1ah2 e_ah1 + s_hp1h1a e_ah2) ,
            N_22 = A^p A^h N_c + (1 - n_p1 - n_p2) A^p s_h1h2h3h4 - (1 - n_h1 - n_h2) A^h s_p1p2p3p4
                   + P_p1p2 P_h1h2 P_p3p4 P_h3h4[(n_p2 - n_h2) d_p2p4 d_h2h4 s_h1p3h3p1] ,   N_c = n_h1 n_h2 nb_p1 nb_p2 - nb_h1 nb_h2 n_p1 n_p2 ,
            A_22 = 1/2 {D,N_22} + V_22 ,
                D ... 1-body operator e acting on the 4 legs (+ for particles, - for holes) of both configurations, the e part of A_22 is
                      the double commutator of the 1-body operator sum_ab e_ab a+_a a_b, i.e. 1/2 (N_22(D c,c') + N_22(c,D c')) with N_22 for
                      arbitrary labels ... it generates (i) e on one leg, (ii) e-dressed sigma, (iii) the contraction terms
                      1/2 S P_p1p2 P_h1h2 P_p3p4[d_p2p4 (n_p1 - n_h1) e_h1p1 s_h2p3h3h4] + 1/2 S P_p1p2 P_h1h2 P_h3h4[d_h2h4 (n_p1 - n_h1) e_h1p1 s_h3p2p3p4]
                      (all of them listed term by term in Sec. 3 of A22_correlated_SRPA) ...
                V_22 = c_pp A^h v_p1p2p3p4 + c_hh A^p v_h1h2h3h4 + P_p1p2 P_h1h2 P_p3p4 P_h3h4[f_ring d_p2p4 d_h2h4 v_h1p3h3p1] ,
                      c_pp, c_hh & f_ring ... occupation polynomials collected from all rho-only v terms of Sec. 3 (see below) ...
            HF limit (n = 0,1 & sigma = 0) ... the expressions of Sec. 4 of the notes ...
        No B type (K matrix) couplings are included (TDA) ...

        2p-2h configurations ... J-coupled & normalized, c = [(p1 p2)_J12 (h1 h2)_J34]_J with p1 <= p2 & h1 <= h2,
            O+_c = sum <J12 M12 J34 M34|J M> <j_p1 m_1 j_p2 m_2|J12 M12> <j_h1 mu_1 j_h2 mu_2|J34 M34> a+_p1 a+_p2 at_h2 at_h1 ,
            at_(j mu) = (-1)^(j + mu) a_(j,-mu) (the same convention as the 1p-1h pairs of RPA_allocate) ...
            p1, p2 (h1, h2) from the particle (hole) orbitals (Tol_par, Tol_hol) with min(n_h1,n_h2) > max(n_p1,n_p2) & N_c > Tol_N ...
        J-coupled building blocks (all angular momenta doubled in the code except J, J12, J34, K & L) ...
            ladders     ... pair MEs <p1 p2;J12|O|p3 p4;J12> & <h1 h2;J34|O|h3 h4;J34> (O = V_NN, Sigma or e-dressed Sigma) ...
            ring        ... 9j recoupling [(p1 p2)J12 (h1 h2)J34]J -> [(p1 h1)K (p2 h2)K']J of both configurations (spectators p2 h2 in K') &
                            the particle-hole (Pandya) MEs X^ph_K(p1 h1;p3 h3) = (-1)^(j_p1 + j_h1 + j_p3 + j_h3) sum_L (2L + 1) {j_p1 j_h1 K; j_p3 j_h3 L} X^L(h1 p3;h3 p1) ...
            contraction ... (p1 h1) coupled to K = 0 by e_h1p1, the remaining 3 legs of the other configuration are recoupled by a 6j symbol ...
            A_12        ... vertices alpha (spectator hole) & beta (spectator particle) recoupled by a single 6j symbol (as in RPA_allocate_SE) ...

        Resolvent G(w) = (w N_22 - A_22)^-1 ... Resolvent =
            "Diagonal" ... (default) strictly single-particle resolvent, no residual interaction (V_22) & no sigma in A_22 & N_22,
                           only the diagonal e_aa with the fractional-occupation quenching N_c of the metric ...
                           N_22 = N_c d_cc' , A_22 = N_c Delta_c d_cc' , Delta_c = e_p1p1 + e_p2p2 - e_h1h1 - e_h2h2 ,
                           G_c(w) = 1 / (N_c (w - Delta_c)) ... (A_12 & N_12 are kept in full, incl. their sigma terms) ...
                           cheap ... no 2p-2h matrices, no diagonalization, Sigma(w) = (A_12 - w N_12) diag(f(w - Delta_c)/N_c) (A_12 - w N_12)^T ,
            "Full"     ... A_22 & N_22 of the leading order given above ...

        Evaluation (modular in w), Resolvent = "Full" ... for every (J,P) channel A_12, N_12, A_22 & N_22 are built once, then
            1) N_22 = X^-T X^-1 on the subspace of N_22 eigenvalues > Tol_N (after diagonal scaling) ... X^T N_22 X = 1
               (X = N_22^-1/2 directly if N_22 is diagonal, i.e. HF or sigma = 0) ,
            2) X^T A_22 X = Z Omega Z^T ... U = X Z (U^T N_22 U = 1, U^T A_22 U = Omega) ,
            3) F_A = A_12 U, F_N = N_12 U ... Sigma(w) = (F_A - w F_N) diag(f(w - Omega)) (F_A - w F_N)^T for every requested w ,
               f(x) = 1/x (Eta = 0) or x/(x^2 + Eta^2) (real part of the imaginary shift Eta) ...
            The 2p-2h matrices are dense (memory ~ 2 N_2p2h^2 Float64 per channel), the elements are computed with Julia threads ...

        Verified (all blocks & Sigma(w), proton & mixed isospin, HF & fractional occupations) to ~1e-14 against
            (a) the brute-force Fock-space double commutators in a quasi-free ensemble state (rho-only terms exact) &
            (b) the m-scheme transcription of the sigma terms of the notes (Sec. 3, incl. 1/2 {D,N_22} = the e sigma terms of A_22) ...

        Arguments ...
            w            ... energy (Float64) or a vector of energies in MeV ,
            Sigma        ... 2-body cumulant (O2B) or nothing (sigma = 0) ,
            HF           ... true = HF limit (occupations rounded to 0 & 1, sigma = 0) ,
            Resolvent    ... "Diagonal" (single-particle resolvent, default) or "Full" (A_22 & N_22 with V & sigma) ,
            Metric_12    ... include the metric coupling N_12 (A_12 -> A_12 - w N_12) ,
            Eta          ... regularization of the 2p-2h poles ,
            Tol_N        ... threshold for the (scaled) eigenvalues of N_22 ,
            Return_blocks ... true = return also Dict((J,P) => (Configurations, A_12, N_12, A_22, N_22)) for inspection & tests ...
            Quantity     ... "Sigma" (self-energy, default) or "Norm" (energy-dependent norm correction dN(w), see RPA_allocate_corr_NE) ...

        Returns ... Sigma[P,J+1] (N_ph x N_ph in the 1p-1h basis Orb_1p1h) for a single w, a vector of these for a vector of w
                    (& the blocks if Return_blocks) ...
    =#

    # Read calculation parameters ...
    N_max = Params.Calc.Nmax
    J_max = 2*N_max + 1
    a_max = length(Orb)

    Tol_par = Params.Calc.RPA.Tol_par
    Tol_hol = Params.Calc.RPA.Tol_hol
    Tol_ord = 1e-8      # occupation ordering of the 2p-2h configurations (as in RPA_allocate_SE) ...

    Correlated = !HF && !isnothing(Sigma)      # sigma terms on/off ...
    Coupling_N = Correlated && Metric_12       # N_12 on/off ...
    if !(Quantity in ("Sigma","Norm"))
        throw(ArgumentError("RPA_allocate_corr_SE ... unknown Quantity = ''" * Quantity * "'' (use ''Sigma'' or ''Norm'')"))
    end
    if !(Resolvent in ("Diagonal","Full"))
        throw(ArgumentError("RPA_allocate_corr_SE ... unknown Resolvent = ''" * Resolvent * "'' (use ''Diagonal'' or ''Full'')"))
    end
    Energies = (w isa Float64) ? [w] : collect(Float64,w)

    JP_list = JP_initialize(J_max)

    println((Quantity == "Sigma") ? "\t\tComputing 1p-1h to 2p-2h self-energy Sigma(w) of the TDA matrix A ..." :
                                    "\t\tComputing 1p-1h to 2p-2h energy-dependent correction dN(w) of the TDA norm matrix N ...")
    println("\t\t\tReference = " * (HF ? "HF" : (Correlated ? "correlated (rho & sigma)" : "correlated (rho only, sigma = 0)")) *
            ", resolvent = " * Resolvent * ", N_12 = " * string(Coupling_N) * ", Eta = " * string(Eta) * " MeV, " * string(length(Energies)) * " energies")

    # 1) Orbitals ... o = a (protons) & o = a + a_max (neutrons), with n_o, j_o (doubled), l_o & T_o ...
    N_o = 2*a_max
    a_o = [(o <= a_max) ? o : o - a_max for o in 1:N_o]
    T_o = [(o <= a_max) ? -1 : 1 for o in 1:N_o]
    j_o = [Orb[a_o[o]].j for o in 1:N_o]
    l_o = [Orb[a_o[o]].l for o in 1:N_o]
    n_raw = [(T_o[o] == -1) ? Orb[a_o[o]].pO : Orb[a_o[o]].nO for o in 1:N_o]
    n_o = HF ? [(x >= 0.5) ? 1.0 : 0.0 for x in n_raw] : n_raw

    orbital(a::Int64,T::Int64) = (T == -1) ? a : a + a_max

        # Generalized mean field e = h_N (symmetrized) & the (T,l,j) blocks of every orbital (e connects only orbitals of one block) ...
    E = zeros(Float64,N_o,N_o)
    Blk = [Int64[] for o in 1:N_o]
    @inbounds for o_1 in 1:N_o, o_2 in 1:N_o
        if (T_o[o_1] == T_o[o_2]) && (l_o[o_1] == l_o[o_2]) && (j_o[o_1] == j_o[o_2])
            push!(Blk[o_1],o_2)
            h = (T_o[o_1] == -1) ? h_N.p : h_N.n
            E[o_1,o_2] = 0.5 * (h[a_o[o_1],a_o[o_2]] + h[a_o[o_2],a_o[o_1]])
        end
    end

        # Particle & hole orbitals of the 2p-2h configurations ...
    Par = HF ? [o for o in 1:N_o if n_o[o] == 0.0] : [o for o in 1:N_o if n_o[o] < Tol_par]
    Hol = HF ? [o for o in 1:N_o if n_o[o] == 1.0] : [o for o in 1:N_o if n_o[o] > Tol_hol]

    # 2) Matrix elements ...
        # J-coupled antisymmetrized ME <o1 o2;K|O|o3 o4;K> of a 2-body operator O (V_NN or Sigma) for all isospin combinations ...
    function ME(O::O2B,o_1::Int64,o_2::Int64,o_3::Int64,o_4::Int64,K::Int64)::Float64
        T_1, T_2, T_3, T_4 = T_o[o_1], T_o[o_2], T_o[o_3], T_o[o_4]
        if (T_1 + T_2) != (T_3 + T_4)
            return 0.0
        end
        if parity(l_o[o_1] + l_o[o_2]) != parity(l_o[o_3] + l_o[o_4])
            return 0.0
        end
        if !triangle(j_o[o_1],j_o[o_2],2*K) || !triangle(j_o[o_3],j_o[o_4],2*K) || (K + 1) > size(Orb_NN.Map,3)
            return 0.0
        end
        if ((o_1 == o_2) || (o_3 == o_4)) && isodd(K)
            return 0.0
        end

        a, b, c, d = a_o[o_1], a_o[o_2], a_o[o_3], a_o[o_4]
        P = parity(l_o[o_1] + l_o[o_2])

        if T_1 == T_2
            # Proton-proton & neutron-neutron MEs ... (only pairs inside the N2max truncation are available)
            if Orb_NN.Map[max(a,b),min(a,b),K+1,2] == 0 || Orb_NN.Map[max(c,d),min(c,d),K+1,2] == 0
                return 0.0
            end
            return (T_1 == -1) ? O2b_pp(a,b,c,d,K,P,O,Orb,Orb_NN) : O2b_nn(a,b,c,d,K,P,O,Orb,Orb_NN)
        end

        # Proton-neutron MEs ... stored as <pn|O|pn>, <np|..> = -(-1)^((j_1 + j_2)/2 - K) <pn|..> & the same for the ket ...
        Amp = 1.0
        if T_1 == 1
            Amp *= - phase(div(j_o[o_1] + j_o[o_2],2) - K)
            a, b = b, a
        end
        if T_3 == 1
            Amp *= - phase(div(j_o[o_3] + j_o[o_4],2) - K)
            c, d = d, c
        end
        if Orb_NN.Map[a,b,K+1,1] == 0 || Orb_NN.Map[c,d,K+1,1] == 0
            return 0.0
        end
        return Amp * O2b_pn(a,b,c,d,K,P,O,Orb_NN)
    end

    Sigma_NN::O2B = Correlated ? Sigma : V_NN       # (type-stable placeholder, never used if !Correlated)
    V(o_1,o_2,o_3,o_4,K) = ME(V_NN,o_1,o_2,o_3,o_4,K)
    S(o_1,o_2,o_3,o_4,K) = Correlated ? ME(Sigma_NN,o_1,o_2,o_3,o_4,K) : 0.0

        # e-dressed sigma ... sum_k s_k sum_a e_(o_k a) S(.. o_k -> a ..) (e is a scalar, the dressing keeps the J coupling) ...
    function eS(o_1,o_2,o_3,o_4,K,s_1,s_2,s_3,s_4)
        X = 0.0
        if !Correlated
            return X
        end
        @inbounds for a in Blk[o_1]
            (s_1 != 0) && (E[o_1,a] != 0.0) && (X += s_1 * E[o_1,a] * S(a,o_2,o_3,o_4,K))
        end
        @inbounds for a in Blk[o_2]
            (s_2 != 0) && (E[o_2,a] != 0.0) && (X += s_2 * E[o_2,a] * S(o_1,a,o_3,o_4,K))
        end
        @inbounds for a in Blk[o_3]
            (s_3 != 0) && (E[o_3,a] != 0.0) && (X += s_3 * E[o_3,a] * S(o_1,o_2,a,o_4,K))
        end
        @inbounds for a in Blk[o_4]
            (s_4 != 0) && (E[o_4,a] != 0.0) && (X += s_4 * E[o_4,a] * S(o_1,o_2,o_3,a,K))
        end
        return X
    end

        # Particle-hole (Pandya) ME X^ph_K(x1 y1;u1 v1) of X_(y1 u1 v1 x1) ... Kind = 1 (V), 2 (sigma), 3 (e-dressed sigma of the ring) ...
        #   cached in Cache (one cache per task, see the A_22 loop) ...

    function X_L(Kind::Int64,y_1,u_1,v_1,x_1,L)
        if Kind == 1
            return V(y_1,u_1,v_1,x_1,L)
        elseif Kind == 2
            return S(y_1,u_1,v_1,x_1,L)
        else
            # 1/2 (- e_h1 + e_p3 - e_h3 + e_p1) on the legs of s_h1p3h3p1 ...
            return 0.5 * eS(y_1,u_1,v_1,x_1,L,-1,1,-1,1)
        end
    end

    function X_ph(Cache::Dict{NTuple{6,Int64},Float64},Kind::Int64,x_1,y_1,u_1,v_1,K)
        Key = (Kind,x_1,y_1,u_1,v_1,K)
        X = get(Cache,Key,NaN)
        if !isnan(X)
            return X
        end

        j_x, j_y, j_u, j_v = j_o[x_1], j_o[y_1], j_o[u_1], j_o[v_1]
        X = 0.0
        @inbounds for L in div(max(abs(j_y - j_u),abs(j_v - j_x)),2):div(min(j_y + j_u,j_v + j_x),2)
            W = w6j(j_x,j_y,2*K,j_u,j_v,2*L)
            if W != 0.0
                X += (2*L + 1) * W * X_L(Kind,y_1,u_1,v_1,x_1,L)
            end
        end
        X *= phase(div(j_x + j_y + j_u + j_v,2))

        Cache[Key] = X
        return X
    end

    # 3) Occupation factors ...
        # N_c of the metric ... n_h1 n_h2 nb_p1 nb_p2 - nb_h1 nb_h2 n_p1 n_p2 ...
    N_cal(n_p1,n_p2,n_h1,n_h2) = n_h1 * n_h2 * (1.0 - n_p1) * (1.0 - n_p2) - (1.0 - n_h1) * (1.0 - n_h2) * n_p1 * n_p2

        # A_12 vertices (the same as in RPA_allocate_SE) ... Fa = Fh(h,h1,p,p1,p2), Fb = Fp(p,p1,h,h1,h2) & G of the e_h1p1 term ...
    function Fh(h,h_1,p,p_1,p_2)
        n_h, n_h1, n_p, n_p1, n_p2 = n_o[h], n_o[h_1], n_o[p], n_o[p_1], n_o[p_2]
        F = 0.5 * (n_h - n_p) * (n_h1 * (1.0 - n_p1) * (1.0 - n_p2) + (1.0 - n_h1) * n_p1 * n_p2)
        F += 0.5 * (n_h * n_h1 * (1.0 - n_p1) * (1.0 - n_p2) - (1.0 - n_h) * (1.0 - n_h1) * n_p1 * n_p2)
        return F
    end

    function Fp(p,p_1,h,h_1,h_2)
        n_h, n_h1, n_h2, n_p, n_p1 = n_o[h], n_o[h_1], n_o[h_2], n_o[p], n_o[p_1]
        F = 0.5 * (n_h - n_p) * ((1.0 - n_p1) * n_h1 * n_h2 + n_p1 * (1.0 - n_h1) * (1.0 - n_h2))
        F += 0.5 * ((1.0 - n_p) * (1.0 - n_p1) * n_h1 * n_h2 - n_p * n_p1 * (1.0 - n_h1) * (1.0 - n_h2))
        return F
    end

    G(h,h_1,p,p_1) = (n_o[h] - n_o[p]) * (n_o[h_1] - n_o[p_1])

        # V_22 ... pp ladder c_pp A^h v_p1p2p3p4 & hh ladder c_hh A^p v_h1h2h3h4 (all rho-only v terms of Sec. 3 in the NAT basis) ...
    function c_pp(p_1,p_2,p_3,p_4,h_1,h_2)
        X, Y = n_o[h_1] * n_o[h_2], n_o[h_1] + n_o[h_2]
        a, b = n_o[p_1] * n_o[p_2], n_o[p_3] * n_o[p_4]
        s, t = n_o[p_1] + n_o[p_2], n_o[p_3] + n_o[p_4]
        return X * (1.0 - s) * (1.0 - t) - 0.5 * (1.0 - Y) * (a * (1.0 - t) + b * (1.0 - s))
    end

    function c_hh(h_1,h_2,h_3,h_4,p_1,p_2)
        X_1, X_2 = n_o[h_1] * n_o[h_2], n_o[h_3] * n_o[h_4]
        Y_1, Y_2 = n_o[h_1] + n_o[h_2], n_o[h_3] + n_o[h_4]
        a, s = n_o[p_1] * n_o[p_2], n_o[p_1] + n_o[p_2]
        return a * (1.0 - Y_1) * (1.0 - Y_2) - 0.5 * (1.0 - s) * (X_1 * (1.0 - Y_2) + X_2 * (1.0 - Y_1))
    end

        # ... ring P_p1p2 P_h1h2 P_p3p4 P_h3h4[f_ring d_p2p4 d_h2h4 v_h1p3h3p1] ... active (p1,h1) & (p3,h3), spectators p2 = p4 & h2 = h4 ...
    function f_ring(p_1,h_1,p_3,h_3,p_s,h_s)
        x_1, x_3, y_1, y_3 = n_o[h_1], n_o[h_3], n_o[p_1], n_o[p_3]
        H, P = n_o[h_s], n_o[p_s]
        f = H * P * (x_1 - y_1) * (x_3 - y_3) - x_1 * x_3 * H - y_1 * y_3 * P + 0.5 * (H + P) * (x_1 * y_3 + x_3 * y_1)
        f += 0.5 * (H - P) * (x_1 * x_3 * (y_1 + y_3) - y_1 * y_3 * (x_1 + x_3))
        return f
    end

    # 4) J-coupled 2p-2h configurations ...
        # Particle-particle & hole-hole pairs (o1 <= o2 in the order of Par & Hol) grouped by (total isospin, parity, J_12) ...
    function pairs_make(Orb_list::Vector{Int64})
        Pairs = Dict{Tuple{Int64,Int64,Int64},Vector{Tuple{Int64,Int64}}}()
        @inbounds for i_1 in 1:length(Orb_list), i_2 in i_1:length(Orb_list)
            o_1, o_2 = Orb_list[i_1], Orb_list[i_2]
            @inbounds for K in div(abs(j_o[o_1] - j_o[o_2]),2):div(j_o[o_1] + j_o[o_2],2)
                if (o_1 == o_2) && isodd(K)
                    continue
                end
                Key = (T_o[o_1] + T_o[o_2],rem(l_o[o_1] + l_o[o_2],2),K)
                push!(get!(Pairs,Key,Vector{Tuple{Int64,Int64}}()),(o_1,o_2))
            end
        end
        return Pairs
    end

    PP_pairs = pairs_make(Par)
    HH_pairs = pairs_make(Hol)

        # Configurations (p1,p2,h1,h2,J12,J34) of the (J,P) channel ...
    function configurations_make(J::Int64,P::Int64)
        Configurations = Vector{NTuple{6,Int64}}()
        @inbounds for ((T_12,L_12,J12),PP_list) in PP_pairs, ((T_34,L_34,J34),HH_list) in HH_pairs
            if (T_12 != T_34) || (parity(L_12 + L_34) != P) || !triangle(2*J12,2*J34,2*J)
                continue
            end
            @inbounds for (p_1,p_2) in PP_list, (h_1,h_2) in HH_list
                if (min(n_o[h_1],n_o[h_2]) - max(n_o[p_1],n_o[p_2])) <= Tol_ord
                    continue
                end
                if abs(N_cal(n_o[p_1],n_o[p_2],n_o[h_1],n_o[h_2])) < Tol_N
                    continue
                end
                push!(Configurations,(p_1,p_2,h_1,h_2,J12,J34))
            end
        end
        sort!(Configurations)
        return Configurations
    end

        # Orientation pi = 1..4 of a configuration ... (p1,p2,h1,h2), (p2,p1,h1,h2), (p1,p2,h2,h1), (p2,p1,h2,h1) with the J-coupled
        #   exchange phases, i.e. P_p1p2[f] = f(p1,p2) - (-1)^((j_p1 + j_p2)/2 - J12) f(p2,p1) for the J-coupled kernels f ...
    function orient(c::NTuple{6,Int64},pi::Int64)
        p_1, p_2, h_1, h_2, J12, J34 = c
        Sign = 1.0
        if (pi == 2) || (pi == 4)
            Sign *= - phase(div(j_o[p_1] + j_o[p_2],2) - J12)
            p_1, p_2 = p_2, p_1
        end
        if (pi == 3) || (pi == 4)
            Sign *= - phase(div(j_o[h_1] + j_o[h_2],2) - J34)
            h_1, h_2 = h_2, h_1
        end
        return p_1, p_2, h_1, h_2, Sign
    end

        # 9j recoupling U[K,K'] of the orientation (x1,x2,y1,y2) ... [(x1 x2)J12 (y1 y2)J34]J -> [(x1 y1)K (x2 y2)K']J ...
    function recoupling_make(c::NTuple{6,Int64},J::Int64)
        J12, J34 = c[5], c[6]
        Rec = Vector{Tuple{Int64,Matrix{Float64}}}(undef,4)
        @inbounds for pi in 1:4
            x_1, x_2, y_1, y_2, ~ = orient(c,pi)
            K_min, K_max = div(abs(j_o[x_1] - j_o[y_1]),2), div(j_o[x_1] + j_o[y_1],2)
            Kp_min, Kp_max = div(abs(j_o[x_2] - j_o[y_2]),2), div(j_o[x_2] + j_o[y_2],2)
            U = zeros(Float64,K_max - K_min + 1,Kp_max - Kp_min + 1)
            @inbounds for K in K_min:K_max, Kp in Kp_min:Kp_max
                if triangle(2*K,2*Kp,2*J)
                    U[K - K_min + 1,Kp - Kp_min + 1] = sqrt((2*J12 + 1) * (2*J34 + 1) * (2*K + 1) * (2*Kp + 1)) *
                                                       w9j(j_o[x_1],j_o[x_2],2*J12,j_o[y_1],j_o[y_2],2*J34,2*K,2*Kp,2*J)
                end
            end
            Rec[pi] = (K_min,U)
        end
        return Rec
    end

    # 5) A_22 & N_22 elements (unnormalized configurations) ...
        # One-body part ... A^p A^h N_c & the e on one leg (1/2 {D,.} of A^p A^h N_c) ... d = orientation of c' (right antisymmetrization) ...
    function element_1b(c,u_1,u_2,v_1,v_2)
        p_1, p_2, h_1, h_2 = c[1], c[2], c[3], c[4]
        e_p1, e_p2, e_h1, e_h2 = p_1 == u_1, p_2 == u_2, h_1 == v_1, h_2 == v_2
        n_p1, n_p2, n_h1, n_h2 = n_o[p_1], n_o[p_2], n_o[h_1], n_o[h_2]
        NE, AE = 0.0, 0.0
        if e_p1 && e_p2 && e_h1 && e_h2
            NE = N_cal(n_p1,n_p2,n_h1,n_h2)
        end
        # e on one leg ... the occupation of the e link is the average of its two ends ...
        if e_p2 && e_h1 && e_h2 && (E[p_1,u_1] != 0.0)
            AE += E[p_1,u_1] * N_cal(0.5 * (n_p1 + n_o[u_1]),n_p2,n_h1,n_h2)
        end
        if e_p1 && e_h1 && e_h2 && (E[p_2,u_2] != 0.0)
            AE += E[p_2,u_2] * N_cal(n_p1,0.5 * (n_p2 + n_o[u_2]),n_h1,n_h2)
        end
        if e_p1 && e_p2 && e_h2 && (E[h_1,v_1] != 0.0)
            AE -= E[h_1,v_1] * N_cal(n_p1,n_p2,0.5 * (n_h1 + n_o[v_1]),n_h2)
        end
        if e_p1 && e_p2 && e_h1 && (E[h_2,v_2] != 0.0)
            AE -= E[h_2,v_2] * N_cal(n_p1,n_p2,n_h1,0.5 * (n_h2 + n_o[v_2]))
        end
        return AE, NE
    end

        # Particle-particle ladder ... hole pair (h1,h2) linked to (v1,v2), the particle pairs by v, sigma & e-dressed sigma ...
    function element_pp(c,u_1,u_2,v_1,v_2,J12)
        p_1, p_2, h_1, h_2 = c[1], c[2], c[3], c[4]
        e_h1, e_h2 = h_1 == v_1, h_2 == v_2
        n_h1, n_h2 = n_o[h_1], n_o[h_2]
        AE, NE = 0.0, 0.0
        if !(e_h1 || e_h2)
            return AE, NE
        end
        S_P = S(p_1,p_2,u_1,u_2,J12)
        if e_h1 && e_h2
            NE += (n_h1 + n_h2 - 1.0) * S_P
            AE += c_pp(p_1,p_2,u_1,u_2,h_1,h_2) * V(p_1,p_2,u_1,u_2,J12)
            AE += (n_h1 + n_h2 - 1.0) * 0.5 * eS(p_1,p_2,u_1,u_2,J12,1,1,1,1)
        end
        if e_h2 && (E[h_1,v_1] != 0.0)
            AE -= E[h_1,v_1] * (0.5 * (n_h1 + n_o[v_1]) + n_h2 - 1.0) * S_P
        end
        if e_h1 && (E[h_2,v_2] != 0.0)
            AE -= E[h_2,v_2] * (n_h1 + 0.5 * (n_h2 + n_o[v_2]) - 1.0) * S_P
        end
        return AE, NE
    end

        # Hole-hole ladder ... particle pair (p1,p2) linked to (u1,u2), the hole pairs by v, sigma & e-dressed sigma ...
    function element_hh(c,u_1,u_2,v_1,v_2,J34)
        p_1, p_2, h_1, h_2 = c[1], c[2], c[3], c[4]
        e_p1, e_p2 = p_1 == u_1, p_2 == u_2
        n_p1, n_p2 = n_o[p_1], n_o[p_2]
        AE, NE = 0.0, 0.0
        if !(e_p1 || e_p2)
            return AE, NE
        end
        S_H = S(h_1,h_2,v_1,v_2,J34)
        if e_p1 && e_p2
            NE += (1.0 - n_p1 - n_p2) * S_H
            AE += c_hh(h_1,h_2,v_1,v_2,p_1,p_2) * V(h_1,h_2,v_1,v_2,J34)
            AE -= (1.0 - n_p1 - n_p2) * 0.5 * eS(h_1,h_2,v_1,v_2,J34,1,1,1,1)
        end
        if e_p2 && (E[p_1,u_1] != 0.0)
            AE += E[p_1,u_1] * (1.0 - 0.5 * (n_p1 + n_o[u_1]) - n_p2) * S_H
        end
        if e_p1 && (E[p_2,u_2] != 0.0)
            AE += E[p_2,u_2] * (1.0 - n_p1 - 0.5 * (n_p2 + n_o[u_2])) * S_H
        end
        return AE, NE
    end

        # Ring ... active (x1,y1) -> (u1,v1) in the ph channel K, spectators x2 ~ u2 & y2 ~ v2 (d or e link) ...
    function element_ring(x_1,x_2,y_1,y_2,u_1,u_2,v_1,v_2,Rec_L,Rec_R,Cache)
        e_p, e_h = x_2 == u_2, y_2 == v_2
        AE, NE = 0.0, 0.0
        if !(e_p || e_h)
            return AE, NE
        end
        n_xs, n_ys = n_o[x_2], n_o[y_2]

        # Coefficients of sigma^ph (c_S, c_N), e-dressed sigma^ph (c_E) & v^ph (c_V) ...
        c_N, c_S, c_E, c_V = 0.0, 0.0, 0.0, 0.0
        if e_p && e_h
            c_N = n_xs - n_ys
            c_E = n_xs - n_ys
            c_V = f_ring(x_1,y_1,u_1,v_1,x_2,y_2)
        end
        if e_h && (E[x_2,u_2] != 0.0)
            c_S += E[x_2,u_2] * (0.5 * (n_xs + n_o[u_2]) - n_ys)
        end
        if e_p && (E[y_2,v_2] != 0.0)
            c_S -= E[y_2,v_2] * (n_xs - 0.5 * (n_ys + n_o[v_2]))
        end
        if !Correlated
            c_N, c_S, c_E = 0.0, 0.0, 0.0
        end
        if (c_N == 0.0) && (c_S == 0.0) && (c_E == 0.0) && (c_V == 0.0)
            return AE, NE
        end

        # W(K) = sum_K' U_L[K,K'] U_R[K,K'] ... (the spectators have the same j's, i.e. the same K' range)
        K_minL, U_L = Rec_L
        K_minR, U_R = Rec_R
        @inbounds for K in max(K_minL,K_minR):min(K_minL + size(U_L,1),K_minR + size(U_R,1)) - 1
            W = 0.0
            @inbounds for k_p in 1:size(U_L,2)
                W += U_L[K - K_minL + 1,k_p] * U_R[K - K_minR + 1,k_p]
            end
            if W == 0.0
                continue
            end
            if c_V != 0.0
                AE += W * c_V * X_ph(Cache,1,x_1,y_1,u_1,v_1,K)
            end
            if (c_N != 0.0) || (c_S != 0.0)
                S_ph = X_ph(Cache,2,x_1,y_1,u_1,v_1,K)
                NE += W * c_N * S_ph
                AE += W * c_S * S_ph
            end
            if c_E != 0.0
                AE += W * c_E * X_ph(Cache,3,x_1,y_1,u_1,v_1,K)
            end
        end
        return AE, NE
    end

        # Contraction ... (x1,y1) of the left configuration contracted by e_y1x1 (K = 0), the 3 remaining legs of the right one via sigma ...
        #   G(c,c') = sum_(pi_L) [sum_(p swaps of c') g1 + sum_(h swaps of c') g2] , A_22 += 1/2 (G(c,c') + G(c',c)) ...
    function element_contraction(c,Rec_c,d,J::Int64)
        AE = 0.0
        J12, J34 = d[5], d[6]
        Pi_d = sqrt((2*J12 + 1) * (2*J34 + 1))
        @inbounds for pi_L in 1:4
            x_1, x_2, y_1, y_2, Sign_L = orient(c,pi_L)
            if E[y_1,x_1] == 0.0
                continue
            end
            K_min, U = Rec_c[pi_L]
            if (K_min != 0) || !triangle(j_o[x_2],j_o[y_2],2*J)
                continue
            end
            Left = Sign_L * (n_o[x_1] - n_o[y_1]) * E[y_1,x_1] * U[1,J - div(abs(j_o[x_2] - j_o[y_2]),2) + 1] * sqrt(j_o[x_1] + 1.0)
            if Left == 0.0
                continue
            end
            # g1 ... d_x2u2 , sigma_(y2 u1 v1 v2) (right config (u1,u2=x2,v1,v2)) ...
            @inbounds for pi_R in (1,2)
                u_1, u_2, v_1, v_2, Sign_R = orient(d,pi_R)
                if x_2 == u_2
                    W = w6j(j_o[x_2],j_o[u_1],2*J12,2*J34,2*J,j_o[y_2])
                    if W != 0.0
                        AE += Left * Sign_R * phase(J12 + J34 + J) * Pi_d * W * S(y_2,u_1,v_1,v_2,J34)
                    end
                end
            end
            # g2 ... d_y2v2 , sigma_(v1 x2 u1 u2) (right config (u1,u2,v1,v2=y2)) ...
            @inbounds for pi_R in (1,3)
                u_1, u_2, v_1, v_2, Sign_R = orient(d,pi_R)
                if y_2 == v_2
                    W = w6j(j_o[y_2],j_o[v_1],2*J34,2*J12,2*J,j_o[x_2])
                    if W != 0.0
                        AE += Left * Sign_R * phase(div(j_o[y_2] + j_o[v_1],2) + J12 + J) * Pi_d * W * S(v_1,x_2,u_1,u_2,J12)
                    end
                end
            end
        end
        return AE
    end

        # Full element (A_22, N_22)[c,c'] ...
    function element_22(c,d,Rec_c,Rec_d,J::Int64,Cache)
        AE, NE = 0.0, 0.0

        # Configurations without a common leg (particle or hole) are not coupled at this order ...
        if !((c[1] == d[1]) || (c[1] == d[2]) || (c[2] == d[1]) || (c[2] == d[2]) ||
             (c[3] == d[3]) || (c[3] == d[4]) || (c[4] == d[3]) || (c[4] == d[4]))
            return AE, NE
        end

        # One-body part & ladders ... conserve J12 & J34, antisymmetrization of the right configuration ...
        if (c[5] == d[5]) && (c[6] == d[6])
            @inbounds for pi_R in 1:4
                u_1, u_2, v_1, v_2, Sign_R = orient(d,pi_R)
                A_1b, N_1b = element_1b(c,u_1,u_2,v_1,v_2)
                AE += Sign_R * A_1b
                NE += Sign_R * N_1b
                if pi_R == 1 || pi_R == 3
                    A_pp, N_pp = element_pp(c,u_1,u_2,v_1,v_2,c[5])
                    AE += Sign_R * A_pp
                    NE += Sign_R * N_pp
                end
                if pi_R == 1 || pi_R == 2
                    A_hh, N_hh = element_hh(c,u_1,u_2,v_1,v_2,c[6])
                    AE += Sign_R * A_hh
                    NE += Sign_R * N_hh
                end
            end
        end

        # Ring ... full antisymmetrization of both configurations ...
        @inbounds for pi_L in 1:4
            x_1, x_2, y_1, y_2, Sign_L = orient(c,pi_L)
            @inbounds for pi_R in 1:4
                u_1, u_2, v_1, v_2, Sign_R = orient(d,pi_R)
                if !((x_2 == u_2) || (y_2 == v_2))
                    continue
                end
                A_r, N_r = element_ring(x_1,x_2,y_1,y_2,u_1,u_2,v_1,v_2,Rec_c[pi_L],Rec_d[pi_R],Cache)
                AE += Sign_L * Sign_R * A_r
                NE += Sign_L * Sign_R * N_r
            end
        end

        # Contraction (e_hp sigma terms) ...
        if Correlated
            AE += 0.5 * (element_contraction(c,Rec_c,d,J) + element_contraction(d,Rec_d,c,J))
        end

        return AE, NE
    end

    # 6) A_12 & N_12 columns ... vertices alpha (spectator hole h ~ h2) & beta (spectator particle p ~ p2) as in RPA_allocate_SE ...
        # alpha^(ph)_(h1;p1p2) = <h1 p;K|alpha|p1 p2;K> ... (the d_hh2 part, the e_hh2 sigma part is added separately)
    function alpha(h,h_1,p,p_1,p_2,K::Int64)
        ME_a = Fh(h,h_1,p,p_1,p_2) * V(h_1,p,p_1,p_2,K)
        ME_a += 0.5 * ((p == p_2) ? 1.0 : 0.0) * E[h_1,p_1] * G(h,h_1,p,p_1)
        ME_a -= 0.5 * phase(div(j_o[p] + j_o[p_2],2) - K) * ((p == p_1) ? 1.0 : 0.0) * E[h_1,p_2] * G(h,h_1,p,p_2)
        ME_a += 0.5 * eS(h_1,p,p_1,p_2,K,1,-1,-1,-1)
        return ME_a
    end

        # beta^(ph)_(p1;h1h2) = <h p1;K|beta|h1 h2;K> ... (the d_pp2 part)
    beta(p,p_1,h,h_1,h_2,K::Int64) = Fp(p,p_1,h,h_1,h_2) * V(h,p_1,h_1,h_2,K) + 0.5 * eS(h,p_1,h_1,h_2,K,1,-1,1,1)

        # Recoupling <(p h^-1)_J| ... |[(p1 p2)_J12 (h1 h2)_J34]_J> of the vertices (as in RPA_allocate_SE) ...
    function C_alpha(j_p::Int64,j_h::Int64,j_x::Int64,J::Int64,J12::Int64,J34::Int64)
        Amp = sqrt((2*J12 + 1) * (2*J34 + 1)) * w6j(j_p,j_h,2*J,2*J34,2*J12,j_x)
        return phase(div(j_p + j_h,2) + div(j_p + j_x,2) + J + J12) * Amp
    end

    function C_beta(j_p::Int64,j_h::Int64,j_x::Int64,J::Int64,J12::Int64,J34::Int64)
        Amp = sqrt((2*J12 + 1) * (2*J34 + 1)) * w6j(j_p,j_h,2*J,2*J34,2*J12,j_x)
        return phase(div(j_p + j_h,2) + div(j_p + j_x,2) + div(j_h + j_x,2) + J + J12 + J34) * Amp
    end

        # Column c of A_12 & N_12 ... Rows_H[o] (Rows_P[o]) ... rows whose 1p-1h hole (particle) is o, with their particle (hole) ...
        #   (C_alpha & C_beta refer to the hole order at_h1 at_h2, i.e. to - O+_c, hence the overall sign -1)
    function column_12!(A_c,N_c,Rows_H,Rows_P,c,J::Int64,Norm_c::Float64)
        p_1, p_2, h_1, h_2, J12, J34 = c
        Norm = - Norm_c

        # Spectator hole ... (spectator h2, active h1) & (spectator h1, active h2 with the exchange phase of the hole pair) ...
        @inbounds for (h_s,h_a,Amp_s) in ((h_2,h_1,Norm),(h_1,h_2,- Norm * phase(div(j_o[h_1] + j_o[h_2],2) - J34)))
            @inbounds for h in Blk[h_s], (i,p) in Rows_H[h]
                C = Amp_s * C_alpha(j_o[p],j_o[h_s],j_o[h_a],J,J12,J34)
                if C == 0.0
                    continue
                end
                S_a = S(h_a,p,p_1,p_2,J12)
                if h == h_s
                    A_c[i] += C * alpha(h_s,h_a,p,p_1,p_2,J12)
                    N_c[i] -= C * S_a
                end
                A_c[i] += C * E[h,h_s] * S_a
            end
        end

        # Spectator particle ... (spectator p2, active p1) & (spectator p1, active p2 with the exchange phase of the particle pair) ...
        @inbounds for (p_s,p_a,Amp_s) in ((p_2,p_1,Norm),(p_1,p_2,- Norm * phase(div(j_o[p_1] + j_o[p_2],2) - J12)))
            @inbounds for p in Blk[p_s], (i,h) in Rows_P[p]
                C = Amp_s * C_beta(j_o[p_s],j_o[h],j_o[p_a],J,J12,J34)
                if C == 0.0
                    continue
                end
                S_b = S(h,p_a,h_1,h_2,J34)
                if p == p_s
                    A_c[i] += C * beta(p_s,p_a,h,h_1,h_2,J34)
                    N_c[i] -= C * S_b
                end
                A_c[i] -= C * E[p,p_s] * S_b
            end
        end
        return
    end

    # 7) Spectral representation of G(w) = (w N_22 - A_22)^-1 = U (w - Omega)^-1 U^T ...
    function spectral(A_22::Matrix{Float64},N_22::Matrix{Float64})
        D = [1.0 / sqrt(abs(N_22[k,k])) for k in 1:size(N_22,1)]
        if isdiag(N_22)
            # Diagonal metric (HF or sigma = 0) ... X = N_22^-1/2 ...
            Omega, Z = eigen(Symmetric(D .* A_22 .* D'))
            return D .* Z, Omega, 0
        end
        Lambda, Q = eigen(Symmetric(D .* N_22 .* D'))
        Keep = Lambda .> Tol_N
        X = (D .* Q[:,Keep]) ./ sqrt.(Lambda[Keep])'
        Omega, Z = eigen(Symmetric(X' * A_22 * X))
        return X * Z, Omega, count(.!Keep)
    end

    f_reg(x::Float64) = (Eta == 0.0) ? 1.0 / x : x / (x^2 + Eta^2)

        # Assembly at energy w from the 2p-2h eigenmodes ... F_A = A_12 U, F_N = N_12 U & the weights g = diag of U^T G(w) U ...
        #   Sigma ... (F_A - w F_N) g (F_A - w F_N)^T ,   Norm ... F_A g F_N^T + F_N g F_A^T - w F_N g F_N^T ...
    function assemble(F_A::Matrix{Float64},F_N::Matrix{Float64},g::Vector{Float64},w_k::Float64)
        if Quantity == "Sigma"
            F = F_A .- w_k .* F_N
            X = (F .* g') * F'
        else
            X = (F_A .* g') * F_N'
            X = X + X' - w_k .* ((F_N .* g') * F_N')
        end
        return 0.5 * (X + X')
    end

    # 8) Loop over the (J,P) channels ...
    Sigma_E = [Matrix{Matrix{Float64}}(undef,2,J_max+1) for k in 1:length(Energies)]
    N_task = Threads.nthreads()
    Caches = [Dict{NTuple{6,Int64},Float64}() for t in 1:N_task]     # particle-hole MEs X_ph, one cache per task (shared by all channels) ...
    Blocks = Dict{Tuple{Int64,Int64},Any}()
    N_conf_max, N_conf_total, N_removed = 0, 0, 0
    Omega_min = Inf

    @inbounds for JP in JP_list
        J, P = JP[1], JP[2]
        N_ph = Orb_1p1h.N[P,J+1]

        Configurations = configurations_make(J,P)
        N_conf = length(Configurations)
        N_conf_max = max(N_conf_max,N_conf)
        N_conf_total += N_conf

        if (N_ph == 0) || (N_conf == 0)
            for k in 1:length(Energies)
                Sigma_E[k][P,J+1] = zeros(Float64,N_ph,N_ph)
            end
            continue
        end

        Norm = [1.0 / sqrt(((c[1] == c[2]) ? 2.0 : 1.0) * ((c[3] == c[4]) ? 2.0 : 1.0)) for c in Configurations]

        # 1p-1h rows ...
        Rows_H = [Tuple{Int64,Int64}[] for o in 1:N_o]
        Rows_P = [Tuple{Int64,Int64}[] for o in 1:N_o]
        @inbounds for (i,Orb_ph) in enumerate(Orb_1p1h.Ind[P,J+1])
            p, h = orbital(Orb_ph.a,Orb_ph.T), orbital(Orb_ph.b,Orb_ph.T)
            push!(Rows_H[h],(i,p))
            push!(Rows_P[p],(i,h))
        end

        # A_12 & N_12 ... (threads over configurations)
        A_12 = zeros(Float64,N_ph,N_conf)
        N_12 = zeros(Float64,N_ph,N_conf)
        Threads.@threads for k in 1:N_conf
            column_12!(view(A_12,:,k),view(N_12,:,k),Rows_H,Rows_P,Configurations[k],J,Norm[k])
        end
        if !Coupling_N
            N_12 .= 0.0
        end

        # Single-particle resolvent ... G_c(w) = 1 / (N_c (w - Delta_c)), Sigma(w) = (A_12 - w N_12) diag(f(w - Delta_c)/N_c) (..)^T ...
        if Resolvent == "Diagonal"
            N_d = [N_cal(n_o[c[1]],n_o[c[2]],n_o[c[3]],n_o[c[4]]) for c in Configurations]
            Delta = [E[c[1],c[1]] + E[c[2],c[2]] - E[c[3],c[3]] - E[c[4],c[4]] for c in Configurations]
            if !isempty(Delta)
                Omega_min = min(Omega_min,minimum(Delta))
            end
            if Return_blocks
                Blocks[(J,P)] = (Configurations = Configurations,A_12 = copy(A_12),N_12 = copy(N_12),
                                 A_22 = Matrix(Diagonal(N_d .* Delta)),N_22 = Matrix(Diagonal(N_d)))
            end
            @inbounds for (k,E_k) in enumerate(Energies)
                Sigma_E[k][P,J+1] = assemble(A_12,N_12,f_reg.(E_k .- Delta) ./ N_d,E_k)
            end
            continue
        end

        # Full resolvent ... A_22 & N_22 (threads over rows, upper triangle)
        Rec = Vector{Vector{Tuple{Int64,Matrix{Float64}}}}(undef,N_conf)
        Threads.@threads for k in 1:N_conf
            Rec[k] = recoupling_make(Configurations[k],J)
        end

        A_22 = zeros(Float64,N_conf,N_conf)
        N_22 = zeros(Float64,N_conf,N_conf)
        @sync for t in 1:N_task
            Threads.@spawn begin
                Cache = Caches[t]
                # Interleaved rows k = t, t + N_task, ... (balanced upper triangle), each task owns its cache ...
                @inbounds for k in t:N_task:N_conf, l in k:N_conf
                    AE, NE = element_22(Configurations[k],Configurations[l],Rec[k],Rec[l],J,Cache)
                    if (AE != 0.0) || (NE != 0.0)
                        A_22[k,l] = A_22[l,k] = Norm[k] * Norm[l] * AE
                        N_22[k,l] = N_22[l,k] = Norm[k] * Norm[l] * NE
                    end
                end
            end
        end
        if Return_blocks
            Blocks[(J,P)] = (Configurations = Configurations,A_12 = copy(A_12),N_12 = copy(N_12),A_22 = copy(A_22),N_22 = copy(N_22))
        end

        # Spectral representation & Sigma(w) for all energies ...
        U, Omega, N_rem = spectral(A_22,N_22)
        N_removed += N_rem
        if !isempty(Omega)
            Omega_min = min(Omega_min,minimum(Omega))
        end
        F_A = A_12 * U
        F_N = N_12 * U
        @inbounds for (k,E_k) in enumerate(Energies)
            Sigma_E[k][P,J+1] = assemble(F_A,F_N,f_reg.(E_k .- Omega),E_k)
        end
    end

    @printf("\t\t\t2p-2h configurations ... %d in total, %d in the largest channel, %d removed (N_22 eigenvalues < Tol_N), lowest 2p-2h pole %.4f MeV\n",
            N_conf_total,N_conf_max,N_removed,Omega_min)
    println("\t\t\tFinished on evaluation of 1p-1h to 2p-2h " * ((Quantity == "Sigma") ? "self-energy ..." : "norm correction ..."))

    Result = (w isa Float64) ? Sigma_E[1] : Sigma_E
    return Return_blocks ? (Result,Blocks) : Result
end

function RPA_allocate_corr_NE(Params::Parameters,w::Union{Float64,AbstractVector{Float64}},Orb::Vector{Orb1B},Orb_1p1h::Orb2B_1p1h,Orb_NN::Orb2B,
                              h_N::O1B,V_NN::O2B,Sigma::Union{O2B,Nothing};
                              Resolvent::String = "Diagonal",Eta::Float64 = 0.0,Tol_N::Float64 = 1e-8,Return_blocks::Bool = false)
    #=
        Energy-dependent 1p-1h to 2p-2h correction dN(w) of the TDA norm (metric) matrix N for a general correlated reference (NAT basis) ...
            The exact downfolding of the SRPA-TDA problem [A_11 A_12; A_21 A_22] = w [N_11 N_12; N_21 N_22] onto the 1p-1h space,
                (A_11 - w N_11) + (A_12 - w N_12) G(w) (A_21 - w N_21) ,   G(w) = (w N_22 - A_22)^-1 ,
            is written as A_eff(w) - w N_eff(w) with all explicit factors w collected into the norm ...
                A_eff(w) = A_11 + A_12 G A_21                     ... RPA_allocate_corr_SE(...; Metric_12 = false) ,
                N_eff(w) = N_11 + dN(w) ,
                dN(w)    = A_12 G N_21 + N_12 G A_21 - w N_12 G N_21 ... this function ,
            equivalently dN = - (A_12 R N_21 + N_12 R A_21) + w N_12 R N_21 with R = (A_22 - w N_22)^-1 = - G ...
            Note ... Sigma(w) of RPA_allocate_corr_SE with Metric_12 = true (default) = A_12 G A_21 - w dN(w), i.e. it already contains
                     dN ... use either [corr_SE(Metric_12 = true), N_11] or [corr_SE(Metric_12 = false), N_11 + dN(w)], not both ...

        Blocks, leading-order power counting & resolvent ... the same as in RPA_allocate_corr_SE (A_12 with V, e & e sigma terms,
            N_12 = - P_p1p2[d_pp2 s_hp1h1h2] - P_h1h2[d_hh2 s_h1pp1p2] = O(V), Resolvent = "Diagonal" (default, G_c = 1/(N_c (w - Delta_c)))
            or "Full") ... dN = O(V^2/h), it vanishes for sigma = 0 (N_12 = 0) ...

        Returns ... dN[P,J+1] (N_ph x N_ph in the 1p-1h basis Orb_1p1h) for a single w, a vector of these for a vector of w
                    (& the blocks if Return_blocks) ...
    =#

    return RPA_allocate_corr_SE(Params,w,Orb,Orb_1p1h,Orb_NN,h_N,V_NN,Sigma;Resolvent = Resolvent,Metric_12 = true,Eta = Eta,Tol_N = Tol_N,
                                Return_blocks = Return_blocks,Quantity = "Norm")
end

function RPA_allocate_HF_S(Params::Parameters,Orb::Vector{Orb1B},Orb_1p1h::Orb2B_1p1h,Orb_NN::Orb2B,h_N::O1B,V_NN::O2B;gamma::Float64 = 5.0)
    #=
        Energy-dependent 1p-1h to 2p-2h self-energy of the TDA matrix A for a Hartree-Fock reference (single energy w) ...
            Sigma(w) = A_12 R(w) A_21 ,   R_c(w) = 1 / (w - Delta_c) ,   Delta_c = e_p1 + e_p2 - e_h1 - e_h2 ,
        HF limit of the notes (N_11 = 1, N_12 = 0, N_22 = 1 for normalized configurations, the norm correction dN vanishes) with the
        single-particle resolvent (no residual interaction in A_22) & the contact * V vertices ...
            A_12 = P_h1h2[d_hh2 v_h1pp1p2] + P_p1p2[d_pp2 v_hp1h1h2] ,
        (the term 1/2 P_p1p2 P_h1h2[d_hh2 d_pp2 e_h1p1] vanishes for a HF reference, Brillouin, & is not included) ...
            e_a = h_N[a,a] ... HF single-particle energies ,  holes ... occupation >= 0.5 , particles ... occupation < 0.5 ...

        2p-2h configurations c = [(p1 p2)_J12 (h1 h2)_J34]_J with p1 <= p2 & h1 <= h2 (normalized, 1/sqrt(2) for equal orbitals) ...
        J-coupled vertices (all angular momenta doubled except J, J12 & J34) ...
            spectator hole h = h2     ... C_alpha(j_p,j_h2,j_h1) <h1 p;J12|V|p1 p2;J12> ,
            spectator hole h = h1     ... - (-1)^((j_h1 + j_h2)/2 - J34) C_alpha(j_p,j_h1,j_h2) <h2 p;J12|V|p1 p2;J12> ,
            spectator particle p = p2 ... C_beta(j_p2,j_h,j_p1) <h p1;J34|V|h1 h2;J34> ,
            spectator particle p = p1 ... - (-1)^((j_p1 + j_p2)/2 - J12) C_beta(j_p1,j_h,j_p2) <h p2;J34|V|h1 h2;J34> ,
            C_alpha & C_beta ... 6j recoupling of the vertices to <(p h^-1)_J| (as in RPA_allocate_SE & RPA_allocate_corr_SE) ...
        The same result follows from RPA_allocate_corr_SE(...; HF = true, Resolvent = "Diagonal") up to the e_h1p1 term
        (the full HF A_22 resolvent ... RPA_allocate_corr_SE(...; HF = true, Resolvent = "Full")) ...

        Returns ... Sigma[P,J+1] (N_ph x N_ph in the 1p-1h basis Orb_1p1h, the same layout as A of RPA_allocate) ...
    =#

    # Read calculation parameters ...
    N_max = Params.Calc.Nmax
    J_max = 2*N_max + 1

    JP_list = JP_initialize(J_max)

    println("\t\tComputing HF 1p-1h to 2p-2h self-energy Sigma(w) of the TDA matrix A ...")

    # Particle & hole orbitals (a,T) of the 2p-2h configurations & their HF single-particle energies e ...
    Par = Vector{Tuple{Int64,Int64}}()
    Hol = Vector{Tuple{Int64,Int64}}()
    @inbounds for Orb_a in Orb
        push!((Orb_a.pO >= 0.5) ? Hol : Par,(Orb_a.a,-1))
        push!((Orb_a.nO >= 0.5) ? Hol : Par,(Orb_a.a,1))
    end

    e(o::Tuple{Int64,Int64}) = (o[2] == -1) ? h_N.p[o[1],o[1]] : h_N.n[o[1],o[1]]
    j(o::Tuple{Int64,Int64}) = Orb[o[1]].j
    l(o::Tuple{Int64,Int64}) = Orb[o[1]].l

    # J-coupled antisymmetrized ME <o1 o2;K|V|o3 o4;K> for all isospin combinations (pp, nn, pn & np) ...
    function V(o_1::Tuple{Int64,Int64},o_2::Tuple{Int64,Int64},o_3::Tuple{Int64,Int64},o_4::Tuple{Int64,Int64},K::Int64)
        (a, T_a), (b, T_b), (c, T_c), (d, T_d) = o_1, o_2, o_3, o_4

        # Charge, parity & angular momentum conservation ...
        if (T_a + T_b) != (T_c + T_d) || parity(l(o_1) + l(o_2)) != parity(l(o_3) + l(o_4))
            return 0.0
        end
        if !triangle(j(o_1),j(o_2),2*K) || !triangle(j(o_3),j(o_4),2*K) || (K + 1) > size(Orb_NN.Map,3)
            return 0.0
        end
        if ((o_1 == o_2) || (o_3 == o_4)) && isodd(K)
            return 0.0
        end
        P = parity(l(o_1) + l(o_2))

        # Proton-proton & neutron-neutron MEs ... (only pairs inside the N2max truncation are available)
        if T_a == T_b
            if Orb_NN.Map[max(a,b),min(a,b),K+1,2] == 0 || Orb_NN.Map[max(c,d),min(c,d),K+1,2] == 0
                return 0.0
            end
            return (T_a == -1) ? O2b_pp(a,b,c,d,K,P,V_NN,Orb,Orb_NN) : O2b_nn(a,b,c,d,K,P,V_NN,Orb,Orb_NN)
        end

        # Proton-neutron MEs ... stored as <pn|V|pn>, <np|..> = -(-1)^((j_a + j_b)/2 - K) <pn|..> & the same for the ket ...
        Amp = 1.0
        if T_a == 1
            Amp *= - phase(div(j(o_1) + j(o_2),2) - K)
            a, b = b, a
        end
        if T_c == 1
            Amp *= - phase(div(j(o_3) + j(o_4),2) - K)
            c, d = d, c
        end
        if Orb_NN.Map[a,b,K+1,1] == 0 || Orb_NN.Map[c,d,K+1,1] == 0
            return 0.0
        end
        return Amp * O2b_pn(a,b,c,d,K,P,V_NN,Orb_NN)
    end

    # Recoupling <(p h^-1)_J| ... |[(p1 p2)_J12 (h1 h2)_J34]_J> of the vertices ... j_x = j of the active (non-spectator) leg ...
    function C_alpha(j_p::Int64,j_h::Int64,j_x::Int64,J::Int64,J12::Int64,J34::Int64)
        Amp = sqrt((2*J12 + 1) * (2*J34 + 1)) * w6j(j_p,j_h,2*J,2*J34,2*J12,j_x)
        return phase(div(j_p + j_h,2) + div(j_p + j_x,2) + J + J12) * Amp
    end

    function C_beta(j_p::Int64,j_h::Int64,j_x::Int64,J::Int64,J12::Int64,J34::Int64)
        Amp = sqrt((2*J12 + 1) * (2*J34 + 1)) * w6j(j_p,j_h,2*J,2*J34,2*J12,j_x)
        return phase(div(j_p + j_h,2) + div(j_p + j_x,2) + div(j_h + j_x,2) + J + J12 + J34) * Amp
    end

    # Self-energy for every (J,P) channel ... Sigma[i,j] = sum_c A_12[i,c] R_c(w) A_12[j,c] ...
    Sigma = Matrix{Matrix{Float64}}(undef,2,J_max+1)

    @inbounds Threads.@threads for JP in JP_list
        J, P = JP[1], JP[2]
        N_ph = Orb_1p1h.N[P,J+1]
        Orb_1p1h_JP = Orb_1p1h.Ind[P,J+1]

        Sigma_JP = zeros(Float64,N_ph,N_ph)
        A_c = zeros(Float64,N_ph)       # column A_12[:,c] of the current configuration c ...

        # Loop over the 2p-2h configurations c = (p1 p2 h1 h2 J12 J34) ...
        @inbounds for i_p1 in 1:length(Par), i_p2 in i_p1:length(Par), i_h1 in 1:length(Hol), i_h2 in i_h1:length(Hol)
            p_1, p_2, h_1, h_2 = Par[i_p1], Par[i_p2], Hol[i_h1], Hol[i_h2]
            e_p1, e_p2, e_h1, e_h2 = e(p_1), e(p_2), e(h_1), e(h_2)

            # Charge & parity of the configuration ...
            if (p_1[2] + p_2[2]) != (h_1[2] + h_2[2]) || parity(l(p_1) + l(p_2) + l(h_1) + l(h_2)) != P
                continue
            end

            # Single-particle resolvent R_c(w) ... & the normalization of equal orbitals ...
            Norm = 1.0 / sqrt(((p_1 == p_2) ? 2.0 : 1.0) * ((h_1 == h_2) ? 2.0 : 1.0))

            @inbounds for J12 in div(abs(j(p_1) - j(p_2)),2):div(j(p_1) + j(p_2),2), J34 in div(abs(j(h_1) - j(h_2)),2):div(j(h_1) + j(h_2),2)
                # Antisymmetry of equal orbitals (even J12 & J34 only) & the coupling to J ...
                if ((p_1 == p_2) && isodd(J12)) || ((h_1 == h_2) && isodd(J34)) || !triangle(2*J12,2*J34,2*J)
                    continue
                end

                Phase_h = - phase(div(j(h_1) + j(h_2),2) - J34)     # exchange of the hole pair ...
                Phase_p = - phase(div(j(p_1) + j(p_2),2) - J12)     # exchange of the particle pair ...

                # Column A_12[:,c] ... contact (spectator) * V terms ...
                @inbounds for i_ph in 1:N_ph
                    Orb_ph = Orb_1p1h_JP[i_ph]
                    p, h = (Orb_ph.a,Orb_ph.T), (Orb_ph.b,Orb_ph.T)

                    ME = 0.0
                    # Spectator hole ... P_h1h2[d_hh2 v_h1pp1p2] ...
                    if h == h_2
                        ME += C_alpha(j(p),j(h_2),j(h_1),J,J12,J34) * V(h_1,p,p_1,p_2,J12)
                    end
                    if h == h_1
                        ME += Phase_h * C_alpha(j(p),j(h_1),j(h_2),J,J12,J34) * V(h_2,p,p_1,p_2,J12)
                    end
                    # Spectator particle ... P_p1p2[d_pp2 v_hp1h1h2] ...
                    if p == p_2
                        ME += C_beta(j(p_2),j(h),j(p_1),J,J12,J34) * V(h,p_1,h_1,h_2,J34)
                    end
                    if p == p_1
                        ME += Phase_p * C_beta(j(p_1),j(h),j(p_2),J,J12,J34) * V(h,p_2,h_1,h_2,J34)
                    end

                    A_c[i_ph] = Norm * ME
                end

                # Sigma += A_12[:,c] R_c(w) A_12[:,c]^T ... (lower triangle)
                @inbounds for i_ph in 1:N_ph
                    Orb_ph = Orb_1p1h_JP[i_ph]
                    p, h = (Orb_ph.a,Orb_ph.T), (Orb_ph.b,Orb_ph.T)
                    E_p, E_h = e(p), e(h)

                    if A_c[i_ph] == 0.0
                        continue
                    end

                    @inbounds for i_qg in 1:i_ph
                        Orb_qg = Orb_1p1h_JP[i_qg]
                        q, g = (Orb_qg.a,Orb_qg.T), (Orb_qg.b,Orb_qg.T)
                        E_q, E_g = e(q), e(g)

                        w_ph = 0.0#E_p - E_h
                        w_qg = 0.0#E_q - E_g

                        R = 1.0 / (w_ph - (e_p1 + e_p2 - e_h1 - e_h2)) +#* (1.0 - exp(-((w_ph - (e_p1 + e_p2 - e_h1 - e_h2)) / gamma)^2)) + 
                            1.0 / (w_qg - (e_p1 + e_p2 - e_h1 - e_h2)) #* (1.0 - exp(-((w_qg - (e_p1 + e_p2 - e_h1 - e_h2)) / gamma)^2))
                        R *= 0.5

                        Sigma_JP[i_ph,i_qg] += A_c[i_ph] * R * A_c[i_qg]
                    end
                end
            end
        end

        # Symmetric self-energy matrix ...
        @inbounds for i_ph in 1:N_ph, i_qg in 1:(i_ph - 1)
            Sigma_JP[i_qg,i_ph] = Sigma_JP[i_ph,i_qg]
        end

        Sigma[P,J+1] = Sigma_JP
    end

    println("\t\t\tFinished on evaluation of HF 1p-1h to 2p-2h self-energy ...")

    return Sigma
end