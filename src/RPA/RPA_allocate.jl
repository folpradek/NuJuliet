function RPA_allocate(Params::Parameters,Orb::Vector{Orb1B},Orb_1p1h::Orb2B_1p1h,Orb_NN::Orb2B,h_N::O1B,V_NN::O2B,Rho::O1B,Sigma::O2B)
    # Read calculation parameters ...
    cB = Params.Calc.RPA.cB
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1

    JP_list = JP_initialize(J_max)
    
    println("\tAllocating RPA matrices A & B ...")

    # Initialize matrices A & B ...
    A = Matrix{Matrix{Float64}}(undef,2,J_max+1)
    B = Matrix{Matrix{Float64}}(undef,2,J_max+1)
    N = Matrix{Matrix{Float64}}(undef,2,J_max+1)
    M = Matrix{Matrix{Float64}}(undef,2,J_max+1)

    # Allocate matrices A & B ... no 2-body correlation function yet ...
    @time @inbounds Threads.@threads for JP in JP_list
        J, P = JP[1], JP[2]
        N_ph = Orb_1p1h.N[P,J+1]
        Orb_1p1h_JP = Orb_1p1h.Ind[P,J+1]

        A_JP = zeros(Float64,N_ph,N_ph)
        B_JP = zeros(Float64,N_ph,N_ph)
        N_JP = zeros(Float64,N_ph,N_ph)
        M_JP = zeros(Float64,N_ph,N_ph)

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
                dn_pq = n_q - n_p
                dn_gh = n_h - n_g

                delta_pq = Float64(kronecker_delta(a_p,a_q))
                delta_hg = Float64(kronecker_delta(a_h,a_g))
                delta_pg = Float64(kronecker_delta(a_p,a_g))
                delta_hq = Float64(kronecker_delta(a_h,a_q))

                P_pg = rem(l_p + l_g,2) + 1
                P_pq = rem(l_p + l_q,2) + 1
    
                ASum = 0.0
                BSum = 0.0
                NSum = 0.0
                MSum = 0.0

                # 1-body mean-field term h & the overlap matrix N ...
                if T_ph == T_qg
                    if T_ph == -1
                        h_pq, h_qg = h_N.p[a_p,a_q], h_N.p[a_h,a_g]

                        AME = 0.5 * (dn_ph + dn_qg) * (h_pq * delta_hg - h_qg * delta_pq)
                        ASum += AME

                        h_pg, h_hq = h_N.p[a_p,a_g], h_N.p[a_h,a_q]

                        BME = 0.5 * (dn_pq + dn_gh) * (delta_hq * h_hq - delta_pg * h_pg)
                        BSum += BME

                        NME = delta_pq * delta_hg * dn_ph
                        NSum += NME

                        MME = delta_pg * delta_hq * dn_ph
                        MSum += MME

                    elseif T_ph == 1
                        h_pq, h_qg = h_N.n[a_p,a_q], h_N.n[a_h,a_g]

                        AME = 0.5 * (dn_ph + dn_qg) * (h_pq * delta_hg - h_qg * delta_pq)
                        ASum += AME

                        h_pg, h_hq = h_N.n[a_p,a_g], h_N.n[a_h,a_q]

                        BME = 0.5 * (dn_pq + dn_gh) * (delta_hq * h_hq - delta_pg * h_pg)
                        BSum += BME

                        NME = delta_pq * delta_hg * dn_ph
                        NSum += NME

                        MME = delta_pg * delta_hq * dn_ph
                        MSum += MME
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

                BSum *= cB

                A_JP[i_ph,i_qg] = ASum
                B_JP[i_ph,i_qg] = BSum
                N_JP[i_ph,i_qg] = NSum
                M_JP[i_ph,i_qg] = MSum

                if i_ph != i_qg
                    A_JP[i_qg,i_ph] = ASum
                    B_JP[i_qg,i_ph] = BSum
                    N_JP[i_qg,i_ph] = NSum
                    M_JP[i_qg,i_ph] = MSum
                end
            end
        end

        # For given J & P allocate A & B ...
        A[P,J+1] = A_JP
        B[P,J+1] = B_JP
        N[P,J+1] = N_JP
        M[P,J+1] = M_JP
    end

    # If available & set to true, include 2-body correlation function effects ...
    if Params.Calc.RPA.TBCF == true && isfile("io/" * Params.Calc.Path * "/" * Params.Calc.RPA.Reference * "/bin/sigma.bin")
        @time A, B = RPA_allocate_TBCF(Params,Orb,Orb_1p1h,Orb_NN,V_NN,Sigma,A,B)
    end

    println("\tRPA matrices A, B, N & M allocated ...")

    return A, B, N, M
end

function RPA_allocate_TBCF(Params::Parameters,Orb::Vector{Orb1B},Orb_1p1h::Orb2B_1p1h,Orb_NN::Orb2B,V_NN::O2B,Sigma::O2B,A::Matrix{Matrix{Float64}},B::Matrix{Matrix{Float64}})
    # Read calculation parameters ...
    cB = Params.Calc.RPA.cB
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1
    a_max = div((N_max + 1)*(N_max + 2),2)

    JP_list = JP_initialize(J_max)
    
    println("\t\tComputing 2-body correlation function contributions to the RPA matrices A & B ...")

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

                P_pg = rem(l_p + l_g,2) + 1
                P_pq = rem(l_p + l_q,2) + 1

                P_hq = rem(l_h + l_q,2) + 1
                P_hg = rem(l_h + l_g,2) + 1
    
                ASum = 0.0
                BSum = 0.0

                # 2-body scattering TDA matrix A ...
                    # Hole-hole polarization diagram ...
                if T_ph == T_qg && T_ph != 0 && a_p == a_q && j_h == j_g
                    @inbounds for Orb_b in Orb
                        a_b, l_b, j_b = Orb_b. a, Orb_b.l, Orb_b.j
                        P_hb = rem(l_h + l_b,2) + 1

                        if P_hb != (rem(l_g + l_b,2) + 1)
                            continue
                        end

                        @inbounds for Orb_c in Orb
                            a_c, l_c, j_c = Orb_c.a, Orb_c.l, Orb_c.j
                        
                            @inbounds for Orb_d in Orb
                                a_d, l_d, j_d = Orb_d.a, Orb_d.l, Orb_d.j

                                if P_hb != (rem(l_c + l_d,2) + 1)
                                    continue
                                end

                                I_min, I_max = div(max(abs(j_h - j_b),abs(j_c - j_d)),2), div(min(j_h + j_b,j_c + j_d),2)
                                @inbounds for I in I_min:I_max
                                    Amp = 0.5 * hat(2*I) / (hat(j_h) * hat(2*J))

                                    if T_ph == -1
                                        ppME = 0.5 * Amp * (O2b_pp(a_g,a_b,a_c,a_d,I,P_hb,Sigma,Orb,Orb_NN) * O2b_pp(a_h,a_b,a_c,a_d,I,P_hb,V_NN,Orb,Orb_NN) + 
                                                            O2b_pp(a_h,a_b,a_c,a_d,I,P_hb,Sigma,Orb,Orb_NN) * O2b_pp(a_g,a_b,a_c,a_d,I,P_hb,V_NN,Orb,Orb_NN))

                                        pnME = Amp * (O2b_pn(a_g,a_b,a_c,a_d,I,P_hb,Sigma,Orb_NN) * O2b_pn(a_h,a_b,a_c,a_d,I,P_hb,V_NN,Orb_NN) + 
                                                      O2b_pn(a_h,a_b,a_c,a_d,I,P_hb,Sigma,Orb_NN) * O2b_pn(a_g,a_b,a_c,a_d,I,P_hb,V_NN,Orb_NN))

                                        ASum += ppME + pnME

                                    elseif T_ph == 1
                                        nnME = 0.5 * Amp * (O2b_nn(a_g,a_b,a_c,a_d,I,P_hb,Sigma,Orb,Orb_NN) * O2b_nn(a_h,a_b,a_c,a_d,I,P_hb,V_NN,Orb,Orb_NN) + 
                                                            O2b_nn(a_h,a_b,a_c,a_d,I,P_hb,Sigma,Orb,Orb_NN) * O2b_nn(a_g,a_b,a_c,a_d,I,P_hb,V_NN,Orb,Orb_NN))

                                        pnME = phase(1 + div(j_h + j_g,2)) * Amp * (O2b_pn(a_b,a_g,a_c,a_d,I,P_hb,Sigma,Orb_NN) * O2b_pn(a_b,a_h,a_c,a_d,I,P_hb,V_NN,Orb_NN) + 
                                                                                O2b_pn(a_b,a_h,a_c,a_d,I,P_hb,Sigma,Orb_NN) * O2b_pn(a_b,a_g,a_c,a_d,I,P_hb,V_NN,Orb_NN))

                                        ASum += nnME + pnME
                                    end
                                end
                            end
                        end

                    end
                end
                    # Particle-particle polarization diagram ...
                if T_ph == T_qg && T_ph != 0 && a_h == a_g && j_p == j_q
                    @inbounds for Orb_b in Orb
                        a_b, l_b, j_b = Orb_b.a, Orb_b.l, Orb_b.j
                        P_pb = rem(l_p + l_b,2) + 1

                        if P_pb != (rem(l_q + l_b,2) + 1)
                            continue
                        end

                        @inbounds for Orb_c in Orb
                            a_c, l_c, j_c = Orb_c.a, Orb_c.l, Orb_c.j
                        
                            @inbounds for Orb_d in Orb
                                a_d, l_d, j_d = Orb_d.a, Orb_d.l, Orb_d.j

                                if P_pb != (rem(l_c + l_d,2) + 1)
                                    continue
                                end

                                I_min, I_max = div(max(abs(j_p - j_b), abs(j_c - j_d)),2), div(min(j_p + j_b, j_c + j_d),2)
                                @inbounds for I in I_min:I_max
                                    Amp = 0.5 * hat(2*I) / (hat(j_p) * hat(2*J))

                                    if T_ph == -1
                                        ppME = 0.5 * Amp * (O2b_pp(a_q,a_b,a_c,a_d,I,P_pb,Sigma,Orb,Orb_NN) * O2b_pp(a_p,a_b,a_c,a_d,I,P_pb,V_NN,Orb,Orb_NN) + 
                                                            O2b_pp(a_p,a_b,a_c,a_d,I,P_pb,Sigma,Orb,Orb_NN) * O2b_pp(a_q,a_b,a_c,a_d,I,P_pb,V_NN,Orb,Orb_NN))

                                        pnME = Amp * (O2b_pn(a_q,a_b,a_c,a_d,I,P_pb,Sigma,Orb_NN) * O2b_pn(a_p,a_b,a_c,a_d,I,P_pb,V_NN,Orb_NN) + 
                                                      O2b_pn(a_p,a_b,a_c,a_d,I,P_pb,Sigma,Orb_NN) * O2b_pn(a_q,a_b,a_c,a_d,I,P_pb,V_NN,Orb_NN))

                                        ASum += ppME + pnME

                                    elseif T_ph == 1
                                        nnME = 0.5 * Amp * (O2b_nn(a_q,a_b,a_c,a_d,I,P_pb,Sigma,Orb,Orb_NN) * O2b_nn(a_p,a_b,a_c,a_d,I,P_pb,V_NN,Orb,Orb_NN) + 
                                                            O2b_nn(a_p,a_b,a_c,a_d,I,P_pb,Sigma,Orb,Orb_NN) * O2b_nn(a_q,a_b,a_c,a_d,I,P_pb,V_NN,Orb,Orb_NN))

                                        pnME = phase(1 + div(j_h + j_g,2)) * Amp * (O2b_pn(a_b,a_q,a_c,a_d,I,P_pb,Sigma,Orb_NN) * O2b_pn(a_b,a_p,a_c,a_d,I,P_pb,V_NN,Orb_NN) + 
                                                                                O2b_pn(a_b,a_p,a_c,a_d,I,P_pb,Sigma,Orb_NN) * O2b_pn(a_b,a_q,a_c,a_d,I,P_pb,V_NN,Orb_NN))

                                        ASum += nnME + pnME
                                    end
                                end
                            end
                        end

                    end
                end
                    # Box diagram 1
                if T_ph == T_qg
                    @inbounds for Orb_b in Orb
                        a_b, l_b, j_b = Orb_b.a, Orb_b.l, Orb_b.j
                        P_pb = rem(l_p + l_b,2) + 1
                        P_hb = rem(l_h + l_b,2) + 1

                        @inbounds for Orb_d in Orb
                            a_d, l_d, j_d = Orb_d.a, Orb_d.l, Orb_d.j

                            if (rem(l_g + l_d,2) + 1 != P_hb) || (rem(l_q + l_d,2) + 1 != P_pb)
                                continue
                            end

                            J_1_min, J_1_max = div(max(abs(j_p - j_b),abs(j_q - j_d)),2), div(min(j_p + j_b, j_q + j_d),2)
                            J_2_min, J_2_max = div(max(abs(j_h - j_b),abs(j_g - j_d)),2), div(min(j_h + j_b, j_g + j_d),2)
                            I_min = div(max(abs(j_p - j_q), abs(j_h - j_g), abs(j_b - j_d)),2)
                            I_max = div(min(j_p + j_q, j_h + j_g, j_b + j_d),2)

                            @inbounds for I in I_min:I_max
                                Amp_I = phase(1 + I + div(j_h + j_g,2)) * w6j(j_p,j_q,2*I,j_g,j_h,2*J)

                                @inbounds for J_1 in J_1_min:J_1_max
                                    Amp_J_1 = hat(2*J_1) * phase(J_1) * w6j(j_p,j_q,2*I,j_d,j_b,2*J_1)

                                    @inbounds for J_2 in J_2_min:J_2_max
                                        Amp_J_2 = hat(2*J_2) * phase(J_2) * w6j(j_h,j_g,2*I,j_d,j_b,2*J_2)

                                        Amp = Amp_I * Amp_J_1 * Amp_J_2

                                        if abs(Amp) < 1e-8
                                            continue
                                        end

                                        if T_ph == -1
                                            ppME = Amp * (O2b_pp(a_h,a_b,a_g,a_d,J_2,P_hb,Sigma,Orb,Orb_NN) * O2b_pp(a_p,a_b,a_q,a_d,J_1,P_pb,V_NN,Orb,Orb_NN) +
                                                            O2b_pp(a_p,a_b,a_q,a_d,J_1,P_pb,Sigma,Orb,Orb_NN) * O2b_pp(a_h,a_b,a_g,a_d,J_2,P_hb,V_NN,Orb,Orb_NN))

                                            pnME = Amp * (O2b_pn(a_h,a_b,a_g,a_d,J_2,P_hb,Sigma,Orb_NN) * O2b_pn(a_p,a_b,a_q,a_d,J_1,P_pb,V_NN,Orb_NN) +
                                                            O2b_pn(a_p,a_b,a_q,a_d,J_1,P_pb,Sigma,Orb_NN) * O2b_pn(a_h,a_b,a_g,a_d,J_2,P_hb,V_NN,Orb_NN))

                                            ASum += ppME + pnME
                                        end

                                        if T_ph == 1
                                            nnME = Amp * (O2b_nn(a_h,a_b,a_g,a_d,J_2,P_hb,Sigma,Orb,Orb_NN) * O2b_nn(a_p,a_b,a_q,a_d,J_1,P_pb,V_NN,Orb,Orb_NN) +
                                                            O2b_nn(a_p,a_b,a_q,a_d,J_1,P_pb,Sigma,Orb,Orb_NN) * O2b_nn(a_h,a_b,a_g,a_d,J_2,P_hb,V_NN,Orb,Orb_NN))

                                            pnME = Amp * (O2b_pn(a_b,a_h,a_d,a_g,J_2,P_hb,Sigma,Orb_NN) * O2b_pn(a_b,a_p,a_d,a_q,J_1,P_pb,V_NN,Orb_NN) +
                                                            O2b_pn(a_b,a_p,a_d,a_q,J_1,P_pb,Sigma,Orb_NN) * O2b_pn(a_b,a_h,a_d,a_g,J_2,P_hb,V_NN,Orb_NN)) * phase(div(j_p + j_h + j_q + j_g,2))

                                            ASum += nnME + pnME
                                        end
                                    end
                                end
                            end
                        end
                    end
                else
                    @inbounds for Orb_b in Orb
                        a_b, l_b, j_b = Orb_b.a, Orb_b.l, Orb_b.j
                        P_pb = rem(l_p + l_b,2) + 1
                        P_hb = rem(l_h + l_b,2) + 1

                        @inbounds for Orb_c in Orb
                            a_c, l_c, j_c = Orb_c.a, Orb_c.l, Orb_c.j

                            if (rem(l_g + l_c,2) + 1 != P_hb) || (rem(l_q + l_c,2) + 1 != P_pb)
                                continue
                            end

                            J_1_min, J_1_max = div(max(abs(j_p - j_b),abs(j_q - j_c)),2), div(min(j_p + j_b, j_q + j_c),2)
                            J_2_min, J_2_max = div(max(abs(j_h - j_b),abs(j_g - j_c)),2), div(min(j_h + j_b, j_g + j_c),2)
                            I_min = div(max(abs(j_p - j_q), abs(j_h - j_g), abs(j_b - j_c)),2)
                            I_max = div(min(j_p + j_q, j_h + j_g, j_b + j_c),2)

                            @inbounds for I in I_min:I_max
                                Amp_I = phase(1 + I + div(j_h + j_q,2)) * w6j(j_p,j_q,2*I,j_g,j_h,2*J)

                                @inbounds for J_1 in J_1_min:J_1_max
                                    Amp_J_1 = hat(2*J_1) * w6j(j_p,j_q,2*I,j_c,j_b,2*J_1)

                                    @inbounds for J_2 in J_2_min:J_2_max
                                        Amp_J_2 = hat(2*J_2) * w6j(j_h,j_g,2*I,j_c,j_b,2*J_2)

                                        Amp = Amp_I * Amp_J_1 * Amp_J_2

                                        if abs(Amp) < 1e-8
                                            continue
                                        end

                                        pnME = Amp * (O2b_pn(a_h,a_b,a_c,a_g,J_2,P_hb,Sigma,Orb_NN) * O2b_pn(a_p,a_b,a_c,a_q,J_1,P_pb,V_NN,Orb_NN) +
                                                      O2b_pn(a_p,a_b,a_c,a_q,J_1,P_pb,Sigma,Orb_NN) * O2b_pn(a_h,a_b,a_c,a_g,J_2,P_hb,V_NN,Orb_NN))

                                        ASum += pnME
                                    end
                                end
                            end
                        end
                    end
                end
                    # Box diagram 2
                if P_pg == P_hq
                    @inbounds for Orb_c in Orb
                        a_c, l_c, j_c = Orb_c.a, Orb_c.l, Orb_c.j

                        @inbounds for Orb_d in Orb
                            a_d, l_d, j_d = Orb_d.a, Orb_d.l, Orb_d.j

                            if (rem(l_c + l_d,2) + 1) != P_pg
                                continue
                            end

                            I_min = div(max(abs(j_p - j_g), abs(j_q - j_h), abs(j_c - j_d)),2)
                            I_max = div(min(j_p + j_g, j_q + j_h, j_c + j_d),2)

                            @inbounds for I in I_min:I_max
                                Amp_I = phase(I + div(j_h + j_q,2)) * hat(2*I) / hat(2*J) * w6j(j_p,j_h,2*J,j_q,j_g,2*I)

                                if T_ph == -1 && T_qg == -1
                                    ppME = Amp_I * (O2b_pp(a_p,a_g,a_c,a_d,I,P_pg,V_NN,Orb,Orb_NN) * O2b_pp(a_h,a_q,a_c,a_d,I,P_pg,Sigma,Orb,Orb_NN) + 
                                                    O2b_pp(a_h,a_q,a_c,a_d,I,P_pg,V_NN,Orb,Orb_NN) * O2b_pp(a_p,a_g,a_c,a_d,I,P_pg,Sigma,Orb,Orb_NN)) * 0.5

                                    ASum += ppME
                                end

                                if T_ph == -1 && T_qg == 1
                                    ppME = Amp_I * (O2b_pn(a_p,a_g,a_c,a_d,I,P_pg,V_NN,Orb_NN) * O2b_pn(a_h,a_q,a_c,a_d,I,P_pg,Sigma,Orb_NN) + 
                                                    O2b_pn(a_h,a_q,a_c,a_d,I,P_pg,V_NN,Orb_NN) * O2b_pn(a_p,a_g,a_c,a_d,I,P_pg,Sigma,Orb_NN))

                                    ASum += ppME
                                end

                                if T_ph == 1 && T_qg == -1
                                    npME = phase(div(j_p + j_h + j_q + j_g,2)) * Amp_I * (O2b_pn(a_q,a_h,a_c,a_d,I,P_pg,V_NN,Orb_NN) * O2b_pn(a_g,a_p,a_c,a_d,I,P_pg,Sigma,Orb_NN) +
                                                                                          O2b_pn(a_g,a_p,a_c,a_d,I,P_pg,V_NN,Orb_NN) * O2b_pn(a_q,a_h,a_c,a_d,I,P_pg,Sigma,Orb_NN))

                                    ASum += npME
                                end

                                if T_ph == 1 && T_qg == 1
                                    ppME = Amp_I * (O2b_nn(a_p,a_g,a_c,a_d,I,P_pg,V_NN,Orb,Orb_NN) * O2b_nn(a_h,a_q,a_c,a_d,I,P_pg,Sigma,Orb,Orb_NN) + 
                                                    O2b_nn(a_h,a_q,a_c,a_d,I,P_pg,V_NN,Orb,Orb_NN) * O2b_nn(a_p,a_g,a_c,a_d,I,P_pg,Sigma,Orb,Orb_NN)) * 0.5

                                    ASum += ppME
                                end
                            end
                        end
                    end
                end
                # 2-body correlation matrix B ...
                    # Box diagram 1
                if T_ph == T_qg
                    @inbounds for Orb_b in Orb
                        a_b, l_b, j_b = Orb_b.a, Orb_b.l, Orb_b.j
                        P_pb = rem(l_p + l_b,2) + 1
                        P_hb = rem(l_h + l_b,2) + 1

                        @inbounds for Orb_d in Orb
                            a_d, l_d, j_d = Orb_d.a, Orb_d.l, Orb_d.j

                            if (rem(l_q + l_d,2) + 1 != P_hb) || (rem(l_g + l_d,2) + 1 != P_pb)
                                continue
                            end

                            J_1_min, J_1_max = div(max(abs(j_p - j_b),abs(j_g - j_d)),2), div(min(j_p + j_b, j_g + j_d),2)
                            J_2_min, J_2_max = div(max(abs(j_h - j_b),abs(j_q - j_d)),2), div(min(j_h + j_b, j_q + j_d),2)
                            I_min = div(max(abs(j_p - j_g), abs(j_h - j_q), abs(j_b - j_d)),2)
                            I_max = div(min(j_p + j_g, j_h + j_q, j_b + j_d),2)

                            @inbounds for I in I_min:I_max
                                Amp_I = phase(1 + I + div(j_h + j_g,2)) * w6j(j_p,j_g,2*I,j_q,j_h,2*J)

                                @inbounds for J_1 in J_1_min:J_1_max
                                    Amp_J_1 = hat(2*J_1) * w6j(j_p,j_g,2*I,j_d,j_b,2*J_1)

                                    @inbounds for J_2 in J_2_min:J_2_max
                                        Amp_J_2 = hat(2*J_2) * w6j(j_h,j_q,2*I,j_d,j_b,2*J_2)

                                        Amp = Amp_I * Amp_J_1 * Amp_J_2

                                        if abs(Amp) < 1e-8
                                            continue
                                        end

                                        if T_ph == -1
                                            ppME = Amp * (O2b_pp(a_h,a_b,a_q,a_d,J_2,P_hb,Sigma,Orb,Orb_NN) * O2b_pp(a_p,a_b,a_g,a_d,J_1,P_pb,V_NN,Orb,Orb_NN) +
                                                            O2b_pp(a_p,a_b,a_g,a_d,J_1,P_pb,Sigma,Orb,Orb_NN) * O2b_pp(a_h,a_b,a_q,a_d,J_2,P_hb,V_NN,Orb,Orb_NN))

                                            pnME = Amp * (O2b_pn(a_h,a_b,a_q,a_d,J_2,P_hb,Sigma,Orb_NN) * O2b_pn(a_p,a_b,a_g,a_d,J_1,P_pb,V_NN,Orb_NN) +
                                                            O2b_pn(a_p,a_b,a_g,a_d,J_1,P_pb,Sigma,Orb_NN) * O2b_pn(a_h,a_b,a_q,a_d,J_2,P_hb,V_NN,Orb_NN))

                                            BSum += ppME + pnME
                                        end

                                        if T_ph == 1
                                            nnME = Amp * (O2b_nn(a_h,a_b,a_q,a_d,J_2,P_hb,Sigma,Orb,Orb_NN) * O2b_nn(a_p,a_b,a_g,a_d,J_1,P_pb,V_NN,Orb,Orb_NN) +
                                                            O2b_nn(a_p,a_b,a_g,a_d,J_1,P_pb,Sigma,Orb,Orb_NN) * O2b_nn(a_h,a_b,a_q,a_d,J_2,P_hb,V_NN,Orb,Orb_NN))

                                            pnME = Amp * (O2b_pn(a_b,a_h,a_d,a_q,J_2,P_hb,Sigma,Orb_NN) * O2b_pn(a_b,a_p,a_d,a_g,J_1,P_pb,V_NN,Orb_NN) +
                                                            O2b_pn(a_b,a_p,a_d,a_g,J_1,P_pb,Sigma,Orb_NN) * O2b_pn(a_b,a_h,a_d,a_q,J_2,P_hb,V_NN,Orb_NN)) * phase(div(j_p + j_h + j_q + j_g,2))

                                            BSum += nnME + pnME
                                        end
                                    end
                                end
                            end
                        end
                    end
                else
                    @inbounds for Orb_b in Orb
                        a_b, l_b, j_b = Orb_b.a, Orb_b.l, Orb_b.j
                        P_pb = rem(l_p + l_b,2) + 1
                        P_hb = rem(l_h + l_b,2) + 1

                        @inbounds for Orb_c in Orb
                            a_c, l_c, j_c = Orb_c.a, Orb_c.l, Orb_c.j

                            if (rem(l_q + l_c,2) + 1 != P_hb) || (rem(l_g + l_c,2) + 1 != P_pb)
                                continue
                            end

                            J_1_min, J_1_max = div(max(abs(j_p - j_b),abs(j_g - j_c)),2), div(min(j_p + j_b, j_g + j_c),2)
                            J_2_min, J_2_max = div(max(abs(j_h - j_b),abs(j_q - j_c)),2), div(min(j_h + j_b, j_q + j_c),2)
                            I_min = div(max(abs(j_p - j_g), abs(j_h - j_q), abs(j_b - j_c)),2)
                            I_max = div(min(j_p + j_g, j_h + j_q, j_b + j_c),2)

                            @inbounds for I in I_min:I_max
                                Amp_I = phase(1 + I + div(j_h + j_q,2)) * w6j(j_p,j_g,2*I,j_q,j_h,2*J)

                                @inbounds for J_1 in J_1_min:J_1_max
                                    Amp_J_1 = hat(2*J_1) * w6j(j_p,j_g,2*I,j_c,j_b,2*J_1)

                                    @inbounds for J_2 in J_2_min:J_2_max
                                        Amp_J_2 = hat(2*J_2) * w6j(j_h,j_q,2*I,j_c,j_b,2*J_2)

                                        Amp = Amp_I * Amp_J_1 * Amp_J_2

                                        if abs(Amp) < 1e-8
                                            continue
                                        end

                                        pnME = Amp * (O2b_pn(a_h,a_b,a_c,a_q,J_2,P_hb,Sigma,Orb_NN) * O2b_pn(a_p,a_b,a_c,a_g,J_1,P_pb,V_NN,Orb_NN) +
                                                        O2b_pn(a_p,a_b,a_c,a_g,J_1,P_pb,Sigma,Orb_NN) * O2b_pn(a_h,a_b,a_c,a_q,J_2,P_hb,V_NN,Orb_NN))

                                        BSum += pnME
                                    end
                                end
                            end
                        end
                    end
                end
                    # Box diagram 2
                if P_pq == P_hg
                    @inbounds for Orb_c in Orb
                        a_c, l_c, j_c = Orb_c.a, Orb_c.l, Orb_c.j

                        @inbounds for Orb_d in Orb
                            a_d, l_d, j_d = Orb_d.a, Orb_d.l, Orb_d.j

                            if (rem(l_c + l_d,2) + 1) != P_pq
                                continue
                            end

                            I_min = div(max(abs(j_p - j_q), abs(j_h - j_g), abs(j_c - j_d)),2)
                            I_max = div(min(j_p + j_q, j_h + j_g, j_c + j_d),2)
                            @inbounds for I in I_min:I_max
                                Amp_I = phase(J + I + div(j_h + j_q,2)) * hat(2*I) / hat(2*J) * w6j(j_p,j_h,2*J,j_g,j_q,2*I)

                                if T_ph == -1 && T_qg == -1
                                    ME = 0.5 * Amp_I * (O2b_pp(a_h,a_g,a_c,a_d,I,P_pq,Sigma,Orb,Orb_NN) * O2b_pp(a_p,a_q,a_c,a_d,I,P_pq,V_NN,Orb,Orb_NN) + 
                                                        O2b_pp(a_p,a_q,a_c,a_d,I,P_pq,Sigma,Orb,Orb_NN) * O2b_pp(a_h,a_g,a_c,a_d,I,P_pq,V_NN,Orb,Orb_NN))

                                    BSum += ME
                                end

                                if T_ph == -1 && T_qg == 1
                                    ME = Amp_I * (O2b_pn(a_h,a_g,a_c,a_d,I,P_pq,Sigma,Orb_NN) * O2b_pn(a_p,a_q,a_c,a_d,I,P_pq,V_NN,Orb_NN) + 
                                                  O2b_pn(a_p,a_q,a_c,a_d,I,P_pq,Sigma,Orb_NN) * O2b_pn(a_h,a_g,a_c,a_d,I,P_pq,V_NN,Orb_NN))

                                    BSum += ME
                                end

                                if T_ph == 1 && T_qg == -1
                                    Amp_I_np = phase(J + I + div(j_g + j_p,2)) * hat(2*I) / hat(2*J) * w6j(j_q,j_g,2*J,j_h,j_p,2*I)

                                    ME = Amp_I_np * (O2b_pn(a_g,a_h,a_c,a_d,I,P_pq,Sigma,Orb_NN) * O2b_pn(a_q,a_p,a_c,a_d,I,P_pq,V_NN,Orb_NN) +
                                                     O2b_pn(a_q,a_p,a_c,a_d,I,P_pq,Sigma,Orb_NN) * O2b_pn(a_g,a_h,a_c,a_d,I,P_pq,V_NN,Orb_NN))

                                    BSum += ME
                                end

                                if T_ph == 1 && T_qg == 1
                                    ME = 0.5 * Amp_I * (O2b_nn(a_h,a_g,a_c,a_d,I,P_pq,Sigma,Orb,Orb_NN) * O2b_nn(a_p,a_q,a_c,a_d,I,P_pq,V_NN,Orb,Orb_NN) + 
                                                        O2b_nn(a_p,a_q,a_c,a_d,I,P_pq,Sigma,Orb,Orb_NN) * O2b_nn(a_h,a_g,a_c,a_d,I,P_pq,V_NN,Orb,Orb_NN))

                                    BSum += ME
                                end
                            end
                        end
                    end
                end
                    # Hole-particle polarization diagram ...
                if T_ph == T_qg && T_ph != 0 && a_p == a_g && j_h == j_q
                    @inbounds for Orb_c in Orb
                        a_c, l_c, j_c = Orb_c.a, Orb_c.l, Orb_c.j
                        P_hc = rem(l_h + l_c,2) + 1

                        if P_hc != (rem(l_q + l_c,2) + 1)
                            continue
                        end

                        @inbounds for Orb_b in Orb
                            a_b, l_b, j_b = Orb_b.a, Orb_b.l, Orb_b.j
                        
                            @inbounds for Orb_a in Orb
                                a_a, l_a, j_a = Orb_a.a, Orb_a.l, Orb_a.j

                                if P_hc != (rem(l_a + l_b,2) + 1)
                                    continue
                                end

                                I_min, I_max = div(max(abs(j_h - j_c), abs(j_a - j_b)),2), div(min(j_h + j_c, j_a + j_b),2)
                                @inbounds for I in I_min:I_max
                                    Amp = 0.5 * hat(2*I) / (hat(j_h) * hat(2*J)) * phase(1 - J + div(j_p + j_h,2))

                                    if T_ph == -1
                                        ppME = 0.5 * Amp * (O2b_pp(a_a,a_b,a_c,a_q,I,P_hc,Sigma,Orb,Orb_NN) * O2b_pp(a_a,a_b,a_c,a_h,I,P_hc,V_NN,Orb,Orb_NN) + 
                                                            O2b_pp(a_a,a_b,a_c,a_h,I,P_hc,Sigma,Orb,Orb_NN) * O2b_pp(a_a,a_b,a_c,a_q,I,P_hc,V_NN,Orb,Orb_NN))

                                        pnME = phase(1 + div(j_h + j_q,2)) * Amp * (O2b_pn(a_a,a_b,a_q,a_c,I,P_hc,Sigma,Orb_NN) * O2b_pn(a_a,a_b,a_h,a_c,I,P_hc,V_NN,Orb_NN) + 
                                                                                    O2b_pn(a_a,a_b,a_h,a_c,I,P_hc,Sigma,Orb_NN) * O2b_pn(a_a,a_b,a_q,a_c,I,P_hc,V_NN,Orb_NN))

                                        BSum += ppME + pnME

                                    elseif T_ph == 1
                                        nnME = 0.5 * Amp * (O2b_nn(a_a,a_b,a_c,a_q,I,P_hc,Sigma,Orb,Orb_NN) * O2b_nn(a_a,a_b,a_c,a_h,I,P_hc,V_NN,Orb,Orb_NN) + 
                                                            O2b_nn(a_a,a_b,a_c,a_h,I,P_hc,Sigma,Orb,Orb_NN) * O2b_nn(a_a,a_b,a_c,a_q,I,P_hc,V_NN,Orb,Orb_NN))

                                        pnME = Amp * (O2b_pn(a_a,a_b,a_c,a_q,I,P_hc,Sigma,Orb_NN) * O2b_pn(a_a,a_b,a_c,a_h,I,P_hc,V_NN,Orb_NN) + 
                                                      O2b_pn(a_a,a_b,a_c,a_h,I,P_hc,Sigma,Orb_NN) * O2b_pn(a_a,a_b,a_c,a_q,I,P_hc,V_NN,Orb_NN))

                                        BSum += nnME + pnME
                                    end
                                end
                            end
                        end

                    end
                end
                    # Particle-hole polarization diagram ...
                if T_ph == T_qg && T_ph != 0 && a_h == a_q && j_p == j_g
                    @inbounds for Orb_c in Orb
                        a_c, l_c, j_c = Orb_c.a, Orb_c.l, Orb_c.j
                        P_pc = rem(l_p + l_c,2) + 1

                        if P_pc != (rem(l_g + l_c,2) + 1)
                            continue
                        end

                        @inbounds for Orb_b in Orb
                            a_b, l_b, j_b = Orb_b.a, Orb_b.l, Orb_b.j
                        
                            @inbounds for Orb_a in Orb
                                a_a, l_a, j_a = Orb_a.a, Orb_a.l, Orb_a.j

                                if P_pc != (rem(l_a + l_b,2) + 1)
                                    continue
                                end

                                I_min, I_max = div(max(abs(j_p - j_c), abs(j_a - j_b)),2), div(min(j_p + j_c, j_a + j_b),2)
                                @inbounds for I in I_min:I_max
                                    Amp = 0.5 * hat(2*I) / (hat(j_p) * hat(2*J)) * phase(1 - J + div(j_p + j_h,2))

                                    if T_ph == -1
                                        ppME = 0.5 * Amp * (O2b_pp(a_a,a_b,a_c,a_q,I,P_pc,Sigma,Orb,Orb_NN) * O2b_pp(a_a,a_b,a_c,a_p,I,P_pc,V_NN,Orb,Orb_NN) + 
                                                            O2b_pp(a_a,a_b,a_c,a_p,I,P_pc,Sigma,Orb,Orb_NN) * O2b_pp(a_a,a_b,a_c,a_q,I,P_pc,V_NN,Orb,Orb_NN))

                                        pnME = phase(1 + div(j_p + j_g,2)) * Amp * (O2b_pn(a_a,a_b,a_q,a_c,I,P_pc,Sigma,Orb_NN) * O2b_pn(a_p,a_b,a_p,a_c,I,P_pc,V_NN,Orb_NN) + 
                                                                                    O2b_pn(a_a,a_b,a_p,a_c,I,P_pc,Sigma,Orb_NN) * O2b_pn(a_q,a_b,a_q,a_c,I,P_pc,V_NN,Orb_NN))

                                        BSum += ppME + pnME

                                    elseif T_ph == 1
                                        nnME = 0.5 * Amp * (O2b_nn(a_a,a_b,a_c,a_q,I,P_pc,Sigma,Orb,Orb_NN) * O2b_nn(a_a,a_b,a_c,a_p,I,P_pc,V_NN,Orb,Orb_NN) + 
                                                            O2b_nn(a_a,a_b,a_c,a_p,I,P_pc,Sigma,Orb,Orb_NN) * O2b_nn(a_a,a_b,a_c,a_q,I,P_pc,V_NN,Orb,Orb_NN))

                                        pnME = Amp * (O2b_pn(a_a,a_b,a_c,a_q,I,P_pc,Sigma,Orb_NN) * O2b_pn(a_a,a_b,a_c,a_p,I,P_pc,V_NN,Orb_NN) + 
                                                      O2b_pn(a_a,a_b,a_c,a_p,I,P_pc,Sigma,Orb_NN) * O2b_pn(a_a,a_b,a_c,a_q,I,P_pc,V_NN,Orb_NN))

                                        BSum += nnME + pnME
                                    end
                                end
                            end
                        end

                    end
                end

                BSum *= cB

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