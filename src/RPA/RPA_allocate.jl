function RPA_allocate(Params::Parameters,Orb::Vector{Orb1B},Orb_1p1h::Orb2B_1p1h,Orb_NN::Orb2B,h_N::O1B,V_NN::O2B,Rho::O1B,Sigma::O2B)
    # Read calculation parameters ...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1

    JP_list = JP_initialize(J_max)
    
    println("\tAllocating RPA matrices A & B ...")

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

                P_pg = rem(l_p + l_g,2) + 1
                P_pq = rem(l_p + l_q,2) + 1
    
                ASum = 0.0
                BSum = 0.0
                NSum = 0.0

                # 1-body mean-field term h & the overlap matrix N ...
                if T_ph == T_qg
                    if T_ph == -1
                        h_pq, h_qg = h_N.p[a_p,a_q], h_N.p[a_h,a_g]

                        Amp = 0.5 * (dn_ph + dn_qg) * (h_pq * delta_hg - h_qg * delta_pq)
                        ASum += Amp

                        NME = dn_ph * delta_pq * delta_hg
                        NSum += NME
                    elseif T_ph == 1
                        h_pq, h_qg = h_N.n[a_p,a_q], h_N.n[a_h,a_g]

                        Amp = 0.5 * (dn_ph + dn_qg) * (h_pq * delta_hg - h_qg * delta_pq)
                        ASum += Amp

                        NME = dn_ph * delta_pq * delta_hg
                        NSum += NME
                    end
                end

                # 2-body TDA matrix A ...
                    # 1-body scattering ...
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

                # Move to another function ...

                #=
                    # 2-body scattering
                        # particle-particle 2p-2h scattering ...
                if t_ph == t_qg && a_h == a_g && j_p == j_q
                    # Case of proton-proton contribution ...
                    if t_ph == -1
                        @inbounds for b in Particle.p
                            a_b, l_b, j_b = b.a, b.l, b.j
                            P_pb = rem(l_p + l_b,2) + 1

                            if P_pb != (rem(l_q + l_b,2) + 1)
                                continue
                            end

                            @inbounds for c in Hole.p
                                a_c, l_c, j_c = c.a, c.l, c.j
                            
                                @inbounds for d in Hole.p
                                    a_d, l_d, j_d = d.a, d.l, d.j

                                    if P_pb != (rem(l_c + l_d,2) + 1)
                                        continue
                                    end

                                    @inbounds for I in div(max(abs(j_p - j_b),abs(j_c - j_d)),2):div(min(j_p + j_b,j_c + j_d),2)
                                        Amp = 0.25 * Float64((2*J + 1)) / Float64(j_p + 1) / Float64(2*J + 1)

                                        if I != J
                                            continue
                                        end

                                        ME = Amp * (O2b_pp(a_p,a_b,a_c,a_d,I,P_pb,V_NN,Orb,Orb_NN) * O2b_pp(a_q,a_b,a_c,a_d,I,P_pb,Sigma_NN,Orb,Orb_NN) + 
                                                    O2b_pp(a_q,a_b,a_c,a_d,I,P_pb,V_NN,Orb,Orb_NN) * O2b_pp(a_p,a_b,a_c,a_d,I,P_pb,Sigma_NN,Orb,Orb_NN))
                                        ASum += ME
                                    end
                                end
                            end

                        end

                        @inbounds for b in Particle.n
                            a_b, l_b, j_b = b.a, b.l, b.j
                            P_pb = rem(l_p + l_b,2) + 1

                            if P_pb != (rem(l_q + l_b,2) + 1)
                                continue
                            end

                            @inbounds for c in Hole.p
                                a_c, l_c, j_c = c.a, c.l, c.j
                            
                                @inbounds for d in Hole.n
                                    a_d, l_d, j_d = d.a, d.l, d.j

                                    if P_pb != (rem(l_c + l_d,2) + 1)
                                        continue
                                    end

                                    @inbounds for I in div(max(abs(j_p - j_b),abs(j_c - j_d)),2):div(min(j_p + j_b,j_c + j_d),2)
                                        Amp = 0.5 * Float64((2*J + 1)) / Float64(j_p + 1) / Float64(2*J + 1)

                                        if I != J
                                            continue
                                        end

                                        ME = Amp * (O2b_pn(a_p,a_b,a_c,a_d,I,P_pb,V_NN,Orb_NN) * O2b_pn(a_q,a_b,a_c,a_d,I,P_pb,Sigma_NN,Orb_NN) + 
                                                    O2b_pn(a_q,a_b,a_c,a_d,I,P_pb,V_NN,Orb_NN) * O2b_pn(a_p,a_b,a_c,a_d,I,P_pb,Sigma_NN,Orb_NN))
                                        ASum += ME
                                    end
                                end
                            end

                        end

                    end

                    # Case of neutron-neutron contribution ...
                    if t_ph == 1
                        @inbounds for b in Particle.n
                            a_b, l_b, j_b = b.a, b.l, b.j
                            P_pb = rem(l_p + l_b,2) + 1

                            if P_pb != (rem(l_q + l_b,2) + 1)
                                continue
                            end

                            @inbounds for c in Hole.n
                                a_c, l_c, j_c = c.a, c.l, c.j
                            
                                @inbounds for d in Hole.n
                                    a_d, l_d, j_d = d.a, d.l, d.j

                                    if P_pb != (rem(l_c + l_d,2) + 1)
                                        continue
                                    end

                                    @inbounds for I in div(max(abs(j_p - j_b), abs(j_c - j_d)),2):div(min(j_p + j_b, j_c + j_d),2)
                                        Amp = 0.25 * Float64((2*J + 1)) / Float64(j_p + 1) / Float64(2*J + 1)

                                        if I != J
                                            continue
                                        end

                                        ME = Amp * (O2b_nn(a_p,a_b,a_c,a_d,I,P_pb,V_NN,Orb,Orb_NN) * O2b_nn(a_q,a_b,a_c,a_d,I,P_pb,Sigma_NN,Orb,Orb_NN) + 
                                                    O2b_nn(a_q,a_b,a_c,a_d,I,P_pb,V_NN,Orb,Orb_NN) * O2b_nn(a_p,a_b,a_c,a_d,I,P_pb,Sigma_NN,Orb,Orb_NN))
                                        ASum += ME
                                    end
                                end
                            end
                        end

                        @inbounds for b in Particle.p
                            a_b, l_b, j_b = b.a, b.l, b.j
                            P_pb = rem(l_p + l_b,2) + 1

                            if P_pb != (rem(l_q + l_b,2) + 1)
                                continue
                            end

                            @inbounds for c in Hole.n
                                a_c, l_c, j_c = c.a, c.l, c.j
                            
                                @inbounds for d in Hole.p
                                    a_d, l_d, j_d = d.a, d.l, d.j

                                    if P_pb != (rem(l_c + l_d,2) + 1)
                                        continue
                                    end

                                    @inbounds for I in div(max(abs(j_p - j_b), abs(j_c - j_d)),2):div(min(j_p + j_b, j_c + j_d),2)
                                        Amp = 0.5 * Float64((2*J + 1)) / Float64(j_p + 1) / Float64(2*J + 1)

                                        if I != J
                                            continue
                                        end

                                        ME = Amp * (O2b_pn(a_b,a_p,a_d,a_c,I,P_pb,V_NN,Orb_NN) * O2b_pn(a_b,a_q,a_d,a_c,I,P_pb,Sigma_NN,Orb_NN) + 
                                                    O2b_pn(a_b,a_q,a_d,a_c,I,P_pb,V_NN,Orb_NN) * O2b_pn(a_b,a_p,a_d,a_c,I,P_pb,Sigma_NN,Orb_NN))
                                        ASum += ME
                                    end
                                end
                            end
                        end
                    end
                end

                        # hole-hole 2p-2h scattering ...
                if t_ph == t_qg && a_p == a_q && j_h == j_g
                    # Case of proton-proton contribution ...
                    if t_ph == -1
                        @inbounds for b in Hole.p
                            a_b, l_b, j_b = b.a, b.l, b.j
                            P_hb = rem(l_h + l_b,2) + 1

                            if P_hb != (rem(l_g + l_b,2) + 1)
                                continue
                            end

                            @inbounds for c in Particle.p
                                a_c, l_c, j_c = c.a, c.l, c.j
                            
                                @inbounds for d in Particle.p
                                    a_d, l_d, j_d = d.a, d.l, d.j

                                    if P_hb != (rem(l_c + l_d,2) + 1)
                                        continue
                                    end

                                    @inbounds for I in div(max(abs(j_h - j_b),abs(j_c - j_d)),2):div(min(j_h + j_b, j_c + j_d),2)
                                        Amp = 0.25 * Float64((2*J + 1)) / Float64(j_h + 1) / Float64(2*J + 1)

                                        if I != J
                                            continue
                                        end

                                        ME = Amp * (O2b_pp(a_h,a_b,a_c,a_d,I,P_hb,V_NN,Orb,Orb_NN) * O2b_pp(a_g,a_b,a_c,a_d,I,P_hb,Sigma_NN,Orb,Orb_NN) + 
                                                    O2b_pp(a_g,a_b,a_c,a_d,I,P_hb,V_NN,Orb,Orb_NN) * O2b_pp(a_h,a_b,a_c,a_d,I,P_hb,Sigma_NN,Orb,Orb_NN))
                                        ASum += ME
                                    end
                                end
                            end
                        end

                        @inbounds for b in Hole.n
                            a_b, l_b, j_b = b.a, b.l, b.j
                            P_hb = rem(l_h + l_b,2) + 1

                            if P_hb != (rem(l_g + l_b,2) + 1)
                                continue
                            end

                            @inbounds for c in Particle.p
                                a_c, l_c, j_c = c.a, c.l, c.j
                            
                                @inbounds for d in Particle.n
                                    a_d, l_d, j_d = d.a, d.l, d.j

                                    if P_hb != (rem(l_c + l_d,2) + 1)
                                        continue
                                    end

                                    @inbounds for I in div(max(abs(j_h - j_b),abs(j_c - j_d)),2):div(min(j_h + j_b, j_c + j_d),2)
                                        Amp = 0.5 * Float64(2*J + 1) / Float64(j_h + 1) / Float64(2*J + 1)

                                        if I != J
                                            continue
                                        end

                                        ME = Amp * (O2b_pn(a_h,a_b,a_c,a_d,I,P_hb,V_NN,Orb_NN) * O2b_pn(a_g,a_b,a_c,a_d,I,P_hb,Sigma_NN,Orb_NN) + 
                                                    O2b_pn(a_g,a_b,a_c,a_d,I,P_hb,V_NN,Orb_NN) * O2b_pn(a_h,a_b,a_c,a_d,I,P_hb,Sigma_NN,Orb_NN))
                                        ASum += ME
                                    end
                                end
                            end
                        end
                    end
                    
                    # Case of neutron-neutron contribution ...
                    if t_ph == 1
                        @inbounds for b in Hole.n
                            a_b, l_b, j_b = b.a, b.l, b.j
                            P_hb = rem(l_h + l_b,2) + 1

                            if P_hb != (rem(l_g + l_b,2) + 1)
                                continue
                            end

                            @inbounds for c in Particle.n
                                a_c, l_c, j_c = c.a, c.l, c.j
                            
                                @inbounds for d in Particle.n
                                    a_d, l_d, j_d = d.a, d.l, d.j

                                    if P_hb != (rem(l_c + l_d,2) + 1)
                                        continue
                                    end

                                    @inbounds for I in div(max(abs(j_h - j_b),abs(j_c - j_d)),2):div(min(j_h + j_b, j_c + j_d),2)
                                        Amp = 0.25 * Float64((2*J + 1)) / Float64(j_h + 1) / Float64(2*J + 1)

                                        if I != J
                                            continue
                                        end

                                        ME = Amp * (O2b_nn(a_h,a_b,a_c,a_d,I,P_hb,V_NN,Orb,Orb_NN) * O2b_nn(a_g,a_b,a_c,a_d,I,P_hb,Sigma_NN,Orb,Orb_NN) + 
                                                    O2b_nn(a_g,a_b,a_c,a_d,I,P_hb,V_NN,Orb,Orb_NN) * O2b_nn(a_h,a_b,a_c,a_d,I,P_hb,Sigma_NN,Orb,Orb_NN))
                                        ASum += ME
                                    end
                                end
                            end
                        end

                        @inbounds for b in Hole.p
                            a_b, l_b, j_b = b.a, b.l, b.j
                            P_hb = rem(l_h + l_b,2) + 1

                            if P_hb != (rem(l_g + l_b,2) + 1)
                                continue
                            end

                            @inbounds for c in Particle.n
                                a_c, l_c, j_c = c.a, c.l, c.j
                            
                                @inbounds for d in Particle.p
                                    a_d, l_d, j_d = d.a, d.l, d.j

                                    if P_hb != (rem(l_c + l_d,2) + 1)
                                        continue
                                    end

                                    @inbounds for I in div(max(abs(j_h - j_b),abs(j_c - j_d)),2):div(min(j_h + j_b, j_c + j_d),2)
                                        Amp = 0.5 * Float64((2*J + 1)) / Float64(j_h + 1) / Float64(2*J + 1)

                                        if I != J
                                            continue
                                        end
                                        ME = Amp * (O2b_pn(a_b,a_h,a_d,a_c,I,P_hb,V_NN,Orb_NN) * O2b_pn(a_b,a_g,a_d,a_c,I,P_hb,Sigma_NN,Orb_NN) + 
                                                    O2b_pn(a_b,a_g,a_d,a_c,I,P_hb,V_NN,Orb_NN) * O2b_pn(a_b,a_h,a_d,a_c,I,P_hb,Sigma_NN,Orb_NN))
                                        ASum += ME
                                    end
                                end
                            end
                        end
                    end
                end
                =#
                
                # 2-body correlation matrix B ...
                    # 1-body scattering ...
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

                # Also move to another function ...

                #=
                    # 2-body scattering ...
                if rem(l_p + l_q, 2) == rem(l_h + l_g, 2)
                    @inbounds for I in div(max(abs(j_p - j_q),abs(j_h - j_g)),2):div(min(j_p + j_q,j_h + j_g),2)

                        if t_ph == -1 && t_qg == -1
                            Amp = 0.5 * Float64(2*I + 1) * phase(I + J + div(j_h + j_q,2)) * f6j(j_p,j_h,2*J,j_g,j_q,2*I) / Float64(2*J + 1)

                            @inbounds for c in Particle.p
                                a_c, l_c, j_c = c.a, c.l, c.j

                                @inbounds for d in Particle.p
                                    a_d, l_d, j_d = d.a, d.l, d.j

                                    if P_pq != (rem(l_c + l_d,2) + 1) || abs(j_c - j_d) > 2*I || (j_c + j_d) < 2*I
                                        continue
                                    end

                                    ME = Amp * O2b_pp(a_p,a_q,a_c,a_d,I,P_pq,V_NN,Orb,Orb_NN) * O2b_pp(a_h,a_g,a_c,a_d,I,P_pq,Sigma_NN,Orb,Orb_NN)
                                    BSum += ME
                                end
                            end

                            @inbounds for c in Hole.p
                                a_c, l_c, j_c = c.a, c.l, c.j

                                @inbounds for d in Hole.p
                                    a_d, l_d, j_d = d.a, d.l, d.j

                                    if P_pq != (rem(l_c + l_d,2) + 1) || abs(j_c - j_d) > 2*I || (j_c + j_d) < 2*I
                                        continue
                                    end

                                    ME = Amp * O2b_pp(a_h,a_g,a_c,a_d,I,P_pq,V_NN,Orb,Orb_NN) * O2b_pp(a_p,a_q,a_c,a_d,I,P_pq,Sigma_NN,Orb,Orb_NN)
                                    BSum += ME
                                end
                            end

                        end

                        if t_ph == -1 && t_qg == 1
                            Amp = Float64(2*I + 1) * phase(I + J + div(j_h + j_q,2)) * f6j(j_p,j_h,2*J,j_g,j_q,2*I) / Float64(2*J + 1)

                            @inbounds for c in Particle.p
                                a_c, l_c, j_c = c.a, c.l, c.j

                                @inbounds for d in Particle.n
                                    a_d, l_d, j_d = d.a, d.l, d.j

                                    if P_pq != (rem(l_c + l_d,2) + 1) || abs(j_c - j_d) > 2*I || (j_c + j_d) < 2*I
                                        continue
                                    end

                                    ME = Amp * O2b_pn(a_p,a_q,a_c,a_d,I,P_pq,V_NN,Orb_NN) * O2b_pn(a_h,a_g,a_c,a_d,I,P_pq,Sigma_NN,Orb_NN)
                                    BSum += ME
                                end
                            end

                            @inbounds for c in Hole.p
                                a_c, l_c, j_c = c.a, c.l, c.j

                                @inbounds for d in Hole.n
                                    a_d, l_d, j_d = d.a, d.l, d.j

                                    if P_pq != (rem(l_c + l_d,2) + 1) || abs(j_c - j_d) > 2*I || (j_c + j_d) < 2*I
                                        continue
                                    end

                                    ME = Amp * O2b_pn(a_h,a_g,a_c,a_d,I,P_pq,V_NN,Orb_NN) * O2b_pn(a_p,a_q,a_c,a_d,I,P_pq,Sigma_NN,Orb_NN)
                                    BSum += ME
                                end
                            end
                        end

                        #=
                        if t_ph == 1 && t_qg == -1
                            Amp = Float64(2*I + 1) * phase(I + J + div(j_g + j_p,2)) * f6j(j_q,j_g,2*J,j_h,j_p,2*I) / Float64(2*J + 1)

                            @inbounds for c in Particle.p
                                a_c, l_c, j_c = c.a, c.l, c.j

                                @inbounds for d in Particle.n
                                    a_d, l_d, j_d = d.a, d.l, d.j

                                    if P_pq != (rem(l_c + l_d,2) + 1) || abs(j_c - j_d) > 2*I || (j_c + j_d) < 2*I
                                        continue
                                    end

                                    ME = Amp * O2b_pn(a_q,a_p,a_d,a_c,I,P_pq,V_NN,Orb_NN) * O2b_pn(a_g,a_h,a_d,a_c,I,P_pq,Sigma_NN,Orb_NN)
                                    BSum += ME
                                end
                            end

                            @inbounds for c in Hole.p
                                a_c, l_c, j_c = c.a, c.l, c.j

                                @inbounds for d in Hole.n
                                    a_d, l_d, j_d = d.a, d.l, d.j

                                    if P_pq != (rem(l_c + l_d,2) + 1) || abs(j_c - j_d) > 2*I || (j_c + j_d) < 2*I
                                        continue
                                    end

                                    ME = Amp * O2b_pn(a_g,a_h,a_d,a_c,I,P_pq,V_NN,Orb_NN) * O2b_pn(a_q,a_p,a_d,a_c,I,P_pq,Sigma_NN,Orb_NN)
                                    BSum += ME
                                end
                            end


                        end
                        =#

                        if t_ph == 1 && t_qg == 1
                            Amp = 0.5 * Float64(2*I + 1) * phase(I + J + div(j_h + j_q,2)) * f6j(j_p,j_h,2*J,j_g,j_q,2*I) / Float64(2*J + 1)

                            @inbounds for c in Particle.n
                                a_c, l_c, j_c = c.a, c.l, c.j

                                @inbounds for d in Particle.n
                                    a_d, l_d, j_d = d.a, d.l, d.j

                                    if P_pq != (rem(l_c + l_d,2) + 1) || abs(j_c - j_d) > 2*I || (j_c + j_d) < 2*I
                                        continue
                                    end

                                    ME = Amp * O2b_nn(a_p,a_q,a_c,a_d,I,P_pq,V_NN,Orb,Orb_NN) * O2b_nn(a_h,a_g,a_c,a_d,I,P_pq,Sigma_NN,Orb,Orb_NN)
                                    BSum += ME
                                end
                            end

                            @inbounds for c in Hole.n
                                a_c, l_c, j_c = c.a, c.l, c.j

                                @inbounds for d in Hole.n
                                    a_d, l_d, j_d = d.a, d.l, d.j

                                    if P_pq != (rem(l_c + l_d,2) + 1) || abs(j_c - j_d) > 2*I || (j_c + j_d) < 2*I
                                        continue
                                    end

                                    ME = Amp * O2b_nn(a_h,a_g,a_c,a_d,I,P_pq,V_NN,Orb,Orb_NN) * O2b_nn(a_p,a_q,a_c,a_d,I,P_pq,Sigma_NN,Orb,Orb_NN)
                                    BSum += ME
                                end
                            end

                        end

                    end
                end
                    # 2-body scattering ... recoupled term ...
                        # pp terms ...
                if t_ph == -1 && t_qg == -1

                    @inbounds for b in Hole.p
                        a_b, l_b, j_b = b.a, b.l, b.j

                        P_hb = rem(l_h + l_b,2) + 1
                        P_pb = rem(l_p + l_b,2) + 1

                        P_qb = rem(l_q + l_b,2) + 1
                        P_gb = rem(l_g + l_b,2) + 1

                        @inbounds for d in Particle.p
                            a_d, l_d, j_d = d.a, d.l, d.j

                            P_qd = rem(l_q + l_d,2) + 1
                            P_gd = rem(l_g + l_d,2) + 1

                            P_pd = rem(l_p + l_d,2) + 1
                            P_hd = rem(l_h + l_d,2) + 1

                            # V_pbgd S_hgqd
                            if P_hb == P_qd && P_pb == P_gd

                                @inbounds for I in div(max(abs(j_p - j_q),abs(j_h - j_g)),2):div(min(j_p + j_q,j_h + j_g),2)
                                    Amp_I = Float64(2*I + 1) * phase(1 + I + div(j_h + j_g,2)) * f6j(j_p,j_h,2*J,j_q,j_g,2*I) / Float64(2*J + 1)

                                    @inbounds for J_1 in div(max(abs(j_p - j_b),abs(j_g - j_d)),2):div(min(j_p + j_b,j_g + j_d),2)
                                        Amp_J_1 = Float64(2*J_1 + 1) * phase(J_1) * f6j(j_p,j_g,2*I,j_d,j_b,2*J_1)

                                        @inbounds for J_2 in div(max(abs(j_b - j_h),abs(j_q - j_d)),2):div(min(j_b + j_h,j_q + j_d),2)
                                            Amp_J_2 = Float64(2*J_2 + 1) * phase(J_2) * f6j(j_h,j_q,2*I,j_d,j_b,2*J_2)

                                            Amp = Amp_I * Amp_J_1 * Amp_J_2

                                            if abs(Amp) < 1e-8
                                                continue
                                            end

                                            ME = Amp * O2b_pp(a_p,a_b,a_g,a_d,J_1,P_pb,V_NN,Orb,Orb_NN) * O2b_pp(a_h,a_b,a_q,a_d,J_2,P_hb,Sigma_NN,Orb,Orb_NN)
                                            BSum += ME
                                        end
                                    end
                                end

                            end

                            # V_qbhd S_gbpd
                            if P_hd == P_qb && P_pd == P_gb

                                @inbounds for I in div(max(abs(j_p - j_q),abs(j_h - j_g)),2):div(min(j_p + j_q,j_h + j_g),2)
                                    Amp_I = Float64(2*I + 1) * phase(1 + I + div(j_h + j_g,2)) * f6j(j_q,j_g,2*J,j_p,j_h,2*I) / Float64(2*J + 1)


                                    @inbounds for J_1 in div(max(abs(j_q -  j_b),abs(j_h - j_d)),2):div(min(j_q + j_b,j_h + j_d),2)
                                        Amp_J_1 = Float64(2*J_1 + 1) * phase(J_1) * f6j(j_q,j_h,2*I,j_d,j_b,2*J_1)

                                        @inbounds for J_2 in div(max(abs(j_g - j_b),abs(j_p - j_d)),2):div(min(j_g + j_b,j_p + j_d),2)
                                            Amp_J_2 = Float64(2*J_2 + 1) * phase(J_2) * f6j(j_g,j_p,2*I,j_d,j_b,2*J_2)

                                            Amp = Amp_I * Amp_J_1 * Amp_J_2

                                            if abs(Amp) < 1e-8
                                                continue
                                            end

                                            ME = Amp * O2b_pp(a_q,a_b,a_h,a_d,J_1,P_qb,V_NN,Orb,Orb_NN) * O2b_pp(a_g,a_b,a_p,a_d,J_2,P_gb,Sigma_NN,Orb,Orb_NN)
                                            BSum += ME
                                        end
                                    end
                                end

                            end

                        end
                    end

                    @inbounds for b in Hole.n
                        a_b, l_b, j_b = b.a, b.l, b.j

                        P_hb = rem(l_h + l_b,2) + 1
                        P_pb = rem(l_p + l_b,2) + 1

                        P_qb = rem(l_q + l_b,2) + 1
                        P_gb = rem(l_g + l_b,2) + 1

                        @inbounds for d in Particle.n
                            a_d, l_d, j_d = d.a, d.l, d.j

                            P_qd = rem(l_q + l_d,2) + 1
                            P_gd = rem(l_g + l_d,2) + 1

                            P_pd = rem(l_p + l_d,2) + 1
                            P_hd = rem(l_h + l_d,2) + 1

                            # V_pbgd S_hgqd
                            if P_hb == P_qd && P_pb == P_gd

                                @inbounds for I in div(max(abs(j_p - j_q),abs(j_h - j_g)),2):div(min(j_p + j_q,j_h + j_g),2)
                                    Amp_I = Float64(2*I + 1) * phase(1 + I + div(j_h + j_g,2)) * f6j(j_p,j_h,2*J,j_q,j_g,2*I) / Float64(2*J + 1)

                                    @inbounds for J_1 in div(max(abs(j_p - j_b),abs(j_g - j_d)),2):div(min(j_p + j_b,j_g + j_d),2)
                                        Amp_J_1 = Float64(2*J_1 + 1) * phase(J_1) * f6j(j_p,j_g,2*I,j_d,j_b,2*J_1)

                                        @inbounds for J_2 in div(max(abs(j_b - j_h),abs(j_q - j_d)),2):div(min(j_b + j_h,j_q + j_d),2)
                                            Amp_J_2 = Float64(2*J_2 + 1) * phase(J_2) * f6j(j_h,j_q,2*I,j_d,j_b,2*J_2)

                                            Amp = Amp_I * Amp_J_1 * Amp_J_2

                                            if abs(Amp) < 1e-8
                                                continue
                                            end

                                            ME = Amp * O2b_pn(a_p,a_b,a_g,a_d,J_1,P_pb,V_NN,Orb_NN) * O2b_pn(a_h,a_b,a_q,a_d,J_2,P_hb,Sigma_NN,Orb_NN)
                                            BSum += ME
                                        end
                                    end
                                end

                            end

                            # V_qbhd S_gbpd
                            if P_hd == P_qb && P_pd == P_gb

                                @inbounds for I in div(max(abs(j_p - j_q),abs(j_h - j_g)),2):div(min(j_p + j_q,j_h + j_g),2)
                                    Amp_I = Float64(2*I + 1) * phase(1 + I + div(j_h + j_g,2)) * f6j(j_q,j_g,2*J,j_p,j_h,2*I)


                                    @inbounds for J_1 in div(max(abs(j_q -  j_b),abs(j_h - j_d)),2):div(min(j_q + j_b,j_h + j_d),2)
                                        Amp_J_1 = Float64(2*J_1 + 1) * phase(J_1) * f6j(j_q,j_h,2*I,j_d,j_b,2*J_1)

                                        @inbounds for J_2 in div(max(abs(j_g - j_b),abs(j_p - j_d)),2):div(min(j_g + j_b,j_p + j_d),2)
                                            Amp_J_2 = Float64(2*J_2 + 1) * phase(J_2) * f6j(j_g,j_p,2*I,j_d,j_b,2*J_2)

                                            Amp = Amp_I * Amp_J_1 * Amp_J_2

                                            if abs(Amp) < 1e-8
                                                continue
                                            end

                                            ME = Amp * O2b_pn(a_q,a_b,a_h,a_d,J_1,P_qb,V_NN,Orb_NN) * O2b_pn(a_g,a_b,a_p,a_d,J_2,P_gb,Sigma_NN,Orb_NN)
                                            BSum += ME
                                        end
                                    end
                                end

                            end

                        end
                    end

                end
                    # nn terms ...
                if t_ph == 1 && t_qg == 1

                    @inbounds for b in Hole.n
                        a_b, l_b, j_b = b.a, b.l, b.j

                        P_hb = rem(l_h + l_b,2) + 1
                        P_pb = rem(l_p + l_b,2) + 1

                        P_qb = rem(l_q + l_b,2) + 1
                        P_gb = rem(l_g + l_b,2) + 1

                        @inbounds for d in Particle.n
                            a_d, l_d, j_d = d.a, d.l, d.j

                            P_qd = rem(l_q + l_d,2) + 1
                            P_gd = rem(l_g + l_d,2) + 1

                            P_pd = rem(l_p + l_d,2) + 1
                            P_hd = rem(l_h + l_d,2) + 1

                            # V_pbgd S_hgqd
                            if P_hb == P_qd && P_pb == P_gd

                                @inbounds for I in div(max(abs(j_p - j_q),abs(j_h - j_g)),2):div(min(j_p + j_q,j_h + j_g),2)
                                    Amp_I = Float64(2*I + 1) * phase(1 + I + div(j_h + j_g,2)) * f6j(j_p,j_h,2*J,j_q,j_g,2*I) / Float64(2*J + 1)


                                    @inbounds for J_1 in div(max(abs(j_p -  j_b),abs(j_g - j_d)),2):div(min(j_p + j_b,j_g + j_d),2)
                                        Amp_J_1 = Float64(2*J_1 + 1) * phase(J_1) * f6j(j_p,j_g,2*I,j_d,j_b,2*J_1)

                                        @inbounds for J_2 in div(max(abs(j_b - j_h),abs(j_q - j_d)),2):div(min(j_b + j_h,j_q + j_d),2)
                                            Amp_J_2 = Float64(2*J_2 + 1) * phase(J_2) * f6j(j_h,j_q,2*I,j_d,j_b,2*J_2)

                                            Amp = Amp_I * Amp_J_1 * Amp_J_2

                                            if abs(Amp) < 1e-8
                                                continue
                                            end

                                            ME = Amp * O2b_nn(a_p,a_b,a_g,a_d,J_1,P_pb,V_NN,Orb,Orb_NN) * O2b_nn(a_h,a_b,a_q,a_d,J_2,P_hb,Sigma_NN,Orb,Orb_NN)
                                            BSum += ME
                                        end
                                    end
                                end

                            end

                            # V_qbhd S_gbpd
                            if P_hd == P_qb && P_pd == P_gb

                                @inbounds for I in div(max(abs(j_p - j_q),abs(j_h - j_g)),2):div(min(j_p + j_q,j_h + j_g),2)
                                    Amp_I = Float64(2*I + 1) * phase(1 + I + div(j_h + j_g,2)) * f6j(j_q,j_g,2*J,j_p,j_h,2*I) / Float64(2*J + 1)

                                    @inbounds for J_1 in div(max(abs(j_q -  j_b),abs(j_h - j_d)),2):div(min(j_q + j_b,j_h + j_d),2)
                                        Amp_J_1 = Float64(2*J_1 + 1) * phase(J_1) * f6j(j_q,j_h,2*I,j_d,j_b,2*J_1)

                                        @inbounds for J_2 in div(max(abs(j_g - j_b),abs(j_p - j_d)),2):div(min(j_g + j_b,j_p + j_d),2)
                                            Amp_J_2 = Float64(2*J_2 + 1) * phase(J_2) * f6j(j_g,j_p,2*I,j_d,j_b,2*J_2)

                                            Amp = Amp_I * Amp_J_1 * Amp_J_2

                                            if abs(Amp) < 1e-8
                                                continue
                                            end

                                            ME = Amp * O2b_nn(a_q,a_b,a_h,a_d,J_1,P_qb,V_NN,Orb,Orb_NN) * O2b_nn(a_g,a_b,a_p,a_d,J_2,P_gb,Sigma_NN,Orb,Orb_NN)
                                            BSum += ME
                                        end
                                    end
                                end

                            end

                        end
                    end

                    @inbounds for b in Hole.p
                        a_b, l_b, j_b = b.a, b.l, b.j

                        P_hb = rem(l_h + l_b,2) + 1
                        P_pb = rem(l_p + l_b,2) + 1

                        P_qb = rem(l_q + l_b,2) + 1
                        P_gb = rem(l_g + l_b,2) + 1

                        @inbounds for d in Particle.p
                            a_d, l_d, j_d = d.a, d.l, d.j

                            P_qd = rem(l_q + l_d,2) + 1
                            P_gd = rem(l_g + l_d,2) + 1

                            P_pd = rem(l_p + l_d,2) + 1
                            P_hd = rem(l_h + l_d,2) + 1

                            # V_pbgd S_hgqd
                            if P_hb == P_qd && P_pb == P_gd

                                @inbounds for I in div(max(abs(j_p - j_q),abs(j_h - j_g)),2):div(min(j_p + j_q,j_h + j_g),2)
                                    Amp_I = Float64(2*I + 1) * phase(1 + I + div(j_h + j_g,2)) * f6j(j_p,j_h,2*J,j_q,j_g,2*I) / Float64(2*J + 1)

                                    @inbounds for J_1 in div(max(abs(j_p -  j_b),abs(j_g - j_d)),2):div(min(j_p + j_b,j_g + j_d),2)
                                        Amp_J_1 = Float64(2*J_1 + 1) * phase(J_1) * f6j(j_p,j_g,2*I,j_d,j_b,2*J_1)

                                        @inbounds for J_2 in div(max(abs(j_b - j_h),abs(j_q - j_d)),2):div(min(j_b + j_h,j_q + j_d),2)
                                            Amp_J_2 = Float64(2*J_2 + 1) * phase(J_2) * f6j(j_h,j_q,2*I,j_d,j_b,2*J_2)

                                            Amp = Amp_I * Amp_J_1 * Amp_J_2

                                            if abs(Amp) < 1e-8
                                                continue
                                            end

                                            ME = Amp * O2b_pn(a_b,a_p,a_d,a_g,J_1,P_pb,V_NN,Orb_NN) * O2b_pn(a_b,a_h,a_d,a_q,J_2,P_hb,Sigma_NN,Orb_NN)
                                            BSum += ME
                                        end
                                    end
                                end

                            end

                            # V_qbhd S_gbpd
                            if P_hd == P_qb && P_pd == P_gb

                                @inbounds for I in div(max(abs(j_p - j_q),abs(j_h - j_g)),2):div(min(j_p + j_q,j_h + j_g),2)
                                    Amp_I = Float64(2*I + 1) * phase(1 + I + div(j_h + j_g,2)) * f6j(j_q,j_g,2*J,j_p,j_h,2*I) / Float64(2*J + 1)

                                    @inbounds for J_1 in div(max(abs(j_q -  j_b),abs(j_h - j_d)),2):div(min(j_q + j_b,j_h + j_d),2)
                                        Amp_J_1 = Float64(2*J_1 + 1) * phase(J_1) * f6j(j_q,j_h,2*I,j_d,j_b,2*J_1)

                                        @inbounds for J_2 in div(max(abs(j_g - j_b),abs(j_p - j_d)),2):div(min(j_g + j_b,j_p + j_d),2)
                                            Amp_J_2 = Float64(2*J_2 + 1) * phase(J_2) * f6j(j_g,j_p,2*I,j_d,j_b,2*J_2)

                                            Amp = Amp_I * Amp_J_1 * Amp_J_2

                                            if abs(Amp) < 1e-8
                                                continue
                                            end

                                            ME = Amp * O2b_pn(a_b,a_q,a_d,a_h,J_1,P_qb,V_NN,Orb_NN) * O2b_pn(a_b,a_g,a_d,a_p,J_2,P_gb,Sigma_NN,Orb_NN)
                                            BSum += ME
                                        end
                                    end
                                end

                            end

                        end
                    end

                end
                    # pn terms ...
                if t_ph == -1 && t_qg == 1

                    @inbounds for b in Hole.n
                        a_b, l_b, j_b = b.a, b.l, b.j

                        P_hb = rem(l_h + l_b,2) + 1
                        P_pb = rem(l_p + l_b,2) + 1

                        @inbounds for c in Particle.p
                            a_c, l_c, j_c = c.a, c.l, c.j

                            P_qc = rem(l_q + l_c,2) + 1
                            P_gc = rem(l_g + l_c,2) + 1

                            # V_pbcg S_hgcq
                            if P_hb == P_qc && P_pb == P_gc

                                @inbounds for I in div(max(abs(j_p - j_q),abs(j_h - j_g)),2):div(min(j_p + j_q,j_h + j_g),2)
                                    Amp_I = Float64(2*I + 1) * phase(I + div(j_h + j_q,2)) * f6j(j_p,j_g,2*I,j_q,j_h,2*J) / Float64(2*J + 1)


                                    @inbounds for J_1 in div(max(abs(j_p -  j_b),abs(j_g - j_c)),2):div(min(j_p + j_b,j_g + j_c),2)
                                        Amp_J_1 = Float64(2*J_1 + 1) * f6j(j_p,j_g,2*I,j_c,j_b,2*J_1)

                                        @inbounds for J_2 in div(max(abs(j_h - j_b),abs(j_q - j_c)),2):div(min(j_h + j_b,j_q + j_c),2)
                                            Amp_J_2 = Float64(2*J_2 + 1)* f6j(j_h,j_q,2*I,j_c,j_b,2*J_2)

                                            Amp = Amp_I * Amp_J_1 * Amp_J_2

                                            if abs(Amp) < 1e-8
                                                continue
                                            end

                                            ME = Amp * O2b_pn(a_p,a_b,a_c,a_g,J_1,P_pb,V_NN,Orb_NN) * O2b_pn(a_h,a_b,a_c,a_q,J_2,P_hb,Sigma_NN,Orb_NN)
                                            BSum += ME
                                        end
                                    end
                                end

                            end

                            # V_hbcq S_pbcg
                            if P_pb == P_gc && P_hb == P_qc

                                @inbounds for I in div(max(abs(j_p - j_q),abs(j_h - j_g)),2):div(min(j_p + j_q,j_h + j_g),2)
                                    Amp_I = Float64(2*I + 1) * phase(I + div(j_h + j_q,2)) * f6j(j_p,j_g,2*I,j_q,j_h,2*J) / Float64(2*J + 1)


                                    @inbounds for J_1 in div(max(abs(j_h -  j_b),abs(j_q - j_c)),2):div(min(j_h + j_b,j_q + j_c),2)
                                        Amp_J_1 = Float64(2*J_1 + 1) * f6j(j_h,j_q,2*I,j_c,j_b,2*J_1)

                                        @inbounds for J_2 in div(max(abs(j_p - j_b),abs(j_q - j_g)),2):div(min(j_p + j_b,j_g + j_c),2)
                                            Amp_J_2 = Float64(2*J_2 + 1)* f6j(j_p,j_g,2*I,j_c,j_b,2*J_2)

                                            Amp = Amp_I * Amp_J_1 * Amp_J_2

                                            if abs(Amp) < 1e-8
                                                continue
                                            end

                                            ME = Amp * O2b_pn(a_h,a_b,a_c,a_q,J_1,P_hb,V_NN,Orb_NN) * O2b_pn(a_p,a_b,a_c,a_g,J_2,P_pb,Sigma_NN,Orb_NN)
                                            BSum += ME
                                        end
                                    end
                                end

                            end
                        end
                    end
                end
                =#

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
    if Params.Calc.RPA.TBCF == true
        #@time A, B = RPA_allocate_TBCF(Params,Orb,Orb_1p1h,Orb_NN,V_NN,Rho,Sigma,A,B)
    end


    println("\tRPA matrices A, B & N allocated ...")

    return A, B, N
end

function RPA_allocate_TBCF(Params::Parameters,Orb::Vector{Orb1B},Orb_1p1h::Orb2B_1p1h,Orb_NN::Orb2B,V_NN::O2B,Rho::O1B,Sigma::O2B,A::Matrix{Matrix{Float64}},B::Matrix{Matrix{Float64}})
 # Read calculation parameters ...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1
    a_max = div((N_max + 1)*(N_max + 2),2)

    JP_list = JP_initialize(J_max)
    
    println("\t\tComputing 2-body correlation function contributions to the RPA matrices A & B ...")

    # Allocate matrices A & B ... no 2-body correlation function yet ...
    @inbounds for JP in JP_list
        J, P = JP[1], JP[2]
        N_ph = Orb_1p1h.N[P,J+1]
        Orb_1p1h_JP = Orb_1p1h.Ind[P,J+1]

        A_JP = zeros(Float64,N_ph,N_ph)
        B_JP = zeros(Float64,N_ph,N_ph)

        @inbounds for i_ph in 1:N_ph
            Orb_ph = Orb_1p1h_JP[i_ph]
            a_p, a_h, T_ph = Orb_ph.a, Orb_ph.b, Orb_ph.T

            l_p, j_p = Orb[a_p].l, Orb[a_p].j
            l_h, j_h = Orb[a_h].l, Orb[a_h].j

            @inbounds Threads.@threads for i_qg in 1:i_ph
                Orb_qg = Orb_1p1h_JP[i_qg]
                a_q, a_g, T_qg = Orb_qg.a, Orb_qg.b, Orb_qg.T

                l_q, j_q = Orb[a_q].l, Orb[a_q].j
                l_g, j_g = Orb[a_g].l, Orb[a_g].j

                P_pg = rem(l_p + l_g,2) + 1
                P_pq = rem(l_p + l_q,2) + 1
    
                ASum = 0.0
                BSum = 0.0

                # 2-body TDA matrix A ...
                    # 2-body scattering
                        # Hole-hole contact term ...
                if T_ph == T_qg && a_p == a_q && j_h == j_g

                    @inbounds for a_b in 1:a_max
                        l_b, j_b = Orb[a_b].l, Orb[a_b].j
                        P_hb = rem(l_h + l_b,2) + 1

                        if P_hb != (rem(l_g + l_b,2) + 1)
                            continue
                        end

                        @inbounds for a_c in 1:a_max
                            l_c, j_c = Orb[a_c].l, Orb[a_c].j
                        
                            @inbounds for a_d in 1:a_max
                                l_d, j_d = Orb[a_d].l, Orb[a_d].j

                                if P_hb != (rem(l_c + l_d,2) + 1)
                                    continue
                                end

                                I_min, I_max = div(max(abs(j_h - j_b),abs(j_c - j_d)),2), div(min(j_h + j_b,j_c + j_d),2)
                                @inbounds for I in I_min:I_max

                                    Amp = 0.5 * Float64(2*I + 1) / Float64(j_h + 1)

                                    if T_ph == -1
                                        ppSigma_hbcd = O2b_pp(a_h,a_b,a_c,a_d,I,P_hb,Sigma,Orb,Orb_NN)
                                        ppSigma_gbcd = O2b_pp(a_g,a_b,a_c,a_d,I,P_hb,Sigma,Orb,Orb_NN)

                                        ppV_hbcd = O2b_pp(a_h,a_b,a_c,a_d,I,P_hb,V_NN,Orb,Orb_NN)
                                        ppV_gbcd = O2b_pp(a_g,a_b,a_c,a_d,I,P_hb,V_NN,Orb,Orb_NN)

                                        pnSigma_hbcd = O2b_pn(a_h,a_b,a_c,a_d,I,P_hb,Sigma,Orb_NN)
                                        pnSigma_gbcd = O2b_pn(a_g,a_b,a_c,a_d,I,P_hb,Sigma,Orb_NN)

                                        pnV_hbcd = O2b_pn(a_h,a_b,a_c,a_d,I,P_hb,V_NN,Orb_NN)
                                        pnV_gbcd = O2b_pn(a_g,a_b,a_c,a_d,I,P_hb,V_NN,Orb_NN)


                                        ME = 0.5 * Amp * (ppSigma_gbcd * ppV_hbcd + ppSigma_hbcd * ppV_gbcd)

                                        ASum += ME

                                        ME = Amp * (pnSigma_gbcd * pnV_hbcd + pnSigma_hbcd * pnV_gbcd)

                                        ASum += ME

                                    elseif T_ph == 1
                                        nnSigma_hbcd = O2b_nn(a_h,a_b,a_c,a_d,I,P_hb,Sigma,Orb,Orb_NN)
                                        nnSigma_gbcd = O2b_nn(a_g,a_b,a_c,a_d,I,P_hb,Sigma,Orb,Orb_NN)

                                        nnV_hbcd = O2b_nn(a_h,a_b,a_c,a_d,I,P_hb,V_NN,Orb,Orb_NN)
                                        nnV_gbcd = O2b_nn(a_g,a_b,a_c,a_d,I,P_hb,V_NN,Orb,Orb_NN)

                                        pnSigma_hbcd = O2b_pn(a_b,a_h,a_c,a_d,I,P_hb,Sigma,Orb_NN)
                                        pnSigma_gbcd = O2b_pn(a_b,a_g,a_c,a_d,I,P_hb,Sigma,Orb_NN)

                                        pnV_hbcd = O2b_pn(a_b,a_h,a_c,a_d,I,P_hb,V_NN,Orb_NN)
                                        pnV_gbcd = O2b_pn(a_b,a_g,a_c,a_d,I,P_hb,V_NN,Orb_NN)


                                        ME = 0.5 * Amp * (nnSigma_gbcd * nnV_hbcd + nnSigma_hbcd * nnV_gbcd)

                                        ASum += ME

                                        ME = Amp * (pnSigma_gbcd * pnV_hbcd + pnSigma_hbcd * pnV_gbcd)

                                        ASum += ME
                                    end
                                    
                                end
                            end
                        end

                    end

                end
                        # Particle-particle contact term ...
                if T_ph == T_qg && a_h == a_g && j_p == j_q

                    @inbounds for a_b in 1:a_max
                        l_b, j_b = Orb[a_b].l, Orb[a_b].j
                        P_pb = rem(l_p + l_b,2) + 1

                        if P_pb != (rem(l_q + l_b,2) + 1)
                            continue
                        end

                        @inbounds for a_c in 1:a_max
                            l_c, j_c = Orb[a_c].l, Orb[a_c].j
                        
                            @inbounds for a_d in 1:a_max
                                l_d, j_d = Orb[a_d].l, Orb[a_d].j

                                if P_pb != (rem(l_c + l_d,2) + 1)
                                    continue
                                end

                                I_min, I_max = div(max(abs(j_p - j_b),abs(j_c - j_d)),2), div(min(j_p + j_b,j_c + j_d),2)
                                @inbounds for I in I_min:I_max

                                    Amp = 0.5 * Float64(2*I + 1) / Float64(j_p + 1)

                                    if T_ph == -1
                                        ppSigma_pbcd = O2b_pp(a_p,a_b,a_c,a_d,I,P_pb,Sigma,Orb,Orb_NN)
                                        ppSigma_qbcd = O2b_pp(a_q,a_b,a_c,a_d,I,P_pb,Sigma,Orb,Orb_NN)

                                        ppV_pbcd = O2b_pp(a_p,a_b,a_c,a_d,I,P_pb,V_NN,Orb,Orb_NN)
                                        ppV_qbcd = O2b_pp(a_q,a_b,a_c,a_d,I,P_pb,V_NN,Orb,Orb_NN)

                                        pnSigma_pbcd = O2b_pn(a_p,a_b,a_c,a_d,I,P_pb,Sigma,Orb_NN)
                                        pnSigma_qbcd = O2b_pn(a_q,a_b,a_c,a_d,I,P_pb,Sigma,Orb_NN)

                                        pnV_pbcd = O2b_pn(a_p,a_b,a_c,a_d,I,P_pb,V_NN,Orb_NN)
                                        pnV_qbcd = O2b_pn(a_q,a_b,a_c,a_d,I,P_pb,V_NN,Orb_NN)


                                        ME = 0.5 * Amp * (ppSigma_qbcd * ppV_pbcd + ppSigma_pbcd * ppV_qbcd)

                                        ASum += ME

                                        ME = Amp * (pnSigma_qbcd * pnV_pbcd + pnSigma_pbcd * pnV_qbcd)

                                        ASum += ME

                                    elseif T_ph == 1
                                        nnSigma_pbcd = O2b_nn(a_p,a_b,a_c,a_d,I,P_pb,Sigma,Orb,Orb_NN)
                                        nnSigma_qbcd = O2b_nn(a_q,a_b,a_c,a_d,I,P_pb,Sigma,Orb,Orb_NN)

                                        nnV_pbcd = O2b_nn(a_p,a_b,a_c,a_d,I,P_pb,V_NN,Orb,Orb_NN)
                                        nnV_qbcd = O2b_nn(a_q,a_b,a_c,a_d,I,P_pb,V_NN,Orb,Orb_NN)

                                        pnSigma_pbcd = O2b_pn(a_b,a_p,a_d,a_c,I,P_pb,Sigma,Orb_NN)
                                        pnSigma_qbcd = O2b_pn(a_b,a_q,a_d,a_c,I,P_pb,Sigma,Orb_NN)

                                        pnV_pbcd = O2b_pn(a_b,a_p,a_d,a_c,I,P_pb,V_NN,Orb_NN)
                                        pnV_qbcd = O2b_pn(a_b,a_q,a_d,a_c,I,P_pb,V_NN,Orb_NN)

                                        ME = 0.5 * Amp * (nnSigma_qbcd * nnV_pbcd + nnSigma_pbcd * nnV_qbcd)

                                        ASum += ME

                                        ME = Amp * (pnSigma_qbcd * pnV_pbcd + pnSigma_pbcd * pnV_qbcd)

                                        ASum += ME
                                    end

                                end
                            end
                        end

                    end

                end

                        # Box diagram 1
                #=


                        # hole-hole 2p-2h scattering ...
                if t_ph == t_qg && a_p == a_q && j_h == j_g
                    # Case of proton-proton contribution ...
                    if t_ph == -1
                        @inbounds for b in Hole.p
                            a_b, l_b, j_b = b.a, b.l, b.j
                            P_hb = rem(l_h + l_b,2) + 1

                            if P_hb != (rem(l_g + l_b,2) + 1)
                                continue
                            end

                            @inbounds for c in Particle.p
                                a_c, l_c, j_c = c.a, c.l, c.j
                            
                                @inbounds for d in Particle.p
                                    a_d, l_d, j_d = d.a, d.l, d.j

                                    if P_hb != (rem(l_c + l_d,2) + 1)
                                        continue
                                    end

                                    @inbounds for I in div(max(abs(j_h - j_b),abs(j_c - j_d)),2):div(min(j_h + j_b, j_c + j_d),2)
                                        Amp = 0.25 * Float64((2*J + 1)) / Float64(j_h + 1) / Float64(2*J + 1)

                                        if I != J
                                            continue
                                        end

                                        ME = Amp * (O2b_pp(a_h,a_b,a_c,a_d,I,P_hb,V_NN,Orb,Orb_NN) * O2b_pp(a_g,a_b,a_c,a_d,I,P_hb,Sigma_NN,Orb,Orb_NN) + 
                                                    O2b_pp(a_g,a_b,a_c,a_d,I,P_hb,V_NN,Orb,Orb_NN) * O2b_pp(a_h,a_b,a_c,a_d,I,P_hb,Sigma_NN,Orb,Orb_NN))
                                        ASum += ME
                                    end
                                end
                            end
                        end

                        @inbounds for b in Hole.n
                            a_b, l_b, j_b = b.a, b.l, b.j
                            P_hb = rem(l_h + l_b,2) + 1

                            if P_hb != (rem(l_g + l_b,2) + 1)
                                continue
                            end

                            @inbounds for c in Particle.p
                                a_c, l_c, j_c = c.a, c.l, c.j
                            
                                @inbounds for d in Particle.n
                                    a_d, l_d, j_d = d.a, d.l, d.j

                                    if P_hb != (rem(l_c + l_d,2) + 1)
                                        continue
                                    end

                                    @inbounds for I in div(max(abs(j_h - j_b),abs(j_c - j_d)),2):div(min(j_h + j_b, j_c + j_d),2)
                                        Amp = 0.5 * Float64(2*J + 1) / Float64(j_h + 1) / Float64(2*J + 1)

                                        if I != J
                                            continue
                                        end

                                        ME = Amp * (O2b_pn(a_h,a_b,a_c,a_d,I,P_hb,V_NN,Orb_NN) * O2b_pn(a_g,a_b,a_c,a_d,I,P_hb,Sigma_NN,Orb_NN) + 
                                                    O2b_pn(a_g,a_b,a_c,a_d,I,P_hb,V_NN,Orb_NN) * O2b_pn(a_h,a_b,a_c,a_d,I,P_hb,Sigma_NN,Orb_NN))
                                        ASum += ME
                                    end
                                end
                            end
                        end
                    end
                    
                    # Case of neutron-neutron contribution ...
                    if t_ph == 1
                        @inbounds for b in Hole.n
                            a_b, l_b, j_b = b.a, b.l, b.j
                            P_hb = rem(l_h + l_b,2) + 1

                            if P_hb != (rem(l_g + l_b,2) + 1)
                                continue
                            end

                            @inbounds for c in Particle.n
                                a_c, l_c, j_c = c.a, c.l, c.j
                            
                                @inbounds for d in Particle.n
                                    a_d, l_d, j_d = d.a, d.l, d.j

                                    if P_hb != (rem(l_c + l_d,2) + 1)
                                        continue
                                    end

                                    @inbounds for I in div(max(abs(j_h - j_b),abs(j_c - j_d)),2):div(min(j_h + j_b, j_c + j_d),2)
                                        Amp = 0.25 * Float64((2*J + 1)) / Float64(j_h + 1) / Float64(2*J + 1)

                                        if I != J
                                            continue
                                        end

                                        ME = Amp * (O2b_nn(a_h,a_b,a_c,a_d,I,P_hb,V_NN,Orb,Orb_NN) * O2b_nn(a_g,a_b,a_c,a_d,I,P_hb,Sigma_NN,Orb,Orb_NN) + 
                                                    O2b_nn(a_g,a_b,a_c,a_d,I,P_hb,V_NN,Orb,Orb_NN) * O2b_nn(a_h,a_b,a_c,a_d,I,P_hb,Sigma_NN,Orb,Orb_NN))
                                        ASum += ME
                                    end
                                end
                            end
                        end

                        @inbounds for b in Hole.p
                            a_b, l_b, j_b = b.a, b.l, b.j
                            P_hb = rem(l_h + l_b,2) + 1

                            if P_hb != (rem(l_g + l_b,2) + 1)
                                continue
                            end

                            @inbounds for c in Particle.n
                                a_c, l_c, j_c = c.a, c.l, c.j
                            
                                @inbounds for d in Particle.p
                                    a_d, l_d, j_d = d.a, d.l, d.j

                                    if P_hb != (rem(l_c + l_d,2) + 1)
                                        continue
                                    end

                                    @inbounds for I in div(max(abs(j_h - j_b),abs(j_c - j_d)),2):div(min(j_h + j_b, j_c + j_d),2)
                                        Amp = 0.5 * Float64((2*J + 1)) / Float64(j_h + 1) / Float64(2*J + 1)

                                        if I != J
                                            continue
                                        end
                                        ME = Amp * (O2b_pn(a_b,a_h,a_d,a_c,I,P_hb,V_NN,Orb_NN) * O2b_pn(a_b,a_g,a_d,a_c,I,P_hb,Sigma_NN,Orb_NN) + 
                                                    O2b_pn(a_b,a_g,a_d,a_c,I,P_hb,V_NN,Orb_NN) * O2b_pn(a_b,a_h,a_d,a_c,I,P_hb,Sigma_NN,Orb_NN))
                                        ASum += ME
                                    end
                                end
                            end
                        end
                    end
                end
                
                # 2-body correlation matrix B ...
                    # 2-body scattering ...
                if rem(l_p + l_q, 2) == rem(l_h + l_g, 2)
                    @inbounds for I in div(max(abs(j_p - j_q),abs(j_h - j_g)),2):div(min(j_p + j_q,j_h + j_g),2)

                        if t_ph == -1 && t_qg == -1
                            Amp = 0.5 * Float64(2*I + 1) * phase(I + J + div(j_h + j_q,2)) * f6j(j_p,j_h,2*J,j_g,j_q,2*I) / Float64(2*J + 1)

                            @inbounds for c in Particle.p
                                a_c, l_c, j_c = c.a, c.l, c.j

                                @inbounds for d in Particle.p
                                    a_d, l_d, j_d = d.a, d.l, d.j

                                    if P_pq != (rem(l_c + l_d,2) + 1) || abs(j_c - j_d) > 2*I || (j_c + j_d) < 2*I
                                        continue
                                    end

                                    ME = Amp * O2b_pp(a_p,a_q,a_c,a_d,I,P_pq,V_NN,Orb,Orb_NN) * O2b_pp(a_h,a_g,a_c,a_d,I,P_pq,Sigma_NN,Orb,Orb_NN)
                                    BSum += ME
                                end
                            end

                            @inbounds for c in Hole.p
                                a_c, l_c, j_c = c.a, c.l, c.j

                                @inbounds for d in Hole.p
                                    a_d, l_d, j_d = d.a, d.l, d.j

                                    if P_pq != (rem(l_c + l_d,2) + 1) || abs(j_c - j_d) > 2*I || (j_c + j_d) < 2*I
                                        continue
                                    end

                                    ME = Amp * O2b_pp(a_h,a_g,a_c,a_d,I,P_pq,V_NN,Orb,Orb_NN) * O2b_pp(a_p,a_q,a_c,a_d,I,P_pq,Sigma_NN,Orb,Orb_NN)
                                    BSum += ME
                                end
                            end

                        end

                        if t_ph == -1 && t_qg == 1
                            Amp = Float64(2*I + 1) * phase(I + J + div(j_h + j_q,2)) * f6j(j_p,j_h,2*J,j_g,j_q,2*I) / Float64(2*J + 1)

                            @inbounds for c in Particle.p
                                a_c, l_c, j_c = c.a, c.l, c.j

                                @inbounds for d in Particle.n
                                    a_d, l_d, j_d = d.a, d.l, d.j

                                    if P_pq != (rem(l_c + l_d,2) + 1) || abs(j_c - j_d) > 2*I || (j_c + j_d) < 2*I
                                        continue
                                    end

                                    ME = Amp * O2b_pn(a_p,a_q,a_c,a_d,I,P_pq,V_NN,Orb_NN) * O2b_pn(a_h,a_g,a_c,a_d,I,P_pq,Sigma_NN,Orb_NN)
                                    BSum += ME
                                end
                            end

                            @inbounds for c in Hole.p
                                a_c, l_c, j_c = c.a, c.l, c.j

                                @inbounds for d in Hole.n
                                    a_d, l_d, j_d = d.a, d.l, d.j

                                    if P_pq != (rem(l_c + l_d,2) + 1) || abs(j_c - j_d) > 2*I || (j_c + j_d) < 2*I
                                        continue
                                    end

                                    ME = Amp * O2b_pn(a_h,a_g,a_c,a_d,I,P_pq,V_NN,Orb_NN) * O2b_pn(a_p,a_q,a_c,a_d,I,P_pq,Sigma_NN,Orb_NN)
                                    BSum += ME
                                end
                            end
                        end

                        #=
                        if t_ph == 1 && t_qg == -1
                            Amp = Float64(2*I + 1) * phase(I + J + div(j_g + j_p,2)) * f6j(j_q,j_g,2*J,j_h,j_p,2*I) / Float64(2*J + 1)

                            @inbounds for c in Particle.p
                                a_c, l_c, j_c = c.a, c.l, c.j

                                @inbounds for d in Particle.n
                                    a_d, l_d, j_d = d.a, d.l, d.j

                                    if P_pq != (rem(l_c + l_d,2) + 1) || abs(j_c - j_d) > 2*I || (j_c + j_d) < 2*I
                                        continue
                                    end

                                    ME = Amp * O2b_pn(a_q,a_p,a_d,a_c,I,P_pq,V_NN,Orb_NN) * O2b_pn(a_g,a_h,a_d,a_c,I,P_pq,Sigma_NN,Orb_NN)
                                    BSum += ME
                                end
                            end

                            @inbounds for c in Hole.p
                                a_c, l_c, j_c = c.a, c.l, c.j

                                @inbounds for d in Hole.n
                                    a_d, l_d, j_d = d.a, d.l, d.j

                                    if P_pq != (rem(l_c + l_d,2) + 1) || abs(j_c - j_d) > 2*I || (j_c + j_d) < 2*I
                                        continue
                                    end

                                    ME = Amp * O2b_pn(a_g,a_h,a_d,a_c,I,P_pq,V_NN,Orb_NN) * O2b_pn(a_q,a_p,a_d,a_c,I,P_pq,Sigma_NN,Orb_NN)
                                    BSum += ME
                                end
                            end


                        end
                        =#

                        if t_ph == 1 && t_qg == 1
                            Amp = 0.5 * Float64(2*I + 1) * phase(I + J + div(j_h + j_q,2)) * f6j(j_p,j_h,2*J,j_g,j_q,2*I) / Float64(2*J + 1)

                            @inbounds for c in Particle.n
                                a_c, l_c, j_c = c.a, c.l, c.j

                                @inbounds for d in Particle.n
                                    a_d, l_d, j_d = d.a, d.l, d.j

                                    if P_pq != (rem(l_c + l_d,2) + 1) || abs(j_c - j_d) > 2*I || (j_c + j_d) < 2*I
                                        continue
                                    end

                                    ME = Amp * O2b_nn(a_p,a_q,a_c,a_d,I,P_pq,V_NN,Orb,Orb_NN) * O2b_nn(a_h,a_g,a_c,a_d,I,P_pq,Sigma_NN,Orb,Orb_NN)
                                    BSum += ME
                                end
                            end

                            @inbounds for c in Hole.n
                                a_c, l_c, j_c = c.a, c.l, c.j

                                @inbounds for d in Hole.n
                                    a_d, l_d, j_d = d.a, d.l, d.j

                                    if P_pq != (rem(l_c + l_d,2) + 1) || abs(j_c - j_d) > 2*I || (j_c + j_d) < 2*I
                                        continue
                                    end

                                    ME = Amp * O2b_nn(a_h,a_g,a_c,a_d,I,P_pq,V_NN,Orb,Orb_NN) * O2b_nn(a_p,a_q,a_c,a_d,I,P_pq,Sigma_NN,Orb,Orb_NN)
                                    BSum += ME
                                end
                            end

                        end

                    end
                end
                    # 2-body scattering ... recoupled term ...
                        # pp terms ...
                if t_ph == -1 && t_qg == -1

                    @inbounds for b in Hole.p
                        a_b, l_b, j_b = b.a, b.l, b.j

                        P_hb = rem(l_h + l_b,2) + 1
                        P_pb = rem(l_p + l_b,2) + 1

                        P_qb = rem(l_q + l_b,2) + 1
                        P_gb = rem(l_g + l_b,2) + 1

                        @inbounds for d in Particle.p
                            a_d, l_d, j_d = d.a, d.l, d.j

                            P_qd = rem(l_q + l_d,2) + 1
                            P_gd = rem(l_g + l_d,2) + 1

                            P_pd = rem(l_p + l_d,2) + 1
                            P_hd = rem(l_h + l_d,2) + 1

                            # V_pbgd S_hgqd
                            if P_hb == P_qd && P_pb == P_gd

                                @inbounds for I in div(max(abs(j_p - j_q),abs(j_h - j_g)),2):div(min(j_p + j_q,j_h + j_g),2)
                                    Amp_I = Float64(2*I + 1) * phase(1 + I + div(j_h + j_g,2)) * f6j(j_p,j_h,2*J,j_q,j_g,2*I) / Float64(2*J + 1)

                                    @inbounds for J_1 in div(max(abs(j_p - j_b),abs(j_g - j_d)),2):div(min(j_p + j_b,j_g + j_d),2)
                                        Amp_J_1 = Float64(2*J_1 + 1) * phase(J_1) * f6j(j_p,j_g,2*I,j_d,j_b,2*J_1)

                                        @inbounds for J_2 in div(max(abs(j_b - j_h),abs(j_q - j_d)),2):div(min(j_b + j_h,j_q + j_d),2)
                                            Amp_J_2 = Float64(2*J_2 + 1) * phase(J_2) * f6j(j_h,j_q,2*I,j_d,j_b,2*J_2)

                                            Amp = Amp_I * Amp_J_1 * Amp_J_2

                                            if abs(Amp) < 1e-8
                                                continue
                                            end

                                            ME = Amp * O2b_pp(a_p,a_b,a_g,a_d,J_1,P_pb,V_NN,Orb,Orb_NN) * O2b_pp(a_h,a_b,a_q,a_d,J_2,P_hb,Sigma_NN,Orb,Orb_NN)
                                            BSum += ME
                                        end
                                    end
                                end

                            end

                            # V_qbhd S_gbpd
                            if P_hd == P_qb && P_pd == P_gb

                                @inbounds for I in div(max(abs(j_p - j_q),abs(j_h - j_g)),2):div(min(j_p + j_q,j_h + j_g),2)
                                    Amp_I = Float64(2*I + 1) * phase(1 + I + div(j_h + j_g,2)) * f6j(j_q,j_g,2*J,j_p,j_h,2*I) / Float64(2*J + 1)


                                    @inbounds for J_1 in div(max(abs(j_q -  j_b),abs(j_h - j_d)),2):div(min(j_q + j_b,j_h + j_d),2)
                                        Amp_J_1 = Float64(2*J_1 + 1) * phase(J_1) * f6j(j_q,j_h,2*I,j_d,j_b,2*J_1)

                                        @inbounds for J_2 in div(max(abs(j_g - j_b),abs(j_p - j_d)),2):div(min(j_g + j_b,j_p + j_d),2)
                                            Amp_J_2 = Float64(2*J_2 + 1) * phase(J_2) * f6j(j_g,j_p,2*I,j_d,j_b,2*J_2)

                                            Amp = Amp_I * Amp_J_1 * Amp_J_2

                                            if abs(Amp) < 1e-8
                                                continue
                                            end

                                            ME = Amp * O2b_pp(a_q,a_b,a_h,a_d,J_1,P_qb,V_NN,Orb,Orb_NN) * O2b_pp(a_g,a_b,a_p,a_d,J_2,P_gb,Sigma_NN,Orb,Orb_NN)
                                            BSum += ME
                                        end
                                    end
                                end

                            end

                        end
                    end

                    @inbounds for b in Hole.n
                        a_b, l_b, j_b = b.a, b.l, b.j

                        P_hb = rem(l_h + l_b,2) + 1
                        P_pb = rem(l_p + l_b,2) + 1

                        P_qb = rem(l_q + l_b,2) + 1
                        P_gb = rem(l_g + l_b,2) + 1

                        @inbounds for d in Particle.n
                            a_d, l_d, j_d = d.a, d.l, d.j

                            P_qd = rem(l_q + l_d,2) + 1
                            P_gd = rem(l_g + l_d,2) + 1

                            P_pd = rem(l_p + l_d,2) + 1
                            P_hd = rem(l_h + l_d,2) + 1

                            # V_pbgd S_hgqd
                            if P_hb == P_qd && P_pb == P_gd

                                @inbounds for I in div(max(abs(j_p - j_q),abs(j_h - j_g)),2):div(min(j_p + j_q,j_h + j_g),2)
                                    Amp_I = Float64(2*I + 1) * phase(1 + I + div(j_h + j_g,2)) * f6j(j_p,j_h,2*J,j_q,j_g,2*I) / Float64(2*J + 1)

                                    @inbounds for J_1 in div(max(abs(j_p - j_b),abs(j_g - j_d)),2):div(min(j_p + j_b,j_g + j_d),2)
                                        Amp_J_1 = Float64(2*J_1 + 1) * phase(J_1) * f6j(j_p,j_g,2*I,j_d,j_b,2*J_1)

                                        @inbounds for J_2 in div(max(abs(j_b - j_h),abs(j_q - j_d)),2):div(min(j_b + j_h,j_q + j_d),2)
                                            Amp_J_2 = Float64(2*J_2 + 1) * phase(J_2) * f6j(j_h,j_q,2*I,j_d,j_b,2*J_2)

                                            Amp = Amp_I * Amp_J_1 * Amp_J_2

                                            if abs(Amp) < 1e-8
                                                continue
                                            end

                                            ME = Amp * O2b_pn(a_p,a_b,a_g,a_d,J_1,P_pb,V_NN,Orb_NN) * O2b_pn(a_h,a_b,a_q,a_d,J_2,P_hb,Sigma_NN,Orb_NN)
                                            BSum += ME
                                        end
                                    end
                                end

                            end

                            # V_qbhd S_gbpd
                            if P_hd == P_qb && P_pd == P_gb

                                @inbounds for I in div(max(abs(j_p - j_q),abs(j_h - j_g)),2):div(min(j_p + j_q,j_h + j_g),2)
                                    Amp_I = Float64(2*I + 1) * phase(1 + I + div(j_h + j_g,2)) * f6j(j_q,j_g,2*J,j_p,j_h,2*I)


                                    @inbounds for J_1 in div(max(abs(j_q -  j_b),abs(j_h - j_d)),2):div(min(j_q + j_b,j_h + j_d),2)
                                        Amp_J_1 = Float64(2*J_1 + 1) * phase(J_1) * f6j(j_q,j_h,2*I,j_d,j_b,2*J_1)

                                        @inbounds for J_2 in div(max(abs(j_g - j_b),abs(j_p - j_d)),2):div(min(j_g + j_b,j_p + j_d),2)
                                            Amp_J_2 = Float64(2*J_2 + 1) * phase(J_2) * f6j(j_g,j_p,2*I,j_d,j_b,2*J_2)

                                            Amp = Amp_I * Amp_J_1 * Amp_J_2

                                            if abs(Amp) < 1e-8
                                                continue
                                            end

                                            ME = Amp * O2b_pn(a_q,a_b,a_h,a_d,J_1,P_qb,V_NN,Orb_NN) * O2b_pn(a_g,a_b,a_p,a_d,J_2,P_gb,Sigma_NN,Orb_NN)
                                            BSum += ME
                                        end
                                    end
                                end

                            end

                        end
                    end

                end
                    # nn terms ...
                if t_ph == 1 && t_qg == 1

                    @inbounds for b in Hole.n
                        a_b, l_b, j_b = b.a, b.l, b.j

                        P_hb = rem(l_h + l_b,2) + 1
                        P_pb = rem(l_p + l_b,2) + 1

                        P_qb = rem(l_q + l_b,2) + 1
                        P_gb = rem(l_g + l_b,2) + 1

                        @inbounds for d in Particle.n
                            a_d, l_d, j_d = d.a, d.l, d.j

                            P_qd = rem(l_q + l_d,2) + 1
                            P_gd = rem(l_g + l_d,2) + 1

                            P_pd = rem(l_p + l_d,2) + 1
                            P_hd = rem(l_h + l_d,2) + 1

                            # V_pbgd S_hgqd
                            if P_hb == P_qd && P_pb == P_gd

                                @inbounds for I in div(max(abs(j_p - j_q),abs(j_h - j_g)),2):div(min(j_p + j_q,j_h + j_g),2)
                                    Amp_I = Float64(2*I + 1) * phase(1 + I + div(j_h + j_g,2)) * f6j(j_p,j_h,2*J,j_q,j_g,2*I) / Float64(2*J + 1)


                                    @inbounds for J_1 in div(max(abs(j_p -  j_b),abs(j_g - j_d)),2):div(min(j_p + j_b,j_g + j_d),2)
                                        Amp_J_1 = Float64(2*J_1 + 1) * phase(J_1) * f6j(j_p,j_g,2*I,j_d,j_b,2*J_1)

                                        @inbounds for J_2 in div(max(abs(j_b - j_h),abs(j_q - j_d)),2):div(min(j_b + j_h,j_q + j_d),2)
                                            Amp_J_2 = Float64(2*J_2 + 1) * phase(J_2) * f6j(j_h,j_q,2*I,j_d,j_b,2*J_2)

                                            Amp = Amp_I * Amp_J_1 * Amp_J_2

                                            if abs(Amp) < 1e-8
                                                continue
                                            end

                                            ME = Amp * O2b_nn(a_p,a_b,a_g,a_d,J_1,P_pb,V_NN,Orb,Orb_NN) * O2b_nn(a_h,a_b,a_q,a_d,J_2,P_hb,Sigma_NN,Orb,Orb_NN)
                                            BSum += ME
                                        end
                                    end
                                end

                            end

                            # V_qbhd S_gbpd
                            if P_hd == P_qb && P_pd == P_gb

                                @inbounds for I in div(max(abs(j_p - j_q),abs(j_h - j_g)),2):div(min(j_p + j_q,j_h + j_g),2)
                                    Amp_I = Float64(2*I + 1) * phase(1 + I + div(j_h + j_g,2)) * f6j(j_q,j_g,2*J,j_p,j_h,2*I) / Float64(2*J + 1)

                                    @inbounds for J_1 in div(max(abs(j_q -  j_b),abs(j_h - j_d)),2):div(min(j_q + j_b,j_h + j_d),2)
                                        Amp_J_1 = Float64(2*J_1 + 1) * phase(J_1) * f6j(j_q,j_h,2*I,j_d,j_b,2*J_1)

                                        @inbounds for J_2 in div(max(abs(j_g - j_b),abs(j_p - j_d)),2):div(min(j_g + j_b,j_p + j_d),2)
                                            Amp_J_2 = Float64(2*J_2 + 1) * phase(J_2) * f6j(j_g,j_p,2*I,j_d,j_b,2*J_2)

                                            Amp = Amp_I * Amp_J_1 * Amp_J_2

                                            if abs(Amp) < 1e-8
                                                continue
                                            end

                                            ME = Amp * O2b_nn(a_q,a_b,a_h,a_d,J_1,P_qb,V_NN,Orb,Orb_NN) * O2b_nn(a_g,a_b,a_p,a_d,J_2,P_gb,Sigma_NN,Orb,Orb_NN)
                                            BSum += ME
                                        end
                                    end
                                end

                            end

                        end
                    end

                    @inbounds for b in Hole.p
                        a_b, l_b, j_b = b.a, b.l, b.j

                        P_hb = rem(l_h + l_b,2) + 1
                        P_pb = rem(l_p + l_b,2) + 1

                        P_qb = rem(l_q + l_b,2) + 1
                        P_gb = rem(l_g + l_b,2) + 1

                        @inbounds for d in Particle.p
                            a_d, l_d, j_d = d.a, d.l, d.j

                            P_qd = rem(l_q + l_d,2) + 1
                            P_gd = rem(l_g + l_d,2) + 1

                            P_pd = rem(l_p + l_d,2) + 1
                            P_hd = rem(l_h + l_d,2) + 1

                            # V_pbgd S_hgqd
                            if P_hb == P_qd && P_pb == P_gd

                                @inbounds for I in div(max(abs(j_p - j_q),abs(j_h - j_g)),2):div(min(j_p + j_q,j_h + j_g),2)
                                    Amp_I = Float64(2*I + 1) * phase(1 + I + div(j_h + j_g,2)) * f6j(j_p,j_h,2*J,j_q,j_g,2*I) / Float64(2*J + 1)

                                    @inbounds for J_1 in div(max(abs(j_p -  j_b),abs(j_g - j_d)),2):div(min(j_p + j_b,j_g + j_d),2)
                                        Amp_J_1 = Float64(2*J_1 + 1) * phase(J_1) * f6j(j_p,j_g,2*I,j_d,j_b,2*J_1)

                                        @inbounds for J_2 in div(max(abs(j_b - j_h),abs(j_q - j_d)),2):div(min(j_b + j_h,j_q + j_d),2)
                                            Amp_J_2 = Float64(2*J_2 + 1) * phase(J_2) * f6j(j_h,j_q,2*I,j_d,j_b,2*J_2)

                                            Amp = Amp_I * Amp_J_1 * Amp_J_2

                                            if abs(Amp) < 1e-8
                                                continue
                                            end

                                            ME = Amp * O2b_pn(a_b,a_p,a_d,a_g,J_1,P_pb,V_NN,Orb_NN) * O2b_pn(a_b,a_h,a_d,a_q,J_2,P_hb,Sigma_NN,Orb_NN)
                                            BSum += ME
                                        end
                                    end
                                end

                            end

                            # V_qbhd S_gbpd
                            if P_hd == P_qb && P_pd == P_gb

                                @inbounds for I in div(max(abs(j_p - j_q),abs(j_h - j_g)),2):div(min(j_p + j_q,j_h + j_g),2)
                                    Amp_I = Float64(2*I + 1) * phase(1 + I + div(j_h + j_g,2)) * f6j(j_q,j_g,2*J,j_p,j_h,2*I) / Float64(2*J + 1)

                                    @inbounds for J_1 in div(max(abs(j_q -  j_b),abs(j_h - j_d)),2):div(min(j_q + j_b,j_h + j_d),2)
                                        Amp_J_1 = Float64(2*J_1 + 1) * phase(J_1) * f6j(j_q,j_h,2*I,j_d,j_b,2*J_1)

                                        @inbounds for J_2 in div(max(abs(j_g - j_b),abs(j_p - j_d)),2):div(min(j_g + j_b,j_p + j_d),2)
                                            Amp_J_2 = Float64(2*J_2 + 1) * phase(J_2) * f6j(j_g,j_p,2*I,j_d,j_b,2*J_2)

                                            Amp = Amp_I * Amp_J_1 * Amp_J_2

                                            if abs(Amp) < 1e-8
                                                continue
                                            end

                                            ME = Amp * O2b_pn(a_b,a_q,a_d,a_h,J_1,P_qb,V_NN,Orb_NN) * O2b_pn(a_b,a_g,a_d,a_p,J_2,P_gb,Sigma_NN,Orb_NN)
                                            BSum += ME
                                        end
                                    end
                                end

                            end

                        end
                    end

                end
                    # pn terms ...
                if t_ph == -1 && t_qg == 1

                    @inbounds for b in Hole.n
                        a_b, l_b, j_b = b.a, b.l, b.j

                        P_hb = rem(l_h + l_b,2) + 1
                        P_pb = rem(l_p + l_b,2) + 1

                        @inbounds for c in Particle.p
                            a_c, l_c, j_c = c.a, c.l, c.j

                            P_qc = rem(l_q + l_c,2) + 1
                            P_gc = rem(l_g + l_c,2) + 1

                            # V_pbcg S_hgcq
                            if P_hb == P_qc && P_pb == P_gc

                                @inbounds for I in div(max(abs(j_p - j_q),abs(j_h - j_g)),2):div(min(j_p + j_q,j_h + j_g),2)
                                    Amp_I = Float64(2*I + 1) * phase(I + div(j_h + j_q,2)) * f6j(j_p,j_g,2*I,j_q,j_h,2*J) / Float64(2*J + 1)


                                    @inbounds for J_1 in div(max(abs(j_p -  j_b),abs(j_g - j_c)),2):div(min(j_p + j_b,j_g + j_c),2)
                                        Amp_J_1 = Float64(2*J_1 + 1) * f6j(j_p,j_g,2*I,j_c,j_b,2*J_1)

                                        @inbounds for J_2 in div(max(abs(j_h - j_b),abs(j_q - j_c)),2):div(min(j_h + j_b,j_q + j_c),2)
                                            Amp_J_2 = Float64(2*J_2 + 1)* f6j(j_h,j_q,2*I,j_c,j_b,2*J_2)

                                            Amp = Amp_I * Amp_J_1 * Amp_J_2

                                            if abs(Amp) < 1e-8
                                                continue
                                            end

                                            ME = Amp * O2b_pn(a_p,a_b,a_c,a_g,J_1,P_pb,V_NN,Orb_NN) * O2b_pn(a_h,a_b,a_c,a_q,J_2,P_hb,Sigma_NN,Orb_NN)
                                            BSum += ME
                                        end
                                    end
                                end

                            end

                            # V_hbcq S_pbcg
                            if P_pb == P_gc && P_hb == P_qc

                                @inbounds for I in div(max(abs(j_p - j_q),abs(j_h - j_g)),2):div(min(j_p + j_q,j_h + j_g),2)
                                    Amp_I = Float64(2*I + 1) * phase(I + div(j_h + j_q,2)) * f6j(j_p,j_g,2*I,j_q,j_h,2*J) / Float64(2*J + 1)


                                    @inbounds for J_1 in div(max(abs(j_h -  j_b),abs(j_q - j_c)),2):div(min(j_h + j_b,j_q + j_c),2)
                                        Amp_J_1 = Float64(2*J_1 + 1) * f6j(j_h,j_q,2*I,j_c,j_b,2*J_1)

                                        @inbounds for J_2 in div(max(abs(j_p - j_b),abs(j_q - j_g)),2):div(min(j_p + j_b,j_g + j_c),2)
                                            Amp_J_2 = Float64(2*J_2 + 1)* f6j(j_p,j_g,2*I,j_c,j_b,2*J_2)

                                            Amp = Amp_I * Amp_J_1 * Amp_J_2

                                            if abs(Amp) < 1e-8
                                                continue
                                            end

                                            ME = Amp * O2b_pn(a_h,a_b,a_c,a_q,J_1,P_hb,V_NN,Orb_NN) * O2b_pn(a_p,a_b,a_c,a_g,J_2,P_pb,Sigma_NN,Orb_NN)
                                            BSum += ME
                                        end
                                    end
                                end

                            end
                        end
                    end
                end
                =#

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