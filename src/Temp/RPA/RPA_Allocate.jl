function RPA_allocate(Params::Parameters,N_nu::Matrix{Int64},Orb_Phonon::Matrix{Vector{Int64}},Phonon::Vector{PhState},Particle::pnSVector,Hole::pnSVector,Orb::Vector{Orb1B},Orb_NN::Orb2B,h_N::O1B,V_NN::O2B,Rho::O1B)
    # Read calculation parameters ...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1

    JP_list = JP_initialize(J_max)
    
    println("\tAllocating RPA matrices A & B ...")

    # Initialize matrices A & B ...
    A = Matrix{Matrix{Float64}}(undef,J_max+1,2)
    B = Matrix{Matrix{Float64}}(undef,J_max+1,2)
    N = Matrix{Matrix{Float64}}(undef,J_max+1,2)

    # Allocate matrices A & B ...
    @inbounds Threads.@threads for JP in JP_list
        J, P = JP[1], JP[2]
        N_ph = N_nu[J+1,P]

        A_JP = zeros(Float64,N_ph,N_ph)
        B_JP = zeros(Float64,N_ph,N_ph)
        N_JP = zeros(Float64,N_ph,N_ph)

        @inbounds for ind_ph in 1:N_ph
            ph = Orb_Phonon[J+1,P][ind_ph]
            p, h = Phonon[ph].p, Phonon[ph].h
            t_ph, J, P = Phonon[ph].tz, Phonon[ph].J, Phonon[ph].P
            n_p, n_h = 0.0, 0.0

            if t_ph == -1
                a_p, l_p, j_p = Particle.p[p].a, Particle.p[p].l, Particle.p[p].j
                a_h, l_h, j_h = Hole.p[h].a, Hole.p[h].l, Hole.p[h].j
                n_p, n_h = Rho.p[a_p,a_p], Rho.p[a_h,a_h]
            elseif t_ph == 1
                a_p, l_p, j_p = Particle.n[p].a, Particle.n[p].l, Particle.n[p].j
                a_h, l_h, j_h = Hole.n[h].a, Hole.n[h].l, Hole.n[h].j
                n_p, n_h = Rho.n[a_p,a_p], Rho.n[a_h,a_h]
            end

            dn_ph = n_h - n_p

            @inbounds for ind_qg in 1:N_ph
                qg = Orb_Phonon[J+1,P][ind_qg]
                q, g = Phonon[qg].p, Phonon[qg].h
                t_qg = Phonon[qg].tz
                n_q, n_g = 0.0, 0.0

                if t_qg == -1
                    a_q, l_q, j_q = Particle.p[q].a, Particle.p[q].l, Particle.p[q].j
                    a_g, l_g, j_g = Hole.p[g].a, Hole.p[g].l, Hole.p[g].j
                    n_q, n_g = Rho.p[a_q,a_q], Rho.p[a_g,a_g]
                elseif t_qg == 1
                    a_q, l_q, j_q = Particle.n[q].a, Particle.n[q].l, Particle.n[q].j
                    a_g, l_g, j_g = Hole.n[g].a, Hole.n[g].l, Hole.n[g].j
                    n_q, n_g = Rho.n[a_q,a_q], Rho.n[a_g,a_g]
                end

                delta_pq = Float64(kronecker_delta(a_p,a_q))
                delta_hg = Float64(kronecker_delta(a_h,a_g))

                dn_qg = n_g - n_q

                P_pg = rem(l_p + l_g,2) + 1
                P_pq = rem(l_p + l_q,2) + 1
    
                ASum = 0.0
                BSum = 0.0
                NSum = 0.0

                # 1-body mean-field term h & the overlap matrix N ...
                if t_ph == t_qg
                    if t_ph == -1
                        Amp = 0.5 * (dn_ph + dn_qg) * (h_N.p[a_p,a_q] * delta_hg - h_N.p[a_h,a_g] * delta_pq)
                        ASum += Amp

                        NME = dn_ph * delta_pq * delta_hg
                        NSum += NME
                    elseif t_ph == 1
                        Amp = 0.5 * (dn_ph + dn_qg) * (h_N.n[a_p,a_q] * delta_hg - h_N.n[a_h,a_g] * delta_pq)
                        ASum += Amp

                        NME = dn_ph * delta_pq * delta_hg
                        NSum += NME
                    end
                end

                # 2-body TDA matrix A ...
                if rem(l_p + l_g, 2) == rem(l_q + l_h, 2)
                    @inbounds for J_r in div(max(abs(j_p - j_g),abs(j_q - j_h)),2):div(min(j_p + j_g,j_q + j_h),2)
                        hat_J_r = Float64(2*J_r + 1)
                        if t_ph == -1 && t_qg == -1
                            Amp = dn_ph * dn_qg * phase(div(j_h + j_q,2) + J_r) * hat_J_r * f6j(j_p,j_h,2*J,j_q,j_g,2*J_r)
                            ME = Amp * O2b_pp(a_p,a_g,a_h,a_q,J_r,P_pg,V_NN,Orb,Orb_NN)
                            ASum +=  ME
                        elseif t_ph == -1 && t_qg == 1
                            Amp = dn_ph * dn_qg * phase(div(j_h + j_q,2) + J_r) * hat_J_r * f6j(j_p,j_h,2*J,j_q,j_g,2*J_r)
                            ME = Amp * O2b_pn(a_p,a_g,a_h,a_q,J_r,P_pg,V_NN,Orb_NN)
                            ASum += ME
                        elseif t_ph == 1 && t_qg == -1
                            Amp = dn_ph * dn_qg * phase(div(j_g + j_p,2) + J_r) * hat_J_r * f6j(j_q,j_g,2*J,j_p,j_h,2*J_r)
                            ME = Amp * O2b_pn(a_q,a_h,a_g,a_p,J_r,P_pg,V_NN,Orb_NN)
                            ASum += ME
                        elseif t_ph == 1 && t_qg == 1
                            Amp = dn_ph * dn_qg * phase(div(j_h + j_q,2) + J_r) * hat_J_r * f6j(j_p,j_h,2*J,j_q,j_g,2*J_r)
                            ME = Amp * O2b_nn(a_p,a_g,a_h,a_q,J_r,P_pg,V_NN,Orb,Orb_NN)
                            ASum +=  ME
                        end
                    end
                end

                # 2-body correlation matrix B ...
                if rem(l_p + l_q, 2) == rem(l_h + l_g, 2)
                    @inbounds for J_r in div(max(abs(j_p - j_q),abs(j_h - j_g)),2):div(min(j_p + j_q,j_h + j_g),2)
                        hat_J_r = Float64(2*J_r + 1)
                        if t_ph == -1 && t_qg == -1
                            Amp = dn_ph * dn_qg * phase(div(j_h + j_q,2) + J + J_r) * hat_J_r * f6j(j_p,j_h,2*J,j_g,j_q,2*J_r) 
                            ME = Amp * O2b_pp(a_p,a_q,a_h,a_g,J_r,P_pq,V_NN,Orb,Orb_NN)
                            BSum += ME
                        elseif t_ph == -1 && t_qg == 1
                            Amp = dn_ph * dn_qg * phase(div(j_h + j_q,2) + J + J_r) * hat_J_r * f6j(j_p,j_h,2*J,j_g,j_q,2*J_r)
                            ME = Amp * O2b_pn(a_p,a_q,a_h,a_g,J_r,P_pq,V_NN,Orb_NN)
                            BSum += ME
                        elseif t_ph == 1 && t_qg == -1
                            Amp = dn_ph * dn_qg * phase(div(j_g + j_p,2) + J + J_r) * hat_J_r * f6j(j_q,j_g,2*J,j_h,j_p,2*J_r)
                            ME = Amp * O2b_pn(a_q,a_p,a_g,a_h,J_r,P_pq,V_NN,Orb_NN)
                            BSum += ME
                        elseif t_ph == 1 && t_qg == 1
                            Amp = dn_ph * dn_qg * phase(div(j_h + j_q,2) + J + J_r) * hat_J_r * f6j(j_p,j_h,2*J,j_g,j_q,2*J_r)
                            ME = Amp * O2b_nn(a_p,a_q,a_h,a_g,J_r,P_pq,V_NN,Orb,Orb_NN)
                            BSum += ME
                        end
                    end
                end

                A_JP[ind_ph,ind_qg] = ASum
                B_JP[ind_ph,ind_qg] = BSum * 0.0
                N_JP[ind_ph,ind_qg] = NSum
            end
        end

        # For given J & P allocate A & B ...
        A[J+1,P] = A_JP
        B[J+1,P] = B_JP
        N[J+1,P] = N_JP
    end

    println("\tMatrices A, B & N allocated ...")

    return A, B, N
end

function RPA_allocate(Params::Parameters,N_nu::Matrix{Int64},Orb_Phonon::Matrix{Vector{Int64}},Phonon::Vector{PhState},Particle::pnSVector,Hole::pnSVector,Orb::Vector{Orb1B},Orb_NN::Orb2B,h_N::O1B,V_NN::O2B,Rho::O1B,Sigma_NN::O2B)
    # Read calculation parameters ...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1

    JP_list = JP_initialize(J_max)
    
    println("\tAllocating RPA matrices A & B ...")

    # Initialize matrices A & B ...
    A = Matrix{Matrix{Float64}}(undef,J_max+1,2)
    B = Matrix{Matrix{Float64}}(undef,J_max+1,2)
    N = Matrix{Matrix{Float64}}(undef,J_max+1,2)

    # Allocate matrices A & B ...
    @inbounds Threads.@threads for JP in JP_list
        J, P = JP[1], JP[2]
        N_ph = N_nu[J+1,P]

        A_JP = zeros(Float64,N_ph,N_ph)
        B_JP = zeros(Float64,N_ph,N_ph)
        N_JP = zeros(Float64,N_ph,N_ph)

        @inbounds for ind_ph in 1:N_ph
            ph = Orb_Phonon[J+1,P][ind_ph]
            p, h = Phonon[ph].p, Phonon[ph].h
            t_ph, J, P = Phonon[ph].tz, Phonon[ph].J, Phonon[ph].P
            n_p, n_h = 0.0, 0.0

            if t_ph == -1
                a_p, l_p, j_p = Particle.p[p].a, Particle.p[p].l, Particle.p[p].j
                a_h, l_h, j_h = Hole.p[h].a, Hole.p[h].l, Hole.p[h].j
                n_p, n_h = Rho.p[a_p,a_p], Rho.p[a_h,a_h]
            elseif t_ph == 1
                a_p, l_p, j_p = Particle.n[p].a, Particle.n[p].l, Particle.n[p].j
                a_h, l_h, j_h = Hole.n[h].a, Hole.n[h].l, Hole.n[h].j
                n_p, n_h = Rho.n[a_p,a_p], Rho.n[a_h,a_h]
            end

            dn_ph = n_h - n_p

            @inbounds for ind_qg in 1:N_ph
                qg = Orb_Phonon[J+1,P][ind_qg]
                q, g = Phonon[qg].p, Phonon[qg].h
                t_qg = Phonon[qg].tz
                n_q, n_g = 0.0, 0.0

                if ind_ph > ind_qg
                    continue
                end

                if t_qg == -1
                    a_q, l_q, j_q = Particle.p[q].a, Particle.p[q].l, Particle.p[q].j
                    a_g, l_g, j_g = Hole.p[g].a, Hole.p[g].l, Hole.p[g].j
                    n_q, n_g = Rho.p[a_q,a_q], Rho.p[a_g,a_g]
                elseif t_qg == 1
                    a_q, l_q, j_q = Particle.n[q].a, Particle.n[q].l, Particle.n[q].j
                    a_g, l_g, j_g = Hole.n[g].a, Hole.n[g].l, Hole.n[g].j
                    n_q, n_g = Rho.n[a_q,a_q], Rho.n[a_g,a_g]
                end

                delta_pq = Float64(kronecker_delta(a_p,a_q))
                delta_hg = Float64(kronecker_delta(a_h,a_g))

                dn_qg = n_g - n_q

                P_pg = rem(l_p + l_g,2) + 1
                P_pq = rem(l_p + l_q,2) + 1
    
                ASum = 0.0
                BSum = 0.0
                NSum = 0.0

                # 1-body mean-field term h & the overlap matrix N ...
                if t_ph == t_qg
                    if t_ph == -1
                        Amp = 0.5 * (dn_ph + dn_qg) * (h_N.p[a_p,a_q] * delta_hg - h_N.p[a_h,a_g] * delta_pq)
                        ASum += Amp

                        NME = dn_ph * delta_pq * delta_hg
                        NSum += NME
                    elseif t_ph == 1
                        Amp = 0.5 * (dn_ph + dn_qg) * (h_N.n[a_p,a_q] * delta_hg - h_N.n[a_h,a_g] * delta_pq)
                        ASum += Amp

                        NME = dn_ph * delta_pq * delta_hg
                        NSum += NME
                    end
                end

                # 2-body TDA matrix A ...
                    # 1-body scattering ...
                if rem(l_p + l_g, 2) == rem(l_q + l_h, 2)
                    @inbounds for J_r in div(max(abs(j_p - j_g),abs(j_q - j_h)),2):div(min(j_p + j_g,j_q + j_h),2)
                        hat_J_r = Float64(2*J_r + 1)
                        if t_ph == -1 && t_qg == -1
                            Amp = dn_ph * dn_qg * phase(div(j_h + j_q,2) + J_r) * hat_J_r * f6j(j_p,j_h,2*J,j_q,j_g,2*J_r)
                            ME = Amp * O2b_pp(a_p,a_g,a_h,a_q,J_r,P_pg,V_NN,Orb,Orb_NN)
                            ASum +=  ME
                        elseif t_ph == -1 && t_qg == 1
                            Amp = dn_ph * dn_qg * phase(div(j_h + j_q,2) + J_r) * hat_J_r * f6j(j_p,j_h,2*J,j_q,j_g,2*J_r)
                            ME = Amp * O2b_pn(a_p,a_g,a_h,a_q,J_r,P_pg,V_NN,Orb_NN)
                            ASum += ME
                        elseif t_ph == 1 && t_qg == -1
                            Amp = dn_ph * dn_qg * phase(div(j_g + j_p,2) + J_r) * hat_J_r * f6j(j_q,j_g,2*J,j_p,j_h,2*J_r)
                            ME = Amp * O2b_pn(a_q,a_h,a_g,a_p,J_r,P_pg,V_NN,Orb_NN)
                            ASum += ME
                        elseif t_ph == 1 && t_qg == 1
                            Amp = dn_ph * dn_qg * phase(div(j_h + j_q,2) + J_r) * hat_J_r * f6j(j_p,j_h,2*J,j_q,j_g,2*J_r)
                            ME = Amp * O2b_nn(a_p,a_g,a_h,a_q,J_r,P_pg,V_NN,Orb,Orb_NN)
                            ASum +=  ME
                        end
                    end
                end
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
                
                # 2-body correlation matrix B ...
                    # 1-body scattering ...
                if rem(l_p + l_q, 2) == rem(l_h + l_g, 2)
                    @inbounds for J_r in div(max(abs(j_p - j_q),abs(j_h - j_g)),2):div(min(j_p + j_q,j_h + j_g),2)
                        hat_J_r = Float64(2*J_r + 1)
                        if t_ph == -1 && t_qg == -1
                            Amp = dn_ph * dn_qg * phase(div(j_h + j_q,2) + J + J_r) * hat_J_r * f6j(j_p,j_h,2*J,j_g,j_q,2*J_r) 
                            ME = Amp * O2b_pp(a_p,a_q,a_h,a_g,J_r,P_pq,V_NN,Orb,Orb_NN)
                            BSum += ME
                        elseif t_ph == -1 && t_qg == 1
                            Amp = dn_ph * dn_qg * phase(div(j_h + j_q,2) + J + J_r) * hat_J_r * f6j(j_p,j_h,2*J,j_g,j_q,2*J_r)
                            ME = Amp * O2b_pn(a_p,a_q,a_h,a_g,J_r,P_pq,V_NN,Orb_NN)
                            BSum += ME
                        elseif t_ph == 1 && t_qg == -1
                            Amp = dn_ph * dn_qg * phase(div(j_g + j_p,2) + J + J_r) * hat_J_r * f6j(j_q,j_g,2*J,j_h,j_p,2*J_r)
                            ME = Amp * O2b_pn(a_q,a_p,a_g,a_h,J_r,P_pq,V_NN,Orb_NN)
                            BSum += ME
                        elseif t_ph == 1 && t_qg == 1
                            Amp = dn_ph * dn_qg * phase(div(j_h + j_q,2) + J + J_r) * hat_J_r * f6j(j_p,j_h,2*J,j_g,j_q,2*J_r)
                            ME = Amp * O2b_nn(a_p,a_q,a_h,a_g,J_r,P_pq,V_NN,Orb,Orb_NN)
                            BSum += ME
                        end
                    end
                end
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

                A_JP[ind_ph,ind_qg] = ASum
                B_JP[ind_ph,ind_qg] = BSum
                N_JP[ind_ph,ind_qg] = NSum

                if ind_ph != ind_qg
                    A_JP[ind_qg,ind_ph] = ASum
                    B_JP[ind_qg,ind_ph] = BSum
                    N_JP[ind_qg,ind_ph] = NSum
                end
            end
        end

        # Check symmetries of A, B and N ...
        if !(isapprox(A_JP,A_JP',atol = 1e-5))
            display("\t\tA is not a symmetric matrix! ... JP = $(J)$(P)")
        end

        if !(isapprox(B_JP,B_JP',atol = 1e-5))
            display("\t\tB is not a symmetric matrix! ... JP = $(J)$(P)")
        end

        if !(isapprox(N_JP,N_JP',atol = 1e-5))
            display("\t\tN is not a symmetric matrix! ... JP = $(J)$(P)")
        end

        # For given J & P allocate A & B ...
        A[J+1,P] = A_JP
        B[J+1,P] = B_JP
        N[J+1,P] = N_JP
    end

    println("\tMatrices A, B & N allocated ...")

    return A, B, N
end

function V2b_no2b(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,Orb_NNN::Orb3B,V_NN::O2B,V_NNN::Array{Vector{Vector{Float32}},4},Rho::O1B,C::O1B)
    # Make density-dependent residual NN interaction ... NO2B approximation ...
    @time V_NN = V2b_residual_no2b(Params,Orb,Orb_NN,Orb_NNN,Rho,V_NN,V_NNN)

    # Transform the density-dependent NN interaction to the target basis ...
    @time V_NN_no2b = O2b_transformation(Params,Orb,Orb_NN,V_NN,C)

    return V_NN_no2b, Orb_NN
end

function RPA_sigma2b_allocate(Params::Parameters,N_nu::Matrix{Int64},Orb_Phonon::Matrix{Vector{Int64}},Phonon::Vector{PhState},N_Particle::pnInteger,Particle::pnSVector,N_Hole::pnInteger,Hole::pnSVector,Orb::Vector{Orb1B},Orb_NN::Orb2B,X_RPA::Matrix{Matrix{ComplexF64}},Y_RPA::Matrix{Matrix{ComplexF64}})
    # Read parameters ...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1
    a_max = div((N_max + 1)*(N_max + 2),2)

    # Preallocate arrays of allowed values of J & P ...
    JP_list = JP_initialize(Params.Calc.N2max + 1)

    # Make Particle-Hole orbitals list ...
    ParticleHole = orbitals_ph_list(J_max,N_Particle,Particle,N_Hole,Hole)
    
    # Initialite the 2-body NN correlation matrix Sigma_NN ...
    Sigma_NN = O2b_initialize(Params,Orb)

    function orbitals_ph_mappping(Params::Parameters,N_Particle::pnInteger,Particle::pnSVector,N_Hole::pnInteger,Hole::pnSVector)
        # Read parameters ...
        N_max = Params.Calc.Nmax
        a_max = div((N_max + 1)*(N_max + 2),2)

        orbitals_ph_list = Vector{Vector{Int64}}(undef,a_max)

        @inbounds for a in 1:a_max
            orbitals_ph_list[a] = Vector{Int64}(undef,2)
        end

        @inbounds for p in 1:N_Particle.p
            a = Particle.p[p].a
            orbitals_ph_list[a][1] = p
        end

        @inbounds for h in 1:N_Hole.p
            a = Hole.p[h].a
            orbitals_ph_list[a][1] = h
        end

        @inbounds for p in 1:N_Particle.n
            a = Particle.n[p].a
            orbitals_ph_list[a][2] = p
        end

        @inbounds for h in 1:N_Hole.n
            a = Hole.n[h].a
            orbitals_ph_list[a][2] = h
        end

        return orbitals_ph_list
    end

    ph_list = orbitals_ph_mappping(Params,N_Particle,Particle,N_Hole,Hole)

    println("\nAllocating 2-body RPA correlation matrix Sigma_NN ...")

    @inbounds Threads.@threads for JP in JP_list;
        J, P = JP[1], JP[2]
        if P == 1
            println("\tCalculating   ...   J = " * string(J) * "/" * string(N_2max+1) * "\tP = +")
        else
            println("\tCalculating   ...   J = " * string(J) * "/" * string(N_2max+1) * "\tP = -")
        end
        N_T0, N_T1 = Orb_NN.N[1,P,J+1], Orb_NN.N[2,P,J+1]
        @inbounds for Bra in 1:max(N_T0, N_T1)
            @inbounds for Ket in 1:Bra

                # Case of pn interaction ... T = 0
                if Bra <= N_T0
                    Ind = Bra + (Ket - 1) * N_T0 - div(Ket * (Ket - 1),2)
                    a, b = Orb_NN.Ind[1,P,J+1][Bra][1], Orb_NN.Ind[1,P,J+1][Bra][2]
                    c, d = Orb_NN.Ind[1,P,J+1][Ket][1], Orb_NN.Ind[1,P,J+1][Ket][2]
                    l_a, j_a = Orb[a].l, Orb[a].j
                    l_b, j_b = Orb[b].l, Orb[b].j
                    l_c, j_c = Orb[c].l, Orb[c].j
                    l_d, j_d = Orb[d].l, Orb[d].j

                    pO_a, nO_b = Orb[a].pO, Orb[b].nO
                    pO_c, nO_d = Orb[c].pO, Orb[d].nO

                    pnSigmaSum = 0.0

                    # 2p-2h terms ...
                    if ((pO_a + nO_b) == 0 && (pO_c + nO_d) == 2) || ((pO_a + nO_b) == 2 && (pO_c + nO_d) == 0)

                        if (pO_a + nO_b) == 0 && (pO_c + nO_d) == 2
                            p, l_p, j_p = a, l_a, j_a
                            q, l_q, j_q = b, l_b, j_b
                            h, l_h, j_h = c, l_c, j_c
                            g, l_g, j_g = d, l_d, j_d
                        else
                            p, l_p, j_p = c, l_c, j_c
                            q, l_q, j_q = d, l_d, j_d
                            h, l_h, j_h = a, l_a, j_a
                            g, l_g, j_g = b, l_b, j_b
                        end

                        p_ind, q_ind = ph_list[p][1], ph_list[q][2]
                        h_ind, g_ind = ph_list[h][1], ph_list[g][2]

                        pnSigmaME = RPA_TBDM_2p2h_T0(p_ind,l_p,j_p,q_ind,l_q,j_q,h_ind,l_h,j_h,g_ind,l_g,j_g,J,X_RPA,Y_RPA,N_nu,ParticleHole)
                        pnSigmaSum += pnSigmaME
                    end
                    # 1p1-h terms ... no proton-neutron 1p-1h 2-body correlation function terms ...

                    @inbounds Sigma_NN.pn[P,J+1][Ind] = pnSigmaSum

                end

                # Case of pp & nn interaction ... T = 1
                if Bra <= N_T1
                    Ind = Bra + (Ket - 1) * N_T1 - div(Ket * (Ket - 1),2)
                    a, b = Orb_NN.Ind[2,P,J+1][Bra][1], Orb_NN.Ind[2,P,J+1][Bra][2]
                    c, d = Orb_NN.Ind[2,P,J+1][Ket][1], Orb_NN.Ind[2,P,J+1][Ket][2]
                    l_a, j_a = Orb[a].l, Orb[a].j
                    l_b, j_b = Orb[b].l, Orb[b].j
                    l_c, j_c = Orb[c].l, Orb[c].j
                    l_d, j_d = Orb[d].l, Orb[d].j

                    pO_a, pO_b = Orb[a].pO, Orb[b].pO
                    pO_c, pO_d = Orb[c].pO, Orb[d].pO

                    nO_a, nO_b = Orb[a].nO, Orb[b].nO
                    nO_c, nO_d = Orb[c].nO, Orb[d].nO

                    ppSigmaSum, nnSigmaSum = 0.0, 0.0

                    # 2p-2h channel ...
                        # proton-proton terms ...
                        if ((pO_a + pO_b) == 0 && (pO_c + pO_d) == 2) || ((pO_a + pO_b) == 2 && (pO_c + pO_d) == 0)

                            if (pO_a + pO_b) == 0 && (pO_c + pO_d) == 2
                                p, l_p, j_p = a, l_a, j_a
                                q, l_q, j_q = b, l_b, j_b
                                h, l_h, j_h = c, l_c, j_c
                                g, l_g, j_g = d, l_d, j_d
                            else
                                p, l_p, j_p = c, l_c, j_c
                                q, l_q, j_q = d, l_d, j_d
                                h, l_h, j_h = a, l_a, j_a
                                g, l_g, j_g = b, l_b, j_b
                            end

                            p_ind, q_ind = ph_list[p][1], ph_list[q][1]
                            h_ind, g_ind = ph_list[h][1], ph_list[g][1]

                            #C += 1
                            #me1 = RPA_TBDM_2p2h_T1(p_ind,l_p,j_p,q_ind,l_q,j_q,h_ind,l_h,j_h,g_ind,l_g,j_g,J,-1,X_RPA,Y_RPA,N_nu,ParticleHole)
                            #me2 = RPA_TBDM_2p2h_T1(p_ind,l_p,j_p,q_ind,l_q,j_q,g_ind,l_g,j_g,h_ind,l_h,j_h,J,-1,X_RPA,Y_RPA,N_nu,ParticleHole)
                            #println("ME1 = $(me1) \t ME2 = $(me2)")
                            #if C == 12
                            #    throw("time to stop ...")
                            #end

                            ppSigmaME = RPA_TBDM_2p2h_T1(p_ind,l_p,j_p,q_ind,l_q,j_q,h_ind,l_h,j_h,g_ind,l_g,j_g,J,-1,X_RPA,Y_RPA,N_nu,ParticleHole)
                            ppSigmaSum += ppSigmaME

                        end

                        # Neutron-neutron terms
                        if ((nO_a + nO_b) == 0 && (nO_c + nO_d) == 2) || ((nO_a + nO_b) == 2 && (nO_c + nO_d) == 0)

                            if (nO_a + nO_b) == 0 && (nO_c + nO_d) == 2
                                p, l_p, j_p = a, l_a, j_a
                                q, l_q, j_q = b, l_b, j_b
                                h, l_h, j_h = c, l_c, j_c
                                g, l_g, j_g = d, l_d, j_d
                            else
                                p, l_p, j_p = c, l_c, j_c
                                q, l_q, j_q = d, l_d, j_d
                                h, l_h, j_h = a, l_a, j_a
                                g, l_g, j_g = b, l_b, j_b
                            end

                            p_ind, q_ind = ph_list[p][2], ph_list[q][2]
                            h_ind, g_ind = ph_list[h][2], ph_list[g][2]

                            nnSigmaME = RPA_TBDM_2p2h_T1(p_ind,l_p,j_p,q_ind,l_q,j_q,h_ind,l_h,j_h,g_ind,l_g,j_g,J,1,X_RPA,Y_RPA,N_nu,ParticleHole)
                            nnSigmaSum += nnSigmaME

                        end
                    

                        ###=
                    # 1p-1h channel ...
                        # proton-proton terms ...
                        if (pO_a + pO_b) == 1 && (pO_c + pO_d) == 1

                            if pO_a == 0
                                p, l_p, j_p = a, l_a, j_a
                                h, l_h, j_h = b, l_b, j_b
                            else
                                p, l_p, j_p = b, l_b, j_b
                                h, l_h, j_h = a, l_a, j_a
                            end

                            if pO_c == 0
                                q, l_q, j_q = c, l_c, j_c
                                g, l_g, j_g = d, l_d, j_d
                            else
                                q, l_q, j_q = d, l_d, j_d
                                g, l_g, j_g = c, l_c, j_c
                            end

                            p_ind, q_ind = ph_list[p][1], ph_list[q][1]
                            h_ind, g_ind = ph_list[h][1], ph_list[g][1]

                            if p != q && h != g
                                ppSigmaME = RPA_TBDM_1p1h(p_ind,l_p,j_p,q_ind,l_q,j_q,h_ind,l_h,j_h,g_ind,l_g,j_g,J,-1,X_RPA,Y_RPA,N_nu,ParticleHole)
                                ppSigmaSum += ppSigmaME
                            end

                        end

                        # neutron-neutron terms ...
                        if (nO_a + nO_b) == 1 && (nO_c + nO_d) == 1

                            if nO_a == 0
                                p, l_p, j_p = a, l_a, j_a
                                h, l_h, j_h = b, l_b, j_b
                            else
                                p, l_p, j_p = b, l_b, j_b
                                h, l_h, j_h = a, l_a, j_a
                            end

                            if nO_c == 0
                                q, l_q, j_q = c, l_c, j_c
                                g, l_g, j_g = d, l_d, j_d
                            else
                                q, l_q, j_q = d, l_d, j_d
                                g, l_g, j_g = c, l_c, j_c
                            end

                            if p != q && h != g
                                p_ind, q_ind = ph_list[p][2], ph_list[q][2]
                                h_ind, g_ind = ph_list[h][2], ph_list[g][2]

                                nnSigmaME = RPA_TBDM_1p1h(p_ind,l_p,j_p,q_ind,l_q,j_q,h_ind,l_h,j_h,g_ind,l_g,j_g,J,1,X_RPA,Y_RPA,N_nu,ParticleHole)
                                nnSigmaSum += nnSigmaME
                            end

                        end
                        ##=#

                    @inbounds Sigma_NN.pp[P,J+1][Ind] = ppSigmaSum
                    @inbounds Sigma_NN.nn[P,J+1][Ind] = nnSigmaSum

                end

            end
        end

    end

    println("\nSuccesfully allocate 2-body RPA correlation matrix Sigma_NN ...")
    
    return Sigma_NN
end

function RPA_TBDM_2p2h_T1(p::Int64,l_p::Int64,j_p::Int64,q::Int64,l_q::Int64,j_q::Int64,h::Int64,l_h::Int64,j_h::Int64,g::Int64,l_g::Int64,j_g::Int64,J::Int64,T::Int64,X_RPA::Matrix{Matrix{ComplexF64}},Y_RPA::Matrix{Matrix{ComplexF64}},N_nu::Matrix{Int64},ParticleHole::pnArray)
    Sigma  = 0.0

    # Direct term ...
    Q = rem(l_p + l_h,2) + 1
    if Q == (rem(l_q + l_g,2) + 1)
        @inbounds for I in max(div(abs(j_p - j_h),2),div(abs(j_q - j_g),2)):min(div((j_p + j_h),2),div((j_q + j_g),2))
            N_ph = N_nu[I+1,Q]
            Amp = 0.25 * Float64(2*I + 1) * (-1)^(J + I + div(j_q + j_h,2)) * f6j(j_p,j_q,2*J,j_g,j_h,2*I)
            @inbounds for nu in 1:N_ph
                if T == -1
                    ph = ParticleHole.p[I+1,Q,p,h]
                    qg = ParticleHole.p[I+1,Q,q,g]
                else
                    ph = ParticleHole.n[I+1,Q,p,h]
                    qg = ParticleHole.n[I+1,Q,q,g]
                end
                ME = Amp * real(Y_RPA[I+1,Q][ph,nu] * conj(X_RPA[I+1,Q][qg,nu]))
                Sigma += ME
            end
        end
    end

    # Exchange term ...
    Q = rem(l_p + l_g,2) + 1
    if Q == (rem(l_q + l_h,2) + 1)
        @inbounds for I in max(div(abs(j_p - j_g),2),div(abs(j_q - j_h),2)):min(div((j_p + j_g),2),div((j_q + j_h),2))
            N_ph = N_nu[I+1,Q]
            Amp = 0.25 * Float64(2*I + 1) * (-1)^(I + div(j_q + j_g,2)) * f6j(j_p,j_q,2*J,j_h,j_g,2*I)
            @inbounds for nu in 1:N_ph
                if T == -1
                    pg = ParticleHole.p[I+1,Q,p,g]
                    qh = ParticleHole.p[I+1,Q,q,h]
                else
                    pg = ParticleHole.n[I+1,Q,p,g]
                    qh = ParticleHole.n[I+1,Q,q,h]
                end
                ME = Amp * real(Y_RPA[I+1,Q][pg,nu] * conj(X_RPA[I+1,Q][qh,nu]))
                Sigma += ME
            end
        end
    end

    return Sigma
end

function RPA_TBDM_2p2h_T0(p::Int64,l_p::Int64,j_p::Int64,q::Int64,l_q::Int64,j_q::Int64,h::Int64,l_h::Int64,j_h::Int64,g::Int64,l_g::Int64,j_g::Int64,J::Int64,X_RPA::Matrix{Matrix{ComplexF64}},Y_RPA::Matrix{Matrix{ComplexF64}},N_nu::Matrix{Int64},ParticleHole::pnArray)
    Sigma  = 0.0

    # Direct term only ...
    Q = rem(l_p + l_h,2) + 1
    if Q == (rem(l_q + l_g,2) + 1)
        @inbounds for I in max(div(abs(j_p - j_h),2),div(abs(j_q - j_g),2)):min(div((j_p + j_h),2),div((j_q + j_g),2))
            N_ph = N_nu[I+1,Q]
            Amp = 0.25 * Float64(2*I + 1) * (-1)^(J + I + div(j_q + j_h,2)) * f6j(j_p,j_q,2*J,j_g,j_h,2*I)
            @inbounds for nu in 1:N_ph
                ph = ParticleHole.p[I+1,Q,p,h]
                qg = ParticleHole.n[I+1,Q,q,g]
                ME = Amp * real(Y_RPA[I+1,Q][ph,nu] * conj(X_RPA[I+1,Q][qg,nu]) + real(Y_RPA[I+1,Q][qg,nu] * conj(X_RPA[I+1,Q][ph,nu])))
                Sigma += ME
            end
        end
    end

    return Sigma
end

function RPA_TBDM_1p1h(p::Int64,l_p::Int64,j_p::Int64,q::Int64,l_q::Int64,j_q::Int64,h::Int64,l_h::Int64,j_h::Int64,g::Int64,l_g::Int64,j_g::Int64,J::Int64,T::Int64,X_RPA::Matrix{Matrix{ComplexF64}},Y_RPA::Matrix{Matrix{ComplexF64}},N_nu::Matrix{Int64},ParticleHole::pnArray)
    Sigma  = 0.0

    # Direct term only ...
    Q = rem(l_p + l_g,2) + 1
    if Q == (rem(l_q + l_h,2) + 1)
        @inbounds for I in max(div(abs(j_p - j_g),2),div(abs(j_q - j_h),2)):min(div((j_p + j_g),2),div((j_q + j_h),2))
            N_ph = N_nu[I+1,Q]
            Amp = 0.5 * Float64(2*I + 1) * f6j(j_p,j_h,2*J,j_q,j_g,2*I)
            @inbounds for nu in 1:N_ph
                if T == -1
                    pg = ParticleHole.p[I+1,Q,p,g]
                    qh = ParticleHole.p[I+1,Q,q,h]
                else
                    pg = ParticleHole.n[I+1,Q,p,g]
                    qh = ParticleHole.n[I+1,Q,q,h]
                end
                ME = Amp * real(Y_RPA[I+1,Q][pg,nu] * conj(Y_RPA[I+1,Q][qh,nu]))
                Sigma += ME
            end
        end
    end

    return Sigma
end

function RPA_TBDM_1p1h_X(p::Int64,l_p::Int64,j_p::Int64,q::Int64,l_q::Int64,j_q::Int64,h::Int64,l_h::Int64,j_h::Int64,g::Int64,l_g::Int64,j_g::Int64,J::Int64,T::Int64,X_RPA::Matrix{Matrix{ComplexF64}},Y_RPA::Matrix{Matrix{ComplexF64}},N_nu::Matrix{Int64},ParticleHole::pnArray)
    Sigma  = 0.0

    # Direct term only ...
    Q = rem(l_p + l_g,2) + 1
    if Q == (rem(l_q + l_h,2) + 1)
        @inbounds for I in max(div(abs(j_p - j_g),2),div(abs(j_q - j_h),2)):min(div((j_p + j_g),2),div((j_q + j_h),2))
            N_ph = N_nu[I+1,Q]
            Amp = 0.5 * Float64(2*I + 1) * f6j(j_p,j_h,2*J,j_q,j_g,2*I)
            @inbounds for nu in 1:N_ph
                if T == -1
                    pg = ParticleHole.p[I+1,Q,p,g]
                    qh = ParticleHole.p[I+1,Q,q,h]
                else
                    pg = ParticleHole.n[I+1,Q,p,g]
                    qh = ParticleHole.n[I+1,Q,q,h]
                end
                ME = Amp * real(X_RPA[I+1,Q][pg,nu] * conj(X_RPA[I+1,Q][qh,nu]))
                Sigma += ME
            end
        end
    end

    return Sigma
end

function RPA_rho2b_allocate_test(Params::Parameters,N_nu::Matrix{Int64},Orb_Phonon::Matrix{Vector{Int64}},Phonon::Vector{PhState},N_Particle::pnInteger,Particle::pnSVector,N_Hole::pnInteger,Hole::pnSVector,Orb::Vector{Orb1B},Orb_NN::Orb2B,X_RPA::Matrix{Matrix{ComplexF64}},Y_RPA::Matrix{Matrix{ComplexF64}})
    # Read parameters ...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1
    a_max = div((N_max + 1)*(N_max + 2),2)

    # Preallocate arrays of allowed values of J & P ...
    JP_list = JP_initialize(Params.Calc.N2max + 1)

    # Make Particle-Hole orbitals list ...
    ParticleHole = orbitals_ph_list(J_max,N_Particle,Particle,N_Hole,Hole)
    
    # Initialite the 2-body NN correlation matrix Sigma_NN ...
    Rho_NN = O2b_initialize(Params,Orb)

    function orbitals_ph_mappping(Params::Parameters,N_Particle::pnInteger,Particle::pnSVector,N_Hole::pnInteger,Hole::pnSVector)
        # Read parameters ...
        N_max = Params.Calc.Nmax
        a_max = div((N_max + 1)*(N_max + 2),2)

        orbitals_ph_list = Vector{Vector{Int64}}(undef,a_max)

        @inbounds for a in 1:a_max
            orbitals_ph_list[a] = Vector{Int64}(undef,2)
        end

        @inbounds for p in 1:N_Particle.p
            a = Particle.p[p].a
            orbitals_ph_list[a][1] = p
        end

        @inbounds for h in 1:N_Hole.p
            a = Hole.p[h].a
            orbitals_ph_list[a][1] = h
        end

        @inbounds for p in 1:N_Particle.n
            a = Particle.n[p].a
            orbitals_ph_list[a][2] = p
        end

        @inbounds for h in 1:N_Hole.n
            a = Hole.n[h].a
            orbitals_ph_list[a][2] = h
        end

        return orbitals_ph_list
    end

    ph_list = orbitals_ph_mappping(Params,N_Particle,Particle,N_Hole,Hole)

    println("\nAllocating 2-body RPA correlation matrix Sigma_NN ...")

    @inbounds Threads.@threads for JP in JP_list;
        J, P = JP[1], JP[2]
        if P == 1
            println("\tCalculating   ...   J = " * string(J) * "/" * string(N_2max+1) * "\tP = +")
        else
            println("\tCalculating   ...   J = " * string(J) * "/" * string(N_2max+1) * "\tP = -")
        end
        N_T0, N_T1 = Orb_NN.N[1,P,J+1], Orb_NN.N[2,P,J+1]
        @inbounds for Bra in 1:max(N_T0, N_T1)
            @inbounds for Ket in 1:Bra

                # Case of pp & nn interaction ... T = 1
                if Bra <= N_T1
                    Ind = Bra + (Ket - 1) * N_T1 - div(Ket * (Ket - 1),2)
                    a, b = Orb_NN.Ind[2,P,J+1][Bra][1], Orb_NN.Ind[2,P,J+1][Bra][2]
                    c, d = Orb_NN.Ind[2,P,J+1][Ket][1], Orb_NN.Ind[2,P,J+1][Ket][2]
                    l_a, j_a = Orb[a].l, Orb[a].j
                    l_b, j_b = Orb[b].l, Orb[b].j
                    l_c, j_c = Orb[c].l, Orb[c].j
                    l_d, j_d = Orb[d].l, Orb[d].j

                    pO_a, pO_b = Orb[a].pO, Orb[b].pO
                    pO_c, pO_d = Orb[c].pO, Orb[d].pO

                    nO_a, nO_b = Orb[a].nO, Orb[b].nO
                    nO_c, nO_d = Orb[c].nO, Orb[d].nO

                    ppRhoSum, nnRhoSum = 0.0, 0.0

                    
                    # 1p-1h channel ...
                        # proton-proton terms ...
                        if (pO_a + pO_b) == 1 && (pO_c + pO_d) == 1

                            if pO_a == 0
                                p, l_p, j_p = a, l_a, j_a
                                h, l_h, j_h = b, l_b, j_b
                            else
                                continue
                                p, l_p, j_p = b, l_b, j_b
                                h, l_h, j_h = a, l_a, j_a
                            end

                            if pO_c == 0
                                q, l_q, j_q = c, l_c, j_c
                                g, l_g, j_g = d, l_d, j_d
                            else
                                continue
                                q, l_q, j_q = d, l_d, j_d
                                g, l_g, j_g = c, l_c, j_c
                            end

                            p_ind, q_ind = ph_list[p][1], ph_list[q][1]
                            h_ind, g_ind = ph_list[h][1], ph_list[g][1]

                            ppRhoME = RPA_TBDM_1p1h(p_ind,l_p,j_p,q_ind,l_q,j_q,h_ind,l_h,j_h,g_ind,l_g,j_g,J,-1,X_RPA,Y_RPA,N_nu,ParticleHole)
                            ppRhoSum += ppRhoME

                        end

                        # neutron-neutron terms ...
                        if (nO_a + nO_b) == 1 && (nO_c + nO_d) == 1

                            if nO_a == 0
                                p, l_p, j_p = a, l_a, j_a
                                h, l_h, j_h = b, l_b, j_b
                            else
                                continue
                                p, l_p, j_p = b, l_b, j_b
                                h, l_h, j_h = a, l_a, j_a
                            end

                            if nO_c == 0
                                q, l_q, j_q = c, l_c, j_c
                                g, l_g, j_g = d, l_d, j_d
                            else
                                continue
                                q, l_q, j_q = d, l_d, j_d
                                g, l_g, j_g = c, l_c, j_c
                            end

                            p_ind, q_ind = ph_list[p][2], ph_list[q][2]
                            h_ind, g_ind = ph_list[h][2], ph_list[g][2]

                            nnRhoME = RPA_TBDM_1p1h(p_ind,l_p,j_p,q_ind,l_q,j_q,h_ind,l_h,j_h,g_ind,l_g,j_g,J,1,X_RPA,Y_RPA,N_nu,ParticleHole)
                            nnRhoSum += nnRhoME

                        end


                    # 1h-1p channel ...
                        # proton-proton terms ...
                        if (pO_a + pO_b) == 1 && (pO_c + pO_d) == 1

                            if pO_a == 0
                                p, l_p, j_p = a, l_a, j_a
                                h, l_h, j_h = b, l_b, j_b
                            else
                                continue
                                p, l_p, j_p = b, l_b, j_b
                                h, l_h, j_h = a, l_a, j_a
                            end

                            if pO_c == 0
                                q, l_q, j_q = c, l_c, j_c
                                g, l_g, j_g = d, l_d, j_d
                            else
                                continue
                                q, l_q, j_q = d, l_d, j_d
                                g, l_g, j_g = c, l_c, j_c
                            end

                            p_ind, q_ind = ph_list[p][1], ph_list[q][1]
                            h_ind, g_ind = ph_list[h][1], ph_list[g][1]

                            ppRhoME = RPA_TBDM_1p1h_X(p_ind,l_p,j_p,q_ind,l_q,j_q,h_ind,l_h,j_h,g_ind,l_g,j_g,J,-1,X_RPA,Y_RPA,N_nu,ParticleHole)
                            ppRhoSum += ppRhoME

                        end

                        # neutron-neutron terms ...
                        if (nO_a + nO_b) == 1 && (nO_c + nO_d) == 1

                            if nO_a == 0
                                p, l_p, j_p = a, l_a, j_a
                                h, l_h, j_h = b, l_b, j_b
                            else
                                p, l_p, j_p = b, l_b, j_b
                                h, l_h, j_h = a, l_a, j_a
                            end

                            if nO_c == 0
                                continue
                                q, l_q, j_q = c, l_c, j_c
                                g, l_g, j_g = d, l_d, j_d
                            else
                                continue
                                q, l_q, j_q = d, l_d, j_d
                                g, l_g, j_g = c, l_c, j_c
                            end

                            p_ind, q_ind = ph_list[p][2], ph_list[q][2]
                            h_ind, g_ind = ph_list[h][2], ph_list[g][2]

                            nnRhoME = RPA_TBDM_1p1h_X(p_ind,l_p,j_p,q_ind,l_q,j_q,h_ind,l_h,j_h,g_ind,l_g,j_g,J,1,X_RPA,Y_RPA,N_nu,ParticleHole)
                            nnRhoSum += nnRhoME

                        end

                    @inbounds Rho_NN.pp[P,J+1][Ind] = ppRhoSum
                    @inbounds Rho_NN.nn[P,J+1][Ind] = nnRhoSum

                end

            end
        end

    end

    pSum, nSum = 0.0, 0.0

    @inbounds for a in 1:a_max
        l_a, j_a = Orb[a].l, Orb[a].j
        @inbounds for b in 1:a_max
            l_b, j_b = Orb[b].l, Orb[b].j

            if l_a == l_b && j_a == j_b && a == b
                @inbounds for c in 1:a_max
                    l_c, j_c = Orb[c].l, Orb[c].j

                    P = rem(l_a + l_c,2) + 1

                    if P != (rem(l_b + l_c,2) + 1)
                        continue
                    end

                    @inbounds for J in div(max(abs(j_a - j_c), abs(j_b - j_c)),2):div(min(j_a + j_c, j_b + j_c),2)
                        J_hat = Float64(2*J + 1)
                        pME = J_hat * O2b_pp(a,c,b,c,J,P,Rho_NN,Orb,Orb_NN)
                        pSum += pME
                        nME = J_hat * O2b_nn(a,c,b,c,J,P,Rho_NN,Orb,Orb_NN)
                        nSum += nME
                    end

                end
            end

        end
    end

    println("pSum = $(pSum) \t nSum = $(nSum)")


    #println("\nSuccesfully allocate 2-body RPA correlation matrix Rho_NN ...")
    
    return
end