function HF_ScRPA_allocate(Params::Parameters,N_nu::Matrix{Int64},Orb_Phonon::Matrix{Vector{Int64}},Phonon::Vector{PhState},Particle::pnSVector,Hole::pnSVector,h_N::O1B,V_NN::O2B,Rho_NN::O2B,Orb::Vector{Orb1B},Orb_NN::Orb2B)
    # Read calculation parameters ...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1

    JP_List = JP_initialize(J_max)
    
    println("\nAllocating TDA & RPA matrices A & B ...")

    #Allocate A & B matrices ...
    A = Matrix{Matrix{Float64}}(undef,J_max+1,2)
    B = Matrix{Matrix{Float64}}(undef,J_max+1,2)
    N = Matrix{Matrix{Float64}}(undef,J_max+1,2)
    M = Matrix{Matrix{Float64}}(undef,J_max+1,2)

    # Fill A & B matrices ...
    @inbounds Threads.@threads for JP in JP_List
        J = JP[1]
        P = JP[2]
        N_ph = N_nu[J+1,P]
        A_JP = zeros(Float64,N_ph,N_ph)
        B_JP = zeros(Float64,N_ph,N_ph)
        N_JP = zeros(Float64,N_ph,N_ph)
        M_JP = zeros(Float64,N_ph,N_ph)

        @inbounds for ind_ph in 1:N_ph
            ph = Orb_Phonon[J+1,P][ind_ph]
            p, h = Phonon[ph].p, Phonon[ph].h
            t_ph, J, P = Phonon[ph].tz, Phonon[ph].J, Phonon[ph].P
            if t_ph == -1
                a_p, l_p, j_p, E_p = Particle.p[p].a, Particle.p[p].l, Particle.p[p].j, Particle.p[p].E
                a_h, l_h, j_h, E_h = Hole.p[h].a, Hole.p[h].l, Hole.p[h].j, Hole.p[h].E

                a_p, l_p, j_p = Particle.p[p].a, Particle.p[p].l, Particle.p[p].j
                a_h, l_h, j_h = Hole.p[h].a, Hole.p[h].l, Hole.p[h].j
            elseif t_ph == 1
                a_p, l_p, j_p, E_p = Particle.n[p].a, Particle.n[p].l, Particle.n[p].j, Particle.n[p].E
                a_h, l_h, j_h, E_h = Hole.n[h].a, Hole.n[h].l, Hole.n[h].j, Hole.n[h].E

                a_p, l_p, j_p = Particle.n[p].a, Particle.n[p].l, Particle.n[p].j
                a_h, l_h, j_h = Hole.n[h].a, Hole.n[h].l, Hole.n[h].j
            end

            @inbounds for ind_qg in 1:N_ph
                qg = Orb_Phonon[J+1,P][ind_qg]
                q, g = Phonon[qg].p, Phonon[qg].h
                t_qg = Phonon[qg].tz
                if t_qg == -1
                    a_q, l_q, j_q, E_q = Particle.p[q].a, Particle.p[q].l, Particle.p[q].j, Particle.p[q].E
                    a_g, l_g, j_g, E_g = Hole.p[g].a, Hole.p[g].l, Hole.p[g].j, Hole.p[g].E

                    a_q, l_q, j_q = Particle.p[q].a, Particle.p[q].l, Particle.p[q].j
                    a_g, l_g, j_g = Hole.p[g].a, Hole.p[g].l, Hole.p[g].j
                elseif t_qg == 1
                    a_q, l_q, j_q, E_q = Particle.n[q].a, Particle.n[q].l, Particle.n[q].j, Particle.n[q].E
                    a_g, l_g, j_g, E_g = Hole.n[g].a, Hole.n[g].l, Hole.n[g].j, Hole.n[g].E

                    a_q, l_q, j_q = Particle.n[q].a, Particle.n[q].l, Particle.n[q].j
                    a_g, l_g, j_g = Hole.n[g].a, Hole.n[g].l, Hole.n[g].j
                end
    
                ASum = 0.0
                BSum = 0.0
                NSum = 0.0
                MSum = 0.0

                P_pg = rem(l_p + l_g,2) + 1
                P_pq = rem(l_p + l_q,2) + 1

                #=
                    if t_ph == t_qg && ph == qg
                        ASum += (E_p - E_h)
                        NSum += 1.0
                    end
                =#

                ###=
                if t_ph == -1 && t_qg == -1# && ph == qg
                    ASum += h_N.p[a_p,a_q] * kronecker_delta(h,g) - h_N.p[a_h,a_g] * kronecker_delta(p,q)
                elseif t_ph == 1 && t_qg == 1# && ph == qg
                    ASum += h_N.n[a_p,a_q] * kronecker_delta(h,g) - h_N.n[a_h,a_g] * kronecker_delta(p,q)
                end

                if ph == qg && t_ph == t_qg
                    NSum += 1.0
                end
                ##=#

                if rem(l_p + l_g, 2) == rem(l_q + l_h, 2)
                    @inbounds for J_r in div(abs(j_p - j_g),2):div((j_p + j_g),2)
                        if (abs(j_q - j_h) <= 2*J_r) && (2*J_r <= (j_q + j_h))
                            if t_ph == -1 && t_qg == -1
                                Phase = (-1)^(div(j_h + j_q,2) + J_r)
                                ME = Phase * (2*J_r + 1) * f6j(j_p, j_h, 2*J, j_q, j_g, 2*J_r) * O2b_pp(a_p,a_g,a_h,a_q,J_r,P_pg,V_NN,Orb,Orb_NN)
                                ASum +=  ME
                            elseif t_ph == -1 && t_qg == 1
                                Phase = (-1)^(div((j_h + j_q),2) + J_r)
                                ME = Phase * (2*J_r + 1) * f6j(j_p, j_h, 2*J, j_q, j_g, 2*J_r) * O2b_pn(a_p,a_g,a_h,a_q,J_r,P_pg,V_NN,Orb_NN)
                                ASum += ME
                            elseif t_ph == 1 && t_qg == -1
                                Phase = (-1)^(div((j_g + j_p),2) + J_r)
                                ME = Phase * (2*J_r + 1) * f6j(j_q, j_g, 2*J, j_p, j_h, 2*J_r) * O2b_pn(a_q,a_h,a_g,a_p,J_r,P_pg,V_NN,Orb_NN)
                                ASum += ME
                            elseif t_ph == 1 && t_qg == 1
                                Phase = (-1)^(div((j_h + j_q),2) + J_r)
                                ME = Phase * (2*J_r + 1) * f6j(j_p, j_h, 2*J, j_q, j_g, 2*J_r) * O2b_nn(a_p,a_g,a_h,a_q,J_r,P_pg,V_NN,Orb,Orb_NN)
                                ASum +=  ME
                            end
                        end
                    end
                end

                if rem(l_p + l_q, 2) == rem(l_h + l_g, 2)
                    @inbounds for J_r in div(abs(j_p - j_q),2):div((j_p + j_q),2)
                        if (abs(j_h - j_g) <= 2*J_r) && (2*J_r <= (j_h + j_g))

                            if t_ph == -1 && t_qg == -1
                                Phase = Float64((-1)^(div(j_h + j_q,2) + J + J_r))
                                ME = Phase * Float64(2*J_r + 1) * f6j(j_p,j_h,2*J,j_g,j_q,2*J_r) * O2b_pp(a_p,a_q,a_h,a_g,J_r,P_pq,V_NN,Orb,Orb_NN)
                                BSum += ME
                            elseif t_ph == -1 && t_qg == 1
                                Phase = Float64((-1)^(div(j_h + j_q,2) + J + J_r))
                                ME = Phase * (2*J_r + 1) * f6j(j_p,j_h,2*J,j_g,j_q,2*J_r) * O2b_pn(a_p,a_q,a_h,a_g,J_r,P_pq,V_NN,Orb_NN)
                                BSum += ME
                            elseif t_ph == 1 && t_qg == -1
                                Phase = Float64((-1)^(div(j_g + j_p,2) + J + J_r))
                                ME = Phase * (2*J_r + 1) * f6j(j_q,j_g,2*J,j_h,j_p,2*J_r) * O2b_pn(a_q,a_p,a_g,a_h,J_r,P_pq,V_NN,Orb_NN)
                                BSum += ME
                            elseif t_ph == 1 && t_qg == 1
                                Phase = Float64((-1)^(div(j_h + j_q,2) + J + J_r))
                                ME = Phase * (2*J_r + 1) * f6j(j_p,j_h,2*J,j_g,j_q,2*J_r) * O2b_nn(a_p,a_q,a_h,a_g,J_r,P_pq,V_NN,Orb,Orb_NN)
                                BSum += ME
                            end
                            
                            # Allocation of overlap matrix M ...
                            #=
                                if t_ph == -1 && t_qg == -1
                                    Phase = Float64((-1)^(div(j_h + j_q,2) + J + J_r))
                                    ME = Phase * Float64(2*J_r + 1) * f6j(j_p,j_h,2*J,j_g,j_q,2*J_r) * V2B(a_p,a_q,a_h,a_g,J_r,1,V_NN.pp,Orb,Orb_NN) / (E_h + E_g - E_p - E_q)
                                    MSum += ME
                                elseif t_ph == -1 && t_qg == 1
                                    Phase = Float64((-1)^(div(j_h + j_q,2) + J + J_r))
                                    ME = Phase * (2*J_r + 1) * f6j(j_p,j_h,2*J,j_g,j_q,2*J_r) * V2B(a_p,a_q,a_h,a_g,J_r,0,V_NN.pn,Orb,Orb_NN) / (E_h + E_g - E_p - E_q)
                                    MSum += ME
                                elseif t_ph == 1 && t_qg == -1
                                    Phase = Float64((-1)^(div(j_g + j_p,2) + J + J_r))
                                    ME = Phase * (2*J_r + 1) * f6j(j_q,j_g,2*J,j_h,j_p,2*J_r) * V2B(a_q,a_p,a_g,a_h,J_r,0,V_NN.pn,Orb,Orb_NN) / (E_h + E_g - E_p - E_q)
                                    MSum += ME
                                elseif t_ph == 1 && t_qg == 1
                                    Phase = Float64((-1)^(div(j_h + j_q,2) + J + J_r))
                                    ME = Phase * (2*J_r + 1) * f6j(j_p,j_h,2*J,j_g,j_q,2*J_r) * V2B(a_p,a_q,a_h,a_g,J_r,1,V_NN.nn,Orb,Orb_NN) / (E_h + E_g - E_p - E_q)
                                    MSum += ME
                                end
                            =#

                            # Works great!!!

                            #=
                            if t_ph == -1 && t_qg == -1
                                Phase = Float64((-1)^(div(j_h + j_q,2) + J + J_r))
                                ME = Phase * Float64(2*J_r + 1) * f6j(j_p,j_h,2*J,j_g,j_q,2*J_r) * O2b_pp(a_p,a_q,a_h,a_g,J_r,P_pq,Rho_NN,Orb,Orb_NN)
                                MSum += ME
                            elseif t_ph == -1 && t_qg == 1
                                Phase = Float64((-1)^(div(j_h + j_q,2) + J + J_r))
                                ME = Phase * (2*J_r + 1) * f6j(j_p,j_h,2*J,j_g,j_q,2*J_r) * O2b_pn(a_p,a_q,a_h,a_g,J_r,P_pq,Rho_NN,Orb_NN)
                                MSum += ME
                            elseif t_ph == 1 && t_qg == -1
                                Phase = Float64((-1)^(div(j_g + j_p,2) + J + J_r))
                                ME = Phase * (2*J_r + 1) * f6j(j_q,j_g,2*J,j_h,j_p,2*J_r) * O2b_pn(a_q,a_p,a_g,a_h,J_r,P_pq,Rho_NN,Orb_NN)
                                MSum += ME
                            elseif t_ph == 1 && t_qg == 1
                                Phase = Float64((-1)^(div(j_h + j_q,2) + J + J_r))
                                ME = Phase * (2*J_r + 1) * f6j(j_p,j_h,2*J,j_g,j_q,2*J_r) * O2b_nn(a_p,a_q,a_h,a_g,J_r,P_pq,Rho_NN,Orb,Orb_NN)
                                MSum += ME
                            end
                            =#

                        end
                    end
                end
                A_JP[ind_ph,ind_qg] = ASum
                B_JP[ind_ph,ind_qg] = BSum
                N_JP[ind_ph,ind_qg] = NSum
                M_JP[ind_ph,ind_qg] = MSum
            end
        end

        # For given J & P allocate A, B, N & M ...
        A[J+1,P] = A_JP
        B[J+1,P] = B_JP
        N[J+1,P] = N_JP
        M[J+1,P] = M_JP
    end

    println("\nMatrices A, B, N & M allocated ...")

    return A, B, N, M
end