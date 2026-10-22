function HF_MBPT(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,h_N::O1B,V_NN::O2B)
    # Make Particle-Hole orbitals ...
    Orb_HF_MBPT = HF_MBPT_orbitals_make(Params,Orb)

    # Evaluate the HF-MBPT(2) ground-state energy correction ...
    @time HF_MBPT_energy(Params,Orb,Orb_HF_MBPT,Orb_NN,h_N,V_NN)

    # Evaluate the HF-MBPT(3) ground-state OBDM correction & radial densities ...
    @time HF_MBPT_OBDM(Params,Orb,Orb_HF_MBPT,Orb_NN,h_N,V_NN)

    return
end

function HF_MBPT_orbitals_make(Params::Parameters,Orb::Vector{Orb1B})
    # Read parameters ...
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1)*(N_max + 2),2)

    # Initialize particle number counters ...
    pN_particle, pN_hole = 0, 0
    nN_particle, nN_hole = 0, 0

    # Determine the number of particle & hole orbitals ...
    @inbounds for a in 1:a_max
        if abs(Orb[a].pO) > 1e-8
            pN_hole += 1
        end
        if abs(1.0 - Orb[a].pO) > 1e-8
            pN_particle += 1
        end
        if abs(Orb[a].nO) > 1e-8
            nN_hole += 1
        end
        if abs(1.0 - Orb[a].nO) > 1e-8
            nN_particle += 1
        end
    end

    # Initialize the particle & hole orbitals ...
    pOrb_particle, pOrb_hole = Vector{Orb1B}(undef,pN_particle), Vector{Orb1B}(undef,pN_hole)
    nOrb_particle, nOrb_hole = Vector{Orb1B}(undef,nN_particle), Vector{Orb1B}(undef,nN_hole)

    # Reset particle number counters ...
    pN_particle, pN_hole = 0, 0
    nN_particle, nN_hole = 0, 0

    # Allocate the particle & hole orbitals ...
    @inbounds for a in 1:a_max
        if abs(1.0 - Orb[a].pO) > 1e-8
            pN_particle += 1
            pOrb_particle[pN_particle] = Orb[a]
        end
        if abs(Orb[a].pO) > 1e-8
            pN_hole += 1
            pOrb_hole[pN_hole] = Orb[a]
        end
        if abs(1.0 - Orb[a].nO) > 1e-8
            nN_particle += 1
            nOrb_particle[nN_particle] = Orb[a]
        end
        if abs(Orb[a].nO) > 1e-8
            nN_hole += 1
            nOrb_hole[nN_hole] = Orb[a]
        end
    end

    # Allocate the 1-body partice & hole orbitals structure ...
    Orb_HF_MBPT = Orb1B_HF_MBPT(pnOrbVector(pOrb_particle,nOrb_particle),
                                pnOrbVector(pOrb_hole,nOrb_hole),
                                pnInteger(pN_particle,nN_particle),
                                pnInteger(pN_hole,nN_hole)
                                )

    return Orb_HF_MBPT

end

@inline function HF_MBPT_h1b(a::Int64,t_a::Int64,h_N::O1B)
    if t_a == -1
        return h_N.p[a,a]
    else
        return h_N.n[a,a]
    end
    return 0.0
end

@inline function HF_MBPT3_V2b_Pandya(a_a::Int64,l_a::Int64,j_a::Int64,t_a::Int64,a_b::Int64,l_b::Int64,j_b::Int64,t_b::Int64,
                                     a_c::Int64,l_c::Int64,j_c::Int64,t_c::Int64,a_d::Int64,l_d::Int64,j_d::Int64,t_d::Int64,
                                     J::Int64,Orb::Vector{Orb1B},Orb_NN::Orb2B,V_NN::O2B)

    P_ab = rem(l_a + l_b,2) + 1

    if (rem(l_c + l_d,2) + 1) != P_ab
        return 0.0
    end

    I_min, I_max = div(max(abs(j_a - j_b), abs(j_c - j_d)),2), div(min(j_a + j_b, j_c + j_d),2)

    V = 0.0

    if (t_a + t_b) == -2
        @inbounds for I in I_min:I_max
            ME = Float64(2*I + 1) * w6j(j_a,j_b,2*I,j_c,j_d,2*J) * O2b_pp(a_a,a_b,a_c,a_d,I,P_ab,V_NN,Orb,Orb_NN)
            V -= ME
        end
        return V
    end

    if (t_a + t_b) == 2
        @inbounds for I in I_min:I_max
            ME = Float64(2*I + 1) * w6j(j_a,j_b,2*I,j_c,j_d,2*J) * O2b_nn(a_a,a_b,a_c,a_d,I,P_ab,V_NN,Orb,Orb_NN)
            V -= ME
        end
        return V
    end

    if (t_a + t_b) == 0

        if (t_a == -1) && (t_b == 1) && (t_c == -1) && (t_d == 1)
            @inbounds for I in I_min:I_max
                ME = Float64(2*I + 1) * w6j(j_a,j_b,2*I,j_c,j_d,2*J) * O2b_pn(a_a,a_b,a_c,a_d,I,P_ab,V_NN,Orb_NN)
                V -= ME
            end
            return V
        end

        if (t_a == -1) && (t_b == 1) && (t_c == 1) && (t_d == -1)
            @inbounds for I in I_min:I_max
                ME = Float64(2*I + 1) * w6j(j_a,j_b,2*I,j_c,j_d,2*J) * O2b_pn(a_a,a_b,a_d,a_c,I,P_ab,V_NN,Orb_NN) * phase(div(j_c + j_d,2) - I)
                V -= ME
            end
            return V
        end

        if (t_a == 1) && (t_b == -1) && (t_c == -1) && (t_d == 1)
            @inbounds for I in I_min:I_max
                ME = Float64(2*I + 1) * w6j(j_a,j_b,2*I,j_c,j_d,2*J) * O2b_pn(a_b,a_a,a_c,a_d,I,P_ab,V_NN,Orb_NN) * phase(div(j_a + j_b,2) - I)
                V -= ME
            end
            return V
        end

        if (t_a == 1) && (t_b == -1) && (t_c == 1) && (t_d ==-1)
            @inbounds for I in I_min:I_max
                ME = Float64(2*I + 1) * w6j(j_a,j_b,2*I,j_c,j_d,2*J) * O2b_pn(a_b,a_a,a_d,a_c,I,P_ab,V_NN,Orb_NN) * phase(div(j_a + j_b + j_c + j_d,2))
                V -= ME
            end
            return V
        end
    end

    return 0.0
end

function HF_MBPT_energy(Params::Parameters,Orb::Vector{Orb1B},Orb_HF_MBPT::Orb1B_HF_MBPT,Orb_NN::Orb2B,h_N::O1B,V_NN::O2B)
    # Read parameters ...
    IO = Params.Calc.Path
    N_max = Params.Calc.Nmax
    J_max = 2*N_max + 1

    # Unpack particle & hole orbitals ...
    pOrb_particle, pOrb_hole = Orb_HF_MBPT.p.p, Orb_HF_MBPT.h.p
    nOrb_particle, nOrb_hole = Orb_HF_MBPT.p.n, Orb_HF_MBPT.h.n

    # Initialize the thread accumulators ...
    dE, dE_threads = 0.0, zeros(Float64,Threads.maxthreadid())
    dE2, dE3_pp, dE3_ph, dE3_hh = 0.0, 0.0, 0.0, 0.0

    # Calculate the HF-MBPT(2) energy corrections ...
    println("\nCalculating HF-MBPT(2) ground-state energy correction ... ")
    
        # proton-proton contribution
        @inbounds Threads.@threads :static for Orb_p in pOrb_particle
            Sum, Tid = 0.0, Threads.threadid()
            a_p, l_p, j_p = Orb_p.a, Orb_p.l, Orb_p.j
            E_p = h_N.p[a_p,a_p]
            @inbounds for Orb_q in pOrb_particle
                a_q, l_q, j_q = Orb_q.a, Orb_q.l, Orb_q.j
                E_q = h_N.p[a_q,a_q]
                P = rem(l_p + l_q, 2) + 1
                @inbounds for J in div(abs(j_p - j_q),2):div((j_p + j_q),2)
                    @inbounds for Orb_h in pOrb_hole
                        a_h, l_h, j_h = Orb_h.a, Orb_h.l, Orb_h.j
                        E_h = h_N.p[a_h,a_h]
                        @inbounds for Orb_g in pOrb_hole
                            a_g, l_g, j_g = Orb_g.a, Orb_g.l, Orb_g.j
                            E_g = h_N.p[a_g,a_g]

                            if (P != (rem(l_h + l_g, 2) + 1)) || (abs(j_h - j_g) > 2*J) || (2*J > (j_h + j_g))
                                continue
                            end

                            E_pqhg = E_h + E_g - E_p - E_q

                            Sum += 0.25 * Float64(2*J + 1) * abs(O2b_pp(a_p,a_q,a_h,a_g,J,P,V_NN,Orb,Orb_NN))^2 / E_pqhg
                        end
                    end
                end
            end
            dE_threads[Tid] += Sum 
        end
        dE2 += sum(dE_threads)

        # neutron-neutron contribution
        dE_threads .= 0.0
        @inbounds Threads.@threads :static for Orb_p in nOrb_particle
            Sum, Tid = 0.0, Threads.threadid()
            a_p, l_p, j_p = Orb_p.a, Orb_p.l, Orb_p.j
            E_p = h_N.n[a_p,a_p]
            @inbounds for Orb_q in nOrb_particle
                a_q, l_q, j_q = Orb_q.a, Orb_q.l, Orb_q.j
                E_q = h_N.n[a_q,a_q]
                P = rem(l_p + l_q, 2) + 1
                @inbounds for J in div(abs(j_p - j_q),2):div((j_p + j_q),2)
                    @inbounds for Orb_h in nOrb_hole
                        a_h, l_h, j_h = Orb_h.a, Orb_h.l, Orb_h.j
                        E_h = h_N.n[a_h,a_h]
                        @inbounds for Orb_g in nOrb_hole
                            a_g, l_g, j_g = Orb_g.a, Orb_g.l, Orb_g.j
                            E_g = h_N.n[a_g,a_g]

                            if (P != (rem(l_h + l_g, 2) + 1)) || (abs(j_h - j_g) > 2*J) || (2*J > (j_h + j_g))
                                continue
                            end

                            E_pqhg = E_h + E_g - E_p - E_q

                            Sum += 0.25 * Float64(2*J + 1) * abs(O2b_nn(a_p,a_q,a_h,a_g,J,P,V_NN,Orb,Orb_NN))^2 / E_pqhg
                        end
                    end
                end
            end
            dE_threads[Tid] += Sum
        end
        dE2 += sum(dE_threads)

        # proton-neutron contribution
        dE_threads .= 0.0
        @inbounds Threads.@threads :static for Orb_p in pOrb_particle
            Sum, Tid = 0.0, Threads.threadid()
            a_p, l_p, j_p = Orb_p.a, Orb_p.l, Orb_p.j
            E_p = h_N.p[a_p,a_p]
            @inbounds for Orb_q in nOrb_particle
                a_q, l_q, j_q = Orb_q.a, Orb_q.l, Orb_q.j
                E_q = h_N.n[a_q,a_q]
                P = rem(l_p + l_q, 2) + 1
                @inbounds for J in div(abs(j_p - j_q),2):div((j_p + j_q),2)
                    @inbounds for Orb_h in pOrb_hole
                        a_h, l_h, j_h = Orb_h.a, Orb_h.l, Orb_h.j
                        E_h = h_N.p[a_h,a_h]
                        @inbounds for Orb_g in nOrb_hole
                            a_g, l_g, j_g = Orb_g.a, Orb_g.l, Orb_g.j
                            E_g = h_N.n[a_g,a_g]

                            if (P != (rem(l_h + l_g, 2) + 1)) || (abs(j_h - j_g) > 2*J) || (2*J > (j_h + j_g))
                                continue
                            end

                            E_pqhg = E_h + E_g - E_p - E_q

                            Sum += Float64(2*J + 1) * abs(O2b_pn(a_p,a_q,a_h,a_g,J,P,V_NN,Orb_NN))^2 / E_pqhg
                        end
                    end
                end
            end
            dE_threads[Tid] += Sum
        end
        dE2 += sum(dE_threads)

    # Calculate the HF-MBPT(3) energy corrections ...
    println("\nCalculating HF-MBPT(3) ground-state energy correction ... ")

        # particle-particle sector
            # proton-proton contribution
            ppOrb_particle_list = Tuple{Orb1B,Orb1B}[(x,y) for x in pOrb_particle, y in pOrb_particle]

            dE_threads .= 0.0
            @time @inbounds Threads.@threads :static for (Orb_p,Orb_q) in ppOrb_particle_list
                Sum, Tid = 0.0, Threads.threadid()

                a_p, l_p, j_p = Orb_p.a, Orb_p.l, Orb_p.j
                E_p = HF_MBPT_h1b(a_p,-1,h_N)

                a_q, l_q, j_q = Orb_q.a, Orb_q.l, Orb_q.j
                E_q = HF_MBPT_h1b(a_q,-1,h_N)

                P_pq = rem(l_p + l_q, 2) + 1

                @inbounds for Orb_h in pOrb_hole
                    a_h, l_h, j_h = Orb_h.a, Orb_h.l, Orb_h.j
                    E_h = HF_MBPT_h1b(a_h,-1,h_N)

                    @inbounds for Orb_g in pOrb_hole
                        a_g, l_g, j_g = Orb_g.a, Orb_g.l, Orb_g.j
                        E_g = HF_MBPT_h1b(a_g,-1,h_N)

                        if (rem(l_h + l_g,2) + 1) != P_pq
                            continue
                        end

                        E_pqhg = E_h + E_g - E_p - E_q

                        @inbounds for Orb_r in pOrb_particle
                            a_r, l_r, j_r = Orb_r.a, Orb_r.l, Orb_r.j
                            E_r = HF_MBPT_h1b(a_r,-1,h_N)

                            @inbounds for Orb_s in pOrb_particle
                                a_s, l_s, j_s = Orb_s.a, Orb_s.l, Orb_s.j
                                E_s = HF_MBPT_h1b(a_s,-1,h_N)

                                if (rem(l_r + l_s,2) + 1) != P_pq
                                    continue
                                end

                                E_rshg = E_h + E_g - E_r - E_s

                                @inbounds for J in div(max(abs(j_h - j_g),abs(j_p - j_q),abs(j_r - j_s)),2):div(min(j_h + j_g, j_p + j_q, j_r + j_s),2)
                                    J_hat = Float64(2*J + 1)

                                    ME = 0.125 * J_hat * O2b_pp(a_h,a_g,a_p,a_q,J,P_pq,V_NN,Orb,Orb_NN) *
                                                         O2b_pp(a_p,a_q,a_r,a_s,J,P_pq,V_NN,Orb,Orb_NN) *
                                                         O2b_pp(a_r,a_s,a_h,a_g,J,P_pq,V_NN,Orb,Orb_NN) / E_pqhg / E_rshg

                                    Sum += ME
                                end
                            end
                        end
                    end
                end

                dE_threads[Tid] += Sum
            end
            dE3_pp += sum(dE_threads)

            # proton-neutron contribution
            pnOrb_particle_list = Tuple{Orb1B,Orb1B}[(x,y) for x in pOrb_particle, y in nOrb_particle]

            dE_threads .= 0.0
            @time @inbounds Threads.@threads :static for (Orb_p,Orb_q) in pnOrb_particle_list
                Sum, Tid = 0.0, Threads.threadid()

                a_p, l_p, j_p = Orb_p.a, Orb_p.l, Orb_p.j
                E_p = HF_MBPT_h1b(a_p,-1,h_N)

                a_q, l_q, j_q = Orb_q.a, Orb_q.l, Orb_q.j
                E_q = HF_MBPT_h1b(a_q,-1,h_N)

                P_pq = rem(l_p + l_q, 2) + 1

                @inbounds for Orb_h in pOrb_hole
                    a_h, l_h, j_h = Orb_h.a, Orb_h.l, Orb_h.j
                    E_h = HF_MBPT_h1b(a_h,-1,h_N)

                    @inbounds for Orb_g in nOrb_hole
                        a_g, l_g, j_g = Orb_g.a, Orb_g.l, Orb_g.j
                        E_g = HF_MBPT_h1b(a_g,-1,h_N)

                        if (rem(l_h + l_g,2) + 1) != P_pq
                            continue
                        end

                        E_pqhg = E_h + E_g - E_p - E_q

                        @inbounds for Orb_r in pOrb_particle
                            a_r, l_r, j_r = Orb_r.a, Orb_r.l, Orb_r.j
                            E_r = HF_MBPT_h1b(a_r,-1,h_N)

                            @inbounds for Orb_s in nOrb_particle
                                a_s, l_s, j_s = Orb_s.a, Orb_s.l, Orb_s.j
                                E_s = HF_MBPT_h1b(a_s,-1,h_N)

                                if (rem(l_r + l_s,2) + 1) != P_pq
                                    continue
                                end

                                E_rshg = E_h + E_g - E_r - E_s

                                @inbounds for J in div(max(abs(j_h - j_g),abs(j_p - j_q),abs(j_r - j_s)),2):div(min(j_h + j_g, j_p + j_q, j_r + j_s),2)
                                    J_hat = Float64(2*J + 1)

                                    ME = J_hat * O2b_pn(a_h,a_g,a_p,a_q,J,P_pq,V_NN,Orb_NN) *
                                                 O2b_pn(a_p,a_q,a_r,a_s,J,P_pq,V_NN,Orb_NN) *
                                                 O2b_pn(a_r,a_s,a_h,a_g,J,P_pq,V_NN,Orb_NN) / E_pqhg / E_rshg

                                    Sum += ME
                                end
                            end
                        end
                    end
                end

                dE_threads[Tid] += Sum
            end
            dE3_pp += sum(dE_threads)

            # neutron-neutron contribution
            nnOrb_particle_list = Tuple{Orb1B,Orb1B}[(x,y) for x in nOrb_particle, y in nOrb_particle]

            dE_threads .= 0.0
            @time @inbounds Threads.@threads :static for (Orb_p,Orb_q) in nnOrb_particle_list
                Sum, Tid = 0.0, Threads.threadid()

                a_p, l_p, j_p = Orb_p.a, Orb_p.l, Orb_p.j
                E_p = HF_MBPT_h1b(a_p,-1,h_N)

                a_q, l_q, j_q = Orb_q.a, Orb_q.l, Orb_q.j
                E_q = HF_MBPT_h1b(a_q,-1,h_N)

                P_pq = rem(l_p + l_q, 2) + 1

                @inbounds for Orb_h in nOrb_hole
                    a_h, l_h, j_h = Orb_h.a, Orb_h.l, Orb_h.j
                    E_h = HF_MBPT_h1b(a_h,-1,h_N)

                    @inbounds for Orb_g in nOrb_hole
                        a_g, l_g, j_g = Orb_g.a, Orb_g.l, Orb_g.j
                        E_g = HF_MBPT_h1b(a_g,-1,h_N)

                        if (rem(l_h + l_g,2) + 1) != P_pq
                            continue
                        end

                        E_pqhg = E_h + E_g - E_p - E_q

                        @inbounds for Orb_r in nOrb_particle
                            a_r, l_r, j_r = Orb_r.a, Orb_r.l, Orb_r.j
                            E_r = HF_MBPT_h1b(a_r,-1,h_N)

                            @inbounds for Orb_s in nOrb_particle
                                a_s, l_s, j_s = Orb_s.a, Orb_s.l, Orb_s.j
                                E_s = HF_MBPT_h1b(a_s,-1,h_N)

                                if (rem(l_r + l_s,2) + 1) != P_pq
                                    continue
                                end

                                E_rshg = E_h + E_g - E_r - E_s

                                @inbounds for J in div(max(abs(j_h - j_g),abs(j_p - j_q),abs(j_r - j_s)),2):div(min(j_h + j_g, j_p + j_q, j_r + j_s),2)
                                    J_hat = Float64(2*J + 1)

                                    ME = 0.125 * J_hat * O2b_nn(a_h,a_g,a_p,a_q,J,P_pq,V_NN,Orb,Orb_NN) *
                                                         O2b_nn(a_p,a_q,a_r,a_s,J,P_pq,V_NN,Orb,Orb_NN) *
                                                         O2b_nn(a_r,a_s,a_h,a_g,J,P_pq,V_NN,Orb,Orb_NN) / E_pqhg / E_rshg

                                    Sum += ME
                                end
                            end
                        end
                    end
                end

                dE_threads[Tid] += Sum
            end
            dE3_pp += sum(dE_threads)

        # particle-hole sector
            Orb_particle = [[(Orb,-1) for Orb in pOrb_particle];
                            [(Orb,+1) for Orb in nOrb_particle]]

            Orb_hole = [[(Orb,-1) for Orb in pOrb_hole];
                        [(Orb,+1) for Orb in nOrb_hole]]

            Orb_particle_list = [(x, y) for x in Orb_particle, y in Orb_particle]

            dE_threads .= 0.0
            @time @inbounds Threads.@threads :static for ((Orb_p,t_p),(Orb_q,t_q)) in Orb_particle_list
                Sum, Tid = 0.0, Threads.threadid()

                a_p, l_p, j_p = Orb_p.a, Orb_p.l, Orb_p.j
                E_p = HF_MBPT_h1b(a_p,t_p,h_N)

                a_q, l_q, j_q = Orb_q.a, Orb_q.l, Orb_q.j
                E_q = HF_MBPT_h1b(a_q,t_q,h_N)

                @inbounds for (Orb_h,t_h) in Orb_hole
                    a_h, l_h, j_h = Orb_h.a, Orb_h.l, Orb_h.j
                    E_h = HF_MBPT_h1b(a_h,t_h,h_N)

                    @inbounds for (Orb_g,t_g) in Orb_hole
                        a_g, l_g, j_g = Orb_g.a, Orb_g.l, Orb_g.j
                        E_g = HF_MBPT_h1b(a_g,t_g,h_N)

                        if (t_p + t_q) != (t_h + t_g)
                            continue
                        end

                        @inbounds for J in div(max(abs(j_p - j_g),abs(j_h - j_q)),2):div(min(j_p + j_g, j_h + j_q),2)
                            J_hat = Float64(2*J + 1)

                            V_pqhg = HF_MBPT3_V2b_Pandya(a_p,l_p,j_p,t_p,a_q,l_q,j_q,t_q,
                                                         a_h,l_h,j_h,t_h,a_g,l_g,j_g,t_g,
                                                         J,Orb,Orb_NN,V_NN)

                            if abs(V_pqhg) < 1e-12
                                continue
                            end

                            E_pqhg = E_h + E_g - E_p - E_q

                            @inbounds for (Orb_r,t_r) in Orb_particle
                                a_r, l_r, j_r = Orb_r.a, Orb_r.l, Orb_r.j
                                E_r = HF_MBPT_h1b(a_r,t_r,h_N)

                                @inbounds for (Orb_f,t_f) in Orb_hole
                                    a_f, l_f, j_f = Orb_f.a, Orb_f.l, Orb_f.j
                                    E_f = HF_MBPT_h1b(a_f,t_f,h_N)

                                    if (t_h + t_r) != (t_q + t_f) || (t_g + t_f) != (t_p + t_r) || triangle(j_f,j_r,2*J) == false
                                        continue
                                    end

                                    V_hfrq = HF_MBPT3_V2b_Pandya(a_h,l_h,j_h,t_h,a_r,l_r,j_r,t_r,
                                                                 a_f,l_f,j_f,t_f,a_q,l_q,j_q,t_q,
                                                                 J,Orb,Orb_NN,V_NN)

                                    if abs(V_hfrq) < 1e-12
                                        continue
                                    end

                                    V_fgpr = HF_MBPT3_V2b_Pandya(a_f,l_f,j_f,t_f,a_g,l_g,j_g,t_g,
                                                                 a_p,l_p,j_p,t_p,a_r,l_r,j_r,t_r,
                                                                 J,Orb,Orb_NN,V_NN)

                                    if abs(V_fgpr) < 1e-12
                                        continue
                                    end

                                    E_prgf = E_g + E_f - E_p - E_r

                                    ME = J_hat * V_pqhg * V_hfrq * V_fgpr / E_pqhg / E_prgf
                                    Sum += ME
                                end
                            end
                        end
                    end
                end

                dE_threads[Tid] += Sum
            end
            dE3_ph += sum(dE_threads)

        # hole-hole sector
            # proton-proton contribution
            dE_threads .= 0.0
            @time @inbounds Threads.@threads :static for (Orb_p,Orb_q) in ppOrb_particle_list
                Sum, Tid = 0.0, Threads.threadid()

                a_p, l_p, j_p = Orb_p.a, Orb_p.l, Orb_p.j
                E_p = HF_MBPT_h1b(a_p,-1,h_N)

                a_q, l_q, j_q = Orb_q.a, Orb_q.l, Orb_q.j
                E_q = HF_MBPT_h1b(a_q,-1,h_N)

                P_pq = rem(l_p + l_q, 2) + 1

                @inbounds for Orb_h in pOrb_hole
                    a_h, l_h, j_h = Orb_h.a, Orb_h.l, Orb_h.j
                    E_h = HF_MBPT_h1b(a_h,-1,h_N)

                    @inbounds for Orb_g in pOrb_hole
                        a_g, l_g, j_g = Orb_g.a, Orb_g.l, Orb_g.j
                        E_g = HF_MBPT_h1b(a_g,-1,h_N)

                        if (rem(l_h + l_g,2) + 1) != P_pq
                            continue
                        end

                        E_pqhg = E_h + E_g - E_p - E_q

                        @inbounds for Orb_f in pOrb_hole
                            a_f, l_f, j_f = Orb_f.a, Orb_f.l, Orb_f.j
                            E_f = HF_MBPT_h1b(a_f,-1,h_N)

                            @inbounds for Orb_e in pOrb_hole
                                a_e, l_e, j_e = Orb_e.a, Orb_e.l, Orb_e.j
                                E_e = HF_MBPT_h1b(a_e,-1,h_N)

                                if (rem(l_f + l_e,2) + 1) != P_pq
                                    continue
                                end

                                E_pqfe = E_f + E_e - E_p - E_q

                                @inbounds for J in div(max(abs(j_h - j_g),abs(j_p - j_q),abs(j_f - j_e)),2):div(min(j_h + j_g, j_p + j_q, j_f + j_e),2)
                                    J_hat = Float64(2*J + 1)

                                    ME = 0.125 * J_hat * O2b_pp(a_p,a_q,a_h,a_g,J,P_pq,V_NN,Orb,Orb_NN) *
                                                         O2b_pp(a_h,a_g,a_f,a_e,J,P_pq,V_NN,Orb,Orb_NN) *
                                                         O2b_pp(a_f,a_e,a_p,a_q,J,P_pq,V_NN,Orb,Orb_NN) / E_pqhg / E_pqfe

                                    Sum += ME
                                end
                            end
                        end
                    end
                end

                dE_threads[Tid] += Sum
            end
            dE3_hh += sum(dE_threads)

            # proton-neutron contribution
            dE_threads .= 0.0
            @time @inbounds Threads.@threads :static for (Orb_p,Orb_q) in pnOrb_particle_list
                Sum, Tid = 0.0, Threads.threadid()

                a_p, l_p, j_p = Orb_p.a, Orb_p.l, Orb_p.j
                E_p = HF_MBPT_h1b(a_p,-1,h_N)

                a_q, l_q, j_q = Orb_q.a, Orb_q.l, Orb_q.j
                E_q = HF_MBPT_h1b(a_q,-1,h_N)

                P_pq = rem(l_p + l_q, 2) + 1

                @inbounds for Orb_h in pOrb_hole
                    a_h, l_h, j_h = Orb_h.a, Orb_h.l, Orb_h.j
                    E_h = HF_MBPT_h1b(a_h,-1,h_N)

                    @inbounds for Orb_g in nOrb_hole
                        a_g, l_g, j_g = Orb_g.a, Orb_g.l, Orb_g.j
                        E_g = HF_MBPT_h1b(a_g,-1,h_N)

                        if (rem(l_h + l_g,2) + 1) != P_pq
                            continue
                        end

                        E_pqhg = E_h + E_g - E_p - E_q

                        @inbounds for Orb_f in pOrb_hole
                            a_f, l_f, j_f = Orb_f.a, Orb_f.l, Orb_f.j
                            E_f = HF_MBPT_h1b(a_f,-1,h_N)

                            @inbounds for Orb_e in nOrb_hole
                                a_e, l_e, j_e = Orb_e.a, Orb_e.l, Orb_e.j
                                E_e = HF_MBPT_h1b(a_e,-1,h_N)

                                if (rem(l_f + l_e,2) + 1) != P_pq
                                    continue
                                end

                                E_pqfe = E_f + E_e - E_p - E_q

                                @inbounds for J in div(max(abs(j_h - j_g),abs(j_p - j_q),abs(j_f - j_e)),2):div(min(j_h + j_g, j_p + j_q, j_f + j_e),2)
                                    J_hat = Float64(2*J + 1)

                                    ME = J_hat * O2b_pn(a_p,a_q,a_h,a_g,J,P_pq,V_NN,Orb_NN) *
                                                 O2b_pn(a_h,a_g,a_f,a_e,J,P_pq,V_NN,Orb_NN) *
                                                 O2b_pn(a_f,a_e,a_p,a_q,J,P_pq,V_NN,Orb_NN) / E_pqhg / E_pqfe

                                    Sum += ME
                                end
                            end
                        end
                    end
                end

                dE_threads[Tid] += Sum
            end
            dE3_hh += sum(dE_threads)

            # neutron-neutron contribution
            dE_threads .= 0.0
            @time @inbounds Threads.@threads :static for (Orb_p,Orb_q) in nnOrb_particle_list
                Sum, Tid = 0.0, Threads.threadid()

                a_p, l_p, j_p = Orb_p.a, Orb_p.l, Orb_p.j
                E_p = HF_MBPT_h1b(a_p,-1,h_N)

                a_q, l_q, j_q = Orb_q.a, Orb_q.l, Orb_q.j
                E_q = HF_MBPT_h1b(a_q,-1,h_N)

                P_pq = rem(l_p + l_q, 2) + 1

                @inbounds for Orb_h in nOrb_hole
                    a_h, l_h, j_h = Orb_h.a, Orb_h.l, Orb_h.j
                    E_h = HF_MBPT_h1b(a_h,-1,h_N)

                    @inbounds for Orb_g in nOrb_hole
                        a_g, l_g, j_g = Orb_g.a, Orb_g.l, Orb_g.j
                        E_g = HF_MBPT_h1b(a_g,-1,h_N)

                        if (rem(l_h + l_g,2) + 1) != P_pq
                            continue
                        end

                        E_pqhg = E_h + E_g - E_p - E_q

                        @inbounds for Orb_f in nOrb_hole
                            a_f, l_f, j_f = Orb_f.a, Orb_f.l, Orb_f.j
                            E_f = HF_MBPT_h1b(a_f,-1,h_N)

                            @inbounds for Orb_e in nOrb_hole
                                a_e, l_e, j_e = Orb_e.a, Orb_e.l, Orb_e.j
                                E_e = HF_MBPT_h1b(a_e,-1,h_N)

                                if (rem(l_f + l_e,2) + 1) != P_pq
                                    continue
                                end

                                E_pqfe = E_f + E_e - E_p - E_q

                                @inbounds for J in div(max(abs(j_h - j_g),abs(j_p - j_q),abs(j_f - j_e)),2):div(min(j_h + j_g, j_p + j_q, j_f + j_e),2)
                                    J_hat = Float64(2*J + 1)

                                    ME = 0.125 * J_hat * O2b_nn(a_p,a_q,a_h,a_g,J,P_pq,V_NN,Orb,Orb_NN) *
                                                         O2b_nn(a_h,a_g,a_f,a_e,J,P_pq,V_NN,Orb,Orb_NN) *
                                                         O2b_nn(a_f,a_e,a_p,a_q,J,P_pq,V_NN,Orb,Orb_NN) / E_pqhg / E_pqfe

                                    Sum += ME
                                end
                            end
                        end
                    end
                end

                dE_threads[Tid] += Sum
            end
            dE3_hh += sum(dE_threads)

    # Calculate the total energy correction ...
    dE = dE2 + dE3_pp + dE3_ph + dE3_hh

    # Print the total HF-MBPT(2) energy correction ...
    println("\tHF-MBPT(2) energy    ...   E^(2)    = " * string(round(dE2, sigdigits=9)) * " MeV")

    # Print the total HF-MBPT(3) energy corrections ...
    println("\tHF-MBPT(3) energy    ...   E^(3)    = " * string(round(dE3_pp + dE3_ph + dE3_hh, sigdigits=9)) * " MeV")
    println("\t\t                           E^(3)_pp = " * string(round(dE3_pp, sigdigits=9)) * " MeV")
    println("\t\t                           E^(3)_ph = " * string(round(dE3_ph, sigdigits=9)) * " MeV")
    println("\t\t                           E^(3)_hh = " * string(round(dE3_hh, sigdigits=9)) * " MeV")

    # Print the total HF-MBPT energy corrections ...
    println("\tHF-MBPT(3) energy    ...   E^(tot)  = " * string(round(dE, sigdigits=9)) * " MeV\n")

    # Write the HF-MBPT energy corrections to the summary file ...
    Summary =  open(string("io/", IO, "/HF/HF_summary.dat"), "a")
        println(Summary, "\nSpherical Hartree-Fock Many-Body Perturbation Theory solution review:")
        @printf(Summary, "\nE_0^(2)        = %15.8f \t MeV \t\t ... \t LO HF-MBPT(2) correction to ground-state energy", dE2)
        @printf(Summary, "\nE_0^(3)        = %15.8f \t MeV \t\t ... \t NLO HF-MBPT(3) correction to ground-state energy", dE3_pp + dE3_ph + dE3_hh)
        @printf(Summary, "\nE_0^(3)_pp     = %15.8f \t MeV \t\t ... \t NLO HF-MBPT(3) pp correction to ground-state energy", dE3_pp)
        @printf(Summary, "\nE_0^(3)_ph     = %15.8f \t MeV \t\t ... \t NLO HF-MBPT(3) ph correction to ground-state energy", dE3_ph)
        @printf(Summary, "\nE_0^(3)_hh     = %15.8f \t MeV \t\t ... \t NLO HF-MBPT(3) hh correction to ground-state energy", dE3_hh)
        println(Summary, "\n____________________________________________________________________________________________________________________________")
        @printf(Summary, "\nE_0^(tot)      = %15.8f \t MeV \t\t ... \t NLO HF-MBPT(2+3) correction to ground-state energy\n\n", dE)
        @printf(Summary, "\nE_0^(2) / A    = %15.8f \t MeV \t\t ... \t NLO HF-MBPT(2) correction to ground-state energy per nucleon", dE2 / Float64(Params.Calc.A))
        @printf(Summary, "\nE_0^(3) / A    = %15.8f \t MeV \t\t ... \t NLO HF-MBPT(3) correction to ground-state energy",  (dE3_hh + dE3_ph + dE3_pp) / Float64(Params.Calc.A))
        @printf(Summary, "\nE_0^(3)_pp / A = %15.8f \t MeV \t\t ... \t NLO HF-MBPT(3) pp correction to ground-state energy per nucleon", dE3_pp / Float64(Params.Calc.A))
        @printf(Summary, "\nE_0^(3)_ph / A = %15.8f \t MeV \t\t ... \t NLO HF-MBPT(3) ph correction to ground-state energy per nucleon", dE3_ph / Float64(Params.Calc.A))
        @printf(Summary, "\nE_0^(3)_hh / A = %15.8f \t MeV \t\t ... \t NLO HF-MBPT(3) hh correction to ground-state energy per nucleon", dE3_hh / Float64(Params.Calc.A))
        println(Summary, "\n____________________________________________________________________________________________________________________________")
        @printf(Summary, "\nE_0^(tot) / A  = %15.8f \t MeV \t\t ... \t NLO HF-MBPT(2+3) correction to ground-state energy per nucleon", dE / Float64(Params.Calc.A))
    close(Summary)
    
    return
end

function HF_MBPT_OBDM(Params::Parameters,Orb::Vector{Orb1B},Orb_HF_MBPT::Orb1B_HF_MBPT,Orb_NN::Orb2B,h_N::O1B,V_NN::O2B)
    # Read calculation params ...
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1)*(N_max + 2),2)
    IO_path = "io/" * Params.Calc.Path

    # Calculate HF-MBPT OBDM corrections ...
    println("\nCalculating HF-MBPT OBDM corrections ...")

    # Unpack particle & hole orbitals ...
    pOrb_particle, pOrb_hole = Orb_HF_MBPT.p.p, Orb_HF_MBPT.h.p
    nOrb_particle, nOrb_hole = Orb_HF_MBPT.p.n, Orb_HF_MBPT.h.n

    # Initialize density matrices ...
    dpRho = zeros(Float64,a_max,a_max)
    dnRho = zeros(Float64,a_max,a_max)

    # pRho 1p-1h pp
    @inbounds Threads.@threads for Orb_p in pOrb_particle
        a_p, l_p, j_p = Orb_p.a, Orb_p.l, Orb_p.j
        E_p = h_N.p[a_p,a_p]
        j_p_hat = Float64(j_p + 1)

        @inbounds for Orb_h in pOrb_hole
            a_h, l_h, j_h = Orb_h.a, Orb_h.l, Orb_h.j
            E_h = h_N.p[a_h,a_h]

            E_hp = E_h - E_p

            if j_p != j_h || l_p != l_h
                continue
            end

            @inbounds for Orb_f in pOrb_hole
                a_f, l_f, j_f = Orb_f.a, Orb_f.l, Orb_f.j
                E_f = h_N.p[a_f,a_f]

                P_pf, P_hf = rem(l_p + l_f,2) + 1, rem(l_h + l_f,2) + 1

                @inbounds for J in div(abs(j_p - j_f),2):div((j_p + j_f),2)
                    J_hat = Float64(2*J + 1)

                    @inbounds for Orb_r in pOrb_particle
                        a_r, l_r, j_r = Orb_r.a, Orb_r.l, Orb_r.j
                        E_r = h_N.p[a_r,a_r]

                        @inbounds for Orb_q in pOrb_particle
                            a_q, l_q, j_q = Orb_q.a, Orb_q.l, Orb_q.j
                            E_q = h_N.p[a_q,a_q]
                            
                            if (abs(j_q - j_r) > 2*J) || (2*J > (j_q + j_r)) || ((rem(l_q + l_r, 2) + 1) != P_hf)
                                continue
                            end

                            E_hfqr = E_h + E_f - E_q - E_r

                            ME = 0.5 * J_hat / j_p_hat * O2b_pp(a_p,a_f,a_q,a_r,J,P_pf,V_NN,Orb,Orb_NN) *
                                O2b_pp(a_q,a_r,a_h,a_f,J,P_hf,V_NN,Orb,Orb_NN) / E_hp / E_hfqr

                            dpRho[a_p,a_h] += ME
                            dpRho[a_h,a_p] += ME
                        end
                    end
                end
            end

            @inbounds for Orb_r in pOrb_particle
                a_r, l_r, j_r = Orb_r.a, Orb_r.l, Orb_r.j
                E_r = h_N.p[a_r,a_r]

                P_pr, P_hr = rem(l_p + l_r,2) + 1, rem(l_p + l_r,2) + 1

                @inbounds for J in div(abs(j_h - j_r),2):div((j_h + j_r),2)
                    J_hat = Float64(2*J + 1)

                    @inbounds for Orb_f in pOrb_hole
                        a_f, l_f, j_f = Orb_f.a, Orb_f.l, Orb_f.j
                        E_f = h_N.p[a_f,a_f]

                        @inbounds for Orb_g in pOrb_hole
                            a_g, l_g, j_g = Orb_g.a, Orb_g.l, Orb_g.j
                            E_g = h_N.p[a_g,a_g]

                            if (abs(j_g - j_f) > 2*J) || (2*J > (j_g + j_f)) || ((rem(l_g + l_f, 2) + 1) != P_pr)
                                continue
                            end

                            E_gfpr = E_g + E_f - E_p - E_r

                            ME = -0.5 * J_hat / j_p_hat * O2b_pp(a_p,a_r,a_g,a_f,J,P_pr,V_NN,Orb,Orb_NN) * 
                                O2b_pp(a_g,a_f,a_h,a_r,J,P_hr,V_NN,Orb,Orb_NN) / E_hp / E_gfpr

                            dpRho[a_p,a_h] += ME
                            dpRho[a_h,a_p] += ME
                        end
                    end
                end
            end
        end
    end

    # pRho 1p-1h pn
    @inbounds Threads.@threads for Orb_p in pOrb_particle
        a_p, l_p, j_p = Orb_p.a, Orb_p.l, Orb_p.j
        E_p = h_N.p[a_p,a_p]
        j_p_hat = Float64(j_p + 1)

        @inbounds for Orb_h in pOrb_hole
            a_h, l_h, j_h = Orb_h.a, Orb_h.l, Orb_h.j
            E_h = h_N.p[a_h,a_h]

            E_hp = E_h - E_p

            if j_p != j_h || l_p != l_h
                continue
            end

            @inbounds for Orb_f in nOrb_hole
                a_f, l_f, j_f = Orb_f.a, Orb_f.l, Orb_f.j
                E_f = h_N.n[a_f,a_f]

                P_pf, P_hf = rem(l_p + l_f,2) + 1, rem(l_h + l_f,2) + 1

                @inbounds for J in div(abs(j_p - j_f),2):div((j_p + j_f),2)
                    J_hat = Float64(2*J + 1)

                    @inbounds for Orb_r in nOrb_particle
                        a_r, l_r, j_r = Orb_r.a, Orb_r.l, Orb_r.j
                        E_r = h_N.n[a_r,a_r]

                        @inbounds for Orb_q in pOrb_particle
                            a_q, l_q, j_q = Orb_q.a, Orb_q.l, Orb_q.j
                            E_q = h_N.p[a_q,a_q]

                            if (abs(j_q - j_r) > 2*J) || (2*J > (j_q + j_r)) || ((rem(l_q + l_r, 2) + 1) != P_hf)
                                continue
                            end

                            E_hgqr = E_h + E_f - E_q - E_r

                            ME = J_hat / j_p_hat * O2b_pn(a_p,a_f,a_q,a_r,J,P_pf,V_NN,Orb_NN) * 
                                O2b_pn(a_q,a_r,a_h,a_f,J,P_hf,V_NN,Orb_NN) / E_hp / E_hgqr

                            dpRho[a_p,a_h] += ME
                            dpRho[a_h,a_p] += ME
                        end
                    end
                end
            end

            @inbounds for Orb_r in nOrb_particle
                a_r, l_r, j_r = Orb_r.a, Orb_r.l, Orb_r.j
                E_r = h_N.n[a_r,a_r]

                P_pr, P_hr = rem(l_p + l_r,2) + 1, rem(l_h + l_r,2) + 1

                @inbounds for J in div(abs(j_h - j_r),2):div((j_h + j_r),2)
                    J_hat = Float64(2*J + 1)

                    @inbounds for Orb_f in nOrb_hole
                        a_f, l_f, j_f = Orb_f.a, Orb_f.l, Orb_f.j
                        E_f = h_N.n[a_f,a_f]

                        @inbounds for Orb_g in pOrb_hole
                            a_g, l_g, j_g = Orb_g.a, Orb_g.l, Orb_g.j
                            E_g = h_N.p[a_g,a_g]

                            if (abs(j_g - j_f) > 2*J) || (2*J > (j_g + j_f)) || ((rem(l_g + l_f, 2) + 1) != P_pr)
                                continue
                            end

                            E_gfpr = E_g + E_f - E_p - E_r

                            ME = -J_hat / j_p_hat * O2b_pn(a_p,a_r,a_g,a_f,J,P_pr,V_NN,Orb_NN) * 
                                O2b_pn(a_g,a_f,a_h,a_r,J,P_hr,V_NN,Orb_NN) / E_hp / E_gfpr

                            dpRho[a_p,a_h] += ME
                            dpRho[a_h,a_p] += ME
                        end
                    end
                end
            end
        end
    end

    # pRho 2p-2h pp
    @inbounds Threads.@threads for Orb_p in pOrb_particle
        a_p, l_p, j_p = Orb_p.a, Orb_p.l, Orb_p.j
        E_p = h_N.p[a_p,a_p]
        j_p_hat = Float64(j_p + 1)

        @inbounds for Orb_q in pOrb_particle
            a_q, l_q, j_q = Orb_q.a, Orb_q.l, Orb_q.j
            E_q = h_N.p[a_q,a_q]

            P_pq = rem(l_p + l_q,2) + 1

            @inbounds for J in div(abs(j_p - j_q),2):div((j_p + j_q),2)
                J_hat = Float64(2*J + 1)

                @inbounds for Orb_h in pOrb_hole
                    a_h, l_h, j_h = Orb_h.a, Orb_h.l, Orb_h.j
                    E_h = h_N.p[a_h,a_h]
                    
                    @inbounds for Orb_g in pOrb_hole
                        a_g, l_g, j_g = Orb_g.a, Orb_g.l, Orb_g.j
                        E_g = h_N.p[a_g,a_g]
                        P_hg = rem(l_h + l_g,2) + 1

                        if (abs(j_h - j_g) > 2*J) || (2*J > (j_h + j_g)) || (P_pq != P_hg)
                            continue
                        end

                        E_hgpq = E_h + E_g - E_p - E_q

                        @inbounds for Orb_a in pOrb_particle
                            a_a, l_a, j_a = Orb_a.a, Orb_a.l, Orb_a.j
                            E_a = h_N.p[a_a,a_a]

                            if ((rem(l_q + l_a, 2) + 1) != P_hg) || (j_a != j_p) || (l_a != l_p)
                                continue
                            end

                            E_hgaq = E_h + E_g - E_a - E_q

                            ME = 0.5 * J_hat / j_p_hat * O2b_pp(a_p,a_q,a_h,a_g,J,P_pq,V_NN,Orb,Orb_NN) *
                                O2b_pp(a_h,a_g,a_a,a_q,J,P_hg,V_NN,Orb,Orb_NN) / E_hgpq / E_hgaq
                            dpRho[a_p,a_a] += ME
                        end

                    end
                end
            end
        end
    end

    @inbounds Threads.@threads for Orb_h in pOrb_hole
        a_h, l_h, j_h = Orb_h.a, Orb_h.l, Orb_h.j
        E_h = h_N.p[a_h,a_h]
        j_h_hat = Float64(j_h + 1)

        @inbounds for Orb_g in pOrb_hole
            a_g, l_g, j_g = Orb_g.a, Orb_g.l, Orb_g.j
            E_g = h_N.p[a_g,a_g]
            P_hg = rem(l_h + l_g,2) + 1

            @inbounds for J in div(abs(j_h - j_g),2):div((j_h + j_g),2)
                J_hat = Float64(2*J + 1)

                @inbounds for Orb_p in pOrb_particle
                    a_p, l_p, j_p = Orb_p.a, Orb_p.l, Orb_p.j
                    E_p = h_N.p[a_p,a_p]

                    @inbounds for Orb_q in pOrb_particle
                        a_q, l_q, j_q = Orb_q.a, Orb_q.l, Orb_q.j
                        E_q = h_N.p[a_q,a_q]
                        P_pq = rem(l_p + l_q,2) + 1

                        if (abs(j_p - j_q) > 2*J) || (2*J > (j_p + j_q)) || (P_hg != P_pq)
                            continue
                        end

                        E_ghpq = E_h + E_g - E_p - E_q

                        @inbounds for Orb_c in pOrb_hole
                            a_c, l_c, j_c = Orb_c.a, Orb_c.l, Orb_c.j
                            E_c = h_N.p[a_c,a_c]

                            if ((rem(l_g + l_c, 2) + 1) != P_pq) || (j_c != j_h) || (l_c != l_h)
                                continue
                            end

                            E_cghp = E_c + E_g - E_p - E_q

                            ME = -0.5 * J_hat / j_h_hat * O2b_pp(a_p,a_q,a_h,a_g,J,P_hg,V_NN,Orb,Orb_NN) *
                                O2b_pp(a_c,a_g,a_p,a_q,J,P_pq,V_NN,Orb,Orb_NN) / E_ghpq / E_cghp
                            dpRho[a_c,a_h] += ME
                        end

                    end
                end
            end
        end
    end

    # pRho 2p-2h pn
    @inbounds Threads.@threads for Orb_p in pOrb_particle
        a_p, l_p, j_p = Orb_p.a, Orb_p.l, Orb_p.j
        E_p = h_N.p[a_p,a_p]
        j_p_hat = Float64(j_p + 1)

        @inbounds for Orb_q in nOrb_particle
            a_q, l_q, j_q = Orb_q.a, Orb_q.l, Orb_q.j
            E_q = h_N.n[a_q,a_q]
            P_pq = rem(l_p + l_q,2) + 1

            @inbounds for J in div(abs(j_p - j_q),2):div((j_p + j_q),2)
                J_hat = Float64(2*J + 1)

                @inbounds for Orb_h in pOrb_hole
                    a_h, l_h, j_h = Orb_h.a, Orb_h.l, Orb_h.j
                    E_h = h_N.p[a_h,a_h]

                    @inbounds for Orb_g in nOrb_hole
                        a_g, l_g, j_g = Orb_g.a, Orb_g.l, Orb_g.j
                        E_g = h_N.n[a_g,a_g]
                        P_hg = rem(l_h + l_g,2) + 1

                        if (abs(j_h - j_g) > 2*J) || (2*J > (j_h + j_g)) || (P_hg != P_pq)
                            continue
                        end

                        E_ghpq = E_h + E_g - E_p - E_q

                        @inbounds for Orb_a in pOrb_particle
                            a_a, l_a, j_a = Orb_a.a, Orb_a.l, Orb_a.j
                            E_a = h_N.p[a_a,a_a]

                            if ((rem(l_q + l_a, 2) + 1) != P_hg) || (j_a != j_p) || (l_a != l_p)
                                continue
                            end

                            E_hgaq = E_h + E_g - E_a - E_q
                            
                            ME = J_hat / j_p_hat * O2b_pn(a_p,a_q,a_h,a_g,J,P_pq,V_NN,Orb_NN) *
                                O2b_pn(a_h,a_g,a_a,a_q,J,P_hg,V_NN,Orb_NN) / E_ghpq / E_hgaq
                            dpRho[a_p,a_a] += ME
                        end
                    end
                end
            end
        end
    end

    @inbounds Threads.@threads for Orb_h in pOrb_hole
        a_h, l_h, j_h = Orb_h.a, Orb_h.l, Orb_h.j
        E_h = h_N.p[a_h,a_h]
        j_h_hat = Float64(j_h + 1)

        @inbounds for Orb_g in nOrb_hole
            a_g, l_g, j_g = Orb_g.a, Orb_g.l, Orb_g.j
            E_g = h_N.n[a_g,a_g]
            P_hg = rem(l_h + l_g,2) + 1

            @inbounds for J in div(abs(j_h - j_g),2):div((j_h + j_g),2)
                J_hat = Float64(2*J + 1)

                @inbounds for Orb_p in pOrb_particle
                    a_p, l_p, j_p = Orb_p.a, Orb_p.l, Orb_p.j
                    E_p = h_N.p[a_p,a_p]

                    @inbounds for Orb_q in nOrb_particle
                        a_q, l_q, j_q = Orb_q.a, Orb_q.l, Orb_q.j
                        E_q = h_N.n[a_q,a_q]
                        P_pq = rem(l_p + l_q,2) + 1

                        if (abs(j_p - j_q) > 2*J) || (2*J > (j_p + j_q)) || (P_hg != P_pq)
                            continue
                        end

                        E_hgpq = E_h + E_g - E_p - E_q

                        @inbounds for Orb_c in pOrb_hole
                            a_c, l_c, j_c = Orb_c.a, Orb_c.l, Orb_c.j
                            E_c = h_N.p[a_c,a_c]

                            if ((rem(l_g + l_c, 2) + 1) != P_pq) || (j_c != j_h) || (l_c != l_h)
                                continue
                            end

                            E_cgpq = E_c + E_g - E_p - E_q


                            ME = -J_hat / j_h_hat * O2b_pn(a_p,a_q,a_h,a_g,J,P_pq,V_NN,Orb_NN) *
                                O2b_pn(a_c,a_g,a_p,a_q,J,P_pq,V_NN,Orb_NN) / E_hgpq / E_cgpq
                            dpRho[a_c,a_h] += ME
                        end
                    end
                end
            end
        end
    end

    # nRho 1p-1h nn
    @inbounds Threads.@threads for Orb_p in nOrb_particle
        a_p, l_p, j_p = Orb_p.a, Orb_p.l, Orb_p.j
        E_p = h_N.n[a_p,a_p]
        j_p_hat = Float64(j_p + 1)

        @inbounds for Orb_h in nOrb_hole
            a_h, l_h, j_h = Orb_h.a, Orb_h.l, Orb_h.j
            E_h = h_N.n[a_h,a_h]

            if j_p != j_h || l_p != l_h
                continue
            end

            E_hp = E_h - E_p

            @inbounds for Orb_f in nOrb_hole
                a_f, l_f, j_f = Orb_f.a, Orb_f.l, Orb_f.j
                E_f = h_N.n[a_f,a_f]
                P_pf, P_hf = rem(l_p + l_f,2) + 1, rem(l_h + l_f,2) + 1

                @inbounds for J in div(abs(j_p - j_f),2):div((j_p + j_f),2)
                    J_hat = Float64(2*J + 1)

                    @inbounds for Orb_r in nOrb_particle
                        a_r, l_r, j_r = Orb_r.a, Orb_r.l, Orb_r.j
                        E_r = h_N.n[a_r,a_r]

                        @inbounds for Orb_q in nOrb_particle
                            a_q, l_q, j_q = Orb_q.a, Orb_q.l, Orb_q.j
                            E_q = h_N.n[a_q,a_q]

                            if (abs(j_q - j_r) > 2*J) || (2*J > (j_q + j_r)) || ((rem(l_q + l_r, 2) + 1) != P_hf)
                                continue
                            end

                            E_hfqr = E_h + E_f - E_q - E_r

                            ME = 0.5 * J_hat / j_p_hat * O2b_nn(a_p,a_f,a_q,a_r,J,P_pf,V_NN,Orb,Orb_NN) * 
                                O2b_nn(a_q,a_r,a_h,a_f,J,P_hf,V_NN,Orb,Orb_NN) / E_hp / E_hfqr

                            dnRho[a_p,a_h] += ME
                            dnRho[a_h,a_p] += ME
                        end
                    end
                end
            end

            @inbounds for Orb_r in nOrb_particle
                a_r, l_r, j_r = Orb_r.a, Orb_r.l, Orb_r.j
                E_r = h_N.n[a_r,a_r]
                P_pr, P_hr = rem(l_p + l_r,2) + 1, rem(l_h + l_r,2) + 1

                @inbounds for J in div(abs(j_h - j_r),2):div((j_h + j_r),2)
                    J_hat = Float64(2*J + 1)

                    @inbounds for Orb_f in nOrb_hole
                        a_f, l_f, j_f = Orb_f.a, Orb_f.l, Orb_f.j
                        E_f = h_N.n[a_f,a_f]

                        @inbounds for Orb_g in nOrb_hole
                            a_g, l_g, j_g = Orb_g.a, Orb_g.l, Orb_g.j
                            E_g = h_N.n[a_g,a_g]

                            if (abs(j_g - j_f) > 2*J) || (2*J > (j_g + j_f)) || ((rem(l_g + l_f, 2) + 1) != P_pr)
                                continue
                            end

                            E_gfpr = E_g + E_f - E_p - E_r

                            ME = -0.5 * J_hat / j_p_hat * O2b_nn(a_p,a_r,a_g,a_f,J,P_pr,V_NN,Orb,Orb_NN) * 
                                O2b_nn(a_g,a_f,a_h,a_r,J,P_hr,V_NN,Orb,Orb_NN) / E_hp / E_gfpr
                            dnRho[a_p,a_h] += ME
                            dnRho[a_h,a_p] += ME
                        end
                    end
                end
            end
        end
    end

    # nRho 1p-1h pn
    @inbounds Threads.@threads for Orb_p in nOrb_particle
        a_p, l_p, j_p = Orb_p.a, Orb_p.l, Orb_p.j
        E_p = h_N.n[a_p,a_p]
        j_p_hat = Float64(j_p + 1)

        @inbounds for Orb_h in nOrb_hole
            a_h, l_h, j_h = Orb_h.a, Orb_h.l, Orb_h.j
            E_h = h_N.n[a_h,a_h]

            if j_p != j_h || l_p != l_h
                continue
            end

            E_hp = E_h - E_p

            @inbounds for Orb_f in pOrb_hole
                a_f, l_f, j_f = Orb_f.a, Orb_f.l, Orb_f.j
                E_f = h_N.p[a_f,a_f]
                P_pf, P_hf = rem(l_p + l_f,2) + 1, rem(l_h + l_f,2) + 1

                @inbounds for J in div(abs(j_p - j_f),2):div((j_p + j_f),2)
                    J_hat = Float64(2*J + 1)

                    @inbounds for Orb_r in pOrb_particle
                        a_r, l_r, j_r = Orb_r.a, Orb_r.l, Orb_r.j
                        E_r = h_N.p[a_r,a_r]

                        @inbounds for Orb_q in nOrb_particle
                            a_q, l_q, j_q = Orb_q.a, Orb_q.l, Orb_q.j
                            E_q = h_N.n[a_q,a_q]

                            if (abs(j_q - j_r) > 2*J) || (2*J > (j_q + j_r)) || ((rem(l_q + l_r, 2) + 1) != P_hf)
                                continue
                            end

                            E_hfqr = E_h + E_f - E_q - E_r

                            ME = J_hat / j_p_hat * O2b_pn(a_f,a_p,a_r,a_q,J,P_pf,V_NN,Orb_NN) * 
                                O2b_pn(a_r,a_q,a_f,a_h,J,P_hf,V_NN,Orb_NN) / E_hp / E_hfqr

                            dnRho[a_p,a_h] += ME
                            dnRho[a_h,a_p] += ME
                        end
                    end
                end
            end

            @inbounds for Orb_r in pOrb_particle
                a_r, l_r, j_r = Orb_r.a, Orb_r.l, Orb_r.j
                E_r = h_N.p[a_r,a_r]
                P_pr, P_hr = rem(l_p + l_r,2) + 1, rem(l_h + l_r,2) + 1

                @inbounds for J in div(abs(j_h - j_r),2):div((j_h + j_r),2)
                    J_hat = Float64(2*J + 1)

                    @inbounds for Orb_f in pOrb_hole
                        a_f, l_f, j_f = Orb_f.a, Orb_f.l, Orb_f.j
                        E_f = h_N.p[a_f,a_f]

                        @inbounds for Orb_g in nOrb_hole
                            a_g, l_g, j_g = Orb_g.a, Orb_g.l, Orb_g.j
                            E_g = h_N.n[a_g,a_g]

                            if (abs(j_g - j_f) > 2*J) || (2*J > (j_g + j_f)) || ((rem(l_g + l_f, 2) + 1) != P_pr)
                                continue
                            end

                            E_gfpr = E_g + E_f - E_p - E_r

                            ME = -J_hat / j_p_hat * O2b_pn(a_r,a_p,a_f,a_g,J,P_pr,V_NN,Orb_NN) * 
                                O2b_pn(a_f,a_g,a_r,a_h,J,P_hr,V_NN,Orb_NN) / E_hp / E_gfpr

                            dnRho[a_p,a_h] += ME
                            dnRho[a_h,a_p] += ME
                        end
                    end
                end
            end
        end
    end

    # nRho 2p-2h nn
    @inbounds Threads.@threads for Orb_p in nOrb_particle
        a_p, l_p, j_p = Orb_p.a, Orb_p.l, Orb_p.j
        E_p = h_N.n[a_p,a_p]
        j_p_hat = Float64(j_p + 1)

        @inbounds for Orb_q in nOrb_particle
            a_q, l_q, j_q = Orb_q.a, Orb_q.l, Orb_q.j
            E_q = h_N.n[a_q,a_q]
            P_pq = rem(l_p + l_q,2) + 1

            @inbounds for J in div(abs(j_p - j_q),2):div((j_p + j_q),2)
                J_hat = Float64(2*J + 1)

                @inbounds for Orb_h in nOrb_hole
                    a_h, l_h, j_h = Orb_h.a, Orb_h.l, Orb_h.j
                    E_h = h_N.n[a_h,a_h]

                    @inbounds for Orb_g in nOrb_hole
                        a_g, l_g, j_g = Orb_g.a, Orb_g.l, Orb_g.j
                        E_g = h_N.n[a_g,a_g]
                        P_hg = rem(l_h + l_g,2) + 1

                        if (abs(j_h - j_g) > 2*J) || (2*J > (j_h + j_g)) || (P_hg != P_pq)
                            continue
                        end

                        E_hgpq = E_h + E_g - E_p - E_q

                        @inbounds for Orb_a in nOrb_particle
                            a_a, l_a, j_a = Orb_a.a, Orb_a.l, Orb_a.j
                            E_a = h_N.n[a_a,a_a]

                            if ((rem(l_q + l_a, 2) + 1) != P_hg) || (j_a != j_p) || (l_a != l_p)
                                continue
                            end

                            E_hgaq = E_h + E_g - E_a - E_q

                            ME = 0.5 * J_hat / j_p_hat * O2b_nn(a_p,a_q,a_h,a_g,J,P_pq,V_NN,Orb,Orb_NN) *
                                O2b_nn(a_h,a_g,a_a,a_q,J,P_hg,V_NN,Orb,Orb_NN) / E_hgpq / E_hgaq

                            dnRho[a_p,a_a] += ME
                        end
                    end
                end
            end
        end
    end

    @inbounds Threads.@threads for Orb_h in nOrb_hole
        a_h, l_h, j_h = Orb_h.a, Orb_h.l, Orb_h.j
        E_h = h_N.n[a_h,a_h]
        j_h_hat = Float64(j_h + 1)

        @inbounds for Orb_g in nOrb_hole
            a_g, l_g, j_g = Orb_g.a, Orb_g.l, Orb_g.j
            E_g = h_N.n[a_g,a_g]
            P_hg = rem(l_h + l_g,2) + 1

            @inbounds for J in div(abs(j_h - j_g),2):div((j_h + j_g),2)
                J_hat = Float64(2*J + 1)

                @inbounds for Orb_p in nOrb_particle
                    a_p, l_p, j_p = Orb_p.a, Orb_p.l, Orb_p.j
                    E_p = h_N.n[a_p,a_p]

                    @inbounds for Orb_q in nOrb_particle
                        a_q, l_q, j_q = Orb_q.a, Orb_q.l, Orb_q.j
                        E_q = h_N.n[a_q,a_q]
                        P_pq = rem(l_p + l_q,2) + 1

                        if (abs(j_p - j_q) > 2*J) || (2*J > (j_p + j_q)) || (P_hg != P_pq)
                            continue
                        end

                        E_hgpq = E_h + E_g - E_p - E_q

                        @inbounds for Orb_c in nOrb_hole
                            a_c, l_c, j_c = Orb_c.a, Orb_c.l, Orb_c.j
                            E_c = h_N.n[a_c,a_c]

                            if ((rem(l_g + l_c, 2) + 1) != P_pq) || (j_c != j_h) || (l_c != l_h)
                                continue
                            end

                            E_cgpq = E_c + E_g - E_p - E_q

                            ME = -0.5 * J_hat / j_h_hat * O2b_nn(a_p,a_q,a_h,a_g,J,P_pq,V_NN,Orb,Orb_NN) *
                                O2b_nn(a_c,a_g,a_p,a_q,J,P_pq,V_NN,Orb,Orb_NN) / E_hgpq / E_cgpq

                            dnRho[a_c,a_h] += ME
                        end
                    end
                end
            end
        end
    end

    # nRho 2p-2h pn
    @inbounds Threads.@threads for Orb_p in nOrb_particle
        a_p, l_p, j_p = Orb_p.a, Orb_p.l, Orb_p.j
        E_p = h_N.n[a_p,a_p]
        j_p_hat = Float64(j_p + 1)

        @inbounds for Orb_q in pOrb_particle
            a_q, l_q, j_q = Orb_q.a, Orb_q.l, Orb_q.j
            E_q = h_N.p[a_q,a_q]
            P_pq = rem(l_p + l_q,2) + 1

            @inbounds for J in div(abs(j_p - j_q),2):div((j_p + j_q),2)
                J_hat = Float64(2*J + 1)

                @inbounds for Orb_h in nOrb_hole
                    a_h, l_h, j_h = Orb_h.a, Orb_h.l, Orb_h.j
                    E_h = h_N.n[a_h,a_h]

                    @inbounds for Orb_g in pOrb_hole
                        a_g, l_g, j_g = Orb_g.a, Orb_g.l, Orb_g.j
                        E_g = h_N.p[a_g,a_g]
                        P_hg = rem(l_h + l_g,2) + 1

                        if (abs(j_h - j_g) > 2*J) || (2*J > (j_h + j_g)) || (P_hg != P_pq)
                            continue
                        end

                        E_hgpq = E_h + E_g - E_p - E_q
                    
                        @inbounds for Orb_a in nOrb_particle
                            a_a, l_a, j_a = Orb_a.a, Orb_a.l, Orb_a.j
                            E_a = h_N.n[a_a,a_a]

                            if ((rem(l_q + l_a, 2) + 1) != P_hg) || (j_a != j_p) || (l_a != l_p)
                                continue
                            end

                            E_hgaq = E_h + E_g - E_a - E_q

                            ME = J_hat / j_p_hat * O2b_pn(a_q,a_p,a_g,a_h,J,P_hg,V_NN,Orb_NN) *
                                O2b_pn(a_g,a_h,a_q,a_a,J,P_hg,V_NN,Orb_NN) / E_hgpq / E_hgaq

                            dnRho[a_p,a_a] += ME
                        end
                    end
                end
            end
        end
    end

    @inbounds Threads.@threads for Orb_h in nOrb_hole
        a_h, l_h, j_h = Orb_h.a, Orb_h.l, Orb_h.j
        E_h = h_N.n[a_h,a_h]
        j_h_hat = Float64(j_h + 1)

        @inbounds for Orb_g in pOrb_hole
            a_g, l_g, j_g = Orb_g.a, Orb_g.l, Orb_g.j
            E_g = h_N.p[a_g,a_g]
            P_hg = rem(l_h + l_g,2) + 1

            @inbounds for J in div(abs(j_h - j_g),2):div((j_h + j_g),2)
                J_hat = Float64(2*J + 1)

                @inbounds for Orb_p in nOrb_particle
                    a_p, l_p, j_p = Orb_p.a, Orb_p.l, Orb_p.j
                    E_p = h_N.n[a_p,a_p]

                    @inbounds for Orb_q in pOrb_particle
                        a_q, l_q, j_q = Orb_q.a, Orb_q.l, Orb_q.j
                        E_q = h_N.p[a_q,a_q]
                        P_pq = rem(l_p + l_q,2) + 1

                        if (abs(j_p - j_q) > 2*J) || (2*J > (j_p + j_q)) || (P_hg != P_pq)
                            continue
                        end

                        E_hgpq = E_h + E_g - E_p - E_q
                    
                        @inbounds for Orb_c in nOrb_hole
                            a_c, l_c, j_c = Orb_c.a, Orb_c.l, Orb_c.j
                            E_c = h_N.n[a_c,a_c]

                            if ((rem(l_g + l_c, 2) + 1) != P_pq) || (j_c != j_h) || (l_c != l_h)
                                continue
                            end

                            E_cgpq = E_c + E_g - E_p - E_q

                            ME = -J_hat / j_h_hat * O2b_pn(a_q,a_p,a_g,a_h,J,P_pq,V_NN,Orb_NN) *
                                O2b_pn(a_g,a_c,a_q,a_p,J,P_pq,V_NN,Orb_NN) / E_hgpq / E_cgpq

                            dnRho[a_c,a_h] += ME
                        end
                    end
                end
            end
        end
    end

    # Read the transformation matrix C ... LHO -> HF ...
    C = O1b_import(Params, Orb,IO_path * "/HF/bin/C.bin")

    # Read the HF OBDM Rho ... in the HF basis ...
    Rho_HF = O1b_import(Params, Orb,IO_path * "/HF/bin/rho.bin")

    # Include HF-MBPT corrections to Rho ...
    pRho = Rho_HF.p .+ dpRho
    nRho = Rho_HF.n .+ dnRho

    # Construct the HF-MBPT natural orbitals basis to extract occupations & radial densities ...
    Orb_NAT, D, Rho_NAT = OBDM_natural_orbitals(Params,Orb,O1B(pRho,nRho))

    # Calculate & export radial HF-MBPT densities & radii ...
    Summary_File = IO_path * "/HF/HF_summary.dat"
    Densities_File = IO_path * "/HF/densities/HF_MBPT_radial_densities.dat"
    OBDM_export(Params,Orb,Summary_File,Densities_File,O1B(C.p * D.p * Rho_NAT.p * D.p' * C.p', C.n * D.n * Rho_NAT.n * D.n' * C.n'),O1B(C.p * D.p, C.n * D.n))

    # Calculate & export HF-MBPT occupation numbers ...
    HF_MBPT_occupation(Params,Rho_HF,Rho_NAT)

    # In case of export, write the HF-MBPT natural orbitals & OBDM to temporary files for further use...
    if Params.Calc.HF.MBPT.Export == true
        # Express natural orbitals in the LHO basis ...
        C_NAT = O1B(C.p * D.p * C.p', C.n * D.n * C.n')

        # Export the HF-MBPT natural orbitals ...
        Orb_export_path = IO_path * "/HF/bin/orb_temp.bin"
        orbitals_export(Params,Orb_NAT,Orb_export_path)

        C_export_path = IO_path * "/HF/bin/C_temp.bin"
        O1b_export(Params,Orb_NAT,C_NAT,C_export_path)

        # Export the HF-MBPT OBDM ...
        Rho_export_path = IO_path * "/HF/bin/rho_temp.bin"
        O1b_export(Params,Orb_NAT,Rho_NAT,Rho_export_path)        
    end

    return
end

function HF_MBPT_TBCF(Params::Parameters,C::O1B,h_N::O1B,V_NN::O2B,Orb::Vector{Orb1B},Orb_HF_MBPT::Orb1B_HF_MBPT,Orb_NN::Orb2B)
    # Read parameters ...
    N_2max = Params.Calc.N2max
    J_max = N_2max + 1

    # Iteraction list for J & P ...
    JP_list = JP_initialize(J_max)

    # Unpack particle & hole orbitals ...
    pOrb_particle, pOrb_hole = Orb_HF_MBPT.p.p, Orb_HF_MBPT.h.p
    nOrb_particle, nOrb_hole = Orb_HF_MBPT.p.n, Orb_HF_MBPT.h.n

    # Create lists of particle & hole orbitals with isospin labels ...
    Orb_particle = [[(Orb,-1) for Orb in pOrb_particle];
                    [(Orb,+1) for Orb in nOrb_particle]]
    Orb_hole = [[(Orb,-1) for Orb in pOrb_hole];
                [(Orb,+1) for Orb in nOrb_hole]]

    # Define local functions to distinguish particle & hole orbitals ...
        @inline particle(O::Float64) = (O <= 1e-3)

        @inline hole(O::Float64) = (O >= 0.999)

    # Define local functions to precreate Bra & Ket iteration indices ...
        function HF_MBPT_sigma_orbitals_JP(J::Int64,P::Int64,Orb_NN::Orb2B)
            N_T0, N_T1 = Orb_NN.N[1,P,J+1], Orb_NN.N[2,P,J+1]

            BraKet_count = 0

            @inbounds for Bra in 1:N_T0
                @inbounds for Ket in 1:Bra
                    BraKet_count += 1
                end
            end

            @inbounds for Bra in 1:N_T1
                @inbounds for Ket in 1:Bra
                    BraKet_count += 2
                end
            end

            BraKet_list = Vector{Tuple{Int64,Int64,Int64}}(undef,BraKet_count)

            BraKet_count = 0

            # Case of proton-proton orbitals ...
            @inbounds for Bra in 1:N_T1
                @inbounds for Ket in 1:Bra
                    BraKet_count += 1
                    BraKet_list[BraKet_count] = (Bra,Ket,-1)
                end
            end

            # Case of proton-neutron orbitals ...
            @inbounds for Bra in 1:N_T0
                @inbounds for Ket in 1:Bra
                    BraKet_count += 1
                    BraKet_list[BraKet_count] = (Bra,Ket,0)
                end
            end

            # Case of neutron-neutron orbitals ...
            @inbounds for Bra in 1:N_T1
                @inbounds for Ket in 1:Bra
                    BraKet_count += 1
                    BraKet_list[BraKet_count] = (Bra,Ket,1)
                end
            end

            return BraKet_list
        end

        # Define local functions to calculate the MBPT(2) & MBPT(3) contributions to 2-body correlation function matrix Sigma ...
        function HF_MBPT2_sigma(a::Int64,t_a::Int64,b::Int64,t_b::Int64,c::Int64,t_c::Int64,d::Int64,t_d::Int64,J::Int64,P::Int64)
            E_a, E_b, E_c, E_d = HF_MBPT_h1b(a,t_a,h_N), HF_MBPT_h1b(b,t_b,h_N), HF_MBPT_h1b(c,t_c,h_N), HF_MBPT_h1b(d,t_d,h_N)

            sigma = 0.0

            if (t_a + t_b) == -2
                ME = O2b_pp(a,b,c,d,J,P,V_NN,Orb,Orb_NN) / (E_a + E_b - E_c - E_d)
                sigma -= ME

            elseif (t_a + t_b) == 0
                ME = O2b_pn(a,b,c,d,J,P,V_NN,Orb_NN) / (E_a + E_b - E_c - E_d)
                sigma -= ME

            elseif (t_a + t_b) == 2
                ME = O2b_nn(a,b,c,d,J,P,V_NN,Orb,Orb_NN) / (E_a + E_b - E_c - E_d)
                sigma -= ME
            end

            return sigma
        end

        function HF_MBPT3_sigma_pp(a::Int64,t_a::Int64,b::Int64,t_b::Int64,c::Int64,t_c::Int64,d::Int64,t_d::Int64,J::Int64,P::Int64)
            E_a, E_b, E_c, E_d = HF_MBPT_h1b(a,t_a,h_N), HF_MBPT_h1b(b,t_b,h_N), HF_MBPT_h1b(c,t_c,h_N), HF_MBPT_h1b(d,t_d,h_N)

            sigma = 0.0

            if (t_a + t_b) == -2
                @inbounds for Orb_r in pOrb_particle
                    r, l_r, j_r, t_r = Orb_r.a, Orb_r.l, Orb_r.j, -1
                    E_r = HF_MBPT_h1b(r,t_r,h_N)

                    @inbounds for Orb_s in pOrb_particle
                        s, l_s, j_s, t_s = Orb_s.a, Orb_s.l, Orb_s.j, -1
                        E_s = HF_MBPT_h1b(s,t_s,h_N)

                        if (triangle(j_r,j_s,2*J) == false) || (rem(l_r + l_s,2) + 1 != P)
                            continue
                        end

                        ME = 0.5 * O2b_pp(a,b,r,s,J,P,V_NN,Orb,Orb_NN) * O2b_pp(r,s,c,d,J,P,V_NN,Orb,Orb_NN) / (E_a + E_b - E_c - E_d) / (E_r + E_s - E_c - E_d)
                        sigma += ME
                    end
                end

            elseif (t_a + t_b) == 0
                @inbounds for Orb_r in pOrb_particle
                    r, l_r, j_r, t_r = Orb_r.a, Orb_r.l, Orb_r.j, -1
                    E_r = HF_MBPT_h1b(r,t_r,h_N)

                    @inbounds for Orb_s in nOrb_particle
                        s, l_s, j_s, t_s = Orb_s.a, Orb_s.l, Orb_s.j, 1
                        E_s = HF_MBPT_h1b(s,t_s,h_N)

                        if (triangle(j_r,j_s,2*J) == false) || (rem(l_r + l_s,2) + 1 != P)
                            continue
                        end

                        ME = O2b_pn(a,b,r,s,J,P,V_NN,Orb_NN) * O2b_pn(r,s,c,d,J,P,V_NN,Orb_NN) / (E_a + E_b - E_c - E_d) / (E_r + E_s - E_c - E_d)
                        sigma += ME
                    end
                end

            elseif (t_a + t_b) == 2
                @inbounds for Orb_r in nOrb_particle
                    r, l_r, j_r, t_r = Orb_r.a, Orb_r.l, Orb_r.j, 1
                    E_r = HF_MBPT_h1b(r,t_r,h_N)

                    @inbounds for Orb_s in nOrb_particle
                        s, l_s, j_s, t_s = Orb_s.a, Orb_s.l, Orb_s.j, 1
                        E_s = HF_MBPT_h1b(s,t_s,h_N)

                        if (triangle(j_r,j_s,2*J) == false) || (rem(l_r + l_s,2) + 1 != P)
                            continue
                        end

                        ME = 0.5 * O2b_nn(a,b,r,s,J,P,V_NN,Orb,Orb_NN) * O2b_nn(r,s,c,d,J,P,V_NN,Orb,Orb_NN) / (E_a + E_b - E_c - E_d) / (E_r + E_s - E_c - E_d)
                        sigma += ME
                    end
                end
            end

            return sigma
        end

        function HF_MBPT3_sigma_hh(a::Int64,t_a::Int64,b::Int64,t_b::Int64,c::Int64,t_c::Int64,d::Int64,t_d::Int64,J::Int64,P::Int64)
            E_a, E_b, E_c, E_d = HF_MBPT_h1b(a,t_a,h_N), HF_MBPT_h1b(b,t_b,h_N), HF_MBPT_h1b(c,t_c,h_N), HF_MBPT_h1b(d,t_d,h_N)

            sigma = 0.0

            if (t_a + t_b) == -2
                @inbounds for Orb_r in pOrb_hole
                    r, l_r, j_r, t_r = Orb_r.a, Orb_r.l, Orb_r.j, -1
                    E_r = HF_MBPT_h1b(r,t_r,h_N)

                    @inbounds for Orb_s in pOrb_hole
                        s, l_s, j_s, t_s = Orb_s.a, Orb_s.l, Orb_s.j, -1
                        E_s = HF_MBPT_h1b(s,t_s,h_N)

                        if (triangle(j_r,j_s,2*J) == false) || (rem(l_r + l_s,2) + 1 != P)
                            continue
                        end

                        ME = 0.5 * O2b_pp(a,b,r,s,J,P,V_NN,Orb,Orb_NN) * O2b_pp(r,s,c,d,J,P,V_NN,Orb,Orb_NN) / (E_a + E_b - E_c - E_d) / (E_a + E_b - E_r - E_s)

                        sigma += ME
                    end
                end

            elseif (t_a + t_b) == 0
                @inbounds for Orb_r in pOrb_hole
                    r, l_r, j_r, t_r = Orb_r.a, Orb_r.l, Orb_r.j, -1
                    E_r = HF_MBPT_h1b(r,t_r,h_N)

                    @inbounds for Orb_s in nOrb_hole
                        s, l_s, j_s, t_s = Orb_s.a, Orb_s.l, Orb_s.j, 1
                        E_s = HF_MBPT_h1b(s,t_s,h_N)

                        if (triangle(j_r,j_s,2*J) == false) || (rem(l_r + l_s,2) + 1 != P)
                            continue
                        end

                        ME = O2b_pn(a,b,r,s,J,P,V_NN,Orb_NN) * O2b_pn(r,s,c,d,J,P,V_NN,Orb_NN) / (E_a + E_b - E_c - E_d) / (E_a + E_b - E_r - E_s)

                        sigma += ME
                    end
                end

            elseif (t_a + t_b) == 2
                @inbounds for Orb_r in nOrb_hole
                    r, l_r, j_r, t_r = Orb_r.a, Orb_r.l, Orb_r.j, 1
                    E_r = HF_MBPT_h1b(r,t_r,h_N)

                    @inbounds for Orb_s in nOrb_hole
                        s, l_s, j_s, t_s = Orb_s.a, Orb_s.l, Orb_s.j, 1
                        E_s = HF_MBPT_h1b(s,t_s,h_N)

                        if (triangle(j_r,j_s,2*J) == false) || (rem(l_r + l_s,2) + 1 != P)
                            continue
                        end

                        ME = 0.5 * O2b_nn(a,b,r,s,J,P,V_NN,Orb,Orb_NN) * O2b_nn(r,s,c,d,J,P,V_NN,Orb,Orb_NN) / (E_a + E_b - E_c - E_d) / (E_a + E_b - E_r - E_s)

                        sigma += ME
                    end
                end
            end

            return sigma
        end

        function HF_MBPT3_sigma_ph_direct(a::Int64,l_a::Int64,j_a::Int64,t_a::Int64,E_a::Float64,
                                            b::Int64,l_b::Int64,j_b::Int64,t_b::Int64,E_b::Float64,
                                            c::Int64,l_c::Int64,j_c::Int64,t_c::Int64,E_c::Float64,
                                            d::Int64,l_d::Int64,j_d::Int64,t_d::Int64,E_d::Float64,J::Int64)
            sigma = 0.0

            I_min, I_max = div(max(abs(j_a - j_d), abs(j_b - j_c)),2), div(min(j_a + j_d, j_b + j_c),2)

            @inbounds for I in I_min:I_max
                Amp_abcd_I = Float64(2*I + 1) * w6j(j_a,j_b,2*J,j_c,j_d,2*I) / (E_a + E_b - E_c - E_d)

                @inbounds for (Orb_r,t_r) in Orb_particle
                    r, l_r, j_r = Orb_r.a, Orb_r.l, Orb_r.j
                    E_r = HF_MBPT_h1b(r,t_r,h_N)

                    @inbounds for (Orb_f,t_f) in Orb_hole
                        f, l_f, j_f = Orb_f.a, Orb_f.l, Orb_f.j
                        E_f = HF_MBPT_h1b(f,t_f,h_N)

                        if (rem(l_f + l_b,2) != rem(l_c + l_r,2)) || (rem(l_a + l_r,2) != rem(l_f + l_d,2)) || (t_b + t_f != t_c + t_r) ||
                            (t_a + t_r != t_d + t_f)  || triangle(j_r,j_f,2*I) == false
                            continue
                        end

                        V_fbcr = HF_MBPT3_V2b_Pandya(f,l_f,j_f,t_f,b,l_b,j_b,t_b,c,l_c,j_c,t_c,r,l_r,j_r,t_r,I,Orb,Orb_NN,V_NN)
                        V_arfd = HF_MBPT3_V2b_Pandya(a,l_a,j_a,t_a,r,l_r,j_r,t_r,f,l_f,j_f,t_f,d,l_d,j_d,t_d,I,Orb,Orb_NN,V_NN)

                        sigma -= Amp_abcd_I * V_fbcr * V_arfd / (E_a + E_r - E_d - E_f)
                    end
                end
            end

            return sigma
        end

        function HF_MBPT3_sigma_ph(a::Int64,t_a::Int64,b::Int64,t_b::Int64,c::Int64,t_c::Int64,d::Int64,t_d::Int64,J::Int64)
            l_a, j_a, E_a = Orb[a].l, Orb[a].j, HF_MBPT_h1b(a,t_a,h_N)
            l_b, j_b, E_b = Orb[b].l, Orb[b].j, HF_MBPT_h1b(b,t_b,h_N)
            l_c, j_c, E_c = Orb[c].l, Orb[c].j, HF_MBPT_h1b(c,t_c,h_N)
            l_d, j_d, E_d = Orb[d].l, Orb[d].j, HF_MBPT_h1b(d,t_d,h_N)

            Phase_ab = phase(div(j_a + j_b,2) - J)
            Phase_cd = phase(div(j_c + j_d,2) - J)

            if t_a == t_b
                Phase_ab *= -1.0
            end

            if t_c == t_d
                Phase_cd *= -1.0
            end

            sigma = HF_MBPT3_sigma_ph_direct(a,l_a,j_a,t_a,E_a,b,l_b,j_b,t_b,E_b,c,l_c,j_c,t_c,E_c,d,l_d,j_d,t_d,E_d,J) +
                    HF_MBPT3_sigma_ph_direct(b,l_b,j_b,t_b,E_b,a,l_a,j_a,t_a,E_a,c,l_c,j_c,t_c,E_c,d,l_d,j_d,t_d,E_d,J) * Phase_ab +
                    HF_MBPT3_sigma_ph_direct(a,l_a,j_a,t_a,E_a,b,l_b,j_b,t_b,E_b,d,l_d,j_d,t_d,E_d,c,l_c,j_c,t_c,E_c,J) * Phase_cd +
                    HF_MBPT3_sigma_ph_direct(b,l_b,j_b,t_b,E_b,a,l_a,j_a,t_a,E_a,d,l_d,j_d,t_d,E_d,c,l_c,j_c,t_c,E_c,J) * Phase_ab * Phase_cd

            return sigma
        end

    println("\nInitializing HF-MBPT 2-body correlation function matrix Sigma ...")

    # Initialize 2-body operator Sigma ...
    Sigma = O2b_initialize(Params,Orb;Make_Orb_NN=false)

    println("\nCalculating 2nd & 3rd order HF-MBPT 2-body correlation function matrix Sigma ...")

    # Evaluate 2nd order particle-hole 2-body correlation function ...
        println("\t\tCalculating 2nd order particle-hole diagram ...")
    @time @inbounds for JP in JP_list;
        J, P = JP[1], JP[2]

        BraKet = HF_MBPT_sigma_orbitals_JP(J,P,Orb_NN)

        N_T0, N_T1 = Orb_NN.N[1,P,J+1], Orb_NN.N[2,P,J+1]

        Orb_NN_JP_T0 = Orb_NN.Ind[1,P,J+1]
        Orb_NN_JP_T1 = Orb_NN.Ind[2,P,J+1]

        @inbounds Threads.@threads for (Bra,Ket,T) in BraKet
            if T == -1
                Ind = Bra + (Ket - 1) * N_T1 - div(Ket * (Ket - 1),2)

                a, b = Orb_NN_JP_T1[Bra][1], Orb_NN_JP_T1[Bra][2]
                c, d = Orb_NN_JP_T1[Ket][1], Orb_NN_JP_T1[Ket][2]

                t_a, t_b, t_c, t_d = -1, -1, -1, -1

                O_a, O_b = Orb[a].pO, Orb[b].pO
                O_c, O_d = Orb[c].pO, Orb[d].pO

                ppSigmaSum = 0.0

                if particle(O_a) && particle(O_b) && hole(O_c) && hole(O_d)
                    ME = HF_MBPT2_sigma(a,t_a,b,t_b,c,t_c,d,t_d,J,P)
                    ppSigmaSum += ME

                elseif hole(O_a) && hole(O_b) && particle(O_c) && particle(O_d)
                    ME = HF_MBPT2_sigma(c,t_c,d,t_d,a,t_a,b,t_b,J,P)
                    ppSigmaSum += ME
                end

                @inbounds Sigma.pp[P,J+1][Ind] += ppSigmaSum

            elseif T == 0
                Ind = Bra + (Ket - 1) * N_T0 - div(Ket * (Ket - 1),2)

                a, b = Orb_NN_JP_T0[Bra][1], Orb_NN_JP_T0[Bra][2]
                c, d = Orb_NN_JP_T0[Ket][1], Orb_NN_JP_T0[Ket][2]

                t_a, t_b, t_c, t_d = -1, 1, -1, 1

                O_a, O_b = Orb[a].pO, Orb[b].nO
                O_c, O_d = Orb[c].pO, Orb[d].nO

                pnSigmaSum = 0.0

                if particle(O_a) && particle(O_b) && hole(O_c) && hole(O_d)
                    ME = HF_MBPT2_sigma(a,t_a,b,t_b,c,t_c,d,t_d,J,P)
                    pnSigmaSum += ME

                elseif hole(O_a) && hole(O_b) && particle(O_c) && particle(O_d)
                    ME = HF_MBPT2_sigma(c,t_c,d,t_d,a,t_a,b,t_b,J,P)
                    pnSigmaSum += ME
                end

                @inbounds Sigma.pn[P,J+1][Ind] += pnSigmaSum

            elseif T == 1
                Ind = Bra + (Ket - 1) * N_T1 - div(Ket * (Ket - 1),2)

                a, b = Orb_NN_JP_T1[Bra][1], Orb_NN_JP_T1[Bra][2]
                c, d = Orb_NN_JP_T1[Ket][1], Orb_NN_JP_T1[Ket][2]

                t_a, t_b, t_c, t_d = 1, 1, 1, 1

                O_a, O_b = Orb[a].pO, Orb[b].pO
                O_c, O_d = Orb[c].pO, Orb[d].pO

                nnSigmaSum = 0.0

                if particle(O_a) && particle(O_b) && hole(O_c) && hole(O_d)
                    ME = HF_MBPT2_sigma(a,t_a,b,t_b,c,t_c,d,t_d,J,P)
                    nnSigmaSum += ME

                elseif hole(O_a) && hole(O_b) && particle(O_c) && particle(O_d)
                    ME = HF_MBPT2_sigma(c,t_c,d,t_d,a,t_a,b,t_b,J,P)
                    nnSigmaSum += ME
                end

                @inbounds Sigma.nn[P,J+1][Ind] += nnSigmaSum
            end
        end
    end

    # Evaluate 3rd order particle-particle 2-body correlation function ...
        println("\t\tCalculating 3rd order particle-particle diagram ...")
    @time @inbounds for JP in JP_list;
        J, P = JP[1], JP[2]

        BraKet = HF_MBPT_sigma_orbitals_JP(J,P,Orb_NN)

        N_T0, N_T1 = Orb_NN.N[1,P,J+1], Orb_NN.N[2,P,J+1]

        Orb_NN_JP_T0 = Orb_NN.Ind[1,P,J+1]
        Orb_NN_JP_T1 = Orb_NN.Ind[2,P,J+1]

        @inbounds Threads.@threads for (Bra,Ket,T) in BraKet
            if T == -1
                Ind = Bra + (Ket - 1) * N_T1 - div(Ket * (Ket - 1),2)

                a, b = Orb_NN_JP_T1[Bra][1], Orb_NN_JP_T1[Bra][2]
                c, d = Orb_NN_JP_T1[Ket][1], Orb_NN_JP_T1[Ket][2]

                t_a, t_b, t_c, t_d = -1, -1, -1, -1

                O_a, O_b = Orb[a].pO, Orb[b].pO
                O_c, O_d = Orb[c].pO, Orb[d].pO

                ppSigmaSum = 0.0

                if particle(O_a) && particle(O_b) && hole(O_c) && hole(O_d)
                    ME = HF_MBPT3_sigma_pp(a,t_a,b,t_b,c,t_c,d,t_d,J,P)
                    ppSigmaSum += ME

                elseif hole(O_a) && hole(O_b) && particle(O_c) && particle(O_d)
                    ME = HF_MBPT3_sigma_pp(c,t_c,d,t_d,a,t_a,b,t_b,J,P)
                    ppSigmaSum += ME
                end

                @inbounds Sigma.pp[P,J+1][Ind] += ppSigmaSum

            elseif T == 0
                Ind = Bra + (Ket - 1) * N_T0 - div(Ket * (Ket - 1),2)

                a, b = Orb_NN_JP_T0[Bra][1], Orb_NN_JP_T0[Bra][2]
                c, d = Orb_NN_JP_T0[Ket][1], Orb_NN_JP_T0[Ket][2]

                t_a, t_b, t_c, t_d = -1, 1, -1, 1

                O_a, O_b = Orb[a].pO, Orb[b].nO
                O_c, O_d = Orb[c].pO, Orb[d].nO

                pnSigmaSum = 0.0

                if particle(O_a) && particle(O_b) && hole(O_c) && hole(O_d)
                    ME = HF_MBPT3_sigma_pp(a,t_a,b,t_b,c,t_c,d,t_d,J,P)
                    pnSigmaSum += ME

                elseif hole(O_a) && hole(O_b) && particle(O_c) && particle(O_d)
                    ME = HF_MBPT3_sigma_pp(c,t_c,d,t_d,a,t_a,b,t_b,J,P)
                    pnSigmaSum += ME
                end

                @inbounds Sigma.pn[P,J+1][Ind] += pnSigmaSum

            elseif T == 1
                Ind = Bra + (Ket - 1) * N_T1 - div(Ket * (Ket - 1),2)

                a, b = Orb_NN_JP_T1[Bra][1], Orb_NN_JP_T1[Bra][2]
                c, d = Orb_NN_JP_T1[Ket][1], Orb_NN_JP_T1[Ket][2]

                t_a, t_b, t_c, t_d = 1, 1, 1, 1

                O_a, O_b = Orb[a].pO, Orb[b].pO
                O_c, O_d = Orb[c].pO, Orb[d].pO

                nnSigmaSum = 0.0

                if particle(O_a) && particle(O_b) && hole(O_c) && hole(O_d)
                    ME = HF_MBPT3_sigma_pp(a,t_a,b,t_b,c,t_c,d,t_d,J,P)
                    nnSigmaSum += ME

                elseif hole(O_a) && hole(O_b) && particle(O_c) && particle(O_d)
                    ME = HF_MBPT3_sigma_pp(c,t_c,d,t_d,a,t_a,b,t_b,J,P)
                    nnSigmaSum += ME
                end

                @inbounds Sigma.nn[P,J+1][Ind] += nnSigmaSum
            end
        end
    end

    # Evaluate 3rd order hole-hole 2-body correlation function ...
        println("\t\tCalculating 3rd order hole-hole diagram ...")
    @time @inbounds for JP in JP_list;
        J, P = JP[1], JP[2]

        BraKet = HF_MBPT_sigma_orbitals_JP(J,P,Orb_NN)

        N_T0, N_T1 = Orb_NN.N[1,P,J+1], Orb_NN.N[2,P,J+1]

        Orb_NN_JP_T0 = Orb_NN.Ind[1,P,J+1]
        Orb_NN_JP_T1 = Orb_NN.Ind[2,P,J+1]

        @inbounds Threads.@threads for (Bra,Ket,T) in BraKet
            if T == -1
                Ind = Bra + (Ket - 1) * N_T1 - div(Ket * (Ket - 1),2)

                a, b = Orb_NN_JP_T1[Bra][1], Orb_NN_JP_T1[Bra][2]
                c, d = Orb_NN_JP_T1[Ket][1], Orb_NN_JP_T1[Ket][2]

                t_a, t_b, t_c, t_d = -1, -1, -1, -1

                O_a, O_b = Orb[a].pO, Orb[b].pO
                O_c, O_d = Orb[c].pO, Orb[d].pO

                ppSigmaSum = 0.0

                if particle(O_a) && particle(O_b) && hole(O_c) && hole(O_d)
                    ME = HF_MBPT3_sigma_hh(a,t_a,b,t_b,c,t_c,d,t_d,J,P)
                    ppSigmaSum += ME

                elseif hole(O_a) && hole(O_b) && particle(O_c) && particle(O_d)
                    ME = HF_MBPT3_sigma_hh(c,t_c,d,t_d,a,t_a,b,t_b,J,P)
                    ppSigmaSum += ME
                end

                @inbounds Sigma.pp[P,J+1][Ind] += ppSigmaSum

            elseif T == 0
                Ind = Bra + (Ket - 1) * N_T0 - div(Ket * (Ket - 1),2)

                a, b = Orb_NN_JP_T0[Bra][1], Orb_NN_JP_T0[Bra][2]
                c, d = Orb_NN_JP_T0[Ket][1], Orb_NN_JP_T0[Ket][2]

                t_a, t_b, t_c, t_d = -1, 1, -1, 1

                O_a, O_b = Orb[a].pO, Orb[b].nO
                O_c, O_d = Orb[c].pO, Orb[d].nO

                pnSigmaSum = 0.0

                if particle(O_a) && particle(O_b) && hole(O_c) && hole(O_d)
                    ME = HF_MBPT3_sigma_hh(a,t_a,b,t_b,c,t_c,d,t_d,J,P)
                    pnSigmaSum += ME

                elseif hole(O_a) && hole(O_b) && particle(O_c) && particle(O_d)
                    ME = HF_MBPT3_sigma_hh(c,t_c,d,t_d,a,t_a,b,t_b,J,P)
                    pnSigmaSum += ME
                end

                @inbounds Sigma.pn[P,J+1][Ind] += pnSigmaSum

            elseif T == 1
                Ind = Bra + (Ket - 1) * N_T1 - div(Ket * (Ket - 1),2)

                a, b = Orb_NN_JP_T1[Bra][1], Orb_NN_JP_T1[Bra][2]
                c, d = Orb_NN_JP_T1[Ket][1], Orb_NN_JP_T1[Ket][2]

                t_a, t_b, t_c, t_d = 1, 1, 1, 1

                O_a, O_b = Orb[a].pO, Orb[b].pO
                O_c, O_d = Orb[c].pO, Orb[d].pO

                nnSigmaSum = 0.0

                if particle(O_a) && particle(O_b) && hole(O_c) && hole(O_d)
                    ME = HF_MBPT3_sigma_hh(a,t_a,b,t_b,c,t_c,d,t_d,J,P)
                    nnSigmaSum += ME

                elseif hole(O_a) && hole(O_b) && particle(O_c) && particle(O_d)
                    ME = HF_MBPT3_sigma_hh(c,t_c,d,t_d,a,t_a,b,t_b,J,P)
                    nnSigmaSum += ME
                end

                @inbounds Sigma.nn[P,J+1][Ind] += nnSigmaSum
            end
        end
    end

    # Evaluate 3rd order particle-hole 2-body correlation function ...
        println("\t\tCalculating 3rd order particle-hole diagram ...")
    @time @inbounds for JP in JP_list;
        J, P = JP[1], JP[2]

        BraKet = HF_MBPT_sigma_orbitals_JP(J,P,Orb_NN)

        N_T0, N_T1 = Orb_NN.N[1,P,J+1], Orb_NN.N[2,P,J+1]

        Orb_NN_JP_T0 = Orb_NN.Ind[1,P,J+1]
        Orb_NN_JP_T1 = Orb_NN.Ind[2,P,J+1]

        @inbounds Threads.@threads for (Bra,Ket,T) in BraKet
            SigmaSum = 0.0

            if T == -1
                Ind = Bra + (Ket - 1) * N_T1 - div(Ket * (Ket - 1),2)

                a, b = Orb_NN_JP_T1[Bra][1], Orb_NN_JP_T1[Bra][2]
                c, d = Orb_NN_JP_T1[Ket][1], Orb_NN_JP_T1[Ket][2]

                O_a, O_b, O_c, O_d = Orb[a].pO, Orb[b].pO, Orb[c].pO, Orb[d].pO

                t_a, t_b, t_c, t_d = -1, -1, -1, -1
            end

            if T == 0
                Ind = Bra + (Ket - 1) * N_T0 - div(Ket * (Ket - 1),2)

                a, b = Orb_NN_JP_T0[Bra][1], Orb_NN_JP_T0[Bra][2]
                c, d = Orb_NN_JP_T0[Ket][1], Orb_NN_JP_T0[Ket][2]

                O_a, O_b, O_c, O_d = Orb[a].pO, Orb[b].nO, Orb[c].pO, Orb[d].nO

                t_a, t_b, t_c, t_d = -1, 1, -1, 1
            end

            if T == 1
                Ind = Bra + (Ket - 1) * N_T1 - div(Ket * (Ket - 1),2)

                a, b = Orb_NN_JP_T1[Bra][1], Orb_NN_JP_T1[Bra][2]
                c, d = Orb_NN_JP_T1[Ket][1], Orb_NN_JP_T1[Ket][2]

                O_a, O_b, O_c, O_d = Orb[a].nO, Orb[b].nO, Orb[c].nO, Orb[d].nO

                t_a, t_b, t_c, t_d = 1, 1, 1, 1
            end

            if particle(O_a) && particle(O_b) && hole(O_c) && hole(O_d)
                SigmaSum = HF_MBPT3_sigma_ph(a,t_a,b,t_b,c,t_c,d,t_d,J)

            elseif hole(O_a) && hole(O_b) && particle(O_c) && particle(O_d)
                SigmaSum = HF_MBPT3_sigma_ph(c,t_c,d,t_d,a,t_a,b,t_b,J)
            end

            if T == -1
                @inbounds Sigma.pp[P,J+1][Ind] += SigmaSum

            elseif T == 0
                @inbounds Sigma.pn[P,J+1][Ind] += SigmaSum

            elseif T == 1
                @inbounds Sigma.nn[P,J+1][Ind] += SigmaSum
            end
        end
    end

    # Perform transformation of the 2-body correlation function into the target basis ...
    @time Sigma_NOB = O2b_transformation(Params,Orb,Orb_NN,Sigma,C)

    # Deallocate & perform garbage collection ...
    Sigma = nothing
    GC.gc()

    println("\n\tFinished allocation & basis transformation of the HF-MBPT 2-body correlation function ...")

    return Sigma_NOB
end

function HF_MBPT_occupation(Params::Parameters,Rho_HF::O1B,Rho_NAT::O1B)
    # Read parameters ...
    IO_path = "io/" * Params.Calc.Path
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1)*(N_max + 2),2)

    # Set the export path ...
    export_path = IO_path * "/HF/densities/HF_MBPT_occupation.dat"

    # Export the occupation probabilities to data file ...
    open(export_path, "w") do write_file
        @printf(write_file, "%-6s %-12s %-12s %-12s %-12s %-12s %-12s\n", "a", "pN_HF", "pN_PT", "d_pN", "nN_HF", "nN_PT", "d_nN")
        @inbounds for a in 1:a_max
            @printf(write_file, "%-6d %-12.6f %-12.6f %-12.6f %-12.6f %-12.6f %-12.6f\n",
                    a, Rho_HF.p[a,a], Rho_NAT.p[a,a], Rho_HF.p[a,a] - Rho_NAT.p[a,a], Rho_HF.n[a,a],
                    Rho_NAT.n[a,a], Rho_HF.n[a,a] - Rho_NAT.n[a,a])
        end
    end

    return
end

function HF_MBPT_export(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,Orb_NNN::Orb3B,C::O1B,T::O1B,h_N::O1B,V_NN::O2B,V_NN_res::O2B,V_NNN::Array{Vector{Vector{Float32}},4})
    # Read parameters ...
    IO_path = "io/" * Params.Calc.Path

    # Make Particle-Hole orbitals ...
    Orb_HF_MBPT = HF_MBPT_orbitals_make(Params,Orb)

    println("\nPreparing export of perturbatively correlated HF-MBPT ground state ...")

    # Set up a new directory for HF-MBPT solutions ...
    if isdir("io/" * Params.Calc.Path * "/HF-MBPT")
        rm("io/" * Params.Calc.Path * "/HF-MBPT", recursive = true)
    end
    mkdir("io/" * Params.Calc.Path * "/HF-MBPT")
    mkdir("io/" * Params.Calc.Path * "/HF-MBPT/bin")

    println("\n\tPreparing HF-MBPT natural orbitals & OBDM ...")

    # Import precomputed natural orbitals basis from temporary files ...
    Orb_NAT = orbitals_import(Params,IO_path * "/HF/bin/orb_temp.bin")
    C_NAT = O1b_import(Params,Orb,IO_path * "/HF/bin/C_temp.bin")
    Rho_NAT = O1b_import(Params,Orb,IO_path * "/HF/bin/rho_temp.bin")

    # Transform Rho to the LHO basis ...
    Rho_LHO = O1B(C_NAT.p * Rho_NAT.p * C_NAT.p', C_NAT.n * Rho_NAT.n * C_NAT.n')

    println("\n\t\tHF-MBPT natural orbitals & OBDM finished ...")

    # If set, evaluate 2-body correlation function & export ...
    if Params.Calc.HF.MBPT.TBCF == true
        println("\n\tCalculating HF-MBPT 2-body correlation function ...")
        # Calculate 2-body correlation function ... in HF basis and transform into LHO basis ...
        Sigma_NN_LHO = HF_MBPT_TBCF(Params,C,h_N,V_NN_res,Orb,Orb_HF_MBPT,Orb_NN)

        # Perform transformation of the 2-body correlation function into the target basis ...
        @time Sigma_NN_NAT = O2b_transformation(Params,Orb,Orb_NN,Sigma_NN_LHO,C_NAT)

        println("\n\t\tHF-MBPT 2-body correlation function constructed ...")
    end

    println("\n\tCalculating HF-MBPT 1-body mean-field Hamiltonain ...")

    # If set, include mean-field polarisation ... new 1-body mean-field ...
    if Params.Calc.HF.MBPT.MFP == true
        if Params.Calc.HF.MBPT.TBCF == true
            h_N_LHO = H1b(Params,Orb,Orb_NN,Orb_NNN,T,V_NN,V_NNN,Rho_LHO,Sigma_NN_LHO)
        else
            h_N_LHO = H1b(Params,Orb,Orb_NN,Orb_NNN,T,V_NN,V_NNN,Rho_LHO)
        end
    else
        h_N_LHO = O1B(C.p * h_N.p * C.p', C.n * h_N.n * C.n')
    end

    # Transform the 1-body mean-field Hamiltonian to the NAT basis ...
    h_N_NAT = O1B(C_NAT.p' * h_N_LHO.p * C_NAT.p, C_NAT.n' * h_N_LHO.n * C_NAT.n)

    println("\n\t\tFinished on HF-MBPT 1-body mean-field Hamiltonain ...")

    println("\n\tConstructing HF-MBPT residual 2-body interaction ...")

    # Construct new residual 2-body interaction ...
    @time V_NN_temp = V2b_residual_no2b(Params,Orb,Orb_NN,Orb_NNN,Rho_LHO,V_NN,V_NNN)

    # Transform the density-dependent NN interaction to the target NAT basis ...
    @time V_NN_NAT = O2b_transformation(Params,Orb,Orb_NN,V_NN_temp,C_NAT)

    println("\n\t\tHF-MBPT residual 2-body interaction finished ...")

    println("\n\tPerforming export of HF-MBPT perturbatively correlated ground state ...")

    # Deallocate & perform garbage collection ...
    h_N_LHO = nothing
    Rho_LHO = nothing
    Sigma_NN_LHO = nothing
    V_NN_temp = nothing
    GC.gc()

    # Remove temporary solutions ...
    rm(IO_path * "/HF/bin/orb_temp.bin")
    rm(IO_path * "/HF/bin/C_temp.bin")
    rm(IO_path * "/HF/bin/rho_temp.bin")

    # Export HF-MBPT solutions ...
    orbitals_export(Params,Orb_NAT,IO_path * "/HF-MBPT/bin/orb.bin")
    O1b_export(Params,Orb_NAT,C_NAT,IO_path * "/HF-MBPT/bin/C.bin")
    O1b_export(Params,Orb_NAT,h_N_NAT,IO_path * "/HF-MBPT/bin/H1b.bin")
    O1b_export(Params,Orb_NAT,Rho_NAT,IO_path * "/HF-MBPT/bin/rho.bin")
    O2b_export(Params,Orb_NN,V_NN_NAT,IO_path * "/HF-MBPT/bin/H2b.bin")
    if Params.Calc.HF.MBPT.TBCF == true
        O2b_export(Params,Orb_NN,Sigma_NN_NAT,IO_path * "/HF-MBPT/bin/sigma.bin")
    end

    # Deallocate & perform garbage collection ...
    C_NAT = nothing
    h_N_NAT = nothing
    Rho_NAT = nothing
    Sigma_NN_NAT = nothing
    V_NN_NAT = nothing
    GC.gc()

    println("\n\t\tFinished performing export & garbage collection for HF-MBPT perturbatively correlated ground state ...")

    return
end