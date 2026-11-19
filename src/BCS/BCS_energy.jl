function BCS_energy(Params::Parameters,Rho::O1B,Kappa::O1B,Orb::Vector{Orb1B},Orb_NN::Orb2B,Orb_NNN::Orb3B,T::O1B,V_NN::O2B,V_NNN::Array{Vector{Vector{Float32}},4})
    # Read parameters ...
    A = Params.Calc.A
    N_max, N_2max, N_3max = Params.Calc.Nmax, Params.Calc.N2max, Params.Calc.N3max
    a_max = div((N_max + 1)*(N_max + 2),2)
    CMS = Params.Calc.CMS
    p2, p3 = Params.Int.cP2N, Params.Int.cP3N

    # Calculate the total mean-field + BCS ground-state energy ...
    println("\nCalculating the total mean-field + BCS pairing ground-state energy ...")

    E_MF, E_BCS = 0.0, [0.0, 0.0]
    E_MF_threads = zeros(Float64,Threads.maxthreadid())
    E_BCS_2N_threads = zeros(Float64,Threads.maxthreadid())
    E_BCS_3N_threads = zeros(Float64,Threads.maxthreadid())

    # Single-particle HF mean-field energy ...
    @inbounds Threads.@threads :static for a in 1:a_max
        Sum, Tid = 0.0, Threads.threadid()
        n_a = Orb[a].n
        l_a = Orb[a].l
        j_a = Orb[a].j
        @inbounds for b = 1:a_max
            n_b = Orb[b].n
            l_b = Orb[b].l
            if (2*(n_a + n_b) + l_a + l_b) <= N_2max
                j_b = Orb[b].j
                P_ab = rem(l_a + l_b,2) + 1
                @inbounds for d = 1:a_max
                    l_d = Orb[d].l
                    j_d = Orb[d].j
                    if (l_a == l_d) && (j_a == j_d)
                        n_d = Orb[d].n
                        pRho_ad = Rho.p[a,d]
                        nRho_ad = Rho.n[a,d]
                        @inbounds for e = 1:a_max
                            n_e = Orb[e].n
                            l_e = Orb[e].l
                            j_e = Orb[e].j
                            if (l_b == l_e && j_b == j_e) && ((2*(n_d + n_e) + l_d + l_e) <= N_2max)
                                pRho_be = Rho.p[b,e]
                                nRho_be = Rho.n[b,e]
                                @inbounds for J = div(abs(j_a - j_b),2):div((j_a + j_b),2)

                                    # 2-body NN interaction ...
                                    if (rem(l_a + l_b, 2) == rem(l_d + l_e, 2))
                                        Hat =  Float64(2*J + 1)
                                        Sum += 0.5 * Hat * pRho_ad * pRho_be * O2b_pp(a,b,d,e,J,P_ab,V_NN,Orb,Orb_NN)
                                        Sum += 0.5 * Hat * nRho_ad * nRho_be * O2b_nn(a,b,d,e,J,P_ab,V_NN,Orb,Orb_NN)
                                        Sum += Hat * pRho_ad * nRho_be * O2b_pn(a,b,d,e,J,P_ab,V_NN,Orb_NN)
                                    end
                                    
                                    # 3-body NNN interaction ...
                                    @inbounds for c = 1:a_max
                                        n_c = Orb[c].n
                                        l_c = Orb[c].l
                                        if ((2*(n_a + n_b + n_c) + l_a + l_b + l_c) <= N_3max)
                                            j_c = Orb[c].j
                                            P_abc = rem(l_a + l_b + l_c, 2) + 1
                                            @inbounds for f = 1:a_max
                                                n_f = Orb[f].n
                                                l_f = Orb[f].l
                                                j_f = Orb[f].j
                                                if (2*(n_d + n_e + n_f) + l_d + l_e + l_f) <= N_3max && l_c == l_f && j_c == j_f && P_abc == (rem(l_d + l_e + l_f, 2)+1)
                                                    pRho_cf, nRho_cf = Rho.p[c,f], Rho.n[c,f]

                                                    me1 = V3b_no2b(a,b,c,0,d,e,f,0,J,1,P_abc,V_NNN,Orb,Orb_NNN)
                                                    me2 = V3b_no2b(a,b,c,1,d,e,f,0,J,1,P_abc,V_NNN,Orb,Orb_NNN)
                                                    me3 = V3b_no2b(a,b,c,0,d,e,f,1,J,1,P_abc,V_NNN,Orb,Orb_NNN)
                                                    me4 = V3b_no2b(a,b,c,1,d,e,f,1,J,1,P_abc,V_NNN,Orb,Orb_NNN)
                                                    me5 = V3b_no2b(a,b,c,1,d,e,f,1,J,3,P_abc,V_NNN,Orb,Orb_NNN)

                                                    Sum += 1.0/6.0 * (me5 * pRho_ad * pRho_be * pRho_cf + (2.0 * me4 + me5) *
                                                    pRho_ad * pRho_be * nRho_cf + (1.5 * me1 + sqrt(3.0/4.0) * me2 +
                                                    sqrt(3.0/4.0) * me3 + 0.5 * me4 + me5)  * pRho_ad * nRho_be * nRho_cf +
                                                    me5 * nRho_ad * nRho_be * nRho_cf)
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
            if CMS == "CMS1+2B"
                @views Sum += (Rho.p[a,b] + Rho.n[a,b]) * T.p[a,b] * Float64(j_a + 1) * (1.0 - 1.0 / Float64(A))
            elseif CMS == "CMS2B"
                @views Sum += 0
            else
                @views Sum += (Rho.p[a,b] + Rho.n[a,b]) * T.p[a,b] * Float64(j_a + 1)
            end
        end
        E_MF_threads[Tid] += Sum
    end

    # Pairing BCS field energy ...
    @inbounds Threads.@threads :static for a = 1:a_max
        Sum2N, Sum3N, Tid = 0.0, 0.0, Threads.threadid()
        n_a = Orb[a].n
        l_a = Orb[a].l
        j_a = Orb[a].j
        j_a_hat = sqrt(Float64(Orb[a].j + 1)) 
        @inbounds for b = 1:a_max
            n_b = Orb[b].n
            l_b = Orb[b].l
            if (2*(n_a + n_b) + l_a + l_b) <= N_2max && l_a == l_b
                j_b = Orb[b].j
                if j_a == j_b
                    pKappa_ab, nKappa_ab = Kappa.p[a,b], Kappa.n[a,b]
                    @inbounds for d = 1:a_max
                        n_d = Orb[d].n
                        l_d = Orb[d].l
                        j_d = Orb[d].j
                        j_d_hat = sqrt(Float64(Orb[d].j + 1)) 
                        @inbounds for e = 1:a_max
                            n_e = Orb[e].n
                            l_e = Orb[e].l
                            if ((2*(n_d + n_e) + l_d + l_e) <= N_2max) && l_d == l_e
                                j_e = Orb[e].j
                                if j_d == j_e
                                    pKappa_de, nKappa_de = Kappa.p[d,e], Kappa.n[d,e]

                                    # 2-body NN interaction ...
                                    Sum2N += p2 * 0.25 * j_a_hat * j_d_hat * pKappa_ab * pKappa_de * O2b_pp(a,b,d,e,0,1,V_NN,Orb,Orb_NN)
                                    Sum2N += p2 * 0.25 * j_a_hat * j_d_hat * nKappa_ab * nKappa_de * O2b_nn(a,b,d,e,0,1,V_NN,Orb,Orb_NN)
                                
                                    # 3-body NNN interaction ...
                                    @inbounds for c = 1:a_max
                                        n_c = Orb[c].n
                                        l_c = Orb[c].l
                                        if ((2*(n_a + n_b + n_c) + l_a + l_b + l_c) <= N_3max)
                                            j_c = Orb[c].j
                                            P_abc = rem(l_a + l_b + l_c, 2) + 1
                                            @inbounds for f = 1:a_max
                                                n_f = Orb[f].n
                                                l_f = Orb[f].l
                                                j_f = Orb[f].j
                                                if (2*(n_d + n_e + n_f) + l_d + l_e + l_f) <= N_3max && l_c == l_f && j_c == j_f && P_abc == (rem(l_d + l_e + l_f, 2)+1)
                                                    pRho_cf, nRho_cf = Rho.p[c,f], Rho.n[c,f]

                                                    ME111 = V3b_no2b(a,b,c,1,d,e,f,1,0,1,P_abc,V_NNN,Orb,Orb_NNN)
                                                    ME113 = V3b_no2b(a,b,c,1,d,e,f,1,0,3,P_abc,V_NNN,Orb,Orb_NNN)

                                                    Sum3N += p3 * 0.25 * j_a_hat * j_d_hat * (pKappa_ab * pKappa_de *
                                                                                            pRho_cf * ME113 + pKappa_ab * pKappa_de * nRho_cf * 1.0 / 3.0 *
                                                                                            (2.0 * ME111 + ME113) + nKappa_ab * nKappa_de * nRho_cf * ME113 +
                                                                                            nKappa_ab * nKappa_de * pRho_cf * 1.0 / 3.0 *(2.0 * ME111 + ME113))
                                                end
                                            end
                                        end
                                    end

                                end
                            end
                        end
                    end

                end
            end
        end
        E_BCS_2N_threads[Tid] += Sum2N
        E_BCS_3N_threads[Tid] += Sum3N
    end

    # Sum the total energy contributions from threads ...
    E_MF, E_BCS = sum(E_MF_threads), [sum(E_BCS_2N_threads), sum(E_BCS_3N_threads)] 

    println("\tMF energy      ...   E_MF  = " * string(E_MF) * " MeV")
    println("\tBCS energy     ...   E_BCS = " * string(sum(E_BCS)) * " MeV")
    println("\tBCS 2N energy ...   E_BCS_2N = " * string(E_BCS[1]) * " MeV")
    println("\tBCS 3N energy ...   E_BCS_3N = " * string(E_BCS[2]) * " MeV")

    return E_MF, E_BCS
end