function HF_energy(Params::Parameters,Rho::O1B,Orb::Vector{Orb1B},Orb_NN::Orb2B,Orb_NNN::Orb3B,T::O1B,V_NN::O2B,V_NNN::Array{Vector{Vector{Float32}},4})
    # Read parameters ...
    A = Params.Calc.A
    N_max = Params.Calc.Nmax
    N_2max = Params.Calc.N2max
    N_3max = Params.Calc.N3max
    CMS = Params.Calc.CMS
    a_max = div((N_max + 1)*(N_max + 2),2)

    # Calculate the HF energy ...
    println("\nCalculating the total HF ground-state energy ...")

    E_HF, E_HF_threads = 0.0, zeros(Float64,Threads.maxthreadid())

    @inbounds Threads.@threads :static for a = 1:a_max
        Sum, Tid = 0.0, Threads.threadid()
        n_a = Orb[a].n
        l_a = Orb[a].l
        j_a = Orb[a].j
        @inbounds for b = 1:a_max
            n_b = Orb[b].n
            l_b = Orb[b].l
            P = rem(l_a + l_b, 2) + 1
            if (2*(n_a + n_b) + l_a + l_b) <= N_2max
                j_b = Orb[b].j
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

                                    # 2-body NN interaction
                                    if (rem(l_a + l_b, 2) == rem(l_d + l_e, 2))
                                        Hat =  Float64(2*J + 1)
                                        Sum += 0.5 * Hat * pRho_ad * pRho_be * O2b_pp(a,b,d,e,J,P,V_NN,Orb,Orb_NN)
                                        Sum += 0.5 * Hat * nRho_ad * nRho_be * O2b_nn(a,b,d,e,J,P,V_NN,Orb,Orb_NN)
                                        Sum += Hat * pRho_ad * nRho_be * O2b_pn(a,b,d,e,J,P,V_NN,Orb_NN)
                                    end
                                    
                                    # 3-body NNN interaction
                                    @inbounds for c = 1:a_max
                                        n_c = Orb[c].n
                                        l_c = Orb[c].l
                                        if ((2*(n_a + n_b + n_c) + l_a + l_b + l_c) <= N_3max)
                                            j_c = Orb[c].j
                                            P3B = rem(l_a + l_b + l_c, 2) + 1
                                            @inbounds for f = 1:a_max
                                                n_f = Orb[f].n
                                                l_f = Orb[f].l
                                                j_f = Orb[f].j
                                                if (2*(n_d + n_e + n_f) + l_d + l_e + l_f) <= N_3max && l_c == l_f && j_c == j_f && P3B == (rem(l_d + l_e + l_f, 2) + 1)
                                                    pRho_cf = Rho.p[c,f]
                                                    nRho_cf = Rho.n[c,f]

                                                    me1 = V3b_no2b(a,b,c,0,d,e,f,0,J,1,P3B,V_NNN,Orb,Orb_NNN)
                                                    me2 = V3b_no2b(a,b,c,1,d,e,f,0,J,1,P3B,V_NNN,Orb,Orb_NNN)
                                                    me3 = V3b_no2b(a,b,c,0,d,e,f,1,J,1,P3B,V_NNN,Orb,Orb_NNN)
                                                    me4 = V3b_no2b(a,b,c,1,d,e,f,1,J,1,P3B,V_NNN,Orb,Orb_NNN)
                                                    me5 = V3b_no2b(a,b,c,1,d,e,f,1,J,3,P3B,V_NNN,Orb,Orb_NNN)

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
                Sum += (Rho.p[a,b] + Rho.n[a,b]) * T.p[a,b] * Float64(j_a + 1) * (1.0 - 1.0 / Float64(A))
            elseif CMS == "CMS2B"
                Sum += 0
            else
                Sum += (Rho.p[a,b] + Rho.n[a,b]) * T.p[a,b] * Float64(j_a + 1)
            end
        end
        E_HF_threads[Tid] += Sum
    end

    # Sum-up the total HF energy ...
    E_HF = sum(E_HF_threads)

    println("\tHF energy    ...   E_HF = " * string(E_HF) * " MeV")

    return E_HF
end