function HFB_energy(Params::Parameters,Rho::O1B,Kappa::O1B,Orb::Vector{Orb1B},Orb_NN::Orb2B,Orb_NNN::Orb3B,T::O1B,V_NN::O2B,V_NNN::Array{Vector{Vector{Float32}},4})
    # Read parameters ...
    A = Params.Calc.A
    N_max, N_2max, N_3max = Params.Calc.Nmax, Params.Calc.N2max, Params.Calc.N3max
    a_max = div((N_max + 1)*(N_max + 2),2)
    p2, p3 = Params.Int.cP2N, Params.Int.cP3N
    CMS = Params.Calc.CMS

    #Calculate the HF energy ...
    println("\nCalculating total HFB ground-state energy ...")

    E_HFB = [0.0, 0.0, 0.0, 0.0]

    E_h_threads = zeros(Float64,Threads.maxthreadid())
    E_p_2N_threads = zeros(Float64,Threads.maxthreadid())
    E_p_3N_threads = zeros(Float64,Threads.maxthreadid())

    # Single-particle mean-field energy ...
    @inbounds Threads.@threads :static for a = 1:a_max
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

                                                    ME001 = V3b_no2b(a,b,c,0,d,e,f,0,J,1,P_abc,V_NNN,Orb,Orb_NNN)
                                                    ME101 = V3b_no2b(a,b,c,1,d,e,f,0,J,1,P_abc,V_NNN,Orb,Orb_NNN)
                                                    ME011 = V3b_no2b(a,b,c,0,d,e,f,1,J,1,P_abc,V_NNN,Orb,Orb_NNN)
                                                    ME111 = V3b_no2b(a,b,c,1,d,e,f,1,J,1,P_abc,V_NNN,Orb,Orb_NNN)
                                                    ME113 = V3b_no2b(a,b,c,1,d,e,f,1,J,3,P_abc,V_NNN,Orb,Orb_NNN)

                                                    Sum += 1.0/6.0 * (ME113 * pRho_ad * pRho_be * pRho_cf + (2.0 * ME111 + ME113) *
                                                    pRho_ad * pRho_be * nRho_cf + (1.5 * ME001 + sqrt(3.0/4.0) * ME101 +
                                                    sqrt(3.0/4.0) * ME011 + 0.5 * ME111 + ME113)  * pRho_ad * nRho_be * nRho_cf +
                                                    ME113 * nRho_ad * nRho_be * nRho_cf)
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
        E_h_threads[Tid] += Sum
    end

    # Pairing field energy ...
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
                    pKappa_ba, nKappa_ba = Kappa.p[b,a], Kappa.n[b,a]
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
                                    Sum2N += p2 * 0.25 * j_a_hat * j_d_hat * pKappa_ba * pKappa_de * O2b_pp(a,b,d,e,0,1,V_NN,Orb,Orb_NN)
                                    Sum2N += p2 * 0.25 * j_a_hat * j_d_hat * nKappa_ba * nKappa_de * O2b_nn(a,b,d,e,0,1,V_NN,Orb,Orb_NN)
                                
                                    # 3-body NNN interaction ...
                                    @inbounds for c = 1:a_max
                                        n_c = Orb[c].n
                                        l_c = Orb[c].l
                                        if ((2*(n_a + n_b + n_c) + l_a + l_b + l_c) <= N_3max)
                                            j_c = Orb[c].j
                                            #j_c_hat = Float64(Orb[c].j + 1)
                                            P_abc = rem(l_a + l_b + l_c, 2) + 1
                                            @inbounds for f = 1:a_max
                                                n_f = Orb[f].n
                                                l_f = Orb[f].l
                                                j_f = Orb[f].j
                                                if (2*(n_d + n_e + n_f) + l_d + l_e + l_f) <= N_3max && l_c == l_f && j_c == j_f && P_abc == (rem(l_d + l_e + l_f, 2)+1)
                                                    pRho_cf, nRho_cf = Rho.p[c,f], Rho.n[c,f]

                                                    ME111 = V3b_no2b(a,b,c,1,d,e,f,1,0,1,P_abc,V_NNN,Orb,Orb_NNN)
                                                    ME113 = V3b_no2b(a,b,c,1,d,e,f,1,0,3,P_abc,V_NNN,Orb,Orb_NNN)

                                                    Sum3N += p3 * 0.25 * j_a_hat * j_d_hat * (pKappa_ba * pKappa_de *
                                                                                            pRho_cf * ME113 + pKappa_ba * pKappa_de * nRho_cf * 1.0 / 3.0 *
                                                                                            (2.0 * ME111 + ME113) + nKappa_ba * nKappa_de * nRho_cf * ME113 +
                                                                                            nKappa_ba * nKappa_de * pRho_cf * 1.0 / 3.0 *(2.0 * ME111 + ME113))
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
        E_p_2N_threads[Tid] += Sum2N
        E_p_3N_threads[Tid] += Sum3N
    end

    E_HFB[1] += sum(E_h_threads)
    E_HFB[2] += sum(E_p_2N_threads)
    E_HFB[3] += sum(E_p_3N_threads)

    # Lipkin-Nogami energy correction ... if enabled ...
    if Params.Calc.HFB.LNT == true
        # Import the LN coefficient Lambda_2 ...
        Lambda_2 = HFB_Lipkin_Nogami_Lambda_2_import(Params)

        # Calculate the LN energy corretion ...
        E_LN = HFB_Lipkin_Nogami_energy(Params,Orb,Lambda_2,Rho)

        E_HFB[4] = E_LN
    end

    println("\n\tHFB energy                 ...   E_HFB = " * string(sum(E_HFB)) * " MeV")
    println("\tMean-field energy          ...   E_MF  = " * string(E_HFB[1]) * " MeV")
    println("\tPairing energy             ...   E_Par = " * string(E_HFB[2] + E_HFB[3]) * " MeV")
    println("\tNN pairing energy        ...   E_Par2N = " * string(E_HFB[2]) * " MeV")
    println("\tNNN pairing energy       ...   E_Par3N = " * string(E_HFB[3]) * " MeV")
    if Params.Calc.HFB.LNT == true
        println("\tLipkin-Nogami energy     ...   E_LN  = " * string(E_HFB[4]) * " MeV")
    end

    return E_HFB
end