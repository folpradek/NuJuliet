function HFB_allocate_indices(Params::Parameters,Orb::Vector{Orb1B})
    # Read # initialize parameters ...
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1)*(N_max + 2),2)
    ab_count, de_count = 0, 0

    # Count how many ab pairs are there ...
    @inbounds for a in 1:a_max
        l_a = Orb[a].l
        j_a = Orb[a].j
        @inbounds for b in 1:a
            l_b = Orb[b].l
            j_b = Orb[b].j
            if j_a == j_b && l_a == l_b
                ab_count += 1
            end
        end
    end

    # Count how many de pairs are there ...
    @inbounds for d in 1:a_max
        l_d = Orb[d].l
        j_d = Orb[d].j
        @inbounds for e in 1:a_max
            l_e = Orb[e].l
            j_e = Orb[e].j
            de_count += 1
        end
    end

    # Initialite the array ab ...
    ab, ab_count = zeros(Int64,2,ab_count), 0
    de, de_count = zeros(Int64,2,de_count), 0

    # Allocate the array ab ...
    @inbounds for a in 1:a_max
        l_a = Orb[a].l
        j_a = Orb[a].j
        @inbounds for b in 1:a
            l_b = Orb[b].l
            j_b = Orb[b].j
            if j_a == j_b && l_a == l_b
                ab_count += 1
                ab[1,ab_count] = a
                ab[2,ab_count] = b
            end
        end
    end

    # Allocate the array de ...
    @inbounds for d in 1:a_max
        l_d = Orb[d].l
        j_d = Orb[d].j
        @inbounds for e in 1:a_max
            l_e = Orb[e].l
            j_e = Orb[e].j
            de_count += 1
            de[1,de_count] = d
            de[2,de_count] = e
        end
    end

    return ab, ab_count, de, de_count
end

function HFB_allocate(Params::Parameters,Lambda::pnFloat,Rho::O1B,Kappa::O1B,Orb::Vector{Orb1B},Orb_NN::Orb2B,Orb_NNN::Orb3B,T::O1B,V_NN::O2B,V_NNN::Array{Vector{Vector{Float32}},4})
    # Read parameters ...
    A = Params.Calc.A
    N_max, N_2max, N_3max = Params.Calc.Nmax, Params.Calc.N2max, Params.Calc.N3max
    a_max = div((N_max + 1)*(N_max + 2),2)
    p2, p3 = Params.Int.cP2N, Params.Int.cP3N
    CMS = Params.Calc.CMS

    # Allocate new HF Hamiltonian matrices ...
    pH, nH = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)
    pDelta, nDelta = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)

    # Allocate indices for HFB iteration ...
    ad, ad_count, be, be_count = HFB_allocate_indices(Params,Orb)

    # Allocate the fields H & Delta ...
    @inbounds for ad_i in 1:ad_count
        a, d = ad[1,ad_i], ad[2,ad_i]
        n_a, l_a, j_a = Orb[a].n, Orb[a].l, Orb[a].j
        n_d, l_d, j_d = Orb[d].n, Orb[d].l, Orb[d].j

        j_a_hat = sqrt(Float64(j_a + 1))
        #s_j_a_hat = Float64(j_a + 1)
        is_j_a_hat = 1.0 / (Float64(j_a) + 1.0)

        # Single-particle field H ..
            # H field thread-local accumulators for each thread ...
        pH_local = zeros(Float64,Threads.maxthreadid())
        nH_local = zeros(Float64,Threads.maxthreadid())

        @inbounds Threads.@threads :static for be_i in 1:be_count
            b, e = be[1,be_i], be[2,be_i]
            n_b, l_b, j_b = Orb[b].n, Orb[b].l, Orb[b].j
            n_e, l_e, j_e = Orb[e].n, Orb[e].l, Orb[e].j

            P_ab = rem(l_a + l_b,2) + 1

            Tid = Threads.threadid()
            pHSum_local, nHSum_local = 0.0, 0.0

            # Normal Density-dependent part ...
            if (2*(n_a + n_b) + l_a + l_b) <= N_2max

                # Normal Density-dependent part ...
                if l_b == l_e && j_b == j_e && (2*(n_d + n_e) + l_d + l_e) <= N_2max
                    pRho_be, nRho_be = Rho.p[b,e], Rho.n[b,e]

                    @inbounds for J = div(abs(j_d - j_e),2):div(j_d + j_e,2)
                        
                        # 2-body NN interaction part ...
                        if rem(l_a + l_b, 2) == rem(l_d + l_e, 2)
                            J_j_a_hat = Float64(2*J + 1) * is_j_a_hat
        
                            pHSum_local += J_j_a_hat * O2b_pp(a,b,d,e,J,P_ab,V_NN,Orb,Orb_NN) * pRho_be
                            pHSum_local += J_j_a_hat * O2b_pn(a,b,d,e,J,P_ab,V_NN,Orb_NN) * nRho_be
                            nHSum_local += J_j_a_hat * O2b_nn(a,b,d,e,J,P_ab,V_NN,Orb,Orb_NN) * nRho_be
                            nHSum_local += J_j_a_hat * O2b_pn(b,a,e,d,J,P_ab,V_NN,Orb_NN) * pRho_be

                        end

                        # 3-body NNN interaction part ...
                        @inbounds for c in 1:a_max
                            n_c = Orb[c].n
                            l_c = Orb[c].l
                            if (2*(n_a + n_b + n_c) + l_a + l_b + l_c) <= N_3max
                                P_abc = rem(l_a + l_b + l_c, 2) + 1
                                j_c = Orb[c].j
                                @inbounds for f in 1:a_max
                                    n_f = Orb[f].n
                                    l_f = Orb[f].l
                                    if (2*(n_d + n_e + n_f) + l_d + l_e + l_f) <= N_3max && l_c == l_f && P_abc == (rem(l_d + l_e + l_f,2) + 1)
                                        j_f = Orb[f].j
                                        if j_c == j_f
                                            pRho_cf, nRho_cf = Rho.p[c,f], Rho.n[c,f]

                                            ME001 = V3b_no2b(a,b,c,0,d,e,f,0,J,1,P_abc,V_NNN,Orb,Orb_NNN)
                                            ME101 = V3b_no2b(a,b,c,1,d,e,f,0,J,1,P_abc,V_NNN,Orb,Orb_NNN)
                                            ME011 = V3b_no2b(a,b,c,0,d,e,f,1,J,1,P_abc,V_NNN,Orb,Orb_NNN)
                                            ME111 = V3b_no2b(a,b,c,1,d,e,f,1,J,1,P_abc,V_NNN,Orb,Orb_NNN)
                                            ME113 = V3b_no2b(a,b,c,1,d,e,f,1,J,3,P_abc,V_NNN,Orb,Orb_NNN)

                                            pHSum_local += is_j_a_hat * (0.5*ME113*pRho_be*pRho_cf + 0.25 * (ME001 +
                                                    sqrt(1.0/3.0)*ME101 + sqrt(1.0/3.0)*ME011 + ME111/3.0 + 2.0/3.0*ME113)*nRho_be*nRho_cf +
                                                    1.0/3.0 * (2.0*ME111 + ME113)*pRho_be*nRho_cf)

                                            nHSum_local += is_j_a_hat * (0.5*ME113*nRho_be*nRho_cf + 0.25 * (ME001 +
                                                    sqrt(1.0/3.0)*ME101 + sqrt(1.0/3.0)*ME011 + ME111/3.0 + 2.0/3.0*ME113)*pRho_be*pRho_cf +
                                                    1.0/3.0 * (2.0*ME111 + ME113)*nRho_be*pRho_cf)

                                        end
                                    end
                                end
                            end
                        end
                    end
                end

            end

            # Anomal Density-dependent part ... Only 3-body NNN interaction part ..
            if (2*(n_b + n_e) + l_b + l_e) <= N_2max && (2*(n_a + n_b + n_e) + l_a + l_b + l_e) <= N_3max && l_b == l_e && j_b == j_e
                pKappa_be, nKappa_be = Kappa.p[b,e], Kappa.n[b,e]
                P_abe = rem(l_a + l_b + l_e, 2) + 1
                @inbounds for c in 1:a_max
                    n_c, l_c, j_c = Orb[c].n, Orb[c].l, Orb[c].j
                    A_NNN_Amp = 0.25 * sqrt(Float64((j_b + 1) * (j_c + 1)))
                    @inbounds for f in 1:a_max
                        n_f = Orb[f].n
                        l_f = Orb[f].l
                        if (2*(n_c + n_d + n_f) + l_c + l_d + l_f) <= N_3max && P_abe == (rem(l_c + l_d + l_f,2) + 1) && l_c == l_f
                            j_f = Orb[f].j
                            if j_c == j_f
                                pKappa_cf, nKappa_cf = Kappa.p[c,f], Kappa.n[c,f]

                                ME111 = V3b_no2b(b,e,a,1,c,f,d,1,0,1,P_abe,V_NNN,Orb,Orb_NNN)
                                ME113 = V3b_no2b(b,e,a,1,c,f,d,1,0,3,P_abe,V_NNN,Orb,Orb_NNN)

                                pHSum_local += p3 * A_NNN_Amp * (ME113 * pKappa_be * pKappa_cf +
                                                (2.0 * ME111 + ME113) / 3.0 * nKappa_be * nKappa_cf)
                                nHSum_local += p3 * A_NNN_Amp * (ME113 * nKappa_be * nKappa_cf +
                                                (2.0 * ME111 + ME113) / 3.0 * pKappa_be * pKappa_cf)

                            end
                        end
                    end
                end
            end

            pH_local[Tid] += pHSum_local
            nH_local[Tid] += nHSum_local

        end

        # Sum over the Thread-local accumulators for H ...
        pHSum = sum(pH_local)
        nHSum = sum(nH_local)

        # Include the 1-body kinetic energy & inclusion of Center-of-Mass motion (CM) correction ...
            # Combined 1- + 2-body kinetic operator with CM correction ...
        if CMS == "CMS1+2B"
            pHSum += T.p[a,d] * (1.0 - 1.0 / A)
            nHSum += T.n[a,d] * (1.0 - 1.0 / A)
            # Pure 1-body kinetic operator with no CM correction ...
        elseif CMS != "CMS2B"
            pHSum += T.p[a,d]
            nHSum += T.n[a,d]
        end
            # No contribution for pure 2-body kinetic operator with CM correction ...


        # Add 0-body chemical potential Lambda ...
        if a == d
            pHSum -= Lambda.p
            nHSum -= Lambda.n
        end

        # Allocate pH & nH ...
        if a != d
            pH[a,d], pH[d,a] = pHSum, pHSum
            nH[a,d], nH[d,a] = nHSum, nHSum
        elseif a == d
            pH[a,a], nH[a,a] = pHSum, nHSum
        end

        # Single-quasiparticle pairing field Delta ...
        if ((2*(n_a + n_d) + l_a + l_d) <= N_2max)
                # Delta field thread-local accumulators for each thread ...
            pDelta_local = zeros(Float64,Threads.maxthreadid())
            nDelta_local = zeros(Float64,Threads.maxthreadid())

            @inbounds Threads.@threads :static for be_i in 1:be_count
                b, e = be[1,be_i], be[2,be_i]
                n_b, l_b, j_b = Orb[b].n, Orb[b].l, Orb[b].j
                n_e, l_e, j_e = Orb[e].n, Orb[e].l, Orb[e].j

                j_b_hat = sqrt(Float64(j_b + 1))
                Amp = 0.5 * j_b_hat / j_a_hat

                Tid = Threads.threadid()
                pDelta2NSum_local, nDelta2NSum_local = 0.0, 0.0
                pDelta3NSum_local, nDelta3NSum_local = 0.0, 0.0

                if (2*(n_b + n_e) + l_b + l_e) <= N_2max && (l_b == l_e) && (j_b == j_e)
                    pKappa_be, nKappa_be = Kappa.p[b,e], Kappa.n[b,e]

                    # 2-body NN interaction part ...
                    pDelta2NSum_local += Amp * pKappa_be * O2b_pp(a,d,b,e,0,1,V_NN,Orb,Orb_NN)
                    nDelta2NSum_local += Amp * nKappa_be * O2b_nn(a,d,b,e,0,1,V_NN,Orb,Orb_NN)

                    # 3-body NNN interaction part ...
                    @inbounds for c in 1:a_max
                        n_c, l_c = Orb[c].n, Orb[c].l
                        if (2*(n_a + n_d + n_c) + l_a + l_d + l_c) <= N_3max
                            P_adc = rem(l_a + l_d + l_c, 2) + 1
                            j_c = Orb[c].j
                            @inbounds for f in 1:a_max
                                n_f = Orb[f].n
                                l_f = Orb[f].l
                                if (2*(n_b + n_e + n_f) + l_b + l_e + l_f) <= N_3max && l_c == l_f && P_adc == (rem(l_b + l_e + l_f,2) + 1)
                                    j_f = Orb[f].j
                                    if j_c == j_f
                                        pRho_cf, nRho_cf = Rho.p[c,f], Rho.n[c,f]

                                        ME111 = V3b_no2b(a,d,c,1,b,e,f,1,0,1,P_adc,V_NNN,Orb,Orb_NNN)
                                        ME113 = V3b_no2b(a,d,c,1,b,e,f,1,0,3,P_adc,V_NNN,Orb,Orb_NNN)

                                        pDelta3NSum_local += Amp * (ME113 * pRho_cf +
                                            (2.0 * ME111 + ME113) / 3.0 * nRho_cf) * pKappa_be
                                        nDelta3NSum_local += Amp * (ME113 * nRho_cf +
                                            (2.0 * ME111 + ME113) / 3.0 * pRho_cf) * nKappa_be

                                    end
                                end
                            end
                        end
                    end
                end

                pDelta_local[Tid] += p2 * pDelta2NSum_local + p3 * pDelta3NSum_local
                nDelta_local[Tid] += p2 * nDelta2NSum_local + p3 * nDelta3NSum_local
            end

            # Sum over the Thread-local accumulators for Delta ...
            pDeltaSum = sum(pDelta_local)
            nDeltaSum = sum(nDelta_local)

            # Allocate Delta ...
            if a != d
                pDelta[a,d], pDelta[d,a] = pDeltaSum, pDeltaSum
                nDelta[a,d], nDelta[d,a] = nDeltaSum, nDeltaSum
            elseif a == d
                pDelta[a,d] = pDeltaSum
                nDelta[a,d] = nDeltaSum
            end

        end

    end

    return O1B(pH,nH), O1B(pDelta,nDelta)
end

function HFB_allocate_H1b(Params::Parameters,SQE::pnVector)
    # Read parameters ...
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1)*(N_max + 2),2)

    # Allocate H_N
    H_N = qpO1B(O1B(diagm(SQE.p),diagm(SQE.n)),
                O1B(zeros(Float64,a_max,a_max),
                zeros(Float64,a_max,a_max)))

    return H_N
end