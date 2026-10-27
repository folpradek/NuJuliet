function HF_BCS_allocate_indices(Params::Parameters,Orb::Vector{Orb1B})
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

function HF_BCS_allocate(Params::Parameters,Rho::O1B,Kappa::O1B,Orb::Vector{Orb1B},Orb_NN::Orb2B,Orb_NNN::Orb3B,T::O1B,V_NN::O2B,V_NNN::Array{Vector{Vector{Float32}},4})
    # Read parameters ...
    A = Params.Calc.A
    N_max = Params.Calc.Nmax
    N_2max = Params.Calc.N2max
    N_3max = Params.Calc.N3max
    p3 = Params.Int.cP3N
    CMS = Params.Calc.CMS
    a_max = div((N_max + 1)*(N_max + 2),2)

    # Allocate  indices for HFB iteration ...
    ad, ad_count, be, be_count = HF_BCS_allocate_indices(Params,Orb)

    # Allocate new HF Hamiltonian matrices ...
    pH, nH = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)

    # Allocate the single-particle field H ...
    @inbounds for ad_i in 1:ad_count
        a, d = ad[1,ad_i], ad[2,ad_i]
        n_a, l_a, j_a = Orb[a].n, Orb[a].l, Orb[a].j
        n_d, l_d, j_d = Orb[d].n, Orb[d].l, Orb[d].j

        is_j_a_hat = 1.0 / (Float64(j_a) + 1.0)

        # Single-particle field H ..
            # H field thread-local accumulators for each thread ...
        pH_local = zeros(Float64,Threads.maxthreadid())
        nH_local = zeros(Float64,Threads.maxthreadid())

        @inbounds Threads.@threads :static for be_i in 1:be_count
            b, e = be[1,be_i], be[2,be_i]
            n_b, l_b, j_b = Orb[b].n, Orb[b].l, Orb[b].j
            n_e, l_e, j_e = Orb[e].n, Orb[e].l, Orb[e].j

            Tid = Threads.threadid()

            pHSum_local, nHSum_local = 0.0, 0.0

            # Normal Density-dependent part ...
            if (2*(n_a + n_b) + l_a + l_b) <= N_2max

                if l_b == l_e && j_b == j_e && (2*(n_d + n_e) + l_d + l_e) <= N_2max
                    pRho_be, nRho_be = Rho.p[b,e], Rho.n[b,e]
                    P_ab = rem(l_a + l_b,2) + 1

                    @inbounds for J = div(abs(j_d - j_e),2):div(j_d + j_e,2)
                        
                        # 2-body NN interaction part ...
                        if rem(l_a + l_b, 2) == rem(l_d + l_e, 2)
                            J_j_a_hat = Float64(2*J + 1) * is_j_a_hat
        
                            pHSum_local += J_j_a_hat * pRho_be * O2b_pp(a,b,d,e,J,P_ab,V_NN,Orb,Orb_NN)
                            pHSum_local += J_j_a_hat * nRho_be * O2b_pn(a,b,d,e,J,P_ab,V_NN,Orb_NN)
                            nHSum_local += J_j_a_hat * nRho_be * O2b_nn(a,b,d,e,J,P_ab,V_NN,Orb,Orb_NN)
                            nHSum_local += J_j_a_hat * pRho_be * O2b_pn(b,a,e,d,J,P_ab,V_NN,Orb_NN)
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
            if l_b == l_e && j_b == j_e &&(2*(n_b + n_e) + l_b + l_e) <= N_2max && (2*(n_a + n_b + n_e) + l_a + l_b + l_e) <= N_3max
                pKappa_be, nKappa_be = Kappa.p[b,e], Kappa.n[b,e]
                P_abe = rem(l_a + l_b + l_e, 2) + 1
                @inbounds for c in 1:a_max
                    n_c, l_c, j_c = Orb[c].n, Orb[c].l, Orb[c].j
                    A_NNN_Amp = 0.25 * sqrt(Float64((j_b + 1) * (j_c + 1)))# * is_j_a_hat^2
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
                                        1.0 / 3.0 * (2.0 * ME111 + ME113) * nKappa_be * nKappa_cf)
                                nHSum_local += p3 * A_NNN_Amp * (ME113 * nKappa_be * nKappa_cf +
                                        1.0 / 3.0 * (2.0 * ME111 + ME113) * pKappa_be * pKappa_cf)

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

        # Allocate pH & nH ...
        if a != d
            pH[a,d], pH[d,a] = pHSum, pHSum
            nH[a,d], nH[d,a] = nHSum, nHSum
        elseif a == d
            pH[a,a], nH[a,a] = pHSum, nHSum
        end

    end

    return O1B(pH,nH)
end

function BCS_allocate_SQE(Params::Parameters,SPE::pnVector,Lambda::pnFloat,Delta::pnVector)
    # Read parameters ...
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1)*(N_max + 2),2)

    # Initialize vectors for SQEs ...
    pSQE, nSQE = zeros(Float64,a_max), zeros(Float64,a_max)

    # Calculate BCS single-quasiparticle energies (SQEs) ...
    println("\nEvalutiang BCS single-quasiparticle energies ...")
    @inbounds for a in 1:a_max
        pE_a = sqrt((SPE.p[a] - Lambda.p)^2 + Delta.p[a]^2)
        nE_a = sqrt((SPE.n[a] - Lambda.n)^2 + Delta.n[a]^2)
        pSQE[a], nSQE[a] = pE_a, nE_a
    end

    return pnVector(pSQE,nSQE)
end

function BCS_allocate_amplitudes(Params::Parameters,Lambda::pnFloat,SPE::pnVector,Delta::pnVector,Orb::Vector{Orb1B})
    # Read parameters ...
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1)*(N_max + 2),2)

    # Initialize U & V amplitudes ...
    pU, pV = zeros(Float64,a_max), zeros(Float64,a_max)
    nU, nV = zeros(Float64,a_max), zeros(Float64,a_max)

    # Calculate U & V from initial guess ... from the initial guess
    @inbounds for a in 1:a_max
        # Correct Condon-Shortley phase factors ... (irrelevant ???)
        Amp = Float64((-1)^Orb[a].l)

        pME = (SPE.p[a] - Lambda.p) / sqrt(((SPE.p[a] - Lambda.p))^2 + Delta.p[a]^2)
        pV[a] = sqrt(0.5 * (1.0 - pME))
        pU[a] = Amp * sqrt(0.5 * (1.0 + pME))

        nME = (SPE.n[a] - Lambda.n) / sqrt(((SPE.n[a] - Lambda.n))^2 + Delta.n[a]^2)
        nV[a] = sqrt(0.5 * (1.0 - nME))
        nU[a] = Amp * sqrt(0.5 * (1.0 + nME))
    end

    return pnVector(pU,nU), pnVector(pV,nV)
end

function BCS_allocate_Delta(Params::Parameters,U::pnVector,V::pnVector,Orb::Vector{Orb1B},Orb_NN_Res::Orb2B,V_NN_Res::O2B)
    # Read parameters ...
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1)*(N_max + 2),2)

    # Initialize the pairing gap Delta ...
    pDelta, nDelta = zeros(Float64,a_max), zeros(Float64,a_max)

    # Allocate the pairing gap Delta ...
    @inbounds for a in 1:a_max
        j_a = Orb[a].j
        pSum, nSum = 0.0, 0.0
        @inbounds for b in 1:a_max
            j_b = Orb[b].j
            ja_jb_hat = sqrt(Float64(j_b + 1) / Float64(j_a + 1))
            pME = - 0.5 * ja_jb_hat * O2b_pp(a,a,b,b,0,1,V_NN_Res,Orb,Orb_NN_Res) * U.p[b] * V.p[b]
            nME = - 0.5 * ja_jb_hat * O2b_nn(a,a,b,b,0,1,V_NN_Res,Orb,Orb_NN_Res) * U.n[b] * V.n[b]
            pSum += pME
            nSum += nME
        end
        pDelta[a] = pSum
        nDelta[a] = nSum
    end    

    return pnVector(pDelta,nDelta)
end

function BCS_allocate_H1b(Params::Parameters,SQE::pnVector)
    # Read parameters ...
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1)*(N_max + 2),2)

    # Allocate H_N
    H_N = qpO1B(O1B(diagm(SQE.p),diagm(SQE.n)),
                O1B(zeros(Float64,a_max,a_max),
                zeros(Float64,a_max,a_max)))

    return H_N
end