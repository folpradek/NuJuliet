function HF_MBPT2_TBDM(Params::Parameters,h_N::O1B,V_NN::O2B,Orb::Vector{Orb1B},Orb_NN::Orb2B)
    # Read parameters ...
    N_2max = Params.Calc.N2max
    J_max = N_2max + 1

    # Iteraction list for J & P ...
    JP = JP_initialize(J_max)

    # Initialize 2-body operator O ...
    Rho_NN = O2b_initialize(Params,Orb;Make_Orb_NN=false)

    # Allocate the components of PT(2) 2-body density matrix Rho2B ...
    println("\nAllocating the components of MBPT(2) 2-body density matrix Rho_NN ...")
    @time @inbounds for i in JP
        J, P = i[1], i[2]
        if P == 1
            println("\tCalculating   ...   J = " * string(J) * "/" * string(N_2max+1) * "\tP = +")
        else
            println("\tCalculating   ...   J = " * string(J) * "/" * string(N_2max+1) * "\tP = -")
        end

        N_T0, N_T1 = Orb_NN.N[1,P,J+1], Orb_NN.N[2,P,J+1]

        @inbounds Threads.@threads for Bra in 1:max(N_T0, N_T1)
            @inbounds for Ket in 1:Bra

                # Case of pn interaction ... T = 0
                if Bra <= N_T0
                    Ind = Bra + (Ket - 1) * N_T0 - div(Ket * (Ket - 1),2)
                    a, b = Orb_NN.Ind[1,P,J+1][Bra][1], Orb_NN.Ind[1,P,J+1][Bra][2]
                    c, d = Orb_NN.Ind[1,P,J+1][Ket][1], Orb_NN.Ind[1,P,J+1][Ket][2]
                    #j_a, j_b, j_c, j_d = Orb[a].j, Orb[b].j, Orb[c].j, Orb[d].j
                    #l_a, l_b, l_c, l_d = Orb[a].l, Orb[b].l, Orb[c].l, Orb[d].l

                    # Case of Rho_pn ...
                    if (Orb[a].pO == 1 && Orb[b].nO == 1) && (Orb[c].pO == 0 && Orb[d].nO == 0)
                        pE_a, nE_b, pE_c ,nE_d = h_N.p[a,a], h_N.n[b,b], h_N.p[c,c], h_N.n[d,d]
                        ME = O2b_pn(a,b,c,d,J,P,V_NN,Orb_NN) / (pE_a + nE_b - pE_c - nE_d)
                        @views Rho_NN.pn[P,J+1][Ind] = ME
                    elseif (Orb[a].pO == 0 && Orb[b].nO == 0) && (Orb[c].pO == 1 && Orb[d].nO == 1)
                        pE_a, nE_b, pE_c ,nE_d = h_N.p[a,a], h_N.n[b,b], h_N.p[c,c], h_N.n[d,d]
                        ME = O2b_pn(a,b,c,d,J,P,V_NN,Orb_NN) / (pE_c + nE_d - pE_a - nE_b)
                        @views Rho_NN.pn[P,J+1][Ind] = ME
                    end

                end

                # Case of pp & nn interaction ... T = 1
                if Bra <= N_T1
                    Ind = Bra + (Ket - 1) * N_T1 - div(Ket * (Ket - 1),2)
                    a, b = Orb_NN.Ind[2,P,J+1][Bra][1], Orb_NN.Ind[2,P,J+1][Bra][2]
                    c, d = Orb_NN.Ind[2,P,J+1][Ket][1], Orb_NN.Ind[2,P,J+1][Ket][2]
                    #j_a, j_b, j_c, j_d = Orb[a].j, Orb[b].j, Orb[c].j, Orb[d].j
                    #l_a, l_b, l_c, l_d = Orb[a].l, Orb[b].l, Orb[c].l, Orb[d].l

                    # Case of Rho_pp ...
                     if (Orb[a].pO == 1 && Orb[b].pO == 1) && (Orb[c].pO == 0 && Orb[d].pO == 0)
                        pE_a, pE_b, pE_c ,pE_d = h_N.p[a,a], h_N.p[b,b], h_N.p[c,c], h_N.p[d,d]
                        ME = O2b_pp(a,b,c,d,J,P,V_NN,Orb,Orb_NN) / (pE_a + pE_b - pE_c - pE_d)
                        @views Rho_NN.pp[P,J+1][Ind] = ME
                    elseif (Orb[a].pO == 0 && Orb[b].pO == 0) && (Orb[c].pO == 1 && Orb[d].pO == 1)
                        pE_a, pE_b, pE_c ,pE_d = h_N.p[a,a], h_N.p[b,b], h_N.p[c,c], h_N.p[d,d]
                        ME = O2b_pp(a,b,c,d,J,P,V_NN,Orb,Orb_NN) / (pE_c + pE_d - pE_a - pE_b)
                        @views Rho_NN.pp[P,J+1][Ind] = ME
                    end

                    # Case of Rho_nn ...
                    if (Orb[a].nO == 1 && Orb[b].nO == 1) && (Orb[c].nO == 0 && Orb[d].nO == 0)
                        nE_a, nE_b, nE_c ,nE_d = h_N.n[a,a], h_N.n[b,b], h_N.n[c,c], h_N.n[d,d]
                        ME = O2b_nn(a,b,c,d,J,P,V_NN,Orb,Orb_NN) / (nE_a + nE_b - nE_c - nE_d)
                        @views Rho_NN.nn[P,J+1][Ind] = ME
                    elseif (Orb[a].nO == 0 && Orb[b].nO == 0) && (Orb[c].nO == 1 && Orb[d].nO == 1)
                        nE_a, nE_b, nE_c ,nE_d = h_N.n[a,a], h_N.n[b,b], h_N.n[c,c], h_N.n[d,d]
                        ME = O2b_nn(a,b,c,d,J,P,V_NN,Orb,Orb_NN) / (nE_c + nE_d - nE_a - nE_b)
                        @views Rho_NN.nn[P,J+1][Ind] = ME
                    end

                end

            end
        end

    end

    return Rho_NN
end

function h1b_correlations(Params::Parameters,h_N::O1B,Rho_NN::O2B,V_NNN::Array{Vector{Vector{Float32}},4},Orb::Vector{Orb1B},Orb_NN::Orb2B,Orb_NNN::Orb3B)
    # Read parameters ...
    N_max, N_2max, N_3max = Params.Calc.Nmax, Params.Calc.N2max, Params.Calc.N3max
    a_max = div((N_max + 1)*(N_max + 2),2)

    # Import LHO -> HF transformation matrices ...
    @time C_HF = O1b_import(Params,Orb,"IO/" * Params.Calc.Path * "/Bin/C_HF.bin")

    Rho_NN_LHO = O2b_transformation(Params,Orb,Orb_NN,Rho_NN,O1B(C_HF.p',C_HF.n'))

    cf, cf_count = HF_allocate_indices(Params,Orb)

    h_p_LHO, h_n_LHO = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)

    function h1b_correlation_indices(Params::Parameters)
        # Read parameters ...
        N_max = Params.Calc.Nmax
        a_max = div((N_max + 1)*(N_max + 2),2)

        ab_list = Vector{Tuple{Int64,Int64}}(undef,a_max*a_max)
        ab_count = 0
        @inbounds for a in 1:a_max
            @inbounds for b in 1:a_max
                ab_count += 1
                ab_list[ab_count] = (a,b)
            end
        end

        return ab_list
    end

    ab_list = h1b_correlation_indices(Params)

    println("\nCalculating correlated 1-body Hamiltonian h1b_correlations ...")
    @inbounds for cf_i in 1:cf_count
        c, f = cf[1,cf_i], cf[2,cf_i]
        n_c, l_c, j_c = Orb[c].n, Orb[c].l, Orb[c].j
        n_f, l_f, j_f = Orb[f].n, Orb[f].l, Orb[f].j

        j_c_hat = Float64(j_c + 1)

        ph_local = zeros(Float64,Threads.maxthreadid())
        nh_local = zeros(Float64,Threads.maxthreadid())

        @inbounds Threads.@threads :static for ab in 1:a_max^2
            a, b = ab_list[ab][1], ab_list[ab][2]
            n_a, l_a, j_a = Orb[a].n, Orb[a].l, Orb[a].j
            n_b, l_b, j_b = Orb[b].n, Orb[b].l, Orb[b].j

            Tid = Threads.threadid()
            pSum, nSum = 0.0, 0.0

            if (2*(n_a + n_b) + l_a + l_b) <= N_2max && (2*(n_a + n_b + n_c) + l_a + l_b + l_c) <= N_3max

                P2B = rem(l_a + l_b, 2) + 1
                P3B = rem(l_a + l_b + l_c, 2) + 1
                @inbounds for J in div(abs(j_a - j_b),2):div((j_a + j_b),2)

                    @inbounds for d in 1:a_max
                        n_d, l_d, j_d = Orb[d].n, Orb[d].l, Orb[d].j

                        @inbounds @simd for e in 1:a_max
                            n_e, l_e, j_e = Orb[e].n, Orb[e].l, Orb[e].j

                            if (2*(n_d + n_e) + l_d + l_e) <= N_2max && (2*(n_d + n_e + n_f) + l_d + l_e + l_f) <= N_3max &&
                            rem(l_d + l_e, 2) + 1 == P2B && rem(l_d + l_e + l_f, 2) + 1 == P3B && abs(j_d - j_e) <= 2*J && 2*J <= (j_d + j_e)

                                ppRho_abde = O2b_pp(a,b,d,e,J,P2B,Rho_NN_LHO,Orb,Orb_NN)
                                pnRho_abde = O2b_pn(a,b,d,e,J,P2B,Rho_NN_LHO,Orb_NN)
                                pnRho_baed = O2b_pn(b,a,e,d,J,P2B,Rho_NN_LHO,Orb_NN)
                                nnRho_abde = O2b_nn(a,b,d,e,J,P2B,Rho_NN_LHO,Orb,Orb_NN)

                                ME001 = V3b_no2b(a,b,c,0,d,e,f,0,J,1,P3B,V_NNN,Orb,Orb_NNN)
                                ME101 = V3b_no2b(a,b,c,1,d,e,f,0,J,1,P3B,V_NNN,Orb,Orb_NNN)
                                ME011 = V3b_no2b(a,b,c,0,d,e,f,1,J,1,P3B,V_NNN,Orb,Orb_NNN)
                                ME111 = V3b_no2b(a,b,c,1,d,e,f,1,J,1,P3B,V_NNN,Orb,Orb_NNN)
                                ME113 = V3b_no2b(a,b,c,1,d,e,f,1,J,3,P3B,V_NNN,Orb,Orb_NNN)

                                npME001 = V3b_no2b(b,a,c,0,e,d,f,0,J,1,P3B,V_NNN,Orb,Orb_NNN)
                                npME101 = V3b_no2b(b,a,c,1,e,d,f,0,J,1,P3B,V_NNN,Orb,Orb_NNN)
                                npME011 = V3b_no2b(b,a,c,0,e,d,f,1,J,1,P3B,V_NNN,Orb,Orb_NNN)
                                npME111 = V3b_no2b(b,a,c,1,e,d,f,1,J,1,P3B,V_NNN,Orb,Orb_NNN)
                                npME113 = V3b_no2b(b,a,c,1,e,d,f,1,J,3,P3B,V_NNN,Orb,Orb_NNN)

                                @inbounds pSum += 0.25 / j_c_hat * (ME113 * ppRho_abde + (2.0 * ME111 + ME113) / 3.0 * nnRho_abde +
                                        2.0 * (ME001 - ME101 / sqrt(3.0) - ME011 / sqrt(3.0) + ME111 / 3.0 + 2.0 / 3.0 * ME113) * pnRho_abde)
                                @inbounds nSum += 0.25 / j_c_hat * (ME113 * nnRho_abde + (2.0 * ME111 + ME113) / 3.0 * ppRho_abde +
                                        2.0 * (npME001 - npME101 / sqrt(3.0) - npME011 / sqrt(3.0) + npME111 / 3.0 + 2.0 / 3.0 * npME113) * pnRho_baed)

                            end

                        end

                    end

                end

            end


            ph_local[Tid] += pSum
            nh_local[Tid] += nSum

        end

        @inbounds h_p_LHO[c,f] += sum(ph_local)
        @inbounds h_n_LHO[c,f] += sum(nh_local)

        if c != f
            @inbounds h_p_LHO[f,c] += sum(ph_local)
            @inbounds h_n_LHO[f,c] += sum(nh_local)
        end


    end

    h_N.p .+= C_HF.p' * h_p_LHO * C_HF.p
    h_N.n .+= C_HF.n' * h_n_LHO * C_HF.n

    return h_N
end

function HF_ScRPA_TBDM_allocate(Params::Parameters,N_nu::Matrix{Int64},Orb_Phonon::Matrix{Vector{Int64}},Phonon::Vector{PhState},N_Particle::pnInteger,Particle::pnSVector,N_Hole::pnInteger,Hole::pnSVector,Orb::Vector{Orb1B},Orb_NN::Orb2B,X_RPA::Matrix{Matrix{ComplexF64}},Y_RPA::Matrix{Matrix{ComplexF64}})
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
                            g, l_h, j_g = b, l_b, j_b
                        end

                        p_ind, q_ind = ph_list[p][1], ph_list[q][2]
                        h_ind, g_ind = ph_list[h][1], ph_list[g][2]

                        pnSigmaME = HF_RPA_TBDM_2p2h_T0(p_ind,l_p,j_p,q_ind,l_q,j_q,h_ind,l_h,j_h,g_ind,l_g,j_g,J,X_RPA,Y_RPA,N_nu,ParticleHole)
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

                            ppSigmaME = HF_RPA_TBDM_2p2h_T1(p_ind,l_p,j_p,q_ind,l_q,j_q,h_ind,l_h,j_h,g_ind,l_g,j_g,J,-1,X_RPA,Y_RPA,N_nu,ParticleHole)
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

                            nnSigmaME = HF_RPA_TBDM_2p2h_T1(p_ind,l_p,j_p,q_ind,l_q,j_q,h_ind,l_h,j_h,g_ind,l_g,j_g,J,1,X_RPA,Y_RPA,N_nu,ParticleHole)
                            nnSigmaSum += nnSigmaME

                        end
                    

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
                                ppSigmaME = HF_RPA_TBDM_1p1h(p_ind,l_p,j_p,q_ind,l_q,j_q,h_ind,l_h,j_h,g_ind,l_g,j_g,J,-1,X_RPA,Y_RPA,N_nu,ParticleHole)
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

                                nnSigmaME = HF_RPA_TBDM_1p1h(p_ind,l_p,j_p,q_ind,l_q,j_q,h_ind,l_h,j_h,g_ind,l_g,j_g,J,1,X_RPA,Y_RPA,N_nu,ParticleHole)
                                nnSigmaSum += nnSigmaME
                            end

                        end

                    @inbounds Sigma_NN.pp[P,J+1][Ind] = ppSigmaSum
                    @inbounds Sigma_NN.nn[P,J+1][Ind] = nnSigmaSum

                end

            end
        end

    end


    println("\nSuccesfully allocate 2-body RPA correlation matrix Sigma_NN ...")
    
    return Sigma_NN
end

function V2b_correlation_energy(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,Sigma::O2B,V_NN::O2B)
    # Read parameters ...
    N_max = Params.Calc.Nmax
    N_2max = Params.Calc.N2max
    a_max = div((N_max + 1)*(N_max + 2),2)

    # Calculate the HF energy ...
    println("\nComputing the correlation ground-state energy by contraction with the provided 2-body correlation function...")

    # Initialize accumulator for the threaded correlation energy ...
    E_corr_threads = zeros(Float64,Threads.maxthreadid())

    @inbounds Threads.@threads :static for a in 1:a_max
        Sum, Tid = 0.0, Threads.threadid()

        Orb_a = Orb[a]
        n_a, l_a, j_a = Orb_a.n, Orb_a.l, Orb_a.j

        @inbounds for b in 1:a_max
            Orb_b = Orb[b]
            n_b, l_b = Orb_b.n, Orb_b.l

            P = rem(l_a + l_b, 2) + 1

            if (2*(n_a + n_b) + l_a + l_b) > N_2max
                continue
            end

            j_b = Orb[b].j

            @inbounds for c = 1:a_max
                Orb_c = Orb[c]
                n_c, l_c,j_c = Orb_c.n, Orb_c.l, Orb_c.j

                @inbounds for d = 1:a_max
                    Orb_d = Orb[d]
                    n_d, l_d, j_d = Orb_d.n, Orb_d.l, Orb_d.j

                    if ((2*(n_c + n_d) + l_c + l_d) > N_2max) || (rem(l_a + l_b, 2) != rem(l_c + l_d, 2))
                        continue
                    end

                    J_min, J_max = max(div(abs(j_a - j_b),2),div(abs(j_c - j_d),2)), min(div((j_a + j_b),2),div((j_c + j_d),2))

                    @inbounds for J in J_min:J_max
                        Hat =  Float64(2*J + 1)
                        Sum += 0.25 * Hat * O2b_pp(a,b,c,d,J,P,V_NN,Orb,Orb_NN) * O2b_pp(a,b,c,d,J,P,Sigma,Orb,Orb_NN)
                        Sum += 0.25 * Hat * O2b_nn(a,b,c,d,J,P,V_NN,Orb,Orb_NN) * O2b_nn(a,b,c,d,J,P,Sigma,Orb,Orb_NN)
                        Sum += Hat * O2b_pn(a,b,c,d,J,P,V_NN,Orb_NN) * O2b_pn(a,b,c,d,J,P,Sigma,Orb_NN)
                    end
                end
            end
        end
        E_corr_threads[Tid] += Sum
    end

    # Accumulate the total correlation energy ...
    E_corr = sum(E_corr_threads)

    println("\tCalculate correlation energy ...   E_corr = " * string(E_corr) * " MeV")

    return E_corr
end

function HF_RPA_TBDM_2p2h_T1(p::Int64,l_p::Int64,j_p::Int64,q::Int64,l_q::Int64,j_q::Int64,h::Int64,l_h::Int64,j_h::Int64,g::Int64,l_g::Int64,j_g::Int64,J::Int64,T::Int64,X_RPA::Matrix{Matrix{ComplexF64}},Y_RPA::Matrix{Matrix{ComplexF64}},N_nu::Matrix{Int64},ParticleHole::pnArray)
    Sigma  = 0.0

    # Direct term ...
    Q = rem(l_p + l_h,2) + 1
    if Q == (rem(l_q + l_g,2) + 1)
        @inbounds for I in max(div(abs(j_p - j_h),2),div(abs(j_q - j_g),2)):min(div((j_p + j_h),2),div((j_q + j_g),2))
            N_ph = N_nu[I+1,Q]
            Amp = 0.5 * Float64(2*I + 1) * (-1)^(J + I + div(j_q + j_h,2)) * f6j(j_p,j_q,2*J,j_g,j_h,2*I)
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
            Amp = 0.5 * Float64(2*I + 1) * (-1)^(I + div(j_q + j_g,2)) * f6j(j_p,j_q,2*J,j_h,j_g,2*I)
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

function HF_RPA_TBDM_2p2h_T0(p::Int64,l_p::Int64,j_p::Int64,q::Int64,l_q::Int64,j_q::Int64,h::Int64,l_h::Int64,j_h::Int64,g::Int64,l_g::Int64,j_g::Int64,J::Int64,X_RPA::Matrix{Matrix{ComplexF64}},Y_RPA::Matrix{Matrix{ComplexF64}},N_nu::Matrix{Int64},ParticleHole::pnArray)
    Sigma  = 0.0

    # Direct term only ...
    Q = rem(l_p + l_h,2) + 1
    if Q == (rem(l_q + l_g,2) + 1)
        @inbounds for I in max(div(abs(j_p - j_h),2),div(abs(j_q - j_g),2)):min(div((j_p + j_h),2),div((j_q + j_g),2))
            N_ph = N_nu[I+1,Q]
            Amp = 0.5 * Float64(2*I + 1) * (-1)^(J + I + div(j_q + j_h,2)) * f6j(j_p,j_q,2*J,j_g,j_h,2*I)
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

function HF_RPA_TBDM_1p1h(p::Int64,l_p::Int64,j_p::Int64,q::Int64,l_q::Int64,j_q::Int64,h::Int64,l_h::Int64,j_h::Int64,g::Int64,l_g::Int64,j_g::Int64,J::Int64,T::Int64,X_RPA::Matrix{Matrix{ComplexF64}},Y_RPA::Matrix{Matrix{ComplexF64}},N_nu::Matrix{Int64},ParticleHole::pnArray)
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

    Sigma *= 0.0

    return Sigma
end