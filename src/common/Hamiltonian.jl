# Temporary solution for Hamiltonian operator related functions ... to be changed!!!
function H1b(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,Orb_NNN::Orb3B,T::O1B,V_NN::O2B,V_NNN::Array{Vector{Vector{Float32}},4},Rho::O1B,Sigma::O2B)
    # Read parameters ...
    A = Float64(Params.Calc.A)
    N_max = Params.Calc.Nmax
    N_2max = Params.Calc.N2max
    N_3max = Params.Calc.N3max
    a_max = div((N_max + 1)*(N_max + 2),2)
    J_max = N_2max + 1
    CMS = Params.Calc.CMS

    println("\nEvaluating 1-body mean-field Hamiltonian h_N ...")

    # Number of work packages ... more than threads to balance the load ...
    N_task = 4*Threads.nthreads()

    function H1b_lj_pairs(Params::Parameters,Orb::Vector{Orb1B})
        # Read # initialize parameters ... (local, otherwise N_max & a_max of H1b are overwritten & become type-unstable ...)
        local N_max = Params.Calc.Nmax
        local a_max = div((N_max + 1)*(N_max + 2),2)

        # All ordered pairs ab & pairs ab with a >= b ... both with l_a = l_b & j_a = j_b ...
        ab_all, ab_low = NTuple{2,Int64}[], NTuple{2,Int64}[]
        @inbounds for a in 1:a_max
            l_a, j_a = Orb[a].l, Orb[a].j
            @inbounds for b in 1:a_max
                l_b, j_b = Orb[b].l, Orb[b].j
                if l_a == l_b && j_a == j_b
                    push!(ab_all,(a,b))
                    if b <= a
                        push!(ab_low,(a,b))
                    end
                end
            end
        end

        return ab_all, ab_low
    end

    function H1b_unpack(O_PJ::Vector{Float64},N::Int64)
        # Unpack the stored lower triangle of one block (P,J) of a 2-body operator (see O2b_index) into a dense symmetric matrix ...
        O_dense = Matrix{Float64}(undef,N,N)
        @inbounds Threads.@threads :dynamic for Ket in 1:N
            Offset = (Ket - 1) * N - div(Ket*(Ket - 1),2)
            @inbounds for Bra in Ket:N
                O_dense[Bra,Ket] = O_PJ[Bra + Offset]
                O_dense[Ket,Bra] = O_PJ[Bra + Offset]
            end
        end
        return O_dense
    end

    function H1b_NNN_triples(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,Orb_NNN::Orb3B)
        # Read # initialize parameters ... (local, otherwise the parameters of H1b are overwritten ...)
        local N_max, N_2max = Params.Calc.Nmax, Params.Calc.N2max
        local a_max = div((N_max + 1)*(N_max + 2),2)
        local J_max = N_2max + 1
        l_max = N_max

        # 3-body triples (a b c) with a >= b of every V_NNN block [P,l_c+1,J+1][j_c] ... stored as (a, b, c, ab, i_0, i_1, i_3),
        #   ab  ... index of the pair a >= b in Orb_NN (Orb_NN.Map[a,b,J+1,2]),
        #   i_0 ... index of (a b, T_ab = 0, c) in V_NNN[P,l_c+1,J+1,1][j_c] (T = 1/2),
        #   i_1 ... index of (a b, T_ab = 1, c) in V_NNN[P,l_c+1,J+1,1][j_c] (T = 1/2),
        #   i_3 ... index of (a b, T_ab = 1, c) in V_NNN[P,l_c+1,J+1,2][j_c] (T = 3/2) ... (see Map_ab[T_ab + T,J - J_min + 1,c] in V3b_no2b_initialize)
        local NNN_triples = Array{Vector{Vector{NTuple{7,Int64}}}}(undef,2,l_max+1,J_max+1)
        @inbounds for P in 1:2
            @inbounds for l in 0:l_max
                @inbounds for J in 0:J_max
                    NNN_triples[P,l+1,J+1] = [NTuple{7,Int64}[] for j_index in 1:length(Orb_NNN.N[P,l+1,J+1,1])]
                end
            end
        end

        @inbounds for a in 1:a_max
            l_a, j_a = Orb[a].l, Orb[a].j
            @inbounds for b in 1:a
                l_b, j_b = Orb[b].l, Orb[b].j
                Map_ab = Orb_NNN.Map[div(a*(a-1),2) + b]
                J_min = div(abs(j_a - j_b),2)
                @inbounds for J_i in 1:size(Map_ab,2)
                    J = J_min + J_i - 1
                    ab = Int64(Orb_NN.Map[a,b,J+1,2])
                    @inbounds for c in 1:a_max
                        l_c, j_c = Orb[c].l, Orb[c].j
                        if Map_ab[1,J_i,c] == 0
                            continue
                        end
                        P = parity(l_a + l_b + l_c)
                        if j_c == abs(2*l_c - 1)
                            j_index = 1
                        else
                            j_index = 2
                        end
                        push!(NNN_triples[P,l_c+1,J+1][j_index],(a,b,c,ab,Int64(Map_ab[1,J_i,c]),Int64(Map_ab[2,J_i,c]),Int64(Map_ab[3,J_i,c])))
                    end
                end
            end
        end

        return NNN_triples
    end

    function H1b_V3b_index(Bra::Int64,Ket::Int64)
        # Index of (Bra|V_NNN|Ket) in the stored lower triangle of a V_NNN block (see V3b_no2b_index) ...
        if Bra >= Ket
            return Ket + div(Bra*(Bra - 1),2)
        else
            return Bra + div(Ket*(Ket - 1),2)
        end
    end

    # Allocate pairs of orbitals with equal l & j ...
    lj_pairs, lj_pairs_low = H1b_lj_pairs(Params,Orb)
    ad, ad_count = lj_pairs_low, length(lj_pairs_low)
    be, be_count = lj_pairs, length(lj_pairs)
    cf = lj_pairs

    # Allocate new 1-body Hamiltonian matrices ...
    ph_N, nh_N = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)

    # Allocate the single-particle field H ...
        # Due to 1-body contractions with the 1-body density Rho ...
            # List of all (ad, be) combinations ... used to distribute the work among tasks ...
        adbe_tasks = NTuple{2,Int64}[]
        @inbounds for ad_i in 1:ad_count
            @inbounds for be_i in 1:be_count
                push!(adbe_tasks,(ad_i,be_i))
            end
        end

        # Initialize the thread local 1-body Hamiltonian matrices ...
        ph_N_local = zeros(Float64,ad_count,N_task)
        nh_N_local = zeros(Float64,ad_count,N_task)

        # Each task takes every N_task-th (ad, be) combination ...
        @inbounds Threads.@threads :dynamic for task in 1:N_task
            @inbounds for task_i in task:N_task:length(adbe_tasks)
                ad_i, be_i = adbe_tasks[task_i]
                a, d = ad[ad_i]
                b, e = be[be_i]
                n_a, l_a, j_a = Orb[a].n, Orb[a].l, Orb[a].j
                n_d, l_d, j_d = Orb[d].n, Orb[d].l, Orb[d].j
                n_b, l_b, j_b = Orb[b].n, Orb[b].l, Orb[b].j
                n_e, l_e, j_e = Orb[e].n, Orb[e].l, Orb[e].j
                is_j_a_hat = 1.0 / hat(j_a)
                P_ab = parity(l_a + l_b)
                pSum_local, nSum_local = 0.0, 0.0

                if (2*(n_a + n_b) + l_a + l_b) <= N_2max && (2*(n_d + n_e) + l_d + l_e) <= N_2max
                    pRho_be, nRho_be = Rho.p[b,e], Rho.n[b,e]

                    @inbounds for J = div(abs(j_d - j_e),2):div(j_d + j_e,2)

                        # 2-body NN interaction part ...
                        if P_ab == parity(l_d + l_e)
                            J_j_a_hat = hat(2*J) * is_j_a_hat
                            pSum_local += J_j_a_hat * pRho_be * O2b_pp(a,b,d,e,J,P_ab,V_NN,Orb,Orb_NN)
                            pSum_local += J_j_a_hat * nRho_be * O2b_pn(a,b,d,e,J,P_ab,V_NN,Orb_NN)
                            nSum_local += J_j_a_hat * nRho_be * O2b_nn(a,b,d,e,J,P_ab,V_NN,Orb,Orb_NN)
                            nSum_local += J_j_a_hat * pRho_be * O2b_pn(b,a,e,d,J,P_ab,V_NN,Orb_NN)
                        end

                        # 3-body NNN interaction part ... only cf with l_c = l_f & j_c = j_f ...
                        @inbounds for (c, f) in cf
                            n_c, l_c = Orb[c].n, Orb[c].l
                            n_f, l_f = Orb[f].n, Orb[f].l
                            P_abc = parity(l_a + l_b + l_c)
                            if (2*(n_a + n_b + n_c) + l_a + l_b + l_c) <= N_3max && (2*(n_d + n_e + n_f) + l_d + l_e + l_f) <= N_3max &&
                                P_abc == parity(l_d + l_e + l_f)
                                pRho_cf, nRho_cf = Rho.p[c,f], Rho.n[c,f]

                                ME001 = V3b_no2b(a,b,c,0,d,e,f,0,J,1,P_abc,V_NNN,Orb,Orb_NNN)
                                ME101 = V3b_no2b(a,b,c,1,d,e,f,0,J,1,P_abc,V_NNN,Orb,Orb_NNN)
                                ME011 = V3b_no2b(a,b,c,0,d,e,f,1,J,1,P_abc,V_NNN,Orb,Orb_NNN)
                                ME111 = V3b_no2b(a,b,c,1,d,e,f,1,J,1,P_abc,V_NNN,Orb,Orb_NNN)
                                ME113 = V3b_no2b(a,b,c,1,d,e,f,1,J,3,P_abc,V_NNN,Orb,Orb_NNN)

                                pSum_local += is_j_a_hat * (0.5*ME113*pRho_be*pRho_cf + 0.25 * (ME001 +
                                                sqrt(1.0/3.0)*ME101 + sqrt(1.0/3.0)*ME011 + ME111/3.0 + 2.0/3.0*ME113)*nRho_be*nRho_cf +
                                                1.0/3.0 * (2.0*ME111 + ME113)*pRho_be*nRho_cf)

                                nSum_local += is_j_a_hat * (0.5*ME113*nRho_be*nRho_cf + 0.25 * (ME001 +
                                                sqrt(1.0/3.0)*ME101 + sqrt(1.0/3.0)*ME011 + ME111/3.0 + 2.0/3.0*ME113)*pRho_be*pRho_cf +
                                                1.0/3.0 * (2.0*ME111 + ME113)*nRho_be*pRho_cf)
                            end
                        end
                    end
                end

                ph_N_local[ad_i,task] += pSum_local
                nh_N_local[ad_i,task] += nSum_local
            end
        end

        # Include kinetic & CM contributions ...
        @inbounds for ad_i in 1:ad_count
            a, d = ad[ad_i]

            # Sum over the task-local accumulators for H ...
            phSum = sum(view(ph_N_local,ad_i,:))
            nhSum = sum(view(nh_N_local,ad_i,:))

            # Include the 1-body kinetic energy & inclusion of Center-of-Mass motion (CM) correction ...
                # Combined 1- + 2-body kinetic operator with CM correction ...
            if CMS == "CMS1+2B"
                phSum += T.p[a,d] * (1.0 - 1.0 / A)
                nhSum += T.n[a,d] * (1.0 - 1.0 / A)
                # Pure 1-body kinetic operator with no CM correction ...
            elseif CMS != "CMS2B"
                phSum += T.p[a,d]
                nhSum += T.n[a,d]
            end
                # No contribution for pure 2-body kinetic operator with CM correction ...

            # Allocate ph & nh ...
            if a != d
                ph_N[a,d], ph_N[d,a] = phSum, phSum
                nh_N[a,d], nh_N[d,a] = nhSum, nhSum
            elseif a == d
                ph_N[a,a], nh_N[a,a] = phSum, nhSum
            end
        end

    # Due to 2-body contractions with the 2-body correlation function Sigma ...
        # h_cf = 1/4 1/j_c_hat sum_J sum_ab sum_de (ab c|V_NNN|de f) Sigma_ab,de ... sums over all ordered pairs ab & de, isospin written out below ...
        # V_NNN is stored only for a >= b & d >= e, per block [P,l_c+1,J+1,T][j_c] (fixed l_c = l_f & j_c = j_f) as a lower triangle ...
        # so the loop runs over pairs of triples (a b c) >= (d e f) of a block & the orderings ba & ed are added through their phases ...
        NNN_triples = H1b_NNN_triples(Params,Orb,Orb_NN,Orb_NNN)

        # Task-local accumulators for h_cf ...
        ph_N_task = [zeros(Float64,a_max,a_max) for task in 1:N_task]
        nh_N_task = [zeros(Float64,a_max,a_max) for task in 1:N_task]

        @inbounds for J in 0:J_max
            @inbounds for P2B in 1:2
                N_ab, N_ab_pn = Orb_NN.N[2,P2B,J+1], Orb_NN.N[1,P2B,J+1]
                if N_ab == 0
                    continue
                end

                # Dense Sigma for this J & P_ab ...
                #   pp & nn for pairs a >= b ([de,ab], Orb_NN.Map[a,b,J+1,2]),
                #   pn for all ordered pairs ([de,ab], Orb_NN.Map[a,b,J+1,1]) ...
                ppSigma_JP = H1b_unpack(Sigma.pp[P2B,J+1],N_ab)
                pnSigma_JP = H1b_unpack(Sigma.pn[P2B,J+1],N_ab_pn)
                nnSigma_JP = H1b_unpack(Sigma.nn[P2B,J+1],N_ab)

                # Each task takes every N_task-th triple (a b c) of every V_NNN block with this J & P_ab ...
                @inbounds Threads.@threads :dynamic for task in 1:N_task
                    ph_task, nh_task = ph_N_task[task], nh_N_task[task]

                    @inbounds for l_c in 0:N_max
                        P3B = parity(P2B - 1 + l_c)

                        @inbounds for (j_index, Triples) in enumerate(NNN_triples[P3B,l_c+1,J+1])
                            V_T1, V_T3 = V_NNN[P3B,l_c+1,J+1,1][j_index], V_NNN[P3B,l_c+1,J+1,2][j_index]

                            @inbounds for abc in task:N_task:length(Triples)
                                a, b, c, ab, abc_0, abc_1, abc_3 = Triples[abc]
                                ab_pn, ba_pn = Int64(Orb_NN.Map[a,b,J+1,1]), Int64(Orb_NN.Map[b,a,J+1,1])

                                # Phase of V_NNN for (b a T_ab = 0| = (-1)^((j_a + j_b)/2 - J) (a b T_ab = 0| ... opposite for T_ab = 1 ...
                                Phase_ab = phase(div(Orb[a].j + Orb[b].j,2) - J)

                                @inbounds for def in 1:abc
                                    d, e, f, de, def_0, def_1, def_3 = Triples[def]
                                    de_pn, ed_pn = Int64(Orb_NN.Map[d,e,J+1,1]), Int64(Orb_NN.Map[e,d,J+1,1])
                                    Phase_de = phase(div(Orb[d].j + Orb[e].j,2) - J)

                                    # 3-body matrix elements (ab T_ab c|V_NNN|de T_de f) with T = 1/2 (ME_T_ab_T_de_1) & T = 3/2 (ME113) ...
                                    ME001 = Float64(V_T1[H1b_V3b_index(abc_0,def_0)])
                                    ME101 = Float64(V_T1[H1b_V3b_index(abc_1,def_0)])
                                    ME011 = Float64(V_T1[H1b_V3b_index(abc_0,def_1)])
                                    ME111 = Float64(V_T1[H1b_V3b_index(abc_1,def_1)])
                                    ME113 = Float64(V_T3[H1b_V3b_index(abc_3,def_3)])

                                    # pp & nn Sigma ... the phases of ba (ed) in V_NNN & Sigma cancel, so every ordering adds the same ...
                                    ppSigma_abde, nnSigma_abde = ppSigma_JP[de,ab], nnSigma_JP[de,ab]
                                    if a != b
                                        ppSigma_abde, nnSigma_abde = 2.0 * ppSigma_abde, 2.0 * nnSigma_abde
                                    end
                                    if d != e
                                        ppSigma_abde, nnSigma_abde = 2.0 * ppSigma_abde, 2.0 * nnSigma_abde
                                    end

                                    # pn Sigma of all orderings ab, ba & de, ed ...
                                    pnSigma_abde = pnSigma_JP[de_pn,ab_pn]
                                    pnSigma_bade, pnSigma_abed, pnSigma_baed = 0.0, 0.0, 0.0
                                    if a != b
                                        pnSigma_bade = Phase_ab * pnSigma_JP[de_pn,ba_pn]
                                    end
                                    if d != e
                                        pnSigma_abed = Phase_de * pnSigma_JP[ed_pn,ab_pn]
                                    end
                                    if a != b && d != e
                                        pnSigma_baed = Phase_ab * Phase_de * pnSigma_JP[ed_pn,ba_pn]
                                    end

                                    # ... projected on the isospin T_ab & T_de of the pairs ab & de (T = 1 flips the sign of the swapped pair) ...
                                    pnSigma_00 = pnSigma_abde + pnSigma_bade + pnSigma_abed + pnSigma_baed
                                    pnSigma_10 = pnSigma_abde - pnSigma_bade + pnSigma_abed - pnSigma_baed
                                    pnSigma_01 = pnSigma_abde + pnSigma_bade - pnSigma_abed - pnSigma_baed
                                    pnSigma_11 = pnSigma_abde - pnSigma_bade - pnSigma_abed + pnSigma_baed

                                    # Proton field (c, f protons) ... <1 0 1/2 +1/2|1/2 +1/2> = -sqrt(1/3) ...
                                    pSum = ME113 * ppSigma_abde + (ME113 + 2.0 * ME111) / 3.0 * nnSigma_abde +
                                           2.0 * (ME001 * pnSigma_00 - (ME101 * pnSigma_10 + ME011 * pnSigma_01) / sqrt(3.0) + (ME111 + 2.0 * ME113) / 3.0 * pnSigma_11)

                                    # Neutron field (c, f neutrons) ... <1 0 1/2 -1/2|1/2 -1/2> = +sqrt(1/3) ...
                                    nSum = ME113 * nnSigma_abde + (ME113 + 2.0 * ME111) / 3.0 * ppSigma_abde +
                                           2.0 * (ME001 * pnSigma_00 + (ME101 * pnSigma_10 + ME011 * pnSigma_01) / sqrt(3.0) + (ME111 + 2.0 * ME113) / 3.0 * pnSigma_11)

                                    # (abc|V|def) adds to h_cf ... (def|V|abc), not stored, adds the same to h_fc ...
                                    ph_task[c,f] += pSum
                                    nh_task[c,f] += nSum

                                    if def != abc
                                        ph_task[f,c] += pSum
                                        nh_task[f,c] += nSum
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end

        # Sum over the task-local accumulators ...
        ph_cf, nh_cf = sum(ph_N_task), sum(nh_N_task)
        @inbounds for (c, f) in cf
            ph_N[c,f] += 0.25 / hat(Orb[c].j) * ph_cf[c,f]
            nh_N[c,f] += 0.25 / hat(Orb[c].j) * nh_cf[c,f]
        end
    
    # Due to 2-body contractions of V_NN with the 2-body correlation function Sigma ...
        function H1b_dense(Orb::Vector{Orb1B},Orb_NN::Orb2B,O::O2B,P::Int64,J::Int64)
            # Dense matrices ppO[de,ac] = <ac|O|de>_J (pnO, nnO similarly) ... ac & de run over all ordered pairs (the pn pairs, Orb_NN.Ind[1,P,J+1]) ...
            N_ac, N_ac_low = Orb_NN.N[1,P,J+1], Orb_NN.N[2,P,J+1]
            ac_pairs = Orb_NN.Ind[1,P,J+1]

            pnO = H1b_unpack(O.pn[P,J+1],N_ac)
            ppO_low, nnO_low = H1b_unpack(O.pp[P,J+1],N_ac_low), H1b_unpack(O.nn[P,J+1],N_ac_low)

            # pp & nn are stored only for a >= c ... pair ac with a < c is the stored pair ca times the phase (-1)^(J + 1 + (j_a + j_c)/2) (see O2b_pp) ...
            ac_low, Phase_ac = zeros(Int64,N_ac), zeros(Float64,N_ac)
            @inbounds for ac in 1:N_ac
                a, c = ac_pairs[ac][1], ac_pairs[ac][2]
                if a >= c
                    ac_low[ac] = Int64(Orb_NN.Map[a,c,J+1,2])
                    Phase_ac[ac] = 1.0
                else
                    ac_low[ac] = Int64(Orb_NN.Map[c,a,J+1,2])
                    Phase_ac[ac] = phase(J + 1 + div(Orb[a].j + Orb[c].j,2))
                end
            end

            ppO, nnO = Matrix{Float64}(undef,N_ac,N_ac), Matrix{Float64}(undef,N_ac,N_ac)
            @inbounds Threads.@threads :dynamic for ac in 1:N_ac
                @inbounds for de in 1:N_ac
                    ppO[de,ac] = Phase_ac[de] * Phase_ac[ac] * ppO_low[ac_low[de],ac_low[ac]]
                    nnO[de,ac] = Phase_ac[de] * Phase_ac[ac] * nnO_low[ac_low[de],ac_low[ac]]
                end
            end

            return ppO, pnO, nnO
        end
        
        ph_abc, nh_abc = zeros(Float64,a_max,a_max,a_max), zeros(Float64,a_max,a_max,a_max)

        # List of all (a, c) ... used to distribute the work among threads ...
        ac_tasks = NTuple{2,Int64}[]
        @inbounds for c in 1:a_max
            @inbounds for a in 1:a_max
                push!(ac_tasks,(a,c))
            end
        end

        @inbounds for J in 0:J_max
            J_hat = hat(2*J)
            @inbounds for P in 1:2
                N_de = Orb_NN.N[1,P,J+1]
                if N_de == 0
                    continue
                end

                # V & Sigma as dense matrices ... e.g. ppV[de,ac] = <ac|V|de>_J for all ordered pairs ac & de with parity P ...
                ppV, pnV, nnV = H1b_dense(Orb,Orb_NN,V_NN,P,J)
                ppS, pnS, nnS = H1b_dense(Orb,Orb_NN,Sigma,P,J)

                @inbounds Threads.@threads :dynamic for (a,c) in ac_tasks
                    l_a, j_a = Orb[a].l, Orb[a].j

                    # Index of the pairs ac & ca ... 0 if not allowed by J & N_2max ...
                    ac, ca = Int64(Orb_NN.Map[a,c,J+1,1]), Int64(Orb_NN.Map[c,a,J+1,1])
                    if ac == 0 || parity(l_a + Orb[c].l) != P
                        continue
                    end

                    @inbounds for b in 1:a_max
                        if Orb[b].l != l_a || Orb[b].j != j_a
                            continue
                        end

                        bc, cb = Int64(Orb_NN.Map[b,c,J+1,1]), Int64(Orb_NN.Map[c,b,J+1,1])
                        if bc == 0
                            continue
                        end

                        pSum, nSum = 0.0, 0.0

                        @inbounds @simd for de in 1:N_de
                            pSum += 0.5 * (ppV[de,ac] * ppS[de,bc] + ppV[de,bc] * ppS[de,ac]) + pnV[de,ac] * pnS[de,bc] + pnV[de,bc] * pnS[de,ac]
                            nSum += 0.5 * (nnV[de,ac] * nnS[de,bc] + nnV[de,bc] * nnS[de,ac]) + pnV[de,ca] * pnS[de,cb] + pnV[de,cb] * pnS[de,ca]
                            #pSum += 0.5 * (ppV[de,bc] * ppS[de,ac]) + pnV[de,bc] * pnS[de,ac]
                            #nSum += 0.5 * (nnV[de,bc] * nnS[de,ac]) + pnV[de,cb] * pnS[de,ca]
                        end

                        ph_abc[a,b,c] += J_hat * pSum * 0.5
                        nh_abc[a,b,c] += J_hat * nSum * 0.5
                    end
                end
            end
        end

        # Sum over c ...
        @inbounds for (a,b) in lj_pairs_low
            pSum, nSum = sum(view(ph_abc,a,b,:)) / hat(Orb[a].j), sum(view(nh_abc,a,b,:)) / hat(Orb[a].j)
            ph_N[a,b] += pSum
            nh_N[a,b] += nSum

            if a != b
                ph_N[b,a] += pSum
                nh_N[b,a] += nSum
            end
        end

    println("\tCompleted the calculation of 1-body mean-field Hamiltonian h_N ...")

    return O1B(ph_N,nh_N)
end

function H1b(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,Orb_NNN::Orb3B,T::O1B,V_NN::O2B,V_NNN::Array{Vector{Vector{Float32}},4},Rho::O1B)
    # Read parameters ...
    A = Float64(Params.Calc.A)
    N_max = Params.Calc.Nmax
    N_2max = Params.Calc.N2max
    N_3max = Params.Calc.N3max
    a_max = div((N_max + 1)*(N_max + 2),2)
    J_max = N_2max + 1
    CMS = Params.Calc.CMS

    println("\nEvaluating 1-body mean-field Hamiltonian h_N ...")

    # Number of work packages ... more than threads to balance the load ...
    N_task = 4*Threads.nthreads()

    function H1b_lj_pairs(Params::Parameters,Orb::Vector{Orb1B})
        # Read # initialize parameters ... (local, otherwise N_max & a_max of H1b are overwritten & become type-unstable ...)
        local N_max = Params.Calc.Nmax
        local a_max = div((N_max + 1)*(N_max + 2),2)

        # All ordered pairs ab & pairs ab with a >= b ... both with l_a = l_b & j_a = j_b ...
        ab_all, ab_low = NTuple{2,Int64}[], NTuple{2,Int64}[]
        @inbounds for a in 1:a_max
            l_a, j_a = Orb[a].l, Orb[a].j
            @inbounds for b in 1:a_max
                l_b, j_b = Orb[b].l, Orb[b].j
                if l_a == l_b && j_a == j_b
                    push!(ab_all,(a,b))
                    if b <= a
                        push!(ab_low,(a,b))
                    end
                end
            end
        end

        return ab_all, ab_low
    end

    # Allocate pairs of orbitals with equal l & j ...
    lj_pairs, lj_pairs_low = H1b_lj_pairs(Params,Orb)
    ad, ad_count = lj_pairs_low, length(lj_pairs_low)
    be, be_count = lj_pairs, length(lj_pairs)
    cf = lj_pairs

    # Allocate new 1-body Hamiltonian matrices ...
    ph_N, nh_N = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)

    # Allocate the single-particle field H ...
        # Due to 1-body contractions with the 1-body density Rho ...
            # List of all (ad, be) combinations ... used to distribute the work among tasks ...
        adbe_tasks = NTuple{2,Int64}[]
        @inbounds for ad_i in 1:ad_count
            @inbounds for be_i in 1:be_count
                push!(adbe_tasks,(ad_i,be_i))
            end
        end

        # Initialize the thread local 1-body Hamiltonian matrices ...
        ph_N_local = zeros(Float64,ad_count,N_task)
        nh_N_local = zeros(Float64,ad_count,N_task)

        # Each task takes every N_task-th (ad, be) combination ...
        @inbounds Threads.@threads :dynamic for task in 1:N_task
            @inbounds for task_i in task:N_task:length(adbe_tasks)
                ad_i, be_i = adbe_tasks[task_i]
                a, d = ad[ad_i]
                b, e = be[be_i]
                n_a, l_a, j_a = Orb[a].n, Orb[a].l, Orb[a].j
                n_d, l_d, j_d = Orb[d].n, Orb[d].l, Orb[d].j
                n_b, l_b, j_b = Orb[b].n, Orb[b].l, Orb[b].j
                n_e, l_e, j_e = Orb[e].n, Orb[e].l, Orb[e].j
                is_j_a_hat = 1.0 / hat(j_a)
                P_ab = parity(l_a + l_b)
                pSum_local, nSum_local = 0.0, 0.0

                if (2*(n_a + n_b) + l_a + l_b) <= N_2max && (2*(n_d + n_e) + l_d + l_e) <= N_2max
                    pRho_be, nRho_be = Rho.p[b,e], Rho.n[b,e]

                    @inbounds for J = div(abs(j_d - j_e),2):div(j_d + j_e,2)

                        # 2-body NN interaction part ...
                        if P_ab == parity(l_d + l_e)
                            J_j_a_hat = hat(2*J) * is_j_a_hat
                            pSum_local += J_j_a_hat * pRho_be * O2b_pp(a,b,d,e,J,P_ab,V_NN,Orb,Orb_NN)
                            pSum_local += J_j_a_hat * nRho_be * O2b_pn(a,b,d,e,J,P_ab,V_NN,Orb_NN)
                            nSum_local += J_j_a_hat * nRho_be * O2b_nn(a,b,d,e,J,P_ab,V_NN,Orb,Orb_NN)
                            nSum_local += J_j_a_hat * pRho_be * O2b_pn(b,a,e,d,J,P_ab,V_NN,Orb_NN)
                        end

                        # 3-body NNN interaction part ... only cf with l_c = l_f & j_c = j_f ...
                        @inbounds for (c,f) in cf
                            n_c, l_c = Orb[c].n, Orb[c].l
                            n_f, l_f = Orb[f].n, Orb[f].l
                            P_abc = parity(l_a + l_b + l_c)
                            
                            if (2*(n_a + n_b + n_c) + l_a + l_b + l_c) <= N_3max && (2*(n_d + n_e + n_f) + l_d + l_e + l_f) <= N_3max &&
                                P_abc == parity(l_d + l_e + l_f)
                                pRho_cf, nRho_cf = Rho.p[c,f], Rho.n[c,f]

                                ME001 = V3b_no2b(a,b,c,0,d,e,f,0,J,1,P_abc,V_NNN,Orb,Orb_NNN)
                                ME101 = V3b_no2b(a,b,c,1,d,e,f,0,J,1,P_abc,V_NNN,Orb,Orb_NNN)
                                ME011 = V3b_no2b(a,b,c,0,d,e,f,1,J,1,P_abc,V_NNN,Orb,Orb_NNN)
                                ME111 = V3b_no2b(a,b,c,1,d,e,f,1,J,1,P_abc,V_NNN,Orb,Orb_NNN)
                                ME113 = V3b_no2b(a,b,c,1,d,e,f,1,J,3,P_abc,V_NNN,Orb,Orb_NNN)

                                pSum_local += is_j_a_hat * (0.5*ME113*pRho_be*pRho_cf + 0.25 * (ME001 +
                                                sqrt(1.0/3.0)*ME101 + sqrt(1.0/3.0)*ME011 + ME111/3.0 + 2.0/3.0*ME113)*nRho_be*nRho_cf +
                                                1.0/3.0 * (2.0*ME111 + ME113)*pRho_be*nRho_cf)

                                nSum_local += is_j_a_hat * (0.5*ME113*nRho_be*nRho_cf + 0.25 * (ME001 +
                                                sqrt(1.0/3.0)*ME101 + sqrt(1.0/3.0)*ME011 + ME111/3.0 + 2.0/3.0*ME113)*pRho_be*pRho_cf +
                                                1.0/3.0 * (2.0*ME111 + ME113)*nRho_be*pRho_cf)
                            end
                        end
                    end
                end

                ph_N_local[ad_i,task] += pSum_local
                nh_N_local[ad_i,task] += nSum_local
            end
        end

        # Include kinetic & CM contributions ...
        @inbounds for ad_i in 1:ad_count
            a, d = ad[ad_i]

            # Sum over the task-local accumulators for H ...
            phSum = sum(view(ph_N_local,ad_i,:))
            nhSum = sum(view(nh_N_local,ad_i,:))

            # Include the 1-body kinetic energy & inclusion of Center-of-Mass motion (CM) correction ...
                # Combined 1- + 2-body kinetic operator with CM correction ...
            if CMS == "CMS1+2B"
                phSum += T.p[a,d] * (1.0 - 1.0 / A)
                nhSum += T.n[a,d] * (1.0 - 1.0 / A)
                # Pure 1-body kinetic operator with no CM correction ...
            elseif CMS != "CMS2B"
                phSum += T.p[a,d]
                nhSum += T.n[a,d]
            end
                # No contribution for pure 2-body kinetic operator with CM correction ...

            # Allocate ph & nh ...
            if a != d
                ph_N[a,d], ph_N[d,a] = phSum, phSum
                nh_N[a,d], nh_N[d,a] = nhSum, nhSum
            elseif a == d
                ph_N[a,a], nh_N[a,a] = phSum, nhSum
            end
        end

    println("\tCompleted the calculation of 1-body mean-field Hamiltonian h_N ...")

    return O1B(ph_N,nh_N)
end

function H2b(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,Orb_NNN::Orb3B,V_NN::O2B,V_NNN::Array{Vector{Vector{Float32}},4},Rho::O1B;Report::Bool=true)
    # Read parameters ...
    N_max = Params.Calc.Nmax
    N_2max = Params.Calc.N2max
    a_max = div((N_max + 1)*(N_max + 2),2)
    J_max = N_2max + 1
    cRes = Params.Int.cRes

    if Report
        println("\nCalculating density dependent residual 2-body interaction Hamiltonian H_NN ... (NO2B approximation of V_NNN)")
    end

    function H2b_lj_pairs(Params::Parameters,Orb::Vector{Orb1B})
        # Read # initialize parameters ... (local, otherwise N_max & a_max of H2b are overwritten & become type-unstable ...)
        local N_max = Params.Calc.Nmax
        local a_max = div((N_max + 1)*(N_max + 2),2)

        # All ordered pairs cf with l_c = l_f & j_c = j_f ... only these are connected by the lj-diagonal Rho ...
        cf_all = NTuple{2,Int64}[]
        @inbounds for Orb_c in Orb
            @inbounds for Orb_f in Orb
                if Orb_c.l == Orb_f.l && Orb_c.j == Orb_f.j
                    push!(cf_all,(Orb_c.a,Orb_f.a))
                end
            end
        end

        return cf_all
    end

    @inline function H2b_V3b_index(Bra::Int64,Ket::Int64)
        # Index of (Bra|V_NNN|Ket) in the stored lower triangle of a V_NNN block (see V3b_no2b_index) ...
        if Bra >= Ket
            return Ket + div(Bra*(Bra - 1),2)
        else
            return Bra + div(Ket*(Ket - 1),2)
        end
    end

    # Allocate pairs of orbitals with equal l & j ...
    cf = H2b_lj_pairs(Params,Orb)

    # Index of the orbital c in V_NNN blocks [P,l_c+1,J+1,T][j_index] ... 1 for j_c = l_c - 1/2, 2 for j_c = l_c + 1/2 ...
    j_index = zeros(Int64,a_max)
    @inbounds for Orb_c in Orb
        c, l_c, j_c = Orb_c.a, Orb_c.l, Orb_c.j
        if j_c == abs(2*l_c- 1)
            j_index[c] = 1
        else
            j_index[c] = 2
        end
    end

    # Start from the 2-body interaction V_NN ...
    H_NN = deepcopy(V_NN)

    # Include V_NNN in the NO2B approximation ...
    @inbounds for J in 0:J_max
        @inbounds for P in 1:2
            if Report
                if P == 1
                    println("\tCalculating   ...   J = " * string(J) * "/" * string(J_max) * "\tP = +")
                else
                    println("\tCalculating   ...   J = " * string(J) * "/" * string(J_max) * "\tP = -")
                end
            end

            # Number of pairs ab with a >= b (pp & nn) ...
            N_pp = Orb_NN.N[2,P,J+1]
            is_J_hat = 1.0 / hat(2*J)

            # Pairs a >= b are distributed among threads ... every Bra writes only its own elements (Bra|V|Ket <= Bra) ...
            @inbounds Threads.@threads :dynamic for Bra in 1:N_pp
                a, b = Orb_NN.Ind[2,P,J+1][Bra][1], Orb_NN.Ind[2,P,J+1][Bra][2]
                l_a, j_a = Orb[a].l, Orb[a].j
                l_b, j_b = Orb[b].l, Orb[b].j

                # Phase of V_NNN for (b a T_ab = 0| = (-1)^((j_a + j_b)/2 - J) (a b T_ab = 0| ... opposite for T_ab = 1 ...
                Phase_ab = phase(div(j_a + j_b,2) - J)

                # Lookup of the 3-body states (ab T_ab c) in V_NNN ... Map_ab[T_ab + T,J - J_min + 1,c] (see V3b_no2b_initialize) ...
                Map_ab = Orb_NNN.Map[div(a*(a-1),2) + b]
                J_ab = J - div(abs(j_a - j_b),2) + 1

                @inbounds for Ket in 1:Bra
                    d, e = Orb_NN.Ind[2,P,J+1][Ket][1], Orb_NN.Ind[2,P,J+1][Ket][2]
                    j_d, j_e = Orb[d].j, Orb[e].j

                    Phase_de = phase(div(j_d + j_e,2) - J)

                    Map_de = Orb_NNN.Map[div(d*(d-1),2) + e]
                    J_de = J - div(abs(j_d - j_e),2) + 1

                    ppSum, nnSum = 0.0, 0.0
                    pnSum_abde, pnSum_bade, pnSum_abed, pnSum_baed = 0.0, 0.0, 0.0, 0.0

                    @inbounds for (c,f) in cf
                        # Index of (ab T_ab c| & |de T_de f) in V_NNN ... 0 if not allowed by N_3max ...
                        abc_0, def_0 = Int64(Map_ab[1,J_ab,c]), Int64(Map_de[1,J_de,f])

                        if abc_0 == 0 || def_0 == 0
                            continue
                        end

                        abc_1, def_1 = Int64(Map_ab[2,J_ab,c]), Int64(Map_de[2,J_de,f])
                        abc_3, def_3 = Int64(Map_ab[3,J_ab,c]), Int64(Map_de[3,J_de,f])

                        l_c = Orb[c].l
                        P3B = parity(l_a + l_b + l_c)
                        V_T1, V_T3 = V_NNN[P3B,l_c+1,J+1,1][j_index[c]], V_NNN[P3B,l_c+1,J+1,2][j_index[c]]

                        ME001 = Float64(V_T1[H2b_V3b_index(abc_0,def_0)])
                        ME101 = Float64(V_T1[H2b_V3b_index(abc_1,def_0)])
                        ME011 = Float64(V_T1[H2b_V3b_index(abc_0,def_1)])
                        ME111 = Float64(V_T1[H2b_V3b_index(abc_1,def_1)])
                        ME113 = Float64(V_T3[H2b_V3b_index(abc_3,def_3)])

                        pRho_fc, nRho_fc = Rho.p[f,c], Rho.n[f,c]

                        ppSum += ME113 * pRho_fc + (2.0 * ME111 + ME113) / 3.0 * nRho_fc
                        nnSum += ME113 * nRho_fc + (2.0 * ME111 + ME113) / 3.0 * pRho_fc

                        pnSum_abde += (0.5 * ME001 + ME111 / 6.0 + ME113 / 3.0) * (pRho_fc + nRho_fc) + ( ME101 + ME011) / sqrt(12.0) * (nRho_fc - pRho_fc)
                        pnSum_bade += (0.5 * ME001 - ME111 / 6.0 - ME113 / 3.0) * (pRho_fc + nRho_fc) + (-ME101 + ME011) / sqrt(12.0) * (nRho_fc - pRho_fc)
                        pnSum_abed += (0.5 * ME001 - ME111 / 6.0 - ME113 / 3.0) * (pRho_fc + nRho_fc) + ( ME101 - ME011) / sqrt(12.0) * (nRho_fc - pRho_fc)
                        pnSum_baed += (0.5 * ME001 + ME111 / 6.0 + ME113 / 3.0) * (pRho_fc + nRho_fc) - ( ME101 + ME011) / sqrt(12.0) * (nRho_fc - pRho_fc)
                    end

                    # Allocate the pp & nn elements ...
                    Ind = O2b_index(a,b,d,e,J,P,1,Orb_NN)
                    H_NN.pp[P,J+1][Ind] = cRes * (H_NN.pp[P,J+1][Ind] + is_J_hat * ppSum)
                    H_NN.nn[P,J+1][Ind] = cRes * (H_NN.nn[P,J+1][Ind] + is_J_hat * nnSum)

                    # Allocate the pn elements of all orderings ... (ba|V|de) & (ab|V|ed) are the same element for ab = de ...
                    Ind = O2b_index(a,b,d,e,J,P,0,Orb_NN)
                    H_NN.pn[P,J+1][Ind] = cRes * (H_NN.pn[P,J+1][Ind] + is_J_hat * pnSum_abde)

                    if a != b
                        Ind = O2b_index(b,a,d,e,J,P,0,Orb_NN)
                        H_NN.pn[P,J+1][Ind] = cRes * (H_NN.pn[P,J+1][Ind] + is_J_hat * Phase_ab * pnSum_bade)
                    end

                    if d != e && Ket != Bra
                        Ind = O2b_index(a,b,e,d,J,P,0,Orb_NN)
                        H_NN.pn[P,J+1][Ind] = cRes * (H_NN.pn[P,J+1][Ind] + is_J_hat * Phase_de * pnSum_abed)
                    end

                    if a != b && d != e
                        Ind = O2b_index(b,a,e,d,J,P,0,Orb_NN)
                        H_NN.pn[P,J+1][Ind] = cRes * (H_NN.pn[P,J+1][Ind] + is_J_hat * Phase_ab * Phase_de * pnSum_baed)
                    end
                end
            end
        end
    end

    if Report
        println("\tCompleted the calculation of density dependent residual 2-body interaction Hamiltonian H_NN ...")
    end

    return H_NN
end

function E1b(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,Orb_NNN::Orb3B,T::O1B,V_NN::O2B,V_NNN::Array{Vector{Vector{Float32}},4},Rho::O1B)
    # Read parameters ...
    A = Float64(Params.Calc.A)
    N_max = Params.Calc.Nmax
    N_2max = Params.Calc.N2max
    N_3max = Params.Calc.N3max
    a_max = div((N_max + 1)*(N_max + 2),2)
    CMS = Params.Calc.CMS

    println("\nCalculating the ground-state energy due to the 1-body density matrix Rho ...")

    # Number of work packages ... more than threads to balance the load ...
    N_task = 4*Threads.nthreads()

    function E1b_lj_pairs(Params::Parameters,Orb::Vector{Orb1B})
        # Read # initialize parameters ... (local, otherwise N_max & a_max of E1b are overwritten & become type-unstable ...)
        local N_max = Params.Calc.Nmax
        local a_max = div((N_max + 1)*(N_max + 2),2)

        # All ordered pairs ad with l_a = l_d & j_a = j_d ... only these are connected by the lj-diagonal Rho ...
        ad_all = NTuple{2,Int64}[]
        @inbounds for Orb_a in Orb
            @inbounds for Orb_d in Orb
                if Orb_a.l == Orb_d.l && Orb_a.j == Orb_d.j
                    push!(ad_all,(Orb_a.a,Orb_d.a))
                end
            end
        end

        return ad_all
    end

    # Allocate pairs of orbitals with equal l & j ...
    lj_pairs = E1b_lj_pairs(Params,Orb)
    ad, be, cf = lj_pairs, lj_pairs, lj_pairs

    # 1-body kinetic energy ... E_T = sum_ad j_a_hat (pRho_ad pT_ad + nRho_ad nT_ad) ...
    E_T = 0.0
    @inbounds for (a, d) in ad
        E_T += hat(Orb[a].j) * (Rho.p[a,d] * T.p[a,d] + Rho.n[a,d] * T.n[a,d])
    end

    # Include the Center-of-Mass motion (CM) correction ...
        # Combined 1- + 2-body kinetic operator with CM correction ...
    if CMS == "CMS1+2B"
        E_T *= (1.0 - 1.0 / A)
        # Pure 2-body kinetic operator with CM correction ... no 1-body contribution ...
    elseif CMS == "CMS2B"
        E_T = 0.0
    end
        # Pure 1-body kinetic operator with no CM correction ... E_T as it is ...

    # 2- & 3-body energy ... sum over all (ad, be) combinations ...
    #   E_NN  = 1/2 sum_J J_hat sum_abde (ppV_ab,de pRho_ad pRho_be + nnV_ab,de nRho_ad nRho_be + 2 pnV_ab,de pRho_ad nRho_be) ...
    #   E_NNN = 1/6 sum_J sum_abcdef (abc|V_NNN|def) Rho_ad Rho_be Rho_cf ... charge contents ppp, ppn, pnn & nnn, with the odd particle
    #           in the positions a, b & c taken together (factor 3) ...
    # List of all (ad, be) combinations ... used to distribute the work among tasks ...
    adbe_tasks = NTuple{2,Int64}[]
    @inbounds for ad_i in 1:length(ad)
        @inbounds for be_i in 1:length(be)
            push!(adbe_tasks,(ad_i,be_i))
        end
    end

    # Task-local accumulators ...
    E_NN_task, E_NNN_task = zeros(Float64,N_task), zeros(Float64,N_task)

    # Each task takes every N_task-th (ad, be) combination ...
    @inbounds Threads.@threads :dynamic for task in 1:N_task
        E_NN_local, E_NNN_local = 0.0, 0.0

        @inbounds for task_i in task:N_task:length(adbe_tasks)
            ad_i, be_i = adbe_tasks[task_i]
            a, d = ad[ad_i]
            b, e = be[be_i]
            n_a, l_a, j_a = Orb[a].n, Orb[a].l, Orb[a].j
            n_b, l_b, j_b = Orb[b].n, Orb[b].l, Orb[b].j
            n_d, l_d = Orb[d].n, Orb[d].l
            n_e, l_e = Orb[e].n, Orb[e].l

            if (2*(n_a + n_b) + l_a + l_b) > N_2max || (2*(n_d + n_e) + l_d + l_e) > N_2max
                continue
            end

            P_ab = parity(l_a + l_b)
            pRho_ad, nRho_ad = Rho.p[a,d], Rho.n[a,d]
            pRho_be, nRho_be = Rho.p[b,e], Rho.n[b,e]

            @inbounds for J in div(abs(j_a - j_b),2):div(j_a + j_b,2)

                # 2-body NN interaction part ...
                if P_ab == parity(l_d + l_e)
                    J_hat = hat(2*J)
                    E_NN_local += 0.5 * J_hat * pRho_ad * pRho_be * O2b_pp(a,b,d,e,J,P_ab,V_NN,Orb,Orb_NN)
                    E_NN_local += 0.5 * J_hat * nRho_ad * nRho_be * O2b_nn(a,b,d,e,J,P_ab,V_NN,Orb,Orb_NN)
                    E_NN_local += J_hat * pRho_ad * nRho_be * O2b_pn(a,b,d,e,J,P_ab,V_NN,Orb_NN)
                end

                # 3-body NNN interaction part ... only cf with l_c = l_f & j_c = j_f ...
                @inbounds for (c, f) in cf
                    n_c, l_c = Orb[c].n, Orb[c].l
                    n_f, l_f = Orb[f].n, Orb[f].l
                    P_abc = parity(l_a + l_b + l_c)

                    if (2*(n_a + n_b + n_c) + l_a + l_b + l_c) > N_3max || (2*(n_d + n_e + n_f) + l_d + l_e + l_f) > N_3max ||
                       P_abc != parity(l_d + l_e + l_f)
                        continue
                    end

                    pRho_cf, nRho_cf = Rho.p[c,f], Rho.n[c,f]

                    ME001 = V3b_no2b(a,b,c,0,d,e,f,0,J,1,P_abc,V_NNN,Orb,Orb_NNN)
                    ME101 = V3b_no2b(a,b,c,1,d,e,f,0,J,1,P_abc,V_NNN,Orb,Orb_NNN)
                    ME011 = V3b_no2b(a,b,c,0,d,e,f,1,J,1,P_abc,V_NNN,Orb,Orb_NNN)
                    ME111 = V3b_no2b(a,b,c,1,d,e,f,1,J,1,P_abc,V_NNN,Orb,Orb_NNN)
                    ME113 = V3b_no2b(a,b,c,1,d,e,f,1,J,3,P_abc,V_NNN,Orb,Orb_NNN)

                    # ppp & nnn ... T = 3/2 only, ppn ... pair pp with c a neutron (x 3),
                    # pnn ... pair pn with c a neutron (x 3), <1 0 1/2 -1/2|1/2 -1/2> = +sqrt(1/3) ...
                    E_NNN_local += 1.0 / 6.0 * (ME113 * pRho_ad * pRho_be * pRho_cf +
                                                (2.0 * ME111 + ME113) * pRho_ad * pRho_be * nRho_cf +
                                                (1.5 * ME001 + sqrt(3.0/4.0) * (ME101 + ME011) + 0.5 * ME111 + ME113) * pRho_ad * nRho_be * nRho_cf +
                                                ME113 * nRho_ad * nRho_be * nRho_cf)
                end
            end
        end

        E_NN_task[task] = E_NN_local
        E_NNN_task[task] = E_NNN_local
    end

    # Sum over the task-local accumulators ...
    E_NN, E_NNN = sum(E_NN_task), sum(E_NNN_task)
    E = E_T + E_NN + E_NNN

    println("\tKinetic energy  ...   E_T   = " * string(round(E_T, digits=8)) * " MeV")
    println("\tNN energy       ...   E_NN  = " * string(round(E_NN, digits=8)) * " MeV")
    println("\tNNN energy      ...   E_NNN = " * string(round(E_NNN, digits=8)) * " MeV")
    println("\tTotal energy    ...   E     = " * string(round(E, digits=8)) * " MeV")

    return E
end

function E2b(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,V_NN::O2B,Sigma::O2B)
    # Read parameters ...
    N_2max = Params.Calc.N2max
    J_max = N_2max + 1

    # Calculate the correlation energy ...
    println("\nComputing the correlation ground-state energy by contraction with the provided 2-body correlation function ...")

    # Number of work packages ... more than threads to balance the load ...
    N_task = 4*Threads.nthreads()

    # E_corr = sum_J J_hat sum_ab,cd [ 1/4 (ppV_ab,cd ppSigma_ab,cd + nnV_ab,cd nnSigma_ab,cd) + pnV_ab,cd pnSigma_ab,cd ] ... over all ordered pairs ab & cd ...
    #
    #   Only the stored elements (Bra|O|Ket <= Bra) are visited ...
    #   pp & nn ... stored for a >= b, the orderings ba (dc) give the same product, as the phases of V & Sigma cancel ... factor 2 for a != b (c != d),
    #   (Ket|O|Bra) gives the same product as (Bra|O|Ket) ... factor 2 for Ket != Bra ...

    # List of all (P, J, Bra) ... used to distribute the work among tasks (Bra runs over the pn pairs, the pp pairs are fewer) ...
    Bra_tasks = NTuple{3,Int64}[]
    @inbounds for J in 0:J_max
        @inbounds for P in 1:2
            @inbounds for Bra in 1:Orb_NN.N[1,P,J+1]
                push!(Bra_tasks,(P,J,Bra))
            end
        end
    end

    # Task-local accumulators ...
    E_corr_task = zeros(Float64,N_task)

    # Each task takes every N_task-th (P, J, Bra) ...
    @inbounds Threads.@threads :dynamic for task in 1:N_task
        E_corr_local = 0.0

        @inbounds for task_i in task:N_task:length(Bra_tasks)
            P, J, Bra = Bra_tasks[task_i]
            J_hat = hat(2*J)

            # pp & nn part ... pairs a >= b ...
            if Bra <= Orb_NN.N[2,P,J+1]
                a, b = Orb_NN.Ind[2,P,J+1][Bra][1], Orb_NN.Ind[2,P,J+1][Bra][2]

                @inbounds for Ket in 1:Bra
                    c, d = Orb_NN.Ind[2,P,J+1][Ket][1], Orb_NN.Ind[2,P,J+1][Ket][2]

                    # Number of equivalent elements ... orderings ab & ba, cd & dc, (Bra|Ket) & (Ket|Bra) ...
                    N_equal = 1.0
                    if a != b
                        N_equal *= 2.0
                    end
                    if c != d
                        N_equal *= 2.0
                    end
                    if Ket != Bra
                        N_equal *= 2.0
                    end

                    E_corr_local += 0.25 * N_equal * J_hat * (O2b_pp(a,b,c,d,J,P,V_NN,Orb,Orb_NN) * O2b_pp(a,b,c,d,J,P,Sigma,Orb,Orb_NN) +
                                                              O2b_nn(a,b,c,d,J,P,V_NN,Orb,Orb_NN) * O2b_nn(a,b,c,d,J,P,Sigma,Orb,Orb_NN))
                end
            end

            # pn part ... all ordered pairs ab ...
            a, b = Orb_NN.Ind[1,P,J+1][Bra][1], Orb_NN.Ind[1,P,J+1][Bra][2]

            @inbounds for Ket in 1:Bra
                c, d = Orb_NN.Ind[1,P,J+1][Ket][1], Orb_NN.Ind[1,P,J+1][Ket][2]

                # Number of equivalent elements ... (Bra|Ket) & (Ket|Bra) ...
                N_equal = 1.0

                if Ket != Bra
                    N_equal *= 2.0
                end

                E_corr_local += N_equal * J_hat * O2b_pn(a,b,c,d,J,P,V_NN,Orb_NN) * O2b_pn(a,b,c,d,J,P,Sigma,Orb_NN)
            end
        end

        E_corr_task[task] = E_corr_local
    end

    # Accumulate the total correlation energy ...
    E_corr = sum(E_corr_task)

    println("\tPerformed calculation of correlation energy by contraction with 2-body correlation function ...\n")

    return E_corr
end

function H1b_H2b(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,Orb_NNN::Orb3B,T::O1B,V_NN::O2B,V_NNN::Array{Vector{Vector{Float32}},4},Rho::O1B,Sigma::O2B;Report::Bool = true)
    # Read parameters ...
    A = Float64(Params.Calc.A)
    N_max = Params.Calc.Nmax
    N_2max = Params.Calc.N2max
    N_3max = Params.Calc.N3max
    a_max = div((N_max + 1)*(N_max + 2),2)
    J_max = N_2max + 1
    CMS = Params.Calc.CMS

    if Report
        println("\nEvaluating 2-body Hamiltonian H_NN ...")
    end

    # First make density-dependent 2-body Hamiltonian ...
    H_NN = H2b(Params,Orb,Orb_NN,Orb_NNN,V_NN,V_NNN,Rho,Report=Report)

    if Report
        println("\nEvaluating 1-body mean-field Hamiltonian h_N ...")
    end

    # Number of work packages ... more than threads to balance the load ...
    N_task = 4*Threads.nthreads()

    function H1b_lj_pairs(Params::Parameters,Orb::Vector{Orb1B})
        # Read # initialize parameters ... (local, otherwise N_max & a_max of H1b are overwritten & become type-unstable ...)
        local N_max = Params.Calc.Nmax
        local a_max = div((N_max + 1)*(N_max + 2),2)

        # All ordered pairs ab & pairs ab with a >= b ... both with l_a = l_b & j_a = j_b ...
        ab_all, ab_low = NTuple{2,Int64}[], NTuple{2,Int64}[]
        @inbounds for a in 1:a_max
            l_a, j_a = Orb[a].l, Orb[a].j
            @inbounds for b in 1:a_max
                l_b, j_b = Orb[b].l, Orb[b].j
                if l_a == l_b && j_a == j_b
                    push!(ab_all,(a,b))
                    if b <= a
                        push!(ab_low,(a,b))
                    end
                end
            end
        end

        return ab_all, ab_low
    end

    function H1b_unpack(O_PJ::Vector{Float64},N::Int64)
        # Unpack the stored lower triangle of one block (P,J) of a 2-body operator (see O2b_index) into a dense symmetric matrix ...
        O_dense = Matrix{Float64}(undef,N,N)
        @inbounds Threads.@threads :dynamic for Ket in 1:N
            Offset = (Ket - 1) * N - div(Ket*(Ket - 1),2)
            @inbounds for Bra in Ket:N
                O_dense[Bra,Ket] = O_PJ[Bra + Offset]
                O_dense[Ket,Bra] = O_PJ[Bra + Offset]
            end
        end
        return O_dense
    end

    function H1b_NNN_triples(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,Orb_NNN::Orb3B)
        # Read # initialize parameters ... (local, otherwise the parameters of H1b are overwritten ...)
        local N_max, N_2max = Params.Calc.Nmax, Params.Calc.N2max
        local a_max = div((N_max + 1)*(N_max + 2),2)
        local J_max = N_2max + 1
        l_max = N_max

        # 3-body triples (a b c) with a >= b of every V_NNN block [P,l_c+1,J+1][j_c] ... stored as (a, b, c, ab, i_0, i_1, i_3),
        #   ab  ... index of the pair a >= b in Orb_NN (Orb_NN.Map[a,b,J+1,2]),
        #   i_0 ... index of (a b, T_ab = 0, c) in V_NNN[P,l_c+1,J+1,1][j_c] (T = 1/2),
        #   i_1 ... index of (a b, T_ab = 1, c) in V_NNN[P,l_c+1,J+1,1][j_c] (T = 1/2),
        #   i_3 ... index of (a b, T_ab = 1, c) in V_NNN[P,l_c+1,J+1,2][j_c] (T = 3/2) ... (see Map_ab[T_ab + T,J - J_min + 1,c] in V3b_no2b_initialize)
        local NNN_triples = Array{Vector{Vector{NTuple{7,Int64}}}}(undef,2,l_max+1,J_max+1)
        @inbounds for P in 1:2
            @inbounds for l in 0:l_max
                @inbounds for J in 0:J_max
                    NNN_triples[P,l+1,J+1] = [NTuple{7,Int64}[] for j_index in 1:length(Orb_NNN.N[P,l+1,J+1,1])]
                end
            end
        end

        @inbounds for a in 1:a_max
            l_a, j_a = Orb[a].l, Orb[a].j
            @inbounds for b in 1:a
                l_b, j_b = Orb[b].l, Orb[b].j
                Map_ab = Orb_NNN.Map[div(a*(a-1),2) + b]
                J_min = div(abs(j_a - j_b),2)
                @inbounds for J_i in 1:size(Map_ab,2)
                    J = J_min + J_i - 1
                    ab = Int64(Orb_NN.Map[a,b,J+1,2])
                    @inbounds for c in 1:a_max
                        l_c, j_c = Orb[c].l, Orb[c].j
                        if Map_ab[1,J_i,c] == 0
                            continue
                        end
                        P = parity(l_a + l_b + l_c)
                        if j_c == abs(2*l_c - 1)
                            j_index = 1
                        else
                            j_index = 2
                        end
                        push!(NNN_triples[P,l_c+1,J+1][j_index],(a,b,c,ab,Int64(Map_ab[1,J_i,c]),Int64(Map_ab[2,J_i,c]),Int64(Map_ab[3,J_i,c])))
                    end
                end
            end
        end

        return NNN_triples
    end

    function H1b_V3b_index(Bra::Int64,Ket::Int64)
        # Index of (Bra|V_NNN|Ket) in the stored lower triangle of a V_NNN block (see V3b_no2b_index) ...
        if Bra >= Ket
            return Ket + div(Bra*(Bra - 1),2)
        else
            return Bra + div(Ket*(Ket - 1),2)
        end
    end

    # Allocate pairs of orbitals with equal l & j ...
    lj_pairs, lj_pairs_low = H1b_lj_pairs(Params,Orb)
    ad, ad_count = lj_pairs_low, length(lj_pairs_low)
    be, be_count = lj_pairs, length(lj_pairs)
    cf = lj_pairs

    # Allocate new 1-body Hamiltonian matrices ...
    ph_N, nh_N = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)

    # Allocate the single-particle field H ...
        # Due to 1-body contractions with the 1-body density Rho ...
            # List of all (ad, be) combinations ... used to distribute the work among tasks ...
        adbe_tasks = NTuple{2,Int64}[]
        @inbounds for ad_i in 1:ad_count
            @inbounds for be_i in 1:be_count
                push!(adbe_tasks,(ad_i,be_i))
            end
        end

        # Initialize the thread local 1-body Hamiltonian matrices ...
        ph_N_local = zeros(Float64,ad_count,N_task)
        nh_N_local = zeros(Float64,ad_count,N_task)

        # Each task takes every N_task-th (ad, be) combination ...
        @inbounds Threads.@threads :dynamic for task in 1:N_task
            @inbounds for task_i in task:N_task:length(adbe_tasks)
                ad_i, be_i = adbe_tasks[task_i]
                a, d = ad[ad_i]
                b, e = be[be_i]
                n_a, l_a, j_a = Orb[a].n, Orb[a].l, Orb[a].j
                n_d, l_d, j_d = Orb[d].n, Orb[d].l, Orb[d].j
                n_b, l_b, j_b = Orb[b].n, Orb[b].l, Orb[b].j
                n_e, l_e, j_e = Orb[e].n, Orb[e].l, Orb[e].j
                is_j_a_hat = 1.0 / hat(j_a)
                P_ab = parity(l_a + l_b)
                pSum_local, nSum_local = 0.0, 0.0

                if (2*(n_a + n_b) + l_a + l_b) <= N_2max && (2*(n_d + n_e) + l_d + l_e) <= N_2max
                    pRho_be, nRho_be = Rho.p[b,e], Rho.n[b,e]

                    @inbounds for J = div(abs(j_d - j_e),2):div(j_d + j_e,2)

                        # 2-body NN interaction part ...
                        if P_ab == parity(l_d + l_e)
                            J_j_a_hat = hat(2*J) * is_j_a_hat
                            pSum_local += J_j_a_hat * pRho_be * O2b_pp(a,b,d,e,J,P_ab,V_NN,Orb,Orb_NN)
                            pSum_local += J_j_a_hat * nRho_be * O2b_pn(a,b,d,e,J,P_ab,V_NN,Orb_NN)
                            nSum_local += J_j_a_hat * nRho_be * O2b_nn(a,b,d,e,J,P_ab,V_NN,Orb,Orb_NN)
                            nSum_local += J_j_a_hat * pRho_be * O2b_pn(b,a,e,d,J,P_ab,V_NN,Orb_NN)
                        end

                        # 3-body NNN interaction part ... only cf with l_c = l_f & j_c = j_f ...
                        @inbounds for (c, f) in cf
                            n_c, l_c = Orb[c].n, Orb[c].l
                            n_f, l_f = Orb[f].n, Orb[f].l
                            P_abc = parity(l_a + l_b + l_c)
                            if (2*(n_a + n_b + n_c) + l_a + l_b + l_c) <= N_3max && (2*(n_d + n_e + n_f) + l_d + l_e + l_f) <= N_3max &&
                                P_abc == parity(l_d + l_e + l_f)
                                pRho_cf, nRho_cf = Rho.p[c,f], Rho.n[c,f]

                                ME001 = V3b_no2b(a,b,c,0,d,e,f,0,J,1,P_abc,V_NNN,Orb,Orb_NNN)
                                ME101 = V3b_no2b(a,b,c,1,d,e,f,0,J,1,P_abc,V_NNN,Orb,Orb_NNN)
                                ME011 = V3b_no2b(a,b,c,0,d,e,f,1,J,1,P_abc,V_NNN,Orb,Orb_NNN)
                                ME111 = V3b_no2b(a,b,c,1,d,e,f,1,J,1,P_abc,V_NNN,Orb,Orb_NNN)
                                ME113 = V3b_no2b(a,b,c,1,d,e,f,1,J,3,P_abc,V_NNN,Orb,Orb_NNN)

                                pSum_local += is_j_a_hat * (0.5*ME113*pRho_be*pRho_cf + 0.25 * (ME001 +
                                                sqrt(1.0/3.0)*ME101 + sqrt(1.0/3.0)*ME011 + ME111/3.0 + 2.0/3.0*ME113)*nRho_be*nRho_cf +
                                                1.0/3.0 * (2.0*ME111 + ME113)*pRho_be*nRho_cf)

                                nSum_local += is_j_a_hat * (0.5*ME113*nRho_be*nRho_cf + 0.25 * (ME001 +
                                                sqrt(1.0/3.0)*ME101 + sqrt(1.0/3.0)*ME011 + ME111/3.0 + 2.0/3.0*ME113)*pRho_be*pRho_cf +
                                                1.0/3.0 * (2.0*ME111 + ME113)*nRho_be*pRho_cf)
                            end
                        end
                    end
                end

                ph_N_local[ad_i,task] += pSum_local
                nh_N_local[ad_i,task] += nSum_local
            end
        end

        # Include kinetic & CM contributions ...
        @inbounds for ad_i in 1:ad_count
            a, d = ad[ad_i]

            # Sum over the task-local accumulators for H ...
            phSum = sum(view(ph_N_local,ad_i,:))
            nhSum = sum(view(nh_N_local,ad_i,:))

            # Include the 1-body kinetic energy & inclusion of Center-of-Mass motion (CM) correction ...
                # Combined 1- + 2-body kinetic operator with CM correction ...
            if CMS == "CMS1+2B"
                phSum += T.p[a,d] * (1.0 - 1.0 / A)
                nhSum += T.n[a,d] * (1.0 - 1.0 / A)
                # Pure 1-body kinetic operator with no CM correction ...
            elseif CMS != "CMS2B"
                phSum += T.p[a,d]
                nhSum += T.n[a,d]
            end
                # No contribution for pure 2-body kinetic operator with CM correction ...

            # Allocate ph & nh ...
            if a != d
                ph_N[a,d], ph_N[d,a] = phSum, phSum
                nh_N[a,d], nh_N[d,a] = nhSum, nhSum
            elseif a == d
                ph_N[a,a], nh_N[a,a] = phSum, nhSum
            end
        end

    # Due to 2-body contractions with the 2-body correlation function Sigma ...
        # h_cf = 1/4 1/j_c_hat sum_J sum_ab sum_de (ab c|V_NNN|de f) Sigma_ab,de ... sums over all ordered pairs ab & de, isospin written out below ...
        # V_NNN is stored only for a >= b & d >= e, per block [P,l_c+1,J+1,T][j_c] (fixed l_c = l_f & j_c = j_f) as a lower triangle ...
        # so the loop runs over pairs of triples (a b c) >= (d e f) of a block & the orderings ba & ed are added through their phases ...
        NNN_triples = H1b_NNN_triples(Params,Orb,Orb_NN,Orb_NNN)

        # Task-local accumulators for h_cf ...
        ph_N_task = [zeros(Float64,a_max,a_max) for task in 1:N_task]
        nh_N_task = [zeros(Float64,a_max,a_max) for task in 1:N_task]

        @inbounds for J in 0:J_max
            @inbounds for P2B in 1:2
                N_ab, N_ab_pn = Orb_NN.N[2,P2B,J+1], Orb_NN.N[1,P2B,J+1]
                if N_ab == 0
                    continue
                end

                # Dense Sigma for this J & P_ab ...
                #   pp & nn for pairs a >= b ([de,ab], Orb_NN.Map[a,b,J+1,2]),
                #   pn for all ordered pairs ([de,ab], Orb_NN.Map[a,b,J+1,1]) ...
                ppSigma_JP = H1b_unpack(Sigma.pp[P2B,J+1],N_ab)
                pnSigma_JP = H1b_unpack(Sigma.pn[P2B,J+1],N_ab_pn)
                nnSigma_JP = H1b_unpack(Sigma.nn[P2B,J+1],N_ab)

                # Each task takes every N_task-th triple (a b c) of every V_NNN block with this J & P_ab ...
                @inbounds Threads.@threads :dynamic for task in 1:N_task
                    ph_task, nh_task = ph_N_task[task], nh_N_task[task]

                    @inbounds for l_c in 0:N_max
                        P3B = parity(P2B - 1 + l_c)

                        @inbounds for (j_index, Triples) in enumerate(NNN_triples[P3B,l_c+1,J+1])
                            V_T1, V_T3 = V_NNN[P3B,l_c+1,J+1,1][j_index], V_NNN[P3B,l_c+1,J+1,2][j_index]

                            @inbounds for abc in task:N_task:length(Triples)
                                a, b, c, ab, abc_0, abc_1, abc_3 = Triples[abc]
                                ab_pn, ba_pn = Int64(Orb_NN.Map[a,b,J+1,1]), Int64(Orb_NN.Map[b,a,J+1,1])

                                # Phase of V_NNN for (b a T_ab = 0| = (-1)^((j_a + j_b)/2 - J) (a b T_ab = 0| ... opposite for T_ab = 1 ...
                                Phase_ab = phase(div(Orb[a].j + Orb[b].j,2) - J)

                                @inbounds for def in 1:abc
                                    d, e, f, de, def_0, def_1, def_3 = Triples[def]
                                    de_pn, ed_pn = Int64(Orb_NN.Map[d,e,J+1,1]), Int64(Orb_NN.Map[e,d,J+1,1])
                                    Phase_de = phase(div(Orb[d].j + Orb[e].j,2) - J)

                                    # 3-body matrix elements (ab T_ab c|V_NNN|de T_de f) with T = 1/2 (ME_T_ab_T_de_1) & T = 3/2 (ME113) ...
                                    ME001 = Float64(V_T1[H1b_V3b_index(abc_0,def_0)])
                                    ME101 = Float64(V_T1[H1b_V3b_index(abc_1,def_0)])
                                    ME011 = Float64(V_T1[H1b_V3b_index(abc_0,def_1)])
                                    ME111 = Float64(V_T1[H1b_V3b_index(abc_1,def_1)])
                                    ME113 = Float64(V_T3[H1b_V3b_index(abc_3,def_3)])

                                    # pp & nn Sigma ... the phases of ba (ed) in V_NNN & Sigma cancel, so every ordering adds the same ...
                                    ppSigma_abde, nnSigma_abde = ppSigma_JP[de,ab], nnSigma_JP[de,ab]
                                    if a != b
                                        ppSigma_abde, nnSigma_abde = 2.0 * ppSigma_abde, 2.0 * nnSigma_abde
                                    end
                                    if d != e
                                        ppSigma_abde, nnSigma_abde = 2.0 * ppSigma_abde, 2.0 * nnSigma_abde
                                    end

                                    # pn Sigma of all orderings ab, ba & de, ed ...
                                    pnSigma_abde = pnSigma_JP[de_pn,ab_pn]
                                    pnSigma_bade, pnSigma_abed, pnSigma_baed = 0.0, 0.0, 0.0
                                    if a != b
                                        pnSigma_bade = Phase_ab * pnSigma_JP[de_pn,ba_pn]
                                    end
                                    if d != e
                                        pnSigma_abed = Phase_de * pnSigma_JP[ed_pn,ab_pn]
                                    end
                                    if a != b && d != e
                                        pnSigma_baed = Phase_ab * Phase_de * pnSigma_JP[ed_pn,ba_pn]
                                    end

                                    # ... projected on the isospin T_ab & T_de of the pairs ab & de (T = 1 flips the sign of the swapped pair) ...
                                    pnSigma_00 = pnSigma_abde + pnSigma_bade + pnSigma_abed + pnSigma_baed
                                    pnSigma_10 = pnSigma_abde - pnSigma_bade + pnSigma_abed - pnSigma_baed
                                    pnSigma_01 = pnSigma_abde + pnSigma_bade - pnSigma_abed - pnSigma_baed
                                    pnSigma_11 = pnSigma_abde - pnSigma_bade - pnSigma_abed + pnSigma_baed

                                    # Proton field (c, f protons) ... <1 0 1/2 +1/2|1/2 +1/2> = -sqrt(1/3) ...
                                    pSum = ME113 * ppSigma_abde + (ME113 + 2.0 * ME111) / 3.0 * nnSigma_abde +
                                           2.0 * (ME001 * pnSigma_00 - (ME101 * pnSigma_10 + ME011 * pnSigma_01) / sqrt(3.0) + (ME111 + 2.0 * ME113) / 3.0 * pnSigma_11)

                                    # Neutron field (c, f neutrons) ... <1 0 1/2 -1/2|1/2 -1/2> = +sqrt(1/3) ...
                                    nSum = ME113 * nnSigma_abde + (ME113 + 2.0 * ME111) / 3.0 * ppSigma_abde +
                                           2.0 * (ME001 * pnSigma_00 + (ME101 * pnSigma_10 + ME011 * pnSigma_01) / sqrt(3.0) + (ME111 + 2.0 * ME113) / 3.0 * pnSigma_11)

                                    # (abc|V|def) adds to h_cf ... (def|V|abc), not stored, adds the same to h_fc ...
                                    ph_task[c,f] += pSum
                                    nh_task[c,f] += nSum

                                    if def != abc
                                        ph_task[f,c] += pSum
                                        nh_task[f,c] += nSum
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end

        # Sum over the task-local accumulators ...
        ph_cf, nh_cf = sum(ph_N_task), sum(nh_N_task)
        @inbounds for (c, f) in cf
            ph_N[c,f] += 0.25 / hat(Orb[c].j) * ph_cf[c,f]
            nh_N[c,f] += 0.25 / hat(Orb[c].j) * nh_cf[c,f]
        end

    #=
        # Due to 2-body contractions of V_NN with the 2-body correlation function Sigma ...
            function H1b_dense(Orb::Vector{Orb1B},Orb_NN::Orb2B,O::O2B,P::Int64,J::Int64)
                # Dense matrices ppO[de,ac] = <ac|O|de>_J (pnO, nnO similarly) ... ac & de run over all ordered pairs (the pn pairs, Orb_NN.Ind[1,P,J+1]) ...
                N_ac, N_ac_low = Orb_NN.N[1,P,J+1], Orb_NN.N[2,P,J+1]
                ac_pairs = Orb_NN.Ind[1,P,J+1]

                pnO = H1b_unpack(O.pn[P,J+1],N_ac)
                ppO_low, nnO_low = H1b_unpack(O.pp[P,J+1],N_ac_low), H1b_unpack(O.nn[P,J+1],N_ac_low)

                # pp & nn are stored only for a >= c ... pair ac with a < c is the stored pair ca times the phase (-1)^(J + 1 + (j_a + j_c)/2) (see O2b_pp) ...
                ac_low, Phase_ac = zeros(Int64,N_ac), zeros(Float64,N_ac)
                @inbounds for ac in 1:N_ac
                    a, c = ac_pairs[ac][1], ac_pairs[ac][2]
                    if a >= c
                        ac_low[ac] = Int64(Orb_NN.Map[a,c,J+1,2])
                        Phase_ac[ac] = 1.0
                    else
                        ac_low[ac] = Int64(Orb_NN.Map[c,a,J+1,2])
                        Phase_ac[ac] = phase(J + 1 + div(Orb[a].j + Orb[c].j,2))
                    end
                end

                ppO, nnO = Matrix{Float64}(undef,N_ac,N_ac), Matrix{Float64}(undef,N_ac,N_ac)
                @inbounds Threads.@threads :dynamic for ac in 1:N_ac
                    @inbounds for de in 1:N_ac
                        ppO[de,ac] = Phase_ac[de] * Phase_ac[ac] * ppO_low[ac_low[de],ac_low[ac]]
                        nnO[de,ac] = Phase_ac[de] * Phase_ac[ac] * nnO_low[ac_low[de],ac_low[ac]]
                    end
                end

                return ppO, pnO, nnO
            end
            
            ph_abc, nh_abc = zeros(Float64,a_max,a_max,a_max), zeros(Float64,a_max,a_max,a_max)

            # List of all (a, c) ... used to distribute the work among threads ...
            ac_tasks = NTuple{2,Int64}[]
            @inbounds for c in 1:a_max
                @inbounds for a in 1:a_max
                    push!(ac_tasks,(a,c))
                end
            end

            @inbounds for J in 0:J_max
                J_hat = hat(2*J)
                @inbounds for P in 1:2
                    N_de = Orb_NN.N[1,P,J+1]
                    if N_de == 0
                        continue
                    end

                    # V & Sigma as dense matrices ... e.g. ppV[de,ac] = <ac|V|de>_J for all ordered pairs ac & de with parity P ...
                    ppV, pnV, nnV = H1b_dense(Orb,Orb_NN,H_NN,P,J)
                    ppS, pnS, nnS = H1b_dense(Orb,Orb_NN,Sigma,P,J)

                    @inbounds Threads.@threads :dynamic for (a,c) in ac_tasks
                        l_a, j_a = Orb[a].l, Orb[a].j

                        # Index of the pairs ac & ca ... 0 if not allowed by J & N_2max ...
                        ac, ca = Int64(Orb_NN.Map[a,c,J+1,1]), Int64(Orb_NN.Map[c,a,J+1,1])
                        if ac == 0 || parity(l_a + Orb[c].l) != P
                            continue
                        end

                        @inbounds for b in 1:a_max
                            if Orb[b].l != l_a || Orb[b].j != j_a
                                continue
                            end

                            bc, cb = Int64(Orb_NN.Map[b,c,J+1,1]), Int64(Orb_NN.Map[c,b,J+1,1])
                            if bc == 0
                                continue
                            end

                            pSum, nSum = 0.0, 0.0

                            @inbounds @simd for de in 1:N_de
                                pSum += 0.5 * (ppV[de,ac] * ppS[de,bc] + ppV[de,bc] * ppS[de,ac]) + pnV[de,ac] * pnS[de,bc] + pnV[de,bc] * pnS[de,ac]
                                nSum += 0.5 * (nnV[de,ac] * nnS[de,bc] + nnV[de,bc] * nnS[de,ac]) + pnV[de,ca] * pnS[de,cb] + pnV[de,cb] * pnS[de,ca]
                                #pSum -= 0.5 * (ppV[de,bc] * ppS[de,ac]) + pnV[de,bc] * pnS[de,ac]
                                #nSum -= 0.5 * (nnV[de,bc] * nnS[de,ac]) + pnV[de,cb] * pnS[de,ca]
                            end

                            ph_abc[a,b,c] += J_hat * pSum * 0.5
                            nh_abc[a,b,c] += J_hat * nSum * 0.5
                        end
                    end
                end
            end


            # Sum over c ...
            @inbounds for (a,b) in lj_pairs#_low
                pSum, nSum = sum(view(ph_abc,a,b,:)) / hat(Orb[a].j), sum(view(nh_abc,a,b,:)) / hat(Orb[a].j)
                ph_N[a,b] += pSum
                nh_N[a,b] += nSum

                if a != b
                    ph_N[b,a] += pSum
                    nh_N[b,a] += nSum
                end
            end
    =#


    if Report
        println("\tCompleted the calculation of 1-body mean-field Hamiltonian h_N ...")
    end

    return O1B(ph_N,nh_N), H_NN
end