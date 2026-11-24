function H1b(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,Orb_NNN::Orb3B,T::O1B,V_NN::O2B,V_NNN::Array{Vector{Vector{Float32}},4},Rho::O1B;Sigma::Union{O2B,Nothing} = nothing,Report::Bool = true)
    # Read parameters ...
    A = Float64(Params.Calc.A)
    N_max = Params.Calc.Nmax
    N_2max = Params.Calc.N2max
    N_3max = Params.Calc.N3max
    a_max = div((N_max + 1)*(N_max + 2),2)
    J_max = N_2max + 1
    CMS = Params.Calc.CMS

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

    @inline function H1b_V3b_index(Bra::Int64,Ket::Int64)
        # Index of (Bra|V_NNN|Ket) in the stored lower triangle of a V_NNN block (see V3b_no2b_index) ...
        if Bra >= Ket
            return Ket + div(Bra*(Bra - 1),2)
        else
            return Bra + div(Ket*(Ket - 1),2)
        end
    end

    @inline function H1b_O2b_index(Bra::Int64,Ket::Int64,N::Int64)
        # Index of (Bra|O|Ket) in the stored lower triangle of one block (P,J) of a 2-body operator (see O2b_index) ...
        Bra_max, Bra_min = max(Bra,Ket), min(Bra,Ket)
        return Bra_max + (Bra_min - 1) * N - div(Bra_min * (Bra_min - 1),2)
    end

    # Allocate pairs of orbitals with equal l & j ...
    lj_pairs, lj_pairs_low = H1b_lj_pairs(Params,Orb)
    ad, ad_count = lj_pairs_low, length(lj_pairs_low)
    be, be_count = lj_pairs, length(lj_pairs)
    cf = lj_pairs

    # Orbitals f with l_f = l_c & j_f = j_c (including c) ... only these are connected to c by the lj-diagonal Rho ...
    lj_partners = [Int64[] for c in 1:a_max]
    @inbounds for (c, f) in cf
        push!(lj_partners[c],f)
    end

    # Index of the orbital c in V_NNN blocks [P,l_c+1,J+1,T][j_index] ... 1 for j_c = l_c - 1/2, 2 for j_c = l_c + 1/2 ...
    Orb_j_index = zeros(Int64,a_max)
    @inbounds for Orb_c in Orb
        c, l_c, j_c = Orb_c.a, Orb_c.l, Orb_c.j
        if j_c == abs(2*l_c - 1)
            Orb_j_index[c] = 1
        else
            Orb_j_index[c] = 2
        end
    end

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

                    # Lookup of the 3-body states (ab T_ab c) & (de T_de f) in V_NNN ... stored only for a >= b & d >= e (see V3b_no2b_initialize) ...
                    Map_ab = Orb_NNN.Map[div(max(a,b)*(max(a,b)-1),2) + min(a,b)]
                    Map_de = Orb_NNN.Map[div(max(d,e)*(max(d,e)-1),2) + min(d,e)]
                    J_min = div(abs(j_d - j_e),2)

                    @inbounds for J = div(abs(j_d - j_e),2):div(j_d + j_e,2)

                        # 2-body NN interaction part ...
                        if P_ab == parity(l_d + l_e)
                            J_j_a_hat = hat(2*J) * is_j_a_hat
                            pSum_local += J_j_a_hat * pRho_be * O2b_pp(a,b,d,e,J,P_ab,V_NN,Orb,Orb_NN)
                            pSum_local += J_j_a_hat * nRho_be * O2b_pn(a,b,d,e,J,P_ab,V_NN,Orb_NN)
                            nSum_local += J_j_a_hat * nRho_be * O2b_nn(a,b,d,e,J,P_ab,V_NN,Orb,Orb_NN)
                            nSum_local += J_j_a_hat * pRho_be * O2b_pn(b,a,e,d,J,P_ab,V_NN,Orb_NN)
                        end

                        # 3-body NNN interaction part ... only cf with l_c = l_f & j_c = j_f (so P_abc = P_def as l_a = l_d & l_b = l_e) ...
                        # V_NNN is contracted first with Rho_cf, (ab T_ab|X_q|de T_de) = sum_cf (ab T_ab c|V_NNN|de T_de f) Rho_cf ...
                        # for q = (T_ab T_de T) = 001, 101, 011, 111 & 113 ... the phases of ba & ed are the same for all cf ...
                        J_i = J - J_min + 1
                        pX001, pX101, pX011, pX111, pX113 = 0.0, 0.0, 0.0, 0.0, 0.0
                        nX001, nX101, nX011, nX111, nX113 = 0.0, 0.0, 0.0, 0.0, 0.0

                        @inbounds for c in 1:a_max
                            # Index of (ab T_ab c| in V_NNN ... 0 if not allowed by N_3max ...
                            abc_0 = Int64(Map_ab[1,J_i,c])
                            if abc_0 == 0
                                continue
                            end
                            abc_1, abc_3 = Int64(Map_ab[2,J_i,c]), Int64(Map_ab[3,J_i,c])

                            l_c = Orb[c].l
                            P_abc = parity(l_a + l_b + l_c)
                            V_T1, V_T3 = V_NNN[P_abc,l_c+1,J+1,1][Orb_j_index[c]], V_NNN[P_abc,l_c+1,J+1,2][Orb_j_index[c]]

                            @inbounds for f in lj_partners[c]
                                # Index of |de T_de f) in V_NNN ... 0 if not allowed by N_3max ...
                                def_0 = Int64(Map_de[1,J_i,f])
                                if def_0 == 0
                                    continue
                                end
                                def_1, def_3 = Int64(Map_de[2,J_i,f]), Int64(Map_de[3,J_i,f])
                                pRho_cf, nRho_cf = Rho.p[c,f], Rho.n[c,f]

                                ME001 = Float64(V_T1[H1b_V3b_index(abc_0,def_0)])
                                ME101 = Float64(V_T1[H1b_V3b_index(abc_1,def_0)])
                                ME011 = Float64(V_T1[H1b_V3b_index(abc_0,def_1)])
                                ME111 = Float64(V_T1[H1b_V3b_index(abc_1,def_1)])
                                ME113 = Float64(V_T3[H1b_V3b_index(abc_3,def_3)])

                                pX001 += ME001 * pRho_cf; nX001 += ME001 * nRho_cf
                                pX101 += ME101 * pRho_cf; nX101 += ME101 * nRho_cf
                                pX011 += ME011 * pRho_cf; nX011 += ME011 * nRho_cf
                                pX111 += ME111 * pRho_cf; nX111 += ME111 * nRho_cf
                                pX113 += ME113 * pRho_cf; nX113 += ME113 * nRho_cf
                            end
                        end

                        # Phases of V_NNN for (b a T_ab| = (-1)^((j_a + j_b)/2 - J - T_ab) (a b T_ab| if a < b ... & the same for |e d T_de) if d < e (see V3b_no2b) ...
                        Phase_ab_0, Phase_ab_1 = 1.0, 1.0
                        if a < b
                            Phase_ab_0 = phase(div(j_a + j_b,2) - J)
                            Phase_ab_1 = -Phase_ab_0
                        end
                        Phase_de_0, Phase_de_1 = 1.0, 1.0
                        if d < e
                            Phase_de_0 = phase(div(j_d + j_e,2) - J)
                            Phase_de_1 = -Phase_de_0
                        end
                        pX001, nX001 = Phase_ab_0 * Phase_de_0 * pX001, Phase_ab_0 * Phase_de_0 * nX001
                        pX101, nX101 = Phase_ab_1 * Phase_de_0 * pX101, Phase_ab_1 * Phase_de_0 * nX101
                        pX011, nX011 = Phase_ab_0 * Phase_de_1 * pX011, Phase_ab_0 * Phase_de_1 * nX011
                        pX111, nX111 = Phase_ab_1 * Phase_de_1 * pX111, Phase_ab_1 * Phase_de_1 * nX111
                        pX113, nX113 = Phase_ab_1 * Phase_de_1 * pX113, Phase_ab_1 * Phase_de_1 * nX113

                        pSum_local += is_j_a_hat * (0.5*pX113*pRho_be + 0.25 * (nX001 +
                                        sqrt(1.0/3.0)*nX101 + sqrt(1.0/3.0)*nX011 + nX111/3.0 + 2.0/3.0*nX113)*nRho_be +
                                        1.0/3.0 * (2.0*nX111 + nX113)*pRho_be)

                        nSum_local += is_j_a_hat * (0.5*nX113*nRho_be + 0.25 * (pX001 +
                                        sqrt(1.0/3.0)*pX101 + sqrt(1.0/3.0)*pX011 + pX111/3.0 + 2.0/3.0*pX113)*pRho_be +
                                        1.0/3.0 * (2.0*pX111 + pX113)*nRho_be)
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
    if !isnothing(Sigma)
        NNN_triples = H1b_NNN_triples(Params,Orb,Orb_NN,Orb_NNN)

        # Task-local accumulators for h_cf ...
        ph_N_task = [zeros(Float64,a_max,a_max) for task in 1:N_task]
        nh_N_task = [zeros(Float64,a_max,a_max) for task in 1:N_task]

        # Sigma combined for all orderings of ab & de (see below) ... allocated once for the largest (P,J) block & reused (Sigma_JP[1:7,de,ab] with de, ab <= N_ab) ...
        Sigma_JP = Array{Float64,3}(undef,7,maximum(Orb_NN.N[2,:,:]),maximum(Orb_NN.N[2,:,:]))

        @inbounds for J in 0:J_max
            @inbounds for P2B in 1:2
                N_ab, N_ab_pn = Orb_NN.N[2,P2B,J+1], Orb_NN.N[1,P2B,J+1]
                if N_ab == 0
                    continue
                end

                ppSigma_PJ, pnSigma_PJ, nnSigma_PJ = Sigma.pp[P2B,J+1], Sigma.pn[P2B,J+1], Sigma.nn[P2B,J+1]

                # Whether the pair a >= b has any 3-body state (ab c) in V_NNN (N_3max) ... Sigma is needed only for these pairs ...
                Is_NNN = falses(N_ab)
                @inbounds for ab in 1:N_ab
                    a, b = Orb_NN.Ind[2,P2B,J+1][ab][1], Orb_NN.Ind[2,P2B,J+1][ab][2]
                    Map_ab = Orb_NNN.Map[div(a*(a-1),2) + b]
                    Is_NNN[ab] = any(!=(0), view(Map_ab,1,J - div(abs(Orb[a].j - Orb[b].j),2) + 1,:))
                end

                # Sigma combined for all orderings of ab & de, for this J & P_ab ... independent of c & f, so evaluated once for all pairs a >= b & d >= e (ab >= de) ...
                #   Sigma_JP[1:7,de,ab] ... coefficients of the 3-body matrix elements in the proton & neutron field (see pSum & nSum below) ...
                #   Only ab >= de is needed ... the triples (a b c) >= (d e f) of a block below are ordered by the pair index ...
                #   Each task takes every N_task-th pair de ... the loop over ab >= de reads the stored Sigma (de|Sigma|ab) mostly in the order of the storage ...
                @inbounds Threads.@threads :dynamic for task in 1:N_task
                    @inbounds for de in task:N_task:N_ab
                        if !Is_NNN[de]
                            continue
                        end

                        d, e = Orb_NN.Ind[2,P2B,J+1][de][1], Orb_NN.Ind[2,P2B,J+1][de][2]
                        de_pn, ed_pn = Int64(Orb_NN.Map[d,e,J+1,1]), Int64(Orb_NN.Map[e,d,J+1,1])
                        Phase_de = phase(div(Orb[d].j + Orb[e].j,2) - J)

                        @inbounds for ab in de:N_ab
                            if !Is_NNN[ab]
                                continue
                            end

                            a, b = Orb_NN.Ind[2,P2B,J+1][ab][1], Orb_NN.Ind[2,P2B,J+1][ab][2]
                            ab_pn, ba_pn = Int64(Orb_NN.Map[a,b,J+1,1]), Int64(Orb_NN.Map[b,a,J+1,1])

                            # Phase of V_NNN for (b a T_ab = 0| = (-1)^((j_a + j_b)/2 - J) (a b T_ab = 0| ... opposite for T_ab = 1 ...
                            Phase_ab = phase(div(Orb[a].j + Orb[b].j,2) - J)

                            # pp & nn Sigma ... the phases of ba (ed) in V_NNN & Sigma cancel, so every ordering adds the same ...
                            ppSigma_abde = ppSigma_PJ[H1b_O2b_index(de,ab,N_ab)]
                            nnSigma_abde = nnSigma_PJ[H1b_O2b_index(de,ab,N_ab)]
                            if a != b
                                ppSigma_abde, nnSigma_abde = 2.0 * ppSigma_abde, 2.0 * nnSigma_abde
                            end
                            if d != e
                                ppSigma_abde, nnSigma_abde = 2.0 * ppSigma_abde, 2.0 * nnSigma_abde
                            end

                            # pn Sigma of all orderings ab, ba & de, ed ...
                            pnSigma_abde = pnSigma_PJ[H1b_O2b_index(de_pn,ab_pn,N_ab_pn)]
                            pnSigma_bade, pnSigma_abed, pnSigma_baed = 0.0, 0.0, 0.0
                            if a != b
                                pnSigma_bade = Phase_ab * pnSigma_PJ[H1b_O2b_index(de_pn,ba_pn,N_ab_pn)]
                            end
                            if d != e
                                pnSigma_abed = Phase_de * pnSigma_PJ[H1b_O2b_index(ed_pn,ab_pn,N_ab_pn)]
                            end
                            if a != b && d != e
                                pnSigma_baed = Phase_ab * Phase_de * pnSigma_PJ[H1b_O2b_index(ed_pn,ba_pn,N_ab_pn)]
                            end

                            # ... projected on the isospin T_ab & T_de of the pairs ab & de (T = 1 flips the sign of the swapped pair) ...
                            pnSigma_00 = pnSigma_abde + pnSigma_bade + pnSigma_abed + pnSigma_baed
                            pnSigma_10 = pnSigma_abde - pnSigma_bade + pnSigma_abed - pnSigma_baed
                            pnSigma_01 = pnSigma_abde + pnSigma_bade - pnSigma_abed - pnSigma_baed
                            pnSigma_11 = pnSigma_abde - pnSigma_bade - pnSigma_abed + pnSigma_baed

                            # Proton field (c, f protons) ... <1 0 1/2 +1/2|1/2 +1/2> = -sqrt(1/3) ...
                            #   pSum = ME113 * ppSigma_abde + (ME113 + 2.0 * ME111) / 3.0 * nnSigma_abde +
                            #          2.0 * (ME001 * pnSigma_00 - (ME101 * pnSigma_10 + ME011 * pnSigma_01) / sqrt(3.0) + (ME111 + 2.0 * ME113) / 3.0 * pnSigma_11)
                            # Neutron field (c, f neutrons) ... <1 0 1/2 -1/2|1/2 -1/2> = +sqrt(1/3) ...
                            #   nSum = ME113 * nnSigma_abde + (ME113 + 2.0 * ME111) / 3.0 * ppSigma_abde +
                            #          2.0 * (ME001 * pnSigma_00 + (ME101 * pnSigma_10 + ME011 * pnSigma_01) / sqrt(3.0) + (ME111 + 2.0 * ME113) / 3.0 * pnSigma_11)
                            # ... rewritten as pSum = sum_q ME_q * pSigma_q & nSum = sum_q ME_q * nSigma_q, with nSigma_q = pSigma_q for q = 001 & -pSigma_q for q = 101, 011 ...
                            Sigma_JP[1,de,ab] = ppSigma_abde + nnSigma_abde / 3.0 + 4.0 / 3.0 * pnSigma_11       # pSigma_113
                            Sigma_JP[2,de,ab] = nnSigma_abde + ppSigma_abde / 3.0 + 4.0 / 3.0 * pnSigma_11       # nSigma_113
                            Sigma_JP[3,de,ab] = 2.0 / 3.0 * (nnSigma_abde + pnSigma_11)                          # pSigma_111
                            Sigma_JP[4,de,ab] = 2.0 / 3.0 * (ppSigma_abde + pnSigma_11)                          # nSigma_111
                            Sigma_JP[5,de,ab] = 2.0 * pnSigma_00                                                 # p/nSigma_001
                            Sigma_JP[6,de,ab] = -2.0 / sqrt(3.0) * pnSigma_10                                    # pSigma_101
                            Sigma_JP[7,de,ab] = -2.0 / sqrt(3.0) * pnSigma_01                                    # pSigma_011
                        end
                    end
                end

                # Each task takes every N_task-th triple (a b c) of every V_NNN block with this J & P_ab ...
                @inbounds Threads.@threads :dynamic for task in 1:N_task
                    ph_task, nh_task = ph_N_task[task], nh_N_task[task]

                    @inbounds for l_c in 0:N_max
                        P3B = parity(P2B - 1 + l_c)

                        @inbounds for (j_index, Triples) in enumerate(NNN_triples[P3B,l_c+1,J+1])
                            V_T1, V_T3 = V_NNN[P3B,l_c+1,J+1,1][j_index], V_NNN[P3B,l_c+1,J+1,2][j_index]

                            @inbounds for abc in task:N_task:length(Triples)
                                a, b, c, ab, abc_0, abc_1, abc_3 = Triples[abc]

                                @inbounds for def in 1:abc
                                    d, e, f, de, def_0, def_1, def_3 = Triples[def]

                                    # 3-body matrix elements (ab T_ab c|V_NNN|de T_de f) with T = 1/2 (ME_T_ab_T_de_1) & T = 3/2 (ME113) ...
                                    ME001 = Float64(V_T1[H1b_V3b_index(abc_0,def_0)])
                                    ME101 = Float64(V_T1[H1b_V3b_index(abc_1,def_0)])
                                    ME011 = Float64(V_T1[H1b_V3b_index(abc_0,def_1)])
                                    ME111 = Float64(V_T1[H1b_V3b_index(abc_1,def_1)])
                                    ME113 = Float64(V_T3[H1b_V3b_index(abc_3,def_3)])

                                    # Proton & neutron field (c, f protons & neutrons) ...
                                    pSum = ME113 * Sigma_JP[1,de,ab] + ME111 * Sigma_JP[3,de,ab] + ME001 * Sigma_JP[5,de,ab] + ME101 * Sigma_JP[6,de,ab] + ME011 * Sigma_JP[7,de,ab]
                                    nSum = ME113 * Sigma_JP[2,de,ab] + ME111 * Sigma_JP[4,de,ab] + ME001 * Sigma_JP[5,de,ab] - ME101 * Sigma_JP[6,de,ab] - ME011 * Sigma_JP[7,de,ab]

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
    end

    if Report
        println("\tCompleted the calculation of 1-body mean-field Hamiltonian h_N ...")
    end

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

    # Number of work packages ... more than threads to balance the load ...
    N_task = 4*Threads.nthreads()

    function H2b_NNN_triples(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,Orb_NNN::Orb3B)
        # Read # initialize parameters ... (local, otherwise the parameters of H2b are overwritten ...)
        local N_max, N_2max = Params.Calc.Nmax, Params.Calc.N2max
        local a_max = div((N_max + 1)*(N_max + 2),2)
        local J_max = N_2max + 1
        l_max = N_max

        # 3-body triples (d e f) with d >= e of every V_NNN block [P,l_f+1,J+1][j_f] ... stored as (f, de, i_0, i_1, i_3),
        #   de  ... index of the pair d >= e in Orb_NN (Orb_NN.Map[d,e,J+1,2]),
        #   i_0 ... index of (d e, T_de = 0, f) in V_NNN[P,l_f+1,J+1,1][j_f] (T = 1/2),
        #   i_1 ... index of (d e, T_de = 1, f) in V_NNN[P,l_f+1,J+1,1][j_f] (T = 1/2),
        #   i_3 ... index of (d e, T_de = 1, f) in V_NNN[P,l_f+1,J+1,2][j_f] (T = 3/2) ... (see Map_ab[T_ab + T,J - J_min + 1,c] in V3b_no2b_initialize)
        # The triples of a block are ordered as in V_NNN, i.e. by (d, e, f) ... so also by the pair index de (Orb_NN orders the pairs by (d, e) too) ...
        # NNN_last[P,l_f+1,J+1][j_f][de] ... number of triples of the block with the pair index <= de ...
        local NNN_triples = Array{Vector{Vector{NTuple{5,Int64}}}}(undef,2,l_max+1,J_max+1)
        local NNN_last = Array{Vector{Vector{Int64}}}(undef,2,l_max+1,J_max+1)
        @inbounds for P in 1:2
            @inbounds for l in 0:l_max
                @inbounds for J in 0:J_max
                    NNN_triples[P,l+1,J+1] = [NTuple{5,Int64}[] for j_index in 1:length(Orb_NNN.N[P,l+1,J+1,1])]
                    NNN_last[P,l+1,J+1] = [zeros(Int64,Orb_NN.N[2,parity(P - 1 + l),J+1]) for j_index in 1:length(Orb_NNN.N[P,l+1,J+1,1])]
                end
            end
        end

        @inbounds for d in 1:a_max
            l_d, j_d = Orb[d].l, Orb[d].j
            @inbounds for e in 1:d
                l_e, j_e = Orb[e].l, Orb[e].j
                Map_de = Orb_NNN.Map[div(d*(d-1),2) + e]
                J_min = div(abs(j_d - j_e),2)
                @inbounds for J_i in 1:size(Map_de,2)
                    J = J_min + J_i - 1
                    de = Int64(Orb_NN.Map[d,e,J+1,2])
                    @inbounds for f in 1:a_max
                        l_f, j_f = Orb[f].l, Orb[f].j
                        if Map_de[1,J_i,f] == 0
                            continue
                        end
                        P = parity(l_d + l_e + l_f)
                        if j_f == abs(2*l_f - 1)
                            j_index = 1
                        else
                            j_index = 2
                        end
                        push!(NNN_triples[P,l_f+1,J+1][j_index],(f,de,Int64(Map_de[1,J_i,f]),Int64(Map_de[2,J_i,f]),Int64(Map_de[3,J_i,f])))
                        NNN_last[P,l_f+1,J+1][j_index][de] = length(NNN_triples[P,l_f+1,J+1][j_index])
                    end
                end
            end
        end

        # Pairs without a triple in the block ... the same number as the previous pair ...
        @inbounds for Last_block in NNN_last
            @inbounds for Last in Last_block
                @inbounds for de in 2:length(Last)
                    Last[de] = max(Last[de],Last[de-1])
                end
            end
        end

        return NNN_triples, NNN_last
    end

    @inline function H2b_V3b_index(Bra::Int64,Ket::Int64)
        # Index of (Bra|V_NNN|Ket) in the stored lower triangle of a V_NNN block (see V3b_no2b_index) ...
        if Bra >= Ket
            return Ket + div(Bra*(Bra - 1),2)
        else
            return Bra + div(Ket*(Ket - 1),2)
        end
    end

    @inline function H2b_O2b_index(Bra::Int64,Ket::Int64,N::Int64)
        # Index of (Bra|O|Ket) in the stored lower triangle of one block (P,J) of a 2-body operator (see O2b_index) ...
        Bra_max, Bra_min = max(Bra,Ket), min(Bra,Ket)
        return Bra_max + (Bra_min - 1) * N - div(Bra_min * (Bra_min - 1),2)
    end

    # Index of the orbital c in V_NNN blocks [P,l_c+1,J+1,T][j_index] ... 1 for j_c = l_c - 1/2, 2 for j_c = l_c + 1/2 ...
    Orb_j_index = zeros(Int64,a_max)
    @inbounds for Orb_c in Orb
        c, l_c, j_c = Orb_c.a, Orb_c.l, Orb_c.j
        if j_c == abs(2*l_c- 1)
            Orb_j_index[c] = 1
        else
            Orb_j_index[c] = 2
        end
    end

    # 3-body triples of every V_NNN block ordered by the pair index ...
    NNN_triples, NNN_last = H2b_NNN_triples(Params,Orb,Orb_NN,Orb_NNN)

    # Start from the 2-body interaction V_NN ... H_NN = cRes * (V_NN + V_NNN in NO2B), the blocks (P,J) are copied in parallel ...
    H_NN = O2B(similar(V_NN.pp),similar(V_NN.pn),similar(V_NN.nn))
    @inbounds Threads.@threads :dynamic for PJ in eachindex(V_NN.pp)
        H_NN.pp[PJ] = cRes .* V_NN.pp[PJ]
        H_NN.pn[PJ] = cRes .* V_NN.pn[PJ]
        H_NN.nn[PJ] = cRes .* V_NN.nn[PJ]
    end

    # Include V_NNN in the NO2B approximation ...
    #   (ab|H|de) += 1/J_hat sum_cf (ab c|V_NNN|de f) Rho_fc ... only with l_c = l_f & j_c = j_f (lj-diagonal Rho) ...
    #   For a fixed pair ab (Bra) & orbital c, the row (ab c| of a V_NNN block is read for all |de f) with de <= ab ... which are stored
    #   contiguously, (ab c|V_NNN|de f) with de < ab are all in the rows of ab as the triples are ordered by the pair index ...
    @inbounds for J in 0:J_max
        @inbounds for P in 1:2
            if Report
                if P == 1
                    println("\tCalculating   ...   J = " * string(J) * "/" * string(J_max) * "\tP = +")
                else
                    println("\tCalculating   ...   J = " * string(J) * "/" * string(J_max) * "\tP = -")
                end
            end

            # Number of pairs ab with a >= b (pp & nn) & of all ordered pairs ab (pn) ...
            N_pp, N_pn = Orb_NN.N[2,P,J+1], Orb_NN.N[1,P,J+1]
            cRes_J_hat = cRes / hat(2*J)
            
            if N_pp == 0
                continue
            end

            # For every pair a >= b ... indices of ab & ba in pn, phase of ba & whether ab has any 3-body state (ab c) in V_NNN (N_3max) ...
            ab_pn, ba_pn = zeros(Int64,N_pp), zeros(Int64,N_pp)
            Phase_ba = zeros(Float64,N_pp)
            Is_NNN = falses(N_pp)
            @inbounds for ab in 1:N_pp
                a, b = Orb_NN.Ind[2,P,J+1][ab][1], Orb_NN.Ind[2,P,J+1][ab][2]
                ab_pn[ab], ba_pn[ab] = Int64(Orb_NN.Map[a,b,J+1,1]), Int64(Orb_NN.Map[b,a,J+1,1])

                # Phase of V_NNN for (b a T_ab = 0| = (-1)^((j_a + j_b)/2 - J) (a b T_ab = 0| ... opposite for T_ab = 1 ...
                Phase_ba[ab] = phase(div(Orb[a].j + Orb[b].j,2) - J)

                Map_ab = Orb_NNN.Map[div(a*(a-1),2) + b]
                Is_NNN[ab] = any(!=(0), view(Map_ab,1,J - div(abs(Orb[a].j - Orb[b].j),2) + 1,:))
            end

            # Each task takes every N_task-th block of N_block consecutive pairs a >= b (Bra) ... every Bra writes only its own elements (Bra|V|Ket <= Bra) ...
            # consecutive Bras write next to each other in H_NN (for the same Ket), so the blocks keep these writes in the cache ...
            N_block = 8
            @inbounds Threads.@threads :dynamic for task in 1:N_task
                # Task-local contractions over cf for all Kets ... X[q,Ket] = sum_cf (Bra c|V_NNN|Ket f) * Rho_fc with q =
                #   1 ... ME113 * pRho_fc,       2 ... ME113 * nRho_fc,
                #   3 ... ME111 * pRho_fc,       4 ... ME111 * nRho_fc,
                #   5 ... ME001 * (pRho_fc + nRho_fc),
                #   6 ... ME101 * (nRho_fc - pRho_fc),  7 ... ME011 * (nRho_fc - pRho_fc) ...
                X = zeros(Float64,7,N_pp)

                @inbounds for Block in task:N_task:cld(N_pp,N_block)
                    @inbounds for Bra in (N_block*(Block - 1) + 1):min(N_block*Block,N_pp)
                        # Nothing to add if (ab c| is not allowed by N_3max for any c ...
                        if !Is_NNN[Bra]
                            continue
                        end

                        a, b = Orb_NN.Ind[2,P,J+1][Bra][1], Orb_NN.Ind[2,P,J+1][Bra][2]
                        l_a, j_a = Orb[a].l, Orb[a].j
                        l_b, j_b = Orb[b].l, Orb[b].j

                        # Lookup of the 3-body states (ab T_ab c) in V_NNN ... Map_ab[T_ab + T,J - J_min + 1,c] (see V3b_no2b_initialize) ...
                        Map_ab = Orb_NNN.Map[div(a*(a-1),2) + b]
                        J_ab = J - div(abs(j_a - j_b),2) + 1

                        fill!(view(X,:,1:Bra),0.0)

                        @inbounds for c in 1:a_max
                            # Index of (ab T_ab c| in V_NNN ... 0 if not allowed by N_3max ...
                            abc_0 = Int64(Map_ab[1,J_ab,c])
                            if abc_0 == 0
                                continue
                            end
                            abc_1, abc_3 = Int64(Map_ab[2,J_ab,c]), Int64(Map_ab[3,J_ab,c])

                            l_c = Orb[c].l
                            P3B = parity(l_a + l_b + l_c)

                            V_T1, V_T3 = V_NNN[P3B,l_c+1,J+1,1][Orb_j_index[c]], V_NNN[P3B,l_c+1,J+1,2][Orb_j_index[c]]
                            Triples = NNN_triples[P3B,l_c+1,J+1][Orb_j_index[c]]

                            # All triples (d e f) of the block with de <= ab ... (l_f = l_c & j_f = j_c by the block) ...
                            @inbounds for def in 1:NNN_last[P3B,l_c+1,J+1][Orb_j_index[c]][Bra]
                                f, Ket, def_0, def_1, def_3 = Triples[def]

                                ME001 = Float64(V_T1[H2b_V3b_index(abc_0,def_0)])
                                ME101 = Float64(V_T1[H2b_V3b_index(abc_1,def_0)])
                                ME011 = Float64(V_T1[H2b_V3b_index(abc_0,def_1)])
                                ME111 = Float64(V_T1[H2b_V3b_index(abc_1,def_1)])
                                ME113 = Float64(V_T3[H2b_V3b_index(abc_3,def_3)])

                                pRho_fc, nRho_fc = Rho.p[f,c], Rho.n[f,c]

                                X[1,Ket] += ME113 * pRho_fc
                                X[2,Ket] += ME113 * nRho_fc
                                X[3,Ket] += ME111 * pRho_fc
                                X[4,Ket] += ME111 * nRho_fc
                                X[5,Ket] += ME001 * (pRho_fc + nRho_fc)
                                X[6,Ket] += ME101 * (nRho_fc - pRho_fc)
                                X[7,Ket] += ME011 * (nRho_fc - pRho_fc)
                            end
                        end

                        @inbounds for Ket in 1:Bra
                            if !Is_NNN[Ket]
                                continue
                            end

                            pX113, nX113, pX111, nX111 = X[1,Ket], X[2,Ket], X[3,Ket], X[4,Ket]
                            X001, X101, X011 = X[5,Ket], X[6,Ket], X[7,Ket]

                            ppSum = pX113 + (2.0 * nX111 + nX113) / 3.0
                            nnSum = nX113 + (2.0 * pX111 + pX113) / 3.0

                            pnSum_abde = 0.5 * X001 + (pX111 + nX111) / 6.0 + (pX113 + nX113) / 3.0 + ( X101 + X011) / sqrt(12.0)
                            pnSum_bade = 0.5 * X001 - (pX111 + nX111) / 6.0 - (pX113 + nX113) / 3.0 + (-X101 + X011) / sqrt(12.0)
                            pnSum_abed = 0.5 * X001 - (pX111 + nX111) / 6.0 - (pX113 + nX113) / 3.0 + ( X101 - X011) / sqrt(12.0)
                            pnSum_baed = 0.5 * X001 + (pX111 + nX111) / 6.0 + (pX113 + nX113) / 3.0 - ( X101 + X011) / sqrt(12.0)

                            # Allocate the pp & nn elements ... (see O2b_index) ...
                            Ind = H2b_O2b_index(Bra,Ket,N_pp)
                            H_NN.pp[P,J+1][Ind] += cRes_J_hat * ppSum
                            H_NN.nn[P,J+1][Ind] += cRes_J_hat * nnSum

                            # Allocate the pn elements of all orderings ... (ba|V|de) & (ab|V|ed) are the same element for ab = de ...
                            Ind = H2b_O2b_index(ab_pn[Bra],ab_pn[Ket],N_pn)
                            H_NN.pn[P,J+1][Ind] += cRes_J_hat * pnSum_abde

                            if a != b
                                Ind = H2b_O2b_index(ba_pn[Bra],ab_pn[Ket],N_pn)
                                H_NN.pn[P,J+1][Ind] += cRes_J_hat * Phase_ba[Bra] * pnSum_bade
                            end

                            if ab_pn[Ket] != ba_pn[Ket] && Ket != Bra
                                Ind = H2b_O2b_index(ab_pn[Bra],ba_pn[Ket],N_pn)
                                H_NN.pn[P,J+1][Ind] += cRes_J_hat * Phase_ba[Ket] * pnSum_abed
                            end

                            if a != b && ab_pn[Ket] != ba_pn[Ket]
                                Ind = H2b_O2b_index(ba_pn[Bra],ba_pn[Ket],N_pn)
                                H_NN.pn[P,J+1][Ind] += cRes_J_hat * Phase_ba[Bra] * Phase_ba[Ket] * pnSum_baed
                            end
                        end
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

function H1b_energy(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,Orb_NNN::Orb3B,T::O1B,V_NN::O2B,V_NNN::Array{Vector{Vector{Float32}},4},Rho::O1B;Report::Bool = true)
    # Total energy of the 1-body density Rho for the bare Hamiltonian T + V_NN + V_NNN ... everything in the LHO basis ...
    #   E = sum_ad T_ad Rho_da + 1/2 sum V_NN Rho Rho + 1/6 sum V_NNN Rho Rho Rho ... (same as HF_energy, but for any Rho & without the l_max & j_max cut)
    # Read parameters ...
    A = Float64(Params.Calc.A)
    N_max = Params.Calc.Nmax
    N_2max = Params.Calc.N2max
    a_max = div((N_max + 1)*(N_max + 2),2)
    CMS = Params.Calc.CMS

    if Report
        println("\nCalculating the total energy of the 1-body density Rho for the bare Hamiltonian ...")
    end

    # Number of work packages ... more than threads to balance the load ...
    N_task = 4*Threads.nthreads()

    # Define local functions ...
    @inline function H1b_energy_V3b_index(Bra::Int64,Ket::Int64)
        # Index of (Bra|V_NNN|Ket) in the stored lower triangle of a V_NNN block (see V3b_no2b_index) ...
        if Bra >= Ket
            return Ket + div(Bra*(Bra - 1),2)
        else
            return Bra + div(Ket*(Ket - 1),2)
        end
    end

    # Pairs of orbitals with l_a = l_d, j_a = j_d & non-zero Rho_ad (Rho is lj-diagonal, zero Rho_ad do not contribute) ...
        # ad ... only a >= d, the term da is the same as ad (Rho & H are symmetric) & is included by the weight 2 ...
        # be ... all ordered pairs ...
    ad, be = NTuple{2,Int64}[], NTuple{2,Int64}[]
    @inbounds for Orb_a in Orb
        a, l_a, j_a = Orb_a.a, Orb_a.l, Orb_a.j
        @inbounds for Orb_d in Orb
            d, l_d, j_d = Orb_d.a, Orb_d.l, Orb_d.j
            if l_a != l_d || j_a != j_d || (Rho.p[a,d] == 0.0 && Rho.n[a,d] == 0.0)
                continue
            end
            push!(be,(a,d))
            if d <= a
                push!(ad,(a,d))
            end
        end
    end

    # Orbitals f with l_f = l_c, j_f = j_c & non-zero Rho_cf (including c) ...
    cf_partners = [Int64[] for c in 1:a_max]
    @inbounds for (c, f) in be
        push!(cf_partners[c],f)
    end

    # Index of the orbital c in V_NNN blocks [P,l_c+1,J+1,T][j_index] ... 1 for j_c = l_c - 1/2, 2 for j_c = l_c + 1/2 ...
    Orb_j_index = zeros(Int64,a_max)
    @inbounds for Orb_c in Orb
        c, l_c, j_c = Orb_c.a, Orb_c.l, Orb_c.j
        if j_c == abs(2*l_c - 1)
            Orb_j_index[c] = 1
        else
            Orb_j_index[c] = 2
        end
    end

    # 1-body kinetic energy & inclusion of Center-of-Mass motion (CM) correction ...
    E_T = 0.0
    @inbounds for (a, d) in be
        j_a = Orb[a].j
        E_T += hat(j_a) * (T.p[a,d] * Rho.p[d,a] + T.n[a,d] * Rho.n[d,a])
    end
        # Combined 1- + 2-body kinetic operator with CM correction ...
    if CMS == "CMS1+2B"
        E_T *= (1.0 - 1.0 / A)
        # No contribution for pure 2-body kinetic operator with CM correction ...
    elseif CMS == "CMS2B"
        E_T = 0.0
    end
        # Pure 1-body kinetic operator with no CM correction otherwise ...

    # 2-body NN & 3-body NNN interaction energy ...
        # List of all (ad, be) combinations ... used to distribute the work among tasks ...
    adbe_tasks = NTuple{2,Int64}[]
    @inbounds for ad_i in 1:length(ad)
        @inbounds for be_i in 1:length(be)
            push!(adbe_tasks,(ad_i,be_i))
        end
    end

    # Task-local accumulators ...
    E_V_task = zeros(Float64,N_task)

    # Each task takes every N_task-th (ad, be) combination ...
    @inbounds Threads.@threads :dynamic for task in 1:N_task
        E_V_local = 0.0

        @inbounds for task_i in task:N_task:length(adbe_tasks)
            ad_i, be_i = adbe_tasks[task_i]
            a, d = ad[ad_i]
            b, e = be[be_i]
            n_a, l_a, j_a = Orb[a].n, Orb[a].l, Orb[a].j
            n_d, l_d, j_d = Orb[d].n, Orb[d].l, Orb[d].j
            n_b, l_b, j_b = Orb[b].n, Orb[b].l, Orb[b].j
            n_e, l_e, j_e = Orb[e].n, Orb[e].l, Orb[e].j
            P_ab = parity(l_a + l_b)

            if (2*(n_a + n_b) + l_a + l_b) > N_2max || (2*(n_d + n_e) + l_d + l_e) > N_2max
                continue
            end

            # Weight of ad ... 2 for a != d (term da) ...
            w_ad = (a != d) ? 2.0 : 1.0

            pRho_ad, nRho_ad = Rho.p[a,d], Rho.n[a,d]
            pRho_be, nRho_be = Rho.p[b,e], Rho.n[b,e]

            # Lookup of the 3-body states (ab T_ab c) & (de T_de f) in V_NNN ... stored only for a >= b & d >= e (see V3b_no2b_initialize) ...
            Map_ab = Orb_NNN.Map[div(max(a,b)*(max(a,b)-1),2) + min(a,b)]
            Map_de = Orb_NNN.Map[div(max(d,e)*(max(d,e)-1),2) + min(d,e)]
            J_min = div(abs(j_d - j_e),2)

            @inbounds for J in div(abs(j_d - j_e),2):div(j_d + j_e,2)
                J_hat = hat(2*J)

                # 2-body NN interaction part ... (P_ab = P_de as l_a = l_d & l_b = l_e) ...
                E_V_local += w_ad * J_hat * (0.5 * pRho_ad * pRho_be * O2b_pp(a,b,d,e,J,P_ab,V_NN,Orb,Orb_NN) +
                                             0.5 * nRho_ad * nRho_be * O2b_nn(a,b,d,e,J,P_ab,V_NN,Orb,Orb_NN) +
                                             pRho_ad * nRho_be * O2b_pn(a,b,d,e,J,P_ab,V_NN,Orb_NN))

                # 3-body NNN interaction part ... only cf with l_c = l_f & j_c = j_f (so P_abc = P_def) ...
                # V_NNN is contracted first with Rho_cf, (ab T_ab|X_q|de T_de) = sum_cf (ab T_ab c|V_NNN|de T_de f) Rho_cf ...
                # for q = (T_ab T_de T) = 001, 101, 011, 111 & 113 ... the phases of ba & ed are the same for all cf ...
                # (only pX113 enters the energy for Rho_cf = pRho_cf, see the isospin combination below) ...
                J_i = J - J_min + 1
                pX113 = 0.0
                nX001, nX101, nX011, nX111, nX113 = 0.0, 0.0, 0.0, 0.0, 0.0

                @inbounds for c in 1:a_max
                    # Index of (ab T_ab c| in V_NNN ... 0 if not allowed by N_3max ...
                    abc_0 = Int64(Map_ab[1,J_i,c])
                    if abc_0 == 0
                        continue
                    end
                    abc_1, abc_3 = Int64(Map_ab[2,J_i,c]), Int64(Map_ab[3,J_i,c])

                    l_c = Orb[c].l
                    P_abc = parity(l_a + l_b + l_c)
                    V_T1, V_T3 = V_NNN[P_abc,l_c+1,J+1,1][Orb_j_index[c]], V_NNN[P_abc,l_c+1,J+1,2][Orb_j_index[c]]

                    @inbounds for f in cf_partners[c]
                        # Index of |de T_de f) in V_NNN ... 0 if not allowed by N_3max ...
                        def_0 = Int64(Map_de[1,J_i,f])
                        if def_0 == 0
                            continue
                        end
                        def_1, def_3 = Int64(Map_de[2,J_i,f]), Int64(Map_de[3,J_i,f])
                        pRho_cf, nRho_cf = Rho.p[c,f], Rho.n[c,f]

                        ME001 = Float64(V_T1[H1b_energy_V3b_index(abc_0,def_0)])
                        ME101 = Float64(V_T1[H1b_energy_V3b_index(abc_1,def_0)])
                        ME011 = Float64(V_T1[H1b_energy_V3b_index(abc_0,def_1)])
                        ME111 = Float64(V_T1[H1b_energy_V3b_index(abc_1,def_1)])
                        ME113 = Float64(V_T3[H1b_energy_V3b_index(abc_3,def_3)])

                        nX001 += ME001 * nRho_cf
                        nX101 += ME101 * nRho_cf
                        nX011 += ME011 * nRho_cf
                        nX111 += ME111 * nRho_cf
                        pX113 += ME113 * pRho_cf; nX113 += ME113 * nRho_cf
                    end
                end

                # Phases of V_NNN for (b a T_ab| = (-1)^((j_a + j_b)/2 - J - T_ab) (a b T_ab| if a < b ... & the same for |e d T_de) if d < e (see V3b_no2b) ...
                Phase_ab_0, Phase_ab_1 = 1.0, 1.0
                if a < b
                    Phase_ab_0 = phase(div(j_a + j_b,2) - J)
                    Phase_ab_1 = -Phase_ab_0
                end
                Phase_de_0, Phase_de_1 = 1.0, 1.0
                if d < e
                    Phase_de_0 = phase(div(j_d + j_e,2) - J)
                    Phase_de_1 = -Phase_de_0
                end
                pX113 = Phase_ab_1 * Phase_de_1 * pX113
                nX001 = Phase_ab_0 * Phase_de_0 * nX001
                nX101 = Phase_ab_1 * Phase_de_0 * nX101
                nX011 = Phase_ab_0 * Phase_de_1 * nX011
                nX111 = Phase_ab_1 * Phase_de_1 * nX111
                nX113 = Phase_ab_1 * Phase_de_1 * nX113

                # Isospin combination as in HF_energy ... ppp, ppn, pnn & nnn ...
                E_V_local += w_ad / 6.0 * (pX113 * pRho_ad * pRho_be + (2.0 * nX111 + nX113) * pRho_ad * pRho_be +
                                           (1.5 * nX001 + sqrt(3.0/4.0) * nX101 + sqrt(3.0/4.0) * nX011 + 0.5 * nX111 + nX113) * pRho_ad * nRho_be +
                                           nX113 * nRho_ad * nRho_be)
            end
        end

        E_V_task[task] = E_V_local
    end

    # Sum-up the total energy ...
    E = E_T + sum(E_V_task)

    if Report
        println("\tTotal energy of Rho    ...   E = " * string(E) * " MeV")
    end

    return E
end

function H2b_energy(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,V_NN::O2B,Sigma::O2B)
    # Read parameters ...
    N_2max = Params.Calc.N2max
    J_max = N_2max + 1

    # Calculate the correlation energy ...
    println("\nComputing the correlation ground-state energy by contraction with the provided 2-body correlation function ...")

    # Number of work packages ... more than threads to balance the load ...
    N_task = 4*Threads.nthreads()

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