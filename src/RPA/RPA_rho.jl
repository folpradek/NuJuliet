function RPA_OBDM_oY2(Params::Parameters,Orb::Vector{Orb1B},Orb_1p1h::Orb2B_1p1h,Y::Matrix{Matrix{ComplexF64}})
    # Read calculation parameters ..
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1
    a_max = div((N_max + 1)*(N_max + 2),2)
    Ortho = Params.Calc.RPA.Ortho

    # Initialize the list of values of J & P for iteration ...
    JP_list = JP_initialize(J_max)

    # Initialize density matrices ...
    pRho, nRho = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)

    @inbounds for a in 1:a_max
        pRho[a,a] = Orb[a].pO
        nRho[a,a] = Orb[a].nO
    end

    # Initialize the storage for diagonal OBDM contributions of individual J & P blocks ...
    N_JP = length(JP_list)

    pRho_JP, nRho_JP = Vector{Vector{Float64}}(undef,N_JP), Vector{Vector{Float64}}(undef,N_JP)
    @inbounds for i_JP in 1:N_JP
        pRho_JP[i_JP] = zeros(Float64,a_max)
        nRho_JP[i_JP] = zeros(Float64,a_max)
    end

    # Evaluate RPA OBDM ... (Threads.@threads requires an indexable range, i.e. enumerate(...) cannot be used here)
    @inbounds Threads.@threads for i_JP in 1:N_JP
        JP = JP_list[i_JP]
        J, P = JP[1], JP[2]
        N_ph = Orb_1p1h.N[P,J+1]
        Orb_1p1h_JP = Orb_1p1h.Ind[P,J+1]
        Y_JP = Y[P,J+1]

        pRho_i, nRho_i = pRho_JP[i_JP], nRho_JP[i_JP]

        # All RPA solutions are included ... the same set of states as in RPA_sigma() & RPA_energy() (algebraic) ...
        #   ... except the spurious 1- Goldstone level (i_nu = 1 of the orthogonalized 1- block) ...
        nu_start = RPA_spurious(Ortho,J,P) ? 2 : 1

        @inbounds for (i_ph,Orb_ph) in enumerate(Orb_1p1h_JP)
            a_p, a_h, T_ph = Orb_ph.a, Orb_ph.b, Orb_ph.T
            j_p, j_h = Orb[a_p].j, Orb[a_h].j

            Sum = 0.0
            @inbounds for i_nu in nu_start:N_ph
                y = abs2(Y_JP[i_ph,i_nu])
                ME = y * hat(2*J)
                Sum += ME
            end

            if T_ph == -1
                n_p, n_h = Orb[a_p].pO, Orb[a_h].pO
                Amp_p, Amp_h = (n_h - n_p)^2 / hat(j_p), (n_h - n_p)^2 / hat(j_h)
                pRho_i[a_h] -= Amp_h * Sum
                pRho_i[a_p] += Amp_p * Sum
            elseif T_ph == 1
                n_p, n_h = Orb[a_p].nO, Orb[a_h].nO
                Amp_p, Amp_h = (n_h - n_p)^2 / hat(j_p), (n_h - n_p)^2 / hat(j_h)
                nRho_i[a_h] -= Amp_h * Sum
                nRho_i[a_p] += Amp_p * Sum
            end
        end
    end

    # Sum up the contributions of individual J & P blocks ...
    @inbounds for i_JP in 1:N_JP
        @inbounds for a in 1:a_max
            pRho[a,a] += pRho_JP[i_JP][a]
            nRho[a,a] += nRho_JP[i_JP][a]
        end
    end

    return O1B(pRho,nRho)
end

#=
    RPA OBDM from the 2p-2h block of the 2-body correlation function Sigma (Pauli-consistent counting) ...
        Rho = diag(occupations of Orb) + drho ,
        drho_pq = + 1/(2j_p + 1) sum_J (2J + 1) [ 1/2 sum_(r, cd like) Sigma_pr,cd Sigma_qr,cd + sum_(r, cd pn) Sigma_pr,cd Sigma_qr,cd ]
        drho_ij = - 1/(2j_i + 1) sum_J (2J + 1) [ 1/2 sum_(k, ab like) Sigma_ab,ik Sigma_ab,jk + sum_(k, ab pn) Sigma_ab,ik Sigma_ab,jk ]
    over all ordered pairs, p, q, r, a, b particles & i, j, k, c, d holes (the same classification as in RPA_sigma) ...

    - uncoupled form n_p = 1/2 sum_p'hh' |lambda_pp'hh'|^2 ... the occupations of the correlated state |0> + 1/4 sum lambda a+a+aa |0>,
      i.e. like-particle 2p-2h configurations are counted once (RPA_OBDM_oY2 & oY4 count them twice, quasi-boson approximation),
      with E = h.drho + 1/4 V Sigma -> MP2 at 2nd order (RPA_OBDM_oY2 ... MP2(pn) only)
    - includes the off-diagonal pp' & hh' elements (same l, j & isospin), no ph elements (absent at this order in RPA) ...
    - conserves the particle number exactly ... every |Sigma_pp'hh'|^2 enters the particle & the hole sums with the same weight ...
    - leading order in Sigma ... valid for small depletion (soft modes with large Y are beyond it), occupations are not bounded to [0,1] ...
    - Sigma carries the factors |D D| of RPA_sigma, i.e. drho ~ D^2 (amplitude)^2 (RPA_OBDM_oY2 ... D^2 |Y|^2) ...
    Same convention as RPA_OBDM_oY2: the returned OBDM starts from the occupations of Orb ...
=#
function RPA_OBDM_sigma(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,Sigma::O2B)
    # Read calculation parameters ...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1
    a_max = div((N_max + 1)*(N_max + 2),2)
    hTol = Params.Calc.RPA.Tol_hol
    pTol = Params.Calc.RPA.Tol_par

    # Define local functions to determine particle & hole orbitals ... (as in RPA_sigma)
    @inline particle(O::Float64) = (O <= pTol)

    @inline hole(O::Float64) = (O >= hTol)

    function OBDM_unpack(O_PJ::Vector{Float64},N::Int64)
        # Unpack the stored lower triangle of one block (P,J) of a 2-body operator into a dense symmetric matrix ...
        O_dense = Matrix{Float64}(undef,N,N)
        @inbounds for Ket in 1:N
            Offset = (Ket - 1) * N - div(Ket*(Ket - 1),2)
            @inbounds for Bra in Ket:N
                O_dense[Bra,Ket] = O_PJ[Bra + Offset]
                O_dense[Ket,Bra] = O_PJ[Bra + Offset]
            end
        end
        return O_dense
    end

    function OBDM_dense(O::O2B,P::Int64,J::Int64)
        # Dense matrices ppO[de,ac] = <ac|O|de>_J (pnO, nnO similarly) ... ac & de run over all ordered pairs (the pn pairs, Orb_NN.Ind[1,P,J+1]) ...
        N_ac, N_ac_low = Orb_NN.N[1,P,J+1], Orb_NN.N[2,P,J+1]
        ac_pairs = Orb_NN.Ind[1,P,J+1]

        pnO = OBDM_unpack(O.pn[P,J+1],N_ac)
        ppO_low, nnO_low = OBDM_unpack(O.pp[P,J+1],N_ac_low), OBDM_unpack(O.nn[P,J+1],N_ac_low)

        # pp & nn are stored only for a >= c ... pair ac with a < c is the stored pair ca times the phase (-1)^(J + 1 + (j_a + j_c)/2) ...
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
        @inbounds for ac in 1:N_ac
            @inbounds for de in 1:N_ac
                ppO[de,ac] = Phase_ac[de] * Phase_ac[ac] * ppO_low[ac_low[de],ac_low[ac]]
                nnO[de,ac] = Phase_ac[de] * Phase_ac[ac] * nnO_low[ac_low[de],ac_low[ac]]
            end
        end

        return ppO, pnO, nnO
    end

    # Particle & hole flags of all orbitals ...
    p_par, p_hol = [particle(Orb[a].pO) for a in 1:a_max], [hole(Orb[a].pO) for a in 1:a_max]
    n_par, n_hol = [particle(Orb[a].nO) for a in 1:a_max], [hole(Orb[a].nO) for a in 1:a_max]

    # Pairs of orbitals with equal l & j ... (p,q) with p >= q ...
    lj_pairs = NTuple{2,Int64}[]
    @inbounds for p in 1:a_max
        @inbounds for q in 1:p
            if Orb[p].l == Orb[q].l && Orb[p].j == Orb[q].j
                push!(lj_pairs,(p,q))
            end
        end
    end

    # OBDM correction ...
    pRho, nRho = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)

    @inbounds for J in 0:J_max
        J_hat = hat(2*J)
        @inbounds for P in 1:2
            N_ac = Orb_NN.N[1,P,J+1]
            if N_ac == 0
                continue
            end

            ppS, pnS, nnS = OBDM_dense(Sigma,P,J)

            # Index of the ordered pair ab in this block ... 0 if not allowed (J, N_2max or parity) ...
            function pair(a::Int64,b::Int64)
                if parity(Orb[a].l + Orb[b].l) != P
                    return 0
                end
                return Int64(Orb_NN.Map[a,b,J+1,1])
            end

            # Ordered pairs of the block ... particle-particle & hole-hole pairs for pp, pn (first index proton) & nn ...
            ac_pairs = Orb_NN.Ind[1,P,J+1]
            pp_pp = [ac for ac in 1:N_ac if p_par[ac_pairs[ac][1]] && p_par[ac_pairs[ac][2]]]
            pp_pn = [ac for ac in 1:N_ac if p_par[ac_pairs[ac][1]] && n_par[ac_pairs[ac][2]]]
            pp_nn = [ac for ac in 1:N_ac if n_par[ac_pairs[ac][1]] && n_par[ac_pairs[ac][2]]]
            hh_pp = [ac for ac in 1:N_ac if p_hol[ac_pairs[ac][1]] && p_hol[ac_pairs[ac][2]]]
            hh_pn = [ac for ac in 1:N_ac if p_hol[ac_pairs[ac][1]] && n_hol[ac_pairs[ac][2]]]
            hh_nn = [ac for ac in 1:N_ac if n_hol[ac_pairs[ac][1]] && n_hol[ac_pairs[ac][2]]]

            # Contraction sum_(cd in List) S[cd,ar] S[cd,br] ...
            function contract(S::Matrix{Float64},List::Vector{Int64},ar::Int64,br::Int64)
                if ar == 0 || br == 0
                    return 0.0
                end
                Sum = 0.0
                @inbounds for cd in List
                    Sum += S[cd,ar] * S[cd,br]
                end
                return Sum
            end

            # Each (p,q) is written by one task only ...
            @inbounds Threads.@threads :dynamic for (p,q) in lj_pairs
                Amp = J_hat / hat(Orb[p].j)
                pSum, nSum = 0.0, 0.0

                @inbounds for r in 1:a_max
                    # Particle part ... p, q particles, r particle, holes cd ...
                    if p_par[p] && p_par[q]
                        if p_par[r]
                            pSum += 0.5 * contract(ppS,hh_pp,pair(p,r),pair(q,r))
                        end
                        if n_par[r]
                            pSum += contract(pnS,hh_pn,pair(p,r),pair(q,r))
                        end
                    end
                    if n_par[p] && n_par[q]
                        if n_par[r]
                            nSum += 0.5 * contract(nnS,hh_nn,pair(p,r),pair(q,r))
                        end
                        if p_par[r]
                            nSum += contract(pnS,hh_pn,pair(r,p),pair(r,q))
                        end
                    end

                    # Hole part ... p, q holes, r hole, particles ab ...
                    if p_hol[p] && p_hol[q]
                        if p_hol[r]
                            pSum -= 0.5 * contract(ppS,pp_pp,pair(p,r),pair(q,r))
                        end
                        if n_hol[r]
                            pSum -= contract(pnS,pp_pn,pair(p,r),pair(q,r))
                        end
                    end
                    if n_hol[p] && n_hol[q]
                        if n_hol[r]
                            nSum -= 0.5 * contract(nnS,pp_nn,pair(p,r),pair(q,r))
                        end
                        if p_hol[r]
                            nSum -= contract(pnS,pp_pn,pair(r,p),pair(r,q))
                        end
                    end
                end

                pRho[p,q] += Amp * pSum
                nRho[p,q] += Amp * nSum
            end
        end
    end

    # Symmetrize ... only p >= q was computed ... & add the occupations of Orb ...
    @inbounds for (p,q) in lj_pairs
        pRho[q,p], nRho[q,p] = pRho[p,q], nRho[p,q]
    end

    @inbounds for a in 1:a_max
        pRho[a,a] += Orb[a].pO
        nRho[a,a] += Orb[a].nO
    end

    return O1B(pRho,nRho)
end

#=
    RPA OBDM at the O(Y^4) level (renormalized RPA) ...
        Rho_new = Rho + dRho ,  occupation differences D_ph = rho_h - rho_p taken from the diagonal of Rho ...

        M_nu,mu = -1/2 sum_qg D_qg^2 X_qg,nu X*_qg,mu                  ... 1p-1h pairs qg of one isospin (pM protons, nM neutrons)
        G_ph,qh = sum_nu,mu Y_ph,nu (delta_nu,mu + M_nu,mu) Y*_qh,mu   ... G = Y (1 + M) Y^+
        dRho_pq = + 1/(2j_p + 1) sum_(h,J) (2J + 1) D_ph D_qh Re G_ph,qh   ... p, q particles of equal l & j
        dRho_hg = - 1/(2j_h + 1) sum_(p,J) (2J + 1) D_ph D_pg Re G_ph,pg   ... h, g holes of equal l & j
    Only the particle & hole orbitals of Orb_ph with D >= 1e-8 contribute ...

    Performance ... M & G are dense matrix products (BLAS) ... each (J,P) block is one task on the Julia threads (largest blocks first)
    with single-threaded BLAS inside (faster than multi-threaded BLAS on the moderate block sizes, BLAS threads are restored at the end),
    the OBDM elements are then independent tasks distributed over the Julia threads (no Dict look-ups inside the nu, mu sums) ...
    pM, nM ... storage of the matrices M, overwritten ...
=#
function RPA_OBDM_oY4(Params::Parameters,Orb::Vector{Orb1B},Orb_ph::Orb1B_RPA_OBDM,Orb_1p1h::Orb2B_1p1h,Rho::O1B,X::Matrix{Matrix{ComplexF64}},Y::Matrix{Matrix{ComplexF64}},pM::Matrix{Matrix{ComplexF64}},nM::Matrix{Matrix{ComplexF64}})
    # Read calculation parameters ..
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1
    a_max = div((N_max + 1)*(N_max + 2),2)

    Ortho = Params.Calc.RPA.Ortho

    # Old OBDM ... the occupation differences D_ph come from its diagonal ...
    pRho_old, nRho_old = Rho.p, Rho.n

    # Particle & hole flags of the orbitals in Orb_ph ...
    p_par, p_hol = falses(a_max), falses(a_max)
    n_par, n_hol = falses(a_max), falses(a_max)

    @inbounds for Orb_a in Orb_ph.p.p
        p_par[Orb_a.a] = true
    end
    @inbounds for Orb_a in Orb_ph.h.p
        p_hol[Orb_a.a] = true
    end
    @inbounds for Orb_a in Orb_ph.p.n
        n_par[Orb_a.a] = true
    end
    @inbounds for Orb_a in Orb_ph.h.n
        n_hol[Orb_a.a] = true
    end

    # 1) Matrices M & G = Y (1 + M) Y^+ of all (J,P) blocks ... one block per Julia thread, single-threaded BLAS inside ...
    pG = Matrix{Matrix{ComplexF64}}(undef,2,J_max+1)
    nG = Matrix{Matrix{ComplexF64}}(undef,2,J_max+1)

    # List of all (J,P) blocks ... sorted by the block size, largest blocks first (better load balance of the threads) ...
    JP_list = JP_initialize(J_max)

    JP_size = zeros(Int64,length(JP_list))
    @inbounds for (i_JP,JP) in enumerate(JP_list)
        J, P = JP[1], JP[2]
        JP_size[i_JP] = Orb_1p1h.N[P,J+1]
    end

    JP_list = JP_list[sortperm(JP_size, rev = true)]

    BLAS_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)

    @inbounds Threads.@threads :dynamic for i_JP in 1:length(JP_list)
        J, P = JP_list[i_JP][1], JP_list[i_JP][2]
        N_ph = Orb_1p1h.N[P,J+1]

        if N_ph == 0
            pM[P,J+1], nM[P,J+1] = zeros(ComplexF64,0,0), zeros(ComplexF64,0,0)
            pG[P,J+1], nG[P,J+1] = zeros(ComplexF64,0,0), zeros(ComplexF64,0,0)
            continue
        end

        Orb_1p1h_JP = Orb_1p1h.Ind[P,J+1]

        # Physical RPA solutions only ... the spurious 1- Goldstone level (i_nu = 1 of the orthogonalized 1- block) is skipped,
        # i.e. M is (N_ph - 1) x (N_ph - 1) in that block, while G = Y (1 + M) Y^+ stays N_ph x N_ph ...
        nu_start = RPA_spurious(Ortho,J,P) ? 2 : 1

        X_JP = view(X[P,J+1],1:N_ph,nu_start:N_ph)
        Y_JP = view(Y[P,J+1],1:N_ph,nu_start:N_ph)

        # Weights D_qg^2 of the proton & neutron 1p-1h pairs qg ...
        pW, nW = zeros(Float64,N_ph), zeros(Float64,N_ph)

        @inbounds for (i_qg,Orb_qg) in enumerate(Orb_1p1h_JP)
            a_q, a_g, T_qg = Orb_qg.a, Orb_qg.b, Orb_qg.T

            if T_qg == -1 && p_par[a_q] && p_hol[a_g]
                D_qg = pRho_old[a_g,a_g] - pRho_old[a_q,a_q]
                if D_qg >= 1e-8
                    pW[i_qg] = D_qg^2
                end
            elseif T_qg == 1 && n_par[a_q] && n_hol[a_g]
                D_qg = nRho_old[a_g,a_g] - nRho_old[a_q,a_q]
                if D_qg >= 1e-8
                    nW[i_qg] = D_qg^2
                end
            end
        end

        # M = -1/2 X^T diag(W) X* ...
        pM[P,J+1] = -0.5 .* (transpose(X_JP) * (pW .* conj.(X_JP)))
        nM[P,J+1] = -0.5 .* (transpose(X_JP) * (nW .* conj.(X_JP)))

        # G = Y (1 + M) Y^+ ...
        pG[P,J+1] = Y_JP * ((I + pM[P,J+1]) * adjoint(Y_JP))
        nG[P,J+1] = Y_JP * ((I + nM[P,J+1]) * adjoint(Y_JP))
    end

    BLAS.set_num_threads(BLAS_threads)

    # 2) List of the OBDM elements (T, Type, a, b) ... a, b of equal l & j, Type = :p particles (a, b in Orb_ph.p), Type = :h holes (a, b in Orb_ph.h) ...
    Tasks = Tuple{Int64,Symbol,Int64,Int64}[]

    @inbounds for (T,Par,Hol) in ((-1,Orb_ph.p.p,Orb_ph.h.p),(1,Orb_ph.p.n,Orb_ph.h.n))
        @inbounds for Orb_a in Par
            @inbounds for Orb_b in Par
                if Orb_a.l == Orb_b.l && Orb_a.j == Orb_b.j
                    push!(Tasks,(T,:p,Orb_a.a,Orb_b.a))
                end
            end
        end

        @inbounds for Orb_a in Hol
            @inbounds for Orb_b in Hol
                if Orb_a.l == Orb_b.l && Orb_a.j == Orb_b.j
                    push!(Tasks,(T,:h,Orb_a.a,Orb_b.a))
                end
            end
        end
    end

    N_task = length(Tasks)
    dRho = zeros(Float64,N_task)

    # 3) OBDM elements ... every task writes only its own dRho[i_task] ...
    @inbounds Threads.@threads :dynamic for i_task in 1:N_task
        T, Type, a, b = Tasks[i_task]
        l_a, j_a = Orb[a].l, Orb[a].j

        if T == -1
            Rho_T, G_T, Par, Hol = pRho_old, pG, Orb_ph.p.p, Orb_ph.h.p
        else
            Rho_T, G_T, Par, Hol = nRho_old, nG, Orb_ph.p.n, Orb_ph.h.n
        end

        Sum = 0.0

        if Type == :p
            # Particles a = p & b = q ... sum over the holes h ...
            @inbounds for Orb_h in Hol
                a_h, l_h, j_h = Orb_h.a, Orb_h.l, Orb_h.j

                D_ah, D_bh = Rho_T[a_h,a_h] - Rho_T[a,a], Rho_T[a_h,a_h] - Rho_T[b,b]

                if D_ah < 1e-8 || D_bh < 1e-8
                    continue
                end

                P = parity(l_a + l_h)

                @inbounds for J in div(abs(j_a - j_h),2):div(j_a + j_h,2)
                    i_ah = orb2b_1p1h_index(a,a_h,J,P,T,Orb_1p1h)
                    i_bh = orb2b_1p1h_index(b,a_h,J,P,T,Orb_1p1h)

                    if i_ah == 0 || i_bh == 0
                        continue
                    end

                    ME = hat(2*J) * D_ah * D_bh * real(G_T[P,J+1][i_ah,i_bh])
                    Sum += ME
                end
            end
        elseif Type == :h
            # Holes a = h & b = g ... sum over the particles p ...
            @inbounds for Orb_p in Par
                a_p, l_p, j_p = Orb_p.a, Orb_p.l, Orb_p.j

                D_pa, D_pb = Rho_T[a,a] - Rho_T[a_p,a_p], Rho_T[b,b] - Rho_T[a_p,a_p]

                if D_pa < 1e-8 || D_pb < 1e-8
                    continue
                end

                P = parity(l_p + l_a)

                @inbounds for J in div(abs(j_p - j_a),2):div(j_p + j_a,2)
                    i_pa = orb2b_1p1h_index(a_p,a,J,P,T,Orb_1p1h)
                    i_pb = orb2b_1p1h_index(a_p,b,J,P,T,Orb_1p1h)

                    if i_pa == 0 || i_pb == 0
                        continue
                    end

                    ME = hat(2*J) * D_pa * D_pb * real(G_T[P,J+1][i_pa,i_pb])
                    Sum -= ME
                end
            end
        end

        dRho[i_task] = Sum / hat(j_a)
    end

    # 4) New OBDM ... Rho_new = Rho_old + dRho ...
    pRho_new, nRho_new = copy(pRho_old), copy(nRho_old)

    @inbounds for i_task in 1:N_task
        T, Type, a, b = Tasks[i_task]
        if T == -1
            pRho_new[a,b] += dRho[i_task]
        else
            nRho_new[a,b] += dRho[i_task]
        end
    end

    return O1B(pRho_new,nRho_new)
end

#=
    Iterative (renormalized RPA) solution for the 1-body density matrix (OBDM) ... always at the O(Y^4) level, RPA_OBDM_oY4().

    Adapted from HF_RRPA_OBDM_iteration() in HF_RRPA/HF_RRPA_OBDM.jl.

    Bases:
        LHO --(C_LHO_HF)--> HF --(C_HF_NAT)--> NAT (current, input) --(C_tot)--> NAT (new, output)

    Each iteration:
        1) Rho_new = Rho_ref + dRho[Rho_old]               ... occupation differences D_ph = rho_h - rho_p taken from Rho_old
        2) (optionally) keep only the diagonal of Rho_new  ... Params.Calc.RPA.OBDM_diagonal
        3) linear mixing: Rho_new = alpha * Rho_new + (1 - alpha) * Rho_old
        4) natural orbitals: Rho_new = C * diag(N) * C'    ... rho_natural_orbitals()
        5) rotate Rho_old, Rho_ref, X & Y into the new natural-orbital (NAT) basis
        6) convergence measure (see below) ->  stop if delta < Tol

    Convergence measure ... all matrices in the new NAT basis, degeneracy-weighted Frobenius norm
        ||O||_w = sqrt( sum_(T = p,n) sum_(a,b) (2j_a + 1) |O_ab|^2 )      ... invariant under the rotations within the (l,j) blocks,
                                                                              in units of nucleons (||Rho_ref||_w ~ sqrt(A) for HF)
        R         = (Rho_new - Rho_old) / alpha                      ... unmixed residual Rho_ref + dRho[Rho_old] - Rho_old
        delta     = ||R||_w / ||Rho_new - Rho_ref||_w                ... change relative to the RPA correction of the OBDM itself,
                                                                         i.e. the relative precision of the depletion (used for Tol)
        delta_abs = ||R||_w                                          ... absolute rms change (printed)
        dn_max    = max_a |R_aa|                                     ... largest change of an occupation number (printed)
    The iteration is stopped as diverging if ||R||_w exceeds ||Rho_ref||_w (a change larger than the whole reference OBDM) ...

    If the iteration does not converge (150 iterations reached, cycling between two solutions or divergence), it falls back
    to a single explicit evaluation:
        Rho = Rho_ref + dRho[Rho_ini]   ... with the input X, Y.

    The orbitals Orb, the particle/hole classes Orb_ph & the 1p-1h structure Orb_1p1h stay fixed during the iteration.
    Only at the very end (converged or fallback solution) the occupations & the transformation matrices are updated.

    Input:
        Orb       ... single-particle orbitals of the current NAT basis
        Orb_ph    ... particle & hole orbitals (fixed, e.g. from RPA_OBDM_orbitals_make() of the reference)
        Orb_1p1h  ... 1p-1h orbitals (fixed), indexing of X & Y
        X, Y      ... RPA amplitudes in the current NAT basis (not modified, rotated copies are made)
        Rho_ini   ... starting OBDM in the current NAT basis (its diagonal gives the initial D_ph)
        C_HF_NAT  ... HF -> current NAT basis; MODIFIED IN PLACE to HF -> new NAT basis

    The reference OBDM (rho.bin) & C_LHO_HF (C.bin) are read from the reference solution directory.

    Output:
        Rho       ... OBDM in the new NAT basis (diagonal)
        C_LHO_NAT ... LHO -> new NAT basis
        Orb_NAT   ... copy of Orb with the new occupations pO, nO
=#
function HF_ScRPA_OBDM_solve(Params::Parameters,Orb::Vector{Orb1B},Orb_ph::Orb1B_RPA_OBDM,Orb_1p1h::Orb2B_1p1h,X::Matrix{Matrix{ComplexF64}},Y::Matrix{Matrix{ComplexF64}},Rho_ini::O1B,C_HF_NAT::O1B)
    # Read calculation parameters ...
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1)*(N_max + 2),2)

    epsilon = 1e-8
    Iteration_max = 150
    Diagonal_OBDM = Params.Calc.RPA.OBDM_diagonal

    alpha = Params.Calc.RPA.OBDM_alpha

    # Storage for the matrices M of RPA_OBDM_oY4() ...
    pM = Matrix{Matrix{ComplexF64}}(undef,size(X))
    nM = Matrix{Matrix{ComplexF64}}(undef,size(X))

    # Define local functions ...
        # One explicit evaluation of the OBDM ... Rho_new = Rho_ref + dRho[Rho_old] (RPA_OBDM_oY4 returns Rho_old + dRho) ...
    function OBDM_update(Rho_ref::O1B,Rho_old::O1B,X_it::Matrix{Matrix{ComplexF64}},Y_it::Matrix{Matrix{ComplexF64}})
        Rho_RPA = RPA_OBDM_oY4(Params,Orb,Orb_ph,Orb_1p1h,Rho_old,X_it,Y_it,pM,nM)

        pRho = Rho_ref.p .+ (Rho_RPA.p .- Rho_old.p)
        nRho = Rho_ref.n .+ (Rho_RPA.n .- Rho_old.n)

        return O1B(pRho,nRho)
    end
        # Natural orbitals function returning the transformation matrix C & density matrix Rho in its diagonal form ...
    function OBDM_natural(Rho::O1B)
        _, C, _ = rho_natural_orbitals(Params,Orb,Rho)

        Rho_NAT = O1b_diagonal(O1b_transformation(Rho,C))

        return C, Rho_NAT
    end
        # Frobenius norm (measure) for 1-body J-scheme operator ... ||O||_w = sqrt( sum_(a,b) (2j_a + 1) (|pO_ab|^2 + |nO_ab|^2) ) ...
    function OBDM_norm(pO::Matrix{Float64},nO::Matrix{Float64})
        Sum = 0.0
        @inbounds for Orb_a in Orb
            a, j_a = Orb_a.a, Orb_a.j
            @inbounds for b in 1:a_max
                ME = hat(j_a) * (pO[a,b]^2 + nO[a,b]^2)
                Sum += ME
            end
        end
        return sqrt(Sum)
    end

    # Import the reference data ... reference OBDM (HF basis) & transformation LHO -> HF ...
    IO_HF = "io/" * Params.Calc.Path * "/" * Params.Calc.RPA.Reference
    Rho_HF = O1b_import(Params,Orb,IO_HF * "/bin/rho.bin")
    C_LHO_HF = O1b_import(Params,Orb,IO_HF * "/bin/C.bin")

    # Reference OBDM in the current NAT basis ...
    Rho_ref = O1b_transformation(Rho_HF,C_HF_NAT)

    # Iteration objects ... (the input amplitudes & OBDMs are never modified, the rotations create new objects)
    X_it, Y_it = X, Y
    Rho_old = O1b_deepcopy(Rho_ini)
    Rho_ref_it = O1b_deepcopy(Rho_ref)

    # Total transformation matrix (current NAT basis -> new NAT basis) ...
    C_tot = O1B(diagm(ones(Float64,a_max)),diagm(ones(Float64,a_max)))

    # Size of the reference OBDM ... the scale for the divergence check (basis invariant) ...
    Norm_ref = OBDM_norm(Rho_ref.p,Rho_ref.n)

    # Iteration control ...
    Iteration = 0
    Converged = false
    delta, delta_odd, delta_even = 1.0, 1.0, 1.0

    println("\n\tStarting RPA O(Y^4) OBDM iteration ...")

    while Iteration < Iteration_max
        # 1) New OBDM ... Rho_ref + dRho[Rho_old] ...
        Rho_new = OBDM_update(Rho_ref_it,Rho_old,X_it,Y_it)

        # 2) Keep only the diagonal if the diagonal approximation is set ...
        if Diagonal_OBDM
            Rho_new = O1b_diagonal(Rho_new)
        end

        # 3) Linear mixing with the old OBDM (both in the current NAT basis) ...
        Rho_new = O1B(alpha .* Rho_new.p .+ (1.0 - alpha) .* Rho_old.p,
                      alpha .* Rho_new.n .+ (1.0 - alpha) .* Rho_old.n)

        # Stop if the iteration diverged (Inf or NaN in the OBDM) ...
        if !all(isfinite, Rho_new.p) || !all(isfinite, Rho_new.n)
            println("\n\tOBDM iteration stopped - diverging solution (Inf or NaN) ...")
            break
        end

        # 4) Natural orbitals ... Orb stays fixed during the iteration ...
        C_step, Rho_new = OBDM_natural(Rho_new)

        # 5) Rotate everything into the new NAT basis ...
        C_tot = O1B(C_tot.p * C_step.p, C_tot.n * C_step.n)

        Rho_old = O1b_transformation(Rho_old,C_step)
        Rho_ref_it = O1b_transformation(Rho_ref_it,C_step)

        X_it, Y_it = RPA_OBDM_transform_amplitudes(Orb_1p1h,X_it,Y_it,C_step)

        # 6) Convergence measures ... unmixed residual R = (Rho_new - Rho_old) / alpha, weighted norms (see above) ...
        pR, nR = (Rho_new.p .- Rho_old.p) ./ alpha, (Rho_new.n .- Rho_old.n) ./ alpha

        delta_abs = OBDM_norm(pR,nR)
        Norm_cor = OBDM_norm(Rho_new.p .- Rho_ref_it.p, Rho_new.n .- Rho_ref_it.n)

        if Norm_cor > 1e-12
            delta = delta_abs / Norm_cor
        else
            delta = delta_abs
        end

        Iteration += 1

        @printf("\t\tOBDM iteration:   Iteration number =  %4d   delta = %12.9f\n", Iteration, delta)

        # Update the old OBDM ...
        Rho_old = Rho_new

        # Diverging ... a change larger than the whole reference OBDM ...
        if (isfinite(delta) == false) || (delta_abs > Norm_ref)
            println("\n\tOBDM iteration stopped - diverging solution ...")
            break
        end

        # Converged ...
        if delta < epsilon
            Converged = true
            break
        end

        # Detect solutions cycling between two states ... delta stalls relative to its own size ...
        if Iteration > 100
            if iseven(Iteration)
                delta_prev = delta_even
            else
                delta_prev = delta_odd
            end

            if abs(delta - delta_prev) < 1e-3 * delta
                println("\n\tOBDM iteration stopped - cycled degenerate solution found ...")
                break
            end

            if iseven(Iteration)
                delta_even = delta
            else
                delta_odd = delta
            end
        end
    end

    if Converged
        println("\n\tRPA O(Y^4) OBDM iteration has converged ...")
        Rho = Rho_old

    else
        # Not converged ... single explicit evaluation Rho = Rho_ref + dRho[Rho_ini] with the input amplitudes ...
        println("\n\tRPA O(Y^4) OBDM iteration has not converged - using a single explicit evaluation of the OBDM ...")

        Rho_expl = OBDM_update(Rho_ref,Rho_ini,X,Y)

        if Diagonal_OBDM
            Rho_expl = O1b_diagonal(Rho_expl)
        end

        C_tot, Rho = OBDM_natural(Rho_expl)
    end

    # Update the occupations of the orbitals ... (a new copy of Orb with the diagonal of Rho)
    Orb_NAT = Vector{Orb1B}(undef,length(Orb))

    @inbounds for Orb_a in Orb
        a, n_a, l_a, j_a = Orb_a.a, Orb_a.n, Orb_a.l, Orb_a.j
        pO_a, nO_a = Rho.p[a,a], Rho.n[a,a]
        Orb_NAT[a] = Orb1B(a,n_a,l_a,j_a,pO_a,nO_a)
    end

    # Update the transformation matrices ... HF -> new NAT (in place) & LHO -> new NAT ...
    C_HF_NAT.p .= C_HF_NAT.p * C_tot.p
    C_HF_NAT.n .= C_HF_NAT.n * C_tot.n

    C_LHO_NAT = O1B(C_LHO_HF.p * C_HF_NAT.p, C_LHO_HF.n * C_HF_NAT.n)

    return Rho, C_LHO_NAT, Orb_NAT
end

function RPA_OBDM_transform_amplitudes(Orb_1p1h::Orb2B_1p1h,X::Matrix{Matrix{ComplexF64}},Y::Matrix{Matrix{ComplexF64}},C::O1B)
    # Prepare arrays for transformed X & Y amplitudes ... (empty blocks keep the old ones, the others are replaced)
    X_new, Y_new = copy(X), copy(Y)

    # List of all (P, J) blocks ... sorted by the block size, largest blocks first (better load balance of the threads) ...
    PJ_list = Tuple{Int64,Int64}[]
    PJ_size = Int64[]

    @inbounds for J_ind in 1:size(Orb_1p1h.N,2)
        @inbounds for P in 1:size(Orb_1p1h.N,1)
            push!(PJ_list,(P,J_ind))
            push!(PJ_size,Orb_1p1h.N[P,J_ind])
        end
    end

    PJ_list = PJ_list[sortperm(PJ_size, rev = true)]

    BLAS_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)

    @inbounds Threads.@threads :dynamic for i_PJ in 1:length(PJ_list)
        P, J_ind = PJ_list[i_PJ][1], PJ_list[i_PJ][2]

        N_ph = Orb_1p1h.N[P,J_ind]
        if N_ph == 0
            continue
        end

        Orb_1p1h_JP = Orb_1p1h.Ind[P,J_ind]

        # Build the 2-body (1p-1h) transformation matrix U ...
        U = zeros(Float64,N_ph,N_ph)

        @inbounds for i_ph in 1:N_ph
            Orb_ph = Orb_1p1h_JP[i_ph]
            a_p, a_h, T_ph = Orb_ph.a, Orb_ph.b, Orb_ph.T

            if T_ph == -1
                C_T = C.p
            else
                C_T = C.n
            end

            @inbounds for (i_qg,Orb_qg) in enumerate(Orb_1p1h_JP)
                a_q, a_g, T_qg = Orb_qg.a, Orb_qg.b, Orb_qg.T

                if T_qg == T_ph
                    U[i_ph,i_qg] = C_T[a_q,a_p] * C_T[a_g,a_h]
                end
            end
        end

        X_new[P,J_ind] = U * X[P,J_ind]
        Y_new[P,J_ind] = U * Y[P,J_ind]
    end

    BLAS.set_num_threads(BLAS_threads)

    return X_new, Y_new
end
