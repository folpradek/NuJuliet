function TDA_rM(Params::Parameters,Orb::Vector{Orb1B},Orb_1p1h::Orb2B_1p1h,Orb_NN::Orb2B,M::Tr1B,Rho::O1B,Sigma::O2B,X::Matrix{Matrix{Float64}})
    # Evaluate the reduced transition matrix elements M^lambda ... exact transition matrix elements <nu||M||0> = <0|Q_nu M|0> with respect
    # to the reference ground state (reference OBDM Rho & for Reference_TBCF = true the reference 2-body correlation function Sigma) ...
    println("\nCalculating reduced transition matrix elements rM ... (Exact)")

    # Case of E0 ...
    J, P = 0, 1
    M_E0 = [M.E0]
    prM_E0, nrM_E0 = TDA_rM_calculate(Params,J,P,M_E0,Orb,Orb_1p1h,Orb_NN,Rho,Sigma,X)

    # Case of E1 ...
    J, P = 1, 2
    M_E1 = [M.E1,M.E1_VC,M.E1_VS,M.E1_TC,M.E1_TS,M.E1_sTC,M.E1_sTS,M.E1_C,M.E1_sC]
    prM_E1, nrM_E1 = TDA_rM_calculate(Params,J,P,M_E1,Orb,Orb_1p1h,Orb_NN,Rho,Sigma,X)

    # Case of E2 ...
    J, P = 2, 1
    M_E2 = [M.E2]
    prM_E2, nrM_E2 = TDA_rM_calculate(Params,J,P,M_E2,Orb,Orb_1p1h,Orb_NN,Rho,Sigma,X)

    # Case of E3 ...
    J, P = 3, 2
    M_E3 = [M.E3]
    prM_E3, nrM_E3 = TDA_rM_calculate(Params,J,P,M_E3,Orb,Orb_1p1h,Orb_NN,Rho,Sigma,X)

    # Allocate the reduced multipole transition matrix elements ... (the shared ReducedMultipole structure is complex, the values are real)
    rM_E0 = pnCVector(prM_E0[1,:],nrM_E0[1,:])
    rM_E1 = pnCVector(prM_E1[1,:],nrM_E1[1,:])
    rM_E2 = pnCVector(prM_E2[1,:],nrM_E2[1,:])
    rM_E3 = pnCVector(prM_E3[1,:],nrM_E3[1,:])
    rM_E1VC = pnCVector(prM_E1[2,:],nrM_E1[2,:])
    rM_E1VS = pnCVector(prM_E1[3,:],nrM_E1[3,:])
    rM_E1TC = pnCVector(prM_E1[4,:],nrM_E1[4,:])
    rM_E1TS = pnCVector(prM_E1[5,:],nrM_E1[5,:])
    rM_E1sTC = pnCVector(prM_E1[6,:],nrM_E1[6,:])
    rM_E1sTS = pnCVector(prM_E1[7,:],nrM_E1[7,:])
    rM_E1C = pnCVector(prM_E1[8,:],nrM_E1[8,:])
    rM_E1sC = pnCVector(prM_E1[9,:],nrM_E1[9,:])

    rM = ReducedMultipole(rM_E0,rM_E1,rM_E2,rM_E3,rM_E1VC,
                          rM_E1VS,rM_E1TC,rM_E1TS,rM_E1sTC,
                          rM_E1sTS,rM_E1C,rM_E1sC)

    println("\tReduced 1-body transition matrix elements rM calculated ...")

    return rM
end

function TDA_rM_calculate(Params::Parameters,J::Int64,P::Int64,M_JP::Vector{O1B},Orb::Vector{Orb1B},Orb_1p1h::Orb2B_1p1h,Orb_NN::Orb2B,Rho::O1B,Sigma::O2B,X::Matrix{Matrix{Float64}})
    # Evaluate the reduced transition matrix elements of all operators M_JP of the multipole (J,P) ...
        # rM^t[i_op,nu] = sum_(ab) M^t_ab[i_op] * Gamma^t_ab(nu) ,   Gamma(nu) = D * X[:,nu] ... (see TDA_transition_kernel)
    Pairs, D = TDA_transition_kernel(Params,J,P,Orb,Orb_1p1h,Orb_NN,Rho,Sigma)

    # Matrix elements of the operators for the pair densities ... proton operators for proton pairs & neutron operators for neutron pairs ...
    pM_pair, nM_pair = TDA_pair_operators(Pairs,M_JP)

    # Transition density matrices Gamma = D * X (BLAS) & contraction with the operators ...
    Gamma = D * X[P,J+1]

    prM = ComplexF64.(pM_pair * Gamma)
    nrM = ComplexF64.(nM_pair * Gamma)

    return prM, nrM
end

function TDA_pair_operators(Pairs::Vector{Ind2B},M_JP::Vector{O1B})
    # Matrix elements M^t_ab of the operators M_JP for the orbital pairs (a,b,t) ... zero for the other isospin ...
    N_o, N_pair = length(M_JP), length(Pairs)

    pM_pair, nM_pair = zeros(Float64,N_o,N_pair), zeros(Float64,N_o,N_pair)

    @inbounds for (i_pair,Pair) in enumerate(Pairs)
        a, b, t = Pair.a, Pair.b, Pair.T
        @inbounds for i_o in 1:N_o
            if t == -1
                pM_pair[i_o,i_pair] = M_JP[i_o].p[a,b]
            else
                nM_pair[i_o,i_pair] = M_JP[i_o].n[a,b]
            end
        end
    end

    return pM_pair, nM_pair
end

function TDA_transition_kernel(Params::Parameters,J::Int64,P::Int64,Orb::Vector{Orb1B},Orb_1p1h::Orb2B_1p1h,Orb_NN::Orb2B,Rho::O1B,Sigma::O2B)
    # Comment on the exact TDA transition densities ...
        # The reduced exact transition density matrix of the TDA state nu with respect to the reference ground state,
        #
        #     Gamma^t_ab(nu) = sum_ph D[(a,b,t),ph] * X[ph,nu] ,      <nu||M||0> = sum_(a,b,t) M^t_ab * Gamma^t_ab(nu) ,
        #
        # is defined on the orbital pairs Pairs = (a,b,t) of the multipole (J,P). The kernel D is the X-term of the exact branch of
        # RPA_rM_contract (rM = <0|Q_nu M|0>, Y = 0) written before the contraction with the operator:
        #
        #   ... 1-body part ...   G = (1 - Rho) * M * Rho  ->  D[(c,d,t_ph),ph] += (-1)^J * (1 - Rho)_pc * Rho_dh ,
        #                         (diagonal Rho ... D[(p,h,t_ph),ph] = (-1)^J * rho_hh * (1 - rho_pp) only) ,
        #   ... 2-body part ...   (Reference_TBCF = true only) with the cross-coupled cumulant <ha;I|Sigma|pb;I> ...
        #                         D[(a,b,t_ph),ph] += (-1)^((j_p + j_h)/2) * K_1 (same isospin as ph, pp/nn part of Sigma) &
        #                         D[(a,b,-t_ph),ph] += (-1)^((j_p + j_h)/2) * K_2 (opposite isospin, pn part of Sigma) ,
        #                         K = sum_I (-1)^(J + I + (j_p + j_a)/2) (2I + 1) {j_p j_h J; j_a j_b I} <ha;I|Sigma|pb;I> ...
        #
        # K_2 multiplies the operator of the OPPOSITE isospin, hence it contributes to rM^(-t_ph) (as in RPA_rM_contract) ...
        # Without Sigma the pairs are the 1p-1h pairs (D diagonal for diagonal Rho), extended by the pairs (c,d,t) coupled through
        # the off-diagonal elements of a non-diagonal Rho; with Sigma all (a,b,t) of parity P coupled to J are used ...
        #
        # Benchmark ... identical to the M-scheme reference <0|Q_nu M|0> (random non-diagonal Rho, random Sigma, Nmax = 2 & 3) to 1e-14 ...
    TBCF = Params.Calc.TDA.Reference_TBCF
    Tol_rho = 1e-12

    Orb_ph_list = Orb_1p1h.Ind[P,J+1]
    N_ph = Orb_1p1h.N[P,J+1]
    a_max = length(Orb)

    # Orbital pairs (a,b,t) & their lookup table ...
    if TBCF == false
        Pairs = copy(Orb_ph_list)

        # Additional pairs (c,d,t) with (1 - Rho)_pc * Rho_dh != 0 for a non-diagonal Rho ... (none for a diagonal Rho)
        Known = Set{Tuple{Int64,Int64,Int64}}((Pair.a,Pair.b,Pair.T) for Pair in Pairs)
        @inbounds for Orb_ph in Orb_ph_list
            a_p, a_h, T_ph = Orb_ph.a, Orb_ph.b, Orb_ph.T
            Rho_1 = T_ph == -1 ? Rho.p : Rho.n
            @inbounds for c in 1:a_max, d in 1:a_max
                if abs(Float64(c == a_p) - Rho_1[a_p,c]) * abs(Rho_1[d,a_h]) > Tol_rho && !((c,d,T_ph) in Known)
                    push!(Pairs,Ind2B(c,d,T_ph))
                    push!(Known,(c,d,T_ph))
                end
            end
        end
    else
        Pairs = Ind2B[]
        @inbounds for t in (-1,1), Orb_a in Orb, Orb_b in Orb
            if parity(Orb_a.l + Orb_b.l) == P && triangle(Orb_a.j,Orb_b.j,2*J)
                push!(Pairs,Ind2B(Orb_a.a,Orb_b.a,t))
            end
        end
    end

    Pair_Map = zeros(Int64,a_max,a_max,2)
    @inbounds for (i_pair,Pair) in enumerate(Pairs)
        Pair_Map[Pair.a,Pair.b,div(Pair.T + 3,2)] = i_pair
    end

    D = zeros(Float64,length(Pairs),N_ph)

    # Evaluate the kernel for each ph pair ... (each thread writes its own column)
    @inbounds Threads.@threads for i_ph in 1:N_ph
        Orb_ph = Orb_ph_list[i_ph]
        a_p, a_h, T_ph = Orb_ph.a, Orb_ph.b, Orb_ph.T

        l_p, j_p = Orb[a_p].l, Orb[a_p].j
        l_h, j_h = Orb[a_h].l, Orb[a_h].j

        # Isospin indices of the pairs ... same (1) & opposite (2) isospin as the ph pair ...
        t_1, t_2 = div(T_ph + 3,2), div(-T_ph + 3,2)

        # 1-body part ... (1 - Rho)_pc * Rho_dh, Rho is a scalar, i.e. (l_c,j_c) = (l_p,j_p) & (l_d,j_d) = (l_h,j_h) ...
        Rho_1 = T_ph == -1 ? Rho.p : Rho.n

        @inbounds for c in 1:a_max
            (Orb[c].l == l_p && Orb[c].j == j_p) || continue
            R_pc = Float64(c == a_p) - Rho_1[a_p,c]

            @inbounds for d in 1:a_max
                (Orb[d].l == l_h && Orb[d].j == j_h) || continue
                i_pair = Pair_Map[c,d,t_1]

                # Pairs below the tolerance are not stored (TBCF = false) ...
                if i_pair == 0
                    continue
                end

                D[i_pair,i_ph] += R_pc * Rho_1[d,a_h] * phase(J)
            end
        end

        # 2-body part (reference 2-body correlation function Sigma) ...
        if TBCF == true
            Phase_X = phase(1 + div(j_p + j_h,2))

            @inbounds for Pair in Pairs
                Pair.T != T_ph && continue

                a_a, a_b = Pair.a, Pair.b
                l_a, j_a = Orb[a_a].l, Orb[a_a].j
                j_b = Orb[a_b].j

                # (the parity of the pair guarantees parity(l_h + l_a) == parity(l_p + l_b))
                P_ha = parity(l_h + l_a)

                # Define 2-body kernels ... same (1) & opposite (2) isospin of the orbitals a & b ...
                K_1, K_2 = 0.0, 0.0

                I_min = div(max(abs(j_h - j_a),abs(j_p - j_b)),2)
                I_max = div(min(j_h + j_a,j_p + j_b),2)

                @inbounds for I in I_min:I_max
                    F = phase(J + I + div(j_p + j_a,2)) * hat(2*I) * w6j(j_p,j_h,2*J,j_a,j_b,2*I)

                    if T_ph == -1
                        K_1 += O2b_pp(a_h,a_a,a_p,a_b,I,P_ha,Sigma,Orb,Orb_NN) * F
                        K_2 += O2b_pn(a_h,a_a,a_p,a_b,I,P_ha,Sigma,Orb_NN) * F
                    else
                        # pn element with the proton first ... both pairs exchanged, (ha) -> (ah) & (pb) -> (bp), phase (-1)^(j_h + j_a + j_p + j_b - 2I) ...
                        K_1 += O2b_nn(a_h,a_a,a_p,a_b,I,P_ha,Sigma,Orb,Orb_NN) * F
                        K_2 += O2b_pn(a_a,a_h,a_b,a_p,I,P_ha,Sigma,Orb_NN) * phase(div(j_p + j_a + j_h + j_b,2)) * F
                    end
                end

                D[Pair_Map[a_a,a_b,t_1],i_ph] += Phase_X * K_1
                D[Pair_Map[a_a,a_b,t_2],i_ph] += Phase_X * K_2
            end
        end
    end

    return Pairs, D
end

function TDA_rB(Params::Parameters,Orb_1p1h::Orb2B_1p1h,rM::ReducedMultipole,E::Matrix{Vector{Float64}})
    # Read parameters ...
    A = Params.Calc.A
    Z = Params.Calc.Z

    # Define basic constants ...
    hc = 197.326980
    g_p = 5.586
    g_n = -3.826

    println("\nCalculating the reduced transition intensities rB ...")

    # Case of E0 ...
    J, P = 0, 1
    N_nu = Orb_1p1h.N[P,J+1]

    # Initialize the reduced transition intensities B ...
    rB_E0 = Vector{Vector{Float64}}(undef,3)

    # Initialize the sub-components of the reduced transition intensities B ...
        # Physical (ph), Isoscalar (is), Isovector (iv) ...
    @inbounds for i in 1:3
        rB_E0[i] = Vector{Float64}(undef,N_nu)
    end

    # Evaluate the reduced transition intensities B ...
    @inbounds Threads.@threads for i_nu in 1:N_nu
        rB_E0[1][i_nu] = abs(rM.E0.p[i_nu])^2
        rB_E0[2][i_nu] = 0.25 * abs(rM.E0.p[i_nu] + rM.E0.n[i_nu])^2
        rB_E0[3][i_nu] = 0.25 * abs(rM.E0.p[i_nu] - rM.E0.n[i_nu])^2
    end

    # Case of E1 ...
    J, P = 1, 2
    N_nu = Orb_1p1h.N[P,J+1]

    # Initialize the reduced transition intensities B ...
    rB_E1 = Vector{Vector{Float64}}(undef,3)
    rB_E1V = Vector{Vector{Float64}}(undef,3)
    rB_E1VC = Vector{Vector{Float64}}(undef,3)
    rB_E1VS = Vector{Vector{Float64}}(undef,3)
    rB_E1T = Vector{Vector{Float64}}(undef,3)
    rB_E1TC = Vector{Vector{Float64}}(undef,3)
    rB_E1TS = Vector{Vector{Float64}}(undef,3)
    rB_E1C = Vector{Vector{Float64}}(undef,3)
    rB_E1CC = Vector{Vector{Float64}}(undef,3)
    rB_E1CS = Vector{Vector{Float64}}(undef,3)
    rB_E1c = Vector{Vector{Float64}}(undef,3)
    rB_E1_NLO_LWA = Vector{Vector{Float64}}(undef,3)

    # Initialize the sub-components of the reduced transition intensities B ...
        # Physical (ph), Isoscalar (is), Isovector (iv) ...
    @inbounds for i in 1:3
        rB_E1[i] = Vector{Float64}(undef,N_nu)
        rB_E1V[i] = Vector{Float64}(undef,N_nu)
        rB_E1VC[i] = Vector{Float64}(undef,N_nu)
        rB_E1VS[i] = Vector{Float64}(undef,N_nu)
        rB_E1T[i] = Vector{Float64}(undef,N_nu)
        rB_E1TC[i] = Vector{Float64}(undef,N_nu)
        rB_E1TS[i] = Vector{Float64}(undef,N_nu)
        rB_E1C[i] = Vector{Float64}(undef,N_nu)
        rB_E1CC[i] = Vector{Float64}(undef,N_nu)
        rB_E1CS[i] = Vector{Float64}(undef,N_nu)
        rB_E1c[i] = Vector{Float64}(undef,N_nu)
        rB_E1_NLO_LWA[i] = Vector{Float64}(undef,N_nu)
    end

    # Define effective E1 isovector charges ...
    e_p = Float64(A - Z) / Float64(A)
    e_n = Float64(Z) / Float64(A)

    # Evaluate the reduced transition intensities B ...
    @inbounds Threads.@threads for i_nu in 1:N_nu
        E_nu = E[P,J+1][i_nu]
            # Physical components ...
        rB_E1[1][i_nu] = abs(rM.E1.p[i_nu])^2
        rB_E1V[1][i_nu] = abs(rM.E1_VC.p[i_nu] + 0.5 * g_p * rM.E1_VS.p[i_nu] + 0.5 * g_n * rM.E1_VS.n[i_nu])^2
        rB_E1VC[1][i_nu] = abs(rM.E1_VC.p[i_nu])^2
        rB_E1VS[1][i_nu] = abs(0.5 * g_p * rM.E1_VS.p[i_nu] + 0.5 * g_n * rM.E1_VS.n[i_nu])^2

        rB_E1T[1][i_nu] = abs(rM.E1_TC.p[i_nu] + 0.5 * g_p * rM.E1_TS.p[i_nu] + 0.5 * g_n * rM.E1_TS.n[i_nu])^2
        rB_E1TC[1][i_nu] = abs(rM.E1_TC.p[i_nu])^2
        rB_E1TS[1][i_nu] = abs(0.5 * g_p * rM.E1_TS.p[i_nu] + 0.5 * g_n * rM.E1_TS.n[i_nu])^2

        rB_E1C[1][i_nu] = abs(rM.E1_TC.p[i_nu] + 0.5 * g_p * rM.E1_TS.p[i_nu] + 0.5 * g_n * rM.E1_TS.n[i_nu] - (rM.E1_VC.p[i_nu] + 0.5 * g_p * rM.E1_VS.p[i_nu] + 0.5 * g_n * rM.E1_VS.n[i_nu]))^2
        rB_E1CC[1][i_nu] = abs(rM.E1_TC.p[i_nu] - rM.E1_VC.p[i_nu])^2
        rB_E1CS[1][i_nu] = abs(0.5 * g_p * rM.E1_TS.p[i_nu] + 0.5 * g_n * rM.E1_TS.n[i_nu] - (0.5 * g_p * rM.E1_VS.p[i_nu] + 0.5 * g_n * rM.E1_VS.n[i_nu]))^2

        rB_E1c[1][i_nu] = abs(rM.E1_C.p[i_nu])^2

        rB_E1_NLO_LWA[1][i_nu] = abs(rM.E1.p[i_nu] + E_nu / hc * (rM.E1_TC.p[i_nu] + 0.5 * g_p * rM.E1_TS.p[i_nu] + 0.5 * g_n * rM.E1_TS.n[i_nu]))^2

            # Isoscalar components ...
        rB_E1[2][i_nu] = 0.25 * abs(rM.E1.p[i_nu] + rM.E1.n[i_nu])^2
        rB_E1V[2][i_nu] = abs(0.5 * (rM.E1_VC.p[i_nu] + rM.E1_VC.n[i_nu]) + 0.125 * (g_p + g_n) * (rM.E1_VS.p[i_nu] + rM.E1_VS.n[i_nu]))^2
        rB_E1VC[2][i_nu] = abs(0.5 * (rM.E1_VC.p[i_nu] + rM.E1_VC.n[i_nu]))^2
        rB_E1VS[2][i_nu] = abs(0.125 * (g_p + g_n) * (rM.E1_VS.p[i_nu] + rM.E1_VS.n[i_nu]))^2
        
        rB_E1T[2][i_nu] = abs(0.5 * (rM.E1_TC.p[i_nu] + rM.E1_sTC.p[i_nu] + rM.E1_TC.n[i_nu] + rM.E1_sTC.n[i_nu]) + 0.125 * (g_p + g_n) * (rM.E1_TS.p[i_nu] + rM.E1_sTS.p[i_nu] + rM.E1_TS.n[i_nu] + rM.E1_sTS.n[i_nu]))^2
        rB_E1TC[2][i_nu] = abs(0.5 * (rM.E1_TC.p[i_nu] + rM.E1_sTC.p[i_nu] + rM.E1_TC.n[i_nu] + rM.E1_sTC.n[i_nu]))^2
        rB_E1TS[2][i_nu] = abs(0.125 * (g_p + g_n) * (rM.E1_TS.p[i_nu] + rM.E1_sTS.p[i_nu] + rM.E1_TS.n[i_nu] + rM.E1_sTS.n[i_nu]))^2

        rB_E1C[2][i_nu] = abs(0.5 * (rM.E1_TC.p[i_nu] + rM.E1_sTC.p[i_nu] + rM.E1_TC.n[i_nu] + rM.E1_sTC.n[i_nu]) + 0.125 * (g_p + g_n) * (rM.E1_TS.p[i_nu] + rM.E1_sTS.p[i_nu] + rM.E1_TS.n[i_nu] + rM.E1_sTS.n[i_nu]) - (0.5 * (rM.E1_VC.p[i_nu] + rM.E1_VC.n[i_nu]) + 0.125 * (g_p + g_n) * (rM.E1_VS.p[i_nu] + rM.E1_VS.n[i_nu])))^2
        rB_E1CC[2][i_nu] = abs(0.5 * (rM.E1_TC.p[i_nu] + rM.E1_sTC.p[i_nu] + rM.E1_TC.n[i_nu] + rM.E1_sTC.n[i_nu]) - (0.5 * (rM.E1_VC.p[i_nu] + rM.E1_VC.n[i_nu])))^2
        rB_E1CS[2][i_nu] = abs(0.125 * (g_p + g_n) * (rM.E1_TS.p[i_nu] + rM.E1_sTS.p[i_nu] + rM.E1_TS.n[i_nu] + rM.E1_sTS.n[i_nu]) - (0.125 * (g_p + g_n) * (rM.E1_VS.p[i_nu] + rM.E1_VS.n[i_nu])))^2

        rB_E1c[2][i_nu] = 0.25 * abs(rM.E1_C.p[i_nu] + rM.E1_sC.p[i_nu] + rM.E1_C.n[i_nu] + rM.E1_sC.n[i_nu])^2

        rB_E1_NLO_LWA[2][i_nu] = 0.25 * abs(rM.E1.p[i_nu] + rM.E1.n[i_nu] + E_nu / hc * ((rM.E1_TC.p[i_nu] + rM.E1_sTC.p[i_nu] + rM.E1_TC.n[i_nu] + rM.E1_sTC.n[i_nu]) + 0.25 * (g_p + g_n) * (rM.E1_TS.p[i_nu] + rM.E1_sTS.p[i_nu] + rM.E1_TS.n[i_nu] + rM.E1_sTS.n[i_nu])))^2

            # Isovector components ...
        rB_E1[3][i_nu] = abs(e_p * rM.E1.p[i_nu] - e_n * rM.E1.n[i_nu])^2
        rB_E1V[3][i_nu] = abs(0.5 * (rM.E1_VC.p[i_nu] - rM.E1_VC.n[i_nu]) + 0.125 * (g_p - g_n) * (rM.E1_VS.p[i_nu] - rM.E1_VS.n[i_nu]))^2
        rB_E1VC[3][i_nu] = abs(0.5 * (rM.E1_VC.p[i_nu] - rM.E1_VC.n[i_nu]))^2
        rB_E1VS[3][i_nu] = abs(0.125 * (g_p - g_n) * (rM.E1_VS.p[i_nu] - rM.E1_VS.n[i_nu]))^2
        
        rB_E1T[3][i_nu] = abs(0.5 * (rM.E1_TC.p[i_nu] - rM.E1_TC.n[i_nu]) + 0.125 * (g_p - g_n) * (rM.E1_TS.p[i_nu] - rM.E1_TS.n[i_nu]))^2
        rB_E1TC[3][i_nu] = abs(0.5 * (rM.E1_TC.p[i_nu] - rM.E1_TC.n[i_nu]))^2
        rB_E1TS[3][i_nu] = abs(0.125 * (g_p - g_n) * (rM.E1_TS.p[i_nu] - rM.E1_TS.n[i_nu]))^2

        rB_E1C[3][i_nu] = abs(0.5 * (rM.E1_TC.p[i_nu] - rM.E1_TC.n[i_nu]) + 0.125 * (g_p - g_n) * (rM.E1_TS.p[i_nu] - rM.E1_TS.n[i_nu]) - (0.5 * (rM.E1_VC.p[i_nu] - rM.E1_VC.n[i_nu]) + 0.125 * (g_p - g_n) * (rM.E1_VS.p[i_nu] - rM.E1_VS.n[i_nu])))^2
        rB_E1CC[3][i_nu] = abs(0.5 * (rM.E1_TC.p[i_nu] - rM.E1_TC.n[i_nu]) - (0.5 * (rM.E1_VC.p[i_nu] - rM.E1_VC.n[i_nu])))^2
        rB_E1CS[3][i_nu] = abs(0.125 * (g_p - g_n) * (rM.E1_TS.p[i_nu] - rM.E1_TS.n[i_nu]) - (0.125 * (g_p - g_n) * (rM.E1_VS.p[i_nu] - rM.E1_VS.n[i_nu])))^2

        rB_E1c[3][i_nu] = 0.25 * abs(rM.E1_C.p[i_nu] - rM.E1_C.n[i_nu])^2

        rB_E1_NLO_LWA[3][i_nu] = abs(e_p * rM.E1.p[i_nu] - e_n * rM.E1.n[i_nu] + 0.5 * E_nu / hc * ((rM.E1_TC.p[i_nu] - rM.E1_TC.n[i_nu]) + 0.25 * (g_p - g_n) * (rM.E1_TS.p[i_nu] - rM.E1_TS.n[i_nu])))^2
    end

    # Case of E2 ...
    J, P = 2, 1
    N_nu = Orb_1p1h.N[P,J+1]

    # Initialize the reduced transition intensities B ...
    rB_E2 = Vector{Vector{Float64}}(undef,3)

    # Initialize the sub-components of the reduced transition intensities B ...
        # Physical (ph), Isoscalar (is), Isovector (iv) ...
    @inbounds for i in 1:3
        rB_E2[i] = Vector{Float64}(undef,N_nu)
    end

    # Evaluate the reduced transition intensities B ...
    @inbounds Threads.@threads for i_nu in 1:N_nu
        rB_E2[1][i_nu] = abs(rM.E2.p[i_nu])^2
        rB_E2[2][i_nu] = 0.25 * abs(rM.E2.p[i_nu] + rM.E2.n[i_nu])^2
        rB_E2[3][i_nu] = 0.25 * abs(rM.E2.p[i_nu] - rM.E2.n[i_nu])^2
    end

    # Case of E3 ...
    J, P = 3, 2
    N_nu = Orb_1p1h.N[P,J+1]

    # Initialize the reduced transition intensities B ...
    rB_E3 = Vector{Vector{Float64}}(undef,3)

    # Initialize the sub-components of the reduced transition intensities B ...
        # Physical (ph), Isoscalar (is), Isovector (iv) ...
    @inbounds for i in 1:3
        rB_E3[i] = Vector{Float64}(undef,N_nu)
    end

    # Evaluate the reduced transition intensities B ...
    @inbounds Threads.@threads for i_nu in 1:N_nu
        rB_E3[1][i_nu] = abs(rM.E3.p[i_nu])^2
        rB_E3[2][i_nu] = 0.25 * abs(rM.E3.p[i_nu] + rM.E3.n[i_nu])^2
        rB_E3[3][i_nu] = 0.25 * abs(rM.E3.p[i_nu] - rM.E3.n[i_nu])^2
    end

    # Allocate the reduced transition intensity matrix elements for the EX transitions ...
    rB_E0 = Transition(rB_E0[1],rB_E0[2],rB_E0[3])
    rB_E1 = Transition(rB_E1[1],rB_E1[2],rB_E1[3])
    rB_E2 = Transition(rB_E2[1],rB_E2[2],rB_E2[3])
    rB_E3 = Transition(rB_E3[1],rB_E3[2],rB_E3[3])
    rB_E1V = Transition(rB_E1V[1],rB_E1V[2],rB_E1V[3])
    rB_E1VC = Transition(rB_E1VC[1],rB_E1VC[2],rB_E1VC[3])
    rB_E1VS = Transition(rB_E1VS[1],rB_E1VS[2],rB_E1VS[3])
    rB_E1T = Transition(rB_E1T[1],rB_E1T[2],rB_E1T[3])
    rB_E1TC = Transition(rB_E1TC[1],rB_E1TC[2],rB_E1TC[3])
    rB_E1TS = Transition(rB_E1TS[1],rB_E1TS[2],rB_E1TS[3])
    rB_E1C = Transition(rB_E1C[1],rB_E1C[2],rB_E1C[3])
    rB_E1CC = Transition(rB_E1CC[1],rB_E1CC[2],rB_E1CC[3])
    rB_E1CS = Transition(rB_E1CS[1],rB_E1CS[2],rB_E1CS[3])
    rB_E1c = Transition(rB_E1c[1],rB_E1c[2],rB_E1c[3])
    rB_E1_NLO_LWA = Transition(rB_E1_NLO_LWA[1],rB_E1_NLO_LWA[2],rB_E1_NLO_LWA[3])

    rB = ReducedTransition(rB_E0,rB_E1,rB_E2,rB_E3,rB_E1V,rB_E1VC,rB_E1VS,rB_E1T,rB_E1TC,rB_E1TS,rB_E1C,rB_E1CC,rB_E1CS,rB_E1c,rB_E1_NLO_LWA)

    println("\tReduced transition intensities rB evaluated ...")

    return rB
end
