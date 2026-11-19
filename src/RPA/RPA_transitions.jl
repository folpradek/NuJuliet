function RPA_rM(Params::Parameters,Orb::Vector{Orb1B},Orb_1p1h::Orb2B_1p1h,Orb_NN::Orb2B,M::Tr1B,Rho::O1B,Sigma::O2B,X::Matrix{Matrix{ComplexF64}},Y::Matrix{Matrix{ComplexF64}})
    # Read the parameters ...
    Transitions = Params.Calc.RPA.Transitions
    TBCF = Params.Calc.RPA.TBCF

    if (Transitions != "Exact") && (Transitions != "Linear")
        error("Invalid value of \"Params.Calc.RPA.Transitions\" = \"$Transitions\"; use \"Exact\" or \"Linear\".")
    end

    # Evaluate the reduced transition matrix elements M^lambda ...
    println("\nCalculating reduced transition matrix elements rM ... (" * Transitions * ")")

    # Case of E0 ...
    J, P = 0, 1
    M_E0 = [M.E0]
    prM_E0, nrM_E0 = RPA_rM_calculate(J,P,M_E0,Orb,Orb_1p1h,Orb_NN,Rho,Sigma,X,Y,Transitions,TBCF)

    # Case of E1 ...
    J, P = 1, 2
    M_E1 = [M.E1,M.E1_VC,M.E1_VS,M.E1_TC,M.E1_TS,M.E1_sTC,M.E1_sTS,M.E1_C,M.E1_sC]
    prM_E1, nrM_E1 = RPA_rM_calculate(J,P,M_E1,Orb,Orb_1p1h,Orb_NN,Rho,Sigma,X,Y,Transitions,TBCF)

    # Case of E2 ...
    J, P = 2, 1
    M_E2 = [M.E2]
    prM_E2, nrM_E2 = RPA_rM_calculate(J,P,M_E2,Orb,Orb_1p1h,Orb_NN,Rho,Sigma,X,Y,Transitions,TBCF)

    # Case of E3 ...
    J, P = 3, 2
    M_E3 = [M.E3]
    prM_E3, nrM_E3 = RPA_rM_calculate(J,P,M_E3,Orb,Orb_1p1h,Orb_NN,Rho,Sigma,X,Y,Transitions,TBCF)

    # Allocate the reduced multipole transition matrix elements ...
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

function RPA_rM_calculate(J::Int64,P::Int64,M_JP::Vector{O1B},Orb::Vector{Orb1B},Orb_1p1h::Orb2B_1p1h,Orb_NN::Orb2B,Rho::O1B,Sigma::O2B,X::Matrix{Matrix{ComplexF64}},Y::Matrix{Matrix{ComplexF64}},Transitions::String,TBCF::Bool)
    # Evaluate the reduced transition matrix elements of all operators M_JP of the multipole (J,P) ...
    N_ph = Orb_1p1h.N[P,J+1]

    # Evaluate the 1-body operator contractions ... M_X & M_Y ...
    pM_X, pM_Y, nM_X, nM_Y = RPA_rM_contract(J,P,M_JP,Orb,Orb_1p1h,Orb_NN,Rho,Sigma,Transitions,TBCF)

    # Contract the coefficients with the RPA amplitudes X & Y ...
        # rM[i_op,i_nu] = sum_ph (M_X[i_op,i_ph] * X[i_ph,i_nu] + M_Y[i_op,i_ph] * Y[i_ph,i_nu]) ... (BLAS matrix-matrix products)
    X_JP = @view X[P,J+1][1:N_ph,1:N_ph]
    Y_JP = @view Y[P,J+1][1:N_ph,1:N_ph]

    prM = ComplexF64.(pM_X) * X_JP
    mul!(prM,ComplexF64.(pM_Y),Y_JP,1.0,1.0)

    nrM = ComplexF64.(nM_X) * X_JP
    mul!(nrM,ComplexF64.(nM_Y),Y_JP,1.0,1.0)

    return prM, nrM
end

function RPA_rM_contract(J::Int64,P::Int64,M_JP::Vector{O1B},Orb::Vector{Orb1B},Orb_1p1h::Orb2B_1p1h,Orb_NN::Orb2B,Rho::O1B,Sigma::O2B,Transitions::String,TBCF::Bool)
    # Evaluate the coefficients M_X & M_Y of the RPA amplitudes X & Y for all operators M_JP of the multipole (J,P) ...
    #
    # Comment on the conventions ...
    #   M^t[a,b] = <a||M^t||b> with <a m_a|M_JM|b m_b> = (j_b m_b J M|j_a m_a) / hat(j_a) * M^t[a,b] (Tr1b) ...
    #   The X-term follows from <A_ph M> (operator in the order (p,h)), the Y-term from <A^dagger_ph M> (operator in the order (h,p)).
    #   For the standard (E-type) operators M[h,p] = (-1)^((j_p - j_h)/2) * M[p,h], so the Y-term coefficient reduces to M[p,h].
    #   The explicit form with M[h,p] is used here, so that operators with the opposite hermiticity phase (E1_VS, E1_TS, ...)
    #   are treated consistently with the 2-body (Sigma) part, which needs the full matrix M[a,b] anyway ...
    #
    #   Linear ...  rM = <0|[Q_nu,M]|0> with the occupations n of Orb ...
    #               M_X = (-1)^J * M[p,h] * (n_h - n_p) ,      M_Y = (-1)^((j_p - j_h)/2) * M[h,p] * (n_h - n_p) ,
    #   Exact  ...  rM = <0|Q_nu M|0> with the OBDM Rho & the 2-body cumulant Sigma ...
    #               M_X = (-1)^J * G[p,h] ,                    M_Y = -(-1)^((j_p - j_h)/2) * G[h,p] ,     G = (1 - Rho) * M * Rho ,
    #               (for diagonal Rho ... G[p,h] = rho_hh * (1 - rho_pp) * M[p,h] & G[h,p] = rho_pp * (1 - rho_hh) * M[h,p]) ,
    #               + 2-body part (Sigma = 2-body cumulant), both X & Y from the cross-coupled elements <ha;I|Sigma|pb;I> ...
    #               M_X^(2) = sum_ab M[a,b] sum_I (-1)^(J + I + j_h + j_a) (2I + 1) {j_p j_h J; j_a j_b I} <ha;I|Sigma|pb;I> ,
    #               M_Y^(2) = sum_ab M[b,a] sum_I (-1)^(I + j_h - j_b) (2I + 1) {j_p j_h J; j_a j_b I} <ha;I|Sigma|pb;I> ,
    #               (verified against the explicit M-scheme CG sums; at leading order the Sigma_hhpp block reproduces +Y.M of the
    #               linear branch, see Claude_HF_RPA/RPA_transitions_exact_check.md) ...
    #   Note on the sign of the exact Y-term ... in the linearized commutator the Y-term is dominated by -<0|M a^dagger_p a_h|0>
    #   (-> -(n_h - n_p) M[h,p]), while the exact Y-term is +<0|a^dagger_p a_h M|0> (-> +rho_pp * (1 - rho_hh) M[h,p]), hence the
    #   opposite sign of M_Y in the two branches (both relative to the same X-term) ... the exact Y-term vanishes for a HF reference ...
    #
    #   Benchmark (M-scheme reference <0|Q_nu M|0> with random scalar non-diagonal Rho, random J-coupled Sigma (pp, nn, pn), random X & Y,
    #   random non-hermitian & all Tr1b operators, Nmax = 2 & 3) ... agreement to 1e-14 for both branches, up to the global phase (-1)^J ...
    #   (NOTE: a random Sigma cannot detect a global sign of the 2-body part; that reference had the 2-body sign flipped, corrected 2026-11-19)
    #
    #   The pn part of Sigma couples the ph pair of isospin t to the operator of the OPPOSITE isospin -t; these contributions
    #   belong to rM^(-t) (not to rM^t), otherwise the physical (proton) & isovector intensities would be wrong ...
    Orb_ph_list = Orb_1p1h.Ind[P,J+1]
    N_ph = Orb_1p1h.N[P,J+1]
    N_o = length(M_JP)

    # Proton & neutron parts of the operators ...
    pM_JP = [M.p for M in M_JP]
    nM_JP = [M.n for M in M_JP]

    # 1-body transition densities G = (1 - Rho) * M * Rho for the exact transition matrix elements ... (Rho is a J = 0 scalar)
        # (single assignment, so that the threaded closure below captures them without boxing; not used in the linear branch)
    pG_JP = Transitions == "Exact" ? [(I - Rho.p) * M * Rho.p for M in pM_JP] : pM_JP
    nG_JP = Transitions == "Exact" ? [(I - Rho.n) * M * Rho.n for M in nM_JP] : nM_JP

    # Initialize the coefficients (proton & neutron parts) ... M_X[i_op,i_ph]
    pM_X, pM_Y = zeros(Float64,N_o,N_ph), zeros(Float64,N_o,N_ph)
    nM_X, nM_Y = zeros(Float64,N_o,N_ph), zeros(Float64,N_o,N_ph)

    # Evaluate the coefficients for each ph pair ... (independent of the RPA state nu)
    @inbounds Threads.@threads for i_ph in 1:N_ph
        Orb_ph = Orb_ph_list[i_ph]
        a_p, a_h, T_ph = Orb_ph.a, Orb_ph.b, Orb_ph.T

        # Operators of the same (1) & opposite (2) isospin as the ph pair ...
        if T_ph == -1
            M_1, M_2 = pM_JP, nM_JP
        elseif T_ph == 1
            M_1, M_2 = nM_JP, pM_JP
        else
            continue
        end

        l_p, j_p = Orb[a_p].l, Orb[a_p].j
        l_h, j_h = Orb[a_h].l, Orb[a_h].j

        # Phase of the Y-term ... M[h,p] -> M[p,h] for the E-type operators ...
        Phase_hp = phase(div(j_p - j_h,2))

        # Define local accumulators ... same (1) & opposite (2) isospin operators ...
        M_X_1, M_Y_1 = zeros(Float64,N_o), zeros(Float64,N_o)
        M_X_2, M_Y_2 = zeros(Float64,N_o), zeros(Float64,N_o)

        # Linearized transition matrix elements ...
        if Transitions == "Linear"
            if T_ph == -1
                D_ph = Orb[a_h].pO - Orb[a_p].pO
            else
                D_ph = Orb[a_h].nO - Orb[a_p].nO
            end

            @inbounds for i_o in 1:N_o
                M_X_1[i_o] = M_1[i_o][a_p,a_h] * D_ph * phase(J)
                M_Y_1[i_o] = M_1[i_o][a_h,a_p] * D_ph * Phase_hp
            end
        end

        # Exact transition matrix elements ...
        if Transitions == "Exact"
            # 1-body part ... (note the opposite sign of the Y-term w.r.t. the linear branch, see above)
            G_1 = T_ph == -1 ? pG_JP : nG_JP

            @inbounds for i_o in 1:N_o
                M_X_1[i_o] = G_1[i_o][a_p,a_h] * phase(J)
                M_Y_1[i_o] = -G_1[i_o][a_h,a_p] * Phase_hp
            end

            # 2-body part (2-body correlation function Sigma) ...
                # X-term ... +sum_ab M[a,b] <h a|Sigma|p b> (cross-coupled, <0|a+_h a_p a+_a a_b|0> = delta_pa rho_bh + <0|a+_h a+_a a_b a_p|0>), Y-term ... the same kernel with M[a,b] -> (-1)^((j_a - j_b)/2) M[b,a]
                # (Sigma hermitian) & the relative phase (-1)^(J+1) ...
            if TBCF == true
                Phase_X = phase(1 + div(j_p + j_h,2))
                Phase_Y = phase(J + div(j_p + j_h,2))

                @inbounds for Orb_a in Orb
                    a_a, l_a, j_a = Orb_a.a, Orb_a.l, Orb_a.j
                    P_ha = parity(l_h + l_a)

                    @inbounds for Orb_b in Orb
                        a_b, l_b, j_b = Orb_b.a, Orb_b.l, Orb_b.j

                        if triangle(j_a,j_b,2*J) == false || parity(l_p + l_b) != P_ha
                            continue
                        end

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

                        # Contract the kernels with all operators of the given multipole ...
                        Phase_ab = phase(div(j_a - j_b,2))

                        @inbounds for i_o in 1:N_o
                            M_X_1[i_o] += Phase_X * M_1[i_o][a_a,a_b] * K_1
                            M_X_2[i_o] += Phase_X * M_2[i_o][a_a,a_b] * K_2
                            M_Y_1[i_o] += Phase_Y * Phase_ab * M_1[i_o][a_b,a_a] * K_1
                            M_Y_2[i_o] += Phase_Y * Phase_ab * M_2[i_o][a_b,a_a] * K_2
                        end
                    end
                end
            end
        end

        # Store the coefficients ... the opposite isospin operator contributions to the opposite isospin part of rM ...
        if T_ph == -1
            pM_X[:,i_ph] .= M_X_1
            pM_Y[:,i_ph] .= M_Y_1
            nM_X[:,i_ph] .= M_X_2
            nM_Y[:,i_ph] .= M_Y_2
        else
            nM_X[:,i_ph] .= M_X_1
            nM_Y[:,i_ph] .= M_Y_1
            pM_X[:,i_ph] .= M_X_2
            pM_Y[:,i_ph] .= M_Y_2
        end
    end

    return pM_X, pM_Y, nM_X, nM_Y
end

function RPA_rB(Params::Parameters,Orb_1p1h::Orb2B_1p1h,rM::ReducedMultipole,E::Matrix{Vector{Float64}})
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
        rB_E1V[3][i_nu] = abs(0.5 * (rM.E1_VC.p[i_nu] - rM.E1_VC.n[i_nu]) + 0.125 * (g_p - g_n) * (rM.E1_VS.p[i_nu] + rM.E1_VS.n[i_nu]))^2
        rB_E1VC[3][i_nu] = abs(0.5 * (rM.E1_VC.p[i_nu] - rM.E1_VC.n[i_nu]))^2
        rB_E1VS[3][i_nu] = abs(0.125 * (g_p - g_n) * (rM.E1_VS.p[i_nu] + rM.E1_VS.n[i_nu]))^2
        
        rB_E1T[3][i_nu] = abs(0.5 * (rM.E1_TC.p[i_nu] - rM.E1_TC.n[i_nu]) + 0.125 * (g_p - g_n) * (rM.E1_TS.p[i_nu] + rM.E1_TS.n[i_nu]))^2
        rB_E1TC[3][i_nu] = abs(0.5 * (rM.E1_TC.p[i_nu] - rM.E1_TC.n[i_nu]))^2
        rB_E1TS[3][i_nu] = abs(0.125 * (g_p - g_n) * (rM.E1_TS.p[i_nu] + rM.E1_TS.n[i_nu]))^2

        rB_E1C[3][i_nu] = abs(0.5 * (rM.E1_TC.p[i_nu] - rM.E1_TC.n[i_nu]) + 0.125 * (g_p - g_n) * (rM.E1_TS.p[i_nu] + rM.E1_TS.n[i_nu]) - (0.5 * (rM.E1_VC.p[i_nu] - rM.E1_VC.n[i_nu]) + 0.125 * (g_p - g_n) * (rM.E1_VS.p[i_nu] + rM.E1_VS.n[i_nu])))^2
        rB_E1CC[3][i_nu] = abs(0.5 * (rM.E1_TC.p[i_nu] - rM.E1_TC.n[i_nu]) - (0.5 * (rM.E1_VC.p[i_nu] - rM.E1_VC.n[i_nu])))^2
        rB_E1CS[3][i_nu] = abs(0.125 * (g_p - g_n) * (rM.E1_TS.p[i_nu] + rM.E1_TS.n[i_nu]) - (0.125 * (g_p - g_n) * (rM.E1_VS.p[i_nu] + rM.E1_VS.n[i_nu])))^2

        rB_E1c[3][i_nu] = 0.25 * abs(rM.E1_C.p[i_nu] - rM.E1_C.n[i_nu])^2

        rB_E1_NLO_LWA[3][i_nu] = abs(e_p * rM.E1.p[i_nu] - e_n * rM.E1.n[i_nu] + 0.5 * E_nu / hc * ((rM.E1_TC.p[i_nu] - rM.E1_TC.n[i_nu]) + 0.125 * (g_p - g_n) * (rM.E1_TS.p[i_nu] + rM.E1_TS.n[i_nu])))^2
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