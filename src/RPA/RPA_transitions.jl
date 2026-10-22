function RPA_rM_linearized(Params::Parameters,Orb_1p1h::Orb2B_1p1h,M::Tr1B,Rho::O1B,X::Matrix{Matrix{ComplexF64}},Y::Matrix{Matrix{ComplexF64}})
    # Evaluate the reduced transition matrix elements M^lambda ...
    println("\nCalculating reduced transition matrix elements rM ...")

    # Define local function to evaluate transition matrix elements ...
    @inline function RPA_rM_calculate(a_p::Int64,a_h::Int64,T::Int64,J::Int64,O::O1B,Rho::O1B,X::ComplexF64,Y::ComplexF64)
        pMSum, nMSum = ComplexF64(0.0), ComplexF64(0.0)

        if T == -1
            pO, pRho = O.p, Rho.p
            D_ph = pRho[a_h,a_h] - pRho[a_p,a_p]

            pM = pO[a_p,a_h] * D_ph * (X * phase(J) + Y)

            pMSum += pM

            return pMSum, nMSum

        elseif T == 1
            nO, nRho = O.n, Rho.n
            D_ph = nRho[a_h,a_h] - nRho[a_p,a_p]

            nM = nO[a_p,a_h] * (D_ph * X * phase(J) + D_ph * Y)

            nMSum += nM

            return pMSum, nMSum
        end

        return pMSum, nMSum
    end

    # Case of E0 ...
    J, P = 0, 1

    N_ph = Orb_1p1h.N[P,J+1]
    Orb_ph_list = Orb_1p1h.Ind[P,J+1]

    X_JP, Y_JP = X[P,J+1], Y[P,J+1]

    # Initialize RPA matrix elements ...
    prM_E0 = Vector{ComplexF64}(undef,N_ph)
    nrM_E0 = Vector{ComplexF64}(undef,N_ph)

    # Evaluate RPA E0 matrix elements ...
    @inbounds Threads.@threads for i_nu in 1:N_ph
        pME_E0Sum = ComplexF64(0.0)
        nME_E0Sum = ComplexF64(0.0)

        @inbounds for (i_ph, Orb_ph) in enumerate(Orb_ph_list)
            a_p, a_h, T_ph = Orb_ph.a, Orb_ph.b, Orb_ph.T

            x, y = X_JP[i_ph,i_nu], Y_JP[i_ph,i_nu]

            pME_E0, nME_E0 = RPA_rM_calculate(a_p,a_h,T_ph,J,M.E0,Rho,x,y)

            pME_E0Sum += pME_E0
            nME_E0Sum += nME_E0
        end

        prM_E0[i_nu] = pME_E0Sum
        nrM_E0[i_nu] = nME_E0Sum
    end

    # Case of E1 ...
    J, P = 1, 2

    N_ph = Orb_1p1h.N[P,J+1]
    Orb_ph_list = Orb_1p1h.Ind[P,J+1]

    X_JP, Y_JP = X[P,J+1], Y[P,J+1]

    # Initialize RPA matrix elements ...
    prM_E1 = Vector{ComplexF64}(undef,N_ph)
    prM_E1VC = Vector{ComplexF64}(undef,N_ph)
    prM_E1VS = Vector{ComplexF64}(undef,N_ph)
    prM_E1TC = Vector{ComplexF64}(undef,N_ph)
    prM_E1TS = Vector{ComplexF64}(undef,N_ph)
    prM_E1sTC = Vector{ComplexF64}(undef,N_ph)
    prM_E1sTS = Vector{ComplexF64}(undef,N_ph)
    prM_E1C = Vector{ComplexF64}(undef,N_ph)
    prM_E1sC = Vector{ComplexF64}(undef,N_ph)

    nrM_E1 = Vector{ComplexF64}(undef,N_ph)
    nrM_E1VC = Vector{ComplexF64}(undef,N_ph)
    nrM_E1VS = Vector{ComplexF64}(undef,N_ph)
    nrM_E1TC = Vector{ComplexF64}(undef,N_ph)
    nrM_E1TS = Vector{ComplexF64}(undef,N_ph)
    nrM_E1sTC = Vector{ComplexF64}(undef,N_ph)
    nrM_E1sTS = Vector{ComplexF64}(undef,N_ph)
    nrM_E1C = Vector{ComplexF64}(undef,N_ph)
    nrM_E1sC = Vector{ComplexF64}(undef,N_ph)

    # Evaluate TDA & RPA E1 matrix elements ...
    @inbounds Threads.@threads for i_nu in 1:N_ph
        pME_E1Sum = ComplexF64(0.0)
        pME_E1VCSum = ComplexF64(0.0)
        pME_E1VSSum = ComplexF64(0.0)
        pME_E1TCSum = ComplexF64(0.0)
        pME_E1TSSum = ComplexF64(0.0)
        pME_E1sTCSum = ComplexF64(0.0)
        pME_E1sTSSum = ComplexF64(0.0)
        pME_E1CSum = ComplexF64(0.0)
        pME_E1sCSum = ComplexF64(0.0)

        nME_E1Sum = ComplexF64(0.0)
        nME_E1VCSum = ComplexF64(0.0)
        nME_E1VSSum = ComplexF64(0.0)
        nME_E1TCSum = ComplexF64(0.0)
        nME_E1TSSum = ComplexF64(0.0)
        nME_E1sTCSum = ComplexF64(0.0)
        nME_E1sTSSum = ComplexF64(0.0)
        nME_E1CSum = ComplexF64(0.0)
        nME_E1sCSum = ComplexF64(0.0)

        @inbounds for (i_ph, Orb_ph) in enumerate(Orb_ph_list)
            a_p, a_h, T_ph = Orb_ph.a, Orb_ph.b, Orb_ph.T

            x, y = X_JP[i_ph,i_nu], Y_JP[i_ph,i_nu]

            pME_E1, nME_E1 = RPA_rM_calculate(a_p,a_h,T_ph,J,M.E1,Rho,x,y)
            pME_E1, nME_E1 = RPA_rM_calculate(a_p,a_h,T_ph,J,M.E1,Rho,x,y)

            pME_E1VC, nME_E1VC = RPA_rM_calculate(a_p,a_h,T_ph,J,M.E1_VC,Rho,x,y)
            pME_E1VS, nME_E1VS = RPA_rM_calculate(a_p,a_h,T_ph,J,M.E1_VS,Rho,x,y)

            pME_E1TC, nME_E1TC = RPA_rM_calculate(a_p,a_h,T_ph,J,M.E1_TC,Rho,x,y)
            pME_E1TS, nME_E1TS = RPA_rM_calculate(a_p,a_h,T_ph,J,M.E1_TS,Rho,x,y)
            pME_E1sTC, nME_E1sTC = RPA_rM_calculate(a_p,a_h,T_ph,J,M.E1_sTC,Rho,x,y)
            pME_E1sTS, nME_E1sTS = RPA_rM_calculate(a_p,a_h,T_ph,J,M.E1_sTS,Rho,x,y)

            pME_E1C, nME_E1C = RPA_rM_calculate(a_p,a_h,T_ph,J,M.E1_C,Rho,x,y)
            pME_E1sC, nME_E1sC = RPA_rM_calculate(a_p,a_h,T_ph,J,M.E1_sC,Rho,x,y)

            pME_E1Sum += pME_E1
            nME_E1Sum += nME_E1

            pME_E1VCSum += pME_E1VC
            nME_E1VCSum += nME_E1VC

            pME_E1VSSum += pME_E1VS
            nME_E1VSSum += nME_E1VS

            pME_E1TCSum += pME_E1TC
            nME_E1TCSum += nME_E1TC

            pME_E1TSSum += pME_E1TS
            nME_E1TSSum += nME_E1TS

            pME_E1sTCSum += pME_E1sTC
            nME_E1sTCSum += nME_E1sTC

            pME_E1sTSSum += pME_E1sTS
            nME_E1sTSSum += nME_E1sTS

            pME_E1CSum += pME_E1C
            nME_E1CSum += nME_E1C

            pME_E1sCSum += pME_E1sC
            nME_E1sCSum += nME_E1sC
        end

        prM_E1[i_nu] = pME_E1Sum
        prM_E1VC[i_nu] = pME_E1VCSum
        prM_E1VS[i_nu] = pME_E1VSSum
        prM_E1TC[i_nu] = pME_E1TCSum
        prM_E1TS[i_nu] = pME_E1TSSum
        prM_E1sTC[i_nu] = pME_E1sTCSum
        prM_E1sTS[i_nu] = pME_E1sTSSum
        prM_E1C[i_nu] = pME_E1CSum
        prM_E1sC[i_nu] = pME_E1sCSum

        nrM_E1[i_nu] = nME_E1Sum
        nrM_E1VC[i_nu] = nME_E1VCSum
        nrM_E1VS[i_nu] = nME_E1VSSum
        nrM_E1TC[i_nu] = nME_E1TCSum
        nrM_E1TS[i_nu] = nME_E1TSSum
        nrM_E1sTC[i_nu] = nME_E1sTCSum
        nrM_E1sTS[i_nu] = nME_E1sTSSum
        nrM_E1C[i_nu] = nME_E1CSum
        nrM_E1sC[i_nu] = nME_E1sCSum
    end

    # Case of E2 ...
    J, P = 2, 1

    N_ph = Orb_1p1h.N[P,J+1]
    Orb_ph_list = Orb_1p1h.Ind[P,J+1]

    X_JP, Y_JP = X[P,J+1], Y[P,J+1]

    # Initialize RPA matrix elements ...
    prM_E2 = Vector{ComplexF64}(undef,N_ph)
    nrM_E2 = Vector{ComplexF64}(undef,N_ph)

    # Evaluate RPA E2 matrix elements ...
    @inbounds Threads.@threads for i_nu in 1:N_ph
        pME_E2Sum = ComplexF64(0.0)
        nME_E2Sum = ComplexF64(0.0)

        @inbounds for (i_ph, Orb_ph) in enumerate(Orb_ph_list)
            a_p, a_h, T_ph = Orb_ph.a, Orb_ph.b, Orb_ph.T

            x, y = X_JP[i_ph,i_nu], Y_JP[i_ph,i_nu]

            pME_E2, nME_E2 = RPA_rM_calculate(a_p,a_h,T_ph,J,M.E2,Rho,x,y)

            pME_E2Sum += pME_E2
            nME_E2Sum += nME_E2
        end

        prM_E2[i_nu] = pME_E2Sum
        nrM_E2[i_nu] = nME_E2Sum
    end

    # Case of E3 ...
    J, P = 3, 2

    N_ph = Orb_1p1h.N[P,J+1]
    Orb_ph_list = Orb_1p1h.Ind[P,J+1]

    X_JP, Y_JP = X[P,J+1], Y[P,J+1]

    # Initialize RPA matrix elements ...
    prM_E3 = Vector{ComplexF64}(undef,N_ph)
    nrM_E3 = Vector{ComplexF64}(undef,N_ph)

    # Evaluate TDA & RPA E3 matrix elements ...
    @inbounds Threads.@threads for i_nu in 1:N_ph
        pME_E3Sum = ComplexF64(0.0)
        nME_E3Sum = ComplexF64(0.0)

        @inbounds for (i_ph, Orb_ph) in enumerate(Orb_ph_list)
            a_p, a_h, T_ph = Orb_ph.a, Orb_ph.b, Orb_ph.T

            x, y = X_JP[i_ph,i_nu], Y_JP[i_ph,i_nu]

            pME_E3, nME_E3 = RPA_rM_calculate(a_p,a_h,T_ph,J,M.E3,Rho,x,y)

            pME_E3Sum += pME_E3
            nME_E3Sum += nME_E3
        end

        prM_E3[i_nu] = pME_E3Sum
        nrM_E3[i_nu] = nME_E3Sum
    end

    # Allocate the reduced multipole transition matrix elements ...
    rM_E0 = pnCVector(prM_E0,nrM_E0)
    rM_E1 = pnCVector(prM_E1,nrM_E1)
    rM_E2 = pnCVector(prM_E2,nrM_E2)
    rM_E3 = pnCVector(prM_E3,nrM_E3)
    rM_E1VC = pnCVector(prM_E1VC,nrM_E1VC)
    rM_E1VS = pnCVector(prM_E1VS,nrM_E1VS)
    rM_E1TC = pnCVector(prM_E1TC,nrM_E1TC)
    rM_E1TS = pnCVector(prM_E1TS,nrM_E1TS)
    rM_E1sTC = pnCVector(prM_E1sTC,nrM_E1sTC)
    rM_E1sTS = pnCVector(prM_E1sTS,nrM_E1sTS)
    rM_E1C = pnCVector(prM_E1C,nrM_E1C)
    rM_E1sC = pnCVector(prM_E1sC,nrM_E1sC)

    rM = ReducedMultipole(rM_E0,rM_E1,rM_E2,rM_E3,rM_E1VC,
                          rM_E1VS,rM_E1TC,rM_E1TS,rM_E1sTC,
                          rM_E1sTS,rM_E1C,rM_E1sC)

    println("\tReduced 1-body transition matrix elements rM calculated ...")

    return rM
end

function RPA_rM(Params::Parameters,Orb::Vector{Orb1B},Orb_1p1h::Orb2B_1p1h,Orb_NN::Orb2B,M::Tr1B,Rho::O1B,Sigma::O2B,X::Matrix{Matrix{ComplexF64}},Y::Matrix{Matrix{ComplexF64}})
    # Read the parameters ...
    TBCF = Params.Calc.RPA.TBCF
    
    # Evaluate the reduced transition matrix elements M^lambda ...
    println("\nCalculating reduced transition matrix elements rM ...")

    # Define local function to evaluate transition matrix elements ...
    @inline function RPA_rM_calculate(a_p::Int64,a_h::Int64,T::Int64,J::Int64,O::O1B,Rho::O1B,Sigma::O2B,X::ComplexF64,Y::ComplexF64,Orb::Vector{Orb1B},Orb_NN::Orb2B,TBCF::Bool)
        pMSum, nMSum = ComplexF64(0.0), ComplexF64(0.0)

        if T == -1
            pO, pRho = O.p, Rho.p
            pM = pO[a_p,a_h] * (pRho[a_h,a_h] * (1.0 - pRho[a_p,a_p]) * X * phase(J) +
                                pRho[a_p,a_p] * (1.0 - pRho[a_h,a_h]) * Y)
            pMSum += pM

            if TBCF == false
                return pMSum, nMSum
            end

            nO = O.n

            l_p, j_p = Orb[a_p].l, Orb[a_p].j
            l_h, j_h = Orb[a_h].l, Orb[a_h].j

            @inbounds for Orb_a in Orb
                a_a, l_a, j_a = Orb_a.a, Orb_a.l, Orb_a.j
                P_pa = rem(l_p + l_a,2) + 1
                P_ha = rem(l_h + l_a,2) + 1

                @inbounds for Orb_b in Orb
                    a_b, l_b, j_b = Orb_b.a, Orb_b.l, Orb_b.j

                    if (rem(l_h + l_b,2) + 1) == P_pa
                        I_min, I_max = div(max(abs(j_p - j_a),abs(j_h - j_b)),2), div(min(j_p + j_a,j_h + j_b),2)

                        @inbounds for I in I_min:I_max
                            I_hat = Float64(2*I + 1)
                            pM = (pO[a_a,a_b] * O2b_pp(a_p,a_a,a_h,a_b,I,P_pa,Sigma,Orb,Orb_NN) * phase(J) + 
                                  nO[a_a,a_b] * O2b_pn(a_p,a_a,a_h,a_b,I,P_pa,Sigma,Orb_NN)) * phase(I + 1) * I_hat * w6j(j_p,j_h,2*J,j_b,j_a,2*I) * X
                            pMSum += pM
                        end
                    end

                    if (rem(l_p + l_b,2) + 1) == P_ha
                        I_min, I_max = div(max(abs(j_h - j_a),abs(j_p - j_b)),2), div(min(j_h + j_a,j_p + j_b),2)

                        @inbounds for I in I_min:I_max
                            I_hat = Float64(2*I + 1)
                            pM = (pO[a_a,a_b] * O2b_pp(a_h,a_a,a_p,a_b,I,P_ha,Sigma,Orb,Orb_NN) + 
                                  nO[a_a,a_b] * O2b_pn(a_h,a_a,a_p,a_b,I,P_ha,Sigma,Orb_NN)) * phase(J + I + div(j_p + j_a,2)) * I_hat * w6j(j_p,j_h,2*J,j_a,j_b,2*I) * Y
                            pMSum += pM
                        end
                    end
                end
            end

            return pMSum, nMSum

        elseif T == 1
            nO, nRho = O.n, Rho.n
            nM = nO[a_p,a_h] * (nRho[a_h,a_h] * (1.0 - nRho[a_p,a_p]) * X * phase(J) +
                                nRho[a_p,a_p] * (1.0 - nRho[a_h,a_h]) * Y)
            nMSum += nM

            if TBCF == false
                return pMSum, nMSum
            end

            pO = O.p

            l_p, j_p = Orb[a_p].l, Orb[a_p].j
            l_h, j_h = Orb[a_h].l, Orb[a_h].j

            @inbounds for Orb_a in Orb
                a_a, l_a, j_a = Orb_a.a, Orb_a.l, Orb_a.j
                P_pa = rem(l_p + l_a,2) + 1
                P_ha = rem(l_h + l_a,2) + 1

                @inbounds for Orb_b in Orb
                    a_b, l_b, j_b = Orb_b.a, Orb_b.l, Orb_b.j

                    if (rem(l_h + l_b,2) + 1) == P_pa
                        I_min, I_max = div(max(abs(j_p - j_a),abs(j_h - j_b)),2), div(min(j_p + j_a,j_h + j_b),2)

                        @inbounds for I in I_min:I_max
                            I_hat = Float64(2*I + 1)
                            nM = (nO[a_a,a_b] * O2b_nn(a_p,a_a,a_h,a_b,I,P_pa,Sigma,Orb,Orb_NN) * phase(J) + 
                                  pO[a_a,a_b] * O2b_pn(a_a,a_p,a_b,a_h,I,P_pa,Sigma,Orb_NN)) * phase(I + 1) * I_hat * w6j(j_p,j_h,2*J,j_b,j_a,2*I) * X
                            nMSum += nM
                        end
                    end

                    if (rem(l_p + l_b,2) + 1) == P_ha
                        I_min, I_max = div(max(abs(j_h - j_a),abs(j_p - j_b)),2), div(min(j_h + j_a,j_p + j_b),2)

                        @inbounds for I in I_min:I_max
                            I_hat = Float64(2*I + 1)
                            nM = (nO[a_a,a_b] * O2b_nn(a_h,a_a,a_p,a_b,I,P_ha,Sigma,Orb,Orb_NN) + 
                                  pO[a_a,a_b] * O2b_pn(a_a,a_h,a_b,a_p,I,P_ha,Sigma,Orb_NN)) * phase(J + I + div(j_p + j_a,2)) * I_hat * w6j(j_p,j_h,2*J,j_a,j_b,2*I) * Y
                            nMSum += nM
                        end
                    end
                end
            end

            return pMSum, nMSum
        end

        return pMSum, nMSum
    end

    # Case of E0 ...
    J, P = 0, 1

    N_ph = Orb_1p1h.N[P,J+1]
    Orb_ph_list = Orb_1p1h.Ind[P,J+1]

    X_JP, Y_JP = X[P,J+1], Y[P,J+1]

    # Initialize RPA matrix elements ...
    prM_E0 = Vector{ComplexF64}(undef,N_ph)
    nrM_E0 = Vector{ComplexF64}(undef,N_ph)

    # Evaluate RPA E0 matrix elements ...
    @inbounds Threads.@threads for i_nu in 1:N_ph
        pME_E0Sum = ComplexF64(0.0)
        nME_E0Sum = ComplexF64(0.0)

        @inbounds for (i_ph, Orb_ph) in enumerate(Orb_ph_list)
            a_p, a_h, T_ph = Orb_ph.a, Orb_ph.b, Orb_ph.T

            x, y = X_JP[i_ph,i_nu], Y_JP[i_ph,i_nu]

            pME_E0, nME_E0 = RPA_rM_calculate(a_p,a_h,T_ph,J,M.E0,Rho,Sigma,x,y,Orb,Orb_NN,TBCF)

            pME_E0Sum += pME_E0
            nME_E0Sum += nME_E0
        end

        prM_E0[i_nu] = pME_E0Sum
        nrM_E0[i_nu] = nME_E0Sum
    end

    # Case of E1 ...
    J, P = 1, 2

    N_ph = Orb_1p1h.N[P,J+1]
    Orb_ph_list = Orb_1p1h.Ind[P,J+1]

    X_JP, Y_JP = X[P,J+1], Y[P,J+1]

    # Initialize RPA matrix elements ...
    prM_E1 = Vector{ComplexF64}(undef,N_ph)
    prM_E1VC = Vector{ComplexF64}(undef,N_ph)
    prM_E1VS = Vector{ComplexF64}(undef,N_ph)
    prM_E1TC = Vector{ComplexF64}(undef,N_ph)
    prM_E1TS = Vector{ComplexF64}(undef,N_ph)
    prM_E1sTC = Vector{ComplexF64}(undef,N_ph)
    prM_E1sTS = Vector{ComplexF64}(undef,N_ph)
    prM_E1C = Vector{ComplexF64}(undef,N_ph)
    prM_E1sC = Vector{ComplexF64}(undef,N_ph)

    nrM_E1 = Vector{ComplexF64}(undef,N_ph)
    nrM_E1VC = Vector{ComplexF64}(undef,N_ph)
    nrM_E1VS = Vector{ComplexF64}(undef,N_ph)
    nrM_E1TC = Vector{ComplexF64}(undef,N_ph)
    nrM_E1TS = Vector{ComplexF64}(undef,N_ph)
    nrM_E1sTC = Vector{ComplexF64}(undef,N_ph)
    nrM_E1sTS = Vector{ComplexF64}(undef,N_ph)
    nrM_E1C = Vector{ComplexF64}(undef,N_ph)
    nrM_E1sC = Vector{ComplexF64}(undef,N_ph)

    # Evaluate TDA & RPA E1 matrix elements ...
    @inbounds Threads.@threads for i_nu in 1:N_ph
        pME_E1Sum = ComplexF64(0.0)
        pME_E1VCSum = ComplexF64(0.0)
        pME_E1VSSum = ComplexF64(0.0)
        pME_E1TCSum = ComplexF64(0.0)
        pME_E1TSSum = ComplexF64(0.0)
        pME_E1sTCSum = ComplexF64(0.0)
        pME_E1sTSSum = ComplexF64(0.0)
        pME_E1CSum = ComplexF64(0.0)
        pME_E1sCSum = ComplexF64(0.0)

        nME_E1Sum = ComplexF64(0.0)
        nME_E1VCSum = ComplexF64(0.0)
        nME_E1VSSum = ComplexF64(0.0)
        nME_E1TCSum = ComplexF64(0.0)
        nME_E1TSSum = ComplexF64(0.0)
        nME_E1sTCSum = ComplexF64(0.0)
        nME_E1sTSSum = ComplexF64(0.0)
        nME_E1CSum = ComplexF64(0.0)
        nME_E1sCSum = ComplexF64(0.0)

        @inbounds for (i_ph, Orb_ph) in enumerate(Orb_ph_list)
            a_p, a_h, T_ph = Orb_ph.a, Orb_ph.b, Orb_ph.T

            x, y = X_JP[i_ph,i_nu], Y_JP[i_ph,i_nu]

            pME_E1, nME_E1 = RPA_rM_calculate(a_p,a_h,T_ph,J,M.E1,Rho,Sigma,x,y,Orb,Orb_NN,TBCF)

            pME_E1VC, nME_E1VC = RPA_rM_calculate(a_p,a_h,T_ph,J,M.E1_VC,Rho,Sigma,x,y,Orb,Orb_NN,TBCF)
            pME_E1VS, nME_E1VS = RPA_rM_calculate(a_p,a_h,T_ph,J,M.E1_VS,Rho,Sigma,x,y,Orb,Orb_NN,TBCF)

            pME_E1TC, nME_E1TC = RPA_rM_calculate(a_p,a_h,T_ph,J,M.E1_TC,Rho,Sigma,x,y,Orb,Orb_NN,TBCF)
            pME_E1TS, nME_E1TS = RPA_rM_calculate(a_p,a_h,T_ph,J,M.E1_TS,Rho,Sigma,x,y,Orb,Orb_NN,TBCF)
            pME_E1sTC, nME_E1sTC = RPA_rM_calculate(a_p,a_h,T_ph,J,M.E1_sTC,Rho,Sigma,x,y,Orb,Orb_NN,TBCF)
            pME_E1sTS, nME_E1sTS = RPA_rM_calculate(a_p,a_h,T_ph,J,M.E1_sTS,Rho,Sigma,x,y,Orb,Orb_NN,TBCF)

            pME_E1C, nME_E1C = RPA_rM_calculate(a_p,a_h,T_ph,J,M.E1_C,Rho,Sigma,x,y,Orb,Orb_NN,TBCF)
            pME_E1sC, nME_E1sC = RPA_rM_calculate(a_p,a_h,T_ph,J,M.E1_sC,Rho,Sigma,x,y,Orb,Orb_NN,TBCF)

            pME_E1Sum += pME_E1
            nME_E1Sum += nME_E1

            pME_E1VCSum += pME_E1VC
            nME_E1VCSum += nME_E1VC

            pME_E1VSSum += pME_E1VS
            nME_E1VSSum += nME_E1VS

            pME_E1TCSum += pME_E1TC
            nME_E1TCSum += nME_E1TC

            pME_E1TSSum += pME_E1TS
            nME_E1TSSum += nME_E1TS

            pME_E1sTCSum += pME_E1sTC
            nME_E1sTCSum += nME_E1sTC

            pME_E1sTSSum += pME_E1sTS
            nME_E1sTSSum += nME_E1sTS

            pME_E1CSum += pME_E1C
            nME_E1CSum += nME_E1C

            pME_E1sCSum += pME_E1sC
            nME_E1sCSum += nME_E1sC
        end

        prM_E1[i_nu] = pME_E1Sum
        prM_E1VC[i_nu] = pME_E1VCSum
        prM_E1VS[i_nu] = pME_E1VSSum
        prM_E1TC[i_nu] = pME_E1TCSum
        prM_E1TS[i_nu] = pME_E1TSSum
        prM_E1sTC[i_nu] = pME_E1sTCSum
        prM_E1sTS[i_nu] = pME_E1sTSSum
        prM_E1C[i_nu] = pME_E1CSum
        prM_E1sC[i_nu] = pME_E1sCSum

        nrM_E1[i_nu] = nME_E1Sum
        nrM_E1VC[i_nu] = nME_E1VCSum
        nrM_E1VS[i_nu] = nME_E1VSSum
        nrM_E1TC[i_nu] = nME_E1TCSum
        nrM_E1TS[i_nu] = nME_E1TSSum
        nrM_E1sTC[i_nu] = nME_E1sTCSum
        nrM_E1sTS[i_nu] = nME_E1sTSSum
        nrM_E1C[i_nu] = nME_E1CSum
        nrM_E1sC[i_nu] = nME_E1sCSum
    end

    # Case of E2 ...
    J, P = 2, 1

    N_ph = Orb_1p1h.N[P,J+1]
    Orb_ph_list = Orb_1p1h.Ind[P,J+1]

    X_JP, Y_JP = X[P,J+1], Y[P,J+1]

    # Initialize RPA matrix elements ...
    prM_E2 = Vector{ComplexF64}(undef,N_ph)
    nrM_E2 = Vector{ComplexF64}(undef,N_ph)

    # Evaluate RPA E2 matrix elements ...
    @inbounds Threads.@threads for i_nu in 1:N_ph
        pME_E2Sum = ComplexF64(0.0)
        nME_E2Sum = ComplexF64(0.0)

        @inbounds for (i_ph, Orb_ph) in enumerate(Orb_ph_list)
            a_p, a_h, T_ph = Orb_ph.a, Orb_ph.b, Orb_ph.T

            x, y = X_JP[i_ph,i_nu], Y_JP[i_ph,i_nu]

            pME_E2, nME_E2 = RPA_rM_calculate(a_p,a_h,T_ph,J,M.E2,Rho,Sigma,x,y,Orb,Orb_NN,TBCF)

            pME_E2Sum += pME_E2
            nME_E2Sum += nME_E2
        end

        prM_E2[i_nu] = pME_E2Sum
        nrM_E2[i_nu] = nME_E2Sum
    end

    # Case of E3 ...
    J, P = 3, 2

    N_ph = Orb_1p1h.N[P,J+1]
    Orb_ph_list = Orb_1p1h.Ind[P,J+1]

    X_JP, Y_JP = X[P,J+1], Y[P,J+1]

    # Initialize RPA matrix elements ...
    prM_E3 = Vector{ComplexF64}(undef,N_ph)
    nrM_E3 = Vector{ComplexF64}(undef,N_ph)

    # Evaluate TDA & RPA E3 matrix elements ...
    @inbounds Threads.@threads for i_nu in 1:N_ph
        pME_E3Sum = ComplexF64(0.0)
        nME_E3Sum = ComplexF64(0.0)

        @inbounds for (i_ph, Orb_ph) in enumerate(Orb_ph_list)
            a_p, a_h, T_ph = Orb_ph.a, Orb_ph.b, Orb_ph.T

            x, y = X_JP[i_ph,i_nu], Y_JP[i_ph,i_nu]

            pME_E3, nME_E3 = RPA_rM_calculate(a_p,a_h,T_ph,J,M.E3,Rho,Sigma,x,y,Orb,Orb_NN,TBCF)

            pME_E3Sum += pME_E3
            nME_E3Sum += nME_E3
        end

        prM_E3[i_nu] = pME_E3Sum
        nrM_E3[i_nu] = nME_E3Sum
    end

    # Allocate the reduced multipole transition matrix elements ...
    rM_E0 = pnCVector(prM_E0,nrM_E0)
    rM_E1 = pnCVector(prM_E1,nrM_E1)
    rM_E2 = pnCVector(prM_E2,nrM_E2)
    rM_E3 = pnCVector(prM_E3,nrM_E3)
    rM_E1VC = pnCVector(prM_E1VC,nrM_E1VC)
    rM_E1VS = pnCVector(prM_E1VS,nrM_E1VS)
    rM_E1TC = pnCVector(prM_E1TC,nrM_E1TC)
    rM_E1TS = pnCVector(prM_E1TS,nrM_E1TS)
    rM_E1sTC = pnCVector(prM_E1sTC,nrM_E1sTC)
    rM_E1sTS = pnCVector(prM_E1sTS,nrM_E1sTS)
    rM_E1C = pnCVector(prM_E1C,nrM_E1C)
    rM_E1sC = pnCVector(prM_E1sC,nrM_E1sC)

    rM = ReducedMultipole(rM_E0,rM_E1,rM_E2,rM_E3,rM_E1VC,
                          rM_E1VS,rM_E1TC,rM_E1TS,rM_E1sTC,
                          rM_E1sTS,rM_E1C,rM_E1sC)

    println("\tReduced 1-body transition matrix elements rM calculated ...")

    return rM
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
        E_nu = E[J+1,P][i_nu]
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

        rB_E1_NLO_LWA[2][i_nu] = 0.25 * abs(rM.E1.p[i_nu] + rM.E1.n[i_nu] +  E_nu / hc * ((rM.E1_TC.p[i_nu] + rM.E1_sTC.p[i_nu] + rM.E1_TC.n[i_nu] + rM.E1_sTC.n[i_nu]) + 0.25 * (g_p + g_n) * (rM.E1_TS.p[i_nu] + rM.E1_sTS.p[i_nu] + rM.E1_TS.n[i_nu] + rM.E1_sTS.n[i_nu])))^2

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

        rB_E1_NLO_LWA[3][i_nu] = abs(e_p * rM.E1.p[i_nu] - e_n * rM.E1.n[i_nu] +  0.5 * E_nu / hc * ((rM.E1_TC.p[i_nu] - rM.E1_TC.n[i_nu]) + 0.125 * (g_p - g_n) * (rM.E1_TS.p[i_nu] + rM.E1_TS.n[i_nu])))^2
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