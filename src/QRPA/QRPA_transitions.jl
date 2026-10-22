function QRPA_rM(Params::Parameters,Orb_2qp::qpOrb2B,X_QRPA::Matrix{Matrix{ComplexF64}},Y_QRPA::Matrix{Matrix{ComplexF64}},qpTrOp::qpTr1B)
    # Evaluate the reduced transition metrix elements M^lambda ...
    println("\nCalculating the QRPA 1-phonon reduced transition matrix elements rM^lambda ...")

    # Case of E0 ...
    J, P = 0, 1
    N_qp = Orb_2qp.N[P,J+1]
    pM_E0 = Vector{ComplexF64}(undef,N_qp)
    nM_E0 = Vector{ComplexF64}(undef,N_qp)
    @inbounds Threads.@threads for nu in 1:N_qp
        pM_E0Sum, nM_E0Sum = ComplexF64(0.0), ComplexF64(0.0)
        @inbounds for i_qp in 1:N_qp
            a, b, T_ab = Orb_2qp.i[P,J+1][i_qp].a, Orb_2qp.i[P,J+1][i_qp].b, Orb_2qp.i[P,J+1][i_qp].T
            Amp = (X_QRPA[P,J+1][i_qp,nu] - Y_QRPA[P,J+1][i_qp,nu]) * ComplexF64(1.0 / sqrt(1.0 + kronecker_delta(a,b)))

            if T_ab == -1
                pE0ME = Amp * ComplexF64(qpTrOp.E0.qp20.p[a,b])
                pM_E0Sum += pE0ME
            end

            if T_ab == 1
                nE0ME = Amp * ComplexF64(qpTrOp.E0.qp20.n[a,b])
                nM_E0Sum += nE0ME
            end
        end
        pM_E0[nu] = pM_E0Sum
        nM_E0[nu] = nM_E0Sum
    end

    # Case of E1 ...
    J, P = 1, 2
    N_qp = Orb_2qp.N[P,J+1]
        # Initialize proton matrix elements ...
    pM_E1 = Vector{ComplexF64}(undef,N_qp)
    pM_E1_VC = Vector{ComplexF64}(undef,N_qp)
    pM_E1_VS = Vector{ComplexF64}(undef,N_qp)
    pM_E1_TC = Vector{ComplexF64}(undef,N_qp)
    pM_E1_TS = Vector{ComplexF64}(undef,N_qp)
    pM_E1_C = Vector{ComplexF64}(undef,N_qp)
    pM_E1_sTC = Vector{ComplexF64}(undef,N_qp)
    pM_E1_sTS = Vector{ComplexF64}(undef,N_qp)
    pM_E1_sC = Vector{ComplexF64}(undef,N_qp)
        # Initialize neutron matrix elements ...
    nM_E1 = Vector{ComplexF64}(undef,N_qp)
    nM_E1_VC = Vector{ComplexF64}(undef,N_qp)
    nM_E1_VS = Vector{ComplexF64}(undef,N_qp)
    nM_E1_TC = Vector{ComplexF64}(undef,N_qp)
    nM_E1_TS = Vector{ComplexF64}(undef,N_qp)
    nM_E1_C = Vector{ComplexF64}(undef,N_qp)
    nM_E1_sTC = Vector{ComplexF64}(undef,N_qp)
    nM_E1_sTS = Vector{ComplexF64}(undef,N_qp)
    nM_E1_sC = Vector{ComplexF64}(undef,N_qp)

    # Evaluate the E1 reduced transition matrix elements ...
    @inbounds Threads.@threads for nu in 1:N_qp
        pM_E1Sum, nM_E1Sum = ComplexF64(0.0), ComplexF64(0.0)
        pM_E1_VCSum, nM_E1_VCSum = ComplexF64(0.0), ComplexF64(0.0)
        pM_E1_VSSum, nM_E1_VSSum = ComplexF64(0.0), ComplexF64(0.0)
        pM_E1_TCSum, nM_E1_TCSum = ComplexF64(0.0), ComplexF64(0.0)
        pM_E1_TSSum, nM_E1_TSSum = ComplexF64(0.0), ComplexF64(0.0)
        pM_E1_CSum, nM_E1_CSum = ComplexF64(0.0), ComplexF64(0.0)
        pM_E1_sTCSum, nM_E1_sTCSum = ComplexF64(0.0), ComplexF64(0.0)
        pM_E1_sTSSum, nM_E1_sTSSum = ComplexF64(0.0), ComplexF64(0.0)
        pM_E1_sCSum, nM_E1_sCSum = ComplexF64(0.0), ComplexF64(0.0)

        @inbounds for i_qp in 1:N_qp
            a, b, T_ab = Orb_2qp.i[P,J+1][i_qp].a, Orb_2qp.i[P,J+1][i_qp].b, Orb_2qp.i[P,J+1][i_qp].T
            Amp = (X_QRPA[P,J+1][i_qp,nu] - Y_QRPA[P,J+1][i_qp,nu]) * ComplexF64(1.0 / sqrt(1.0 + kronecker_delta(a,b)))

            if T_ab == -1
                pE1ME = Amp * ComplexF64(qpTrOp.E1.qp20.p[a,b])
                pE1_VCME = Amp * ComplexF64(qpTrOp.E1_VC.qp20.p[a,b])
                pE1_VSME = Amp * ComplexF64(qpTrOp.E1_VS.qp20.p[a,b])
                pE1_TCME = Amp * ComplexF64(qpTrOp.E1_TC.qp20.p[a,b])
                pE1_TSME = Amp * ComplexF64(qpTrOp.E1_TS.qp20.p[a,b])
                pE1_CME = Amp * ComplexF64(qpTrOp.E1_C.qp20.p[a,b])
                pE1_sTCME = Amp * ComplexF64(qpTrOp.E1_sTC.qp20.p[a,b])
                pE1_sTSME = Amp * ComplexF64(qpTrOp.E1_sTS.qp20.p[a,b])
                pE1_sCME = Amp * ComplexF64(qpTrOp.E1_sC.qp20.p[a,b])

                pM_E1Sum += pE1ME
                pM_E1_VCSum += pE1_VCME
                pM_E1_VSSum += pE1_VSME
                pM_E1_TCSum += pE1_TCME
                pM_E1_TSSum += pE1_TSME
                pM_E1_CSum += pE1_CME
                pM_E1_sTCSum += pE1_sTCME
                pM_E1_sTSSum += pE1_sTSME
                pM_E1_sCSum += pE1_sCME
            end

            if T_ab == 1
                nE1ME = Amp * ComplexF64(qpTrOp.E1.qp20.n[a,b])
                nE1_VCME = Amp * ComplexF64(qpTrOp.E1_VC.qp20.n[a,b])
                nE1_VSME = Amp * ComplexF64(qpTrOp.E1_VS.qp20.n[a,b])
                nE1_TCME = Amp * ComplexF64(qpTrOp.E1_TC.qp20.n[a,b])
                nE1_TSME = Amp * ComplexF64(qpTrOp.E1_TS.qp20.n[a,b])
                nE1_CME = Amp * ComplexF64(qpTrOp.E1_C.qp20.n[a,b])
                nE1_sTCME = Amp * ComplexF64(qpTrOp.E1_sTC.qp20.n[a,b])
                nE1_sTSME = Amp * ComplexF64(qpTrOp.E1_sTS.qp20.n[a,b])
                nE1_sCME = Amp * ComplexF64(qpTrOp.E1_sC.qp20.n[a,b])

                nM_E1Sum += nE1ME
                nM_E1_VCSum += nE1_VCME
                nM_E1_VSSum += nE1_VSME
                nM_E1_TCSum += nE1_TCME
                nM_E1_TSSum += nE1_TSME
                nM_E1_CSum += nE1_CME
                nM_E1_sTCSum += nE1_sTCME
                nM_E1_sTSSum += nE1_sTSME
                nM_E1_sCSum += nE1_sCME
            end
        end
        pM_E1[nu], nM_E1[nu] = pM_E1Sum, nM_E1Sum
        pM_E1_VC[nu], nM_E1_VC[nu] = pM_E1_VCSum, nM_E1_VCSum
        pM_E1_VS[nu], nM_E1_VS[nu] = pM_E1_VSSum, nM_E1_VSSum
        pM_E1_TC[nu], nM_E1_TC[nu] = pM_E1_TCSum, nM_E1_TCSum
        pM_E1_TS[nu], nM_E1_TS[nu] = pM_E1_TSSum, nM_E1_TSSum
        pM_E1_C[nu], nM_E1_C[nu] = pM_E1_CSum, nM_E1_CSum
        pM_E1_sTC[nu], nM_E1_sTC[nu] = pM_E1_sTCSum, nM_E1_sTCSum
        pM_E1_sTS[nu], nM_E1_sTS[nu] = pM_E1_sTSSum, nM_E1_sTSSum
        pM_E1_sC[nu], nM_E1_sC[nu] = pM_E1_sCSum, nM_E1_sCSum
    end

    # Case of E2 ...
    J, P = 2, 1
    N_qp = Orb_2qp.N[P,J+1]
    pM_E2 = Vector{ComplexF64}(undef,N_qp)
    nM_E2 = Vector{ComplexF64}(undef,N_qp)

    # Evaluate the E2 reduced transition matrix elements ...
    @inbounds Threads.@threads for nu in 1:N_qp
        pM_E2Sum, nM_E2Sum = ComplexF64(0.0), ComplexF64(0.0)
        @inbounds for i_qp in 1:N_qp
            a, b, T_ab = Orb_2qp.i[P,J+1][i_qp].a, Orb_2qp.i[P,J+1][i_qp].b, Orb_2qp.i[P,J+1][i_qp].T
            Amp = (X_QRPA[P,J+1][i_qp,nu] - Y_QRPA[P,J+1][i_qp,nu]) * ComplexF64(1.0 / sqrt(1.0 + kronecker_delta(a,b)))
            if T_ab == -1
                pE2ME = Amp * ComplexF64(qpTrOp.E2.qp20.p[a,b])
                pM_E2Sum += pE2ME
            end

            if T_ab == 1
                nE2ME = Amp * ComplexF64(qpTrOp.E2.qp20.n[a,b])
                nM_E2Sum += nE2ME
            end
        end
        pM_E2[nu] = pM_E2Sum
        nM_E2[nu] = nM_E2Sum
    end

    # Case of E3 ...
    J, P = 3, 2
    N_qp = Orb_2qp.N[P,J+1]
    pM_E3 = Vector{ComplexF64}(undef,N_qp)
    nM_E3 = Vector{ComplexF64}(undef,N_qp)

    # Evaluate the E3 reduced transition matrix elements ...
    @inbounds Threads.@threads for nu in 1:N_qp
        pM_E3Sum, nM_E3Sum = ComplexF64(0.0), ComplexF64(0.0)
        @inbounds for i_qp in 1:N_qp
            a, b, T_ab = Orb_2qp.i[P,J+1][i_qp].a, Orb_2qp.i[P,J+1][i_qp].b, Orb_2qp.i[P,J+1][i_qp].T
            Amp = (X_QRPA[P,J+1][i_qp,nu] - Y_QRPA[P,J+1][i_qp,nu]) * ComplexF64(1.0 / sqrt(1.0 + kronecker_delta(a,b)))
            if T_ab == -1
                pE3ME = Amp * ComplexF64(qpTrOp.E3.qp20.p[a,b])
                pM_E3Sum += pE3ME
            end

            if T_ab == 1
                nE3ME = Amp * ComplexF64(qpTrOp.E3.qp20.n[a,b])
                nM_E3Sum += nE3ME
            end
        end
        pM_E3[nu] = pM_E3Sum
        nM_E3[nu] = nM_E3Sum
    end

    # Allocate the reduced multipole transition matrix elements ...
        # Standard EX multipoles ...
    rM_E0 = pnCVector(pM_E0,nM_E0)
    rM_E1 = pnCVector(pM_E1,nM_E1)
    rM_E2 = pnCVector(pM_E2,nM_E2)
    rM_E3 = pnCVector(pM_E3,nM_E3)
        # Toroidal E1 multipoles ...
    rM_E1_VC = pnCVector(pM_E1_VC,nM_E1_VC)
    rM_E1_VS = pnCVector(pM_E1_VS,nM_E1_VS)
    rM_E1_TC = pnCVector(pM_E1_TC,nM_E1_TC)
    rM_E1_TS = pnCVector(pM_E1_TS,nM_E1_TS)
    rM_E1_C = pnCVector(pM_E1_C,nM_E1_C)
    rM_E1_sTC = pnCVector(pM_E1_sTC,nM_E1_sTC)
    rM_E1_sTS = pnCVector(pM_E1_sTS,nM_E1_sTS)
    rM_E1_sC = pnCVector(pM_E1_sC,nM_E1_sC)

    rM_QRPA = ReducedMultipole(rM_E0,rM_E1,rM_E2,rM_E3,rM_E1_VC,rM_E1_VS,rM_E1_TC,
                               rM_E1_TS,rM_E1_sTC,rM_E1_sTS,rM_E1_C,rM_E1_sC)

    println("\tQRPA 1-phonon reduced transition matrix elements rM^lambda succesfully calculated ...")

    return rM_QRPA
end

function QRPA_rB(Params::Parameters,Orb_2qp::qpOrb2B,rM::ReducedMultipole,E_QRPA::Matrix{Vector{ComplexF64}})
    # Read parameters ...
    A = Params.Calc.A
    Z = Params.Calc.Z

    # Define basic constants ...
    hc = 197.326980
    g_p = 5.586
    g_n = -3.826

    # Evaluate the reduced transition intensities B ...
    println("\nCalculating the QRPA 1-phonon reduced transition intensities rB^lambda ...")

    # Case of E0 ...
    J, P = 0, 1
    N_qp = Orb_2qp.N[P,J+1]

    # Initialize the components of B_E0 ...
    rB_E0 = Vector{Vector{Float64}}(undef,3)
    @inbounds for i in 1:3
        rB_E0[i] = Vector{Float64}(undef,N_qp)
    end

    # Evaluate the components of B_E0 ...
    @inbounds Threads.@threads for nu in 1:N_qp
        rB_E0[1][nu] = abs(rM.E0.p[nu])^2
        rB_E0[2][nu] = 0.25 * abs(rM.E0.p[nu] + rM.E0.n[nu])^2
        rB_E0[3][nu] = 0.25 * abs(rM.E0.p[nu] - rM.E0.n[nu])^2
    end

    # Case of E1 ...
    J, P = 1, 2
    N_qp = Orb_2qp.N[P,J+1]

    # Initialize the components of B_E1 ...
    rB_E1 = Vector{Vector{Float64}}(undef,3)
    rB_E1_V = Vector{Vector{Float64}}(undef,3)
    rB_E1_VC = Vector{Vector{Float64}}(undef,3)
    rB_E1_VS = Vector{Vector{Float64}}(undef,3)
    rB_E1_T = Vector{Vector{Float64}}(undef,3)
    rB_E1_TC = Vector{Vector{Float64}}(undef,3)
    rB_E1_TS = Vector{Vector{Float64}}(undef,3)
    rB_E1_C = Vector{Vector{Float64}}(undef,3)
    rB_E1_CC = Vector{Vector{Float64}}(undef,3)
    rB_E1_CS = Vector{Vector{Float64}}(undef,3)
    rB_E1_c = Vector{Vector{Float64}}(undef,3)
    rB_E1_NLO_LWA = Vector{Vector{Float64}}(undef,3)

    @inbounds for i in 1:3
        rB_E1[i] = Vector{Float64}(undef,N_qp)
        rB_E1_V[i] = Vector{Float64}(undef,N_qp)
        rB_E1_VC[i] = Vector{Float64}(undef,N_qp)
        rB_E1_VS[i] = Vector{Float64}(undef,N_qp)
        rB_E1_T[i] = Vector{Float64}(undef,N_qp)
        rB_E1_TC[i] = Vector{Float64}(undef,N_qp)
        rB_E1_TS[i] = Vector{Float64}(undef,N_qp)
        rB_E1_C[i] = Vector{Float64}(undef,N_qp)
        rB_E1_CC[i] = Vector{Float64}(undef,N_qp)
        rB_E1_CS[i] = Vector{Float64}(undef,N_qp)
        rB_E1_c[i] = Vector{Float64}(undef,N_qp)
        rB_E1_NLO_LWA[i] = Vector{Float64}(undef,N_qp)
    end

    # Define effective E1 isovector charges ...
    e_p = Float64(A - Z) / Float64(A)
    e_n = Float64(Z) / Float64(A)

    # Evaluate the components of B_E1 ...
    @inbounds Threads.@threads for nu in 1:N_qp
        E = real(E_QRPA[P,J+1][nu])
            # Physical components ...
        rB_E1[1][nu] = abs(rM.E1.p[nu])^2
        rB_E1_V[1][nu] = abs(rM.E1_VC.p[nu] + 0.5 * g_p * rM.E1_VS.p[nu] + 0.5 * g_n * rM.E1_VS.n[nu])^2
        rB_E1_VC[1][nu] = abs(rM.E1_VC.p[nu])^2
        rB_E1_VS[1][nu] = abs(0.5 * g_p * rM.E1_VS.p[nu] + 0.5 * g_n * rM.E1_VS.n[nu])^2

        rB_E1_T[1][nu] = abs(rM.E1_TC.p[nu] + 0.5 * g_p * rM.E1_TS.p[nu] + 0.5 * g_n * rM.E1_TS.n[nu])^2
        rB_E1_TC[1][nu] = abs(rM.E1_TC.p[nu])^2
        rB_E1_TS[1][nu] = abs(0.5 * g_p * rM.E1_TS.p[nu] + 0.5 * g_n * rM.E1_TS.n[nu])^2

        rB_E1_C[1][nu] = abs(rM.E1_TC.p[nu] + 0.5 * g_p * rM.E1_TS.p[nu] + 0.5 * g_n * rM.E1_TS.n[nu] - (rM.E1_VC.p[nu] + 0.5 * g_p * rM.E1_VS.p[nu] + 0.5 * g_n * rM.E1_VS.n[nu]))^2
        rB_E1_CC[1][nu] = abs(rM.E1_TC.p[nu] - (rM.E1_VC.p[nu]))^2
        rB_E1_CS[1][nu] = abs(0.5 * g_p * rM.E1_TS.p[nu] + 0.5 * g_n * rM.E1_TS.n[nu] - (0.5 * g_p * rM.E1_VS.p[nu] + 0.5 * g_n * rM.E1_VS.n[nu]))^2

        rB_E1_c[1][nu] = abs(rM.E1_C.p[nu])^2

        rB_E1_NLO_LWA[1][nu] = abs(rM.E1.p[nu] + E / hc * (rM.E1_TC.p[nu] + 0.5 * g_p * rM.E1_TS.p[nu] + 0.5 * g_n * rM.E1_TS.n[nu]))^2

            # Isoscalar components ...
        rB_E1[2][nu] = 0.25 * abs(rM.E1.p[nu] + rM.E1.n[nu])^2
        rB_E1_V[2][nu] = abs(0.5 * (rM.E1_VC.p[nu] + rM.E1_VC.n[nu]) + 0.125 * (g_p + g_n) * (rM.E1_VS.p[nu] + rM.E1_VS.n[nu]))^2
        rB_E1_VC[2][nu] = abs(0.5 * (rM.E1_VC.p[nu] + rM.E1_VC.n[nu]))^2
        rB_E1_VS[2][nu] = abs(0.125 * (g_p + g_n) * (rM.E1_VS.p[nu] + rM.E1_VS.n[nu]))^2
        
        rB_E1_T[2][nu] = abs(0.5 * (rM.E1_TC.p[nu] + rM.E1_sTC.p[nu] + rM.E1_TC.n[nu] + rM.E1_sTC.n[nu]) + 0.125 * (g_p + g_n) * (rM.E1_TS.p[nu] + rM.E1_sTS.p[nu] + rM.E1_TS.n[nu] + rM.E1_sTS.n[nu]))^2
        rB_E1_TC[2][nu] = abs(0.5 * (rM.E1_TC.p[nu] + rM.E1_sTC.p[nu] + rM.E1_TC.n[nu] + rM.E1_sTC.n[nu]))^2
        rB_E1_TS[2][nu] = abs(0.125 * (g_p + g_n) * (rM.E1_TS.p[nu] + rM.E1_sTS.p[nu] + rM.E1_TS.n[nu] + rM.E1_sTS.n[nu]))^2

        rB_E1_C[2][nu] = abs(0.5 * (rM.E1_TC.p[nu] + rM.E1_sTC.p[nu] + rM.E1_TC.n[nu] + rM.E1_sTC.n[nu]) + 0.125 * (g_p + g_n) * (rM.E1_TS.p[nu] + rM.E1_sTS.p[nu] + rM.E1_TS.n[nu] + rM.E1_sTS.n[nu]) - (0.5 * (rM.E1_VC.p[nu] + rM.E1_VC.n[nu]) + 0.125 * (g_p + g_n) * (rM.E1_VS.p[nu] + rM.E1_VS.n[nu])))^2
        rB_E1_CC[2][nu] = abs(0.5 * (rM.E1_TC.p[nu] + rM.E1_sTC.p[nu] + rM.E1_TC.n[nu] + rM.E1_sTC.n[nu]) - (0.5 * (rM.E1_VC.p[nu] + rM.E1_VC.n[nu])))^2
        rB_E1_CS[2][nu] = abs(0.125 * (g_p + g_n) * (rM.E1_TS.p[nu] + rM.E1_sTS.p[nu] + rM.E1_TS.n[nu] + rM.E1_sTS.n[nu]) - (0.125 * (g_p + g_n) * (rM.E1_VS.p[nu] + rM.E1_VS.n[nu])))^2

        rB_E1_c[2][nu] = 0.25 * abs(rM.E1_C.p[nu] + rM.E1_sC.p[nu] + rM.E1_C.n[nu] + rM.E1_sC.n[nu])^2

        rB_E1_NLO_LWA[2][nu] = 0.25 * abs(rM.E1.p[nu] + rM.E1.n[nu] +  E / hc * ((rM.E1_TC.p[nu] + rM.E1_sTC.p[nu] + rM.E1_TC.n[nu] + rM.E1_sTC.n[nu]) + 0.25 * (g_p + g_n) * (rM.E1_TS.p[nu] + rM.E1_sTS.p[nu] + rM.E1_TS.n[nu] + rM.E1_sTS.n[nu])))^2

            # Isovector components ...
        rB_E1[3][nu] = abs(e_p * rM.E1.p[nu] - e_n * rM.E1.n[nu])^2
        rB_E1_V[3][nu] = abs(0.5 * (rM.E1_VC.p[nu] - rM.E1_VC.n[nu]) + 0.125 * (g_p - g_n) * (rM.E1_VS.p[nu] + rM.E1_VS.n[nu]))^2
        rB_E1_VC[3][nu] = abs(0.5 * (rM.E1_VC.p[nu] - rM.E1_VC.n[nu]))^2
        rB_E1_VS[3][nu] = abs(0.125 * (g_p - g_n) * (rM.E1_VS.p[nu] + rM.E1_VS.n[nu]))^2
        
        rB_E1_T[3][nu] = abs(0.5 * (rM.E1_TC.p[nu] - rM.E1_TC.n[nu]) + 0.125 * (g_p - g_n) * (rM.E1_TS.p[nu] + rM.E1_TS.n[nu]))^2
        rB_E1_TC[3][nu] = abs(0.5 * (rM.E1_TC.p[nu] - rM.E1_TC.n[nu]))^2
        rB_E1_TS[3][nu] = abs(0.125 * (g_p - g_n) * (rM.E1_TS.p[nu] + rM.E1_TS.n[nu]))^2

        rB_E1_C[3][nu] = abs(0.5 * (rM.E1_TC.p[nu] - rM.E1_TC.n[nu]) + 0.125 * (g_p - g_n) * (rM.E1_TS.p[nu] + rM.E1_TS.n[nu]) - (0.5 * (rM.E1_VC.p[nu] - rM.E1_VC.n[nu]) + 0.125 * (g_p - g_n) * (rM.E1_VS.p[nu] + rM.E1_VS.n[nu])))^2
        rB_E1_CC[3][nu] = abs(0.5 * (rM.E1_TC.p[nu] - rM.E1_TC.n[nu]) - (0.5 * (rM.E1_VC.p[nu] - rM.E1_VC.n[nu])))^2
        rB_E1_CS[3][nu] = abs(0.125 * (g_p - g_n) * (rM.E1_TS.p[nu] + rM.E1_TS.n[nu]) - (0.125 * (g_p - g_n) * (rM.E1_VS.p[nu] + rM.E1_VS.n[nu])))^2

        rB_E1_c[3][nu] = 0.25 * abs(rM.E1_C.p[nu] - rM.E1_C.n[nu])^2

        rB_E1_NLO_LWA[3][nu] = abs(e_p * rM.E1.p[nu] - e_n * rM.E1.n[nu] +  0.5 * E / hc * ((rM.E1_TC.p[nu] - rM.E1_TC.n[nu]) + 0.125 * (g_p - g_n) * (rM.E1_TS.p[nu] + rM.E1_TS.n[nu])))^2
    end

    # Case of E2 ...
    J, P = 2, 1
    N_qp = Orb_2qp.N[P,J+1]

    # Initialize the components of B_E2 ...
    rB_E2 = Vector{Vector{Float64}}(undef,3)
    @inbounds for i in 1:3
        rB_E2[i] = Vector{Float64}(undef,N_qp)
    end

    # Evaluate the components of B_E2 ...
    @inbounds Threads.@threads for nu in 1:N_qp
        rB_E2[1][nu] = abs(rM.E2.p[nu])^2
        rB_E2[2][nu] = 0.25 * abs(rM.E2.p[nu] + rM.E2.n[nu])^2
        rB_E2[3][nu] = 0.25 * abs(rM.E2.p[nu] - rM.E2.n[nu])^2
    end

    # Case of E3 ...
    J, P = 3, 2
    N_qp = Orb_2qp.N[P,J+1]

    # Initialize the components of B_E3 ...
    rB_E3 = Vector{Vector{Float64}}(undef,3)
    @inbounds for i in 1:3
        rB_E3[i] = Vector{Float64}(undef,N_qp)
    end

    # Evaluate the components of B_E3 ...
    @inbounds Threads.@threads for nu in 1:N_qp
        rB_E3[1][nu] = abs(rM.E3.p[nu])^2
        rB_E3[2][nu] = 0.25 * abs(rM.E3.p[nu] + rM.E3.n[nu])^2
        rB_E3[3][nu] = 0.25 * abs(rM.E3.p[nu] - rM.E3.n[nu])^2
    end

    # Allocate the reduced transition intensity matrix elements for the EX transitions ...
    rB_E0 = Transition(rB_E0[1],rB_E0[2],rB_E0[3])
    rB_E1 = Transition(rB_E1[1],rB_E1[2],rB_E1[3])
    rB_E2 = Transition(rB_E2[1],rB_E2[2],rB_E2[3])
    rB_E3 = Transition(rB_E3[1],rB_E3[2],rB_E3[3])
    rB_E1V = Transition(rB_E1_V[1],rB_E1_V[2],rB_E1_V[3])
    rB_E1VC = Transition(rB_E1_VC[1],rB_E1_VC[2],rB_E1_VC[3])
    rB_E1VS = Transition(rB_E1_VS[1],rB_E1_VS[2],rB_E1_VS[3])
    rB_E1T = Transition(rB_E1_T[1],rB_E1_T[2],rB_E1_T[3])
    rB_E1TC = Transition(rB_E1_TC[1],rB_E1_TC[2],rB_E1_TC[3])
    rB_E1TS = Transition(rB_E1_TS[1],rB_E1_TS[2],rB_E1_TS[3])
    rB_E1C = Transition(rB_E1_C[1],rB_E1_C[2],rB_E1_C[3])
    rB_E1CC = Transition(rB_E1_CC[1],rB_E1_CC[2],rB_E1_CC[3])
    rB_E1CS = Transition(rB_E1_CS[1],rB_E1_CS[2],rB_E1_CS[3])
    rB_E1c = Transition(rB_E1_c[1],rB_E1_c[2],rB_E1_c[3])
    rB_E1_NLO_LWA = Transition(rB_E1_NLO_LWA[1],rB_E1_NLO_LWA[2],rB_E1_NLO_LWA[3])

    rB = ReducedTransition(rB_E0,rB_E1,rB_E2,rB_E3,rB_E1V,rB_E1VC,rB_E1VS,rB_E1T,rB_E1TC,rB_E1TS,rB_E1C,rB_E1CC,rB_E1CS,rB_E1c,rB_E1_NLO_LWA)

    println("\tQRPA reduced transition intensities rB evaluated ...")

    return rB
end