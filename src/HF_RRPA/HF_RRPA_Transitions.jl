function HF_RRPA_rM(Params::Parameters,N_nu::Matrix{Int64},Orb_Phonon::Matrix{Vector{Int64}},Phonon::Vector{PhState},Particle::pnSVector,Hole::pnSVector,Rho::O1B,TrOp::Tr1B,X_RPA::Matrix{Matrix{ComplexF64}},Y_RPA::Matrix{Matrix{ComplexF64}})
    
    println("\nCalculating reduced transition matrix elements rM ...")

    # Case of E0 ...
    J, P = 0, 1
    N_ph = N_nu[J+1,P]

    # Initialize RRPA matrix elements ...
    prM_E0 = Vector{ComplexF64}(undef,N_ph)
    nrM_E0 = Vector{ComplexF64}(undef,N_ph)

    # Evaluate RRPA E0 matrix elements ...
    @inbounds Threads.@threads for nu in 1:N_ph
        pM_E0Sum = ComplexF64(0.0)
        nM_E0Sum = ComplexF64(0.0)

        @inbounds for ph in 1:N_ph
            i_ph = Orb_Phonon[J+1,P][ph]
            p, h, t_ph = Phonon[i_ph].p, Phonon[i_ph].h, Phonon[i_ph].tz

            if t_ph == -1
                a_p, a_h = Particle.p[p].a, Hole.p[h].a
                rM_E0 = ComplexF64(TrOp.E0.p[a_p,a_h] * sqrt(Rho.p[a_h,a_h] - Rho.p[a_p,a_p])) * (X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])
                pM_E0Sum += rM_E0

            elseif t_ph == 1
                a_p, a_h = Particle.n[p].a, Hole.n[h].a
                rM_E0 = ComplexF64(TrOp.E0.n[a_p,a_h] * sqrt(Rho.n[a_h,a_h] - Rho.n[a_p,a_p])) * (X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])
                nM_E0Sum += rM_E0
            end

        end
        prM_E0[nu] = pM_E0Sum
        nrM_E0[nu] = nM_E0Sum
    end

    # Case of E1 ...
    J, P = 1, 2
    N_ph = N_nu[J+1,P]

    # Initialize RRPA matrix elements ...
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

    # Evaluate RRPA E1 matrix elements ...
    @inbounds Threads.@threads for nu in 1:N_ph
        pM_E1Sum = ComplexF64(0.0)
        pM_E1VCSum = ComplexF64(0.0)
        pM_E1VSSum = ComplexF64(0.0)
        pM_E1TCSum = ComplexF64(0.0)
        pM_E1TSSum = ComplexF64(0.0)
        pM_E1sTCSum = ComplexF64(0.0)
        pM_E1sTSSum = ComplexF64(0.0)
        pM_E1CSum = ComplexF64(0.0)
        pM_E1sCSum = ComplexF64(0.0)

        nM_E1Sum = ComplexF64(0.0)
        nM_E1VCSum = ComplexF64(0.0)
        nM_E1VSSum = ComplexF64(0.0)
        nM_E1TCSum = ComplexF64(0.0)
        nM_E1TSSum = ComplexF64(0.0)
        nM_E1sTCSum = ComplexF64(0.0)
        nM_E1sTSSum = ComplexF64(0.0)
        nM_E1CSum = ComplexF64(0.0)
        nM_E1sCSum = ComplexF64(0.0)

        @inbounds for ph in 1:N_ph
            i_ph = Orb_Phonon[J+1,P][ph]
            p, h, t_ph = Phonon[i_ph].p, Phonon[i_ph].h, Phonon[i_ph].tz

            if t_ph == -1
                a_p, a_h = Particle.p[p].a, Hole.p[h].a

                pM_E1 = ComplexF64(TrOp.E1.p[a_p,a_h] * sqrt(Rho.p[a_h,a_h] - Rho.p[a_p,a_p])) * (-1.0 * X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])
                pM_E1VCME = ComplexF64(-1.0 * TrOp.E1_VC.p[a_p,a_h] * sqrt(Rho.p[a_h,a_h] - Rho.p[a_p,a_p])) * (-1.0 * X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])
                pM_E1VSME = ComplexF64(-1.0 * TrOp.E1_VS.p[a_p,a_h] * sqrt(Rho.p[a_h,a_h] - Rho.p[a_p,a_p])) * (-1.0 * X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])
                pM_E1TCME = ComplexF64(-1.0 * TrOp.E1_TC.p[a_p,a_h] * sqrt(Rho.p[a_h,a_h] - Rho.p[a_p,a_p])) * (-1.0 * X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])
                pM_E1TSME = ComplexF64(-1.0 * TrOp.E1_TS.p[a_p,a_h] * sqrt(Rho.p[a_h,a_h] - Rho.p[a_p,a_p])) * (-1.0 * X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])
                pM_E1sTCME = ComplexF64(-1.0 * TrOp.E1_sTC.p[a_p,a_h] * sqrt(Rho.p[a_h,a_h] - Rho.p[a_p,a_p])) * (-1.0 * X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])
                pM_E1sTSME = ComplexF64(-1.0 * TrOp.E1_sTS.p[a_p,a_h] * sqrt(Rho.p[a_h,a_h] - Rho.p[a_p,a_p])) * (-1.0 * X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])
                pM_E1CME = ComplexF64(-1.0 * TrOp.E1_C.p[a_p,a_h] * sqrt(Rho.p[a_h,a_h] - Rho.p[a_p,a_p])) * (-1.0 * X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])
                pM_E1sCME = ComplexF64(-1.0 * TrOp.E1_sC.p[a_p,a_h] * sqrt(Rho.p[a_h,a_h] - Rho.p[a_p,a_p])) * (-1.0 * X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])

                pM_E1Sum += pM_E1
                pM_E1VCSum += pM_E1VCME
                pM_E1VSSum += pM_E1VSME
                pM_E1TCSum += pM_E1TCME
                pM_E1TSSum += pM_E1TSME
                pM_E1sTCSum += pM_E1sTCME
                pM_E1sTSSum += pM_E1sTSME
                pM_E1CSum += pM_E1CME
                pM_E1sCSum += pM_E1sCME

            elseif t_ph == 1
                a_p, a_h = Particle.n[p].a, Hole.n[h].a

                nM_E1 = ComplexF64(TrOp.E1.n[a_p,a_h] * sqrt(Rho.n[a_h,a_h] - Rho.n[a_p,a_p])) * (-1.0 * X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])
                nM_E1VCME = ComplexF64(-1.0 * TrOp.E1_VC.n[a_p,a_h] * sqrt(Rho.n[a_h,a_h] - Rho.n[a_p,a_p])) * (-1.0 * X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])
                nM_E1VSME = ComplexF64(-1.0 * TrOp.E1_VS.n[a_p,a_h] * sqrt(Rho.n[a_h,a_h] - Rho.n[a_p,a_p]))* (-1.0 * X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])
                nM_E1TCME = ComplexF64(-1.0 * TrOp.E1_TC.n[a_p,a_h] * sqrt(Rho.n[a_h,a_h] - Rho.n[a_p,a_p])) * (-1.0 * X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])
                nM_E1TSME = ComplexF64(-1.0 * TrOp.E1_TS.n[a_p,a_h] * sqrt(Rho.n[a_h,a_h] - Rho.n[a_p,a_p])) * (-1.0 * X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])
                nM_E1sTCME = ComplexF64(-1.0 * TrOp.E1_sTC.n[a_p,a_h] * sqrt(Rho.n[a_h,a_h] - Rho.n[a_p,a_p])) * (-1.0 * X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])
                nM_E1sTSME = ComplexF64(-1.0 * TrOp.E1_sTS.n[a_p,a_h] * sqrt(Rho.n[a_h,a_h] - Rho.n[a_p,a_p])) * (-1.0 * X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])
                nM_E1CME = ComplexF64(-1.0 * TrOp.E1_C.n[a_p,a_h] * sqrt(Rho.n[a_h,a_h] - Rho.n[a_p,a_p])) * (-1.0 * X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])
                nM_E1sCME = ComplexF64(-1.0 * TrOp.E1_sC.n[a_p,a_h] * sqrt(Rho.n[a_h,a_h] - Rho.n[a_p,a_p])) * (-1.0 * X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])

                nM_E1Sum += nM_E1
                nM_E1VCSum += nM_E1VCME
                nM_E1VSSum += nM_E1VSME
                nM_E1TCSum += nM_E1TCME
                nM_E1TSSum += nM_E1TSME
                nM_E1sTCSum += nM_E1sTCME
                nM_E1sTSSum += nM_E1sTSME
                nM_E1CSum += nM_E1CME
                nM_E1sCSum += nM_E1sCME
            end
        end

        prM_E1[nu] = pM_E1Sum
        prM_E1VC[nu] = pM_E1VCSum
        prM_E1VS[nu] = pM_E1VSSum
        prM_E1TC[nu] = pM_E1TCSum
        prM_E1TS[nu] = pM_E1TSSum
        prM_E1sTC[nu] = pM_E1sTCSum
        prM_E1sTS[nu] = pM_E1sTSSum
        prM_E1C[nu] = pM_E1CSum
        prM_E1sC[nu] = pM_E1sCSum

        nrM_E1[nu] = nM_E1Sum
        nrM_E1VC[nu] = nM_E1VCSum
        nrM_E1VS[nu] = nM_E1VSSum
        nrM_E1TC[nu] = nM_E1TCSum
        nrM_E1TS[nu] = nM_E1TSSum
        nrM_E1sTC[nu] = nM_E1sTCSum
        nrM_E1sTS[nu] = nM_E1sTSSum
        nrM_E1C[nu] = nM_E1CSum
        nrM_E1sC[nu] = nM_E1sCSum

    end

    # Case of E2 ...
    J, P = 2, 1
    N_ph = N_nu[J+1,P]

    # Initialize RRPA matrix elements ...
    prM_E2 = Vector{ComplexF64}(undef,N_ph)
    nrM_E2 = Vector{ComplexF64}(undef,N_ph)

    # Evaluate RRPA E2 matrix elements ...
    @inbounds Threads.@threads for nu in 1:N_ph
        pM_E2Sum = ComplexF64(0.0)
        nM_E2Sum = ComplexF64(0.0)

        @inbounds for ph in 1:N_ph
            i_ph = Orb_Phonon[J+1,P][ph]
            p, h, t_ph = Phonon[i_ph].p, Phonon[i_ph].h, Phonon[i_ph].tz

            if t_ph == -1
                a_p, a_h = Particle.p[p].a, Hole.p[h].a

                pM_E2 = ComplexF64(TrOp.E2.p[a_p,a_h] * sqrt(Rho.p[a_h,a_h] - Rho.p[a_p,a_p])) * (X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])
                pM_E2Sum += pM_E2

            elseif t_ph == 1
                a_p, a_h = Particle.n[p].a, Hole.n[h].a

                nM_E2 = ComplexF64(TrOp.E2.n[a_p,a_h] * sqrt(Rho.n[a_h,a_h] - Rho.n[a_p,a_p])) * (X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])
                nM_E2Sum += nM_E2
            end
        end
        prM_E2[nu] = pM_E2Sum
        nrM_E2[nu] = nM_E2Sum
    end

    # Case of E3 ...
    J, P = 3, 2
    N_ph = N_nu[J+1,P]

    # Initialize RRPA matrix elements ...
    prM_E3 = Vector{ComplexF64}(undef,N_ph)
    nrM_E3 = Vector{ComplexF64}(undef,N_ph)

    # Evaluate RRPA E3 matrix elements ...
    @inbounds Threads.@threads for nu in 1:N_ph
        pM_E3Sum = ComplexF64(0.0)
        nM_E3Sum = ComplexF64(0.0)
        @inbounds for ph in 1:N_ph
            i_ph = Orb_Phonon[J+1,P][ph]
            p, h, t_ph = Phonon[i_ph].p, Phonon[i_ph].h, Phonon[i_ph].tz

            if t_ph == -1
                a_p, a_h = Particle.p[p].a, Hole.p[h].a

                pM_E3 = ComplexF64(TrOp.E3.p[a_p,a_h] * sqrt(Rho.p[a_h,a_h] - Rho.p[a_p,a_p])) * (X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])
                pM_E3Sum += pM_E3

            elseif t_ph == 1
                a_p, a_h = Particle.n[p].a, Hole.n[h].a

                nM_E3 = ComplexF64(TrOp.E3.n[a_p,a_h] * sqrt(Rho.n[a_h,a_h] - Rho.n[a_p,a_p])) * (X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])
                nM_E3Sum += nM_E3
            end
        end
        prM_E3[nu] = pM_E3Sum
        nrM_E3[nu] = nM_E3Sum
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

    rM = ReducedMultipole(rM_E0,rM_E1,rM_E2,rM_E3,rM_E1VC,rM_E1VS,rM_E1TC,rM_E1TS,
                          rM_E1sTC,rM_E1sTS,rM_E1C,rM_E1sC)

    println("\tReduced 1-body transition matrix elements rM calculated ...")

    return rM
end

function HF_RRPA_rB(Params::Parameters,N_nu::Matrix{Int64},rM::ReducedMultipole,E_RPA::Matrix{Vector{ComplexF64}})
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
    N_ph = N_nu[J+1,P]

    # Initialize the reduced transition intensities B ...
    rB_E0 = Vector{Vector{Float64}}(undef,3)

    # Initialize the sub-components of the reduced transition intensities B ...
        # Physical (ph), Isoscalar (is), Isovector (iv) ...
    @inbounds for i in 1:3
        rB_E0[i] = Vector{Float64}(undef,N_ph)
    end

    # Evaluate the reduced transition intensities B ...
    @inbounds Threads.@threads for nu in 1:N_ph
        rB_E0[1][nu] = abs(rM.E0.p[nu])^2
        rB_E0[2][nu] = 0.25 * abs(rM.E0.p[nu] + rM.E0.n[nu])^2
        rB_E0[3][nu] = 0.25 * abs(rM.E0.p[nu] - rM.E0.n[nu])^2
    end

    # Case of E1 ...
    J, P = 1, 2
    N_ph = N_nu[J+1,P]

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
        rB_E1[i] = Vector{Float64}(undef,N_ph)
        rB_E1V[i] = Vector{Float64}(undef,N_ph)
        rB_E1VC[i] = Vector{Float64}(undef,N_ph)
        rB_E1VS[i] = Vector{Float64}(undef,N_ph)
        rB_E1T[i] = Vector{Float64}(undef,N_ph)
        rB_E1TC[i] = Vector{Float64}(undef,N_ph)
        rB_E1TS[i] = Vector{Float64}(undef,N_ph)
        rB_E1C[i] = Vector{Float64}(undef,N_ph)
        rB_E1CC[i] = Vector{Float64}(undef,N_ph)
        rB_E1CS[i] = Vector{Float64}(undef,N_ph)
        rB_E1c[i] = Vector{Float64}(undef,N_ph)
        rB_E1_NLO_LWA[i] = Vector{Float64}(undef,N_ph)
    end

    # Define effective E1 isovector charges ...
    e_p = Float64(A - Z) / Float64(A)
    e_n = Float64(Z) / Float64(A)

    # Evaluate the reduced transition intensities B ...
    @inbounds Threads.@threads for nu in 1:N_ph
        E = real(E_RPA[J+1,P][nu])
            # Physical components ...
        rB_E1[1][nu] = abs(rM.E1.p[nu])^2
        rB_E1V[1][nu] = abs(rM.E1_VC.p[nu] + 0.5 * g_p * rM.E1_VS.p[nu] + 0.5 * g_n * rM.E1_VS.n[nu])^2
        rB_E1VC[1][nu] = abs(rM.E1_VC.p[nu])^2
        rB_E1VS[1][nu] = abs(0.5 * g_p * rM.E1_VS.p[nu] + 0.5 * g_n * rM.E1_VS.n[nu])^2

        rB_E1T[1][nu] = abs(rM.E1_TC.p[nu] + 0.5 * g_p * rM.E1_TS.p[nu] + 0.5 * g_n * rM.E1_TS.n[nu])^2
        rB_E1TC[1][nu] = abs(rM.E1_TC.p[nu])^2
        rB_E1TS[1][nu] = abs(0.5 * g_p * rM.E1_TS.p[nu] + 0.5 * g_n * rM.E1_TS.n[nu])^2

        rB_E1C[1][nu] = abs(rM.E1_TC.p[nu] + 0.5 * g_p * rM.E1_TS.p[nu] + 0.5 * g_n * rM.E1_TS.n[nu] - (rM.E1_VC.p[nu] + 0.5 * g_p * rM.E1_VS.p[nu] + 0.5 * g_n * rM.E1_VS.n[nu]))^2
        rB_E1CC[1][nu] = abs(rM.E1_TC.p[nu] - rM.E1_VC.p[nu])^2
        rB_E1CS[1][nu] = abs(0.5 * g_p * rM.E1_TS.p[nu] + 0.5 * g_n * rM.E1_TS.n[nu] - (0.5 * g_p * rM.E1_VS.p[nu] + 0.5 * g_n * rM.E1_VS.n[nu]))^2

        rB_E1c[1][nu] = abs(rM.E1_C.p[nu])^2

        rB_E1_NLO_LWA[1][nu] = abs(rM.E1.p[nu] + E / hc * (rM.E1_TC.p[nu] + 0.5 * g_p * rM.E1_TS.p[nu] + 0.5 * g_n * rM.E1_TS.n[nu]))^2

            # Isoscalar components ...
        rB_E1[2][nu] = 0.25 * abs(rM.E1.p[nu] + rM.E1.n[nu])^2
        rB_E1V[2][nu] = abs(0.5 * (rM.E1_VC.p[nu] + rM.E1_VC.n[nu]) + 0.125 * (g_p + g_n) * (rM.E1_VS.p[nu] + rM.E1_VS.n[nu]))^2
        rB_E1VC[2][nu] = abs(0.5 * (rM.E1_VC.p[nu] + rM.E1_VC.n[nu]))^2
        rB_E1VS[2][nu] = abs(0.125 * (g_p + g_n) * (rM.E1_VS.p[nu] + rM.E1_VS.n[nu]))^2
        
        rB_E1T[2][nu] = abs(0.5 * (rM.E1_TC.p[nu] + rM.E1_sTC.p[nu] + rM.E1_TC.n[nu] + rM.E1_sTC.n[nu]) + 0.125 * (g_p + g_n) * (rM.E1_TS.p[nu] + rM.E1_sTS.p[nu] + rM.E1_TS.n[nu] + rM.E1_sTS.n[nu]))^2
        rB_E1TC[2][nu] = abs(0.5 * (rM.E1_TC.p[nu] + rM.E1_sTC.p[nu] + rM.E1_TC.n[nu] + rM.E1_sTC.n[nu]))^2
        rB_E1TS[2][nu] = abs(0.125 * (g_p + g_n) * (rM.E1_TS.p[nu] + rM.E1_sTS.p[nu] + rM.E1_TS.n[nu] + rM.E1_sTS.n[nu]))^2

        rB_E1C[2][nu] = abs(0.5 * (rM.E1_TC.p[nu] + rM.E1_sTC.p[nu] + rM.E1_TC.n[nu] + rM.E1_sTC.n[nu]) + 0.125 * (g_p + g_n) * (rM.E1_TS.p[nu] + rM.E1_sTS.p[nu] + rM.E1_TS.n[nu] + rM.E1_sTS.n[nu]) - (0.5 * (rM.E1_VC.p[nu] + rM.E1_VC.n[nu]) + 0.125 * (g_p + g_n) * (rM.E1_VS.p[nu] + rM.E1_VS.n[nu])))^2
        rB_E1CC[2][nu] = abs(0.5 * (rM.E1_TC.p[nu] + rM.E1_sTC.p[nu] + rM.E1_TC.n[nu] + rM.E1_sTC.n[nu]) - (0.5 * (rM.E1_VC.p[nu] + rM.E1_VC.n[nu])))^2
        rB_E1CS[2][nu] = abs(0.125 * (g_p + g_n) * (rM.E1_TS.p[nu] + rM.E1_sTS.p[nu] + rM.E1_TS.n[nu] + rM.E1_sTS.n[nu]) - (0.125 * (g_p + g_n) * (rM.E1_VS.p[nu] + rM.E1_VS.n[nu])))^2

        rB_E1c[2][nu] = 0.25 * abs(rM.E1_C.p[nu] + rM.E1_sC.p[nu] + rM.E1_C.n[nu] + rM.E1_sC.n[nu])^2

        rB_E1_NLO_LWA[2][nu] = 0.25 * abs(rM.E1.p[nu] + rM.E1.n[nu] +  E / hc * ((rM.E1_TC.p[nu] + rM.E1_sTC.p[nu] + rM.E1_TC.n[nu] + rM.E1_sTC.n[nu]) + 0.25 * (g_p + g_n) * (rM.E1_TS.p[nu] + rM.E1_sTS.p[nu] + rM.E1_TS.n[nu] + rM.E1_sTS.n[nu])))^2

            # Isovector components ...
        rB_E1[3][nu] = abs(e_p * rM.E1.p[nu] - e_n * rM.E1.n[nu])^2
        rB_E1V[3][nu] = abs(0.5 * (rM.E1_VC.p[nu] - rM.E1_VC.n[nu]) + 0.125 * (g_p - g_n) * (rM.E1_VS.p[nu] + rM.E1_VS.n[nu]))^2
        rB_E1VC[3][nu] = abs(0.5 * (rM.E1_VC.p[nu] - rM.E1_VC.n[nu]))^2
        rB_E1VS[3][nu] = abs(0.125 * (g_p - g_n) * (rM.E1_VS.p[nu] + rM.E1_VS.n[nu]))^2
        
        rB_E1T[3][nu] = abs(0.5 * (rM.E1_TC.p[nu] - rM.E1_TC.n[nu]) + 0.125 * (g_p - g_n) * (rM.E1_TS.p[nu] + rM.E1_TS.n[nu]))^2
        rB_E1TC[3][nu] = abs(0.5 * (rM.E1_TC.p[nu] - rM.E1_TC.n[nu]))^2
        rB_E1TS[3][nu] = abs(0.125 * (g_p - g_n) * (rM.E1_TS.p[nu] + rM.E1_TS.n[nu]))^2

        rB_E1C[3][nu] = abs(0.5 * (rM.E1_TC.p[nu] - rM.E1_TC.n[nu]) + 0.125 * (g_p - g_n) * (rM.E1_TS.p[nu] + rM.E1_TS.n[nu]) - (0.5 * (rM.E1_VC.p[nu] - rM.E1_VC.n[nu]) + 0.125 * (g_p - g_n) * (rM.E1_VS.p[nu] + rM.E1_VS.n[nu])))^2
        rB_E1CC[3][nu] = abs(0.5 * (rM.E1_TC.p[nu] - rM.E1_TC.n[nu]) - (0.5 * (rM.E1_VC.p[nu] - rM.E1_VC.n[nu])))^2
        rB_E1CS[3][nu] = abs(0.125 * (g_p - g_n) * (rM.E1_TS.p[nu] + rM.E1_TS.n[nu]) - (0.125 * (g_p - g_n) * (rM.E1_VS.p[nu] + rM.E1_VS.n[nu])))^2

        rB_E1c[3][nu] = 0.25 * abs(rM.E1_C.p[nu] - rM.E1_C.n[nu])^2

        rB_E1_NLO_LWA[3][nu] = abs(e_p * rM.E1.p[nu] - e_n * rM.E1.n[nu] +  0.5 * E / hc * ((rM.E1_TC.p[nu] - rM.E1_TC.n[nu]) + 0.125 * (g_p - g_n) * (rM.E1_TS.p[nu] + rM.E1_TS.n[nu])))^2
    end

    # Case of E2 ...
    J, P = 2, 1
    N_ph = N_nu[J+1,P]

    # Initialize the reduced transition intensities B ...
    rB_E2 = Vector{Vector{Float64}}(undef,3)

    # Initialize the sub-components of the reduced transition intensities B ...
        # Physical (ph), Isoscalar (is), Isovector (iv) ...
    @inbounds for i in 1:3
        rB_E2[i] = Vector{Float64}(undef,N_ph)
    end

    # Evaluate the reduced transition intensities B ...
    @inbounds Threads.@threads for nu in 1:N_ph
        rB_E2[1][nu] = abs(rM.E2.p[nu])^2
        rB_E2[2][nu] = 0.25 * abs(rM.E2.p[nu] + rM.E2.n[nu])^2
        rB_E2[3][nu] = 0.25 * abs(rM.E2.p[nu] - rM.E2.n[nu])^2
    end

    # Case of E3 ...
    J, P = 3, 2
    N_ph = N_nu[J+1,P]

    # Initialize the reduced transition intensities B ...
    rB_E3 = Vector{Vector{Float64}}(undef,3)

    # Initialize the sub-components of the reduced transition intensities B ...
        # Physical (ph), Isoscalar (is), Isovector (iv) ...
    @inbounds for i in 1:3
        rB_E3[i] = Vector{Float64}(undef,N_ph)
    end

    # Evaluate the reduced transition intensities B ...
    @inbounds Threads.@threads for nu in 1:N_ph
        rB_E3[1][nu] = abs(rM.E3.p[nu])^2
        rB_E3[2][nu] = 0.25 * abs(rM.E3.p[nu] + rM.E3.n[nu])^2
        rB_E3[3][nu] = 0.25 * abs(rM.E3.p[nu] - rM.E3.n[nu])^2
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

#=
function HF_RRPA_transition_densities_export(Params::Parameters,Orb::Vector{Orb1B},N_nu::Matrix{Int64},Orb_Phonon::Matrix{Vector{Int64}},Phonon::Vector{PhState},Particle::pnSVector,Hole::pnSVector,X_RPA::Matrix{Matrix{ComplexF64}},Y_RPA::Matrix{Matrix{ComplexF64}},TrOp::Tr1B)
    # Read parameters ...
    A = Params.Calc.A
    Z = Params.Calc.Z
    HbarOmega = Params.Calc.hw
    N_max = Params.Calc.Nmax
    a_max = div((N_max+1)*(N_max+2),2)
    Output_File = Params.Calc.Path

    # Calc params ...
    J = 1
    P = 2
    N_nu_max = 15
    N_ph = N_nu[J+1,P]

    # Basic constants ...
    HbarC = 197.326980
    m_n = 939.565346
    m_p = 938.272013
    nu_proton = 0.5 * m_p * HbarOmega / HbarC^2
    nu_neutron = 0.5 * m_n * HbarOmega / HbarC^2

    # Import HF basis transformation matrices ...
    C = transformation_matrix_read(Params,Orb,"IO/" * Params.Calc.Path * "/Bin/C_HF.bin")

    # Preallocate grid ...
    r1 = 0.0 + 1e-8
    r2 = 2.5 * 1.2 * A^(1/3)
    N_Sampling = 10000
    r_grid = range(r1, stop = r2, length = N_Sampling)
    r_grid = collect(r_grid)
    pRho_ERPA_rad = Matrix{Float64}(undef,N_Sampling,N_nu_max)
    nRho_ERPA_rad = Matrix{Float64}(undef,N_Sampling,N_nu_max)

    # Evaluate radial densities on grid ...
    @inbounds Threads.@threads for i in 1:N_Sampling
        r = r_grid[i]
        @inbounds for nu in 1:N_nu_max
            pRad, nRad = 0.0, 0.0
            @inbounds for ph in 1:N_ph
                ind_ph = Orb_Phonon[J+1,P][ph]
                p, h = Phonon[ind_ph].p, Phonon[ind_ph].h
                t_ph = Phonon[ind_ph].tz
                if t_ph == -1
                    a_p, l_p, j_p = Particle.p[p].a, Particle.p[p].l, Particle.p[p].j
                    a_h, l_h, j_h = Hole.p[h].a, Hole.p[h].l, Hole.p[h].j
                elseif t_ph == 1
                    a_p, l_p, j_p = Particle.n[p].a, Particle.n[p].l, Particle.n[p].j
                    a_h, l_h, j_h = Hole.n[h].a, Hole.n[h].l, Hole.n[h].j
                end
                
                pPsi, nPsi = 0.0, 0.0
                @inbounds for a_k in 1:a_max
                    l_k = Orb[a_k].l
                    j_k = Orb[a_k].j
                    n_k = Orb[a_k].n
                    if (j_p == j_k) && (l_p == l_k)
                        @inbounds for a_l in 1:a_max
                            l_l = Orb[a_l].l
                            j_l = Orb[a_l].j
                            n_l = Orb[a_l].n
                            if (j_h == j_l) && (l_h == l_l)
                                if t_ph == -1
                                    Psi = Psi_rad_LHO(r,n_k,l_k,nu_proton) * Psi_rad_LHO(r,n_l,l_l,nu_proton) * C.p[a_k,a_p] * C.p[a_l,a_h]
                                    pPsi += Psi
                                elseif t_ph == 1
                                    Psi = Psi_rad_LHO(r,n_k,l_k,nu_neutron) * Psi_rad_LHO(r,n_l,l_l,nu_neutron) * C.n[a_k,a_p] * C.n[a_l,a_h]
                                    nPsi += Psi
                                end
                            end
                        end
                    end
                end
                if t_ph == -1
                    Amp = TrOp.E1.p[a_p,a_h] * (-1.0 * real(X_RPA[J+1,P][ph,nu]) + real(Y_RPA[J+1,P][ph,nu])) / Float64(2*J + 1)
                    pRad += pPsi * Amp
                elseif t_ph == 1
                    Amp = TrOp.E1.n[a_p,a_h] * (-1.0 * real(X_RPA[J+1,P][ph,nu]) + real(Y_RPA[J+1,P][ph,nu])) / Float64(2*J + 1)
                    nRad += nPsi * Amp
                end
            end
            pRho_ERPA_rad[i,nu] = pRad * r^2
            nRho_ERPA_rad[i,nu] = nRad * r^2
        end
    end

    @views r_grid = round.(r_grid,digits = 7)
    @views pRho_ERPA_rad = round.(pRho_ERPA_rad,digits = 7)
    @views nRho_ERPA_rad = round.(nRho_ERPA_rad,digits = 7)

    # Export path ...
    Output_Path = Output_File * "/RRPA/Densities/HF_RRPA_Radial_Transition_Densities.dat"

    open(Output_Path, "w") do Export_File
        @inbounds for i in 1:N_Sampling
            print(Export_File, r_grid[i])
            print(Export_File, "\t")
            @inbounds for nu in 1:N_nu_max
                print(Export_File, pRho_ERPA_rad[i,nu])
                print(Export_File, "\t")
                print(Export_File, nRho_ERPA_rad[i,nu])
                print(Export_File, "\t")
            end
            print(Export_File, "\n")
        end
    end

    return
end
=#