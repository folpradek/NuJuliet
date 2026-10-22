function RPA_rM(Params::Parameters,N_nu::Matrix{Int64},Orb_Phonon::Matrix{Vector{Int64}},Phonon::Vector{PhState},Particle::pnSVector,Hole::pnSVector,X_RPA::Matrix{Matrix{ComplexF64}},Y_RPA::Matrix{Matrix{ComplexF64}},TrOp::Tr1B)
    # Evaluate the reduced transition matrix elements M^lambda ...
    println("\nCalculating reduced transition matrix elements rM ...")

    # Case of E0 ...
    J, P = 0, 1
    N_ph = N_nu[J+1,P]

    # Initialize RPA matrix elements ...
    prM_E0_RPA = Vector{ComplexF64}(undef,N_ph)
    nrM_E0_RPA = Vector{ComplexF64}(undef,N_ph)

    # Evaluate RPA E0 matrix elements ...
    @inbounds Threads.@threads for nu in 1:N_ph

        pM_E0Sum_RPA = ComplexF64(0.0)
        nM_E0Sum_RPA = ComplexF64(0.0)

        @inbounds for ph in 1:N_ph
            i_ph = Orb_Phonon[J+1,P][ph]
            p, h, t_ph = Phonon[i_ph].p, Phonon[i_ph].h, Phonon[i_ph].tz

            if t_ph == -1
                a_p, a_h = Particle.p[p].a, Hole.p[h].a

                pM_E0_RPA = ComplexF64(TrOp.E0.p[a_p,a_h]) * (X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])
                pM_E0Sum_RPA += pM_E0_RPA

            elseif t_ph == 1
                a_p, a_h = Particle.n[p].a, Hole.n[h].a

                nM_E0_RPA = ComplexF64(TrOp.E0.n[a_p,a_h]) * (X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])
                nM_E0Sum_RPA += nM_E0_RPA

            end


        end

        prM_E0_RPA[nu] = pM_E0Sum_RPA
        nrM_E0_RPA[nu] = nM_E0Sum_RPA
    end

    # Case of E1 ...
    J, P = 1, 2
    N_ph = N_nu[J+1,P]

    # Initialize RPA matrix elements ...
    prM_E1_RPA = Vector{ComplexF64}(undef,N_ph)
    prM_E1VC_RPA = Vector{ComplexF64}(undef,N_ph)
    prM_E1VS_RPA = Vector{ComplexF64}(undef,N_ph)
    prM_E1TC_RPA = Vector{ComplexF64}(undef,N_ph)
    prM_E1TS_RPA = Vector{ComplexF64}(undef,N_ph)
    prM_E1sTC_RPA = Vector{ComplexF64}(undef,N_ph)
    prM_E1sTS_RPA = Vector{ComplexF64}(undef,N_ph)
    prM_E1C_RPA = Vector{ComplexF64}(undef,N_ph)
    prM_E1sC_RPA = Vector{ComplexF64}(undef,N_ph)

    nrM_E1_RPA = Vector{ComplexF64}(undef,N_ph)
    nrM_E1VC_RPA = Vector{ComplexF64}(undef,N_ph)
    nrM_E1VS_RPA = Vector{ComplexF64}(undef,N_ph)
    nrM_E1TC_RPA = Vector{ComplexF64}(undef,N_ph)
    nrM_E1TS_RPA = Vector{ComplexF64}(undef,N_ph)
    nrM_E1sTC_RPA = Vector{ComplexF64}(undef,N_ph)
    nrM_E1sTS_RPA = Vector{ComplexF64}(undef,N_ph)
    nrM_E1C_RPA = Vector{ComplexF64}(undef,N_ph)
    nrM_E1sC_RPA = Vector{ComplexF64}(undef,N_ph)

    # Evaluate TDA & RPA E1 matrix elements ...
    @inbounds Threads.@threads for nu in 1:N_ph
        pM_E1Sum_RPA = ComplexF64(0.0)
        pM_E1VCSum_RPA = ComplexF64(0.0)
        pM_E1VSSum_RPA = ComplexF64(0.0)
        pM_E1TCSum_RPA = ComplexF64(0.0)
        pM_E1TSSum_RPA = ComplexF64(0.0)
        pM_E1sTCSum_RPA = ComplexF64(0.0)
        pM_E1sTSSum_RPA = ComplexF64(0.0)
        pM_E1CSum_RPA = ComplexF64(0.0)
        pM_E1sCSum_RPA = ComplexF64(0.0)

        nM_E1Sum_RPA = ComplexF64(0.0)
        nM_E1VCSum_RPA = ComplexF64(0.0)
        nM_E1VSSum_RPA = ComplexF64(0.0)
        nM_E1TCSum_RPA = ComplexF64(0.0)
        nM_E1TSSum_RPA = ComplexF64(0.0)
        nM_E1sTCSum_RPA = ComplexF64(0.0)
        nM_E1sTSSum_RPA = ComplexF64(0.0)
        nM_E1CSum_RPA = ComplexF64(0.0)
        nM_E1sCSum_RPA = ComplexF64(0.0)

        @inbounds for ph in 1:N_ph
            i_ph = Orb_Phonon[J+1,P][ph]
            p, h, t_ph = Phonon[i_ph].p, Phonon[i_ph].h, Phonon[i_ph].tz

            if t_ph == -1
                a_p, a_h = Particle.p[p].a, Hole.p[h].a

                pM_E1_RPA = ComplexF64(TrOp.E1.p[a_p,a_h]) * (-1.0 * X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])
                pM_E1VCME_RPA = ComplexF64(TrOp.E1_VC.p[a_p,a_h]) * (-1.0 * X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])
                pM_E1VSME_RPA = ComplexF64(TrOp.E1_VS.p[a_p,a_h]) * (-1.0 * X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])
                pM_E1TCME_RPA = ComplexF64(TrOp.E1_TC.p[a_p,a_h]) * (-1.0 * X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])
                pM_E1TSME_RPA = ComplexF64(TrOp.E1_TS.p[a_p,a_h]) * (-1.0 * X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])
                pM_E1sTCME_RPA = ComplexF64(TrOp.E1_sTC.p[a_p,a_h]) * (-1.0 * X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])
                pM_E1sTSME_RPA = ComplexF64(TrOp.E1_sTS.p[a_p,a_h]) * (-1.0 * X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])
                pM_E1CME_RPA = ComplexF64(TrOp.E1_C.p[a_p,a_h]) * (-1.0 * X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])
                pM_E1sCME_RPA = ComplexF64(TrOp.E1_sC.p[a_p,a_h]) * (-1.0 * X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])

                pM_E1Sum_RPA += pM_E1_RPA
                pM_E1VCSum_RPA += pM_E1VCME_RPA
                pM_E1VSSum_RPA += pM_E1VSME_RPA
                pM_E1TCSum_RPA += pM_E1TCME_RPA
                pM_E1TSSum_RPA += pM_E1TSME_RPA
                pM_E1sTCSum_RPA += pM_E1sTCME_RPA
                pM_E1sTSSum_RPA += pM_E1sTSME_RPA
                pM_E1CSum_RPA += pM_E1CME_RPA
                pM_E1sCSum_RPA += pM_E1sCME_RPA

            elseif t_ph == 1
                a_p, a_h = Particle.n[p].a, Hole.n[h].a

                nM_E1_RPA = ComplexF64(TrOp.E1.n[a_p,a_h]) * (-1.0 * X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])
                nM_E1VCME_RPA = ComplexF64(TrOp.E1_VC.n[a_p,a_h]) * (-1.0 * X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])
                nM_E1VSME_RPA = ComplexF64(TrOp.E1_VS.n[a_p,a_h])* (-1.0 * X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])
                nM_E1TCME_RPA = ComplexF64(TrOp.E1_TC.n[a_p,a_h]) * (-1.0 * X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])
                nM_E1TSME_RPA = ComplexF64(TrOp.E1_TS.n[a_p,a_h]) * (-1.0 * X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])
                nM_E1sTCME_RPA = ComplexF64(TrOp.E1_sTC.n[a_p,a_h]) * (-1.0 * X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])
                nM_E1sTSME_RPA = ComplexF64(TrOp.E1_sTS.n[a_p,a_h]) * (-1.0 * X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])
                nM_E1CME_RPA = ComplexF64(TrOp.E1_C.n[a_p,a_h]) * (-1.0 * X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])
                nM_E1sCME_RPA = ComplexF64(TrOp.E1_sC.n[a_p,a_h]) * (-1.0 * X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])

                nM_E1Sum_RPA += nM_E1_RPA
                nM_E1VCSum_RPA += nM_E1VCME_RPA
                nM_E1VSSum_RPA += nM_E1VSME_RPA
                nM_E1TCSum_RPA += nM_E1TCME_RPA
                nM_E1TSSum_RPA += nM_E1TSME_RPA
                nM_E1sTCSum_RPA += nM_E1sTCME_RPA
                nM_E1sTSSum_RPA += nM_E1sTSME_RPA
                nM_E1CSum_RPA += nM_E1CME_RPA
                nM_E1sCSum_RPA += nM_E1sCME_RPA
            end

        end

        prM_E1_RPA[nu] = pM_E1Sum_RPA
        prM_E1VC_RPA[nu] = pM_E1VCSum_RPA
        prM_E1VS_RPA[nu] = pM_E1VSSum_RPA
        prM_E1TC_RPA[nu] = pM_E1TCSum_RPA
        prM_E1TS_RPA[nu] = pM_E1TSSum_RPA
        prM_E1sTC_RPA[nu] = pM_E1sTCSum_RPA
        prM_E1sTS_RPA[nu] = pM_E1sTSSum_RPA
        prM_E1C_RPA[nu] = pM_E1CSum_RPA
        prM_E1sC_RPA[nu] = pM_E1sCSum_RPA

        nrM_E1_RPA[nu] = nM_E1Sum_RPA
        nrM_E1VC_RPA[nu] = nM_E1VCSum_RPA
        nrM_E1VS_RPA[nu] = nM_E1VSSum_RPA
        nrM_E1TC_RPA[nu] = nM_E1TCSum_RPA
        nrM_E1TS_RPA[nu] = nM_E1TSSum_RPA
        nrM_E1sTC_RPA[nu] = nM_E1sTCSum_RPA
        nrM_E1sTS_RPA[nu] = nM_E1sTSSum_RPA
        nrM_E1C_RPA[nu] = nM_E1CSum_RPA
        nrM_E1sC_RPA[nu] = nM_E1sCSum_RPA
    end

    # Case of E2 ...
    J, P = 2, 1
    N_ph = N_nu[J+1,P]

    # Initialize RPA matrix elements ...
    prM_E2_RPA = Vector{ComplexF64}(undef,N_ph)
    nrM_E2_RPA = Vector{ComplexF64}(undef,N_ph)

    # Evaluate RPA E2 matrix elements ...
    @inbounds Threads.@threads for nu in 1:N_ph
        pM_E2Sum_RPA = ComplexF64(0.0)
        nM_E2Sum_RPA = ComplexF64(0.0)

        @inbounds for ph in 1:N_ph
            i_ph = Orb_Phonon[J+1,P][ph]
            p, h, t_ph = Phonon[i_ph].p, Phonon[i_ph].h, Phonon[i_ph].tz

            if t_ph == -1
                a_p, a_h = Particle.p[p].a, Hole.p[h].a

                pM_E2_RPA = ComplexF64(TrOp.E2.p[a_p,a_h]) * (X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])
                pM_E2Sum_RPA += pM_E2_RPA

            elseif t_ph == 1
                a_p, a_h = Particle.n[p].a, Hole.n[h].a

                nM_E2_RPA = ComplexF64(TrOp.E2.n[a_p,a_h]) * (X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])
                nM_E2Sum_RPA += nM_E2_RPA

            end

        end

        prM_E2_RPA[nu] = pM_E2Sum_RPA
        nrM_E2_RPA[nu] = nM_E2Sum_RPA

    end

    # Case of E3 ...
    J, P = 3, 2
    N_ph = N_nu[J+1,P]

    # Initialize RPA matrix elements ...
    prM_E3_RPA = Vector{ComplexF64}(undef,N_ph)
    nrM_E3_RPA = Vector{ComplexF64}(undef,N_ph)

    # Evaluate TDA & RPA E3 matrix elements ...
    @inbounds Threads.@threads for nu in 1:N_ph

        pM_E3Sum_RPA = ComplexF64(0.0)
        nM_E3Sum_RPA = ComplexF64(0.0)

        @inbounds for ph in 1:N_ph
            i_ph = Orb_Phonon[J+1,P][ph]
            p, h, t_ph = Phonon[i_ph].p, Phonon[i_ph].h, Phonon[i_ph].tz

            if t_ph == -1
                a_p, a_h = Particle.p[p].a, Hole.p[h].a

                pM_E3_RPA = ComplexF64(TrOp.E3.p[a_p,a_h]) * (X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])
                pM_E3Sum_RPA += pM_E3_RPA

            elseif t_ph == 1
                a_p, a_h = Particle.n[p].a, Hole.n[h].a

                nM_E3_RPA = ComplexF64(TrOp.E3.n[a_p,a_h]) * (X_RPA[J+1,P][ph,nu] + Y_RPA[J+1,P][ph,nu])
                nM_E3Sum_RPA += nM_E3_RPA

            end

        end

        prM_E3_RPA[nu] = pM_E3Sum_RPA
        nrM_E3_RPA[nu] = nM_E3Sum_RPA

    end

    # Allocate the reduced multipole transition matrix elements ...
    rM_E0_RPA = pnCVector(prM_E0_RPA,nrM_E0_RPA)
    rM_E1_RPA = pnCVector(prM_E1_RPA,nrM_E1_RPA)
    rM_E2_RPA = pnCVector(prM_E2_RPA,nrM_E2_RPA)
    rM_E3_RPA = pnCVector(prM_E3_RPA,nrM_E3_RPA)
    rM_E1VC_RPA = pnCVector(prM_E1VC_RPA,nrM_E1VC_RPA)
    rM_E1VS_RPA = pnCVector(prM_E1VS_RPA,nrM_E1VS_RPA)
    rM_E1TC_RPA = pnCVector(prM_E1TC_RPA,nrM_E1TC_RPA)
    rM_E1TS_RPA = pnCVector(prM_E1TS_RPA,nrM_E1TS_RPA)
    rM_E1sTC_RPA = pnCVector(prM_E1sTC_RPA,nrM_E1sTC_RPA)
    rM_E1sTS_RPA = pnCVector(prM_E1sTS_RPA,nrM_E1sTS_RPA)
    rM_E1C_RPA = pnCVector(prM_E1C_RPA,nrM_E1C_RPA)
    rM_E1sC_RPA = pnCVector(prM_E1sC_RPA,nrM_E1sC_RPA)

    rM_RPA = ReducedMultipole(rM_E0_RPA,rM_E1_RPA,rM_E2_RPA,rM_E3_RPA,rM_E1VC_RPA,
                             rM_E1VS_RPA,rM_E1TC_RPA,rM_E1TS_RPA,rM_E1sTC_RPA,
                             rM_E1sTS_RPA,rM_E1C_RPA,rM_E1sC_RPA)

    println("\tReduced 1-body transition matrix elements rM calculated ...")

    return rM_RPA
end

function RPA_rB(Params::Parameters,N_nu::Matrix{Int64},rM::ReducedMultipole,E_phonon::Matrix{Vector{Float64}})
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
        rB_E1_NLO_LWA[i] = Vector{Float64}(undef,N_ph)
    end

    # Define effective E1 isovector charges ...
    e_p = Float64(A - Z) / Float64(A)
    e_n = Float64(Z) / Float64(A)

    # Evaluate the reduced transition intensities B ...
    @inbounds Threads.@threads for nu in 1:N_ph
        E = E_phonon[J+1,P][nu]
            # Physical components ...
        rB_E1[1][nu] = abs(rM.E1.p[nu])^2
        rB_E1V[1][nu] = abs(rM.E1_VC.p[nu] + 0.5 * g_p * rM.E1_VS.p[nu] + 0.5 * g_n * rM.E1_VS.n[nu])^2
        rB_E1VC[1][nu] = abs(rM.E1_VC.p[nu])^2
        rB_E1VS[1][nu] = abs(0.5 * g_p * rM.E1_VS.p[nu] + 0.5 * g_n * rM.E1_VS.n[nu])^2

        rB_E1T[1][nu] = abs(rM.E1_TC.p[nu] + 0.5 * g_p * rM.E1_TS.p[nu] + 0.5 * g_n * rM.E1_TS.n[nu])^2
        rB_E1TC[1][nu] = abs(rM.E1_TC.p[nu])^2
        rB_E1TS[1][nu] = abs(0.5 * g_p * rM.E1_TS.p[nu] + 0.5 * g_n * rM.E1_TS.n[nu])^2

        rB_E1C[1][nu] = abs(rM.E1_C.p[nu])^2

        rB_E1_NLO_LWA[1][nu] = abs(rM.E1.p[nu] + E / hc * (rM.E1_TC.p[nu] + 0.5 * g_p * rM.E1_TS.p[nu] + 0.5 * g_n * rM.E1_TS.n[nu]))^2

            # Isoscalar components ...
        rB_E1[2][nu] = 0.25 * abs(rM.E1.p[nu] + rM.E1.n[nu])^2
        rB_E1V[2][nu] = abs(0.5 * (rM.E1_VC.p[nu] + rM.E1_VC.n[nu]) + 0.125 * (g_p + g_n) * (rM.E1_VS.p[nu] + rM.E1_VS.n[nu]))^2
        rB_E1VC[2][nu] = abs(0.5 * (rM.E1_VC.p[nu] + rM.E1_VC.n[nu]))^2
        rB_E1VS[2][nu] = abs(0.125 * (g_p + g_n) * (rM.E1_VS.p[nu] + rM.E1_VS.n[nu]))^2
        
        rB_E1T[2][nu] = abs(0.5 * (rM.E1_TC.p[nu] + rM.E1_sTC.p[nu] + rM.E1_TC.n[nu] + rM.E1_sTC.n[nu]) + 0.125 * (g_p + g_n) * (rM.E1_TS.p[nu] + rM.E1_sTS.p[nu] + rM.E1_TS.n[nu] + rM.E1_sTS.n[nu]))^2
        rB_E1TC[2][nu] = abs(0.5 * (rM.E1_TC.p[nu] + rM.E1_sTC.p[nu] + rM.E1_TC.n[nu] + rM.E1_sTC.n[nu]))^2
        rB_E1TS[2][nu] = abs(0.125 * (g_p + g_n) * (rM.E1_TS.p[nu] + rM.E1_sTS.p[nu] + rM.E1_TS.n[nu] + rM.E1_sTS.n[nu]))^2

        rB_E1C[2][nu] = 0.25 * abs(rM.E1_C.p[nu] + rM.E1_sC.p[nu] + rM.E1_C.n[nu] + rM.E1_sC.n[nu])^2

        rB_E1_NLO_LWA[2][nu] = 0.25 * abs(rM.E1.p[nu] + rM.E1.n[nu] +  E / hc * ((rM.E1_TC.p[nu] + rM.E1_sTC.p[nu] + rM.E1_TC.n[nu] + rM.E1_sTC.n[nu]) + 0.25 * (g_p + g_n) * (rM.E1_TS.p[nu] + rM.E1_sTS.p[nu] + rM.E1_TS.n[nu] + rM.E1_sTS.n[nu])))^2

            # Isovector components ...
        rB_E1[3][nu] = abs(e_p * rM.E1.p[nu] - e_n * rM.E1.n[nu])^2
        rB_E1V[3][nu] = abs(0.5 * (rM.E1_VC.p[nu] - rM.E1_VC.n[nu]) + 0.125 * (g_p - g_n) * (rM.E1_VS.p[nu] + rM.E1_VS.n[nu]))^2
        rB_E1VC[3][nu] = abs(0.5 * (rM.E1_VC.p[nu] - rM.E1_VC.n[nu]))^2
        rB_E1VS[3][nu] = abs(0.125 * (g_p - g_n) * (rM.E1_VS.p[nu] + rM.E1_VS.n[nu]))^2
        
        rB_E1T[3][nu] = abs(0.5 * (rM.E1_TC.p[nu] - rM.E1_TC.n[nu]) + 0.125 * (g_p - g_n) * (rM.E1_TS.p[nu] + rM.E1_TS.n[nu]))^2
        rB_E1TC[3][nu] = abs(0.5 * (rM.E1_TC.p[nu] - rM.E1_TC.n[nu]))^2
        rB_E1TS[3][nu] = abs(0.125 * (g_p - g_n) * (rM.E1_TS.p[nu] + rM.E1_TS.n[nu]))^2

        rB_E1C[3][nu] = 0.25 * abs(rM.E1_C.p[nu] - rM.E1_C.n[nu])^2

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
    rB_E1_NLO_LWA = Transition(rB_E1_NLO_LWA[1],rB_E1_NLO_LWA[2],rB_E1_NLO_LWA[3])

    rB = ReducedTransition(rB_E0,rB_E1,rB_E2,rB_E3,rB_E1V,rB_E1VC,rB_E1VS,rB_E1T,rB_E1TC,rB_E1TS,rB_E1C,rB_E1_NLO_LWA)

    println("\tReduced transition intensities rB evaluated ...")

    return rB
end