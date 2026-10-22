function Tr1b_initialize(Params::Parameters,Orb::Vector{Orb1B},chR2::Float64)
    println("\nConstructing electromagnetic transition operators E^lambda, M^lambda in the LHO basis ...")

    # Make standard 1-body electric transition operators ...
    TrE0, TrE1, TrE2, TrE3 = Tr1b_EX_initialize(Params,Orb)

    # Make 1-body electric vortical transition operators ...
    TrE1_VC, TrE1_VS, TrE1_TC, TrE1_TS, TrE1_sTC, TrE1_sTS, TrE1_C, TrE1_sC = Tr1b_E1_NLO_initialize(Params,Orb,chR2)

    # Make standard 1-body magnetic transition operators ...
    TrM1, TrM2, TrM3 = Tr1b_MX_initialize(Params,Orb)

    # Allocate 1-body electromagnetic transition operator ...
    TrOp = Tr1B(TrE0,TrE1,TrE2,TrE3,TrE1_VC,TrE1_VS,TrE1_TC,TrE1_TS,TrE1_sTC,TrE1_sTS,TrE1_C,TrE1_sC,TrM1,TrM2,TrM3)

    println("\tTransition 1-body electromagnetic operators in the LHO basis succesfully constructed ...")

    return TrOp
end

function Tr1b_EX_initialize(Params::Parameters,Orb::Vector{Orb1B})
    # Read parameters
    hw = Params.Calc.hw
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1)*(N_max+2),2)

    # Physical constant ...
    hc = 197.326980
    nmc2 = 939.565346
    pmc2 = 938.272013

    # Osciilator length b ...
    b_osc = sqrt(0.5 * (pmc2 + nmc2) * hw) / hc

    println("\tConstructing 1-body EX transition operators in the LHO basis ...")

    # Reduced EX matrix element ...
    function rEX(n_a::Int64,l_a::Int64,j_a::Int64,n_b::Int64,l_b::Int64,j_b::Int64,X::Int64,b_osc::Float64)
        if X == 0
            Amp = Float64((-1)^(X + div(j_a - 1,2))) * sqrt(Float64((j_a + 1) * (j_b + 1))) * fCG(j_a,j_b,2*X,1,-1,0) / sqrt(4.0 * pi)
            ME = Amp * radial_moment_LHO(X+2,n_a,l_a,n_b,l_b,b_osc)
            return ME
        else
            Amp = Float64((-1)^(X + div(j_a - 1,2))) * sqrt(Float64((j_a + 1) * (j_b + 1))) * fCG(j_a,j_b,2*X,1,-1,0) / sqrt(4.0 * pi)
            ME = Amp * radial_moment_LHO(X,n_a,l_a,n_b,l_b,b_osc)
            return ME
        end
    end

    # Initialize fields for EX transition operators ...
    pE0, nE0 = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)
    pE1, nE1 = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)
    pE2, nE2 = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)
    pE3, nE3 = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)

    # Allocate components of 1-body EX transition operators ...
    @inbounds Threads.@threads for a in 1:a_max
        n_a, l_a, j_a = Orb[a].n, Orb[a].l, Orb[a].j
        @inbounds for b in 1:a_max
            n_b, l_b, j_b = Orb[b].n, Orb[b].l, Orb[b].j

            pE0Sum, nE0Sum = 0.0, 0.0
            pE1Sum, nE1Sum = 0.0, 0.0
            pE2Sum, nE2Sum = 0.0, 0.0
            pE3Sum, nE3Sum = 0.0, 0.0

            # Case of E0 ...
            if (rem(l_a + l_b,2)) == 0
                rME = rEX(n_a,l_a,j_a,n_b,l_b,j_b,0,b_osc)
                pE0Sum += rME
                nE0Sum += rME
            end

            # Case of E1 ...
            if (rem(l_a + l_b + 1,2)) == 0
                rME = rEX(n_a,l_a,j_a,n_b,l_b,j_b,1,b_osc)
                pE1Sum += rME
                nE1Sum += rME
            end

            # Case of E2 ...
            if (rem(l_a + l_b + 2,2)) == 0
                rME = rEX(n_a,l_a,j_a,n_b,l_b,j_b,2,b_osc)
                pE2Sum += rME
                nE2Sum += rME
            end

            # Case of E3 ...
            if (rem(l_a + l_b + 3,2)) == 0
                rME = rEX(n_a,l_a,j_a,n_b,l_b,j_b,3,b_osc)
                pE3Sum += rME
                nE3Sum += rME
            end

            pE0[a,b], nE0[a,b] = pE0Sum, nE0Sum
            pE1[a,b], nE1[a,b] = pE1Sum, nE1Sum
            pE2[a,b], nE2[a,b] = pE2Sum, nE2Sum
            pE3[a,b], nE3[a,b] = pE3Sum, nE3Sum
        end
    end

    # Allocate the 1-body EX transition operators ...
    TrE0 = O1B(pE0,nE0)
    TrE1 = O1B(pE1,nE1)
    TrE2 = O1B(pE2,nE2)
    TrE3 = O1B(pE3,nE3)

    println("\tEX 1-body transition operators succesfully constructed in the LHO basis ...")

    return TrE0, TrE1, TrE2, TrE3
end

function Tr1b_E1_NLO_initialize(Params::Parameters,Orb::Vector{Orb1B},chR2::Float64)
    # Read parameters
    hw = Params.Calc.hw
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1)*(N_max+2),2)

    # Physical constant ...
    hc_pmc2 = 197.326980 / 938.272013
    hc_nmc2 = 197.326980 / 939.565346

    println("\tConstructing 1-body E1 vortical transition operators in the LHO basis ...")

    # Initialize the transitions operator matrix elements ...
    pE1_VC, nE1_VC = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)
    pE1_VS, nE1_VS = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)
    pE1_TC, nE1_TC = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)
    pE1_TS, nE1_TS = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)
    pE1_sTC, nE1_sTC = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)
    pE1_sTS, nE1_sTS = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)
    pE1_C, nE1_C = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)
    pE1_sC, nE1_sC = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)

    # Define local functions for evaluation of the reduced matrix elements ...
    @inline function rME_YL(l_a::Int64,j_a::Int64,l_b::Int64,j_b::Int64,L::Int64)
        if rem(l_a + l_b + L,2) != 0
            return 0.0
        end
        rME = phase(L + div(j_a - 1,2)) * sqrt(Float64((j_a + 1) * (j_b + 1)) / (4.0 * pi)) * fCG(j_a,j_b,2*L,1,-1,0)
        return rME
    end
    
    @inline function rME_rN(n_a::Int64,l_a::Int64,n_b::Int64,l_b::Int64,N::Int64,hw::Float64)
        b_osc = sqrt(0.5 * (939.565346 + 938.272013) * hw) / 197.326980
        rME = radial_moment_LHO(N,n_a,l_a,n_b,l_b,b_osc)
        return rME
    end

    @inline function rME_rN_dr(n_a::Int64,l_a::Int64,n_b::Int64,l_b::Int64,N::Int64,hw::Float64)
        b_osc = sqrt(0.5 * (939.565346 + 938.272013) * hw) / 197.326980
        rME = Float64(l_b) * radial_moment_LHO(N-1,n_a,l_a,n_b,l_b,b_osc) - b_osc^2 * radial_moment_LHO(N+1,n_a,l_a,n_b,l_b,b_osc)
        if 1 <= n_b
            rME -= 2.0 * b_osc * sqrt(Float64(n_b)) * radial_moment_LHO(N,n_a,l_a,n_b-1,l_b+1,b_osc)
        end
        return rME
    end

    @inline function rME_YL(l_a::Int64,l_b::Int64,L::Int64)
        rME = sqrt(Float64((2*l_b + 1) * (2*L + 1)) / (4.0 * pi)) * fCG(2*l_b,2*L,2*l_a,0,0,0)
        return rME
    end

    @inline function rME_nabla_dot_rN_YJL(a::Int64,b::Int64,N::Int64,L::Int64,J::Int64,hw::Float64,Orb::Vector{Orb1B})
        n_a, l_a, j_a = Orb[a].n, Orb[a].l, Orb[a].j
        n_b, l_b, j_b = Orb[b].n, Orb[b].l, Orb[b].j

        rME = 0.0

        Amp_Bra = phase(l_b + div(j_b + 1,2)) * f6j(2*l_b,1,j_b,j_a,2*J,2*l_a) *
                  sqrt(Float64((2*J + 1) * (2*L + 1) * (j_a + 1) * (j_b + 1) * (2*l_a + 1) * (2*l_b + 1)) / (4.0 * pi))

        Amp_Ket = phase(l_a + div(j_b + 1,2)) * f6j(2*l_a,1,j_a,j_b,2*J,2*l_b) *
                  sqrt(Float64((2*J + 1) * (2*L + 1) * (j_a + 1) * (j_b + 1) * (2*l_a + 1) * (2*l_b + 1)) / (4.0 * pi))

        if abs(L - l_b) <= (l_a - 1) && (l_a - 1) <= (L + l_b)
            ME = Amp_Bra * fCG(2*l_b,2*L,2*l_a-2,0,0,0) * f6j(2*L,2,2*J,2*l_a,2*l_b,2*l_a-2) * sqrt(Float64(l_a) / Float64(2*l_a + 1)) *
                (rME_rN_dr(n_b,l_b,n_a,l_a,N,hw) + Float64(l_a + 1) * rME_rN(n_a,l_a,n_b,l_b,N-1,hw))
            rME += ME
        end

        if abs(L - l_b) <= (l_a + 1) && (l_a + 1) <= (L + l_b) && l_a >= 1
            ME = Amp_Bra * fCG(2*l_b,2*L,2*l_a+2,0,0,0) * f6j(2*L,2,2*J,2*l_a,2*l_b,2*l_a+2) * sqrt(Float64(l_a - 1) / Float64(2*l_a + 1)) *
                (rME_rN_dr(n_b,l_b,n_a,l_a,N,hw) - Float64(l_a) * rME_rN(n_a,l_a,n_b,l_b,N-1,hw))
            rME -= ME
        end

        if abs(L - l_a) <= (l_b - 1) && (l_b - 1) <= (L + l_a)
            ME = Amp_Ket * fCG(2*l_a,2*L,2*l_b-2,0,0,0) * f6j(2*L,2,2*J,2*l_b,2*l_a,2*l_b-2) * sqrt(Float64(l_b) / Float64(2*l_b + 1)) *
                (rME_rN_dr(n_a,l_a,n_b,l_b,N,hw) + Float64(l_b + 1) * rME_rN(n_a,l_a,n_b,l_b,N-1,hw))
            rME += ME
        end

        if abs(L - l_a) <= (l_b + 1) && (l_b + 1) <= (L + l_a) && l_b >= 1
            ME = Amp_Ket * fCG(2*l_a,2*L,2*l_b+2,0,0,0) * f6j(2*L,2,2*J,2*l_b,2*l_a,2*l_b+2) * sqrt(Float64(l_b - 1) / Float64(2*l_b + 1)) *
                (rME_rN_dr(n_a,l_a,n_b,l_b,N,hw) - Float64(l_b) * rME_rN(n_a,l_a,n_b,l_b,N-1,hw))
            rME -= ME
        end

        return rME
    end

    @inline function rME_nabla_cross_S_dot_rN_YJL(a::Int64,b::Int64,N::Int64,L::Int64,J::Int64,hw::Float64,Orb::Vector{Orb1B})
        n_a, l_a, j_a = Orb[a].n, Orb[a].l, Orb[a].j
        n_b, l_b, j_b = Orb[b].n, Orb[b].l, Orb[b].j

        if abs(l_a - l_b) > J || J > (l_a + l_b)
            return 0.0
        end

        Amp = phase(l_a + l_b + J + div(j_b + 3,2)) * sqrt(Float64((j_a + 1) * (j_b + 1) * (2*l_a + 1) * (2*l_b + 1)) / (8.0 * pi * Float64(J * (J + 1)))) *
              Float64((l_a - l_b) * (l_a + l_b + 1) - div((j_a - j_b) * (j_a + j_b + 2),4)) * fCG(2*l_a,2*l_b,2*J,0,0,0) * f6j(2*l_a,j_a,1,j_b,2*l_b,2*J)

        if abs(Amp) < 1e-12
            return 0.0
        end

        rME = 0.0

        if L == J + 1
            ME = Amp * sqrt(Float64(J) / Float64(2*J + 1)) * (rME_rN_dr(n_a,l_a,n_b,l_b,N,hw) + rME_rN_dr(n_b,l_b,n_a,l_a,N,hw) - Float64(J) * rME_rN(n_a,l_a,n_b,l_b,N-1,hw))
            rME += ME
        end

        if L == J - 1
            ME = Amp * sqrt(Float64(J + 1) / Float64(2*J + 1)) * (rME_rN_dr(n_a,l_a,n_b,l_b,N,hw) + rME_rN_dr(n_b,l_b,n_a,l_a,N,hw) + Float64(J + 1) * rME_rN(n_a,l_a,n_b,l_b,N-1,hw))
            rME += ME
        end

        return rME
    end

    @inline function rME_rN_YL(a::Int64,b::Int64,N::Int64,L::Int64,hw::Float64,Orb::Vector{Orb1B})
        l_a, l_b = Orb[a].l, Orb[b].l
        if rem(l_a + l_b + L,2) == 0
            n_a, j_a = Orb[a].n, Orb[a].j
            n_b, j_b = Orb[b].n, Orb[b].j
            rY = phase(L + div(j_a - 1,2)) * sqrt(Float64((j_a + 1) * (j_b + 1)) / (4.0 * pi)) * fCG(j_a,j_b,2*L,1,-1,0)

            b_osc = sqrt(0.5 * (939.565346 + 938.272013) * hw) / 197.326980
            rN = radial_moment_LHO(N,n_a,l_a,n_b,l_b,b_osc)

            rME = rN * rY
            return rME
        else
            return 0.0
        end
    end

    # Allocate the 1-body E1 vortical transition reduced matrix elements ...
    @inbounds Threads.@threads for a in 1:a_max
        l_a, j_a = Orb[a].l, Orb[a].j
        @inbounds for b in 1:a_max
            l_b, j_b = Orb[b].l, Orb[b].j

            pVCSum, nVCSum = 0.0, 0.0
            pVSSum, nVSSum = 0.0, 0.0
            pTCSum, nTCSum = 0.0, 0.0
            pTSSum, nTSSum = 0.0, 0.0
            psTCSum, nsTCSum = 0.0, 0.0
            psTSSum, nsTSSum = 0.0, 0.0
            pCSum, nCSum = 0.0, 0.0
            psCSum, nsCSum = 0.0, 0.0

            # E1 selection rule ...
            if (rem(l_a + l_b + 1,2)) == 0 && abs(j_b - 2) <= j_a && j_a <= (j_b + 2)

                rG_dot_r2_Y10_ab = rME_nabla_dot_rN_YJL(a,b,2,0,1,hw,Orb)

                rG_dot_Y10_ab = rME_nabla_dot_rN_YJL(a,b,0,0,1,hw,Orb)
                
                rG_dot_r2_Y12_ab = rME_nabla_dot_rN_YJL(a,b,2,2,1,hw,Orb)

                rG_cross_S_dot_Y10_ab = rME_nabla_cross_S_dot_rN_YJL(a,b,0,0,1,hw,Orb)

                rG_cross_S_dot_r2_Y10_ab = rME_nabla_cross_S_dot_rN_YJL(a,b,2,0,1,hw,Orb)

                rG_cross_S_dot_r2_Y12_ab = rME_nabla_cross_S_dot_rN_YJL(a,b,2,2,1,hw,Orb)

                rr1_Y1_ab = rME_rN_YL(a,b,1,1,hw,Orb)
                rr3_Y1_ab = rME_rN_YL(a,b,3,1,hw,Orb)

                # Proton components ...
                pVCME = hc_pmc2 * sqrt(3.0 / 2.0) / 5.0 * rG_dot_r2_Y12_ab

                pVSME = hc_pmc2 * sqrt(3.0 / 2.0) / 5.0 * rG_cross_S_dot_r2_Y12_ab

                pTCME = hc_pmc2 * (sqrt(3.0 / 2.0) / 15.0 * rG_dot_r2_Y12_ab + sqrt(3.0) / 6.0 * rG_dot_r2_Y10_ab)

                pTSME = hc_pmc2 * (sqrt(3.0 / 2.0) / 15.0 * rG_cross_S_dot_r2_Y12_ab + sqrt(3.0) / 6.0 * rG_cross_S_dot_r2_Y10_ab)

                psTCME = -hc_pmc2 * sqrt(3.0) / 6.0 * chR2 * rG_dot_Y10_ab

                psTSME = -hc_pmc2 * sqrt(3.0) / 6.0 * chR2 * rG_cross_S_dot_Y10_ab
                
                pCME = 1.0 / 10.0 * rr3_Y1_ab

                psCME = -1.0 / 6.0 * chR2 *  rr1_Y1_ab

                pVCSum += pVCME
                pVSSum += pVSME
                pTCSum += pTCME
                pTSSum += pTSME
                psTCSum += psTCME
                psTSSum += psTSME
                pCSum += pCME
                psCSum += psCME

                # Neutron components ...
                nVCME = hc_nmc2 * sqrt(3.0 / 2.0) / 5.0 * rG_dot_r2_Y12_ab

                nVSME = hc_nmc2 * sqrt(3.0 / 2.0) / 5.0 * rG_cross_S_dot_r2_Y12_ab

                nTCME = hc_nmc2 * (sqrt(3.0 / 2.0) / 15.0 * rG_dot_r2_Y12_ab + sqrt(3.0) / 6.0 * rG_dot_r2_Y10_ab)

                nTSME = hc_nmc2 * (sqrt(3.0 / 2.0) / 15.0 * rG_cross_S_dot_r2_Y12_ab + sqrt(3.0) / 6.0 * rG_cross_S_dot_r2_Y10_ab)

                nsTCME = -hc_nmc2 * sqrt(3.0) / 6.0 * chR2 * rG_dot_Y10_ab

                nsTSME = -hc_nmc2 * sqrt(3.0) / 6.0 * chR2 * rG_cross_S_dot_Y10_ab

                nCME = 1.0 / 10.0 * rr3_Y1_ab

                nsCME = -1.0 / 6.0 * chR2 *  rr1_Y1_ab

                nVCSum += nVCME
                nVSSum += nVSME
                nTCSum += nTCME
                nTSSum += nTSME
                nsTCSum += nsTCME
                nsTSSum += nsTSME
                nCSum += nCME
                nsCSum += nsCME
            end

            pE1_VC[a,b] = pVCSum
            pE1_VS[a,b] = pVSSum
            pE1_TC[a,b] = pTCSum
            pE1_TS[a,b] = pTSSum
            pE1_sTC[a,b] = psTCSum
            pE1_sTS[a,b] = psTSSum
            pE1_C[a,b] = pCSum
            pE1_sC[a,b] = psCSum

            nE1_VC[a,b] = nVCSum
            nE1_VS[a,b] = nVSSum
            nE1_TC[a,b] = nTCSum
            nE1_TS[a,b] = nTSSum
            nE1_sTC[a,b] = nsTCSum
            nE1_sTS[a,b] = nsTSSum
            nE1_C[a,b] = nCSum
            nE1_sC[a,b] = nsCSum
        end
    end
    
    # Allocate the 1-body E1 vortical transition operator ...
    TrE1_VC = O1B(pE1_VC,nE1_VC)
    TrE1_VS = O1B(pE1_VS,nE1_VS)
    TrE1_TC = O1B(pE1_TC,nE1_TC)
    TrE1_TS = O1B(pE1_TS,nE1_TS)
    TrE1_sTC = O1B(pE1_sTC,nE1_sTC)
    TrE1_sTS = O1B(pE1_sTS,nE1_sTS)
    TrE1_C = O1B(pE1_C,nE1_C)
    TrE1_sC = O1B(pE1_sC,nE1_sC)

    println("\tE1 vortical 1-body transition operators succesfully constructed in the LHO basis ...")

    return TrE1_VC, TrE1_VS, TrE1_TC, TrE1_TS, TrE1_sTC, TrE1_sTS, TrE1_C, TrE1_sC
end

function Tr1b_MX_initialize(Params::Parameters,Orb::Vector{Orb1B})
    # Read parameters
    hw = Params.Calc.hw
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1)*(N_max+2),2)

    # Physical constant ...
    hc = 197.326980
    nmc2 = 939.565346
    pmc2 = 938.272013
    mu_N = 0.10515
    g_p = 5.586
    g_n = -3.826

    # Osciilator length b ...
    b_osc = sqrt(0.5 * (pmc2 + nmc2) * hw) / hc

    println("\tConstructing 1-body MX transition operators in the LHO basis ...")

    # Reduced MX matrix element ...
    function rMX(n_a::Int64,l_a::Int64,j_a::Int64,n_b::Int64,l_b::Int64,j_b::Int64,X::Int64,T::Int64,b_osc::Float64,mu_N::Float64,g_p::Float64,g_n::Float64)
        kappa = 0.5 * Float64((-1)^(div(j_a+1,2) + l_a) * (j_a + 1) + (-1)^(div(j_b+1,2) + l_b) * (j_b + 1))
        if T == 1
            Amp = mu_N * Float64((-1)^(X + div(j_a - 1,2))) * sqrt(Float64((j_a + 1) * (j_b + 1))) * fCG(j_a,j_b,2*X,1,-1,0) *
                    (X  - kappa) * (-0.5 * g_n) / sqrt(4.0 * pi)
            rME = Amp * radial_moment_LHO(X,n_a,l_a,n_b,l_b,b_osc)
            return rME
        elseif T == -1
            Amp = mu_N * Float64((-1)^(X + div(j_a - 1,2))) * sqrt(Float64((j_a + 1) * (j_b + 1))) * fCG(j_a,j_b,2*X,1,-1,0) *
                    (X  - kappa) * (1.0 + kappa / (X + 1.0) - 0.5 * g_p) / sqrt(4.0 * pi)
            rME = Amp * radial_moment_LHO(X,n_a,l_a,n_b,l_b,b_osc)
            return rME   
        end
    end

    # Initialize fields for MX transition operators ...
    pM1, nM1 = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)
    pM2, nM2 = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)
    pM3, nM3 = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)

    # Allocate components of 1-body MX transition operators ...
    @inbounds Threads.@threads for a in 1:a_max
        n_a, l_a, j_a = Orb[a].n, Orb[a].l, Orb[a].j
        @inbounds for b in 1:a_max
            n_b, l_b, j_b = Orb[b].n, Orb[b].l, Orb[b].j

            pM1Sum, nM1Sum = 0.0, 0.0
            pM2Sum, nM2Sum = 0.0, 0.0
            pM3Sum, nM3Sum = 0.0, 0.0

            # Case of M1 ...
            if rem(l_a + l_b + 2,2) == 0
                rpME = rMX(n_a,l_a,j_a,n_b,l_b,j_b,1,-1,b_osc,mu_N,g_p,g_n)
                rnME = rMX(n_a,l_a,j_a,n_b,l_b,j_b,1,1,b_osc,mu_N,g_p,g_n)
                pM1Sum += rpME
                nM1Sum += rnME
            end

            # Case of M2 ...
            if (rem(l_a + l_b + 3,2)) == 0
                rpME = rMX(n_a,l_a,j_a,n_b,l_b,j_b,2,-1,b_osc,mu_N,g_p,g_n)
                rnME = rMX(n_a,l_a,j_a,n_b,l_b,j_b,2,1,b_osc,mu_N,g_p,g_n)
                pM2Sum += rpME
                nM2Sum += rnME
            end

            # Case of M3 ...
            if (rem(l_a + l_b + 4,2)) == 0
                rpME = rMX(n_a,l_a,j_a,n_b,l_b,j_b,3,-1,b_osc,mu_N,g_p,g_n)
                rnME = rMX(n_a,l_a,j_a,n_b,l_b,j_b,3,1,b_osc,mu_N,g_p,g_n)
                pM3Sum += rpME
                nM3Sum += rnME
            end

            pM1[a,b], nM1[a,b] = pM1Sum, nM1Sum
            pM2[a,b], nM2[a,b] = pM2Sum, nM2Sum
            pM3[a,b], nM3[a,b] = pM3Sum, nM3Sum
        end
    end

    # Allocate the 1-body MX transition operators ...
    TrM1 = O1B(pM1,nM1)
    TrM2 = O1B(pM2,nM2)
    TrM3 = O1B(pM3,nM3)

    println("\tMX 1-body transition operators succesfully constructed in the LHO basis ...")

    return TrM1, TrM2, TrM3
end

function Tr1b_transformation(Params::Parameters,Orb::Vector{Orb1B},C::O1B,TrOp::Tr1B)
    # Parameter initialization...
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1)*(N_max + 2),2)

    println("\nTransforming 1-body transition operators from the reference basis to the given target basis ...")

    # Initialize arrays for 1-body transition operators in the target basis ...
    pE0, nE0 = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)
    pE1, nE1 = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)
    pE2, nE2 = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)
    pE3, nE3 = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)

    pE1_VC, nE1_VC = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)
    pE1_VS, nE1_VS = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)
    pE1_TC, nE1_TC = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)
    pE1_TS, nE1_TS = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)
    pE1_sTC, nE1_sTC = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)
    pE1_sTS, nE1_sTS = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)
    pE1_C, nE1_C = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)
    pE1_sC, nE1_sC = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)

    pM1, nM1 = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)
    pM2, nM2 = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)
    pM3, nM3 = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)

    # Auxiliary function precomputing all possible combinations of orbitals a & b ...
    function Tr1b_transformation_indices(Params::Parameters)
        # Read # initialize parameters ...
        N_max = Params.Calc.Nmax
        a_max = div((N_max + 1)*(N_max + 2),2)

        # Initialize the counter for combinations of ab orbitals ...
        ab_count = 0

        # Initialize the list of ab orbitals ...
        ab = Vector{Tuple{Int64,Int64}}(undef,a_max^2)

        # Allocate the list with combionations of ab orbitals ...
        @inbounds for a in 1:a_max
            @inbounds for b in 1:a_max
                ab_count += 1
                ab[ab_count] = (a,b)
            end
        end

        return ab, ab_count
    end

    # Precomputer all possible combinations of orbitals a & b ...
    ab, ab_count = Tr1b_transformation_indices(Params)

   # Perform the transformation from the LHO basis to the target basis ...
    @inbounds Threads.@threads for ab_ind in 1:ab_count
        (a,b) = ab[ab_ind]
        l_a, j_a = Orb[a].l, Orb[a].j
        l_b, j_b = Orb[b].l, Orb[b].j

        pE0Sum, nE0Sum = 0.0, 0.0
        pE1Sum, nE1Sum = 0.0, 0.0
        pE2Sum, nE2Sum = 0.0, 0.0
        pE3Sum, nE3Sum = 0.0, 0.0

        pE1VCSum, nE1VCSum = 0.0, 0.0
        pE1VSSum, nE1VSSum = 0.0, 0.0
        pE1TCSum, nE1TCSum = 0.0, 0.0
        pE1TSSum, nE1TSSum = 0.0, 0.0
        pE1sTCSum, nE1sTCSum = 0.0, 0.0
        pE1sTSSum, nE1sTSSum = 0.0, 0.0
        pE1CSum, nE1CSum = 0.0, 0.0
        pE1sCSum, nE1sCSum = 0.0, 0.0

        pM1Sum, nM1Sum = 0.0, 0.0
        pM2Sum, nM2Sum = 0.0, 0.0
        pM3Sum, nM3Sum = 0.0, 0.0

        @inbounds for k in 1:a_max
            l_k, j_k = Orb[k].l, Orb[k].j
            if l_a == l_k && j_a == j_k
                @inbounds for l in 1:a_max
                    l_l, j_l = Orb[l].l, Orb[l].j
                    if l_b == l_l && j_b == j_l

                        # Proton matrix elements ...
                        pC_ka, pC_lb = C.p[k,a], C.p[l,b]

                        pE0ME = TrOp.E0.p[k,l] * pC_ka * pC_lb
                        pE1ME = TrOp.E1.p[k,l] * pC_ka * pC_lb
                        pE2ME = TrOp.E2.p[k,l] * pC_ka * pC_lb
                        pE3ME = TrOp.E3.p[k,l] * pC_ka * pC_lb

                        pE1VCME = TrOp.E1_VC.p[k,l] * pC_ka * pC_lb
                        pE1VSME = TrOp.E1_VS.p[k,l] * pC_ka * pC_lb
                        pE1TCME = TrOp.E1_TC.p[k,l] * pC_ka * pC_lb
                        pE1TSME = TrOp.E1_TS.p[k,l] * pC_ka * pC_lb
                        pE1sTCME = TrOp.E1_sTC.p[k,l] * pC_ka * pC_lb
                        pE1sTSME = TrOp.E1_sTS.p[k,l] * pC_ka * pC_lb
                        pE1CME = TrOp.E1_C.p[k,l] * pC_ka * pC_lb
                        pE1sCME = TrOp.E1_sC.p[k,l] * pC_ka * pC_lb

                        pM1ME = TrOp.M1.p[k,l] * pC_ka * pC_lb
                        pM2ME = TrOp.M2.p[k,l] * pC_ka * pC_lb
                        pM3ME = TrOp.M3.p[k,l] * pC_ka * pC_lb

                        pE0Sum += pE0ME
                        pE1Sum += pE1ME
                        pE2Sum += pE2ME
                        pE3Sum += pE3ME

                        pE1VCSum += pE1VCME
                        pE1VSSum += pE1VSME
                        pE1TCSum += pE1TCME
                        pE1TSSum += pE1TSME
                        pE1sTCSum += pE1sTCME
                        pE1sTSSum += pE1sTSME
                        pE1CSum += pE1CME
                        pE1sCSum += pE1sCME

                        pM1Sum += pM1ME
                        pM2Sum += pM2ME
                        pM3Sum += pM3ME

                        # Neutron matrix elements ...
                        nC_ka, nC_lb = C.n[k,a], C.n[l,b]

                        nE0ME = TrOp.E0.n[k,l] * nC_ka * nC_lb
                        nE1ME = TrOp.E1.n[k,l] * nC_ka * nC_lb
                        nE2ME = TrOp.E2.n[k,l] * nC_ka * nC_lb
                        nE3ME = TrOp.E3.n[k,l] * nC_ka * nC_lb

                        nE1VCME = TrOp.E1_VC.n[k,l] * nC_ka * nC_lb
                        nE1VSME = TrOp.E1_VS.n[k,l] * nC_ka * nC_lb
                        nE1TCME = TrOp.E1_TC.n[k,l] * nC_ka * nC_lb
                        nE1TSME = TrOp.E1_TS.n[k,l] * nC_ka * nC_lb
                        nE1sTCME = TrOp.E1_sTC.n[k,l] * nC_ka * nC_lb
                        nE1sTSME = TrOp.E1_sTS.n[k,l] * nC_ka * nC_lb
                        nE1CME = TrOp.E1_C.n[k,l] * nC_ka * nC_lb
                        nE1sCME = TrOp.E1_sC.n[k,l] * nC_ka * nC_lb

                        nM1ME = TrOp.M1.n[k,l] * nC_ka * nC_lb
                        nM2ME = TrOp.M2.n[k,l] * nC_ka * nC_lb
                        nM3ME = TrOp.M3.n[k,l] * nC_ka * nC_lb

                        nE0Sum += nE0ME
                        nE1Sum += nE1ME
                        nE2Sum += nE2ME
                        nE3Sum += nE3ME

                        nE1VCSum += nE1VCME
                        nE1VSSum += nE1VSME
                        nE1TCSum += nE1TCME
                        nE1TSSum += nE1TSME
                        nE1sTCSum += nE1sTCME
                        nE1sTSSum += nE1sTSME
                        nE1CSum += nE1CME
                        nE1sCSum += nE1sCME

                        nM1Sum += nM1ME
                        nM2Sum += nM2ME
                        nM3Sum += nM3ME

                    end
                end
            end
        end

        pE0[a,b], nE0[a,b] = pE0Sum, nE0Sum
        pE1[a,b], nE1[a,b] = pE1Sum, nE1Sum
        pE2[a,b], nE2[a,b] = pE2Sum, nE2Sum
        pE3[a,b], nE3[a,b] = pE3Sum, nE3Sum

        pE1_VC[a,b], nE1_VC[a,b] = pE1VCSum, nE1VCSum
        pE1_VS[a,b], nE1_VS[a,b] = pE1VSSum, nE1VSSum
        pE1_TC[a,b], nE1_TC[a,b] = pE1TCSum, nE1TCSum
        pE1_TS[a,b], nE1_TS[a,b] = pE1TSSum, nE1TSSum
        pE1_sTC[a,b], nE1_sTC[a,b] = pE1sTCSum, nE1sTCSum
        pE1_sTS[a,b], nE1_sTS[a,b] = pE1sTSSum, nE1sTSSum
        pE1_C[a,b], nE1_C[a,b] = pE1CSum, nE1CSum
        pE1_sC[a,b], nE1_sC[a,b] = pE1sCSum, nE1sCSum

        pM1[a,b], nM1[a,b] = pM1Sum, nM1Sum
        pM2[a,b], nM2[a,b] = pM2Sum, nM2Sum
        pM3[a,b], nM3[a,b] = pM3Sum, nM3Sum

    end

    # Allocate components of the transformed 1-body transition operators ...
    TrE0_new = O1B(pE0,nE0)
    TrE1_new = O1B(pE1,nE1)
    TrE2_new = O1B(pE2,nE2)
    TrE3_new = O1B(pE3,nE3)

    TrE1_VC_new = O1B(pE1_VC,nE1_VC)
    TrE1_VS_new = O1B(pE1_VS,nE1_VS)
    TrE1_TC_new = O1B(pE1_TC,nE1_TC)
    TrE1_TS_new = O1B(pE1_TS,nE1_TS)
    TrE1_sTC_new = O1B(pE1_sTC,nE1_sTC)
    TrE1_sTS_new = O1B(pE1_sTS,nE1_sTS)
    TrE1_C_new = O1B(pE1_C,nE1_C)
    TrE1_sC_new = O1B(pE1_sC,nE1_sC)

    TrM1_new = O1B(pM1,nM1)
    TrM2_new = O1B(pM2,nM2)
    TrM3_new = O1B(pM3,nM3)

    # Allocate the transformed 1-body transition operator ... 
    TrOp_new = Tr1B(TrE0_new,TrE1_new,TrE2_new,TrE3_new,TrE1_VC_new,TrE1_VS_new,TrE1_TC_new,TrE1_TS_new,
                    TrE1_sTC_new,TrE1_sTS_new,TrE1_C_new,TrE1_sC_new,TrM1_new,TrM2_new,TrM3_new)

    println("\tTransition 1-body operators transformed to the given target basis ...")
    
    return TrOp_new
end

function qpTr1b_initialize(Params::Parameters,Orb::Vector{Orb1B},chR2::Float64,C::O1B,U::O1B,V::O1B)
    println("\nPreparing 1-body electromagnetic transition operators in quasiparticle representation ...")
    
    # Make the 1-body electromagnetic transitions operator in particle representation ...
    TrOp = Tr1b_initialize(Params,Orb,chR2)

    # Transform 1-body electromagnetic transitions operator in particle representation to the target particle basis ...
    TrOp = Tr1b_transformation(Params,Orb,C,TrOp)

    # Evluate the 1-body electromagnetic transitions operator in quasiparticle representation ...
        # Case of standard EX electric transition operators ...
    qpTrE0 = qpTr1b_EX_initialize(Params,Orb,U,V,TrOp.E0,0)
    qpTrE1 = qpTr1b_EX_initialize(Params,Orb,U,V,TrOp.E1,1)
    qpTrE2 = qpTr1b_EX_initialize(Params,Orb,U,V,TrOp.E2,2)
    qpTrE3 = qpTr1b_EX_initialize(Params,Orb,U,V,TrOp.E3,3)
        # Case of toroidal E1 electric transition operators ...
    qpTrE1_VC = qpTr1b_EX_initialize(Params,Orb,U,V,TrOp.E1_VC,1)
    qpTrE1_VS = qpTr1b_EX_initialize(Params,Orb,U,V,TrOp.E1_VS,1)
    qpTrE1_TC = qpTr1b_EX_initialize(Params,Orb,U,V,TrOp.E1_TC,1)
    qpTrE1_TS = qpTr1b_EX_initialize(Params,Orb,U,V,TrOp.E1_TS,1)
    qpTrE1_C = qpTr1b_EX_initialize(Params,Orb,U,V,TrOp.E1_C,1)
    qpTrE1_sTC = qpTr1b_EX_initialize(Params,Orb,U,V,TrOp.E1_sTC,1)
    qpTrE1_sTS = qpTr1b_EX_initialize(Params,Orb,U,V,TrOp.E1_sTS,1)
    qpTrE1_sC = qpTr1b_EX_initialize(Params,Orb,U,V,TrOp.E1_sC,1)
        # Case of standard MX magnetic transition operators ...
    qpTrM1 = qpTr1b_MX_initialize(Params,Orb,U,V,TrOp.M1,1)
    qpTrM2 = qpTr1b_MX_initialize(Params,Orb,U,V,TrOp.M2,2)
    qpTrM3 = qpTr1b_MX_initialize(Params,Orb,U,V,TrOp.M3,3)

    # Allocate the full form of 1-body transition operator in quasiparticle representation ...
    qpTrOp = qpTr1B(qpTrE0,qpTrE1,qpTrE2,qpTrE3,qpTrE1_VC,qpTrE1_VS,qpTrE1_TC,qpTrE1_TS,
                    qpTrE1_sTC,qpTrE1_sTS,qpTrE1_C,qpTrE1_sC,qpTrM1,qpTrM2,qpTrM3)

    println("\tElectromagnetic 1-body transition operators succesfully evaluated in the quasiparticle representation ...")

    return qpTrOp
end

function qpTr1b_EX_initialize(Params::Parameters,Orb::Vector{Orb1B},U::O1B,V::O1B,TrOp_EX::O1B,X::Int64)
    # Read parameters
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1)*(N_max+2),2)

     # Initialize the quasiparticle representation of EX transition operator ...
    pEX_11, nEX_11 = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)
    pEX_20, nEX_20 = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)

    # Calculate the quasiparticle representation of the EX transition operator ...
    @inbounds Threads.@threads for a in 1:a_max
        l_a , j_a = Orb[a].l, Orb[a].j
        j_a_hat = sqrt(Float64(j_a + 1))
        @inbounds for b in 1:a_max
            l_b, j_b = Orb[b].l, Orb[b].j
            if (div(abs(j_a - j_b),2) <= X) && (X <= div(j_a + j_b,2)) && (rem(l_a + l_b + X,2) == 0)
                pEX_11Sum, nEX_11Sum = 0.0, 0.0
                pEX_20Sum, nEX_20Sum = 0.0, 0.0
                @inbounds for k in 1:a_max
                    l_k, j_k = Orb[k].l, Orb[k].j
                    if l_a == l_k && j_a == j_k
                        @inbounds for l in 1:a_max
                            l_l, j_l = Orb[l].l, Orb[l].j
                            if l_l == l_b && j_l == j_b
                                # Determine phase of qp11 & qp20 components ...
                                Phase_11 = Float64((-1)^(div(j_a + j_b,2) + X))
                                Phase_20 = Float64((-1)^(X))

                                # Read the quasiparticle amplitudes ...
                                pU_ka, pV_ka = U.p[k,a], V.p[k,a]
                                pU_lb, pV_lb = U.p[l,b], V.p[l,b]
                                nU_ka, nV_ka = U.n[k,a], V.n[k,a]
                                nU_lb, nV_lb = U.n[l,b], V.n[l,b]

                                # Read the transition operator matrix element ...
                                pTrOp_EX_kl = TrOp_EX.p[k,l]
                                nTrOp_EX_kl = TrOp_EX.n[k,l]

                                # Special case of E0 ...
                                if X == 0
                                    # 11 components ...
                                    pEX_11ME = pTrOp_EX_kl * (pU_ka * pU_lb - Phase_11 * pV_ka * pV_lb + j_a_hat * kronecker_delta(a,b) * pV_ka * pV_lb)
                                    nEX_11ME = nTrOp_EX_kl * (nU_ka * nU_lb - Phase_11 * nV_ka * nV_lb + j_a_hat * kronecker_delta(a,b) * nV_ka * nV_lb)

                                    # 20 components ...
                                    pEX_20ME = pTrOp_EX_kl * (pV_ka * pU_lb + pU_ka * pV_lb)
                                    nEX_20ME = nTrOp_EX_kl * (nV_ka * nU_lb + nU_ka * nV_lb)

                                    pEX_11Sum += pEX_11ME
                                    nEX_11Sum += nEX_11ME
                                    pEX_20Sum += pEX_20ME
                                    nEX_20Sum += nEX_20ME

                                # General case of EX, X > 0 ...
                                elseif X > 0
                                    # 11 components ...
                                    pEX_11ME = pTrOp_EX_kl * (pU_ka * pU_lb - Phase_11 * pV_ka * pV_lb)
                                    nEX_11ME = nTrOp_EX_kl * (nU_ka * nU_lb - Phase_11 * nV_ka * nV_lb)

                                    # 20 components ...
                                    pEX_20ME = pTrOp_EX_kl * (pV_ka * pU_lb + Phase_20 * pU_ka * pV_lb)
                                    nEX_20ME = nTrOp_EX_kl * (nV_ka * nU_lb + Phase_20 * nU_ka * nV_lb)

                                    pEX_11Sum += pEX_11ME
                                    nEX_11Sum += nEX_11ME
                                    pEX_20Sum += pEX_20ME
                                    nEX_20Sum += nEX_20ME
                                end

                            end
                        end
                    end
                end
                pEX_11[a,b], nEX_11[a,b] = pEX_11Sum, nEX_11Sum
                pEX_20[a,b], nEX_20[a,b] = pEX_20Sum, nEX_20Sum
            end
        end
    end

    # Allocate the EX transition operator in quasiparticle representation ...
    qpTrEX = qpO1B(O1B(pEX_11,nEX_11),O1B(pEX_20,nEX_20))

    return qpTrEX
end

function qpTr1b_MX_initialize(Params::Parameters,Orb::Vector{Orb1B},U::O1B,V::O1B,TrOp_MX::O1B,X::Int64)
    # Read parameters
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1)*(N_max+2),2)

    # Initialize the quasiparticle representation of MX transition operator ...
    pMX_11, nMX_11 = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)
    pMX_20, nMX_20 = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)

    # Calculate the quasiparticle representation of the MX transition operator ...
    @inbounds Threads.@threads for a in 1:a_max
        l_a , j_a = Orb[a].l, Orb[a].j
        @inbounds for b in 1:a_max
            l_b, j_b = Orb[b].l, Orb[b].j
            if (div(abs(j_a - j_b),2) <= X) && (X <= div(j_a + j_b,2)) && (rem(l_a + l_b + X,2) == 1)
                pMX_11Sum, nMX_11Sum = 0.0, 0.0
                pMX_20Sum, nMX_20Sum = 0.0, 0.0
                @inbounds for k in 1:a_max
                    l_k, j_k = Orb[k].l, Orb[k].j
                    if l_a == l_k && j_a == j_k
                        @inbounds for l in 1:a_max
                            l_l, j_l = Orb[l].l, Orb[l].j
                            if l_l == l_b && j_l == j_b
                                # Determine phase of qp11 & qp20 components ...
                                Phase_11 = Float64((-1)^(div(j_a + j_b,2) + X))
                                Phase_20 = Float64((-1)^(X))

                                # Read the quasiparticle amplitudes ...
                                pU_ka, pV_ka = U.p[k,a], V.p[k,a]
                                pU_lb, pV_lb = U.p[l,b], V.p[l,b]
                                nU_ka, nV_ka = U.n[k,a], V.n[k,a]
                                nU_lb, nV_lb = U.n[l,b], V.n[l,b]

                                # Read the transition operator matrix element ...
                                pTrOp_MX_kl = TrOp_MX.p[k,l]
                                nTrOp_MX_kl = TrOp_MX.n[k,l]

                                # 11 components ...
                                pMX_11ME = pTrOp_MX_kl * (pU_ka * pU_lb - Phase_11 * pV_ka * pV_lb)
                                nMX_11ME = nTrOp_MX_kl * (nU_ka * nU_lb - Phase_11 * nV_ka * nV_lb)

                                # 20 components ...
                                pMX_20ME = pTrOp_MX_kl * (pV_ka * pU_lb - Phase_20 * pU_ka * pV_lb)
                                nMX_20ME = nTrOp_MX_kl * (nV_ka * nU_lb - Phase_20 * nU_ka * nV_lb)

                                pMX_11Sum += pMX_11ME
                                nMX_11Sum += nMX_11ME
                                pMX_20Sum += pMX_20ME
                                nMX_20Sum += nMX_20ME
                            end
                        end
                    end
                end
                pMX_11[a,b], nMX_11[a,b] = pMX_11Sum, nMX_11Sum
                pMX_20[a,b], nMX_20[a,b] = pMX_20Sum, nMX_20Sum
            end
        end
    end

    # Allocate the MX transition operator in quasiparticle representation ...
    qpTrMX = qpO1B(O1B(pMX_11,nMX_11),O1B(pMX_20,nMX_20))

    return qpTrMX
end