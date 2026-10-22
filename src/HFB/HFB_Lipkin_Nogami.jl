function HFB_Lipkin_Nogami(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,Orb_NNN::Orb3B,Rho::O1B,Kappa::O1B,H::O1B,Delta::O1B,V_NN::O2B,V_NNN::Array{Vector{Vector{Float32}},4})
    # Calculate the value of Lambda_2 ...
    Lambda_2 = HFB_Lipkin_Nogami_Lambda_2(Params,Orb,Orb_NN,Orb_NNN,Rho,Kappa,V_NN,V_NNN)

    # Export the value of Lambda_2 ...
    HFB_Lipkin_Nogami_Lambda_2_export(Params,Lambda_2)

    # Modify the fields H & Delta ...
    H, Delta = HFB_Lipkin_Nogami_allocate(Params,Lambda_2,Rho,Kappa,H,Delta)

    # Return modified fields H & Delta ...
    return H, Delta
end

function HFB_Lipkin_Nogami_allocate_indices(Params::Parameters,Orb::Vector{Orb1B})
    # Read # initialize parameters ...
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1)*(N_max + 2),2)
    ab_count, de_count = 0, 0

    # Count how many ab pairs are there ...
    @inbounds for a in 1:a_max
        l_a = Orb[a].l
        j_a = Orb[a].j
        @inbounds for b in 1:a_max
            l_b = Orb[b].l
            j_b = Orb[b].j
            if j_a == j_b && l_a == l_b
                ab_count += 1
            end
        end
    end

    # Count how many de pairs are there ...
    @inbounds for d in 1:a_max
        l_d = Orb[d].l
        j_d = Orb[d].j
        @inbounds for e in 1:a_max
            l_e = Orb[e].l
            j_e = Orb[e].j
            de_count += 1
        end
    end

    # Initialite the array ab ...
    ab, ab_count = zeros(Int64,2,ab_count), 0
    de, de_count = zeros(Int64,2,de_count), 0

    # Allocate the array ab ...
    @inbounds for a in 1:a_max
        l_a = Orb[a].l
        j_a = Orb[a].j
        @inbounds for b in 1:a_max
            l_b = Orb[b].l
            j_b = Orb[b].j
            if j_a == j_b && l_a == l_b
                ab_count += 1
                ab[1,ab_count] = a
                ab[2,ab_count] = b
            end
        end
    end

    # Allocate the array de ...
    @inbounds for d in 1:a_max
        l_d = Orb[d].l
        j_d = Orb[d].j
        @inbounds for e in 1:a_max
            l_e = Orb[e].l
            j_e = Orb[e].j
            de_count += 1
            de[1,de_count] = d
            de[2,de_count] = e
        end
    end

    return ab, ab_count, de, de_count
end

function HFB_Lipkin_Nogami_Lambda_2(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,Orb_NNN::Orb3B,Rho::O1B,Kappa::O1B,V_NN::O2B,V_NNN::Array{Vector{Vector{Float32}},4})
    # Read parameters ...
    N_max, N_2max, N_3max = Params.Calc.Nmax, Params.Calc.N2max, Params.Calc.N3max
    a_max = div((N_max + 1)*(N_max + 2),2)
    p2, p3 = Params.Int.cP2N, Params.Int.cP3N
    Tol = Params.Calc.HFB.Tol

    # Initialize the LNT coefficient Lambda_2 ...
    pLambda_2, nLambda_2 = 0.0, 0.0

    # Initialize temporary fields H & Delta & densities Rho & Kappa ...
    pRho_bar, pKappa_bar = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)
    nRho_bar, nKappa_bar = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)

    # Allocate the temporary densities Rho & Kappa ...
    @inbounds Threads.@threads for a in 1:a_max
        l_a ,j_a = Orb[a].l, Orb[a].j
        @inbounds for b in 1:a_max
            l_b ,j_b = Orb[b].l, Orb[b].j
            if l_a == l_b && j_a == j_b
                pRhoSum, pKappaSum = 0.0, 0.0
                nRhoSum, nKappaSum = 0.0, 0.0
                @inbounds for c in 1:a_max
                    l_c ,j_c = Orb[c].l, Orb[c].j
                    if l_a == l_c && j_a == j_c
                        pRhoSum -= Rho.p[a,c] * Rho.p[c,b]
                        nRhoSum -= Rho.n[a,c] * Rho.n[c,b]

                        pKappaSum += Rho.p[a,c] * Kappa.p[c,b]
                        nKappaSum += Rho.n[a,c] * Kappa.n[c,b]
                    end
                end
                pRhoSum += Rho.p[a,b]
                nRhoSum += Rho.n[a,b]

                pRho_bar[a,b] = pRhoSum
                nRho_bar[a,b] = nRhoSum
                pKappa_bar[a,b] = pKappaSum
                nKappa_bar[a,b] = nKappaSum
            end
        end
    end

    # Allocate indices for allocation of H & Delta fields ... same as for HFB ...
    ad, ad_count, be, be_count = HFB_Lipkin_Nogami_allocate_indices(Params,Orb)

    pTrace1_threads = zeros(Float64,Threads.maxthreadid())
    pTrace2_threads = zeros(Float64,Threads.maxthreadid())
    nTrace1_threads = zeros(Float64,Threads.maxthreadid())
    nTrace2_threads = zeros(Float64,Threads.maxthreadid())

    # Evaluate the traces over H & Delta ...
    @inbounds Threads.@threads :static for ad_i in 1:ad_count
        pTr1Sum, pTr2Sum = 0.0, 0.0
        nTr1Sum, nTr2Sum = 0.0, 0.0
        Tid = Threads.threadid()

        a, d = ad[1,ad_i], ad[2,ad_i]
        n_a, l_a, j_a = Orb[a].n, Orb[a].l, Orb[a].j
        n_d, l_d, j_d = Orb[d].n, Orb[d].l, Orb[d].j

        j_a_hat = sqrt(Float64(j_a + 1))

        pRho_ad, nRho_ad = pRho_bar[a,d], nRho_bar[a,d]

        # Single-particle field H ..
        @inbounds for be_i in 1:be_count
            b, e = be[1,be_i], be[2,be_i]
            n_b, l_b, j_b = Orb[b].n, Orb[b].l, Orb[b].j
            n_e, l_e, j_e = Orb[e].n, Orb[e].l, Orb[e].j
            P_ab = rem(l_a + l_b,2) + 1

            if (2*(n_a + n_b) + l_a + l_b) <= N_2max && a == d && b == e

                if l_b == l_e && j_b == j_e && (2*(n_d + n_e) + l_d + l_e) <= N_2max
                    pRho_be, nRho_be = pRho_bar[b,e], nRho_bar[b,e]

                    @inbounds for J = div(abs(j_d - j_e),2):div(j_d + j_e,2)
                        
                        # 2-body NN interaction part ...
                        if rem(l_a + l_b, 2) == rem(l_d + l_e, 2)
                            J_hat = Float64(2*J + 1)
                            pTr1Sum += J_hat * O2b_pp(a,b,d,e,J,P_ab,V_NN,Orb,Orb_NN) * pRho_ad * pRho_be
                            nTr1Sum += J_hat * O2b_nn(a,b,d,e,J,P_ab,V_NN,Orb,Orb_NN) * nRho_ad * nRho_be
                        end

                        # 3-body NNN interaction part ...
                        @inbounds for c in 1:a_max
                            n_c = Orb[c].n
                            l_c = Orb[c].l
                            if (2*(n_a + n_b + n_c) + l_a + l_b + l_c) <= N_3max
                                P_abc = rem(l_a + l_b + l_c, 2) + 1
                                j_c = Orb[c].j
                                @inbounds for f in 1:a_max
                                    n_f = Orb[f].n
                                    l_f = Orb[f].l
                                    if (2*(n_d + n_e + n_f) + l_d + l_e + l_f) <= N_3max && l_c == l_f && P_abc == (rem(l_d + l_e + l_f,2) + 1)
                                        j_f = Orb[f].j
                                        if j_c == j_f
                                            pRho_cf, nRho_cf = Rho.p[c,f], Rho.n[c,f]

                                            ME111 = V3b_no2b(a,b,c,1,d,e,f,1,J,1,P_abc,V_NNN,Orb,Orb_NNN)
                                            ME113 = V3b_no2b(a,b,c,1,d,e,f,1,J,3,P_abc,V_NNN,Orb,Orb_NNN)

                                            pME = (ME113 * pRho_cf + (2.0 * ME111 + ME113) / 3.0 * nRho_cf) * pRho_ad * pRho_be
                                            nME = (ME113 * nRho_cf + (2.0 * ME111 + ME113) / 3.0 * pRho_cf) * nRho_ad * nRho_be

                                            pTr1Sum += pME
                                            nTr1Sum += nME
                                        end
                                    end
                                end
                            end
                        end

                    end
                end

            end

        end

        # Single-quasiparticle pairing field Delta ...
        if ((2*(n_a + n_d) + l_a + l_d) <= N_2max)
            pKappa_ad, nKappa_ad = Kappa.p[a,d] - pKappa_bar[a,d], Kappa.n[a,d] - nKappa_bar[a,d]

            @inbounds for be_i in 1:be_count
                b, e = be[1,be_i], be[2,be_i]
                n_b, l_b, j_b = Orb[b].n, Orb[b].l, Orb[b].j
                n_e, l_e, j_e = Orb[e].n, Orb[e].l, Orb[e].j

                j_b_hat = sqrt(Float64(j_b + 1))
                Amp = j_b_hat * j_a_hat

                if (2*(n_b + n_e) + l_b + l_e) <= N_2max && (l_b == l_e) && (j_b == j_e) && a == d && b == e
                    pKappa_be, nKappa_be = pKappa_bar[b,e], nKappa_bar[b,e]

                    # 2-body NN interaction part ...
                    pTr2Sum += p2 * Amp * pKappa_ad * pKappa_be * O2b_pp(a,d,b,e,0,1,V_NN,Orb,Orb_NN)
                    nTr2Sum += p2 * Amp * nKappa_ad * nKappa_be * O2b_nn(a,d,b,e,0,1,V_NN,Orb,Orb_NN)

                    # 3-body NNN interaction part ...
                    @inbounds for c in 1:a_max
                        n_c, l_c = Orb[c].n, Orb[c].l
                        if (2*(n_a + n_d + n_c) + l_a + l_d + l_c) <= N_3max
                            P_adc = rem(l_a + l_d + l_c, 2) + 1
                            j_c = Orb[c].j
                            @inbounds for f in 1:a_max
                                n_f = Orb[f].n
                                l_f = Orb[f].l
                                if (2*(n_b + n_e + n_f) + l_b + l_e + l_f) <= N_3max && l_c == l_f && P_adc == (rem(l_b + l_e + l_f,2) + 1)
                                    j_f = Orb[f].j
                                    if j_c == j_f
                                        pRho_cf, nRho_cf = Rho.p[c,f], Rho.n[c,f]

                                        ME111 = V3b_no2b(a,d,c,1,b,e,f,1,0,1,P_adc,V_NNN,Orb,Orb_NNN)
                                        ME113 = V3b_no2b(a,d,c,1,b,e,f,1,0,3,P_adc,V_NNN,Orb,Orb_NNN)

                                        pTr2Sum += p3 * Amp * (ME113 * pRho_cf + (2.0 * ME111 + ME113) / 3.0 * nRho_cf) * pKappa_ad * pKappa_be
                                        nTr2Sum += p3 * Amp * (ME113 * nRho_cf + (2.0 * ME111 + ME113) / 3.0 * pRho_cf) * nKappa_ad * nKappa_be
                                    end
                                end
                            end
                        end
                    end

                end

            end

        end

        pTrace1_threads[Tid] += pTr1Sum
        pTrace2_threads[Tid] -= pTr2Sum
        nTrace1_threads[Tid] += nTr1Sum
        nTrace2_threads[Tid] -= nTr2Sum
    end

    # Evaluate the LN coefficients Lambda_2 ...
        # Evaluate the needed matrix products ...
    pMatrix1 = pRho_bar
    pMatrix2 = pRho_bar^2

    nMatrix1 = nRho_bar
    nMatrix2 = nRho_bar^2

        # Initialize the counters for Lambda_2 formula ...
    pDenominator, pNominator = 0.0, sum(pTrace1_threads) + sum(pTrace2_threads)
    nDenominator, nNominator = 0.0, sum(nTrace1_threads) + sum(nTrace2_threads)
    pTraceSquared, nTraceSquared = 0.0, 0.0

        # Evaluate the trace expressions ...
    @inbounds for a in 1:a_max
        j_a_hat = Float64(Orb[a].j + 1)

        pDenominator -= 4.0 * j_a_hat * pMatrix2[a,a]
        pTraceSquared += j_a_hat * pMatrix1[a,a]

        nDenominator -= 4.0 * j_a_hat * nMatrix2[a,a]
        nTraceSquared += j_a_hat * nMatrix1[a,a]
    end

    pDenominator = pDenominator + 2.0 * pTraceSquared^2
    nDenominator = nDenominator + 2.0 * nTraceSquared^2

        # Calculate the value of Lambda_2 ...
    if abs(pDenominator) > max(Tol,1e-4)
        pLambda_2 = pNominator / pDenominator
    end

    if abs(nDenominator) > max(Tol,1e-4)
        nLambda_2 = nNominator / nDenominator
    end

    return pnFloat(pLambda_2,nLambda_2)
end

function HFB_Lipkin_Nogami_Lambda_2_export(Params::Parameters,Lambda_2::pnFloat)
    # Read parameters
    Output_File = Params.Calc.Path

    # Determine the export path for Lambda_2 ...
    Lambda_2_Export_Path = "io/" * Output_File * "/HFB/bin/lambda_2.bin"

    # Export the LNT coefficient Lambda_2 ...
    open(Lambda_2_Export_Path, "w") do Export_File
        # Write the value of pLambda_2 ...
        write(Export_File,Float64(Lambda_2.p))
        # Write the value of nLambda_2 ...
        write(Export_File,Float64(Lambda_2.n))
    end

    return
end

function HFB_Lipkin_Nogami_Lambda_2_import(Params::Parameters)
    # Read parameters ...
    Output_File = Params.Calc.Path

    # Initialize the values of Lambda_2 ...
    pLambda_2, nLambda_2 = 0.0, 0.0

    # Determine the import path for Lambda_2 ...
    Lambda_2_Import_Path = "io/" * Output_File * "/HFB/bin/lambda_2.bin"

    # Import the LNT coefficient Lambda_2 ...
    open(Lambda_2_Import_Path, "r") do Import_File
        # Read the values of Lambda_2 ...
        pLambda_2 = read(Import_File,Float64)
        nLambda_2 = read(Import_File,Float64)
    end

    return pnFloat(pLambda_2,nLambda_2)
end

function HFB_Lipkin_Nogami_allocate(Params::Parameters,Lambda_2::pnFloat,Rho::O1B,Kappa::O1B,H::O1B,Delta::O1B)
    # Read parameters ...
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1)*(N_max + 2),2)
    
    # Initialize temporary fields H & Delta ...
    I = diagm(ones(Float64,a_max))

    # Include the LN correction of H & Delta fields ...
        # Proton fields ...
    pH = H.p .- 2.0 * Lambda_2.p .* (I .- 2.0 .* Rho.p)
    pDelta = Delta.p #.- 2.0 * Lambda_2.p .* Kappa.p
        # Neutron fields ...
    nH = H.n .- 2.0 * Lambda_2.n .* (I .- 2.0 .* Rho.n)
    nDelta = Delta.n #.- 2.0 * Lambda_2.n .* Kappa.n
    
    return O1B(pH,nH), O1B(pDelta,nDelta)
end

function HFB_Lipkin_Nogami_energy(Params::Parameters,Orb::Vector{Orb1B},Lambda_2::pnFloat,Rho::O1B)
    # Read parameters ...
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1)*(N_max + 2),2)

    # Initialize the LN energy ...
    E_LN = 0.0

    # Calculate the square of OBDM ...
    pRho_squared = Rho.p^2
    nRho_squared = Rho.n^2

    # Evaluate the contribution of the LN correction to the energy ...
    @inbounds for a in 1:a_max
        j_a_hat = Float64(Orb[a].j + 1)
        E_LN -= 2.0 * j_a_hat * Lambda_2.p * (Rho.p[a,a] - pRho_squared[a,a])
        E_LN -= 2.0 * j_a_hat * Lambda_2.n * (Rho.n[a,a] - nRho_squared[a,a])
    end

    return E_LN
end

function HFB_Lipkin_Nogami_V2b_residual_no2b(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,V_NN::O2B)
    # Read parameters ...
    N_2max = Params.Calc.N2max
    cRes = Params.Int.cRes

    # Initialize list of J & P values ...
    JP_list = JP_initialize(N_2max + 1)

    # Import the LN coefficient Lambda_2 ...
    Lambda_2 = HFB_Lipkin_Nogami_Lambda_2_import(Params)

    println("\nAdding the LN N^2 correction to the residual NN interaction ...\n")

    @inbounds for JP in JP_list
        J, P = JP[1], JP[2]
        N_T1 = Orb_NN.N[2,P,J+1]
        @inbounds Threads.@threads for Bra in 1:N_T1

            # proton-proton & neutron-neutron contributions only ...
            a, b = Orb_NN.Ind[2,P,J+1][Bra][1], Orb_NN.Ind[2,P,J+1][Bra][2]
            @inbounds for Ket in 1:Bra
                Ind = Bra + (Ket - 1) * N_T1 - div(Ket * (Ket - 1),2)
                c, d = Orb_NN.Ind[2,P,J+1][Ket][1], Orb_NN.Ind[2,P,J+1][Ket][2]
                j_c, j_d = Orb[c].j, Orb[d].j
                Phase = phase(div(j_c + j_d,2) - J)
                ME = -2.0 * cRes * Float64(kronecker_delta(a,c) * kronecker_delta(b,d) - Phase * kronecker_delta(a,d) * kronecker_delta(b,c)) / Float64(1 + kronecker_delta(a,b))
                pME = Lambda_2.p * ME
                nME = Lambda_2.n * ME

                @inbounds V_NN.pp[P,J+1][Ind] += pME
                @inbounds V_NN.nn[P,J+1][Ind] += nME
            end
    
        end
    end

    return V_NN
end