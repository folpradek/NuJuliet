function BCS_V2b(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,Orb_NNN::Orb3B,V_NN::O2B,V_NNN::Array{Vector{Vector{Float32}},4},C::O1B,Rho::O1B)
    # Read parameters ...
    N_max = Params.Calc.Nmax
    N_2max = Params.Calc.N2max
    N_3max = Params.Calc.N3max
    a_max = div((N_max + 1)*(N_max + 2),2)
    J_max = 0

    # Read the pairing rescaling coefficients p2 & p3 ...
    p2, p3 = Params.Int.cP2N, Params.Int.cP3N

    # Initialize the array of values J & P ...
    JP = JP_initialize(J_max)

    # Prepare array for residual interaction ...
    V_NN_Res = deepcopy(V_NN)

    println("\nStarting calculation of residual NN interaction...\n")

    # Include NO2B NNN interaction to NN component ...
    println("\nMaking density dependent residual 2-body interaction...")

    @time @inbounds for i in JP
        J, P = i[1], i[2]
        if P == 1
            println("Calculating   ...   J = " * string(J) * "/" * string(N_2max+1) * "\tP = +")
        else
            println("Calculating   ...   J = " * string(J) * "/" * string(N_2max+1) * "\tP = -")
        end
        @views N_T0 = Orb_NN.N[1,P,J+1]
        @views N_T1 = Orb_NN.N[2,P,J+1]
        
        @inbounds Threads.@threads for Bra in 1:max(N_T0, N_T1)

            if Bra <= N_T0
                @views a = Orb_NN.Ind[1,P,J+1][Bra][1]
                @views b = Orb_NN.Ind[1,P,J+1][Bra][2]
                l_a = Orb[a].l
                n_a = Orb[a].n
                l_b = Orb[b].l
                n_b = Orb[b].n
                for Ket in 1:Bra
                    Ind = Bra + (Ket - 1) * N_T0 - div(Ket * (Ket - 1),2)
                    @views d = Orb_NN.Ind[1,P,J+1][Ket][1]
                    @views e = Orb_NN.Ind[1,P,J+1][Ket][2]
                    l_d = Orb[d].l
                    n_d = Orb[d].n
                    l_e = Orb[e].l
                    n_e = Orb[e].n
                    pnSum = 0.0
                    @inbounds for c in 1:a_max
                        n_c = Orb[c].n
                        l_c = Orb[c].l
                        j_c = Orb[c].j
                        if ((2*(n_a + n_b + n_c) + l_a + l_b + l_c) <= N_3max)
                            P3B = rem(l_a + l_b + l_c, 2) + 1
                            @inbounds for f in 1:a_max
                                n_f = Orb[f].n
                                l_f = Orb[f].l
                                j_f = Orb[f].j
                                if (l_c == l_f) && (j_c == j_f) && ((2*(n_d + n_e + n_f) + l_d + l_e + l_f) <= N_3max)

                                    Hat = 1.0 / Float64(2*J + 1)
                                    ME001 = V3b_no2b(a,b,c,0,d,e,f,0,J,1,P3B,V_NNN,Orb,Orb_NNN)
                                    ME101 = V3b_no2b(a,b,c,1,d,e,f,0,J,1,P3B,V_NNN,Orb,Orb_NNN)
                                    ME011 = V3b_no2b(a,b,c,0,d,e,f,1,J,1,P3B,V_NNN,Orb,Orb_NNN)
                                    ME111 = V3b_no2b(a,b,c,1,d,e,f,1,J,1,P3B,V_NNN,Orb,Orb_NNN)
                                    ME113 = V3b_no2b(a,b,c,1,d,e,f,1,J,3,P3B,V_NNN,Orb,Orb_NNN)

                                    @views pnSum += Hat * ((0.5 * ME001 - 1.0 / sqrt(12.0) * ME101 - 1.0 / sqrt(12.0) * ME011 +
                                                1.0 / 6.0 * ME111 + 1.0 / 3.0 * ME113) * Rho.p[c,f] + (0.5 * ME001 + 1.0 / sqrt(12.0) *
                                                ME101 + 1.0 / sqrt(12.0) * ME011 + 1.0 / 6.0 * ME111 +  1.0 / 3.0 * ME113) * Rho.n[c,f])
                                end
                            end
                        end
                    end
                    @views V_NN_Res.pn[P,J+1][Ind] = p2 * V_NN_Res.pn[P,J+1][Ind] + p3 * pnSum
                end
            end

            if Bra <= N_T1
                @views a = Orb_NN.Ind[2,P,J+1][Bra][1]
                @views b = Orb_NN.Ind[2,P,J+1][Bra][2]
                l_a = Orb[a].l
                n_a = Orb[a].n
                l_b = Orb[b].l
                n_b = Orb[b].n
                for Ket in 1:Bra
                    Ind = Bra + (Ket - 1) * N_T1 - div(Ket * (Ket - 1),2)
                    @views d = Orb_NN.Ind[2,P,J+1][Ket][1]
                    @views e = Orb_NN.Ind[2,P,J+1][Ket][2]
                    l_d = Orb[d].l
                    n_d = Orb[d].n
                    l_e = Orb[e].l
                    n_e = Orb[e].n
                    ppSum = 0.0
                    nnSum = 0.0
                    @inbounds for c in 1:a_max
                        n_c = Orb[c].n
                        l_c = Orb[c].l
                        j_c = Orb[c].j
                        if ((2*(n_a + n_b + n_c) + l_a + l_b + l_c) <= N_3max)
                            P3B = rem(l_a + l_b + l_c, 2) + 1
                            @inbounds for f in 1:a_max
                                n_f = Orb[f].n
                                l_f = Orb[f].l
                                j_f = Orb[f].j
                                if (l_c == l_f) && (j_c == j_f) && ((2*(n_d + n_e + n_f) + l_d + l_e + l_f) <= N_3max)

                                    Hat = 1.0 / Float64(2*J + 1)
                                    ME111 = V3b_no2b(a,b,c,1,d,e,f,1,J,1,P3B,V_NNN,Orb,Orb_NNN)
                                    ME113 = V3b_no2b(a,b,c,1,d,e,f,1,J,3,P3B,V_NNN,Orb,Orb_NNN)

                                    @views ppSum += Hat * (ME113 * Rho.p[c,f] + (2.0/3.0 * ME111 + 1.0/3.0 * ME113) * Rho.n[c,f])
                                    @views nnSum += Hat * (ME113 * Rho.n[c,f] + (2.0/3.0 * ME111 + 1.0/3.0 * ME113) * Rho.p[c,f])

                                end
                            end
                        end
                    end
                    @views V_NN_Res.pp[P,J+1][Ind] = p2 * V_NN_Res.pp[P,J+1][Ind] + p3 * ppSum
                    @views V_NN_Res.nn[P,J+1][Ind] = p2 * V_NN_Res.nn[P,J+1][Ind] + p3 * nnSum
                end
    
            end
        end
    end

    # Perform transformation of the residual NN interaction to the canonical mean-field basis ...
    println("\nTransforming residual 2-body interaction from the LHO to the target basis...")

    @time V_NN_Res = O2b_transformation(Params,Orb,Orb_NN,V_NN_Res,C)

    println("\nResidual 2-body interaction ready...\n")

    return V_NN_Res, Orb_NN
end