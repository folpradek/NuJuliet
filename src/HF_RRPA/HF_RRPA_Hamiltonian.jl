function HF_RRPA_h1b(Params::Parameters,Rho::O1B,Orb::Vector{Orb1B},Orb_NN::Orb2B,Orb_NNN::Orb3B,T::O1B,V_NN::O2B,V_NNN::Array{Vector{Vector{Float32}},4})
    # Read parameters ...
    A = Params.Calc.A
    N_max = Params.Calc.Nmax
    N_2max = Params.Calc.N2max
    N_3max = Params.Calc.N3max
    CMS = Params.Calc.CMS

    a_max = div((N_max + 1)*(N_max + 2),2)

    # Initialize mean-field Hamiltonian matrix ...
    pH, nH = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)

    # Fill MF Hamiltonian ...
    @inbounds Threads.@threads for a = 1:a_max
        n_a = Orb[a].n
        l_a = Orb[a].l
        j_a = Orb[a].j
        Hat = 1.0 / (Float64(j_a) + 1.0)
        @inbounds for d = 1:a
            l_d = Orb[d].l
            j_d = Orb[d].j
            if l_a == l_d && j_a == j_d
                n_d = Orb[d].n
                pSum = 0.0
                nSum = 0.0
                @inbounds for b = 1:a_max
                    n_b = Orb[b].n
                    l_b = Orb[b].l
                    if (2*(n_a + n_b) + l_a + l_b) <= N_2max
                        P_ab = rem(l_a + l_b, 2) + 1
                        j_b = Orb[b].j
                        @inbounds for e = 1:a_max
                            n_e = Orb[e].n
                            l_e = Orb[e].l
                            j_e = Orb[e].j
                            if l_b == l_e && j_b == j_e && (2*(n_d + n_e) + l_d + l_e) <= N_2max
                                pRho_be = Rho.p[b,e]
                                nRho_be = Rho.n[b,e]

                                @inbounds for J = div(abs(j_d - j_e),2):div(j_d + j_e,2)

                                    if rem(l_a + l_b, 2) == rem(l_d + l_e, 2)
                                        Hat_2b = (Float64(2*J) + 1.0) * Hat
                                        pSum += Hat_2b * pRho_be * O2b_pp(a,b,d,e,J,P_ab,V_NN,Orb,Orb_NN)
                                        pSum += Hat_2b * nRho_be * O2b_pn(a,b,d,e,J,P_ab,V_NN,Orb_NN)
                                        nSum += Hat_2b * nRho_be * O2b_nn(a,b,d,e,J,P_ab,V_NN,Orb,Orb_NN)
                                        nSum += Hat_2b * pRho_be * O2b_pn(b,a,e,d,J,P_ab,V_NN,Orb_NN)
                                    end

                                    # 3-body NNN interaction
                                    @inbounds for c = 1:a_max
                                        n_c = Orb[c].n
                                        l_c = Orb[c].l
                                        if (2*(n_a + n_b + n_c) + l_a + l_b + l_c) <= N_3max
                                            P_abc = rem(l_a + l_b + l_c, 2) + 1
                                            j_c = Orb[c].j
                                            @inbounds for f = 1:a_max
                                                n_f = Orb[f].n
                                                l_f = Orb[f].l
                                                if (2*(n_d + n_e + n_f) + l_d + l_e + l_f) <= N_3max && l_c == l_f && P_abc == (rem(l_d + l_e + l_f,2) + 1)
                                                    j_f = Orb[f].j
                                                    if j_c == j_f
                                                        pRho_cf = Rho.p[c,f]
                                                        nRho_cf = Rho.n[c,f]

                                                        ME001 = V3b_no2b(a,b,c,0,d,e,f,0,J,1,P_abc,V_NNN,Orb,Orb_NNN)
                                                        ME101 = V3b_no2b(a,b,c,1,d,e,f,0,J,1,P_abc,V_NNN,Orb,Orb_NNN)
                                                        ME011 = V3b_no2b(a,b,c,0,d,e,f,1,J,1,P_abc,V_NNN,Orb,Orb_NNN)
                                                        ME111 = V3b_no2b(a,b,c,1,d,e,f,1,J,1,P_abc,V_NNN,Orb,Orb_NNN)
                                                        ME113 = V3b_no2b(a,b,c,1,d,e,f,1,J,3,P_abc,V_NNN,Orb,Orb_NNN)

                                                        pSum += Hat * (0.5*ME113*pRho_be*pRho_cf + 0.25 * (ME001 +
                                                                sqrt(1.0/3.0)*ME101 + sqrt(1.0/3.0)*ME011 + 1.0/3.0*ME111 + 2.0/3.0*ME113)*nRho_be*nRho_cf +
                                                                1.0/3.0 * (2*ME111 + ME113)*pRho_be*nRho_cf)

                                                        nSum += Hat * (0.5*ME113*nRho_be*nRho_cf + 0.25 * (ME001 +
                                                                sqrt(1.0/3.0)*ME101 + sqrt(1.0/3.0)*ME011 + 1.0/3.0*ME111 + 2.0/3.0*ME113)*pRho_be*pRho_cf +
                                                                1.0/3.0 * (2*ME111 + ME113)*nRho_be*pRho_cf)

                                                    end
                                                end
                                            end
                                        end
                                    end
                                end

                            end
                        end
                    end
                end

                # 1-body kinetic operator & CMS correction
                if CMS == "CMS1+2B"
                    pH[a,d] = pSum + T.p[a,d] * (1.0 - 1.0/A)
                    nH[a,d] = nSum + T.n[a,d] * (1.0 - 1.0/A)
                    pH[d,a] = pH[a,d]
                    nH[d,a] = nH[a,d]
                elseif CMS == "CMS2B"
                    pH[a,d] = pSum
                    nH[a,d] = nSum
                    pH[d,a] = pH[a,d]
                    nH[d,a] = nH[a,d]
                else
                    pH[a,d] = pSum + T.p[a,d]
                    nH[a,d] = nSum + T.n[a,d]
                    pH[d,a] = pH[a,d]
                    nH[d,a] = nH[a,d]
                end
            end
        end
    end

    return O1B(pH,nH)
end

function HF_RRPA_V2b(Params::Parameters,Orb::Vector{Orb1B},Orb_NN_Bare::Orb2B,Orb_NNN_Bare::Orb3B,V_NN_Bare::O2B,V_NNN_Bare::Array{Vector{Vector{Float32}},4},Rho::O1B,C::O1B)
    # Copy NN interation array ...
    V_NN = deepcopy(V_NN_Bare)
    
    Out = "/dev/null"
    if Sys.iswindows()
        Out = "NUL"
    end

    # Update Density-dependent residual NN interaction & transform to the HF-RRPA basis ...
    open(Out, "w") do devnull_io
        redirect_stdout(devnull_io) do
            redirect_stderr(devnull_io) do
                # Update residual interaction
                V_NN = V2b_residual_no2b(Params,Orb,Orb_NN_Bare,Orb_NNN_Bare,Rho,V_NN,V_NNN_Bare)

                # Transform residual interaction to the HF-RRPA basis
                V_NN_Res = O2b_transformation(Params,Orb,Orb_NN_Bare,V_NN,C)

                # Deallocate the NN interaction & perform garbace collection ...
                V_NN = nothing
                GC.gc()

                return V_NN_Res
            end
        end
    end
end

function HF_RRPA_V2b_transform(Params::Parameters,Orb::Vector{Orb1B},Orb_NN_Res::Orb2B,V_NN_Res::O2B,C::O1B)
    # Copy NN interation array ...
    V_NN = deepcopy(V_NN_Res)
    
    Out = "/dev/null"
    if Sys.iswindows()
        Out = "NUL"
    end

    # Transform the residual NN interaction to the new HF-ERPA basis...
    open(Out, "w") do devnull_io
        redirect_stdout(devnull_io) do
            redirect_stderr(devnull_io) do
                V_NN_Res = O2b_transformation(Params,Orb,Orb_NN_Res,V_NN,C)
                return V_NN_Res
            end
        end
    end
end