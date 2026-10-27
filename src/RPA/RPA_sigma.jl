function RPA_sigma(Params::Parameters,Orb::Vector{Orb1B},Orb_1p1h::Orb2B_1p1h,Orb_NN::Orb2B,Rho::O1B,X::Matrix{Matrix{ComplexF64}},Y::Matrix{Matrix{ComplexF64}})
    # Read parameters ...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1
    a_max = div((N_max + 1)*(N_max + 2),2)
    hTol = Params.Calc.RPA.Tol_hol
    pTol = Params.Calc.RPA.Tol_par

    # Preallocate arrays of allowed values of J & P ...
    JP_list = JP_initialize(J_max)
    
    # Initialize the 2-body NN correlation matrix Sigma_NN ...
    # Note: Sigma_NN is stored in the unnormalized antisymmetrized J-coupled form <ab|Sigma|cd>_A (no 1/sqrt(2) for a == b or c == d),
    #       i.e. the same convention as V_NN, so that E_2b = 1/4 sum_{abcd,J} (2J+1) V_abcd Sigma_abcd ...
    Sigma_NN = O2b_initialize(Params,Orb)

    # Define local functions to determine particle & hole orbitals ...
    @inline particle(O::Float64) = (O <= pTol)

    @inline hole(O::Float64) = (O >= hTol)

    # Define local functions to precreate Bra & Ket iteration indices ...
    function RPA_sigma_orbitals_JP(J::Int64,P::Int64,Orb_NN::Orb2B)
        N_T0, N_T1 = Orb_NN.N[1,P,J+1], Orb_NN.N[2,P,J+1]

        BraKet_count = 0

        @inbounds for Bra in 1:N_T0
            @inbounds for Ket in 1:Bra
                BraKet_count += 1
            end
        end

        @inbounds for Bra in 1:N_T1
            @inbounds for Ket in 1:Bra
                BraKet_count += 2
            end
        end

        BraKet_list = Vector{Tuple{Int64,Int64,Int64}}(undef,BraKet_count)

        BraKet_count = 0

        # Case of proton-proton orbitals ...
        @inbounds for Bra in 1:N_T1
            @inbounds for Ket in 1:Bra
                BraKet_count += 1
                BraKet_list[BraKet_count] = (Bra,Ket,-1)
            end
        end

        # Case of proton-neutron orbitals ...
        @inbounds for Bra in 1:N_T0
            @inbounds for Ket in 1:Bra
                BraKet_count += 1
                BraKet_list[BraKet_count] = (Bra,Ket,0)
            end
        end

        # Case of neutron-neutron orbitals ...
        @inbounds for Bra in 1:N_T1
            @inbounds for Ket in 1:Bra
                BraKet_count += 1
                BraKet_list[BraKet_count] = (Bra,Ket,1)
            end
        end

        return BraKet_list
    end

    println("\nAllocating 2-body RPA correlation function matrix Sigma ...")

    # Allocate the components of the 2-body NN correlation matrix Sigma_NN ...
    @inbounds for JP in JP_list;
        J, P = JP[1], JP[2]

        if P == 1
            println("\tCalculating   ...   J = " * string(J) * "/" * string(N_2max+1) * "\tP = +")
        else
            println("\tCalculating   ...   J = " * string(J) * "/" * string(N_2max+1) * "\tP = -")
        end

        BraKet = RPA_sigma_orbitals_JP(J,P,Orb_NN)

        N_T0, N_T1 = Orb_NN.N[1,P,J+1], Orb_NN.N[2,P,J+1]

        Orb_NN_JP_T0 = Orb_NN.Ind[1,P,J+1]
        Orb_NN_JP_T1 = Orb_NN.Ind[2,P,J+1]

        pRho = Rho.p
        nRho = Rho.n

        @inbounds Threads.@threads for (Bra,Ket,T) in BraKet

            if T == -1
                Ind = Bra + (Ket - 1) * N_T1 - div(Ket * (Ket - 1),2)

                a, b = Orb_NN_JP_T1[Bra][1], Orb_NN_JP_T1[Bra][2]
                c, d = Orb_NN_JP_T1[Ket][1], Orb_NN_JP_T1[Ket][2]

                l_a, j_a = Orb[a].l, Orb[a].j
                l_b, j_b = Orb[b].l, Orb[b].j
                l_c, j_c = Orb[c].l, Orb[c].j
                l_d, j_d = Orb[d].l, Orb[d].j

                ppSigmaSum = 0.0

                # Parities of the 1p-1h pairings (a,c),(b,d) & exchanged pairings (a,d),(b,c) ...
                Q = rem(l_a + l_c,2) + 1
                Qx = rem(l_a + l_d,2) + 1
                if Q == (rem(l_b + l_d,2) + 1)
                    O_a, O_b = pRho[a,a], pRho[b,b]
                    O_c, O_d = pRho[c,c], pRho[d,d]

                    t_a, t_b = -1, -1
                    t_c, t_d = -1, -1

                    # 2p-2h terms ...
                    if particle(O_a) && particle(O_b) && hole(O_c) && hole(O_d)
                        ME = (RPA_sigma_pphh(a,j_a,t_a,O_a,b,j_b,t_b,O_b,c,j_c,t_c,O_c,d,j_d,t_d,O_d,J,Q,X,Y,Orb_1p1h) - phase(J + div(j_c + j_d,2)) * 
                              RPA_sigma_pphh(a,j_a,t_a,O_a,b,j_b,t_b,O_b,d,j_d,t_d,O_d,c,j_c,t_c,O_c,J,Qx,X,Y,Orb_1p1h))
                        ppSigmaSum += ME
                    end

                    if hole(O_a) && hole(O_b) && particle(O_c) && particle(O_d)
                        ME = (RPA_sigma_pphh(c,j_c,t_c,O_c,d,j_d,t_d,O_d,a,j_a,t_a,O_a,b,j_b,t_b,O_b,J,Q,X,Y,Orb_1p1h) - phase(J + div(j_c + j_d,2)) * 
                              RPA_sigma_pphh(d,j_d,t_d,O_d,c,j_c,t_c,O_c,a,j_a,t_a,O_a,b,j_b,t_b,O_b,J,Qx,X,Y,Orb_1p1h))
                        ppSigmaSum += ME
                    end

                    # 1p-1h terms ...
                    if particle(O_a) && hole(O_b) && particle(O_c) && hole(O_d)
                        ME = (RPA_sigma_phhp(a,j_a,t_a,O_a,b,j_b,t_b,O_b,c,j_c,t_c,O_c,d,j_d,t_d,O_d,J,Q,Y,Orb_1p1h) - phase(J + div(j_c + j_d,2)) * 
                              RPA_sigma_phhp(a,j_a,t_a,O_a,b,j_b,t_b,O_b,d,j_d,t_d,O_d,c,j_c,t_c,O_c,J,Qx,Y,Orb_1p1h))
                        ppSigmaSum += ME
                    end
                    
                    if particle(O_a) && hole(O_b) && hole(O_c) && particle(O_d)
                        ME = (RPA_sigma_phhp(a,j_a,t_a,O_a,b,j_b,t_b,O_b,c,j_c,t_c,O_c,d,j_d,t_d,O_d,J,Q,Y,Orb_1p1h) - phase(J + div(j_c + j_d,2)) * 
                              RPA_sigma_phhp(a,j_a,t_a,O_a,b,j_b,t_b,O_b,d,j_d,t_d,O_d,c,j_c,t_c,O_c,J,Qx,Y,Orb_1p1h))
                        ppSigmaSum += ME
                    end
                    
                    if hole(O_a) && particle(O_b) && hole(O_c) && particle(O_d)
                        ME = (RPA_sigma_hpph(a,j_a,t_a,O_a,b,j_b,t_b,O_b,c,j_c,t_c,O_c,d,j_d,t_d,O_d,J,Q,X,Orb_1p1h) - phase(J + div(j_c + j_d,2)) * 
                              RPA_sigma_hpph(a,j_a,t_a,O_a,b,j_b,t_b,O_b,d,j_d,t_d,O_d,c,j_c,t_c,O_c,J,Qx,X,Orb_1p1h))
                        ppSigmaSum += ME
                    end

                    if hole(O_a) && particle(O_b) && particle(O_c) && hole(O_d)
                        ME = (RPA_sigma_hpph(a,j_a,t_a,O_a,b,j_b,t_b,O_b,c,j_c,t_c,O_c,d,j_d,t_d,O_d,J,Q,X,Orb_1p1h) - phase(J + div(j_c + j_d,2)) * 
                              RPA_sigma_hpph(a,j_a,t_a,O_a,b,j_b,t_b,O_b,d,j_d,t_d,O_d,c,j_c,t_c,O_c,J,Qx,X,Orb_1p1h))
                        ppSigmaSum += ME
                    end
                end

                @inbounds Sigma_NN.pp[P,J+1][Ind] = ppSigmaSum
            end

            if T == 0
                Ind = Bra + (Ket - 1) * N_T0 - div(Ket * (Ket - 1),2)

                a, b = Orb_NN_JP_T0[Bra][1], Orb_NN_JP_T0[Bra][2]
                c, d = Orb_NN_JP_T0[Ket][1], Orb_NN_JP_T0[Ket][2]

                l_a, j_a = Orb[a].l, Orb[a].j
                l_b, j_b = Orb[b].l, Orb[b].j
                l_c, j_c = Orb[c].l, Orb[c].j
                l_d, j_d = Orb[d].l, Orb[d].j

                O_a, O_b = pRho[a,a], nRho[b,b]
                O_c, O_d = pRho[c,c], nRho[d,d]

                t_a, t_b = -1, 1
                t_c, t_d = -1, 1

                pnSigmaSum = 0.0

                # Parities of the 1p-1h pairings (a,c),(b,d) & exchanged pairings (a,d),(b,c) ...
                Q = rem(l_a + l_c,2) + 1
                Qx = rem(l_a + l_d,2) + 1
                if Q == (rem(l_b + l_d,2) + 1)
                    # 2p-2h terms ...
                    if particle(O_a) && particle(O_b) && hole(O_c) && hole(O_d)
                        ME = RPA_sigma_pphh(a,j_a,t_a,O_a,b,j_b,t_b,O_b,c,j_c,t_c,O_c,d,j_d,t_d,O_d,J,Q,X,Y,Orb_1p1h)
                        pnSigmaSum += ME
                    end

                    if hole(O_a) && hole(O_b) && particle(O_c) && particle(O_d)
                        ME = RPA_sigma_pphh(c,j_c,t_c,O_c,d,j_d,t_d,O_d,a,j_a,t_a,O_a,b,j_b,t_b,O_b,J,Q,X,Y,Orb_1p1h)
                        pnSigmaSum += ME
                    end

                    # 1p-1h terms ...
                    if particle(O_a) && hole(O_b) && hole(O_c) && particle(O_d)
                        ME = RPA_sigma_phhp(a,j_a,t_a,O_a,b,j_b,t_b,O_b,c,j_c,t_c,O_c,d,j_d,t_d,O_d,J,Q,Y,Orb_1p1h)
                        pnSigmaSum += ME
                    end
    
                    if hole(O_a) && particle(O_b) && particle(O_c) && hole(O_d)
                        ME = RPA_sigma_hpph(a,j_a,t_a,O_a,b,j_b,t_b,O_b,c,j_c,t_c,O_c,d,j_d,t_d,O_d,J,Q,Y,Orb_1p1h)
                        pnSigmaSum += ME
                    end
                end

                @inbounds Sigma_NN.pn[P,J+1][Ind] = pnSigmaSum
            end

            if T == 1
                Ind = Bra + (Ket - 1) * N_T1 - div(Ket * (Ket - 1),2)

                a, b = Orb_NN_JP_T1[Bra][1], Orb_NN_JP_T1[Bra][2]
                c, d = Orb_NN_JP_T1[Ket][1], Orb_NN_JP_T1[Ket][2]

                l_a, j_a = Orb[a].l, Orb[a].j
                l_b, j_b = Orb[b].l, Orb[b].j
                l_c, j_c = Orb[c].l, Orb[c].j
                l_d, j_d = Orb[d].l, Orb[d].j

                nnSigmaSum = 0.0

                # Parities of the 1p-1h pairings (a,c),(b,d) & exchanged pairings (a,d),(b,c) ...
                Q = rem(l_a + l_c,2) + 1
                Qx = rem(l_a + l_d,2) + 1
                if Q == (rem(l_b + l_d,2) + 1)
                    O_a, O_b = nRho[a,a], nRho[b,b]
                    O_c, O_d = nRho[c,c], nRho[d,d]

                    t_a, t_b = 1, 1
                    t_c, t_d = 1, 1

                    # 2p-2h terms ...
                    if particle(O_a) && particle(O_b) && hole(O_c) && hole(O_d)
                        ME = (RPA_sigma_pphh(a,j_a,t_a,O_a,b,j_b,t_b,O_b,c,j_c,t_c,O_c,d,j_d,t_d,O_d,J,Q,X,Y,Orb_1p1h) - phase(J + div(j_c + j_d,2)) * 
                              RPA_sigma_pphh(a,j_a,t_a,O_a,b,j_b,t_b,O_b,d,j_d,t_d,O_d,c,j_c,t_c,O_c,J,Qx,X,Y,Orb_1p1h))                        
                        nnSigmaSum += ME
                    end

                    if hole(O_a) && hole(O_b) && particle(O_c) && particle(O_d)
                        ME = (RPA_sigma_pphh(c,j_c,t_c,O_c,d,j_d,t_d,O_d,a,j_a,t_a,O_a,b,j_b,t_b,O_b,J,Q,X,Y,Orb_1p1h) - phase(J + div(j_a + j_b,2)) * 
                              RPA_sigma_pphh(c,j_c,t_c,O_c,d,j_d,t_d,O_d,b,j_b,t_b,O_b,a,j_a,t_a,O_a,J,Qx,X,Y,Orb_1p1h))
                        nnSigmaSum += ME
                    end

                    # 1p-1h terms ...
                    if particle(O_a) && hole(O_b) && particle(O_c) && hole(O_d)
                        ME = (RPA_sigma_phhp(a,j_a,t_a,O_a,b,j_b,t_b,O_b,c,j_c,t_c,O_c,d,j_d,t_d,O_d,J,Q,Y,Orb_1p1h) - phase(J + div(j_c + j_d,2)) * 
                              RPA_sigma_phhp(a,j_a,t_a,O_a,b,j_b,t_b,O_b,d,j_d,t_d,O_d,c,j_c,t_c,O_c,J,Qx,Y,Orb_1p1h))
                        nnSigmaSum += ME
                    end
                    
                    if particle(O_a) && hole(O_b) && hole(O_c) && particle(O_d)
                        ME = (RPA_sigma_phhp(a,j_a,t_a,O_a,b,j_b,t_b,O_b,c,j_c,t_c,O_c,d,j_d,t_d,O_d,J,Q,Y,Orb_1p1h) - phase(J + div(j_c + j_d,2)) * 
                              RPA_sigma_phhp(a,j_a,t_a,O_a,b,j_b,t_b,O_b,d,j_d,t_d,O_d,c,j_c,t_c,O_c,J,Qx,Y,Orb_1p1h))
                        nnSigmaSum += ME
                    end
                    
                    if hole(O_a) && particle(O_b) && hole(O_c) && particle(O_d)
                        ME = (RPA_sigma_hpph(a,j_a,t_a,O_a,b,j_b,t_b,O_b,c,j_c,t_c,O_c,d,j_d,t_d,O_d,J,Q,X,Orb_1p1h) - phase(J + div(j_c + j_d,2)) * 
                              RPA_sigma_hpph(a,j_a,t_a,O_a,b,j_b,t_b,O_b,d,j_d,t_d,O_d,c,j_c,t_c,O_c,J,Qx,X,Orb_1p1h))
                        nnSigmaSum += ME
                    end

                    if hole(O_a) && particle(O_b) && particle(O_c) && hole(O_d)
                        ME = (RPA_sigma_hpph(a,j_a,t_a,O_a,b,j_b,t_b,O_b,c,j_c,t_c,O_c,d,j_d,t_d,O_d,J,Q,X,Orb_1p1h) - phase(J + div(j_c + j_d,2)) * 
                              RPA_sigma_hpph(a,j_a,t_a,O_a,b,j_b,t_b,O_b,d,j_d,t_d,O_d,c,j_c,t_c,O_c,J,Qx,X,Orb_1p1h))
                        nnSigmaSum += ME
                    end
                end

                @inbounds Sigma_NN.nn[P,J+1][Ind] = nnSigmaSum
            end
        end
    end

    println("\tSuccesfully allocated 2-body RPA correlation matrix Sigma ...")
    
    return Sigma_NN
end

@inline function RPA_sigma_pphh(a_a::Int64,j_a::Int64,t_a::Int64,O_a::Float64,a_b::Int64,j_b::Int64,t_b::Int64,O_b::Float64,
                                a_c::Int64,j_c::Int64,t_c::Int64,O_c::Float64,a_d::Int64,j_d::Int64,t_d::Int64,O_d::Float64,
                                J::Int64,P::Int64,X::Matrix{Matrix{ComplexF64}},Y::Matrix{Matrix{ComplexF64}},Orb_1p1h::Orb2B_1p1h)
    sigma = 0.0

    I_min, I_max = max(div(abs(j_a - j_c),2),div(abs(j_b - j_d),2)),min(div((j_a + j_c),2),div((j_b + j_d),2))
    T_ac, T_bd = div(t_a + t_c,2), div(t_b + t_d,2)
    @inbounds for I in I_min:I_max
        i_ac = orb2b_1p1h_index(a_a,a_c,I,P,T_ac,Orb_1p1h)
        i_bd = orb2b_1p1h_index(a_b,a_d,I,P,T_bd,Orb_1p1h)

        if i_ac == 0 || i_bd == 0
            continue
        end
        
        N_nu = Orb_1p1h.N[P,I+1]

        Amp = Float64(2*I + 1) * phase(J + I + div(j_b + j_c,2)) * w6j(j_a,j_b,2*J,j_d,j_c,2*I) * abs((O_c - O_a) * (O_d - O_b))

        @inbounds for i_nu in 1:N_nu
            Y_ac = Y[P,I+1][i_ac,i_nu]
            X_bd = X[P,I+1][i_bd,i_nu]

            ME = Amp * real(Y_ac * conj(X_bd))
            sigma += ME
        end
    end

    if a_a == a_c && a_b == a_d
        ME = O_a * O_b
        sigma -= ME
    end

    if (a_a == a_d && t_a == t_d) && (a_b == a_c && t_b == t_c)
        ME = phase(J + div(j_a + j_b,2)) * O_a
        sigma -= ME
    end

    return sigma
end

@inline function RPA_sigma_phhp(a_a::Int64,j_a::Int64,t_a::Int64,O_a::Float64,a_b::Int64,j_b::Int64,t_b::Int64,O_b::Float64,
                                a_c::Int64,j_c::Int64,t_c::Int64,O_c::Float64,a_d::Int64,j_d::Int64,t_d::Int64,O_d::Float64,
                                J::Int64,P::Int64,Y::Matrix{Matrix{ComplexF64}},Orb_1p1h::Orb2B_1p1h)
    sigma = 0.0

    T_ac, T_db = div(t_a + t_c,2), div(t_b + t_d,2)
    I_min, I_max = max(div(abs(j_a - j_c),2),div(abs(j_b - j_d),2)),min(div((j_a + j_c),2),div((j_b + j_d),2))
    @inbounds for I in I_min:I_max
        i_ac = orb2b_1p1h_index(a_a,a_c,I,P,T_ac,Orb_1p1h)
        i_db = orb2b_1p1h_index(a_d,a_b,I,P,T_db,Orb_1p1h)

        if i_ac == 0 || i_db == 0
            continue
        end

        N_nu = Orb_1p1h.N[P,I+1]

        Amp = Float64(2*I + 1) * phase(J + div(j_c + j_d,2)) * w6j(j_a,j_b,2*J,j_d,j_c,2*I) * abs((O_c - O_a) * (O_b - O_d))

        @inbounds for i_nu in 1:N_nu
            Y_ac = Y[P,I+1][i_ac,i_nu]
            Y_db = Y[P,I+1][i_db,i_nu]

            ME = Amp * real(Y_ac * conj(Y_db))
            sigma += ME
        end
    end

    if a_a == a_c && a_b == a_d
        ME = O_a * O_b
        sigma -= ME
    end

    if (a_a == a_d && t_a == t_d) && (a_b == a_c && t_b == t_c)
        ME = phase(J + div(j_a + j_b,2)) * O_a
        sigma -= ME
    end

    return sigma
end

@inline function RPA_sigma_hpph(a_a::Int64,j_a::Int64,t_a::Int64,O_a::Float64,a_b::Int64,j_b::Int64,t_b::Int64,O_b::Float64,
                                a_c::Int64,j_c::Int64,t_c::Int64,O_c::Float64,a_d::Int64,j_d::Int64,t_d::Int64,O_d::Float64,
                                J::Int64,P::Int64,X::Matrix{Matrix{ComplexF64}},Orb_1p1h::Orb2B_1p1h)
    sigma = 0.0

    T_ca, T_bd = div(t_a + t_c,2), div(t_b + t_d,2)
    I_min, I_max = max(div(abs(j_a - j_c),2),div(abs(j_b - j_d),2)),min(div((j_a + j_c),2),div((j_b + j_d),2))
    @inbounds for I in I_min:I_max
        i_ca = orb2b_1p1h_index(a_c,a_a,I,P,T_ca,Orb_1p1h)
        i_bd = orb2b_1p1h_index(a_b,a_d,I,P,T_bd,Orb_1p1h)

        if i_ca == 0 || i_bd == 0
            continue
        end

        N_nu = Orb_1p1h.N[P,I+1]

        Amp = Float64(2*I + 1) * phase(J + div(j_a + j_b,2)) * w6j(j_a,j_b,2*J,j_d,j_c,2*I) * abs((O_a - O_c) * (O_b - O_d))

        @inbounds for i_nu in 1:N_nu
            X_ca = X[P,I+1][i_ca,i_nu]
            X_bd = X[P,I+1][i_bd,i_nu]

            ME = Amp * real(X_ca * conj(X_bd))
            sigma += ME
        end
    end

    if a_a == a_c && a_b == a_d
        ME = O_a * O_b
        sigma -= ME
    end

    if (a_a == a_d && t_a == t_d) && (a_b == a_c && t_b == t_c)
        ME = phase(J + div(j_a + j_b,2)) * O_a
        sigma -= ME
    end

    return sigma
end