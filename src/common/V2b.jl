function V2b_read(Params::Parameters,Orb::Vector{Orb1B})
    # Read interactions parameters ...
    Int_File = Params.Int.NN_File
    hw = Params.Int.hw
    N_max_Int = Params.Int.Nmax
    N_2max_Int = Params.Int.N2max
    N_max_Calc = Params.Calc.Nmax
    N_2max_Calc = Params.Calc.N2max
    A = Params.Calc.A
    CMS = Params.Calc.CMS

    a_max = div((N_max_Int+1)*(N_max_Int+2),2)
    J_max = N_2max_Calc + 1

    # Initialize 1-body kinetic operator ...
    T = T1b(Params,Orb)

    # Initialize NN interaction array ...
    println("\nPreparing 2-body NN interaction array...")
    V_NN, Orb_NN = O2b_initialize(Params,Orb,Make_Orb_NN=true)

    # Precalculate chunks of bytes to skip ...
    N_Chunk_skip = V2b_count(N_max_Int,N_2max_Int,Orb)

    # Read NN interaction matrix elements ...
    println("\nReading 2-body NN interaction file...")
    @inbounds Threads.@threads for a in 1:a_max
        n_a = Orb[a].n
        l_a = Orb[a].l
        j_a = Orb[a].j
        N_a = 2*n_a + l_a
        open(Int_File, "r") do Bin_Read
            seek(Bin_Read, N_Chunk_skip[a])
            @inbounds for b in 1:a
                n_b = Orb[b].n
                l_b = Orb[b].l
                j_b = Orb[b].j
                N_b = 2*n_b + l_b
                N_ab = N_a + N_b
                P = rem(l_a + l_b,2) + 1
                if N_ab <= N_2max_Int
                    @inbounds for c in 1:a
                        n_c = Orb[c].n
                        l_c = Orb[c].l
                        j_c = Orb[c].j
                        N_c = 2*n_c + l_c
                        d_max = c
                        if a == c
                            d_max = b
                        end
                        @inbounds for d in 1:d_max
                            n_d = Orb[d].n
                            l_d = Orb[d].l
                            j_d = Orb[d].j
                            N_d = 2*n_d + l_d
                            N_cd = N_c + N_d
                            if (N_cd <= N_2max_Int) && (rem(l_a + l_b,2) == rem(l_c + l_d,2))
                                @inbounds for J in div(max(abs(j_a - j_b), abs(j_c - j_d)),2):div(min(j_a + j_b, j_c + j_d),2)
                                    ME_00 = read(Bin_Read,Float64)
                                    ME_nn = read(Bin_Read,Float64)
                                    ME_10 = read(Bin_Read,Float64)
                                    ME_pp = read(Bin_Read,Float64)

                                    if (N_a <= N_max_Calc) && (N_b <= N_max_Calc) && (N_c <= N_max_Calc) && (N_d <= N_max_Calc) &&
                                       (N_ab <= N_2max_Calc) && (N_cd <= N_2max_Calc)
                                        Amp = 1.0
                                        if a == b
                                            Amp = Amp / sqrt(2)
                                        end
                                        if c == d
                                            Amp = Amp / sqrt(2)
                                        end

                                        if (abs(Amp - 1.0) < 1e-4) || (rem(J,2) == 0)
                                            Ind = O2b_index(a,b,c,d,J,P,1,Orb_NN)
                                            V_NN.pp[P,J+1][Ind] += ME_pp
                                            V_NN.nn[P,J+1][Ind] += ME_nn
                                        end

                                        if (abs(Amp - 1.0) < 1e-4) || (rem(J,2) == 0)
                                            Ind = O2b_index(a,b,c,d,J,P,0,Orb_NN)
                                            V_NN.pn[P,J+1][Ind] += 0.5 * ME_10

                                            if c != d
                                                Phase = (-1)^(div(j_c + j_d,2) + J + 1)
                                                Ind = O2b_index(a,b,d,c,J,P,0,Orb_NN)
                                                V_NN.pn[P,J+1][Ind] += 0.5 * Phase * ME_10
                                            end

                                            if (a != b) && ( c!= d)
                                                Phase = (-1)^(div(j_a + j_b + j_c + j_d,2))
                                                Ind = O2b_index(b,a,d,c,J,P,0,Orb_NN)
                                                V_NN.pn[P,J+1][Ind] += 0.5 * Phase * ME_10
                                            end

                                            if (a != b) && ((a != c) || (b != d))
                                                Phase = (-1)^(div(j_a + j_b,2) + J + 1)
                                                Ind = O2b_index(b,a,c,d,J,P,0,Orb_NN)
                                                V_NN.pn[P,J+1][Ind] += 0.5 * Phase * ME_10
                                            end

                                        end

                                        if (abs(Amp - 1.0) < 1e-4) || (rem(J,2) == 1)
                                            Ind = O2b_index(a,b,c,d,J,P,0,Orb_NN)
                                            V_NN.pn[P,J+1][Ind] += 0.5 * ME_00

                                            if c != d
                                                Phase = (-1)^(div(j_c + j_d,2) + J)
                                                Ind = O2b_index(a,b,d,c,J,P,0,Orb_NN)
                                                V_NN.pn[P,J+1][Ind] += 0.5 * Phase * ME_00
                                            end

                                            if (a != b) && (c != d)
                                                Phase = (-1)^(div(j_a + j_b + j_c + j_d,2))
                                                Ind = O2b_index(b,a,d,c,J,P,0,Orb_NN)
                                                V_NN.pn[P,J+1][Ind] += 0.5 * Phase * ME_00
                                            end

                                            if (a != b) && ((a != c) || (b != d))
                                                Phase = (-1)^(div(j_a + j_b,2) + J)
                                                Ind = O2b_index(b,a,c,d,J,P,0,Orb_NN)
                                                V_NN.pn[P,J+1][Ind] += 0.5 * Phase * ME_00
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
    end

    println("\nNN 2-body interaction loaded...")

    if CMS == "CMS1+2B" || CMS == "CMS2B"
        println("\nLoading 2-body CMS correction...")
        JP_List = JP_initialize(J_max)
        @inbounds Threads.@threads for JP in JP_List
            J, P = JP[1], JP[2]
            N_t0, N_t1 = Orb_NN.N[1,P,J+1], Orb_NN.N[2,P,J+1]
            @inbounds for Bra in 1:max(N_t0,N_t1)
                @inbounds for Ket in 1:Bra

                    if Bra <= N_t0
                        Ind = Bra + (Ket - 1) * N_t0 - div(Ket * (Ket - 1),2)
                        a = Orb_NN.Ind[1,P,J+1][Bra][1]
                        b = Orb_NN.Ind[1,P,J+1][Bra][2]
                        c = Orb_NN.Ind[1,P,J+1][Ket][1]
                        d = Orb_NN.Ind[1,P,J+1][Ket][2]
                        if CMS == "CMS1+2B"
                            T_sym = T2b(Orb,a,b,c,d,J) * hw / Float64(A)
                        elseif CMS == "CMS2B"
                            T_sym = (hw * T2b(Orb,a,b,c,d,J) + T[a,c] * Float64(kronecker_delta(b,d)) + T[b,d] * Float64(kronecker_delta(a,c))) / Float64(A)
                        end
                        V_NN.pn[P,J+1][Ind] += T_sym
                    end

                    if Bra <= N_t1
                        Ind = Bra + (Ket - 1) * N_t1 - div(Ket * (Ket - 1),2)
                        a = Orb_NN.Ind[2,P,J+1][Bra][1]
                        b = Orb_NN.Ind[2,P,J+1][Bra][2]
                        c = Orb_NN.Ind[2,P,J+1][Ket][1]
                        d = Orb_NN.Ind[2,P,J+1][Ket][2]
                        j_c = Orb[c].j
                        j_d = Orb[d].j
                        if CMS == "CMS1+2B"
                            j_c = Orb[c].j
                            j_d = Orb[d].j
                            T_antisym = 1.0 / sqrt(Float64(1 + kronecker_delta(a,b)) * Float64(1 + kronecker_delta(c,d))) * hw * 
                                        (T2b(Orb,a,b,c,d,J) - Float64((-1)^(div(j_c + j_d,2) - J)) * T2b(Orb,a,b,d,c,J)) / Float64(A)
                        elseif CMS == "CMS2B"
                            j_c = Orb[c].j
                            j_d = Orb[d].j
                            Amp = hw / sqrt(Float64(1 + kronecker_delta(a,b)) * Float64(1 + kronecker_delta(c,d)))
                            Amp_2 = 1.0 / sqrt(Float64(1 + kronecker_delta(a,b)) * Float64(1 + kronecker_delta(c,d)))
                            T_antisym = (Amp * T2b(Orb,a,b,c,d,J) + Amp_2 * (kronecker_delta(b,d) * T[a,c] + kronecker_delta(a,c) * T[b,d])
                                        - Float64((-1)^(div(j_c + j_d,2) - J)) * (Amp * T2b(Orb,a,b,d,c,J) + Amp_2 * (Float64(kronecker_delta(b,c)) *
                                        T[a,d] + Float64(kronecker_delta(a,d)) * T[b,c]))) / Float64(A)
                        end
                        ME =  T_antisym * sqrt(Float64(1 + kronecker_delta(a,b)) * Float64(1 + kronecker_delta(c,d)))
                        V_NN.pp[P,J+1][Ind] += ME
                        V_NN.nn[P,J+1][Ind] += ME
                    end

                end
            end
        end
        println("\nCMS 2-body correction added...")
    end

    # Enhance contact V_aabb^J=0 matrix elements ...
    pG, nG = Params.Int.pG2N, Params.Int.nG2N
    N_min, N_max = -1, 25
    if abs(pG) > 1e-8 || abs(nG) > 1e-8
        N_T1 = Orb_NN.N[2,1,1]
        @inbounds Threads.@threads for Bra in 1:N_T1
            a, b = Orb_NN.Ind[2,1,1][Bra][1], Orb_NN.Ind[2,1,1][Bra][2]
            if a == b
                l_a, j_a = Orb[a].l, Orb[a].j
                N_a = 2*Orb[a].n + l_a
                if N_a < N_min || N_a > N_max
                    break
                end
                @inbounds for Ket in 1:Bra
                    c, d = Orb_NN.Ind[2,1,1][Ket][1], Orb_NN.Ind[2,1,1][Ket][2]
                    if c == d
                        l_c, j_c = Orb[c].l, Orb[c].j
                        N_c = 2*Orb[c].n + l_c
                        if N_c < N_min || N_c > N_max
                            break
                        end
                        Ind = Bra + (Ket - 1) * N_T1 - div(Ket * (Ket - 1),2)
                        Amp = -0.5 * sqrt(Float64((j_a + 1) * (j_c + 1))) * Float64((-1)^(l_a + l_c))
                        V_NN.pp[1,1][Ind] += Amp * pG
                        V_NN.nn[1,1][Ind] += Amp * nG
                    end
                end
            end
        end
    end

    #=
        # Export variable ... change in order to allow/forbid the following export ...
        V2b_0p_analysis = true

        if V2b_0p_analysis

            if !(isdir("IO/" * Params.Calc.Path * "/V2b_pairing"))
                mkdir("IO/" * Params.Calc.Path * "/V2b_pairing")
            end

            # Pairing JP = 0+ channel analysis export ...
            V2b_0p_full_Path = "IO/" * Params.Calc.Path * "/V2b_pairing/V2b_0+_full.dat"
            V2b_0p_mono_Path = "IO/" * Params.Calc.Path * "/V2b_pairing/V2b_0+_mono.dat"

            open(V2b_0p_full_Path, "w") do Export_File
                @printf(Export_File, "%6s\t%6s\t%12s\t%12s\n", "Bra", "Ket", "ppV", "nnV")
                J, P = 0, 1

                N_T1 = Orb_NN.N[2,P,J+1]
                @inbounds for Bra in 1:N_T1
                    @inbounds for Ket in 1:Bra
                        Ind = Bra + (Ket - 1) * N_T1 - div(Ket * (Ket - 1),2)
                        ppME = @views V_NN.pp[P,J+1][Ind]
                        nnME = @views V_NN.nn[P,J+1][Ind]
                        @printf(Export_File, "%6d\t%6d\t%12.8f\t%12.8f\n", Bra, Ket, ppME, nnME)
                        if Bra != Ket
                            @printf(Export_File, "%6d\t%6d\t%12.8f\t%12.8f\n", Ket, Bra, ppME, nnME)
                        end
                    end
                end
    
            end

            open(V2b_0p_mono_Path, "w") do Export_File
                @printf(Export_File, "%6s\t%6s\t%12s\t%12s\n", "Bra", "Ket", "ppV", "nnV")
                J, P = 0, 1

                N_T1 = Orb_NN.N[2,P,J+1]
                @inbounds for Bra in 1:N_T1
                    a, b = Orb_NN.Ind[2,1,1][Bra][1], Orb_NN.Ind[2,1,1][Bra][2]
                    if a == b
                        @inbounds for Ket in 1:Bra
                            c, d = Orb_NN.Ind[2,1,1][Ket][1], Orb_NN.Ind[2,1,1][Ket][2]
                            if c == d
                                Ind = Bra + (Ket - 1) * N_T1 - div(Ket * (Ket - 1),2)
                                ppME = @views V_NN.pp[P,J+1][Ind]
                                nnME = @views V_NN.nn[P,J+1][Ind]

                                @printf(Export_File, "%6d\t%6d\t%12.8f\t%12.8f\n", Bra, Ket, ppME, nnME)
                                if Bra != Ket
                                    @printf(Export_File, "%6d\t%6d\t%12.8f\t%12.8f\n", Ket, Bra, ppME, nnME)
                                end
                            end
                        end
                    end
                end

            end

            println("\nDimension of the JP = 0+ NN interaction channel: dim = " * string(Orb_NN.N[2,1,1]))

        end
    =#

    return V_NN, Orb_NN
end

function V2b_count(N_max::Int64,N_2max::Int64,Orb::Vector{Orb1B})
    # Calculate the total number of orbitals ...
    a_max = div((N_max+1)*(N_max+2),2)

    # Initialize the arrays for chunk sizes ...
    N_Chunk = Vector{Int64}(undef,a_max)
    N_Chunk_skip = Vector{Int64}(undef,a_max)

    # Calculate the chunk sizes ...
    @inbounds Threads.@threads for a in 1:a_max
        n_a = Orb[a].n
        l_a = Orb[a].l
        j_a = Orb[a].j
        N = 0
        @inbounds for b in 1:a
            n_b = Orb[b].n
            l_b = Orb[b].l
            j_b = Orb[b].j
            if (2*(n_a + n_b) + l_a + l_b) <= N_2max
                @inbounds for c in 1:a
                    n_c = Orb[c].n
                    l_c = Orb[c].l
                    j_c = Orb[c].j
                    d_max = c

                    if a == c
                        d_max = b
                    end

                    @inbounds for d in 1:d_max
                        n_d = Orb[d].n
                        l_d = Orb[d].l
                        j_d = Orb[d].j
                        if ((2*(n_c + n_d) + l_c + l_d) <= N_2max) && (rem(l_a + l_b,2) == rem(l_c + l_d,2))
                            @inbounds for J in div(max(abs(j_a - j_b), abs(j_c - j_d)),2):div(min(j_a + j_b, j_c + j_d),2)
                                N += 4
                            end
                        end
                    end
                end
            end
        end
        N_Chunk[a] = N
    end

    # Calculate the chunks to be skipped ...
    @inbounds for a in 1:a_max
        Sum = 0
        @inbounds for b in 1:(a-1)
            Sum += 8 * N_Chunk[b]
        end
        N_Chunk_skip[a] = Sum
    end

    return N_Chunk_skip
end

function V2b_residual_no2b(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,Orb_NNN::Orb3B,Rho::O1B,V_NN::O2B,V_NNN::Array{Vector{Vector{Float32}},4})
    # Read parameters ...
    N_max, N_2max, N_3max = Params.Calc.Nmax, Params.Calc.N2max, Params.Calc.N3max
    a_max, J_max = div((N_max + 1)*(N_max + 2),2), N_2max + 1
    cRes = Params.Int.cRes

    # Define local constants
    isr12 = 1.0 / sqrt(12.0)
    i6 = 1.0 / 6.0
    i3 = 1.0 / 3.0

    # Initialize list of J & P values ...
    JP = JP_initialize(J_max)

    V_NN_res = deepcopy(V_NN)

    println("\nStarting calculation of residual NN interaction ...\n")

    # Include NO2B NNN interaction to V_NN...
    println("\nMaking density dependent residual 2-body interaction...")
    @inbounds for jp in JP
        J, P = jp[1], jp[2]
        if P == 1
            println("\tCalculating   ...   J = " * string(J) * "/" * string(N_2max+1) * "\tP = +")
        else
            println("\tCalculating   ...   J = " * string(J) * "/" * string(N_2max+1) * "\tP = -")
        end
        N_T0, N_T1 = Orb_NN.N[1,P,J+1], Orb_NN.N[2,P,J+1]
        Hat = 1.0 / Float64(2*J + 1)

        @inbounds Threads.@threads for Bra_ind in 1:(N_T0 + N_T1)

            # pnV loop ...
            if Bra_ind <= N_T0
                Bra = Bra_ind
                a, b = Orb_NN.Ind[1,P,J+1][Bra][1], Orb_NN.Ind[1,P,J+1][Bra][2]
                n_a, l_a = Orb[a].n, Orb[a].l
                n_b, l_b = Orb[b].n, Orb[b].l
                @inbounds for Ket in 1:Bra
                    Ind = Bra + (Ket - 1) * N_T0 - div(Ket * (Ket - 1),2)
                    d, e = Orb_NN.Ind[1,P,J+1][Ket][1], Orb_NN.Ind[1,P,J+1][Ket][2]
                    n_d, l_d = Orb[d].n, Orb[d].l
                    n_e, l_e = Orb[e].n, Orb[e].l
                    pnSum = 0.0
                    @inbounds for c in 1:a_max
                        n_c, l_c, j_c = Orb[c].n, Orb[c].l, Orb[c].j
                        if ((2*(n_a + n_b + n_c) + l_a + l_b + l_c) <= N_3max)
                            P3B = rem(l_a + l_b + l_c, 2) + 1
                            @inbounds @simd for f in 1:a_max
                                n_f, l_f, j_f = Orb[f].n, Orb[f].l, Orb[f].j
                                if (l_c == l_f) && (j_c == j_f) && ((2*(n_d + n_e + n_f) + l_d + l_e + l_f) <= N_3max)
                                    ME001 = V3b_no2b(a,b,c,0,d,e,f,0,J,1,P3B,V_NNN,Orb,Orb_NNN)
                                    ME101 = V3b_no2b(a,b,c,1,d,e,f,0,J,1,P3B,V_NNN,Orb,Orb_NNN)
                                    ME011 = V3b_no2b(a,b,c,0,d,e,f,1,J,1,P3B,V_NNN,Orb,Orb_NNN)
                                    ME111 = V3b_no2b(a,b,c,1,d,e,f,1,J,1,P3B,V_NNN,Orb,Orb_NNN)
                                    ME113 = V3b_no2b(a,b,c,1,d,e,f,1,J,3,P3B,V_NNN,Orb,Orb_NNN)
                                    @inbounds pnSum += Hat * ((0.5 * ME001 - ME101 * isr12 - ME011 * isr12 +
                                                        ME111 * i6 + ME113 * i3) * Rho.p[f,c] +
                                                       (0.5 * ME001 + ME101 * isr12 + ME011 * isr12 +
                                                        ME111 * i6 + ME113 * i3) * Rho.n[f,c])
                                end
                            end
                        end
                    end
                    @inbounds V_NN_res.pn[P,J+1][Ind] += pnSum
                    @inbounds V_NN_res.pn[P,J+1][Ind] *= cRes
                end
            end

            # ppV && nnV loop ...
            if Bra_ind > N_T0
                Bra = Bra_ind - N_T0
                a, b = Orb_NN.Ind[2,P,J+1][Bra][1], Orb_NN.Ind[2,P,J+1][Bra][2]
                n_a, l_a = Orb[a].n, Orb[a].l
                n_b, l_b = Orb[b].n, Orb[b].l
                @inbounds for Ket in 1:Bra
                    Ind = Bra + (Ket - 1) * N_T1 - div(Ket * (Ket - 1),2)
                    d, e = Orb_NN.Ind[2,P,J+1][Ket][1], Orb_NN.Ind[2,P,J+1][Ket][2]
                    n_d, l_d = Orb[d].n, Orb[d].l
                    n_e, l_e = Orb[e].n, Orb[e].l
                    ppSum, nnSum = 0.0, 0.0

                    @inbounds for c in 1:a_max
                        n_c, l_c, j_c = Orb[c].n, Orb[c].l, Orb[c].j
                        if ((2*(n_a + n_b + n_c) + l_a + l_b + l_c) <= N_3max)
                            P3B = rem(l_a + l_b + l_c, 2) + 1
                            @inbounds @simd for f in 1:a_max
                                n_f, l_f, j_f = Orb[f].n, Orb[f].l, Orb[f].j
                                if (l_c == l_f) && (j_c == j_f) && ((2*(n_d + n_e + n_f) + l_d + l_e + l_f) <= N_3max)
                                    ME111 = V3b_no2b(a,b,c,1,d,e,f,1,J,1,P3B,V_NNN,Orb,Orb_NNN)
                                    ME113 = V3b_no2b(a,b,c,1,d,e,f,1,J,3,P3B,V_NNN,Orb,Orb_NNN)
                                    @inbounds ppSum += Hat * (ME113 * Rho.p[f,c] + (2.0 * ME111 + ME113) * Rho.n[f,c] * i3)
                                    @inbounds nnSum += Hat * (ME113 * Rho.n[f,c] + (2.0 * ME111 + ME113) * Rho.p[f,c] * i3)

                                end
                            end
                        end
                    end
                    @inbounds V_NN_res.pp[P,J+1][Ind] += ppSum
                    @inbounds V_NN_res.nn[P,J+1][Ind] += nnSum
                    @inbounds V_NN_res.pp[P,J+1][Ind] *= cRes
                    @inbounds V_NN_res.nn[P,J+1][Ind] *= cRes
                end
    
            end

        end
    end

    return V_NN_res
end

function V2b_correlation_energy(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,Sigma::O2B,V_NN::O2B)
    # Read parameters ...
    N_max = Params.Calc.Nmax
    N_2max = Params.Calc.N2max
    a_max = div((N_max + 1)*(N_max + 2),2)

    # Calculate the HF energy ...
    println("\nComputing the correlation ground-state energy by contraction with the provided 2-body correlation function ...")

    # Initialize accumulator for the threaded correlation energy ...
    E_corr_threads = zeros(Float64,Threads.maxthreadid())

    @inbounds Threads.@threads :static for a in 1:a_max
        Sum, Tid = 0.0, Threads.threadid()

        Orb_a = Orb[a]
        n_a, l_a, j_a = Orb_a.n, Orb_a.l, Orb_a.j

        @inbounds for b in 1:a_max
            Orb_b = Orb[b]
            n_b, l_b = Orb_b.n, Orb_b.l

            P = rem(l_a + l_b, 2) + 1

            if (2*(n_a + n_b) + l_a + l_b) > N_2max
                continue
            end

            j_b = Orb[b].j

            @inbounds for c = 1:a_max
                Orb_c = Orb[c]
                n_c, l_c,j_c = Orb_c.n, Orb_c.l, Orb_c.j

                @inbounds for d = 1:a_max
                    Orb_d = Orb[d]
                    n_d, l_d, j_d = Orb_d.n, Orb_d.l, Orb_d.j

                    if ((2*(n_c + n_d) + l_c + l_d) > N_2max) || (rem(l_a + l_b, 2) != rem(l_c + l_d, 2))
                        continue
                    end

                    J_min, J_max = max(div(abs(j_a - j_b),2),div(abs(j_c - j_d),2)), min(div((j_a + j_b),2),div((j_c + j_d),2))

                    @inbounds for J in J_min:J_max
                        Hat =  Float64(2*J + 1)
                        Sum += 0.25 * Hat * O2b_pp(a,b,c,d,J,P,V_NN,Orb,Orb_NN) * O2b_pp(a,b,c,d,J,P,Sigma,Orb,Orb_NN)
                        Sum += 0.25 * Hat * O2b_nn(a,b,c,d,J,P,V_NN,Orb,Orb_NN) * O2b_nn(a,b,c,d,J,P,Sigma,Orb,Orb_NN)
                        Sum += Hat * O2b_pn(a,b,c,d,J,P,V_NN,Orb_NN) * O2b_pn(a,b,c,d,J,P,Sigma,Orb_NN)
                    end
                end
            end
        end
        E_corr_threads[Tid] += Sum
    end

    # Accumulate the total correlation energy ...
    E_corr = sum(E_corr_threads)

    println("\tPerformed calculation of correlation energy by contraction with 2-body correlation function ...\n")

    return E_corr
end
