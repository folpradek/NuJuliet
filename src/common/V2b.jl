function V2b_read(Params::Parameters)
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

    # Initialize orbitals for 2-body NN interaction read ...
    Orb = orbitals_make(Params,Int = true)

    # Initialize 1-body kinetic operator ...
    T = T1b(Params,Orb)

    # Initialize NN interaction array ...
    println("\nPreparing 2-body NN interaction array...")
    V_NN, Orb_NN = O2b_initialize(Params,Orb,Make_Orb_NN=true)

    # Precalculate chunks of matrix elements & bytes to skip ...
    N_Chunk, N_Chunk_skip = V2b_count(N_max_Int,N_2max_Int,Orb)

    # Size of the reading buffer ... has to be a multiple of 4 (4 matrix elements per J are stored together) ...
    Buffer_size = 1000000

    # Read NN interaction matrix elements ...
    println("\nReading 2-body NN interaction file...")
    @inbounds Threads.@threads for a in 1:a_max
        n_a = Orb[a].n
        l_a = Orb[a].l
        j_a = Orb[a].j
        N_a = 2*n_a + l_a
        open(Int_File, "r") do Bin_Read
            seek(Bin_Read, N_Chunk_skip[a])

            # The chunk of the orbital a is read by blocks of at most Buffer_size matrix elements ...
            N_left = N_Chunk[a]
            Buffer = Vector{Float64}(undef, min(N_left, Buffer_size))
            n = length(Buffer)
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
                                    # Read next block of the chunk, if the buffer was used up ...
                                    if n == length(Buffer)
                                        resize!(Buffer, min(N_left, Buffer_size))
                                        read!(Bin_Read, Buffer)
                                        N_left -= length(Buffer)
                                        n = 0
                                    end

                                    ME_00 = Buffer[n+1]
                                    ME_nn = Buffer[n+2]
                                    ME_10 = Buffer[n+3]
                                    ME_pp = Buffer[n+4]
                                    n += 4

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

    return N_Chunk, N_Chunk_skip
end

function V2b_parse(Params::Parameters,N_max_Out::Int64,N_2max_Out::Int64,Out_File::String)
    # Read parameters
        # Interaction file path
    In_File = Params.Int.NN_File
        # Interaction configuration space ...
    N_max_In = Params.Int.Nmax
    N_2max_In = Params.Int.N2max
    a_max_Out = div((N_max_Out+1)*(N_max_Out+2),2)

    # Check that the output space is contained in the interaction space ...
    if (N_max_Out > N_max_In) || (N_2max_Out > N_2max_In)
        error("Output space (" * string(N_max_Out) * "," * string(N_2max_Out) * ") is not contained in the interaction space (" *
              string(N_max_In) * "," * string(N_2max_In) * ") ...")
    end
    if abspath(In_File) == abspath(Out_File)
        error("Output file has to differ from the interaction file ...")
    end

    # Initialize orbitals for 2-body NN interaction read ...
    Orb = orbitals_make(Params,Int = true)

    # Initialize the buffer size (# of Float64 matrix elements read/written at once by each thread) ...
    Buffer_size = 1000000

    # Precount NN interaction matrix elements of both spaces ... the a chunks of the smaller space are the first ones ...
    println("\nPrecounting # of 2-body NN interaction matrix elements ...")
    N_Chunk_In, N_Chunk_skip_In = V2b_count(N_max_In,N_2max_In,Orb)
    N_Chunk_Out, N_Chunk_skip_Out = V2b_count(N_max_Out,N_2max_Out,Orb)

    # Check the size of the interaction file ...
    Size_In = 8*sum(N_Chunk_In)
    Size_Out = 8*sum(N_Chunk_Out)
    if filesize(In_File) != Size_In
        error("Size of " * In_File * " (" * string(filesize(In_File)) * " B) does not match the interaction space (" * string(Size_In) * " B) ...")
    end

    # Prepare the output file of the final size ...
    open(Out_File, "w") do Out_IO
        truncate(Out_IO,Size_Out)
    end

    # Queue the a chunks of the output space ... the largest first to balance the threads ...
    Queue = Channel{Int64}(a_max_Out)
    @inbounds for a in sortperm(N_Chunk_Out, rev = true)
        put!(Queue,a)
    end
    close(Queue)

    # Copy NN interaction matrix elements ... each thread streams its chunks through its own small buffers ...
    println("\nExporting 2-body NN interaction file into N_max = " * string(N_max_Out) * " , N_2max = " * string(N_2max_Out) * " ...")
    @sync for thread in 1:Threads.nthreads()
        Threads.@spawn begin
            Buffer_In = Vector{Float64}(undef,Buffer_size)
            Buffer_Out = Vector{Float64}(undef,Buffer_size)
            open(In_File, "r") do In_IO
            open(Out_File, "r+") do Out_IO
                for a in Queue
                    N_skip_In, N_In = N_Chunk_skip_In[a], N_Chunk_In[a]
                    N_skip_Out, N_Out = N_Chunk_skip_Out[a], N_Chunk_Out[a]

                    n_a = Orb[a].n
                    l_a = Orb[a].l
                    j_a = Orb[a].j
                    N_a = 2*n_a + l_a

                    # Chunk_i ... index of the matrix element within the input a chunk, the Buffer_In holds the chunk elements
                    # (Buffer_0 + 1):(Buffer_0 + Buffer_N) ... Out_i ... # of matrix elements in Buffer_Out ...
                    Chunk_i = 0
                    Buffer_0, Buffer_N = 0, 0
                    Out_i, Out_N = 0, 0
                    seek(Out_IO,N_skip_Out)

                    @inbounds for b in 1:a
                        n_b = Orb[b].n
                        l_b = Orb[b].l
                        j_b = Orb[b].j
                        N_b = 2*n_b + l_b
                        N_ab = N_a + N_b
                        if N_ab <= N_2max_In
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
                                    if (N_cd <= N_2max_In) && (rem(l_a + l_b,2) == rem(l_c + l_d,2))
                                        J_min = div(max(abs(j_a - j_b), abs(j_c - j_d)),2)
                                        J_max = div(min(j_a + j_b, j_c + j_d),2)

                                        if (N_a <= N_max_Out) && (N_b <= N_max_Out) && (N_c <= N_max_Out) && (N_d <= N_max_Out) &&
                                           (N_ab <= N_2max_Out) && (N_cd <= N_2max_Out)

                                            # Copy the 4 matrix elements (00,nn,10,pp) per J ...
                                            @inbounds for _ in 1:4*max(J_max - J_min + 1,0)
                                                Chunk_i += 1

                                                # Refill the input buffer from the file once Chunk_i is out of it ...
                                                if Chunk_i > Buffer_0 + Buffer_N
                                                    Buffer_0 = Chunk_i - 1
                                                    Buffer_N = min(Buffer_size,N_In - Buffer_0)
                                                    resize!(Buffer_In,Buffer_N)
                                                    seek(In_IO,N_skip_In + 8*Buffer_0)
                                                    read!(In_IO,Buffer_In)
                                                end

                                                # Flush the output buffer into the file once it is full ...
                                                if Out_i == Buffer_size
                                                    write(Out_IO,Buffer_Out)
                                                    Out_i = 0
                                                end

                                                Out_i += 1
                                                Out_N += 1
                                                Buffer_Out[Out_i] = Buffer_In[Chunk_i - Buffer_0]
                                            end
                                        else
                                            # Skip the elements outside the output space ... 4 matrix elements per J ...
                                            Chunk_i += 4*max(J_max - J_min + 1,0)
                                        end

                                    end
                                end
                            end
                        end
                    end

                    # Flush the rest of the output buffer ...
                    write(Out_IO,view(Buffer_Out,1:Out_i))

                    # Check the # of matrix elements in the chunks ...
                    if (Chunk_i != N_In) || (Out_N != N_Out)
                        error("Wrong # of matrix elements in the chunk a = " * string(a) * " ...")
                    end
                end
            end
            end
        end
    end

    println("\nFinished export of 2-body NN interaction into " * Out_File * " ...")

    return
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
                        J_hat =  hat(2*J)
                        Sum += 0.25 * J_hat * O2b_pp(a,b,c,d,J,P,V_NN,Orb,Orb_NN) * O2b_pp(a,b,c,d,J,P,Sigma,Orb,Orb_NN)
                        Sum += 0.25 * J_hat * O2b_nn(a,b,c,d,J,P,V_NN,Orb,Orb_NN) * O2b_nn(a,b,c,d,J,P,Sigma,Orb,Orb_NN)
                        Sum += J_hat * O2b_pn(a,b,c,d,J,P,V_NN,Orb_NN) * O2b_pn(a,b,c,d,J,P,Sigma,Orb_NN)
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