function V3b_no2b_initialize(N_max::Int64,N_2max::Int64,N_3max::Int64,Orb::Vector{Orb1B})
    # Read parameters ...
    a_max = div((N_max + 1)*(N_max + 2),2)
    l_max = N_max
    J_max = N_2max + 1

    # Initiliaze array for counting of NNN orbitals ...
    N_Orb_NNN = Array{Vector{Int64}}(undef,2,l_max+1,J_max+1,2)

    # Initiliaze the dense lookup for NNN orbitals ... Map[ab][TT,J-J_min+1,c] (see Orb3B) ...
    Orb_NNN_Map = Vector{Array{Int32,3}}(undef,div(a_max*(a_max+1),2))
    @inbounds for a in 1:a_max
        @inbounds for b in 1:a
            N_ab = 2*(Orb[a].n + Orb[b].n) + Orb[a].l + Orb[b].l
            J_count = (N_ab <= N_2max) ? div(Orb[a].j + Orb[b].j,2) - div(abs(Orb[a].j - Orb[b].j),2) + 1 : 0
            Orb_NNN_Map[div(a*(a-1),2) + b] = zeros(Int32,3,J_count,a_max)
        end
    end

    # Initiliaze the lookup of orbital indices from (n,l,j) ...
    Orb_nlj = zeros(Int64,div(N_max,2)+1,l_max+1,2)
    @inbounds for a in 1:a_max
        n_a, l_a, j_a = Orb[a].n, Orb[a].l, Orb[a].j
        Orb_nlj[n_a+1,l_a+1,(j_a == abs(2*l_a - 1)) ? 1 : 2] = a
    end

    # Count the number of NNN orbitals ...
    @inbounds for P in 1:2
        @inbounds for l in 0:l_max
            @inbounds for J in 0:J_max
                @inbounds for T in 1:2
                    j_count = 0
                    @inbounds for j in Int64(abs(2*l-1)):2:Int64(2*l+1)
                        j_count += 1
                    end
                    N_Orb_NNN[P,l+1,J+1,T] = zeros(Int64,j_count)
                end
            end
        end
    end

    # Allocate the dense lookup and indices for NNN orbitals ...
   @inbounds for a in 1:a_max
        n_a, l_a, j_a = Orb[a].n, Orb[a].l, Orb[a].j
        if (2*n_a + l_a) <= N_max
            @inbounds for b in 1:a
                n_b = Orb[b].n
                l_b = Orb[b].l
                j_b = Orb[b].j
                if (2*(n_a + n_b) + l_a + l_b) <= N_2max
                    Map_ab = Orb_NNN_Map[div(a*(a-1),2) + b]
                    J_min = div(abs(j_a - j_b),2)
                    @inbounds for c = 1:a_max
                        n_c = Orb[c].n
                        l_c = Orb[c].l
                        j_c = Orb[c].j
                        P = rem(l_a + l_b + l_c,2) + 1
                        if (2*(n_a + n_b + n_c) + l_a + l_b + l_c) <= N_3max &&
                           (2*(n_a + n_c) + l_a + l_c) <= N_2max &&
                           (2*(n_b + n_c) + l_b + l_c) <= N_2max
                           @inbounds for J = Int64(abs(j_a - j_b)/2):Int64((j_a + j_b)/2)
                                @inbounds for T in 1:2
                                    @inbounds for T_ab in (T-1):1
                                        # Dense lookup instead of Orb_NNN_Dic[V3b_no2b_key(a,b,n_c,l_c,j_c,J,P,T_ab,2*T-1)] ...
                                        if j_c == Int64(abs(2*l_c-1))
                                            N_Orb_NNN[P,l_c+1,J+1,T][1] += 1
                                            Map_ab[T_ab + T,J - J_min + 1,c] = N_Orb_NNN[P,l_c+1,J+1,T][1]
                                        elseif j_c == Int64(2*l_c+1)
                                            N_Orb_NNN[P,l_c+1,J+1,T][2] += 1
                                            Map_ab[T_ab + T,J - J_min + 1,c] = N_Orb_NNN[P,l_c+1,J+1,T][2]
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

    # Initialize arrays for 3-body interaction operator V_NNN ...
    V_NNN = Array{Vector{Vector{Float32}}}(undef,2,l_max+1,J_max+1,2)

    @inbounds for P = 1:2
        @inbounds for l = 0:l_max
            @inbounds for J = 0:J_max
                @inbounds for T = 1:2
                    j_count = 0
                    @inbounds for j in Int64(abs(2*l-1)):2:Int64(2*l+1)
                        j_count += 1
                    end

                    V_NNN[P,l+1,J+1,T] = Vector{Vector{Float32}}(undef,j_count)

                    j_count = 0
                    @inbounds for j in Int64(abs(2*l-1)):2:Int64(2*l+1)
                        j_count += 1

                        N_count = N_Orb_NNN[P,l+1,J+1,T][j_count]
                        Length = div(N_count*(N_count+1),2)
                        V_NNN[P,l+1,J+1,T][j_count] = zeros(Float32, Length)
                    end
                end
            end
        end
    end

    Orb_NNN = Orb3B(Orb_NNN_Map,N_Orb_NNN,Orb_nlj)

    return V_NNN, Orb_NNN
end

function V3b_no2b_parse(Params::Parameters,N_max_Out::Int64,N_2max_Out::Int64,N_3max_Out::Int64,Out_File::String)
    # Read parameters
        # Interaction file path
    In_File = Params.Int.NNN_File
        # Interaction configuration space ...
    N_max_In = Params.Int.Nmax
    N_2max_In = Params.Int.N2max
    N_3max_In = Params.Int.N3max
    a_max = div((N_max_In+1)*(N_max_In+2),2)

    # Check that the output space is contained in the interaction space ...
    if (N_max_Out > N_max_In) || (N_2max_Out > N_2max_In) || (N_3max_Out > N_3max_In)
        error("Output space (" * string(N_max_Out) * "," * string(N_2max_Out) * "," * string(N_3max_Out) *
              ") is not contained in the interaction space (" * string(N_max_In) * "," * string(N_2max_In) * "," * string(N_3max_In) * ") ...")
    end
    if abspath(In_File) == abspath(Out_File)
        error("Output file has to differ from the interaction file ...")
    end

    # Initialize orbitals for 3-body NNN interaction read ...
    Orb = orbitals_make(Params,Int = true)

    # Initialize the buffer size (# of Float32 matrix elements read/written at once by each thread) ...
    Buffer_size = 1000000

    # Precount NNN interaction matrix elements of both spaces ...
    println("\nPrecounting # of 3-body NNN interaction matrix elements ...")
    N_chunk_skip_In, N_chunk_In, ab_In, ab_max_In = V3b_no2b_count(N_max_In,N_2max_In,N_3max_In,Orb)
    N_chunk_skip_Out, N_chunk_Out, ab_Out, ab_max_Out = V3b_no2b_count(N_max_Out,N_2max_Out,N_3max_Out,Orb)

    # Check the size of the interaction file ...
    Size_In = N_chunk_skip_In[ab_max_In] + 4*N_chunk_In[ab_max_In]
    Size_Out = N_chunk_skip_Out[ab_max_Out] + 4*N_chunk_Out[ab_max_Out]
    if filesize(In_File) != Size_In
        error("Size of " * In_File * " (" * string(filesize(In_File)) * " B) does not match the interaction space (" * string(Size_In) * " B) ...")
    end

    # Find the input chunk of each output ab chunk ... the orbitals of the smaller space are the first ones, so the ab chunks follow in the same order ...
    ab_In_index = Vector{Int64}(undef,ab_max_Out)
    ab_i = 1
    @inbounds for ab_o in 1:ab_max_Out
        while (ab_In[ab_i,1] != ab_Out[ab_o,1]) || (ab_In[ab_i,2] != ab_Out[ab_o,2])
            ab_i += 1
        end
        ab_In_index[ab_o] = ab_i
    end

    # Prepare the output file of the final size ...
    open(Out_File, "w") do Out_IO
        truncate(Out_IO,Size_Out)
    end

    # Queue the ab chunks of the output space ... the largest first to balance the threads ...
    Queue = Channel{Int64}(ab_max_Out)
    @inbounds for ab_o in sortperm(N_chunk_Out, rev = true)
        put!(Queue,ab_o)
    end
    close(Queue)

    # Copy NNN interaction matrix elements ... each thread streams its chunks through its own small buffers ...
    println("\nExporting 3-body NO2B NNN interaction file into N_max = " * string(N_max_Out) * " , N_2max = " * string(N_2max_Out) *
            " , N_3max = " * string(N_3max_Out) * " ...")
    @sync for thread in 1:Threads.nthreads()
        Threads.@spawn begin
            Buffer_In = Vector{Float32}(undef,Buffer_size)
            Buffer_Out = Vector{Float32}(undef,Buffer_size)
            open(In_File, "r") do In_IO
            open(Out_File, "r+") do Out_IO
                for ab_o in Queue
                    a, b = ab_Out[ab_o,1], ab_Out[ab_o,2]
                    N_skip_In, N_In = N_chunk_skip_In[ab_In_index[ab_o]], N_chunk_In[ab_In_index[ab_o]]
                    N_skip_Out, N_Out = N_chunk_skip_Out[ab_o], N_chunk_Out[ab_o]

                    n_a, l_a , j_a = Orb[a].n, Orb[a].l, Orb[a].j
                    n_b, l_b , j_b = Orb[b].n, Orb[b].l, Orb[b].j
                    N_a , N_b = 2*n_a + l_a, 2*n_b + l_b
                    N_ab = N_a + N_b

                    # Chunk_i ... index of the matrix element within the input ab chunk, the Buffer_In holds the chunk elements
                    # (Buffer_0 + 1):(Buffer_0 + Buffer_N) ... Out_i ... # of matrix elements in Buffer_Out ...
                    Chunk_i = 0
                    Buffer_0, Buffer_N = 0, 0
                    Out_i, Out_N = 0, 0
                    seek(Out_IO,N_skip_Out)

                    @inbounds for c in 1:a_max
                        n_c = Orb[c].n
                        l_c = Orb[c].l
                        j_c = Orb[c].j
                        N_c = 2*n_c + l_c
                        N_ac = N_a + N_c
                        N_bc = N_b + N_c
                        N_abc = N_a + N_b + N_c
                        if (N_ac <= N_2max_In) && (N_bc <= N_2max_In) && (N_abc <= N_3max_In)
                            P_abc = rem(l_a + l_b + l_c,2) + 1
                            @inbounds for d in 1:a
                                n_d = Orb[d].n
                                l_d = Orb[d].l
                                j_d = Orb[d].j
                                N_d = 2*n_d + l_d
                                @inbounds for e in 1:d
                                    n_e = Orb[e].n
                                    l_e = Orb[e].l
                                    j_e = Orb[e].j
                                    N_e = 2*n_e + l_e
                                    N_de = N_d + N_e
                                    if (N_de <= N_2max_In)
                                        @inbounds for f in 1:a_max
                                            n_f = Orb[f].n
                                            l_f = Orb[f].l
                                            j_f = Orb[f].j
                                            N_f = 2*n_f + l_f
                                            N_df = N_d + N_f
                                            N_ef = N_e + N_f
                                            N_def = N_d + N_e + N_f
                                            if (j_c == j_f) && (l_c == l_f) && (N_df <= N_2max_In) && (N_ef <= N_2max_In) && (N_def <= N_3max_In)
                                                P_def = rem(l_d + l_e + l_f,2) + 1
                                                if (P_abc == P_def)
                                                    J_min = div(max(abs(j_a - j_b),abs(j_d - j_e)),2)
                                                    J_max = div(min((j_a + j_b),(j_d + j_e)),2)

                                                    if (N_a <= N_max_Out) && (N_b <= N_max_Out) && (N_c <= N_max_Out) &&
                                                       (N_d <= N_max_Out) && (N_e <= N_max_Out) && (N_f <= N_max_Out) &&
                                                       (N_ab <= N_2max_Out) && (N_ac <= N_2max_Out) && (N_bc <= N_2max_Out) &&
                                                       (N_de <= N_2max_Out) && (N_df <= N_2max_Out) && (N_ef <= N_2max_Out) &&
                                                       (N_abc <= N_3max_Out) && (N_def <= N_3max_Out)

                                                        # Copy the 5 (T_ab,T_de,T) combinations per J ...
                                                        @inbounds for _ in 1:5*max(J_max - J_min + 1,0)
                                                            Chunk_i += 1

                                                            # Refill the input buffer from the file once Chunk_i is out of it ...
                                                            if Chunk_i > Buffer_0 + Buffer_N
                                                                Buffer_0 = Chunk_i - 1
                                                                Buffer_N = min(Buffer_size,N_In - Buffer_0)
                                                                resize!(Buffer_In,Buffer_N)
                                                                seek(In_IO,N_skip_In + 4*Buffer_0)
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
                                                        # Skip the elements outside the output space ... 5 (T_ab,T_de,T) combinations per J ...
                                                        Chunk_i += 5*max(J_max - J_min + 1,0)
                                                    end

                                                end
                                            end
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
                        error("Wrong # of matrix elements in the chunk (a,b) = (" * string(a) * "," * string(b) * ") ...")
                    end
                end
            end
            end
        end
    end

    println("\nFinished export of 3-body NO2B NNN interaction into " * Out_File * " ...")

    return
end

@noinline V3b_no2b_throw(Key::Tuple) = throw(KeyError(Key))

@inline function V3b_no2b_lookup(a::Int64,b::Int64,c::Int64,T_ab::Int64,J::Int64,T_ind::Int64,Orb::Vector{Orb1B},Orb_NNN::Orb3B)
    # Only (a >= b) & (T_ab,T) != (0,3/2) are stored ... 0 = not allowed ...
    ab = div(a*(a-1),2) + b
    if b > a || b < 1 || ab > length(Orb_NNN.Map) || T_ab < T_ind - 1 || T_ab > 1
        return 0
    end
    Map_ab = Orb_NNN.Map[ab]
    J_min = div(abs(Orb[a].j - Orb[b].j),2)
    if J < J_min || (J - J_min + 1) > size(Map_ab,2) || c < 1 || c > size(Map_ab,3)
        return 0
    end
    return Int64(Map_ab[T_ab + T_ind,J - J_min + 1,c])
end

@inline function V3b_no2b_index(a::Int64,b::Int64,c::Int64,T_ab::Int64,d::Int64,e::Int64,f::Int64,T_de::Int64,J::Int64,T::Int64,P::Int64,Orb::Vector{Orb1B},Orb_NNN::Orb3B)
    T_ind = div(T+1,2)
    # Dense lookup instead of Orb_NNN.Dic[V3b_no2b_key(...)] ...
    Bra = V3b_no2b_lookup(a,b,c,T_ab,J,T_ind,Orb,Orb_NNN)
    if Bra == 0
        V3b_no2b_throw((a,b,c,T_ab,J,T,P))
    end
    l_c = Orb[c].l
    j_c = Orb[c].j
    j_index = (j_c == abs(2*l_c - 1)) ? 1 : 2
    # The Ket orbital is (n_f,l_c,j_c) ... f itself unless (l_f,j_f) != (l_c,j_c) ...
    n_f = Orb[f].n
    if (Orb[f].l == l_c) && (Orb[f].j == j_c)
        f_c = f
    else
        f_c = (n_f + 1 <= size(Orb_NNN.nlj,1)) ? Orb_NNN.nlj[n_f+1,l_c+1,j_index] : 0
    end
    Ket = (f_c > 0) ? V3b_no2b_lookup(d,e,f_c,T_de,J,T_ind,Orb,Orb_NNN) : 0
    P_abc, P_def = rem(Orb[a].l + Orb[b].l + l_c,2) + 1, rem(Orb[d].l + Orb[e].l + l_c,2) + 1
    if Ket == 0 || P != P_abc || P != P_def
        V3b_no2b_throw((a,b,c,T_ab,d,e,f,T_de,J,T,P))
    end
    if Bra < Ket
        Bra, Ket = Ket, Bra
    end
    # Packed lower triangle stored row by row (Bra >= Ket) ... Kets of the same Bra are contiguous in memory ...
    Ind = Ket + div(Bra * (Bra - 1),2)
    return Ind
end

@inline function V3b_no2b(a::Int64,b::Int64,c::Int64,T_ab::Int64,d::Int64,e::Int64,f::Int64,T_de::Int64,J_de::Int64,T::Int64,P::Int64,V_NNN::Array{Vector{Vector{Float32}},4},Orb::Vector{Orb1B},Orb_NNN::Orb3B)
    if a >= b
        if d >= e
            Ind = V3b_no2b_index(a,b,c,T_ab,d,e,f,T_de,J_de,T,P,Orb,Orb_NNN)
            phase = 1.0
        else
            j_d = Orb[d].j
            j_e = Orb[e].j
            Ind = V3b_no2b_index(a,b,c,T_ab,e,d,f,T_de,J_de,T,P,Orb,Orb_NNN)
            exp = div(j_d + j_e,2) - J_de - T_de
            phase = isodd(exp) ? -1.0 : 1.0
        end
    else
        if d >= e
            j_a = Orb[a].j
            j_b = Orb[b].j
            Ind = V3b_no2b_index(b,a,c,T_ab,d,e,f,T_de,J_de,T,P,Orb,Orb_NNN)
            exp = div(j_a + j_b,2) - J_de - T_ab
            phase = isodd(exp) ? -1.0 : 1.0
        else
            j_a = Orb[a].j
            j_b = Orb[b].j
            j_d = Orb[d].j
            j_e = Orb[e].j
            Ind = V3b_no2b_index(b,a,c,T_ab,e,d,f,T_de,J_de,T,P,Orb,Orb_NNN)
            exp = div(j_a + j_b + j_d + j_e,2) - T_de - T_ab
            phase = isodd(exp) ? -1.0 : 1.0
        end
    end
    T_ind = div((T+1),2)
    l_c = Orb[c].l
    j_c = Orb[c].j
    if j_c == abs(2*l_c - 1)
        return phase * Float64(V_NNN[P,l_c+1,J_de+1,T_ind][1][Ind])
    else
        return phase * Float64(V_NNN[P,l_c+1,J_de+1,T_ind][2][Ind])
    end
end

function V3b_no2b_read(Params::Parameters)
    # Read parameters
        # Interaction file path
    Int_File = Params.Int.NNN_File
        # Interaction configuration space...
    N_max_Int = Params.Int.Nmax
    N_2max_Int = Params.Int.N2max
    N_3max_Int = Params.Int.N3max
    a_max = div((N_max_Int+1)*(N_max_Int+2),2)
        # Calculation configuration space
    N_max_Calc = Params.Calc.Nmax
    N_2max_Calc = Params.Calc.N2max
    N_3max_Calc = Params.Calc.N3max

    # Initialize orbitals for 2-body NN interaction read ...
    Orb = orbitals_make(Params,Int = true)

    # Initialize the buffer size (# of Float32 matrix elements read at once by each thread) ...
    Buffer_size = 1000000

    # Initialize NNN interaction arrays
    println("\nPreparing 3-body NNN interaction array ...")
    V_NNN, Orb_NNN = V3b_no2b_initialize(N_max_Calc,N_2max_Calc,N_3max_Calc,Orb)

    # Precount NNN interaction matrix elements
    println("\nPrecounting # of 3-body NNN interaction matrix elements ...")
    N_chunk_skip, N_chunk, ab, ab_max = V3b_no2b_count(N_max_Int,N_2max_Int,N_3max_Int,Orb)

    # Queue the ab chunks needed in the calculation space ... the largest first to balance the threads ...
    Queue = Channel{Int64}(ab_max)
    @inbounds for ab_i in sortperm(N_chunk, rev = true)
        a, b = ab[ab_i,1], ab[ab_i,2]
        N_a , N_b = 2*Orb[a].n + Orb[a].l, 2*Orb[b].n + Orb[b].l
        if (N_a <= N_max_Calc) && (N_b <= N_max_Calc) && ((N_a + N_b) <= N_2max_Calc)
            put!(Queue,ab_i)
        end
    end
    close(Queue)

    # Read NNN interaction from designated binary ... each thread streams its chunks through its own small buffer ...
    println("\nReading 3-body NO2B NNN interaction file. ..")
    @sync for thread in 1:Threads.nthreads()
        Threads.@spawn begin
            Buffer = Vector{Float32}(undef,Buffer_size)
            open(Int_File, "r") do Int_IO
                for ab_i in Queue
                    V3b_no2b_read_chunk!(V_NNN,Orb_NNN,Orb,Params,Int_IO,Buffer,Buffer_size,ab[ab_i,1],ab[ab_i,2],N_chunk_skip[ab_i],N_chunk[ab_i])
                end
            end
        end
    end

    # Perform the Garbage collection ...
    GC.gc()

    println("\nFinished loading of 3-body NO2B NNN interaction ...")

    return V_NNN, Orb_NNN
end

function V3b_no2b_read_chunk!(V_NNN::Array{Vector{Vector{Float32}},4},Orb_NNN::Orb3B,Orb::Vector{Orb1B},Params::Parameters,
                              Int_IO::IOStream,Buffer::Vector{Float32},Buffer_size::Int64,a::Int64,b::Int64,N_skip::Int64,N::Int64)
    # Read parameters
        # Interaction configuration space...
    N_max_Int = Params.Int.Nmax
    N_2max_Int = Params.Int.N2max
    N_3max_Int = Params.Int.N3max
    a_max = div((N_max_Int+1)*(N_max_Int+2),2)
        # Calculation configuration space
    N_max_Calc = Params.Calc.Nmax
    N_2max_Calc = Params.Calc.N2max
    N_3max_Calc = Params.Calc.N3max

    n_a, l_a , j_a = Orb[a].n, Orb[a].l, Orb[a].j
    n_b, l_b , j_b = Orb[b].n, Orb[b].l, Orb[b].j
    N_a , N_b = 2*n_a + l_a, 2*n_b + l_b
    N_ab = N_a + N_b

    # Chunk_i ... index of the matrix element within the ab chunk (starting N_skip bytes into the file),
    # the Buffer holds the chunk elements (Buffer_0 + 1):(Buffer_0 + Buffer_N) ...
    Chunk_i = 0
    Buffer_0, Buffer_N = 0, 0

    @inbounds for c in 1:a_max
        n_c = Orb[c].n
        l_c = Orb[c].l
        j_c = Orb[c].j
        N_c = 2*n_c + l_c
        N_ac = N_a + N_c
        N_bc = N_b + N_c
        N_abc = N_a + N_b + N_c
        P = rem(l_a + l_b + l_c,2) + 1
        if (N_ac<= N_2max_Int) && (N_bc <= N_2max_Int) && (N_abc <= N_3max_Int)
            P_abc = rem(l_a + l_b + l_c,2) + 1
            @inbounds for d in 1:a
                n_d = Orb[d].n
                l_d = Orb[d].l
                j_d = Orb[d].j
                N_d = 2*n_d + l_d
                @inbounds for e in 1:d
                    n_e = Orb[e].n
                    l_e = Orb[e].l
                    j_e = Orb[e].j
                    N_e = 2*n_e + l_e
                    N_de = N_d + N_e
                    if (N_de <= N_2max_Int)
                        @inbounds for f in 1:a_max
                            n_f = Orb[f].n
                            l_f = Orb[f].l
                            j_f = Orb[f].j
                            N_f = 2*n_f + l_f
                            N_df = N_d + N_f
                            N_ef = N_e + N_f
                            N_def = N_d + N_e + N_f
                            if (j_c == j_f) && (l_c == l_f) && (N_df <= N_2max_Int) && (N_ef <= N_2max_Int) && (N_def <= N_3max_Int)
                                P_def = rem(l_d + l_e + l_f,2) + 1
                                if (P_abc == P_def)
                                    J_min = div(max(abs(j_a - j_b),abs(j_d - j_e)),2)
                                    J_max = div(min((j_a + j_b),(j_d + j_e)),2)

                                    if (N_a <= N_max_Calc) && (N_b <= N_max_Calc) && (N_c <= N_max_Calc) &&
                                        (N_d <= N_max_Calc) && (N_e <= N_max_Calc) && (N_f <= N_max_Calc) &&
                                        (N_ab <= N_2max_Calc) && (N_ac <= N_2max_Calc) && (N_bc <= N_2max_Calc) &&
                                        (N_de <= N_2max_Calc) && (N_df <= N_2max_Calc) && (N_ef <= N_2max_Calc) &&
                                        (N_abc <= N_3max_Calc) && (N_def <= N_3max_Calc)

                                        @inbounds for J in J_min:J_max
                                            @inbounds for T_ab in 0:1
                                                @inbounds for T_de in 0:1
                                                    @inbounds for T in max(abs(2*T_ab-1),abs(2*T_de-1)):2:min((2*T_ab+1),(2*T_de+1))

                                                        Chunk_i += 1

                                                        # Refill the buffer from the file once Chunk_i is out of it ...
                                                        if Chunk_i > Buffer_0 + Buffer_N
                                                            Buffer_0 = Chunk_i - 1
                                                            Buffer_N = min(Buffer_size,N - Buffer_0)
                                                            resize!(Buffer,Buffer_N)
                                                            seek(Int_IO,N_skip + 4*Buffer_0)
                                                            read!(Int_IO,Buffer)
                                                        end

                                                        V = Buffer[Chunk_i - Buffer_0]

                                                        T_ind = div((T+1),2)
                                                        Ind = V3b_no2b_index(a,b,c,T_ab,d,e,f,T_de,J,T,P,Orb,Orb_NNN)
                                                        if j_c == abs(2*l_c-1)
                                                            V_NNN[P,l_c+1,J+1,T_ind][1][Ind] = V
                                                        elseif j_c == (2*l_c+1)
                                                            V_NNN[P,l_c+1,J+1,T_ind][2][Ind] = V
                                                        end

                                                    end
                                                end
                                            end
                                        end
                                    else
                                        # Skip the elements outside the calculation space ... 5 (T_ab,T_de,T) combinations per J ...
                                        Chunk_i += 5*max(J_max - J_min + 1,0)
                                    end

                                end
                            end
                        end
                    end
                end
            end
        end
    end

    return nothing
end

function V3b_no2b_allocate(N_max::Int64,N_2max::Int64,Orb::Vector{Orb1B})
    # Initialize a_max, ab_count
    a_max, ab_count = div((N_max+1)*(N_max+2),2), 0

    # Evaluate ab_count
    @inbounds for a in 1:a_max
        n_a = Orb[a].n
        l_a = Orb[a].l
        N_a = 2*n_a + l_a
        if (N_a <= N_max)
            @inbounds for b in 1:a
                n_b = Orb[b].n
                l_b = Orb[b].l
                N_b = 2*n_b + l_b
                if ((N_a + N_b) <= N_2max)
                    ab_count += 1
                end
            end
        end
    end

    # Initialize ab, and restart ab_count
    ab = Matrix{Int64}(undef,ab_count,2)
    ab_count = 0

    # Allocate ab, and evaluate ab_count
    @inbounds for a in 1:a_max
        n_a = Orb[a].n
        l_a = Orb[a].l
        N_a = 2*n_a + l_a
        if (N_a <= N_max)
            @inbounds for b in 1:a
                n_b = Orb[b].n
                l_b = Orb[b].l
                N_b = 2*n_b + l_b
                if ((N_a + N_b) <= N_2max)
                    ab_count += 1
                    ab[ab_count,1] = a
                    ab[ab_count,2] = b
                end
            end
        end
    end

    return ab, ab_count
end

function V3b_no2b_count(N_max::Int64,N_2max::Int64,N_3max::Int64,Orb::Vector{Orb1B})
    # Evaluate a_max
    a_max = div((N_max+1)*(N_max+2),2)

    # Initialize ab, ab_max
    ab, ab_max = V3b_no2b_allocate(N_max,N_2max,Orb)

    # Initialize N_chunk, N_chunk_skip
    N_chunk = Vector{Int64}(undef,ab_max)
    N_chunk_skip = Vector{Int64}(undef,ab_max)

    # Allocate N_chunk
    @inbounds Threads.@threads for ab_i in 1:ab_max
        a, b = ab[ab_i,1], ab[ab_i,2]
        n_a, l_a , j_a = Orb[a].n, Orb[a].l, Orb[a].j
        n_b, l_b , j_b = Orb[b].n, Orb[b].l, Orb[b].j
        N_a , N_b = 2*n_a + l_a, 2*n_b + l_b
        N = 0
        @inbounds for c in 1:a_max
            n_c = Orb[c].n
            l_c = Orb[c].l
            j_c = Orb[c].j
            N_c = 2*n_c + l_c
            if ((N_a + N_c) <= N_2max) && ((N_b + N_c) <= N_2max) && ((N_a + N_b + N_c) <= N_3max)
                P_abc = rem(l_a + l_b + l_c,2) + 1
                @inbounds for d in 1:a
                    n_d = Orb[d].n
                    l_d = Orb[d].l
                    j_d = Orb[d].j
                    N_d = 2*n_d + l_d
                    @inbounds for e in 1:d
                        n_e = Orb[e].n
                        l_e = Orb[e].l
                        j_e = Orb[e].j
                        N_e = 2*n_e + l_e
                        if ((N_d + N_e) <= N_2max)
                            @inbounds for f in 1:a_max
                                n_f = Orb[f].n
                                l_f = Orb[f].l
                                j_f = Orb[f].j
                                N_f = 2*n_f + l_f
                                if (j_c == j_f) && (l_c == l_f) && ((N_d + N_f) <= N_2max) && ((N_e + N_f) <= N_2max) && ((N_d + N_e + N_f) <= N_3max)
                                    P_def = rem(l_d + l_e + l_f,2) + 1
                                    if (P_abc == P_def)
                                        # 5 (T_ab,T_de,T) combinations per J ... (0,0,1/2),(0,1,1/2),(1,0,1/2),(1,1,1/2),(1,1,3/2) ...
                                        J_min = div(max(abs(j_a - j_b),abs(j_d - j_e)),2)
                                        J_max = div(min((j_a + j_b),(j_d + j_e)),2)
                                        N += 5*max(J_max - J_min + 1,0)
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
        N_chunk[ab_i] = N
    end

    # Allocate N_chunk_skip ... running sum of the byte offsets ...
    Sum = 0
    @inbounds for ab_i in 1:ab_max
        N_chunk_skip[ab_i] = Sum
        Sum += 4 * N_chunk[ab_i]
    end

    return N_chunk_skip, N_chunk, ab, ab_max
end