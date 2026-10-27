function V3b_no2b_initialize(N_max::Int64,N_2max::Int64,N_3max::Int64,Orb::Vector{Orb1B})
    # Read parameters ...
    a_max = div((N_max + 1)*(N_max + 2),2)
    l_max = N_max
    J_max = N_2max + 1

    # Initiliaze array for counting of NNN orbitals ...
    N_Orb_NNN = Array{Vector}(undef,2,l_max+1,J_max+1,2)

    # Initiliaze dictionary for NN orbitals ...
    Orb_NNN_Dic = Dict{UInt64,Int64}()

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


    # Allocate the dictionary and indices for NNN orbitals ...
   @inbounds for a in 1:a_max
        n_a, l_a, j_a = Orb[a].n, Orb[a].l, Orb[a].j
        if (2*n_a + l_a) <= N_max
            @inbounds for b in 1:a
                n_b = Orb[b].n
                l_b = Orb[b].l
                j_b = Orb[b].j
                if (2*(n_a + n_b) + l_a + l_b) <= N_2max
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
                                        #key = (Int8(P),Int8(J),Int8(2*T-1),Int8(T_ab),Int16(a),Int16(b),Int8(l_c),Int8(j_c),Int8(n_c))
                                        T_tot = 2*T-1
                                        Key = V3b_no2b_key(a,b,n_c,l_c,j_c,J,P,T_ab,T_tot)
                                        if j_c == Int64(abs(2*l_c-1))
                                            N_Orb_NNN[P,l_c+1,J+1,T][1] += 1
                                            Orb_NNN_Dic[Key] = N_Orb_NNN[P,l_c+1,J+1,T][1]
                                        elseif j_c == Int64(2*l_c+1)
                                            N_Orb_NNN[P,l_c+1,J+1,T][2] += 1
                                            Orb_NNN_Dic[Key] = N_Orb_NNN[P,l_c+1,J+1,T][2]
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

    Orb_NNN = Orb3B(Orb_NNN_Dic,N_Orb_NNN)

    return V_NNN, Orb_NNN
end

@inline function V3b_no2b_key(a::Int64,b::Int64,n_c::Int64,l_c::Int64,j_c::Int64,J::Int64,P::Int64,T_ab::Int64,T::Int64)
    return UInt64(J) |
          (UInt64(P) << 6) |
          (UInt64(T_ab) << 8) |
          (UInt64(T) << 10) |
          (UInt64(a) << 12) |
          (UInt64(b) << 20) |
          (UInt64(n_c) << 28) |
          (UInt64(l_c) << 34) |
          (UInt64(j_c) << 40)
end

@inline function V3b_no2b_index(a::Int64,b::Int64,c::Int64,T_ab::Int64,d::Int64,e::Int64,f::Int64,T_de::Int64,J::Int64,T::Int64,P::Int64,Orb::Vector{Orb1B},Orb_NNN::Orb3B)
    n_c = Orb[c].n
    l_c = Orb[c].l
    j_c = Orb[c].j
    n_f = Orb[f].n
    Bra_key = V3b_no2b_key(a,b,n_c,l_c,j_c,J,P,T_ab,T)
    Ket_key = V3b_no2b_key(d,e,n_f,l_c,j_c,J,P,T_de,T)
    Bra = Orb_NNN.Dic[Bra_key]
    Ket = Orb_NNN.Dic[Ket_key]
    if Bra < Ket
        Bra, Ket = Ket, Bra
    end
    T_ind = div(T+1,2)
    j_index = (j_c == abs(2*l_c - 1)) ? 1 : 2
    N = Orb_NNN.N[P, l_c+1, J+1, T_ind][j_index]
    Ind = Bra + (Ket - 1)*N - div(Ket * (Ket - 1),2)
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

function V3b_no2b_read(Params::Parameters,Orb::Vector{Orb1B})
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

    Int = Mmap.mmap(Int_File)

    @inline function V3b_no2b_read_chunk(Int::Vector{UInt8},N_skip::Int64,N::Int64)
        Bytes = @view Int[(N_skip + 1):(N_skip + 4*N)]
        return reinterpret(Float32,Bytes)
    end

    # Initialize the buffer
    Buffer_size = 1000000

    # Initialize NNN interaction arrays
    println("\nPreparing 3-body NNN interaction array ...")
    V_NNN, Orb_NNN = V3b_no2b_initialize(N_max_Calc,N_2max_Calc,N_3max_Calc,Orb)

    # Precount NNN interaction matrix elements
    println("\nPrecounting # of 3-body NNN interaction matrix elements ...")
    N_chunk_skip, N_chunk, ab, ab_max = V3b_no2b_count(N_max_Int,N_2max_Int,N_3max_Int,Orb)

    # Read NNN interaction from designated binary
    println("\nReading 3-body NO2B NNN interaction file. ..")
    @inbounds Threads.@threads for ab_i in 1:ab_max
        a, b = ab[ab_i,1], ab[ab_i,2]
        n_a, l_a , j_a = Orb[a].n, Orb[a].l, Orb[a].j
        n_b, l_b , j_b = Orb[b].n, Orb[b].l, Orb[b].j
        N_a , N_b = 2*n_a + l_a, 2*n_b + l_b
        N_ab = N_a + N_b

        N_skip = N_chunk_skip[ab_i]
        N = N_chunk[ab_i]
        Buffer_N = min(N,Buffer_size)
        Buffer_i = 1
        Buffer = Float32[]

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
                                        @inbounds for J in div(max(abs(j_a - j_b),abs(j_d - j_e)),2):div(min((j_a + j_b),(j_d + j_e)),2)
                                            @inbounds for T_ab in 0:1
                                                @inbounds for T_de in 0:1
                                                    @inbounds for T in max(abs(2*T_ab-1),abs(2*T_de-1)):2:min((2*T_ab+1),(2*T_de+1))

                                                        if (Buffer_i > Buffer_N) || (Buffer_i == 1)
                                                            Buffer_N = min(N,Buffer_size)
                                                            N = N - Buffer_N
                                                            Buffer_i = 1
                                                            Buffer = V3b_no2b_read_chunk(Int,N_skip,Buffer_N)
                                                            N_skip = N_skip + 4*Buffer_N
                                                        end

                                                        V = Buffer[Buffer_i]

                                                        Buffer_i += 1

                                                        if (N_a <= N_max_Calc) && (N_b <= N_max_Calc) && (N_c <= N_max_Calc) &&
                                                            (N_d <= N_max_Calc) && (N_e <= N_max_Calc) && (N_f <= N_max_Calc) &&
                                                            (N_ab <= N_2max_Calc) && (N_ac <= N_2max_Calc) && (N_bc <= N_2max_Calc) &&
                                                            (N_de <= N_2max_Calc) && (N_df <= N_2max_Calc) && (N_ef <= N_2max_Calc) &&
                                                            (N_abc <= N_3max_Calc) && (N_def <= N_3max_Calc)

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

    # Deallocate NNN interaction memory map Int ...
    Int = nothing

    # Perform the Garbage collection ...
    GC.gc()

    println("\nFinished loading of 3-body NO2B NNN interaction ...")

    return V_NNN, Orb_NNN
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
                                        @inbounds for J in div(max(abs(j_a - j_b),abs(j_d - j_e)),2):div(min((j_a + j_b),(j_d + j_e)),2)
                                            @inbounds for T_ab in 0:1
                                                @inbounds for T_de in 0:1
                                                    @inbounds for T in max(abs(2*T_ab-1),abs(2*T_de-1)):2:min((2*T_ab+1),(2*T_de+1))
                                                        N += 1
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
        end
        N_chunk[ab_i] = N
    end

    # Allocate N_chunk_skip
    @inbounds for ab_i in 1:ab_max
        Sum = 0
        @inbounds for ab_j in 1:(ab_i-1)
            Sum += 4 * N_chunk[ab_j]
        end
        N_chunk_skip[ab_i] = Sum
    end

    return N_chunk_skip, N_chunk, ab, ab_max
end