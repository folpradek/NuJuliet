function O2b_initialize(Params::Parameters,Orb::Vector{Orb1B};Make_Orb_NN::Bool=false)
    # Read parameters ...
    N_max, N_2max = Params.Calc.Nmax, Params.Calc.N2max
    a_max = div((N_max + 1)*(N_max + 2),2)
    J_max = Int64(N_2max + 1)

    # Initiliaze array for counting of NN orbitals ...
    N_Orb_NN = zeros(Int64,2,2,J_max+1)

    # Initiliaze the dense lookup for NN orbitals ... Map[a,b,J+1,T+1] (see Orb2B) ...
    Orb_NN_Map = zeros(Int32,a_max,a_max,J_max+1,2)

    # Count the number of NN orbitals ...
    @inbounds for a in 1:a_max
        n_a, l_a, j_a = Orb[a].n, Orb[a].l, Orb[a].j
        if (2*n_a + l_a) <= N_max
            @inbounds for b = 1:a_max
                n_b, l_b, j_b = Orb[b].n, Orb[b].l, Orb[b].j
                if ((2*(n_a + n_b) + l_a + l_b) <= N_2max)
                    P = rem(l_a + l_b, 2) + 1
                    @inbounds for J = div(abs(j_a - j_b),2):div((j_a + j_b),2)
                        if b <= a
                            N_Orb_NN[2,P,J+1] += 1
                        end
                        N_Orb_NN[1,P,J+1] += 1
                    end
                end
            end
        end
    end

    # Initialite the array for NN orbital indices ...
    Ind_Orb_NN = Array{Vector{Vector{Int64}}}(undef,2,2,J_max+1)
    @inbounds for P = 1:2
        @inbounds for J = 0:J_max
            Ind_Orb_NN[1,P,J+1] = Vector{Vector{Int64}}(undef,N_Orb_NN[1,P,J+1])
            Ind_Orb_NN[2,P,J+1] = Vector{Vector{Int64}}(undef,N_Orb_NN[2,P,J+1])
        end
    end

    # Reinitialize array for counting of NN orbitals ...
    N_Orb_NN = zeros(Int64,2,2,J_max+1)

    # Allocate the dense lookup and indices for NN orbitals ...
    @inbounds for a in 1:a_max
        n_a, l_a, j_a = Orb[a].n, Orb[a].l, Orb[a].j
        if (2*n_a + l_a) <= N_max
            @inbounds for b = 1:a_max
                n_b, l_b, j_b = Orb[b].n, Orb[b].l, Orb[b].j
                l_b = Orb[b].l
                j_b = Orb[b].j
                if ((2*(n_a + n_b) + l_a + l_b) <= N_2max)
                    P = rem(l_a + l_b, 2) + 1
                    @inbounds for J = abs(div(j_a - j_b,2)):div((j_a + j_b),2)
                        if b <= a
                            # Dense lookup instead of Orb_NN_Dic[O2b_key(a,b,J,P,1)] ...
                            N_Orb_NN[2,P,J+1] += 1
                            Orb_NN_Map[a,b,J+1,2] = N_Orb_NN[2,P,J+1]
                            Ind_Orb_NN[2,P,J+1][N_Orb_NN[2,P,J+1]] = zeros(Int64, 2)
                            Ind_Orb_NN[2,P,J+1][N_Orb_NN[2,P,J+1]][1] = a
                            Ind_Orb_NN[2,P,J+1][N_Orb_NN[2,P,J+1]][2] = b
                        end
                        # Dense lookup instead of Orb_NN_Dic[O2b_key(a,b,J,P,0)] ...
                        N_Orb_NN[1,P,J+1] += 1
                        Orb_NN_Map[a,b,J+1,1] = N_Orb_NN[1,P,J+1]
                        Ind_Orb_NN[1,P,J+1][N_Orb_NN[1,P,J+1]] = zeros(Int64, 2)
                        Ind_Orb_NN[1,P,J+1][N_Orb_NN[1,P,J+1]][1] = a
                        Ind_Orb_NN[1,P,J+1][N_Orb_NN[1,P,J+1]][2] = b
                    end
                end
            end
        end
    end

    # Initialize arrays for 2-body operator O ...
    O_pp = Matrix{Vector{Float64}}(undef,2,J_max+1)
    O_pn = Matrix{Vector{Float64}}(undef,2,J_max+1)
    O_nn = Matrix{Vector{Float64}}(undef,2,J_max+1)
    
    # Allocate zero entries of O ...
    @inbounds for P = 1:2
        @inbounds for J=0:J_max
            Length1 = div(N_Orb_NN[1,P,J+1]*(N_Orb_NN[1,P,J+1]+1),2)
            Length2 = div(N_Orb_NN[2,P,J+1]*(N_Orb_NN[2,P,J+1]+1),2)
            O_pp[P,J+1] = zeros(Float64, Length2)
            O_pn[P,J+1] = zeros(Float64, Length1)
            O_nn[P,J+1] = zeros(Float64, Length2)
        end
    end

    # Allocate structure for 2-body operator O ...
    O_NN = O2B(O_pp,O_pn,O_nn)

    if Make_Orb_NN == false
        return O_NN
    elseif Make_Orb_NN == true
        # Allocate the array for NN orbitals ...
        Orb_NN = Orb2B(Orb_NN_Map,N_Orb_NN,Ind_Orb_NN)

        return O_NN, Orb_NN
    end
end

function O2b_initialize_empty()
    # Initialize uninitialized arrays for 2-body operator O ...
    O_pp = Matrix{Vector{Float64}}(undef,1,1)
    O_pn = Matrix{Vector{Float64}}(undef,1,1)
    O_nn = Matrix{Vector{Float64}}(undef,1,1)

    # Allocate structure for 2-body operator O ...
    O_NN = O2B(O_pp,O_pn,O_nn)
    return O_NN
end

@noinline O2b_throw(Key::Tuple) = throw(KeyError(Key))

@inline function O2b_lookup(a::Int64,b::Int64,J::Int64,T::Int64,Orb_NN::Orb2B)
    # Dense lookup instead of Orb_NN.Dic[O2b_key(a,b,J,P,T)] ... P is fixed by the parity of l_a + l_b ...
    # checkbounds(Bool,...) is not removed by @inbounds of the caller, so a wrong (a,b,J,T) still throws ...
    if !checkbounds(Bool,Orb_NN.Map,a,b,J+1,T+1)
        O2b_throw((a,b,J,T))
    end
    @inbounds Ind = Int64(Orb_NN.Map[a,b,J+1,T+1])
    if Ind == 0
        O2b_throw((a,b,J,T))
    end
    return Ind
end

@inline function O2b_index(a::Int64,b::Int64,c::Int64,d::Int64,J::Int64,P::Int64,T::Int64,Orb_NN::Orb2B)
    N = Orb_NN.N[T+1,P,J+1]
    Bra = O2b_lookup(a,b,J,T,Orb_NN)
    Ket = O2b_lookup(c,d,J,T,Orb_NN)
    Braket_max, Bracket_min = max(Bra,Ket), min(Bra,Ket)
    Ind = Braket_max + (Bracket_min - 1) * N - div(Bracket_min * (Bracket_min - 1),2)
    return Ind
end

@inline function O2b_pp(a::Int64,b::Int64,c::Int64,d::Int64,J::Int64,P::Int64,O::O2B,Orb::Vector{Orb1B},Orb_NN::Orb2B)
    if a < b
        a_s, b_s = b, a
        exp_ab = J + 1 + div(Orb[a].j + Orb[b].j, 2)
        Sign_ab = isodd(exp_ab) ? -1.0 : 1.0
    else
        a_s, b_s = a, b
        Sign_ab = 1.0
    end
    if c < d
        c_s, d_s = d, c
        exp_cd = J + 1 + div(Orb[c].j + Orb[d].j, 2)
        Sign_cd = isodd(exp_cd) ? -1.0 : 1.0
    else
        c_s, d_s = c, d
        Sign_cd = 1.0
    end
    Amp = Sign_ab * Sign_cd
    if (a < b) && (c < d)
        exp = div(Orb[a].j + Orb[b].j + Orb[c].j + Orb[d].j, 2)
        Amp = isodd(exp) ? -1.0 : 1.0
    end
    @inbounds Ind = O2b_index(a_s,b_s,c_s,d_s,J,P,1,Orb_NN)
    return Amp * O.pp[P,J+1][Ind]
end

@inline function O2b_pn(a::Int64,b::Int64,c::Int64,d::Int64,J::Int64,P::Int64,O::O2B,Orb_NN::Orb2B)
    @inbounds return O.pn[P,J+1][O2b_index(a,b,c,d,J,P,0,Orb_NN)]
end

@inline function O2b_nn(a::Int64,b::Int64,c::Int64,d::Int64,J::Int64,P::Int64,O::O2B,Orb::Vector{Orb1B},Orb_NN::Orb2B)
    if a < b
        a_s, b_s = b, a
        exp_ab = J + 1 + div(Orb[a].j + Orb[b].j, 2)
        Sign_ab = isodd(exp_ab) ? -1.0 : 1.0
    else
        a_s, b_s = a, b
        Sign_ab = 1.0
    end
    if c < d
        c_s, d_s = d, c
        exp_cd = J + 1 + div(Orb[c].j + Orb[d].j, 2)
        Sign_cd = isodd(exp_cd) ? -1.0 : 1.0
    else
        c_s, d_s = c, d
        Sign_cd = 1.0
    end
    Amp = Sign_ab * Sign_cd
    if (a < b) && (c < d)
        exp = div(Orb[a].j + Orb[b].j + Orb[c].j + Orb[d].j, 2)
        Amp = isodd(exp) ? -1.0 : 1.0
    end
    @inbounds Ind = O2b_index(a_s,b_s,c_s,d_s,J,P,1,Orb_NN)
    return Amp * O.nn[P,J+1][Ind]
end

function O2b_export(Params::Parameters,Orb_NN::Orb2B,O_NN::O2B,Export_Path::String)
    # Read parameters ...
    N_max = Params.Calc.Nmax
    J_max = 2*N_max + 1

    # Export 2-body NN operator ...
        # Each (J,P) block is first collected into a buffer & then written at once (same file layout as before) ...
    println("\nExporting given 2-body operator in internal binary format ...")
    open(Export_Path, "w") do Export_File
        # ppO
        @inbounds for J in 0:J_max
            @inbounds for P in 1:2
                N_T1 = Orb_NN.N[2,P,J+1]
                Buffer, n = Vector{Float64}(undef,div(N_T1*(N_T1+1),2)), 0
                @inbounds for Bra in 1:N_T1
                    @inbounds for Ket in 1:Bra
                        Ind = Bra + (Ket - 1) * N_T1 - div(Ket * (Ket - 1),2)
                        n += 1
                        Buffer[n] = O_NN.pp[P,J+1][Ind]
                    end
                end
                write(Export_File, Buffer)
            end
        end

        # pnO
        @inbounds for J in 0:J_max
            @inbounds for P in 1:2
                N_T0 = Orb_NN.N[1,P,J+1]
                Buffer, n = Vector{Float64}(undef,N_T0^2), 0
                @inbounds for Bra in 1:N_T0
                    @inbounds for Ket in 1:N_T0
                        Ind = Bra + (Ket - 1) * N_T0 - div(Ket * (Ket - 1),2)
                        n += 1
                        Buffer[n] = O_NN.pn[P,J+1][Ind]
                    end
                end
                write(Export_File, Buffer)
            end
        end

        # nnO
        @inbounds for J in 0:J_max
            @inbounds for P in 1:2
                N_T1 = Orb_NN.N[2,P,J+1]
                Buffer, n = Vector{Float64}(undef,div(N_T1*(N_T1+1),2)), 0
                @inbounds for Bra in 1:N_T1
                    @inbounds for Ket in 1:Bra
                        Ind = Bra + (Ket - 1) * N_T1 - div(Ket * (Ket - 1),2)
                        n += 1
                        Buffer[n] = O_NN.nn[P,J+1][Ind]
                    end
                end
                write(Export_File, Buffer)
            end
        end

    end

    println("\nGiven 2-body NN operator succesfully exported in the binary format ... ''" * string(Export_Path) * "''")

    return
end

function O2b_export_HRF(Params::Parameters,Orb_NN::Orb2B,O_NN::O2B,Export_Path::String)
    # Read parameters ...
    N_max = Params.Calc.Nmax
    J_max = 2*N_max + 1

    # Export given 2-body NN operator ...
    println("\nExporting given 2-body operator in the Human-Readable-Format ...")
    open(Export_Path, "w") do Export_File
        println(Export_File, "t\ta\tb\tc\td\tJ\tV")
        # ppO
        @inbounds for J in 0:J_max
            @inbounds for P in 1:2
                N_t1 = Orb_NN.N[2,P,J+1]
                @inbounds for Bra in 1:N_t1
                    a = 2*Orb_NN.Ind[2,P,J+1][Bra][1] - 1
                    b = 2*Orb_NN.Ind[2,P,J+1][Bra][2] - 1
                    @inbounds for Ket in 1:Bra
                        c = 2*Orb_NN.Ind[2,P,J+1][Ket][1] - 1
                        d = 2*Orb_NN.Ind[2,P,J+1][Ket][2] - 1
                        Ind = Bra + (Ket - 1) * N_t1 - div(Ket * (Ket - 1),2)
                        ME = O_NN.pp[P,J+1][Ind]
                        Row = "-1\t" * string(a) * "\t" * string(b) * "\t" *
                                string(c) * "\t" * string(d) * "\t" * string(J) *
                                "\t" * string(round(ME, digits = 10))
                        println(Export_File, Row)
                    end
                end
            end
        end

        # pnO
        @inbounds for J in 0:J_max
            @inbounds for P in 1:2
                N_t0 = Orb_NN.N[1,P,J+1]
                @inbounds for Bra in 1:N_t0
                    a = 2*Orb_NN.Ind[1,P,J+1][Bra][1] - 1
                    b = 2*Orb_NN.Ind[1,P,J+1][Bra][2]
                    @inbounds for Ket in 1:N_t0
                        c = 2*Orb_NN.Ind[1,P,J+1][Ket][1] - 1
                        d = 2*Orb_NN.Ind[1,P,J+1][Ket][2]
                        Ind = Bra + (Ket - 1) * N_t0 - div(Ket * (Ket - 1),2)
                        ME = O_NN.pn[P,J+1][Ind]
                        Row = "0\t" * string(a) * "\t" * string(b) * "\t" *
                                string(c) * "\t" * string(d) * "\t" * string(J) *
                                "\t" * string(round(ME, digits = 10))
                        println(Export_File, Row)
                    end
                end
            end
        end

        # nnO
        @inbounds for J in 0:J_max
            @inbounds for P in 1:2
                N_t1 = Orb_NN.N[2,P,J+1]
                @inbounds for Bra in 1:N_t1
                    a = 2*Orb_NN.Ind[2,P,J+1][Bra][1]
                    b = 2*Orb_NN.Ind[2,P,J+1][Bra][2]
                    @inbounds for Ket in 1:Bra
                        c = 2*Orb_NN.Ind[2,P,J+1][Ket][1]
                        d = 2*Orb_NN.Ind[2,P,J+1][Ket][2]
                        Ind = Bra + (Ket - 1) * N_t1 - div(Ket * (Ket - 1),2)
                        ME = O_NN.nn[P,J+1][Ind]
                        Row = "1\t" * string(a) * "\t" * string(b) * "\t" *
                                string(c) * "\t" * string(d) * "\t" * string(J) *
                                "\t" * string(round(ME, digits = 10))
                        println(Export_File, Row)
                    end
                end
            end
        end

    end

    println("\nGiven 2-body NN operator succesfully exported in the Human-Readable-Format ... ''" * string(Export_Path) * "''")

    return
end

function O2b_import(Params::Parameters,Orb::Vector{Orb1B},Import_Path::String;Make_Orb_NN::Bool=false)
    # Read parameters ...
    N_max = Params.Calc.Nmax
    J_max = 2 * N_max + 1

    # Read given 2-body NN operator ...
    println("\nReading given 2-body NN operator ...")

    # Initialize NN residual interaction arrays ...
    O_NN, Orb_NN = O2b_initialize(Params,Orb;Make_Orb_NN=true)

    # Iteraction list for J & P ...
    JP_list = JP_initialize(J_max)

    # Precalculate chunks of data for parallelized reading ...
    N_chunk_skip = O2b_count(N_max,Orb_NN)

    # Load the given 2-body NN operator ...
        # Each (J,P) block is read at once into a buffer & then distributed into O_NN ...

    # ppO
    @inbounds Threads.@threads for JP in JP_list
        J, P = JP[1], JP[2]
        N_T1 = Orb_NN.N[2,P,J+1]
        Buffer, n = Vector{Float64}(undef,div(N_T1*(N_T1+1),2)), 0
        open(Import_Path, "r") do Bin_Read
            seek(Bin_Read, N_chunk_skip[1,P,J+1])
            read!(Bin_Read, Buffer)
        end
        @inbounds for Bra in 1:N_T1
            @inbounds for Ket in 1:Bra
                Ind = Bra + (Ket - 1) * N_T1 - div(Ket * (Ket - 1),2)
                n += 1
                O_NN.pp[P,J+1][Ind] = Buffer[n]
            end
        end
    end

    # pnO
    @inbounds Threads.@threads for JP in JP_list
        J, P = JP[1], JP[2]
        N_T0 = Orb_NN.N[1,P,J+1]
        Buffer, n = Vector{Float64}(undef,N_T0^2), 0
        open(Import_Path, "r") do Bin_Read
            seek(Bin_Read, N_chunk_skip[2,P,J+1])
            read!(Bin_Read, Buffer)
        end
        @inbounds for Bra in 1:N_T0
            @inbounds for Ket in 1:N_T0
                Ind = Bra + (Ket - 1) * N_T0 - div(Ket * (Ket - 1),2)
                n += 1
                O_NN.pn[P,J+1][Ind] = Buffer[n]
            end
        end
    end

    # nnO
    @inbounds Threads.@threads for JP in JP_list
        J, P = JP[1], JP[2]
        N_T1 = Orb_NN.N[2,P,J+1]
        Buffer, n = Vector{Float64}(undef,div(N_T1*(N_T1+1),2)), 0
        open(Import_Path, "r") do Bin_Read
            seek(Bin_Read, N_chunk_skip[3,P,J+1])
            read!(Bin_Read, Buffer)
        end
        @inbounds for Bra in 1:N_T1
            @inbounds for Ket in 1:Bra
                Ind = Bra + (Ket - 1) * N_T1 - div(Ket * (Ket - 1),2)
                n += 1
                O_NN.nn[P,J+1][Ind] = Buffer[n]
            end
        end
    end

    println("\nGiven 2-body NN operator succesfully read ...")

    if Make_Orb_NN == true
        return O_NN, Orb_NN
    else
        return O_NN
    end
end

function O2b_count(N_max::Int64,Orb_NN::Orb2B)
    # Read parameters ...
    J_max = 2 * N_max + 1

    JP_list = JP_initialize(J_max)

    N_chunk = Array{Int64}(undef,3,2,J_max+1)
    N_chunk_skip = Array{Int64}(undef,3,2,J_max+1)

    # ppO
    @inbounds for JP in JP_list
        N, J, P = 0, JP[1], JP[2]
        N_T1 = Orb_NN.N[2,P,J+1]
        @inbounds for Bra in 1:N_T1
            @inbounds for Ket in 1:Bra
                N += 1
            end
        end
        N_chunk[1,P,J+1] = N
    end

    # pnO
    @inbounds for JP in JP_list
        N, J, P = 0, JP[1], JP[2]
        N_T0 = Orb_NN.N[1,P,J+1]
        @inbounds for Bra in 1:N_T0
            @inbounds for Ket in 1:N_T0
                N += 1
            end
        end
        N_chunk[2,P,J+1] = N
    end

    # nnO
    @inbounds for JP in JP_list
        N, J, P = 0, JP[1], JP[2]
        N_T1 = Orb_NN.N[2,P,J+1]
        @inbounds for Bra in 1:N_T1
            @inbounds for Ket in 1:Bra
                N += 1
            end
        end
        N_chunk[3,P,J+1] = N
    end

    @inbounds for T_1 in 1:3
        @inbounds for JP_1 in 1:(2*(J_max+1))
            J_1 = JP_list[JP_1][1]
            P_1 = JP_list[JP_1][2]
            Sum = 0
            @inbounds for T_2 in 1:T_1
                if T_2 < T_1
                    @inbounds for JP_2 in 1:(2*(J_max+1))
                        J_2 = JP_list[JP_2][1]
                        P_2 = JP_list[JP_2][2]
                        Sum += 8 * N_chunk[T_2,P_2,J_2+1]
                    end
                elseif  T_2 == T_1
                    @inbounds for JP_2 in 1:(JP_1-1)
                        J_2 = JP_list[JP_2][1]
                        P_2 = JP_list[JP_2][2]
                        Sum += 8 * N_chunk[T_2,P_2,J_2+1]
                    end
                end
            end
            N_chunk_skip[T_1,P_1,J_1+1] = Sum
        end
    end

    return N_chunk_skip
end

function O2b_temp_initialize(Params::Parameters,Orb::Vector{Orb1B};Make_Orb_NN::Bool=false)
    # Read parameters ...
    N_max, N_2max = Params.Calc.Nmax, Params.Calc.N2max
    a_max = div((N_max + 1)*(N_max + 2),2)
    J_max = Int64(N_2max + 1)

    # Initiliaze array for counting of NN orbitals ...
    N_Orb_NN = zeros(Int64,2,J_max+1)

    # Initiliaze the dense lookup for NN orbitals ... Map[a,b,J+1] (see Orb2B_Temp) ...
    Orb_NN_Map = zeros(Int32,a_max,a_max,J_max+1)

    # Count the number of NN orbitals ...
    @inbounds for a in 1:a_max
        n_a = Orb[a].n
        l_a = Orb[a].l
        j_a = Orb[a].j
        if (2*n_a + l_a) <= N_max
            @inbounds for b = 1:a_max
                n_b = Orb[b].n
                l_b = Orb[b].l
                j_b = Orb[b].j
                if ((2*(n_a + n_b) + l_a + l_b) <= N_2max)
                    P = rem(l_a + l_b, 2) + 1
                    @inbounds for J = Int64(abs(j_a - j_b)/2):Int64((j_a + j_b)/2)
                        N_Orb_NN[P,J+1] += 1
                    end
                end
            end
        end
    end

    # Initialite the array for NN orbital indices ...
    Ind_Orb_NN = Matrix{Vector{Vector{Int64}}}(undef,2,J_max+1)
    @inbounds for P = 1:2
        @inbounds for J = 0:J_max
            Ind_Orb_NN[P,J+1] = Vector{Vector{Int64}}(undef,N_Orb_NN[P,J+1])
        end
    end

    # Reinitialize array for counting of NN orbitals ...
    N_Orb_NN = zeros(Int64,2,J_max+1)

    # Allocate the dense lookup and indices for NN orbitals ...
    @inbounds for a in 1:a_max
        n_a = Orb[a].n
        l_a = Orb[a].l
        j_a = Orb[a].j
        if (2*n_a + l_a) <= N_max
            @inbounds for b = 1:a_max
                n_b = Orb[b].n
                l_b = Orb[b].l
                j_b = Orb[b].j
                if ((2*(n_a + n_b) + l_a + l_b) <= N_2max)
                    P = rem(l_a + l_b, 2) + 1
                    @inbounds for J = Int64(abs(j_a - j_b)/2):Int64((j_a + j_b)/2)
                        # Dense lookup instead of Orb_NN_Dic[O2b_temp_key(a,b,J,P)] ...
                        N_Orb_NN[P,J+1] += 1
                        Orb_NN_Map[a,b,J+1] = N_Orb_NN[P,J+1]
                        Ind_Orb_NN[P,J+1][N_Orb_NN[P,J+1]] = zeros(Int64, 2)
                        Ind_Orb_NN[P,J+1][N_Orb_NN[P,J+1]][1] = a
                        Ind_Orb_NN[P,J+1][N_Orb_NN[P,J+1]][2] = b
                    end
                end
            end
        end
    end

    # Initialize the NN interaction arrays for O ...
    O_pp = Matrix{Matrix{Float64}}(undef,2,J_max+1)
    O_pn = Matrix{Matrix{Float64}}(undef,2,J_max+1)
    O_nn = Matrix{Matrix{Float64}}(undef,2,J_max+1)
    
    # Allocate entries of O ...
    @inbounds for P = 1:2
        @inbounds Threads.@threads for J=0:J_max
            Length = N_Orb_NN[P,J+1]
            O_pp[P,J+1] = zeros(Float64,Length,Length)
            O_pn[P,J+1] = zeros(Float64,Length,Length)
            O_nn[P,J+1] = zeros(Float64,Length,Length)
        end
    end

    # Allocate O ...
    O_NN_t = O2B_Temp(O_pp,O_pn,O_nn)

    if Make_Orb_NN == false
        return O_NN_t
    elseif Make_Orb_NN == true
        # Allocate the array for NN orbitals ...
        Orb_NN_t = Orb2B_Temp(Orb_NN_Map,N_Orb_NN,Ind_Orb_NN)

        return O_NN_t, Orb_NN_t
    end
end

@inline function O2b_temp_index(a::Int64,b::Int64,J::Int64,P::Int64,Orb_NN_temp::Orb2B_Temp)
    # Dense lookup instead of Orb_NN_temp.Dic[O2b_temp_key(a,b,J,P)] ... P is fixed by the parity of l_a + l_b ...
    if !checkbounds(Bool,Orb_NN_temp.Map,a,b,J+1)
        O2b_throw((a,b,J,P))
    end
    @inbounds Ind = Int64(Orb_NN_temp.Map[a,b,J+1])
    if Ind == 0
        O2b_throw((a,b,J,P))
    end
    return Ind
end

@inline function O2b_temp(a::Int64,b::Int64,c::Int64,d::Int64,J::Int64,P::Int64,T::Int64,O::O2B_Temp,Orb_NN_temp::Orb2B_Temp)
    Bra = O2b_temp_index(a,b,J,P,Orb_NN_temp)
    Ket = O2b_temp_index(c,d,J,P,Orb_NN_temp)
    if T == -1
        return O.pp[P,J+1][Bra,Ket]

    elseif T == 0
        return O.pn[P,J+1][Bra,Ket]

    elseif T == 1
        return O.nn[P,J+1][Bra,Ket]
    end
end

function O2b_transformation(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,O_NN::O2B,C::O1B;Report::Bool=true)
    # Transformation of the 2-body operator O_NN into the target basis defiend by C ...
    if Report
        println("\nPerforming the Transformation of 2-body operator O_NN into the target basis due to C ...")
    end

    # Preallocate arrays of allowed values of J & P ...
    JP = JP_initialize(Params.Calc.N2max + 1)

    # Transformation of the 1st index ...
    O_NN_t1, Orb_NN_t = O2b_transformation_index1(Params,JP,Orb,Orb_NN,O_NN,C,Report)

    # Transformation of the 2nd index ...
    O_NN_t2 = O2b_transformation_index2(Params,JP,Orb,Orb_NN_t,O_NN_t1,C,Report)

    # Transformation of the 3rd index ...
    O_NN_t1 = O2b_transformation_index3(Params,JP,Orb,Orb_NN_t,O_NN_t1,O_NN_t2,C,Report)

    # Transformation of the 4th index ...
    O_NN = O2b_transformation_index4(Params,JP,Orb,Orb_NN,Orb_NN_t,O_NN,O_NN_t1,C,Report)

    # Deallocate temporary interaction arrays ...
    O_NN_t1 = nothing
    O_NN_t2 = nothing

    # Perform the Garbage collection ...
    GC.gc()

    if Report
        println("\nGiven 2-body operator O_NN transformed ...\n")
    end

    return O_NN
end

function O2b_transformation_index1(Params::Parameters,JP::Vector{Vector{Int64}},Orb::Vector{Orb1B},Orb_NN::Orb2B,O_NN::O2B,C::O1B,Report::Bool)
    # Parameter initialization...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max

    # Precompute orbitals lists for each combination of (l,j) ...
    Orb_lj = orbitals_lj_make(Params,Orb)

    # Make temporary arrays for NN interaction V ...
    O_NN_t1, Orb_NN_t = O2b_temp_initialize(Params,Orb,Make_Orb_NN = true)

    # Define function for the transformation MEs for the 1st index ...
    @inline function O2b_temp_transformation_ind1_MEs(J::Int64,P::Int64,i::Int64,a::Int64,b::Int64,c::Int64,d::Int64,C::O1B,O_NN::O2B,Orb::Vector{Orb1B},Orb_NN::Orb2B)
        return @views C.p[i,a] * O2b_pp(i,b,c,d,J,P,O_NN,Orb,Orb_NN), C.p[i,a] * O2b_pn(i,b,c,d,J,P,O_NN,Orb_NN), C.n[i,a] * O2b_nn(i,b,c,d,J,P,O_NN,Orb,Orb_NN)
    end

    # Transformation of the 1st index ...
    if Report
        println("\nPerforming transformation of the given 2-body operator in the 1st index...")
    end
    @inbounds Threads.@threads for jp in JP
        J, P = jp[1], jp[2]
        if Report
            if P == 1
                println("\tCalculating   ...   J = " * string(J) * "/" * string(N_2max+1) * "\tP = +")
            else
                println("\tCalculating   ...   J = " * string(J) * "/" * string(N_2max+1) * "\tP = -")
            end
        end
        N = Orb_NN_t.N[P,J+1]
        @inbounds for Bra in 1:N
            a = Orb_NN_t.Ind[P,J+1][Bra][1]
            b = Orb_NN_t.Ind[P,J+1][Bra][2]
            l_a, j_a = Orb[a].l, Orb[a].j

            Orb_x = Orb_lj[l_a+1,j_a]
            @inbounds for Ket in 1:N
                c = Orb_NN_t.Ind[P,J+1][Ket][1]
                d = Orb_NN_t.Ind[P,J+1][Ket][2]

                OppSum, OpnSum, OnnSum = 0.0, 0.0, 0.0
                
                @inbounds for i in Orb_x
                    OppME, OpnME, OnnME = O2b_temp_transformation_ind1_MEs(J,P,i,a,b,c,d,C,O_NN,Orb,Orb_NN)
                    OppSum += OppME
                    OpnSum += OpnME
                    OnnSum += OnnME
                end
                @views O_NN_t1.pp[P,J+1][Bra,Ket] = OppSum
                @views O_NN_t1.pn[P,J+1][Bra,Ket] = OpnSum
                @views O_NN_t1.nn[P,J+1][Bra,Ket] = OnnSum
            end
        end
    end

    return O_NN_t1, Orb_NN_t
end

function O2b_transformation_index2(Params::Parameters,JP::Vector{Vector{Int64}},Orb::Vector{Orb1B},Orb_NN_t::Orb2B_Temp,O_NN_t1::O2B_Temp,C::O1B,Report::Bool)
    # Parameter initialization...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max

    # Precompute orbitals lists for each combination of (l,j) ...
    Orb_lj = orbitals_lj_make(Params,Orb)

    # Make another temporary arrays for NN interaction V ...
    O_NN_t2 = O2b_temp_initialize(Params,Orb,Make_Orb_NN = false)

    # Define function for the transformation MEs for the 2nd index ...
    @inline function O2b_temp_transformation_ind2_MEs(P::Int64,J::Int64,Bra::Int64,Ket::Int64,i::Int64,b::Int64,C::O1B,O_NN_t::O2B_Temp)
        return @views C.p[i,b] * O_NN_t.pp[P,J+1][Bra,Ket], C.n[i,b] * O_NN_t.pn[P,J+1][Bra,Ket], C.n[i,b] * O_NN_t.nn[P,J+1][Bra,Ket]
    end

    # Transformation of the 2nd index ...
    if Report
        println("\nPerforming transformation of the given 2-body operator O_NN in the 2nd index...")
    end
    @inbounds Threads.@threads for jp in JP
        J, P = jp[1], jp[2]
        if Report
            if P == 1
                println("\tCalculating   ...   J = " * string(J) * "/" * string(N_2max+1) * "\tP = +")
            else
                println("\tCalculating   ...   J = " * string(J) * "/" * string(N_2max+1) * "\tP = -")
            end
        end
        N = Orb_NN_t.N[P,J+1]
        @inbounds for Bra in 1:N
            a = Orb_NN_t.Ind[P,J+1][Bra][1]
            b = Orb_NN_t.Ind[P,J+1][Bra][2]
            l_b, j_b = Orb[b].l, Orb[b].j

            Orb_x = Orb_lj[l_b+1,j_b]

            @inbounds for Ket in 1:N
                OppSum, OpnSum, OnnSum = 0.0, 0.0, 0.0
                
                @inbounds for i in Orb_x
                    Bra_ai = O2b_temp_index(a,i,J,P,Orb_NN_t)
                    OppME, OpnME, OnnME = O2b_temp_transformation_ind2_MEs(P,J,Bra_ai,Ket,i,b,C,O_NN_t1)
                    OppSum += OppME
                    OpnSum += OpnME
                    OnnSum += OnnME
                end

                @views O_NN_t2.pp[P,J+1][Bra,Ket] = OppSum
                @views O_NN_t2.pn[P,J+1][Bra,Ket] = OpnSum
                @views O_NN_t2.nn[P,J+1][Bra,Ket] = OnnSum
            end
        end
    end

    return O_NN_t2
end

function O2b_transformation_index3(Params::Parameters,JP::Vector{Vector{Int64}},Orb::Vector{Orb1B},Orb_NN_t::Orb2B_Temp,O_NN_t1::O2B_Temp,O_NN_t2::O2B_Temp,C::O1B,Report::Bool)
    # Parameter initialization...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max

    # Precompute orbitals lists for each combination of (l,j) ...
    Orb_lj = orbitals_lj_make(Params,Orb)

    # Define function for the transformation MEs for the 3rd index ...
    @inline function O2b_temp_transformation_ind3_MEs(P::Int64,J::Int64,Bra::Int64,Ket::Int64,i::Int64,c::Int64,C::O1B,O_NN_t::O2B_Temp)
        return @views C.p[i,c] * O_NN_t.pp[P,J+1][Bra,Ket], C.p[i,c] * O_NN_t.pn[P,J+1][Bra,Ket], C.n[i,c] * O_NN_t.nn[P,J+1][Bra,Ket]
    end

    # Transformation of the 3rd index ...
    if Report
        println("\nPerforming transformation of the given 2-body operator O_NN in the 3rd index...")
    end
    @inbounds Threads.@threads for jp in JP
        J, P = jp[1], jp[2]
        if Report
            if P == 1
                println("\tCalculating   ...   J = " * string(J) * "/" * string(N_2max+1) * "\tP = +")
            else
                println("\tCalculating   ...   J = " * string(J) * "/" * string(N_2max+1) * "\tP = -")
            end
        end
        N = Orb_NN_t.N[P,J+1]
        @inbounds for Ket in 1:N
            c = Orb_NN_t.Ind[P,J+1][Ket][1]
            d = Orb_NN_t.Ind[P,J+1][Ket][2]
            l_c, j_c = Orb[c].l, Orb[c].j

            Orb_x = Orb_lj[l_c+1,j_c]
            @inbounds for Bra in 1:N
                OppSum, OpnSum, OnnSum = 0.0, 0.0, 0.0

                @inbounds for i in Orb_x
                    Ket_id = O2b_temp_index(i,d,J,P,Orb_NN_t)

                    OppME, OpnME, OnnME = O2b_temp_transformation_ind3_MEs(P,J,Bra,Ket_id,i,c,C,O_NN_t2)

                    OppSum += OppME
                    OpnSum += OpnME
                    OnnSum += OnnME
                end

                @views O_NN_t1.pp[P,J+1][Bra,Ket] = OppSum
                @views O_NN_t1.pn[P,J+1][Bra,Ket] = OpnSum
                @views O_NN_t1.nn[P,J+1][Bra,Ket] = OnnSum
            end
        end
    end

    return O_NN_t1
end

function O2b_transformation_index4(Params::Parameters,JP::Vector{Vector{Int64}},Orb::Vector{Orb1B},Orb_NN::Orb2B,Orb_NN_t::Orb2B_Temp,O_NN::O2B,O_NN_t1::O2B_Temp,C::O1B,Report::Bool)
    # Parameter initialization...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max

    # Precompute orbitals lists for each combination of (l,j) ...
    Orb_lj = orbitals_lj_make(Params,Orb)

    # Define function for the transformation MEs for the 4th index ...
        # Case of O_NN ... T = 0 ...
    @inline function O2b_temp_transformation_ind4_T0_MEs(P::Int64,J::Int64,Bra::Int64,Ket::Int64,i::Int64,d::Int64,C::O1B,O_NN_t::O2B_Temp)
        return @views C.n[i,d] * O_NN_t.pn[P,J+1][Bra,Ket]
    end
        # Case of O_NN ... T = 1 ...
    @inline function O2b_temp_transformation_ind4_T1_MEs(P::Int64,J::Int64,Bra::Int64,Ket::Int64,i::Int64,d::Int64,C::O1B,O_NN_t::O2B_Temp)
        return @views C.p[i,d] * O_NN_t.pp[P,J+1][Bra,Ket], C.n[i,d] * O_NN_t.nn[P,J+1][Bra,Ket]
    end

    # Index 4
    if Report
        println("\nPerforming transformation of the given 2-body operator O_NN in the 4th index...")
    end
    @inbounds Threads.@threads for jp in JP
        J, P = jp[1], jp[2]
        if Report
            if P == 1
                println("\tCalculating   ...   J = " * string(J) * "/" * string(N_2max+1) * "\tP = +")
            else
                println("\tCalculating   ...   J = " * string(J) * "/" * string(N_2max+1) * "\tP = -")
            end
        end
        N_T0, N_T1 = Orb_NN.N[1,P,J+1], Orb_NN.N[2,P,J+1]
        @inbounds for Bra in 1:max(N_T0, N_T1)
            @inbounds for Ket in 1:Bra

                # Case of pn interaction ... T = 0
                if Bra <= N_T0
                    Ind = Bra + (Ket - 1) * N_T0 - div(Ket * (Ket - 1),2)
                    a, b = Orb_NN.Ind[1,P,J+1][Bra][1], Orb_NN.Ind[1,P,J+1][Bra][2]
                    c, d = Orb_NN.Ind[1,P,J+1][Ket][1], Orb_NN.Ind[1,P,J+1][Ket][2]
                    l_d, j_d = Orb[d].l, Orb[d].j

                    Bra_ab = O2b_temp_index(a,b,J,P,Orb_NN_t)

                    Orb_x = Orb_lj[l_d+1,j_d]

                    OpnSum = 0.0

                    @inbounds for i in Orb_x
                        Ket_ci = O2b_temp_index(c,i,J,P,Orb_NN_t)

                        OpnME = O2b_temp_transformation_ind4_T0_MEs(P,J,Bra_ab,Ket_ci,i,d,C,O_NN_t1)

                        OpnSum += OpnME
                    end

                    O_NN.pn[P,J+1][Ind] = OpnSum
                end

                # Case of pp & nn interaction ... T = 1
                if Bra <= N_T1
                    Ind = Bra + (Ket - 1) * N_T1 - div(Ket * (Ket - 1),2)
                    a, b = Orb_NN.Ind[2,P,J+1][Bra][1], Orb_NN.Ind[2,P,J+1][Bra][2]
                    c, d = Orb_NN.Ind[2,P,J+1][Ket][1], Orb_NN.Ind[2,P,J+1][Ket][2]
                    l_d, j_d = Orb[d].l, Orb[d].j

                    Bra_ab = O2b_temp_index(a,b,J,P,Orb_NN_t)

                    Orb_x = Orb_lj[l_d+1,j_d]

                    OppSum, OnnSum = 0.0, 0.0

                    @inbounds for i in Orb_x
                        Ket_ci = O2b_temp_index(c,i,J,P,Orb_NN_t)

                        OppME, OnnME = O2b_temp_transformation_ind4_T1_MEs(P,J,Bra_ab,Ket_ci,i,d,C,O_NN_t1)

                        OppSum += OppME
                        OnnSum += OnnME
                    end

                    O_NN.pp[P,J+1][Ind] = OppSum
                    O_NN.nn[P,J+1][Ind] = OnnSum
                end
            end
        end

    end

    return O_NN
end

function O2b_Pandya_initialize(Params::Parameters,Orb::Vector{Orb1B};Make_Orb_NN::Bool=false)
    # Read parameters ...
    N_max, N_2max = Params.Calc.Nmax, Params.Calc.N2max
    a_max = div((N_max + 1)*(N_max + 2),2)
    J_max = Int64(N_2max + 1)

    # Initiliaze array for counting of NN orbitals ...
    N_Orb_NN = zeros(Int64,2,J_max+1)

    # Initiliaze the dense lookup for NN orbitals ... Map[a,b,J+1] (see Orb2B_Temp) ...
    Orb_NN_Map = zeros(Int32,a_max,a_max,J_max+1)

    # Count the number of NN orbitals ...
    @inbounds for a in 1:a_max
        n_a = Orb[a].n
        l_a = Orb[a].l
        j_a = Orb[a].j
        if (2*n_a + l_a) <= N_max
            @inbounds for b = 1:a_max
                n_b = Orb[b].n
                l_b = Orb[b].l
                j_b = Orb[b].j
                if ((2*(n_a + n_b) + l_a + l_b) <= N_2max)
                    P = rem(l_a + l_b, 2) + 1
                    @inbounds for J = Int64(abs(j_a - j_b)/2):Int64((j_a + j_b)/2)
                        N_Orb_NN[P,J+1] += 1
                    end
                end
            end
        end
    end

    # Initialite the array for NN orbital indices ...
    Ind_Orb_NN = Matrix{Vector{Vector{Int64}}}(undef,2,J_max+1)
    @inbounds for P = 1:2
        @inbounds for J = 0:J_max
            Ind_Orb_NN[P,J+1] = Vector{Vector{Int64}}(undef,N_Orb_NN[P,J+1])
        end
    end

    # Reinitialize array for counting of NN orbitals ...
    N_Orb_NN = zeros(Int64,2,J_max+1)

    # Allocate the dense lookup and indices for NN orbitals ...
    @inbounds for a in 1:a_max
        n_a = Orb[a].n
        l_a = Orb[a].l
        j_a = Orb[a].j
        if (2*n_a + l_a) <= N_max
            @inbounds for b = 1:a_max
                n_b = Orb[b].n
                l_b = Orb[b].l
                j_b = Orb[b].j
                if ((2*(n_a + n_b) + l_a + l_b) <= N_2max)
                    P = rem(l_a + l_b, 2) + 1
                    @inbounds for J = Int64(abs(j_a - j_b)/2):Int64((j_a + j_b)/2)
                        # Dense lookup instead of Orb_NN_Dic[O2b_temp_key(a,b,J,P)] ...
                        N_Orb_NN[P,J+1] += 1
                        Orb_NN_Map[a,b,J+1] = N_Orb_NN[P,J+1]
                        Ind_Orb_NN[P,J+1][N_Orb_NN[P,J+1]] = zeros(Int64, 2)
                        Ind_Orb_NN[P,J+1][N_Orb_NN[P,J+1]][1] = a
                        Ind_Orb_NN[P,J+1][N_Orb_NN[P,J+1]][2] = b
                    end
                end
            end
        end
    end

    # Initialize the NN interaction arrays for O ...
    O_pp = Matrix{Matrix{Float64}}(undef,2,J_max+1)
    O_pnnp = Matrix{Matrix{Float64}}(undef,2,J_max+1)
    O_pnpn = Matrix{Matrix{Float64}}(undef,2,J_max+1)
    O_nn = Matrix{Matrix{Float64}}(undef,2,J_max+1)
    
    # Allocate entries of O ...
    @inbounds for P = 1:2
        @inbounds Threads.@threads for J=0:J_max
            Length = N_Orb_NN[P,J+1]
            O_pp[P,J+1] = zeros(Float64,Length,Length)
            O_pnnp[P,J+1] = zeros(Float64,Length,Length)
            O_pnpn[P,J+1] = zeros(Float64,Length,Length)
            O_nn[P,J+1] = zeros(Float64,Length,Length)
        end
    end

    # Allocate O ...
    O_NN_t = O2B_Pandya(O_pp,O_pnnp,O_pnpn,O_nn)

    if Make_Orb_NN == false
        return O_NN_t
    elseif Make_Orb_NN == true
        # Allocate the array for NN orbitals ...
        Orb_NN_t = Orb2B_Temp(Orb_NN_Map,N_Orb_NN,Ind_Orb_NN)

        return O_NN_t, Orb_NN_t
    end
end

@inline function O2b_Pandya_pp(a::Int64,b::Int64,c::Int64,d::Int64,J::Int64,P::Int64,O::O2B_Pandya,Orb_NN_temp::Orb2B_Temp)
    Bra = O2b_temp_index(a,b,J,P,Orb_NN_temp)
    Ket = O2b_temp_index(c,d,J,P,Orb_NN_temp)
    return O.pp[P,J+1][Bra,Ket]
end

@inline function O2b_Pandya_pnpn(a::Int64,b::Int64,c::Int64,d::Int64,J::Int64,P::Int64,O::O2B_Pandya,Orb_NN_temp::Orb2B_Temp)
    Bra = O2b_temp_index(a,b,J,P,Orb_NN_temp)
    Ket = O2b_temp_index(c,d,J,P,Orb_NN_temp)
    return O.pnpn[P,J+1][Bra,Ket]
end

@inline function O2b_Pandya_pnnp(a::Int64,b::Int64,c::Int64,d::Int64,J::Int64,P::Int64,O::O2B_Pandya,Orb_NN_temp::Orb2B_Temp)
    Bra = O2b_temp_index(a,b,J,P,Orb_NN_temp)
    Ket = O2b_temp_index(c,d,J,P,Orb_NN_temp)
    return O.pnnp[P,J+1][Bra,Ket]
end

@inline function O2b_Pandya_nn(a::Int64,b::Int64,c::Int64,d::Int64,J::Int64,P::Int64,O::O2B_Pandya,Orb_NN_temp::Orb2B_Temp)
    Bra = O2b_temp_index(a,b,J,P,Orb_NN_temp)
    Ket = O2b_temp_index(c,d,J,P,Orb_NN_temp)
    return O.nn[P,J+1][Bra,Ket]
end

function O2b_Pandya_transformation(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,O_NN::O2B)
    # Preallocate arrays of allowed values of J & P ...
    JP_list = JP_initialize(Params.Calc.N2max + 1)

    # Make full arrays for the Pandya transformed operatoe O_NN ...
    F_NN, Orb_NN_t = O2b_Pandya_initialize(Params,Orb,Make_Orb_NN = true)
    
    # Define local functions performing the Pandya transformation of ph matrix elements ...
    @inline function O2b_Pandya_transformation_ppF(a::Int64,b::Int64,c::Int64,d::Int64,J::Int64,Orb::Vector{Orb1B},Orb_NN::Orb2B,O_NN::O2B)
        l_a, l_b, l_c, l_d = Orb[a].l, Orb[b].l, Orb[c].l, Orb[d].l
        Q = parity(l_a + l_d)
        ppFSum = 0.0
        if Q == parity(l_b + l_c)
            j_a, j_b, j_c, j_d = Orb[a].j, Orb[b].j, Orb[c].j, Orb[d].j
            I_min, I_max = max(div(abs(j_a - j_d),2),div(abs(j_b - j_c),2)), min(div(j_a + j_d,2),div(j_b + j_c,2))
            @inbounds for I in I_min:I_max
                ME = phase(div(j_b + j_c,2) + I) * hat(2*I) * w6j(j_a,j_b,2*J,j_c,j_d,2*I) * O2b_pp(a,d,b,c,I,Q,O_NN,Orb,Orb_NN)
                ppFSum += ME
            end
        end
        return ppFSum
    end

    @inline function O2b_Pandya_transformation_pnpnF(a::Int64,b::Int64,c::Int64,d::Int64,J::Int64,Orb::Vector{Orb1B},Orb_NN::Orb2B,O_NN::O2B)
        l_a, l_b, l_c, l_d = Orb[a].l, Orb[b].l, Orb[c].l, Orb[d].l
        Q = parity(l_a + l_d)
        pnpnFSum = 0.0
        if Q == parity(l_b + l_c)
            j_a, j_b, j_c, j_d = Orb[a].j, Orb[b].j, Orb[c].j, Orb[d].j
            I_min, I_max = max(div(abs(j_a - j_d),2),div(abs(j_b - j_c),2)), min(div(j_a + j_d,2),div(j_b + j_c,2))
            @inbounds for I in I_min:I_max
                ME = phase(1) * hat(2*I) * w6j(j_a,j_b,2*J,j_c,j_d,2*I) * O2b_pn(a,d,c,b,I,Q,O_NN,Orb_NN)
                pnpnFSum += ME
            end
        end
        return pnpnFSum
    end

    @inline function O2b_Pandya_transformation_pnnpF(a::Int64,b::Int64,c::Int64,d::Int64,J::Int64,Orb::Vector{Orb1B},Orb_NN::Orb2B,O_NN::O2B)
        l_a, l_b, l_c, l_d = Orb[a].l, Orb[b].l, Orb[c].l, Orb[d].l
        Q = parity(l_a + l_d)
        pnnpFSum = 0.0
        if Q == parity(l_b + l_c)
            j_a, j_b, j_c, j_d = Orb[a].j, Orb[b].j, Orb[c].j, Orb[d].j
            I_min, I_max = max(div(abs(j_a - j_d),2),div(abs(j_b - j_c),2)), min(div(j_a + j_d,2),div(j_b + j_c,2))
            @inbounds for I in I_min:I_max
                ME = phase(div(j_b + j_c,2) + I) * hat(2*I)* w6j(j_a,j_b,2*J,j_c,j_d,2*I) * O2b_pn(a,d,b,c,I,Q,O_NN,Orb_NN)
                pnnpFSum += ME
            end
        end
        return pnnpFSum
    end

    @inline function O2b_Pandya_transformation_nnF(a::Int64,b::Int64,c::Int64,d::Int64,J::Int64,Orb::Vector{Orb1B},Orb_NN::Orb2B,O_NN::O2B)
        l_a, l_b, l_c, l_d = Orb[a].l, Orb[b].l, Orb[c].l, Orb[d].l
        Q = parity(l_a + l_d)
        nnFSum = 0.0
        if Q == parity(l_b + l_c)
            j_a, j_b, j_c, j_d = Orb[a].j, Orb[b].j, Orb[c].j, Orb[d].j
            I_min, I_max = max(div(abs(j_a - j_d),2),div(abs(j_b - j_c),2)), min(div(j_a + j_d,2),div(j_b + j_c,2))
            @inbounds for I in I_min:I_max
                ME = phase(div(j_b + j_c,2) + I) * hat(2*I) * w6j(j_a,j_b,2*J,j_c,j_d,2*I) * O2b_nn(a,d,b,c,I,Q,O_NN,Orb,Orb_NN)
                nnFSum += ME
            end
        end
        return nnFSum
    end

    # Perform the Pandya transformation of the given 2-body O_NN operator ...
    println("\nPerforming the Pandya transformation of the given 2-body operator ..")
    @inbounds for JP in JP_list
        J, P = JP[1], JP[2]
        N = Orb_NN_t.N[P,J+1]
        @inbounds Threads.@threads for Bra in 1:N
            a = Orb_NN_t.Ind[P,J+1][Bra][1]
            b = Orb_NN_t.Ind[P,J+1][Bra][2]

            @inbounds for Ket in 1:N
                c = Orb_NN_t.Ind[P,J+1][Ket][1]
                d = Orb_NN_t.Ind[P,J+1][Ket][2]

                ppF = O2b_Pandya_transformation_ppF(a,b,c,d,J,Orb,Orb_NN,O_NN)
                pnpnF = O2b_Pandya_transformation_pnpnF(a,b,c,d,J,Orb,Orb_NN,O_NN)
                pnnpF = O2b_Pandya_transformation_pnnpF(a,b,c,d,J,Orb,Orb_NN,O_NN)
                nnF = O2b_Pandya_transformation_nnF(a,b,c,d,J,Orb,Orb_NN,O_NN)

                F_NN.pp[P,J+1][Bra,Ket] = ppF
                F_NN.pnpn[P,J+1][Bra,Ket] = pnpnF
                F_NN.pnnp[P,J+1][Bra,Ket] = pnnpF
                F_NN.nn[P,J+1][Bra,Ket] = nnF
            end
        end
    end

    return F_NN
end