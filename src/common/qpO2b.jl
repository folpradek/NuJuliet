function qpO2b_initialize(Params::Parameters,Orb::Vector{Orb1B};Make_Orb_NN::Bool=false)
    # Read parameters ...
    N_max, N_2max = Params.Calc.Nmax, Params.Calc.N2max
    a_max = div((N_max + 1)*(N_max + 2),2)
    J_max = Int64(N_2max + 1)

    # Initiliaze array for counting of NN orbitals ...
    N_Orb_NN = zeros(Int64,2,2,J_max+1)

    # Initiliaze dictionary for NN orbitals ...
    Orb_NN_Dic = Dict{UInt64,Int64}()

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

    # Allocate the dictionary and inidices for NN orbitals ...
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
                        if b <= a
                            #key_1 = (Int8(1),Int8(P),Int8(J),Int16(a),Int16(b))
                            Key_1 = O2b_key(a,b,J,P,1)
                            N_Orb_NN[2,P,J+1] += 1
                            Orb_NN_Dic[Key_1] = Int32(N_Orb_NN[2,P,J+1])
                            Ind_Orb_NN[2,P,J+1][N_Orb_NN[2,P,J+1]] = zeros(Int64, 2)
                            Ind_Orb_NN[2,P,J+1][N_Orb_NN[2,P,J+1]][1] = a
                            Ind_Orb_NN[2,P,J+1][N_Orb_NN[2,P,J+1]][2] = b
                        end
                        #key_0 = (Int8(0),Int8(P),Int8(J),Int16(a),Int16(b))
                        Key_0 = O2b_key(a,b,J,P,0)
                        N_Orb_NN[1,P,J+1] += 1
                        Orb_NN_Dic[Key_0] = Int32(N_Orb_NN[1,P,J+1])
                        Ind_Orb_NN[1,P,J+1][N_Orb_NN[1,P,J+1]] = zeros(Int64, 2)
                        Ind_Orb_NN[1,P,J+1][N_Orb_NN[1,P,J+1]][1] = a
                        Ind_Orb_NN[1,P,J+1][N_Orb_NN[1,P,J+1]][2] = b
                    end
                end
            end
        end
    end

    # Initialize the NN interaction matrices ...
        O40_pp = Matrix{Vector{Float64}}(undef,2,J_max+1)
        O40_pn = Matrix{Matrix{Float64}}(undef,2,J_max+1)
        O40_nn = Matrix{Vector{Float64}}(undef,2,J_max+1)

        #O31_pp = Matrix{Vector{Float64}}(undef,2,J_max+1)
        #O31_pn_2011 = Matrix{Vector{Float64}}(undef,2,J_max+1)
        #O31_pn_1120 = Matrix{Vector{Float64}}(undef,2,J_max+1)
        #O31_nn = Matrix{Vector{Float64}}(undef,2,J_max+1)
        O31_pp = Matrix{Matrix{Float64}}(undef,2,J_max+1)
        O31_pn_2011 = Matrix{Matrix{Float64}}(undef,2,J_max+1)
        O31_pn_1120 = Matrix{Matrix{Float64}}(undef,2,J_max+1)
        O31_nn = Matrix{Matrix{Float64}}(undef,2,J_max+1)


        O22_pp = Matrix{Vector{Float64}}(undef,2,J_max+1)
        O22_pn2002 = Matrix{Matrix{Float64}}(undef,2,J_max+1)
        O22_pn1111 = Matrix{Vector{Float64}}(undef,2,J_max+1)
        O22_nn = Matrix{Vector{Float64}}(undef,2,J_max+1)

    
    # Allocate entries of H ...
    @inbounds for P = 1:2
        @inbounds Threads.@threads for J=0:J_max
            L_T0_symm = div(N_Orb_NN[1,P,J+1]*(N_Orb_NN[1,P,J+1]+1),2)
            L_T1_symm = div(N_Orb_NN[2,P,J+1]*(N_Orb_NN[2,P,J+1]+1),2)
            L_T0_full = N_Orb_NN[1,P,J+1]
            L_T1_full = N_Orb_NN[2,P,J+1]

                O40_pp[P,J+1] = zeros(Float64,L_T1_symm)
                O40_pn[P,J+1] = zeros(Float64,L_T1_full,L_T1_full)
                O40_nn[P,J+1] = zeros(Float64,L_T1_symm)

                #O31_pp[P,J+1] = zeros(Float64,Length_T1)
                #O31_pn_2011[P,J+1] = zeros(Float64,Length_T0)
                #O31_pn_1120[P,J+1] = zeros(Float64,Length_T0)
                #O31_nn[P,J+1] = zeros(Float64,Length_T1)
                O31_pp[P,J+1] = zeros(Float64,L_T0_full,L_T0_full)
                O31_pn_2011[P,J+1] = zeros(Float64,L_T0_full,L_T0_full)
                O31_pn_1120[P,J+1] = zeros(Float64,L_T0_full,L_T0_full)
                O31_nn[P,J+1] = zeros(Float64,L_T0_full,L_T0_full)


                O22_pp[P,J+1] = zeros(Float64,L_T1_symm)
                O22_pn2002[P,J+1] = zeros(Float64,L_T1_full,L_T1_full)
                O22_pn1111[P,J+1] = zeros(Float64,L_T0_symm)
                O22_nn[P,J+1] = zeros(Float64,L_T1_symm)
        end
    end

    # Allocate the NN interaction in quasiparticle picture ...
    O_NN = qpO2B(qpO40(O40_pp,O40_pn,O40_nn),
                 qpO31(O31_pp,O31_pn_2011,O31_pn_1120,O31_nn),
                 qpO22(O22_pp,O22_pn2002,O22_pn1111,O22_nn))

    if Make_Orb_NN == false
        return O_NN
    elseif Make_Orb_NN == true
        # Allocate the array for NN orbitals ...
        Orb_NN = Orb2B(Orb_NN_Dic,N_Orb_NN,Ind_Orb_NN)
        return O_NN, Orb_NN
    end
end

function qpO2b(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,O_NN::O2B,C::O1B,U::O1B,V::O1B)
    println("\nPreparing the given 2-body NN operator O expressed in the J-scheme quasiparticle representation ...")

    # Perform the Canonical Transformation of the given 2-body operator into the target basis ...
    O_NN = O2b_transformation(Params,Orb,Orb_NN,O_NN,C)

    # Initialize the 2-body quasiparticle Hamiltonian H^(kl) ...
    Q_NN = qpO2b_initialize(Params,Orb)

    # Calculate the Pandya transformation of O_NN ...
    @time F_NN = O2b_Pandya_transformation(Params,Orb,Orb_NN,O_NN)

    # Allocate H^(40) ...
    @time Q_NN = qpO2b_40_allocate(Params,Orb,Orb_NN,Q_NN,O_NN,F_NN,U,V)
        #@time Q_NN = qpO2b_40_allocate_canonical(Params,Orb,Orb_NN,Q_NN,O_NN,U,V)

        # Under construction ....
        # Allocate H^(31) ...
        #Q_NN = qpO2b_31_allocate(Params,Orb,Orb_NN,Q_NN,O_NN,U,V)

    # Allocate H^(22) ...
    @time Q_NN = qpO2b_22_allocate(Params,Orb,Orb_NN,Q_NN,O_NN,F_NN,U,V)
        #@time Q_NN = qpO2b_22_allocate_canonical(Params,Orb,Orb_NN,Q_NN,O_NN,U,V)

    # Deallocate the Pandya transformed O_NN temporary operator F_NN ...
    F_NN = nothing
    
    # Perform the Garbage Collection ...
    GC.gc()
    
    println("\nThe given 2-body NN operator O expressed in the quasiparticke represation with components (kl) ...\n")

    return Q_NN
end

function qpO2b_export(Params::Parameters,Orb_NN::Orb2B,O_NN::qpO2B,Export_Path::String)
    # Read parameters ...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1

    # Prepare buffer for export  ...
    Buffer_size, Buffer_count = 10000000, 0
    Buffer = Vector{Float64}(undef,Buffer_size)

    # Export the given 2-body quasiparticle NN operator ...
    println("\nExporting the given quasiparticle 2-body operator in the binary format ...")
    open(Export_Path, "w") do Export_File
        # ppO40
        @inbounds for J in 0:J_max
            @inbounds for P in 1:2
                N = Orb_NN.N[2,P,J+1]
                @inbounds for Bra in 1:N
                    @inbounds for Ket in 1:Bra
                        Ind = Bra + (Ket - 1) * N - div(Ket * (Ket - 1),2)
                        @inbounds ME = O_NN.qp40.pp[P,J+1][Ind]
                        Buffer_count += 1
                        Buffer[Buffer_count] = Float64(ME)
                        if Buffer_count == Buffer_size
                            write(Export_File,Buffer)
                            Buffer_count = 0
                        end
                    end
                end
            end
        end

        # pnO40
        @inbounds for J in 0:J_max
            @inbounds for P in 1:2
                N = Orb_NN.N[2,P,J+1]
                @inbounds for Bra in 1:N
                    @inbounds for Ket in 1:N
                        @inbounds ME = O_NN.qp40.pn[P,J+1][Bra,Ket]
                        Buffer_count += 1
                        Buffer[Buffer_count] = Float64(ME)
                        if Buffer_count == Buffer_size
                            write(Export_File,Buffer)
                            Buffer_count = 0
                        end
                    end
                end
            end
        end

        # nnO40
        @inbounds for J in 0:J_max
            @inbounds for P in 1:2
                N = Orb_NN.N[2,P,J+1]
                @inbounds for Bra in 1:N
                    @inbounds for Ket in 1:Bra
                        Ind = Bra + (Ket - 1) * N - div(Ket * (Ket - 1),2)
                        @inbounds ME = O_NN.qp40.nn[P,J+1][Ind]
                        Buffer_count += 1
                        Buffer[Buffer_count] = Float64(ME)
                        if Buffer_count == Buffer_size
                            write(Export_File,Buffer)
                            Buffer_count = 0
                        end
                    end
                end
            end
        end

        # ppO31
        @inbounds for J in 0:J_max
            @inbounds for P in 1:2
                N = Orb_NN.N[1,P,J+1]
                @inbounds for Bra in 1:N
                    @inbounds for Ket in 1:N
                        @inbounds ME = O_NN.qp31.pp[P,J+1][Bra,Ket]
                        Buffer_count += 1
                        Buffer[Buffer_count] = Float64(ME)
                        if Buffer_count == Buffer_size
                            write(Export_File,Buffer)
                            Buffer_count = 0
                        end
                    end
                end
            end
        end

        # pnO2011
        @inbounds for J in 0:J_max
            @inbounds for P in 1:2
                N = Orb_NN.N[1,P,J+1]
                @inbounds for Bra in 1:N
                    @inbounds for Ket in 1:N
                        @inbounds ME = O_NN.qp31.pn2011[P,J+1][Bra,Ket]
                        Buffer_count += 1
                        Buffer[Buffer_count] = Float64(ME)
                        if Buffer_count == Buffer_size
                            write(Export_File,Buffer)
                            Buffer_count = 0
                        end
                    end
                end
            end
        end

        # pnO1120
        @inbounds for J in 0:J_max
            @inbounds for P in 1:2
                N = Orb_NN.N[1,P,J+1]
                @inbounds for Bra in 1:N
                    @inbounds for Ket in 1:N
                        @inbounds ME = O_NN.qp31.pn1120[P,J+1][Bra,Ket]
                        Buffer_count += 1
                        Buffer[Buffer_count] = Float64(ME)
                        if Buffer_count == Buffer_size
                            write(Export_File,Buffer)
                            Buffer_count = 0
                        end
                    end
                end
            end
        end

        # nnO31
        @inbounds for J in 0:J_max
            @inbounds for P in 1:2
                N = Orb_NN.N[1,P,J+1]
                @inbounds for Bra in 1:N
                    @inbounds for Ket in 1:N
                        @inbounds ME = O_NN.qp31.nn[P,J+1][Bra,Ket]
                        Buffer_count += 1
                        Buffer[Buffer_count] = Float64(ME)
                        if Buffer_count == Buffer_size
                            write(Export_File,Buffer)
                            Buffer_count = 0
                        end
                    end
                end
            end
        end

        # ppO22
        @inbounds for J in 0:J_max
            @inbounds for P in 1:2
                N = Orb_NN.N[2,P,J+1]
                @inbounds for Bra in 1:N
                    @inbounds for Ket in 1:Bra
                        Ind = Bra + (Ket - 1) * N - div(Ket * (Ket - 1),2)
                        @inbounds ME = O_NN.qp22.pp[P,J+1][Ind]
                        Buffer_count += 1
                        Buffer[Buffer_count] = Float64(ME)
                        if Buffer_count == Buffer_size
                            write(Export_File,Buffer)
                            Buffer_count = 0
                        end
                    end
                end
            end
        end

        # pnO2002
        @inbounds for J in 0:J_max
            @inbounds for P in 1:2
                N = Orb_NN.N[2,P,J+1]
                @inbounds for Bra in 1:N
                    @inbounds for Ket in 1:N
                        @inbounds ME = O_NN.qp22.pn2002[P,J+1][Bra,Ket]
                        Buffer_count += 1
                        Buffer[Buffer_count] = Float64(ME)
                        if Buffer_count == Buffer_size
                            write(Export_File,Buffer)
                            Buffer_count = 0
                        end
                    end
                end
            end
        end

        # pnO1111
        @inbounds for J in 0:J_max
            @inbounds for P in 1:2
                N = Orb_NN.N[1,P,J+1]
                @inbounds for Bra in 1:N
                    @inbounds for Ket in 1:Bra
                        Ind = Bra + (Ket - 1) * N - div(Ket * (Ket - 1),2)
                        @inbounds ME = O_NN.qp22.pn1111[P,J+1][Ind]
                        Buffer_count += 1
                        Buffer[Buffer_count] = Float64(ME)
                        if Buffer_count == Buffer_size
                            write(Export_File,Buffer)
                            Buffer_count = 0
                        end
                    end
                end
            end
        end

        # nnO22
        @inbounds for J in 0:J_max
            @inbounds for P in 1:2
                N = Orb_NN.N[2,P,J+1]
                @inbounds for Bra in 1:N
                    @inbounds for Ket in 1:Bra
                        Ind = Bra + (Ket - 1) * N - div(Ket * (Ket - 1),2)
                        @inbounds ME = O_NN.qp22.nn[P,J+1][Ind]
                        Buffer_count += 1
                        Buffer[Buffer_count] = Float64(ME)
                        if Buffer_count == Buffer_size
                            write(Export_File,Buffer)
                            Buffer_count = 0
                        end
                    end
                end
            end
        end

        if Buffer_count > 0
            write(Export_File,Buffer[1:Buffer_count])
        end
    end

    println("\nGiven 2-body quasiparticle operator succesfully exported in the binary format ...")

    return
end

function qpO2b_import(Params::Parameters,Orb::Vector{Orb1B},Import_Path::String)
    # Initialize the 2-body quasiparticle NN operator...
    O_NN, Orb_NN = qpO2b_initialize(Params,Orb,Make_Orb_NN=true)

    # Precalculate chunks of data for parallelized reading ...
    HJP_chunk, N_chunk, N_chunk_skip = qpO2b_import_count(Params,Orb_NN)

    # Memory mapping of the given 2-body quasiparticle NN operator into memory ...
    qpO = Mmap.mmap(Import_Path)

    # Define auxilliary function for reading chunks ...
    @inline function qpO2b_read_chunk(qpO::Vector{UInt8},N_skip::Int64,N::Int64)
        Bytes = @view qpO[(N_skip + 1):(N_skip + 8*N)]
        return reinterpret(Float64,Bytes)
    end

    # Define the buffer size ...
    Buffer_size = 1000000

    # Import the given 2-body quasiparticle NN operator ...
    println("\nReading given 2-body quasiparticle NN operator ...")
    @inbounds Threads.@threads for HJP in HJP_chunk
        H, J, P = HJP[1], HJP[2], HJP[3]

        N = N_chunk[H,J+1,P]
        N_skip = N_chunk_skip[H,J+1,P]
        Buffer_n, Buffer_N, Buffer = 1, min(N,Buffer_size), Float64[]

        if H == 1
            N_T = Orb_NN.N[2,P,J+1]
            @inbounds for Bra in 1:N_T
                @inbounds for Ket in 1:Bra
                    if (Buffer_n > Buffer_N) || (Buffer_n == 1)
                        Buffer_N = min(N,Buffer_size)
                        N = N - Buffer_N
                        Buffer_n = 1
                        Buffer = qpO2b_read_chunk(qpO,N_skip,Buffer_N)
                        N_skip = N_skip + 8*Buffer_N
                    end
                    Ind = Bra + (Ket - 1) * N_T - div(Ket * (Ket - 1),2)
                    ME = Buffer[Buffer_n]
                    Buffer_n += 1
                    O_NN.qp40.pp[P,J+1][Ind] = ME
                end
            end

        elseif H == 2
            N_T = Orb_NN.N[2,P,J+1]
            @inbounds for Bra in 1:N_T
                @inbounds for Ket in 1:N_T
                    if (Buffer_n > Buffer_N) || (Buffer_n == 1)
                        Buffer_N = min(N,Buffer_size)
                        N = N - Buffer_N
                        Buffer_n = 1
                        Buffer = qpO2b_read_chunk(qpO,N_skip,Buffer_N)
                        N_skip = N_skip + 8*Buffer_N
                    end
                    ME = Buffer[Buffer_n]
                    Buffer_n += 1
                    O_NN.qp40.pn[P,J+1][Bra,Ket] = ME
                end
            end

        elseif H == 3
            N_T = Orb_NN.N[2,P,J+1]
            @inbounds for Bra in 1:N_T
                @inbounds for Ket in 1:Bra
                    if (Buffer_n > Buffer_N) || (Buffer_n == 1)
                        Buffer_N = min(N,Buffer_size)
                        N = N - Buffer_N
                        Buffer_n = 1
                        Buffer = qpO2b_read_chunk(qpO,N_skip,Buffer_N)
                        N_skip = N_skip + 8*Buffer_N
                    end
                    Ind = Bra + (Ket - 1) * N_T - div(Ket * (Ket - 1),2)
                    ME = Buffer[Buffer_n]
                    Buffer_n += 1
                    O_NN.qp40.nn[P,J+1][Ind] = ME
                end
            end

        elseif H == 4
            N_T = Orb_NN.N[1,P,J+1]
            @inbounds for Bra in 1:N_T
                @inbounds for Ket in 1:N_T
                    if (Buffer_n > Buffer_N) || (Buffer_n == 1)
                        Buffer_N = min(N,Buffer_size)
                        N = N - Buffer_N
                        Buffer_n = 1
                        Buffer = qpO2b_read_chunk(qpO,N_skip,Buffer_N)
                        N_skip = N_skip + 8*Buffer_N
                    end
                    ME = Buffer[Buffer_n]
                    Buffer_n += 1
                    O_NN.qp31.pp[P,J+1][Bra,Ket] = ME
                end
            end

        elseif H == 5
            N_T = Orb_NN.N[1,P,J+1]
            @inbounds for Bra in 1:N_T
                @inbounds for Ket in 1:N_T
                    if (Buffer_n > Buffer_N) || (Buffer_n == 1)
                        Buffer_N = min(N,Buffer_size)
                        N = N - Buffer_N
                        Buffer_n = 1
                        Buffer = qpO2b_read_chunk(qpO,N_skip,Buffer_N)
                        N_skip = N_skip + 8*Buffer_N
                    end
                    ME = Buffer[Buffer_n]
                    Buffer_n += 1
                    O_NN.qp31.pn2011[P,J+1][Bra,Ket] = ME
                end
            end

        elseif H == 6
            N_T = Orb_NN.N[1,P,J+1]
            @inbounds for Bra in 1:N_T
                @inbounds for Ket in 1:N_T
                    if (Buffer_n > Buffer_N) || (Buffer_n == 1)
                        Buffer_N = min(N,Buffer_size)
                        N = N - Buffer_N
                        Buffer_n = 1
                        Buffer = qpO2b_read_chunk(qpO,N_skip,Buffer_N)
                        N_skip = N_skip + 8*Buffer_N
                    end
                    ME = Buffer[Buffer_n]
                    Buffer_n += 1
                    O_NN.qp31.pn1120[P,J+1][Bra,Ket] = ME
                end
            end
        elseif H == 7
            N_T = Orb_NN.N[1,P,J+1]
            @inbounds for Bra in 1:N_T
                @inbounds for Ket in 1:N_T
                    if (Buffer_n > Buffer_N) || (Buffer_n == 1)
                        Buffer_N = min(N,Buffer_size)
                        N = N - Buffer_N
                        Buffer_n = 1
                        Buffer = qpO2b_read_chunk(qpO,N_skip,Buffer_N)
                        N_skip = N_skip + 8*Buffer_N
                    end
                    ME = Buffer[Buffer_n]
                    Buffer_n += 1
                    O_NN.qp31.nn[P,J+1][Bra,Ket] = ME
                end
            end

        elseif H == 8
            N_T = Orb_NN.N[2,P,J+1]
            @inbounds for Bra in 1:N_T
                @inbounds for Ket in 1:Bra
                    if (Buffer_n > Buffer_N) || (Buffer_n == 1)
                        Buffer_N = min(N,Buffer_size)
                        N = N - Buffer_N
                        Buffer_n = 1
                        Buffer = qpO2b_read_chunk(qpO,N_skip,Buffer_N)
                        N_skip = N_skip + 8*Buffer_N
                    end
                    Ind = Bra + (Ket - 1) * N_T - div(Ket * (Ket - 1),2)
                    ME = Buffer[Buffer_n]
                    Buffer_n += 1
                    O_NN.qp22.pp[P,J+1][Ind] = ME
                end
            end
        
        elseif H == 9
            N_T = Orb_NN.N[2,P,J+1]
            @inbounds for Bra in 1:N_T
                @inbounds for Ket in 1:N_T
                    if (Buffer_n > Buffer_N) || (Buffer_n == 1)
                        Buffer_N = min(N,Buffer_size)
                        N = N - Buffer_N
                        Buffer_n = 1
                        Buffer = qpO2b_read_chunk(qpO,N_skip,Buffer_N)
                        N_skip = N_skip + 8*Buffer_N
                    end
                    ME = Buffer[Buffer_n]
                    Buffer_n += 1
                    O_NN.qp22.pn2002[P,J+1][Bra,Ket] = ME
                end
            end
        
        elseif H == 10
            N_T = Orb_NN.N[1,P,J+1]
            @inbounds for Bra in 1:N_T
                @inbounds for Ket in 1:Bra
                    if (Buffer_n > Buffer_N) || (Buffer_n == 1)
                        Buffer_N = min(N,Buffer_size)
                        N = N - Buffer_N
                        Buffer_n = 1
                        Buffer = qpO2b_read_chunk(qpO,N_skip,Buffer_N)
                        N_skip = N_skip + 8*Buffer_N
                    end
                    Ind = Bra + (Ket - 1) * N_T - div(Ket * (Ket - 1),2)
                    ME = Buffer[Buffer_n]
                    Buffer_n += 1
                    O_NN.qp22.pn1111[P,J+1][Ind] = ME
                end
            end

        elseif H == 11
            N_T = Orb_NN.N[2,P,J+1]
            @inbounds for Bra in 1:N_T
                @inbounds for Ket in 1:Bra
                    if (Buffer_n > Buffer_N) || (Buffer_n == 1)
                        Buffer_N = min(N,Buffer_size)
                        N = N - Buffer_N
                        Buffer_n = 1
                        Buffer = qpO2b_read_chunk(qpO,N_skip,Buffer_N)
                        N_skip = N_skip + 8*Buffer_N
                    end
                    Ind = Bra + (Ket - 1) * N_T - div(Ket * (Ket - 1),2)
                    ME = Buffer[Buffer_n]
                    Buffer_n += 1
                    O_NN.qp22.nn[P,J+1][Ind] = ME
                end
            end

        end

    end

    println("\nGiven 2-body quasiparticle NN operator succesfully imported ...")

    return O_NN, Orb_NN
end

function qpO2b_import_count(Params::Parameters,Orb_NN::Orb2B)
    # Read parameters ...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1

    # Initialize H, J & P values ...
    HJP = Vector{Tuple{Int64,Int64,Int64}}(undef,11*2*(J_max+1))
    HJP_chunk = Array{Int64}(undef,11,J_max+1,2)
    HJP_chunk_skip = Array{Int64}(undef,11,J_max+1,2)

    # Allocate the values of HJP ...
    c = 0
    @inbounds for H in 1:11
        @inbounds for J in 0:J_max
            @inbounds for P in 1:2
                c += 1
                HJP[c] = (H,J,P)
            end
        end
    end

    # Calculate number of MEs in each HJP_chunk ... 
    @inbounds for i in 1:(11*(J_max+1)*2)
        H, J, P = HJP[i][1], HJP[i][2], HJP[i][3]
        n = 0

        if H == 1 || H == 3 || H == 8 || H == 11
            N = Orb_NN.N[2,P,J+1]
            @inbounds for Bra in 1:N
                @inbounds for Ket in 1:Bra
                    n += 1
                end
            end
        elseif H == 2 || H == 9
            N = Orb_NN.N[2,P,J+1]
            @inbounds for Bra in 1:N
                @inbounds for Ket in 1:N
                    n += 1
                end
            end
        elseif H == 4 || H == 5 || H == 6 || H == 7
            N = Orb_NN.N[1,P,J+1]
            @inbounds for Bra in 1:N
                @inbounds for Ket in 1:N
                    n += 1
                end
            end
        elseif H == 10
            N = Orb_NN.N[1,P,J+1]
            @inbounds for Bra in 1:N
                @inbounds for Ket in 1:Bra
                    n += 1
                end
            end
        end


        # To be included later ...
        #=
            if H == 1 || H == 3 || H == 4 || H == 7 || H == 8 || H == 12
                N_T = Orb_NN.N[2,P,J+1]
                @inbounds for Bra in 1:N_T
                    @inbounds for Ket in 1:Bra
                        N += 1
                    end
                end

            else
                N_T = Orb_NN.N[1,P,J+1]
                @inbounds for Bra in 1:N_T
                    @inbounds for Ket in 1:N_T
                        N += 1
                    end
                end
            end
        =#

        HJP_chunk[H,J+1,P] = n
    end

    # Calculate the number of bytes for each HJP chunk to be skipped ...
    @inbounds for i in 1:(11*(J_max+1)*2)
        H, J, P = HJP[i][1], HJP[i][2], HJP[i][3]
        n = 0
        @inbounds for j in 1:(i-1)
            H_s, J_s, P_s = HJP[j][1], HJP[j][2], HJP[j][3]
            n += 8*HJP_chunk[H_s,J_s+1,P_s]
        end
        HJP_chunk_skip[H,J+1,P] = n
    end

    return HJP, HJP_chunk, HJP_chunk_skip
end