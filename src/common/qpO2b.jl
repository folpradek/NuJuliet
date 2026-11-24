function qpO2b_initialize(Params::Parameters,Orb::Vector{Orb1B};Make_Orb_NN::Bool=false)
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
        Orb_NN = Orb2B(Orb_NN_Map,N_Orb_NN,Ind_Orb_NN)
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

            # Under construction ....
        # Allocate H^(31) ...
        #Q_NN = qpO2b_31_allocate(Params,Orb,Orb_NN,Q_NN,O_NN,U,V)

    # Allocate H^(22) ...
    @time Q_NN = qpO2b_22_allocate(Params,Orb,Orb_NN,Q_NN,O_NN,F_NN,U,V)

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

    # Define local functions ...
        # Define auxilliary function for counting the number of MEs in each (H,J,P) chunk ...
    function qpO2b_import_count(Params::Parameters,Orb_NN::Orb2B)
        # Read parameters (same as in qpO2b_export) ...
        N_max = Params.Calc.Nmax
        N_2max = 2*N_max
        J_max = N_2max + 1

        # Components H of the 2-body quasiparticle operator in the order as they are exported ...
            # H = 1 ... ppO40,  2 ... pnO40,   3 ... nnO40,
            # H = 4 ... ppO31,  5 ... pnO2011, 6 ... pnO1120, 7 ... nnO31,
            # H = 8 ... ppO22,  9 ... pnO2002, 10 ... pnO1111, 11 ... nnO22
        N_H = 11

        # Index of NN orbitals in Orb_NN.N (1 ... all pairs (a,b), 2 ... pairs with b <= a) for each H ...
        H_Orb = [2, 2, 2, 1, 1, 1, 1, 2, 2, 1, 2]

        # Storage of each H (true ... symmetric, only Bra >= Ket, false ... full matrix) ...
        H_symm = [true, false, true, false, false, false, false, true, false, true, true]

        # Initialize the list of (H,J,P) chunks, number of MEs in each chunk & number of MEs to be skipped before each chunk ...
        HJP_list = Vector{Vector{Int64}}(undef,N_H*2*(J_max+1))
        N_chunk = zeros(Int64,N_H,2,J_max+1)
        N_chunk_skip = zeros(Int64,N_H,2,J_max+1)

        # Calculate N_chunk & N_chunk_skip by going through the chunks in the same order as in qpO2b_export ...
        c, N_ME = 0, 0
        @inbounds for H in 1:N_H
            @inbounds for J in 0:J_max
                @inbounds for P in 1:2
                    N = Orb_NN.N[H_Orb[H],P,J+1]
                    if H_symm[H] == true
                        N_chunk[H,P,J+1] = div(N*(N+1),2)
                    else
                        N_chunk[H,P,J+1] = N^2
                    end
                    N_chunk_skip[H,P,J+1] = N_ME
                    N_ME += N_chunk[H,P,J+1]

                    c += 1
                    HJP_list[c] = [H, J, P]
                end
            end
        end

        # Sort the chunks from the largest to the smallest for better load balancing of threads ...
        sort!(HJP_list, by = HJP -> N_chunk[HJP[1],HJP[3],HJP[2]+1], rev = true)

        return HJP_list, N_chunk, N_chunk_skip, N_ME
    end
        # Define auxilliary functions for distributing the chunks into O_NN ...
    function qpO2b_import_symm!(O::Vector{Float64},Buffer::Vector{Float64},N::Int64)
        # Symmetric chunk ... exported for Bra in 1:N & Ket in 1:Bra ... stored in O as packed lower triangle ...
        n = 0
        @inbounds for Bra in 1:N
            @inbounds for Ket in 1:Bra
                Ind = Bra + (Ket - 1) * N - div(Ket * (Ket - 1),2)
                n += 1
                O[Ind] = Buffer[n]
            end
        end
        return
    end
        # Define auxilliary function for distributing the chunks into O_NN ...
    function qpO2b_import_full!(O::Matrix{Float64},Buffer::Vector{Float64},N::Int64)
        # Full chunk ... exported row by row (Bra outer, Ket inner), whereas Julia stores matrices column by column ...
            # Thus the chunk viewed as N x N matrix is the transposition of O[Bra,Ket] ... transpose! is cache-friendly (blocked) ...
        transpose!(O,reshape(Buffer,N,N))
        return
    end

    # Initialize the 2-body quasiparticle NN operator...
    O_NN, Orb_NN = qpO2b_initialize(Params,Orb,Make_Orb_NN=true)

    # Precalculate chunks of data for parallelized reading ...
    HJP_list, N_chunk, N_chunk_skip, N_ME = qpO2b_import_count(Params,Orb_NN)

    # Check the size of the given binary file (8 bytes per ME) ...
    if filesize(Import_Path) != 8*N_ME
        error("Size of the binary file ''" * Import_Path * "'' (" * string(filesize(Import_Path)) *
              " B) does not match the expected size of the 2-body quasiparticle operator (" * string(8*N_ME) * " B) ...")
    end

    # Import the given 2-body quasiparticle NN operator ...
        # Each (H,J,P) chunk is read at once into a buffer by its own task & then distributed into O_NN ...
        # Chunks are sorted from the largest to the smallest ...
    println("\nReading given 2-body quasiparticle NN operator ...")
    @sync for HJP in HJP_list
        H, J, P = HJP[1], HJP[2], HJP[3]

        # Skip empty chunks ...
        if N_chunk[H,P,J+1] == 0
            continue
        end

        Threads.@spawn begin
            N_T0 = Orb_NN.N[1,P,J+1]
            N_T1 = Orb_NN.N[2,P,J+1]

            # Read the whole (H,J,P) chunk (8 bytes per ME) ...
            Buffer = Vector{Float64}(undef,N_chunk[H,P,J+1])
            open(Import_Path, "r") do Bin_Read
                seek(Bin_Read, 8*N_chunk_skip[H,P,J+1])
                read!(Bin_Read, Buffer)
            end

            # O40 ...
            if H == 1
                qpO2b_import_symm!(O_NN.qp40.pp[P,J+1],Buffer,N_T1)
            elseif H == 2
                qpO2b_import_full!(O_NN.qp40.pn[P,J+1],Buffer,N_T1)
            elseif H == 3
                qpO2b_import_symm!(O_NN.qp40.nn[P,J+1],Buffer,N_T1)

            # O31 ...
            elseif H == 4
                qpO2b_import_full!(O_NN.qp31.pp[P,J+1],Buffer,N_T0)
            elseif H == 5
                qpO2b_import_full!(O_NN.qp31.pn2011[P,J+1],Buffer,N_T0)
            elseif H == 6
                qpO2b_import_full!(O_NN.qp31.pn1120[P,J+1],Buffer,N_T0)
            elseif H == 7
                qpO2b_import_full!(O_NN.qp31.nn[P,J+1],Buffer,N_T0)

            # O22 ...
            elseif H == 8
                qpO2b_import_symm!(O_NN.qp22.pp[P,J+1],Buffer,N_T1)
            elseif H == 9
                qpO2b_import_full!(O_NN.qp22.pn2002[P,J+1],Buffer,N_T1)
            elseif H == 10
                qpO2b_import_symm!(O_NN.qp22.pn1111[P,J+1],Buffer,N_T0)
            elseif H == 11
                qpO2b_import_symm!(O_NN.qp22.nn[P,J+1],Buffer,N_T1)
            end
        end
    end

    println("\nGiven 2-body quasiparticle NN operator succesfully imported ...")

    return O_NN, Orb_NN
end