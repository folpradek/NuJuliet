function qpO2b_31_allocate(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,Q_NN::qpO2B,O_NN::O2B,U::O1B,V::O1B)
    println("\nAllocating (31) components of the given operator in the quasiparticle representation ...")

    # Preallocate arrays of allowed values of J & P ...
    JP = JP_initialize(Params.Calc.N2max + 1)

    # Initialize temporary operator arrays ...
    Q_NN_t1, Q_NN_t2, Orb_NN_t = qpO2b_31_initialize(Params,Orb)

    # Transformation of the 1st index ...
    @time Q_NN_t1 = qpO2b_31_allocate_ind1(Params,JP,Orb,Orb_NN,Orb_NN_t,O_NN,Q_NN_t1,U,V)

    # Transformation of the 2nd index ...
    @time Q_NN_t2 = qpO2b_31_allocate_ind2(Params,JP,Orb,Orb_NN_t,Q_NN_t1,Q_NN_t2,U,V)

    # Transformation of the 3rd index ...
    @time Q_NN_t1 = qpO2b_31_allocate_ind3(Params,JP,Orb,Orb_NN_t,Q_NN_t1,Q_NN_t2,U,V)

    # Transformation of the 4th index ...
    @time O_NN = qpO2b_31_allocate_ind4(Params,JP,Orb,Orb_NN,Orb_NN_t,Q_NN,Q_NN_t1,U,V)

    # Deallocate temporary arrays ...
    Q_NN_t1 = nothing
    Q_NN_t2 = nothing

    # Perform the Garbage Collection ...
    GC.gc()

    println("\nAllocation of (31) components of the given 2-body NN quasiparticle operator done ...")

    return Q_NN
end

function qpO2b_31_initialize(Params::Parameters,Orb::Vector{Orb1B})
    # Read parameters ...
    N_max = Params.Calc.Nmax
    N_2max = 2 * N_max
    a_max = div((N_max + 1)*(N_max + 2),2)
    J_max = Int64(2*N_max + 1)

    # Initiliaze array for counting of NN orbitals ...
    N_Orb_NN = zeros(Int64,2,J_max+1)

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
                        #key = (Int8(P),Int8(J),Int16(a),Int16(b))
                        Key = O2b_key_temp(a,b,J,P)
                        N_Orb_NN[P,J+1] += 1
                        Orb_NN_Dic[Key] = Int32(N_Orb_NN[P,J+1])
                        Ind_Orb_NN[P,J+1][N_Orb_NN[P,J+1]] = zeros(Int64, 2)
                        Ind_Orb_NN[P,J+1][N_Orb_NN[P,J+1]][1] = a
                        Ind_Orb_NN[P,J+1][N_Orb_NN[P,J+1]][2] = b
                    end
                end
            end
        end
    end

    # Initialize the NN operator matrices ...
        # Case of O_t1 ...
    O_pp_t1 = Vector{Matrix{Matrix{Float64}}}(undef,2)
    O_pn2011_t1 = Vector{Matrix{Matrix{Float64}}}(undef,2)
    O_pn1120_t1 = Vector{Matrix{Matrix{Float64}}}(undef,2)
    O_nn_t1 = Vector{Matrix{Matrix{Float64}}}(undef,2)
        # Case of O_t2 ...
    O_pp_t2 = Vector{Matrix{Matrix{Float64}}}(undef,2)
    O_pn2011_t2 = Vector{Matrix{Matrix{Float64}}}(undef,2)
    O_pn1120_t2 = Vector{Matrix{Matrix{Float64}}}(undef,2)
    O_nn_t2 = Vector{Matrix{Matrix{Float64}}}(undef,2)

    # Initialite sub-entries of O_t ...
    @inbounds for i in 1:2
        O_pp_t1[i] = Matrix{Matrix{Float64}}(undef,2,J_max+1)
        O_pn2011_t1[i] = Matrix{Matrix{Float64}}(undef,2,J_max+1)
        O_pn1120_t1[i] = Matrix{Matrix{Float64}}(undef,2,J_max+1)
        O_nn_t1[i] = Matrix{Matrix{Float64}}(undef,2,J_max+1)

        O_pp_t2[i] = Matrix{Matrix{Float64}}(undef,2,J_max+1)
        O_pn2011_t2[i] = Matrix{Matrix{Float64}}(undef,2,J_max+1)
        O_pn1120_t2[i] = Matrix{Matrix{Float64}}(undef,2,J_max+1)
        O_nn_t2[i] = Matrix{Matrix{Float64}}(undef,2,J_max+1)
    end
 
    # Allocate entries of O ...
    @inbounds for P = 1:2
        @inbounds Threads.@threads for J=0:J_max
            Length = N_Orb_NN[P,J+1]

            @inbounds for i in 1:2
                O_pp_t1[i][P,J+1] = zeros(Float64,Length,Length)
                O_pn2011_t1[i][P,J+1] = zeros(Float64,Length,Length)
                O_pn1120_t1[i][P,J+1] = zeros(Float64,Length,Length)
                O_nn_t1[i][P,J+1] = zeros(Float64,Length,Length)

                O_pp_t2[i][P,J+1] = zeros(Float64,Length,Length)
                O_pn2011_t2[i][P,J+1] = zeros(Float64,Length,Length)
                O_pn1120_t2[i][P,J+1] = zeros(Float64,Length,Length)
                O_nn_t2[i][P,J+1] = zeros(Float64,Length,Length)
            end

        end

    end

    # Allocate the temporary NN operators in the quasiparticle picture ...
    O_NN_t1 = O2B_31_Temp(O_pp_t1,O_pn2011_t1,O_pn1120_t1,O_nn_t1)
    O_NN_t2 = O2B_31_Temp(O_pp_t2,O_pn2011_t2,O_pn1120_t2,O_nn_t2)

    # Allocate the array for NN orbitals ...
    Orb_NN_t = Orb2B_Temp(Orb_NN_Dic,N_Orb_NN,Ind_Orb_NN)

    return O_NN_t1, O_NN_t2, Orb_NN_t
end

function qpO2b_31_allocate_ind1(Params::Parameters,JP::Vector{Vector{Int64}},Orb::Vector{Orb1B},Orb_NN::Orb2B,Orb_NN_t::Orb2B_Temp,O_NN::O2B,O_NN_t1::O2B_31_Temp,U::O1B,V::O1B)
    # Parameter initialization...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    a_max = div((N_max + 1)*(N_max + 2),2)

    # Define function for the transformation MEs for the 1st index ...
    @inline function O2b_temp_transformation_ind1_MEs(P::Int64,J::Int64,i::Int64,a::Int64,b::Int64,c::Int64,d::Int64,U::O1B,V::O1B,O_NN::O2B,Orb::Vector{Orb1B},Orb_NN::Orb2B)
        return @views U.p[i,a] * O2b_pp(i,b,c,d,J,P,O_NN,Orb,Orb_NN), V.p[i,a] * O2b_pp(i,b,c,d,J,P,O_NN,Orb,Orb_NN),
                      U.p[i,a] * O2b_pn(i,b,c,d,J,P,O_NN,Orb_NN), V.p[i,a] * O2b_pn(i,b,c,d,J,P,O_NN,Orb_NN),
                      U.n[i,a] * O2b_nn(i,b,c,d,J,P,O_NN,Orb,Orb_NN), V.n[i,a] * O2b_nn(i,b,c,d,J,P,O_NN,Orb,Orb_NN)
    end

    # Transformation of the 1st index ...
    println("\nPerforming Quasiparticle Transformation of the residual 2-body NN interaction H31 in the 1st index...")
    @inbounds Threads.@threads for i in JP
        J, P = i[1], i[2]
        if P == 1
            println("\tCalculating   ...   J = " * string(J) * "/" * string(N_2max+1) * "\tP = +")
        else
            println("\tCalculating   ...   J = " * string(J) * "/" * string(N_2max+1) * "\tP = -")
        end
        N = Orb_NN_t.N[P,J+1]
        @inbounds for Bra in 1:N
            a = Orb_NN_t.Ind[P,J+1][Bra][1]
            b = Orb_NN_t.Ind[P,J+1][Bra][2]
            l_a, j_a = Orb[a].l, Orb[a].j

            Orb_x = Orb_PreComp(a_max,j_a,l_a,Orb)
            @inbounds for Ket in 1:N
                c = Orb_NN_t.Ind[P,J+1][Ket][1]
                d = Orb_NN_t.Ind[P,J+1][Ket][2]

                OppSum1, OppSum2 = 0.0, 0.0
                Opn2011Sum1, Opn2011Sum2, Opn1120Sum1, Opn1120Sum2 = 0.0, 0.0, 0.0, 0.0
                OnnSum1, OnnSum2 = 0.0, 0.0
                
                @inbounds for i in Orb_x
                    OppUME, OppVME, OpnUME, OpnVME, OnnUME, OnnVME = O2b_temp_transformation_ind1_MEs(P,J,i,a,b,c,d,U,V,O_NN,Orb,Orb_NN)

                    OppSum1 += OppUME
                    OppSum2 += OppVME

                    Opn2011Sum1 += OpnUME
                    Opn2011Sum2 += OpnVME
                    Opn1120Sum1 += OpnUME
                    Opn1120Sum2 += OpnVME

                    OnnSum1 += OnnUME
                    OnnSum2 += OnnVME
                end

                @views O_NN_t1.pp[1][P,J+1][Bra,Ket] = OppSum1
                @views O_NN_t1.pp[2][P,J+1][Bra,Ket] = OppSum2

                @views O_NN_t1.pn2011[1][P,J+1][Bra,Ket] = Opn2011Sum1
                @views O_NN_t1.pn2011[2][P,J+1][Bra,Ket] = Opn2011Sum2
                @views O_NN_t1.pn1120[1][P,J+1][Bra,Ket] = Opn1120Sum1
                @views O_NN_t1.pn1120[2][P,J+1][Bra,Ket] = Opn1120Sum2

                @views O_NN_t1.nn[1][P,J+1][Bra,Ket] = OnnSum1
                @views O_NN_t1.nn[2][P,J+1][Bra,Ket] = OnnSum2
            end
        end
    end

    return O_NN_t1
end

function qpO2b_31_allocate_ind2(Params::Parameters,JP::Vector{Vector{Int64}},Orb::Vector{Orb1B},Orb_NN_t::Orb2B_Temp,O_NN_t1::O2B_31_Temp,O_NN_t2::O2B_31_Temp,U::O1B,V::O1B)
    # Parameter initialization...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    a_max = div((N_max + 1)*(N_max + 2),2)

    # Define function for the transformation MEs for the 2nd index ...
    @inline function O2b_temp_transformation_ind2_MEs(P::Int64,J::Int64,Bra::Int64,Ket::Int64,i::Int64,b::Int64,T::Matrix{Float64},O_NN_t::Matrix{Matrix{Float64}})
        return @views T[i,b] * O_NN_t[P,J+1][Bra,Ket]
    end

    # Transformation of the 2nd index ...
    println("\nPerforming Quasiparticle Transformation of the residual 2-body NN interaction H31 in the 2nd index...")

    @inbounds Threads.@threads for i in JP
        J, P = i[1], i[2]
        if P == 1
            println("\tCalculating   ...   J = " * string(J) * "/" * string(N_2max+1) * "\tP = +")
        else
            println("\tCalculating   ...   J = " * string(J) * "/" * string(N_2max+1) * "\tP = -")
        end
        N = Orb_NN_t.N[P,J+1]
        @inbounds for Bra in 1:N
            a = Orb_NN_t.Ind[P,J+1][Bra][1]
            b = Orb_NN_t.Ind[P,J+1][Bra][2]
            l_b, j_b = Orb[b].l, Orb[b].j

            Orb_x = Orb_PreComp(a_max,j_b,l_b,Orb)
            @inbounds for Ket in 1:N
                c = Orb_NN_t.Ind[P,J+1][Ket][1]
                d = Orb_NN_t.Ind[P,J+1][Ket][2]

                OppSum1, OppSum2 = 0.0, 0.0
                Opn2011Sum1, Opn2011Sum2, Opn1120Sum1, Opn1120Sum2 = 0.0, 0.0, 0.0, 0.0
                OnnSum1, OnnSum2 = 0.0, 0.0
                
                @inbounds for i in Orb_x
                    Bra_ai = O2b_temp_index(a,i,J,P,Orb_NN_t)

                    OppME1 = O2b_temp_transformation_ind2_MEs(P,J,Bra_ai,Ket,i,b,U.p,O_NN_t1.pp[1])
                    OppME2 = O2b_temp_transformation_ind2_MEs(P,J,Bra_ai,Ket,i,b,V.p,O_NN_t1.pp[2])

                    Opn2011ME1 = O2b_temp_transformation_ind2_MEs(P,J,Bra_ai,Ket,i,b,U.n,O_NN_t1.pn2011[1])
                    Opn2011ME2 = O2b_temp_transformation_ind2_MEs(P,J,Bra_ai,Ket,i,b,V.n,O_NN_t1.pn2011[2])
                    Opn1120ME1 = O2b_temp_transformation_ind2_MEs(P,J,Bra_ai,Ket,i,b,U.n,O_NN_t1.pn1120[1])
                    Opn1120ME2 = O2b_temp_transformation_ind2_MEs(P,J,Bra_ai,Ket,i,b,V.n,O_NN_t1.pn1120[2])

                    OnnME1 = O2b_temp_transformation_ind2_MEs(P,J,Bra_ai,Ket,i,b,U.n,O_NN_t1.nn[1])
                    OnnME2 = O2b_temp_transformation_ind2_MEs(P,J,Bra_ai,Ket,i,b,V.n,O_NN_t1.nn[2])

                    OppSum1 += OppME1
                    OppSum2 += OppME2

                    Opn2011Sum1 += Opn2011ME1
                    Opn2011Sum2 += Opn2011ME2
                    Opn1120Sum1 += Opn1120ME1
                    Opn1120Sum2 += Opn1120ME2

                    OnnSum1 += OnnME1
                    OnnSum2 += OnnME2
                end

                @views O_NN_t2.pp[1][P,J+1][Bra,Ket] = OppSum1
                @views O_NN_t2.pp[2][P,J+1][Bra,Ket] = OppSum2

                @views O_NN_t2.pn2011[1][P,J+1][Bra,Ket] = Opn2011Sum1
                @views O_NN_t2.pn2011[2][P,J+1][Bra,Ket] = Opn2011Sum2
                @views O_NN_t2.pn1120[1][P,J+1][Bra,Ket] = Opn1120Sum1
                @views O_NN_t2.pn1120[2][P,J+1][Bra,Ket] = Opn1120Sum2

                @views O_NN_t2.nn[1][P,J+1][Bra,Ket] = OnnSum1
                @views O_NN_t2.nn[2][P,J+1][Bra,Ket] = OnnSum2
            end
        end
    end

    return O_NN_t2
end

function qpO2b_31_allocate_ind3(Params::Parameters,JP::Vector{Vector{Int64}},Orb::Vector{Orb1B},Orb_NN_t::Orb2B_Temp,O_NN_t1::O2B_31_Temp,O_NN_t2::O2B_31_Temp,U::O1B,V::O1B)
    # Parameter initialization...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    a_max = div((N_max + 1)*(N_max + 2),2)

    # Define function for the transformation MEs for the 3rd index ...
    @inline function O2b_temp_transformation_ind3_MEs(P::Int64,J::Int64,Bra::Int64,Ket::Int64,i::Int64,c::Int64,T::Matrix{Float64},O_NN_t::Matrix{Matrix{Float64}})
        return @views T[i,c] * O_NN_t[P,J+1][Bra,Ket]
    end

    # Transformation of the 3rd index ...
    println("\nPerforming Quasiparticle Transformation of the residual 2-body NN interaction H31 in the 3rd index...")

    @inbounds Threads.@threads for i in JP
        J, P = i[1], i[2]
        if P == 1
            println("\tCalculating   ...   J = " * string(J) * "/" * string(N_2max+1) * "\tP = +")
        else
            println("\tCalculating   ...   J = " * string(J) * "/" * string(N_2max+1) * "\tP = -")
        end
        N = Orb_NN_t.N[P,J+1]
        @inbounds for Ket in 1:N
            c = Orb_NN_t.Ind[P,J+1][Ket][1]
            d = Orb_NN_t.Ind[P,J+1][Ket][2]
            l_c, j_c = Orb[c].l, Orb[c].j

            Orb_x = Orb_PreComp(a_max,j_c,l_c,Orb)

            @inbounds for Bra in 1:N

                OppSum1, OppSum2 = 0.0, 0.0
                Opn2011Sum1, Opn2011Sum2, Opn1120Sum1, Opn1120Sum2 = 0.0, 0.0, 0.0, 0.0
                OnnSum1, OnnSum2 = 0.0, 0.0

                @inbounds for i in Orb_x
                    Ket_id = O2b_temp_index(i,d,J,P,Orb_NN_t)

                    OppME1 = O2b_temp_transformation_ind3_MEs(P,J,Bra,Ket_id,i,c,V.p,O_NN_t2.pp[1])
                    OppME2 = O2b_temp_transformation_ind3_MEs(P,J,Bra,Ket_id,i,c,V.p,O_NN_t2.pp[2])

                    Opn2011ME1 = O2b_temp_transformation_ind3_MEs(P,J,Bra,Ket_id,i,c,V.p,O_NN_t2.pn2011[1])
                    Opn2011ME2 = O2b_temp_transformation_ind3_MEs(P,J,Bra,Ket_id,i,c,V.p,O_NN_t2.pn2011[2])
                    Opn1120ME1 = O2b_temp_transformation_ind3_MEs(P,J,Bra,Ket_id,i,c,U.p,O_NN_t2.pn1120[1])
                    Opn1120ME2 = O2b_temp_transformation_ind3_MEs(P,J,Bra,Ket_id,i,c,V.p,O_NN_t2.pn1120[2])

                    OnnME1 = O2b_temp_transformation_ind3_MEs(P,J,Bra,Ket_id,i,c,V.n,O_NN_t2.nn[1])
                    OnnME2 = O2b_temp_transformation_ind3_MEs(P,J,Bra,Ket_id,i,c,V.n,O_NN_t2.nn[2])

                    OppSum1 += OppME1
                    OppSum2 += OppME2

                    Opn2011Sum1 += Opn2011ME1
                    Opn2011Sum2 += Opn2011ME2
                    Opn1120Sum1 += Opn1120ME1
                    Opn1120Sum2 += Opn1120ME2

                    OnnSum1 += OnnME1
                    OnnSum2 += OnnME2
                end

                @views O_NN_t1.pp[1][P,J+1][Bra,Ket] = OppSum1
                @views O_NN_t1.pp[2][P,J+1][Bra,Ket] = OppSum2

                @views O_NN_t1.pn2011[1][P,J+1][Bra,Ket] = Opn2011Sum1
                @views O_NN_t1.pn2011[2][P,J+1][Bra,Ket] = Opn2011Sum2
                @views O_NN_t1.pn1120[1][P,J+1][Bra,Ket] = Opn1120Sum1
                @views O_NN_t1.pn1120[2][P,J+1][Bra,Ket] = Opn1120Sum2

                @views O_NN_t1.nn[1][P,J+1][Bra,Ket] = OnnSum1
                @views O_NN_t1.nn[2][P,J+1][Bra,Ket] = OnnSum2
            end
        end
    end

    return O_NN_t1
end

function qpO2b_31_allocate_ind4(Params::Parameters,JP::Vector{Vector{Int64}},Orb::Vector{Orb1B},Orb_NN::Orb2B,Orb_NN_t::Orb2B_Temp,O_NN::qpO2B,O_NN_t1::O2B_31_Temp,U::O1B,V::O1B)
    # Parameter initialization...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    a_max = div((N_max + 1)*(N_max + 2),2)

    # Define function for the transformation MEs for the 4th index ...
    @inline function O2b_temp_transformation_ind4_MEs(P::Int64,J::Int64,Bra::Int64,Ket::Int64,i::Int64,d::Int64,T::Matrix{Float64},O_NN_t::Matrix{Matrix{Float64}})
        return @views T[i,d] * O_NN_t[P,J+1][Bra,Ket]
    end

    # Transformation of the 4th index ...
    println("\nPerforming Quasiparticle Transformation of the residual 2-body NN interaction H31 in the 4th index...")

    @inbounds Threads.@threads for i in JP
        J, P = i[1], i[2]
        if P == 1
            println("\tCalculating   ...   J = " * string(J) * "/" * string(N_2max+1) * "\tP = +")
        else
            println("\tCalculating   ...   J = " * string(J) * "/" * string(N_2max+1) * "\tP = -")
        end
        N = Orb_NN.N[1,P,J+1]

        @inbounds for Bra in 1:N
            @inbounds for Ket in 1:N

                #a, b = Orb_NN.Ind[1,P,J+1][Bra][1], Orb_NN.Ind[1,P,J+1][Bra][2]
                c, d = Orb_NN.Ind[1,P,J+1][Ket][1], Orb_NN.Ind[1,P,J+1][Ket][2]
                l_d, j_d = Orb[d].l, Orb[d].j

                Orb_x = Orb_PreComp(a_max,j_d,l_d,Orb)

                OppSum, OnnSum = 0.0, 0.0
                Opn2011Sum, Opn1120Sum = 0.0, 0.0

                @inbounds for i in Orb_x
                    Ket_ci = O2b_temp_index(c,i,J,P,Orb_NN_t)

                    Opn2011ME1 = O2b_temp_transformation_ind4_MEs(P,J,Bra,Ket_ci,i,c,U.n,O_NN_t1.pn2011[1])
                    Opn2011ME2 = O2b_temp_transformation_ind4_MEs(P,J,Bra,Ket_ci,i,c,V.n,O_NN_t1.pn2011[2])
                    Opn1120ME1 = O2b_temp_transformation_ind4_MEs(P,J,Bra,Ket_ci,i,c,V.n,O_NN_t1.pn1120[1])
                    Opn1120ME2 = O2b_temp_transformation_ind4_MEs(P,J,Bra,Ket_ci,i,c,V.n,O_NN_t1.pn1120[2])

                    Opn2011Sum += (Opn2011ME1 + Opn2011ME2)
                    Opn1120Sum += (Opn1120ME1 + Opn1120ME2)


                    OppME1 = O2b_temp_transformation_ind4_MEs(P,J,Bra,Ket_ci,i,c,U.p,O_NN_t1.pp[1])
                    OppME2 = O2b_temp_transformation_ind4_MEs(P,J,Bra,Ket_ci,i,c,V.p,O_NN_t1.pp[2])

                    OnnME1 = O2b_temp_transformation_ind4_MEs(P,J,Bra,Ket_ci,i,c,U.n,O_NN_t1.nn[1])
                    OnnME2 = O2b_temp_transformation_ind4_MEs(P,J,Bra,Ket_ci,i,c,V.n,O_NN_t1.nn[2])

                    OppSum += (OppME1 - OppME2)
                    OnnSum += (OnnME1 - OnnME2)
                end

                Amp = Float64((-1)^J)

                Opn2011Sum = Amp * Opn2011Sum
                Opn1120Sum = Amp * Opn1120Sum

                @views O_NN.qp31.pn2011[P,J+1][Bra,Ket] = Opn2011Sum
                @views O_NN.qp31.pn1120[P,J+1][Bra,Ket] = Opn1120Sum


                Amp = 0.5 * Float64((-1)^(J + 1))

                OppSum = Amp * OppSum
                OnnSum = Amp * OnnSum

                @views O_NN.qp31.pp[P,J+1][Bra,Ket] = OppSum
                @views O_NN.qp31.nn[P,J+1][Bra,Ket] = OnnSum
            end
        end

    end

    return O_NN
end

@inline function qpO2b_31_index(a::Int64,b::Int64,c::Int64,d::Int64,J::Int64,P::Int64,Orb_NN::Orb2B)
    #Bra_key = (Int8(1),Int8(P),Int8(J),Int16(a),Int16(b))
    #Ket_key = (Int8(1),Int8(P),Int8(J),Int16(c),Int16(d))
    Bra_key = O2b_key(a,b,J,P,1)
    Ket_key = O2b_key(c,d,J,P,1)
    Bra = Int64(Orb_NN.Dic[Bra_key])
    Ket = Int64(Orb_NN.Dic[Ket_key])
    return Bra, Ket
end

@inline function qpO2b_31_pp(a::Int64,b::Int64,c::Int64,d::Int64,J::Int64,P::Int64,O_NN::qpO2B,Orb_NN::Orb2B)
    @inbounds Bra, Ket = qpO2b_31_index(a,b,c,d,J,P,Orb_NN)
    @inbounds return O_NN.qp31.pp[P,J+1][Bra,Ket]
end

@inline function qpO2b_2011_pn(a::Int64,b::Int64,c::Int64,d::Int64,J::Int64,P::Int64,O_NN::qpO2B,Orb_NN::Orb2B)
    @inbounds Bra, Ket = qpO2b_31_index(a,b,c,d,J,P,Orb_NN)
    @inbounds return O_NN.qp31.pn2011[P,J+1][Bra,Ket]
end

@inline function qpO2b_1120_pn(a::Int64,b::Int64,c::Int64,d::Int64,J::Int64,P::Int64,O_NN::qpO2B,Orb_NN::Orb2B)
    @inbounds Bra, Ket = qpO2b_31_index(a,b,c,d,J,P,Orb_NN)
    @inbounds return O_NN.qp31.pn1120[P,J+1][Bra,Ket]
end

@inline function qpO2b_31_nn(a::Int64,b::Int64,c::Int64,d::Int64,J::Int64,P::Int64,O_NN::qpO2B,Orb_NN::Orb2B)
    @inbounds Bra, Ket = qpO2b_31_index(a,b,c,d,J,P,Orb_NN)
    @inbounds return O_NN.qp31.nn[P,J+1][Bra,Ket]
end