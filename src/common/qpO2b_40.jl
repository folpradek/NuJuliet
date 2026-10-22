function qpO2b_40_allocate(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,Q_NN::qpO2B,O_NN::O2B,F_NN::O2B_Temp,U::O1B,V::O1B)
    println("\n\tAllocating components O^(40) of the given NN operator 2-body in the quasiparticle representation ...")

    # Initialize temporary operator arrays ...
    Q_NN_t1, Q_NN_t2, Orb_NN_t = qpO2b_40_initialize(Params,Orb)

    # Allocate the proton-proton components of H40 ...
    Q_NN = qpO2B_40_allocate_pp(Params,Orb,Orb_NN,Orb_NN_t,O_NN,F_NN,Q_NN,Q_NN_t1,Q_NN_t2,U,V)

    # Allocate the proton-neutron components of H40 ...
    Q_NN = qpO2B_40_allocate_pn(Params,Orb,Orb_NN,Orb_NN_t,O_NN,F_NN,Q_NN,Q_NN_t1,Q_NN_t2,U,V)

    # Allocate the neutron-neutron components of H40 ...
    Q_NN = qpO2B_40_allocate_nn(Params,Orb,Orb_NN,Orb_NN_t,O_NN,F_NN,Q_NN,Q_NN_t1,Q_NN_t2,U,V)

    # Deallocate temporary arrays ...
    Q_NN_t1 = nothing
    Q_NN_t2 = nothing
    
    # Perform the Garbage Collection ...
    GC.gc()

    println("\n\tComponents of O^(40) of the given 2-body NN quasiparticle operator succesfully allocated ...")

    return Q_NN
end

function qpO2b_40_initialize(Params::Parameters,Orb::Vector{Orb1B})
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
                        Key = O2b_temp_key(a,b,J,P)
                        N_Orb_NN[P,J+1] += 1
                        Orb_NN_Dic[Key] = N_Orb_NN[P,J+1]
                        Ind_Orb_NN[P,J+1][N_Orb_NN[P,J+1]] = zeros(Int64, 2)
                        Ind_Orb_NN[P,J+1][N_Orb_NN[P,J+1]][1] = a
                        Ind_Orb_NN[P,J+1][N_Orb_NN[P,J+1]][2] = b
                    end
                end
            end
        end
    end

    # Initialize the NN operator matrices ...
    O_NN_t1 = Matrix{Vector{Matrix{Float64}}}(undef,2,J_max+1)
    O_NN_t2 = Matrix{Vector{Matrix{Float64}}}(undef,2,J_max+1)

    @inbounds for P = 1:2
        @inbounds Threads.@threads for J=0:J_max
            O_NN_t1[P,J+1] = Vector{Matrix{Float64}}(undef,6)
            O_NN_t2[P,J+1] = Vector{Matrix{Float64}}(undef,6)
        end
    end
 
    # Allocate entries of O ...
    @inbounds for P = 1:2
        @inbounds Threads.@threads for J=0:J_max
            Length = N_Orb_NN[P,J+1]
            @inbounds for i in 1:6
                O_NN_t1[P,J+1][i] = zeros(Float64,Length,Length)
                O_NN_t2[P,J+1][i] = zeros(Float64,Length,Length)
            end
        end
    end

    # Allocate the array for NN orbitals ...
    Orb_NN_t = Orb2B_Temp(Orb_NN_Dic,N_Orb_NN,Ind_Orb_NN)

    return O_NN_t1, O_NN_t2, Orb_NN_t
end

function qpO2B_40_allocate_pp(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,Orb_NN_t::Orb2B_Temp,O_NN::O2B,F_NN::O2B_Temp,Q_NN::qpO2B,O_NN_t1::Matrix{Vector{Matrix{Float64}}},O_NN_t2::Matrix{Vector{Matrix{Float64}}},U::O1B,V::O1B)
    # Preallocate arrays of allowed values of J & P ...
    JP_list = JP_initialize(Params.Calc.N2max + 1)

    # Precalculate the array of orbitals contributing for each combination of l & j ...
    Orb_lj = orbitals_lj_make(Params,Orb)

    println("\n\t\tAllocating the components of the 2-body proton-proton operator O^(40) ...")

    # Perform quasiparticle transformation in the 1st index ...
    @inbounds for JP in JP_list
        J, P = JP[1], JP[2]
        N = Orb_NN_t.N[P,J+1]
        @inbounds Threads.@threads for BraKet in 1:N^2
            Bra = div(BraKet-1,N) + 1
            Ket = rem(BraKet-1,N) + 1

            a = Orb_NN_t.Ind[P,J+1][Bra][1]
            b = Orb_NN_t.Ind[P,J+1][Bra][2]
            c = Orb_NN_t.Ind[P,J+1][Ket][1]
            d = Orb_NN_t.Ind[P,J+1][Ket][2]

            l_a, j_a = Orb[a].l, Orb[a].j
            j_c, j_d = Orb[c].j, Orb[d].j

            Phase_cdJ = Float64((-1)^(J + div(j_c + j_d,2)))

            Ket_dc =  O2b_temp_index(d,c,J,P,Orb_NN_t)

            Orb_i = Orb_lj[l_a+1,j_a]

            Sum1, Sum2, Sum3 = 0.0, 0.0, 0.0
            Sum4, Sum5, Sum6 = 0.0, 0.0, 0.0

            @inbounds for i in Orb_i
                Bra_ib = O2b_temp_index(i,b,J,P,Orb_NN_t)
                V_ibcd = O2b_pp(i,b,c,d,J,P,O_NN,Orb,Orb_NN)
                F_ibdc = F_NN.pp[P,J+1][Bra_ib,Ket_dc]
                F_ibcd = F_NN.pp[P,J+1][Bra_ib,Ket]

                ME1 = U.p[i,a] * V_ibcd
                ME2 = V.p[i,a] * V_ibcd
                ME3 = Phase_cdJ * U.p[i,a] * F_ibdc
                ME4 = Phase_cdJ * V.p[i,a] * F_ibdc
                ME5 = U.p[i,a] * F_ibcd
                ME6 = V.p[i,a] * F_ibcd

                Sum1 += ME1
                Sum2 += ME2
                Sum3 += ME3
                Sum4 += ME4
                Sum5 += ME5
                Sum6 += ME6
            end
            O_NN_t1[P,J+1][1][Bra,Ket] = Sum1
            O_NN_t1[P,J+1][2][Bra,Ket] = Sum2
            O_NN_t1[P,J+1][3][Bra,Ket] = Sum3
            O_NN_t1[P,J+1][4][Bra,Ket] = Sum4
            O_NN_t1[P,J+1][5][Bra,Ket] = Sum5
            O_NN_t1[P,J+1][6][Bra,Ket] = Sum6
        end
    end

    # Perform quasiparticle transformation in the 2nd index ...
    @inbounds for JP in JP_list
        J, P = JP[1], JP[2]
        N = Orb_NN_t.N[P,J+1]
        @inbounds Threads.@threads for Bra in 1:N
            a = Orb_NN_t.Ind[P,J+1][Bra][1]
            b = Orb_NN_t.Ind[P,J+1][Bra][2]
            l_b, j_b = Orb[b].l, Orb[b].j
            Orb_i = Orb_lj[l_b+1,j_b]

            @inbounds for Ket in 1:N
                Sum1, Sum2, Sum3 = 0.0, 0.0, 0.0
                Sum4, Sum5, Sum6 = 0.0, 0.0, 0.0

                @inbounds for i in Orb_i
                    Bra_ai = O2b_temp_index(a,i,J,P,Orb_NN_t)

                    ME1 = U.p[i,b] * O_NN_t1[P,J+1][1][Bra_ai,Ket]
                    ME2 = V.p[i,b] * O_NN_t1[P,J+1][2][Bra_ai,Ket]
                    ME3 = V.p[i,b] * O_NN_t1[P,J+1][3][Bra_ai,Ket]
                    ME4 = U.p[i,b] * O_NN_t1[P,J+1][4][Bra_ai,Ket]
                    ME5 = V.p[i,b] * O_NN_t1[P,J+1][5][Bra_ai,Ket]
                    ME6 = U.p[i,b] * O_NN_t1[P,J+1][6][Bra_ai,Ket]

                    Sum1 += ME1
                    Sum2 += ME2
                    Sum3 += ME3
                    Sum4 += ME4
                    Sum5 += ME5
                    Sum6 += ME6
                end

                O_NN_t2[P,J+1][1][Bra,Ket] = Sum1
                O_NN_t2[P,J+1][2][Bra,Ket] = Sum2
                O_NN_t2[P,J+1][3][Bra,Ket] = Sum3
                O_NN_t2[P,J+1][4][Bra,Ket] = Sum4
                O_NN_t2[P,J+1][5][Bra,Ket] = Sum5
                O_NN_t2[P,J+1][6][Bra,Ket] = Sum6
            end
        end
    end

    # Perform quasiparticle transformation in the 3rd index ...
    @inbounds for JP in JP_list
        J, P = JP[1], JP[2]
        N = Orb_NN_t.N[P,J+1]
        @inbounds Threads.@threads for Bra in 1:N
            @inbounds for Ket in 1:N
                c = Orb_NN_t.Ind[P,J+1][Ket][1]
                d = Orb_NN_t.Ind[P,J+1][Ket][2]
                l_c, j_c = Orb[c].l, Orb[c].j

                Orb_i = Orb_lj[l_c+1,j_c]

                Sum1, Sum2, Sum3 = 0.0, 0.0, 0.0
                Sum4, Sum5, Sum6 = 0.0, 0.0, 0.0

                @inbounds for i in Orb_i
                    Ket_id = O2b_temp_index(i,d,J,P,Orb_NN_t)

                    ME1 = V.p[i,c] * O_NN_t2[P,J+1][1][Bra,Ket_id]
                    ME2 = U.p[i,c] * O_NN_t2[P,J+1][2][Bra,Ket_id]
                    ME3 = U.p[i,c] * O_NN_t2[P,J+1][3][Bra,Ket_id]
                    ME4 = V.p[i,c] * O_NN_t2[P,J+1][4][Bra,Ket_id]
                    ME5 = V.p[i,c] * O_NN_t2[P,J+1][5][Bra,Ket_id]
                    ME6 = U.p[i,c] * O_NN_t2[P,J+1][6][Bra,Ket_id]

                    Sum1 += ME1
                    Sum2 += ME2
                    Sum3 += ME3
                    Sum4 += ME4
                    Sum5 += ME5
                    Sum6 += ME6
                end
                O_NN_t1[P,J+1][1][Bra,Ket] = Sum1
                O_NN_t1[P,J+1][2][Bra,Ket] = Sum2
                O_NN_t1[P,J+1][3][Bra,Ket] = Sum3
                O_NN_t1[P,J+1][4][Bra,Ket] = Sum4
                O_NN_t1[P,J+1][5][Bra,Ket] = Sum5
                O_NN_t1[P,J+1][6][Bra,Ket] = Sum6
            end
        end
    end

    # Perform quasiparticle transformation in the 4th index & allocate the components of ppO^(40) ...
    @inbounds for JP in JP_list
        J, P = JP[1], JP[2]
        N = Orb_NN.N[2,P,J+1]
        @inbounds Threads.@threads for Bra in 1:N
            a = Orb_NN.Ind[2,P,J+1][Bra][1]
            b = Orb_NN.Ind[2,P,J+1][Bra][2]
            Bra_ab = O2b_temp_index(a,b,J,P,Orb_NN_t)
            @inbounds for Ket in 1:Bra
                Ind = Bra + (Ket - 1) * N - div(Ket * (Ket - 1),2)
                c = Orb_NN.Ind[2,P,J+1][Ket][1]
                d = Orb_NN.Ind[2,P,J+1][Ket][2]
                l_d, j_d = Orb[d].l, Orb[d].j
                
                Orb_i = Orb_lj[l_d+1,j_d]

                Sum = 0.0

                @inbounds for i in Orb_i
                    Ket_ci = O2b_temp_index(c,i,J,P,Orb_NN_t)

                    ME1 = V.p[i,d] * O_NN_t1[P,J+1][1][Bra_ab,Ket_ci]
                    ME2 = U.p[i,d] * O_NN_t1[P,J+1][2][Bra_ab,Ket_ci]
                    ME3 = V.p[i,d] * O_NN_t1[P,J+1][3][Bra_ab,Ket_ci]
                    ME4 = U.p[i,d] * O_NN_t1[P,J+1][4][Bra_ab,Ket_ci]
                    ME5 = U.p[i,d] * O_NN_t1[P,J+1][5][Bra_ab,Ket_ci]
                    ME6 = V.p[i,d] * O_NN_t1[P,J+1][6][Bra_ab,Ket_ci]

                    Sum += (-ME1 - ME2 + ME3 + ME4 - ME5 - ME6)
                end

                Q_NN.qp40.pp[P,J+1][Ind] = Sum
            end
        end

    end
    
    return Q_NN
end

function qpO2B_40_allocate_pn(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,Orb_NN_t::Orb2B_Temp,O_NN::O2B,F_NN::O2B_Temp,Q_NN::qpO2B,O_NN_t1::Matrix{Vector{Matrix{Float64}}},O_NN_t2::Matrix{Vector{Matrix{Float64}}},U::O1B,V::O1B)
    # Preallocate arrays of allowed values of J & P ...
    JP_list = JP_initialize(Params.Calc.N2max + 1)

    # Precalculate the array of orbitals contributing for each combination of l & j ...
    Orb_lj = orbitals_lj_make(Params,Orb)

    println("\n\t\tAllocating the components of the 2-body proton-neutron operator O^(40) ...")

    # Perform quasiparticle transformation in the 1st index ...
    @inbounds for JP in JP_list
        J, P = JP[1], JP[2]
        N = Orb_NN_t.N[P,J+1]
        @inbounds Threads.@threads for BraKet in 1:N^2
            Bra = div(BraKet-1,N) + 1
            Ket = rem(BraKet-1,N) + 1

            a = Orb_NN_t.Ind[P,J+1][Bra][1]
            b = Orb_NN_t.Ind[P,J+1][Bra][2]
            c = Orb_NN_t.Ind[P,J+1][Ket][1]
            d = Orb_NN_t.Ind[P,J+1][Ket][2]

            l_a, j_a = Orb[a].l, Orb[a].j
            j_c, j_d = Orb[c].j, Orb[d].j

            Phase_cdJ = Float64((-1)^(J + div(j_c + j_d,2)))

            Ket_dc =  O2b_temp_index(d,c,J,P,Orb_NN_t)

            Orb_i = Orb_lj[l_a+1,j_a]

            Sum1, Sum2, Sum3, Sum4 = 0.0, 0.0, 0.0, 0.0

            @inbounds for i in Orb_i
                Bra_ib = O2b_temp_index(i,b,J,P,Orb_NN_t)
                F_ibdc = F_NN.pn[P,J+1][Bra_ib,Ket_dc]
                F_ibcd = F_NN.pn[P,J+1][Bra_ib,Ket]

                ME1 = Phase_cdJ * U.p[i,a] * F_ibdc
                ME2 = Phase_cdJ * V.p[i,a] * F_ibdc
                ME3 = U.p[i,a] * F_ibcd
                ME4 = V.p[i,a] * F_ibcd

                Sum1 += ME1
                Sum2 += ME2
                Sum3 += ME3
                Sum4 += ME4
            end
            O_NN_t1[P,J+1][1][Bra,Ket] = Sum1
            O_NN_t1[P,J+1][2][Bra,Ket] = Sum2
            O_NN_t1[P,J+1][3][Bra,Ket] = Sum3
            O_NN_t1[P,J+1][4][Bra,Ket] = Sum4
        end
    end

    # Perform quasiparticle transformation in the 2nd index ...
    @inbounds for JP in JP_list
        J, P = JP[1], JP[2]
        N = Orb_NN_t.N[P,J+1]
        @inbounds Threads.@threads for Bra in 1:N
            a = Orb_NN_t.Ind[P,J+1][Bra][1]
            b = Orb_NN_t.Ind[P,J+1][Bra][2]
            l_b, j_b = Orb[b].l, Orb[b].j
            Orb_i = Orb_lj[l_b+1,j_b]

            @inbounds for Ket in 1:N
                Sum1, Sum2, Sum3, Sum4 = 0.0, 0.0, 0.0, 0.0

                @inbounds for i in Orb_i
                    Bra_ai = O2b_temp_index(a,i,J,P,Orb_NN_t)

                    ME1 = V.p[i,b] * O_NN_t1[P,J+1][1][Bra_ai,Ket]
                    ME2 = U.p[i,b] * O_NN_t1[P,J+1][2][Bra_ai,Ket]
                    ME3 = V.p[i,b] * O_NN_t1[P,J+1][3][Bra_ai,Ket]
                    ME4 = U.p[i,b] * O_NN_t1[P,J+1][4][Bra_ai,Ket]

                    Sum1 += ME1
                    Sum2 += ME2
                    Sum3 += ME3
                    Sum4 += ME4
                end

                O_NN_t2[P,J+1][1][Bra,Ket] = Sum1
                O_NN_t2[P,J+1][2][Bra,Ket] = Sum2
                O_NN_t2[P,J+1][3][Bra,Ket] = Sum3
                O_NN_t2[P,J+1][4][Bra,Ket] = Sum4
            end
        end
    end

    # Perform quasiparticle transformation in the 3rd index ...
    @inbounds for JP in JP_list
        J, P = JP[1], JP[2]
        N = Orb_NN_t.N[P,J+1]
        @inbounds Threads.@threads for Bra in 1:N
            @inbounds for Ket in 1:N
                c = Orb_NN_t.Ind[P,J+1][Ket][1]
                d = Orb_NN_t.Ind[P,J+1][Ket][2]
                l_c, j_c = Orb[c].l, Orb[c].j

                Orb_i = Orb_lj[l_c+1,j_c]

                Sum1, Sum2, Sum3, Sum4 = 0.0, 0.0, 0.0, 0.0

                @inbounds for i in Orb_i
                    Ket_id = O2b_temp_index(i,d,J,P,Orb_NN_t)

                    ME1 = U.n[i,c] * O_NN_t2[P,J+1][1][Bra,Ket_id]
                    ME2 = V.n[i,c] * O_NN_t2[P,J+1][2][Bra,Ket_id]
                    ME3 = V.n[i,c] * O_NN_t2[P,J+1][3][Bra,Ket_id]
                    ME4 = U.n[i,c] * O_NN_t2[P,J+1][4][Bra,Ket_id]

                    Sum1 += ME1
                    Sum2 += ME2
                    Sum3 += ME3
                    Sum4 += ME4
                end
                O_NN_t1[P,J+1][1][Bra,Ket] = Sum1
                O_NN_t1[P,J+1][2][Bra,Ket] = Sum2
                O_NN_t1[P,J+1][3][Bra,Ket] = Sum3
                O_NN_t1[P,J+1][4][Bra,Ket] = Sum4
            end
        end
    end

    # Perform quasiparticle transformation in the 4th index & allocate the components of pnO^(40) ...
    @inbounds for JP in JP_list
        J, P = JP[1], JP[2]
        N = Orb_NN.N[2,P,J+1]
        @inbounds Threads.@threads for Bra in 1:N
            a = Orb_NN.Ind[2,P,J+1][Bra][1]
            b = Orb_NN.Ind[2,P,J+1][Bra][2]
            Bra_ab = O2b_temp_index(a,b,J,P,Orb_NN_t)
            @inbounds for Ket in 1:N
                c = Orb_NN.Ind[2,P,J+1][Ket][1]
                d = Orb_NN.Ind[2,P,J+1][Ket][2]
                l_d, j_d = Orb[d].l, Orb[d].j
                
                Orb_i = Orb_lj[l_d+1,j_d]

                Sum = 0.0

                @inbounds for i in Orb_i
                    Ket_ci = O2b_temp_index(c,i,J,P,Orb_NN_t)

                    ME1 = V.n[i,d] * O_NN_t1[P,J+1][1][Bra_ab,Ket_ci]
                    ME2 = U.n[i,d] * O_NN_t1[P,J+1][2][Bra_ab,Ket_ci]
                    ME3 = U.n[i,d] * O_NN_t1[P,J+1][3][Bra_ab,Ket_ci]
                    ME4 = V.n[i,d] * O_NN_t1[P,J+1][4][Bra_ab,Ket_ci]

                    Sum += (ME1 + ME2 - ME3 - ME4)
                end

                Q_NN.qp40.pn[P,J+1][Bra,Ket] = Sum
            end
        end

    end
    
    return Q_NN
end

function qpO2B_40_allocate_nn(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,Orb_NN_t::Orb2B_Temp,O_NN::O2B,F_NN::O2B_Temp,Q_NN::qpO2B,O_NN_t1::Matrix{Vector{Matrix{Float64}}},O_NN_t2::Matrix{Vector{Matrix{Float64}}},U::O1B,V::O1B)
    # Preallocate arrays of allowed values of J & P ...
    JP_list = JP_initialize(Params.Calc.N2max + 1)

    # Precalculate the array of orbitals contributing for each combination of l & j ...
    Orb_lj = orbitals_lj_make(Params,Orb)

    println("\n\t\tAllocating the components of the 2-body neutron-neutron operator O^(40) ...")

    # Perform quasiparticle transformation in the 1st index ...
    @inbounds for JP in JP_list
        J, P = JP[1], JP[2]
        N = Orb_NN_t.N[P,J+1]
        @inbounds Threads.@threads for BraKet in 1:N^2
            Bra = div(BraKet-1,N) + 1
            Ket = rem(BraKet-1,N) + 1

            a = Orb_NN_t.Ind[P,J+1][Bra][1]
            b = Orb_NN_t.Ind[P,J+1][Bra][2]
            c = Orb_NN_t.Ind[P,J+1][Ket][1]
            d = Orb_NN_t.Ind[P,J+1][Ket][2]

            l_a, j_a = Orb[a].l, Orb[a].j
            j_c, j_d = Orb[c].j, Orb[d].j

            Phase_cdJ = Float64((-1)^(J + div(j_c + j_d,2)))

            Ket_dc =  O2b_temp_index(d,c,J,P,Orb_NN_t)

            Orb_i = Orb_lj[l_a+1,j_a]

            Sum1, Sum2, Sum3 = 0.0, 0.0, 0.0
            Sum4, Sum5, Sum6 = 0.0, 0.0, 0.0

            @inbounds for i in Orb_i
                Bra_ib = O2b_temp_index(i,b,J,P,Orb_NN_t)
                V_ibcd = O2b_nn(i,b,c,d,J,P,O_NN,Orb,Orb_NN)
                F_ibdc = F_NN.nn[P,J+1][Bra_ib,Ket_dc]
                F_ibcd = F_NN.nn[P,J+1][Bra_ib,Ket]

                ME1 = U.n[i,a] * V_ibcd
                ME2 = V.n[i,a] * V_ibcd
                ME3 = Phase_cdJ * U.n[i,a] * F_ibdc
                ME4 = Phase_cdJ * V.n[i,a] * F_ibdc
                ME5 = U.n[i,a] * F_ibcd
                ME6 = V.n[i,a] * F_ibcd

                Sum1 += ME1
                Sum2 += ME2
                Sum3 += ME3
                Sum4 += ME4
                Sum5 += ME5
                Sum6 += ME6
            end
            O_NN_t1[P,J+1][1][Bra,Ket] = Sum1
            O_NN_t1[P,J+1][2][Bra,Ket] = Sum2
            O_NN_t1[P,J+1][3][Bra,Ket] = Sum3
            O_NN_t1[P,J+1][4][Bra,Ket] = Sum4
            O_NN_t1[P,J+1][5][Bra,Ket] = Sum5
            O_NN_t1[P,J+1][6][Bra,Ket] = Sum6
        end
    end

    # Perform quasiparticle transformation in the 2nd index ...
    @inbounds for JP in JP_list
        J, P = JP[1], JP[2]
        N = Orb_NN_t.N[P,J+1]
        @inbounds Threads.@threads for Bra in 1:N
            a = Orb_NN_t.Ind[P,J+1][Bra][1]
            b = Orb_NN_t.Ind[P,J+1][Bra][2]
            l_b, j_b = Orb[b].l, Orb[b].j
            Orb_i = Orb_lj[l_b+1,j_b]

            @inbounds for Ket in 1:N
                Sum1, Sum2, Sum3 = 0.0, 0.0, 0.0
                Sum4, Sum5, Sum6 = 0.0, 0.0, 0.0

                @inbounds for i in Orb_i
                    Bra_ai = O2b_temp_index(a,i,J,P,Orb_NN_t)

                    ME1 = U.n[i,b] * O_NN_t1[P,J+1][1][Bra_ai,Ket]
                    ME2 = V.n[i,b] * O_NN_t1[P,J+1][2][Bra_ai,Ket]
                    ME3 = V.n[i,b] * O_NN_t1[P,J+1][3][Bra_ai,Ket]
                    ME4 = U.n[i,b] * O_NN_t1[P,J+1][4][Bra_ai,Ket]
                    ME5 = V.n[i,b] * O_NN_t1[P,J+1][5][Bra_ai,Ket]
                    ME6 = U.n[i,b] * O_NN_t1[P,J+1][6][Bra_ai,Ket]

                    Sum1 += ME1
                    Sum2 += ME2
                    Sum3 += ME3
                    Sum4 += ME4
                    Sum5 += ME5
                    Sum6 += ME6
                end

                O_NN_t2[P,J+1][1][Bra,Ket] = Sum1
                O_NN_t2[P,J+1][2][Bra,Ket] = Sum2
                O_NN_t2[P,J+1][3][Bra,Ket] = Sum3
                O_NN_t2[P,J+1][4][Bra,Ket] = Sum4
                O_NN_t2[P,J+1][5][Bra,Ket] = Sum5
                O_NN_t2[P,J+1][6][Bra,Ket] = Sum6
            end
        end
    end

    # Perform quasiparticle transformation in the 3rd index ...
    @inbounds for JP in JP_list
        J, P = JP[1], JP[2]
        N = Orb_NN_t.N[P,J+1]
        @inbounds Threads.@threads for Bra in 1:N
            @inbounds for Ket in 1:N
                c = Orb_NN_t.Ind[P,J+1][Ket][1]
                d = Orb_NN_t.Ind[P,J+1][Ket][2]
                l_c, j_c = Orb[c].l, Orb[c].j

                Orb_i = Orb_lj[l_c+1,j_c]

                Sum1, Sum2, Sum3 = 0.0, 0.0, 0.0
                Sum4, Sum5, Sum6 = 0.0, 0.0, 0.0

                @inbounds for i in Orb_i
                    Ket_id = O2b_temp_index(i,d,J,P,Orb_NN_t)

                    ME1 = V.n[i,c] * O_NN_t2[P,J+1][1][Bra,Ket_id]
                    ME2 = U.n[i,c] * O_NN_t2[P,J+1][2][Bra,Ket_id]
                    ME3 = U.n[i,c] * O_NN_t2[P,J+1][3][Bra,Ket_id]
                    ME4 = V.n[i,c] * O_NN_t2[P,J+1][4][Bra,Ket_id]
                    ME5 = V.n[i,c] * O_NN_t2[P,J+1][5][Bra,Ket_id]
                    ME6 = U.n[i,c] * O_NN_t2[P,J+1][6][Bra,Ket_id]

                    Sum1 += ME1
                    Sum2 += ME2
                    Sum3 += ME3
                    Sum4 += ME4
                    Sum5 += ME5
                    Sum6 += ME6
                end
                O_NN_t1[P,J+1][1][Bra,Ket] = Sum1
                O_NN_t1[P,J+1][2][Bra,Ket] = Sum2
                O_NN_t1[P,J+1][3][Bra,Ket] = Sum3
                O_NN_t1[P,J+1][4][Bra,Ket] = Sum4
                O_NN_t1[P,J+1][5][Bra,Ket] = Sum5
                O_NN_t1[P,J+1][6][Bra,Ket] = Sum6
            end
        end
    end

    # Perform quasiparticle transformation in the 4th index & allocate the components of nnO^(40) ...
    @inbounds for JP in JP_list
        J, P = JP[1], JP[2]
        N = Orb_NN.N[2,P,J+1]
        @inbounds Threads.@threads for Bra in 1:N
            a = Orb_NN.Ind[2,P,J+1][Bra][1]
            b = Orb_NN.Ind[2,P,J+1][Bra][2]
            Bra_ab = O2b_temp_index(a,b,J,P,Orb_NN_t)
            @inbounds for Ket in 1:Bra
                Ind = Bra + (Ket - 1) * N - div(Ket * (Ket - 1),2)
                c = Orb_NN.Ind[2,P,J+1][Ket][1]
                d = Orb_NN.Ind[2,P,J+1][Ket][2]
                l_d, j_d = Orb[d].l, Orb[d].j
                
                Orb_i = Orb_lj[l_d+1,j_d]

                Sum = 0.0

                @inbounds for i in Orb_i
                    Ket_ci = O2b_temp_index(c,i,J,P,Orb_NN_t)

                    ME1 = V.n[i,d] * O_NN_t1[P,J+1][1][Bra_ab,Ket_ci]
                    ME2 = U.n[i,d] * O_NN_t1[P,J+1][2][Bra_ab,Ket_ci]
                    ME3 = V.n[i,d] * O_NN_t1[P,J+1][3][Bra_ab,Ket_ci]
                    ME4 = U.n[i,d] * O_NN_t1[P,J+1][4][Bra_ab,Ket_ci]
                    ME5 = U.n[i,d] * O_NN_t1[P,J+1][5][Bra_ab,Ket_ci]
                    ME6 = V.n[i,d] * O_NN_t1[P,J+1][6][Bra_ab,Ket_ci]

                    Sum += (-ME1 - ME2 + ME3 + ME4 - ME5 - ME6)
                end

                Q_NN.qp40.nn[P,J+1][Ind] = Sum
            end
        end

    end
    
    return Q_NN
end

@inline function qpO2b_40_T0_index(a::Int64,b::Int64,c::Int64,d::Int64,J::Int64,P::Int64,Orb_NN::Orb2B)
    #Bra_key = (Int8(1),Int8(P),Int8(J),Int16(a),Int16(b))
    #Ket_key = (Int8(1),Int8(P),Int8(J),Int16(c),Int16(d))
    Bra_key = O2b_key(a,b,J,P,1)
    Ket_key = O2b_key(c,d,J,P,1)
    Bra = Orb_NN.Dic[Bra_key]
    Ket = Orb_NN.Dic[Ket_key]
    return Bra, Ket
end

@inline function qpO2b_40_T1_index(a::Int64,b::Int64,c::Int64,d::Int64,J::Int64,P::Int64,Orb_NN::Orb2B)
    N = Orb_NN.N[2,P,J+1]
    #Bra_key = (Int8(1),Int8(P),Int8(J),Int16(a),Int16(b))
    #Ket_key = (Int8(1),Int8(P),Int8(J),Int16(c),Int16(d))
    Bra_key = O2b_key(a,b,J,P,1)
    Ket_key = O2b_key(c,d,J,P,1)
    Bra = Orb_NN.Dic[Bra_key]
    Ket = Orb_NN.Dic[Ket_key]
    Braket_max, Bracket_min = max(Bra,Ket), min(Bra,Ket)
    Ind = Braket_max + (Bracket_min - 1) * N - div(Bracket_min * (Bracket_min - 1),2)
    return Ind
end

@inline function qpO2b_40_pp(a::Int64,b::Int64,c::Int64,d::Int64,J::Int64,P::Int64,O_NN::qpO2B,Orb::Vector{Orb1B},Orb_NN::Orb2B)
    if a < b
        a_s, b_s = b, a
        exp_ab = J + 1 + div(Orb[a].j + Orb[b].j, 2)
        Sign_ab = isodd(exp_ab) ? -1.0 : 1.0
    else
        a_s, b_s = a, b
        Sign_ab = 1
    end
    if c < d
        c_s, d_s = d, c
        exp_cd = J + 1 + div(Orb[c].j + Orb[d].j, 2)
        Sign_cd = isodd(exp_cd) ? -1.0 : 1.0
    else
        c_s, d_s = c, d
        Sign_cd = 1
    end
    Amp = Sign_ab * Sign_cd
    if (a < b) && (c < d)
        exp = div(Orb[a].j + Orb[b].j + Orb[c].j + Orb[d].j, 2)
        Amp = isodd(exp) ? -1.0 : 1.0
    end
    @inbounds Ind = qpO2b_40_T1_index(a_s,b_s,c_s,d_s,J,P,Orb_NN)
    return Amp * O_NN.qp40.pp[P,J+1][Ind]
end

@inline function qpO2b_40_pn(a::Int64,b::Int64,c::Int64,d::Int64,J::Int64,P::Int64,O_NN::qpO2B,Orb::Vector{Orb1B},Orb_NN::Orb2B)
    if a < b
        a_s, b_s = b, a
        exp_ab = J + 1 + div(Orb[a].j + Orb[b].j, 2)
        Sign_ab = isodd(exp_ab) ? -1.0 : 1.0
    else
        a_s, b_s = a, b
        Sign_ab = 1
    end
    if c < d
        c_s, d_s = d, c
        exp_cd = J + 1 + div(Orb[c].j + Orb[d].j, 2)
        Sign_cd = isodd(exp_cd) ? -1.0 : 1.0
    else
        c_s, d_s = c, d
        Sign_cd = 1
    end
    Amp = Sign_ab * Sign_cd
    if (a < b) && (c < d)
        exp = div(Orb[a].j + Orb[b].j + Orb[c].j + Orb[d].j, 2)
        Amp = isodd(exp) ? -1.0 : 1.0
    end
    @inbounds Bra, Ket = qpO2b_40_T0_index(a_s,b_s,c_s,d_s,J,P,Orb_NN)
    return Amp * O_NN.qp40.pn[P,J+1][Bra,Ket]
end

@inline function qpO2b_40_nn(a::Int64,b::Int64,c::Int64,d::Int64,J::Int64,P::Int64,O_NN::qpO2B,Orb::Vector{Orb1B},Orb_NN::Orb2B)
    if a < b
        a_s, b_s = b, a
        exp_ab = J + 1 + div(Orb[a].j + Orb[b].j, 2)
        Sign_ab = isodd(exp_ab) ? -1.0 : 1.0
    else
        a_s, b_s = a, b
        Sign_ab = 1
    end
    if c < d
        c_s, d_s = d, c
        exp_cd = J + 1 + div(Orb[c].j + Orb[d].j, 2)
        Sign_cd = isodd(exp_cd) ? -1.0 : 1.0
    else
        c_s, d_s = c, d
        Sign_cd = 1
    end
    Amp = Sign_ab * Sign_cd
    if (a < b) && (c < d)
        exp = div(Orb[a].j + Orb[b].j + Orb[c].j + Orb[d].j, 2)
        Amp = isodd(exp) ? -1.0 : 1.0
    end
    @inbounds Ind = qpO2b_40_T1_index(a_s,b_s,c_s,d_s,J,P,Orb_NN)
    return Amp * O_NN.qp40.nn[P,J+1][Ind]
end

# For validation ...
function qpO2b_40_allocate_canonical(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,Q_NN::qpO2B,O_NN::O2B,U::O1B,V::O1B)
    # Read parameters ...
    N_max = Params.Calc.Nmax
    N_2max = Params.Calc.N2max

    # Preallocate arrays of allowed values of J & P ...
    JP = JP_initialize(N_2max + 1)

    # Define local functions performing the Pandya transformation of ph matrix elements ...
    @inline function qpO2B_40_allocate_canonical_ppF(a::Int64,b::Int64,c::Int64,d::Int64,J::Int64,Orb::Vector{Orb1B},Orb_NN::Orb2B,O_NN::O2B)
        l_a, l_b, l_c, l_d = Orb[a].l, Orb[b].l, Orb[c].l, Orb[d].l
        Q = rem(l_a + l_d,2) + 1
        ppFSum = 0.0
        if Q == (rem(l_b + l_c,2) + 1)
            j_a, j_b, j_c, j_d = Orb[a].j, Orb[b].j, Orb[c].j, Orb[d].j
            @inbounds for I in max(div(abs(j_a - j_d),2),div(abs(j_b - j_c),2)):min(div(j_a + j_d,2),div(j_b + j_c,2))
                ME = Float64((-1)^(div(j_b + j_c,2) + I) * (2*I + 1)) * f6j(j_a,j_b,2*J,j_c,j_d,2*I) * O2b_pp(a,d,b,c,I,Q,O_NN,Orb,Orb_NN)
                ppFSum += ME
            end
        end
        return ppFSum
    end

    @inline function qpO2B_40_allocate_canonical_pnF(a::Int64,b::Int64,c::Int64,d::Int64,J::Int64,Orb::Vector{Orb1B},Orb_NN::Orb2B,O_NN::O2B)
        l_a, l_b, l_c, l_d = Orb[a].l, Orb[b].l, Orb[c].l, Orb[d].l
        Q = rem(l_a + l_d,2) + 1
        pnFSum = 0.0
        if Q == (rem(l_b + l_c,2) + 1)
            j_a, j_b, j_c, j_d = Orb[a].j, Orb[b].j, Orb[c].j, Orb[d].j
            @inbounds for I in max(div(abs(j_a - j_d),2),div(abs(j_b - j_c),2)):min(div(j_a + j_d,2),div(j_b + j_c,2))
                ME = Float64((-1)^(div(j_b + j_c,2) + I) * (2*I + 1)) * f6j(j_a,j_b,2*J,j_c,j_d,2*I) * O2b_pn(a,d,b,c,I,Q,O_NN,Orb_NN)
                pnFSum += ME
            end
        end
        return pnFSum
    end

    @inline function qpO2B_40_allocate_canonical_nnF(a::Int64,b::Int64,c::Int64,d::Int64,J::Int64,Orb::Vector{Orb1B},Orb_NN::Orb2B,O_NN::O2B)
        l_a, l_b, l_c, l_d = Orb[a].l, Orb[b].l, Orb[c].l, Orb[d].l
        Q = rem(l_a + l_d,2) + 1
        nnFSum = 0.0
        if Q == (rem(l_b + l_c,2) + 1)
            j_a, j_b, j_c, j_d = Orb[a].j, Orb[b].j, Orb[c].j, Orb[d].j
            @inbounds for I in max(div(abs(j_a - j_d),2),div(abs(j_b - j_c),2)):min(div(j_a + j_d,2),div(j_b + j_c,2))
                ME = Float64((-1)^(div(j_b + j_c,2) + I) * (2*I + 1)) * f6j(j_a,j_b,2*J,j_c,j_d,2*I) * O2b_nn(a,d,b,c,I,Q,O_NN,Orb,Orb_NN)
                nnFSum += ME
            end
        end
        return nnFSum
    end

    println("\nPerforming Canonical Quasiparticle Transformation of the residual 2-body NN operator O^(40) ...")

    # Directly allocated Q^(40) components ...
    @inbounds for i in JP
        J, P = i[1], i[2]
        if P == 1
            println("\tCalculating   ...   J = " * string(J) * "/" * string(N_2max+1) * "\tP = +")
        else
            println("\tCalculating   ...   J = " * string(J) * "/" * string(N_2max+1) * "\tP = -")
        end

        N = Orb_NN.N[2,P,J+1]

        @inbounds Threads.@threads for Bra in 1:N
            @inbounds for Ket in 1:N

                # Case of pn interaction ... T = 0
                a, b = Orb_NN.Ind[2,P,J+1][Bra][1], Orb_NN.Ind[2,P,J+1][Bra][2]
                c, d = Orb_NN.Ind[2,P,J+1][Ket][1], Orb_NN.Ind[2,P,J+1][Ket][2]
                j_c, j_d = Orb[c].j, Orb[d].j

                OpnME = 0.0

                OpnME = (U.p[a,a] * V.p[b,b] * U.n[c,c] * V.n[d,d] + V.p[a,a] * U.p[b,b] * V.n[c,c] * U.n[d,d]) * qpO2B_40_allocate_canonical_pnF(a,b,d,c,J,Orb,Orb_NN,O_NN) * Float64((-1)^(div(j_c + j_d,2))) -
                        (U.p[a,a] * V.p[b,b] * V.n[c,c] * U.n[d,d] + V.p[a,a] * U.p[b,b] * U.n[c,c] * V.n[d,d]) * qpO2B_40_allocate_canonical_pnF(a,b,c,d,J,Orb,Orb_NN,O_NN) * Float64((-1)^(J))

                Q_NN.qp40.pn[P,J+1][Bra,Ket] = OpnME

                # Case of pp & nn interaction ... T = 1
                if Ket <= Bra
                    Ind = Bra + (Ket - 1) * N - div(Ket * (Ket - 1),2)
                    #a, b = Orb_NN.Ind[2,P,J+1][Bra][1], Orb_NN.Ind[2,P,J+1][Bra][2]
                    #c, d = Orb_NN.Ind[2,P,J+1][Ket][1], Orb_NN.Ind[2,P,J+1][Ket][2]
                    #j_c, j_d = Orb[c].j, Orb[d].j

                    OppME, OnnME = 0.0, 0.0

                    OppME = (U.p[a,a] * U.p[b,b] * V.p[c,c] * V.p[d,d] + V.p[a,a] * V.p[b,b] * U.p[c,c] * U.p[d,d]) * O2b_pp(a,b,c,d,J,P,O_NN,Orb,Orb_NN) * (-1)^(J + 1) +
                            (U.p[a,a] * V.p[b,b] * U.p[c,c] * V.p[d,d] + V.p[a,a] * U.p[b,b] * V.p[c,c] * U.p[d,d]) * qpO2B_40_allocate_canonical_ppF(a,b,d,c,J,Orb,Orb_NN,O_NN) * Float64((-1)^(div(j_c + j_d,2))) -
                            (U.p[a,a] * V.p[b,b] * V.p[c,c] * U.p[d,d] + V.p[a,a] * U.p[b,b] * U.p[c,c] * V.p[d,d]) * qpO2B_40_allocate_canonical_ppF(a,b,c,d,J,Orb,Orb_NN,O_NN) * (-1)^(J)
                    

                    OnnME = (U.n[a,a] * U.n[b,b] * V.n[c,c] * V.n[d,d] + V.n[a,a] * V.n[b,b] * U.n[c,c] * U.n[d,d]) * O2b_nn(a,b,c,d,J,P,O_NN,Orb,Orb_NN) * (-1)^(J + 1) +
                            (U.n[a,a] * V.n[b,b] * U.n[c,c] * V.n[d,d] + V.n[a,a] * U.n[b,b] * V.n[c,c] * U.n[d,d]) * qpO2B_40_allocate_canonical_nnF(a,b,d,c,J,Orb,Orb_NN,O_NN) * Float64((-1)^(div(j_c + j_d,2))) -
                            (U.n[a,a] * V.n[b,b] * V.n[c,c] * U.n[d,d] + V.n[a,a] * U.n[b,b] * U.n[c,c] * V.n[d,d]) * qpO2B_40_allocate_canonical_nnF(a,b,c,d,J,Orb,Orb_NN,O_NN) * (-1)^(J)

                    Q_NN.qp40.pp[P,J+1][Ind] = OppME
                    Q_NN.qp40.nn[P,J+1][Ind] = OnnME
                end

            end
        end
    end

    #=
    function test_call(a_max::Int64)

        ppSum, pnSum, nnSum = 0.0, 0.0, 0.0

        @inbounds for a in 1:a_max
            @inbounds for b in 1:a_max
                l_a, j_a = Orb[a].l, Orb[a].j
                l_b, j_b = Orb[b].l, Orb[b].j
                Q = rem(l_a + l_b,2) + 1
                @inbounds for c in 1:a_max
                    @inbounds for d in 1:a_max
                        l_c, j_c = Orb[c].l, Orb[c].j
                        l_d, j_d = Orb[d].l, Orb[d].j
                        if (rem(l_c + l_d,2) + 1) == Q
                            @inbounds for J in max(div(abs(j_a - j_b),2),div(abs(j_c - j_d),2)):min(div(j_a + j_b,2),div(j_c + j_d,2))
                                ppSum += abs(qpO2b_40_pp(a,b,c,d,J,Q,Q_NN,Orb,Orb_NN) + (-1)^(div(j_a + j_b,2) + J) * qpO2b_40_pp(b,a,c,d,J,Q,Q_NN,Orb,Orb_NN))
                                ppSum += abs(qpO2b_40_pp(a,b,c,d,J,Q,Q_NN,Orb,Orb_NN) + (-1)^(div(j_c + j_d,2) + J) * qpO2b_40_pp(a,b,d,c,J,Q,Q_NN,Orb,Orb_NN))
                                ppSum += abs(qpO2b_40_pp(a,b,c,d,J,Q,Q_NN,Orb,Orb_NN) - (-1)^(div(j_a + j_b + j_c + j_d,2)) * qpO2b_40_pp(b,a,d,c,J,Q,Q_NN,Orb,Orb_NN))
                                #ppSum += abs(qpO2b_40_pp(a,b,c,d,J,Q,Q_NN,Orb,Orb_NN) - qpO2b_40_pp(c,d,a,b,J,Q,Q_NN,Orb,Orb_NN))

                                #ppSum += qpO2b_40_pp(a,b,c,d,J,Q,Q_NN,Orb,Orb_NN) - qpO2b_40_pp(a,b,c,d,J,Q,Q_NN,Orb,Orb_NN)

                                #ppSum += qpO2b_40_pp(a,b,c,d,J,Q,Q_NN,Orb,Orb_NN) + (-1)^(div(j_a + j_b,2) + J) * qpO2b_40_pp(b,a,c,d,J,Q,Q_NN,Orb,Orb_NN)
                                ##=

                                nnSum += abs(qpO2b_40_nn(a,b,c,d,J,Q,Q_NN,Orb,Orb_NN) + (-1)^(div(j_a + j_b,2) + J) * qpO2b_40_nn(b,a,c,d,J,Q,Q_NN,Orb,Orb_NN))
                                nnSum += abs(qpO2b_40_nn(a,b,c,d,J,Q,Q_NN,Orb,Orb_NN) + (-1)^(div(j_c + j_d,2) + J) * qpO2b_40_nn(a,b,d,c,J,Q,Q_NN,Orb,Orb_NN))
                                nnSum += abs(qpO2b_40_nn(a,b,c,d,J,Q,Q_NN,Orb,Orb_NN) - (-1)^(div(j_a + j_b + j_c + j_d,2)) * qpO2b_40_nn(b,a,d,c,J,Q,Q_NN,Orb,Orb_NN))
                                #nnSum += abs(qpO2b_40_nn(a,b,c,d,J,Q,Q_NN,Orb,Orb_NN) - qpO2b_40_nn(c,d,a,b,J,Q,Q_NN,Orb,Orb_NN))
                                ##=#

                                pnSum += abs(qpO2b_40_pn(a,b,c,d,J,Q,Q_NN,Orb_NN) + (-1)^(div(j_a + j_b,2) + J) * qpO2b_40_pn(b,a,c,d,J,Q,Q_NN,Orb_NN))
                                pnSum += abs(qpO2b_40_pn(a,b,c,d,J,Q,Q_NN,Orb_NN) + (-1)^(div(j_c + j_d,2) + J) * qpO2b_40_pn(a,b,d,c,J,Q,Q_NN,Orb_NN))
                                pnSum += abs(qpO2b_40_pn(a,b,c,d,J,Q,Q_NN,Orb_NN) - (-1)^(div(j_a + j_b + j_c + j_d,2)) * qpO2b_40_pn(b,a,d,c,J,Q,Q_NN,Orb_NN))

                            end
                        end
                    end
                end
            end
        end

        println("\nppSum = " * string(ppSum))
        println("pnSum = " * string(pnSum))
        println("nnSum = " * string(nnSum))

        return
    end

    =#
    =#
    N_max = Params.Calc.Nmax
    N_2max = 2 * N_max
    a_max = div((N_max + 1)*(N_max + 2),2)
    test_call(a_max)
    =#

    return Q_NN
end