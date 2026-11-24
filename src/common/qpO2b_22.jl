function qpO2b_22_allocate(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,Q_NN::qpO2B,O_NN::O2B,F_NN::O2B_Pandya,U::O1B,V::O1B)
    println("\n\tAllocating components O^(22) of the given NN operator 2-body in the quasiparticle representation ...")

    # Initialize temporary V & W interaction arrays ...
    Q_NN_t1, Q_NN_t2, Orb_NN_t = qpO2b_22_initialize(Params,Orb)

    # Allocate the proton-proton components of O^(22) ...
    Q_NN = qpO2B_22_allocate_pp(Params,Orb,Orb_NN,Orb_NN_t,O_NN,F_NN,Q_NN,Q_NN_t1,Q_NN_t2,U,V)

    # Allocate the proton-neutron (2002) components of O^(22) ...
    Q_NN = qpO2B_22_allocate_pn2002(Params,Orb,Orb_NN,Orb_NN_t,F_NN,Q_NN,Q_NN_t1,Q_NN_t2,U,V)

    # Allocate the proton-neutron (1111) components of O^(22) ...
    Q_NN = qpO2B_22_allocate_pn1111(Params,Orb,Orb_NN,Orb_NN_t,O_NN,F_NN,Q_NN,Q_NN_t1,Q_NN_t2,U,V)

    # Allocate the neutron-neutron components of O^(22) ...
    Q_NN = qpO2B_22_allocate_nn(Params,Orb,Orb_NN,Orb_NN_t,O_NN,F_NN,Q_NN,Q_NN_t1,Q_NN_t2,U,V)

    # Deallocate Q_NN_t ...
    Q_NN_t1 = nothing
    Q_NN_t2 = nothing

    # Perform the Garbage Collection ...
    GC.gc()

    println("\n\tComponents of O^(22) of the given 2-body NN quasiparticle operator succesfully allocated ...")

    return Q_NN
end

function qpO2b_22_initialize(Params::Parameters,Orb::Vector{Orb1B})
    # Read parameters ...
    N_max = Params.Calc.Nmax
    N_2max = 2 * N_max
    a_max = div((N_max + 1)*(N_max + 2),2)
    J_max = Int64(2*N_max + 1)

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
    Orb_NN_t = Orb2B_Temp(Orb_NN_Map,N_Orb_NN,Ind_Orb_NN)

    return O_NN_t1, O_NN_t2, Orb_NN_t
end

function qpO2B_22_allocate_pp(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,Orb_NN_t::Orb2B_Temp,O_NN::O2B,F_NN::O2B_Pandya,Q_NN::qpO2B,O_NN_t1::Matrix{Vector{Matrix{Float64}}},O_NN_t2::Matrix{Vector{Matrix{Float64}}},U::O1B,V::O1B)
    # Preallocate arrays of allowed values of J & P ...
    JP_list = JP_initialize(Params.Calc.N2max + 1)

    # Precalculate the array of orbitals contributing for each combination of l & j ...
    Orb_lj = orbitals_lj_make(Params,Orb)

    println("\n\t\tAllocating the components of the 2-body proton-proton operator O^(22) ...")

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
                ME3 = U.p[i,a] * F_ibcd
                ME4 = V.p[i,a] * F_ibcd
                ME5 = Phase_cdJ * U.p[i,a] * F_ibdc
                ME6 = Phase_cdJ * V.p[i,a] * F_ibdc

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

                    ME1 = U.p[i,c] * O_NN_t2[P,J+1][1][Bra,Ket_id]
                    ME2 = V.p[i,c] * O_NN_t2[P,J+1][2][Bra,Ket_id]
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

    # Perform quasiparticle transformation in the 4th index & allocate the components of ppO^(22) ...
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

                    ME1 = U.p[i,d] * O_NN_t1[P,J+1][1][Bra_ab,Ket_ci]
                    ME2 = V.p[i,d] * O_NN_t1[P,J+1][2][Bra_ab,Ket_ci]
                    ME3 = V.p[i,d] * O_NN_t1[P,J+1][3][Bra_ab,Ket_ci]
                    ME4 = U.p[i,d] * O_NN_t1[P,J+1][4][Bra_ab,Ket_ci]
                    ME5 = U.p[i,d] * O_NN_t1[P,J+1][5][Bra_ab,Ket_ci]
                    ME6 = V.p[i,d] * O_NN_t1[P,J+1][6][Bra_ab,Ket_ci]

                    Sum += (ME1 + ME2 + ME3 + ME4 - ME5 - ME6)
                end

                Q_NN.qp22.pp[P,J+1][Ind] = Sum
            end
        end

    end
    
    return Q_NN
end

function qpO2B_22_allocate_pn2002(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,Orb_NN_t::Orb2B_Temp,F_NN::O2B_Pandya,Q_NN::qpO2B,O_NN_t1::Matrix{Vector{Matrix{Float64}}},O_NN_t2::Matrix{Vector{Matrix{Float64}}},U::O1B,V::O1B)
    # Preallocate arrays of allowed values of J & P ...
    JP_list = JP_initialize(Params.Calc.N2max + 1)

    # Precalculate the array of orbitals contributing for each combination of l & j ...
    Orb_lj = orbitals_lj_make(Params,Orb)

    println("\n\t\tAllocating the components of the 2-body proton-neutron operator O^(20,02) ...")

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
                F_ibdc = F_NN.pnnp[P,J+1][Bra_ib,Ket_dc]
                F_ibcd = F_NN.pnnp[P,J+1][Bra_ib,Ket]

                ME1 = U.p[i,a] * F_ibcd
                ME2 = V.p[i,a] * F_ibcd
                ME3 = Phase_cdJ * U.p[i,a] * F_ibdc
                ME4 = Phase_cdJ * V.p[i,a] * F_ibdc

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

    # Perform quasiparticle transformation in the 4th index & allocate the components of pnO^(22) ...
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

                Q_NN.qp22.pn2002[P,J+1][Bra,Ket] = Sum
            end
        end

    end
    
    return Q_NN
end

function qpO2B_22_allocate_pn1111(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,Orb_NN_t::Orb2B_Temp,O_NN::O2B,F_NN::O2B_Pandya,Q_NN::qpO2B,O_NN_t1::Matrix{Vector{Matrix{Float64}}},O_NN_t2::Matrix{Vector{Matrix{Float64}}},U::O1B,V::O1B)
    # Preallocate arrays of allowed values of J & P ...
    JP_list = JP_initialize(Params.Calc.N2max + 1)

    # Precalculate the array of orbitals contributing for each combination of l & j ...
    Orb_lj = orbitals_lj_make(Params,Orb)

    println("\n\t\tAllocating the components of the 2-body proton-neutron operator O^(11,11) ...")

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

            Orb_i = Orb_lj[l_a+1,j_a]

            Sum1, Sum2, Sum3, Sum4 = 0.0, 0.0, 0.0, 0.0

            @inbounds for i in Orb_i
                Bra_ib = O2b_temp_index(i,b,J,P,Orb_NN_t)
                
                O_ibcd = O2b_pn(i,b,c,d,J,P,O_NN,Orb_NN)
                F_ibcd = F_NN.pnpn[P,J+1][Bra_ib,Ket]

                ME1 = U.p[i,a] * O_ibcd
                ME2 = V.p[i,a] * O_ibcd
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

                    ME1 = U.n[i,b] * O_NN_t1[P,J+1][1][Bra_ai,Ket]
                    ME2 = V.n[i,b] * O_NN_t1[P,J+1][2][Bra_ai,Ket]
                    ME3 = V.n[i,b] * O_NN_t1[P,J+1][3][Bra_ai,Ket]
                    ME4 = U.n[i,b] * O_NN_t1[P,J+1][4][Bra_ai,Ket]

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

                    ME1 = U.p[i,c] * O_NN_t2[P,J+1][1][Bra,Ket_id]
                    ME2 = V.p[i,c] * O_NN_t2[P,J+1][2][Bra,Ket_id]
                    ME3 = U.p[i,c] * O_NN_t2[P,J+1][3][Bra,Ket_id]
                    ME4 = V.p[i,c] * O_NN_t2[P,J+1][4][Bra,Ket_id]

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

    # Perform quasiparticle transformation in the 4th index & allocate the components of pnO^(22) ...
    @inbounds for JP in JP_list
        J, P = JP[1], JP[2]
        N = Orb_NN.N[1,P,J+1]
        @inbounds Threads.@threads for Bra in 1:N
            a = Orb_NN.Ind[1,P,J+1][Bra][1]
            b = Orb_NN.Ind[1,P,J+1][Bra][2]
            Bra_ab = O2b_temp_index(a,b,J,P,Orb_NN_t)
            @inbounds for Ket in 1:Bra
                Ind = Bra + (Ket - 1) * N - div(Ket * (Ket - 1),2)
                c = Orb_NN.Ind[1,P,J+1][Ket][1]
                d = Orb_NN.Ind[1,P,J+1][Ket][2]
                l_d, j_d = Orb[d].l, Orb[d].j
                
                Orb_i = Orb_lj[l_d+1,j_d]

                Sum = 0.0

                @inbounds for i in Orb_i
                    Ket_ci = O2b_temp_index(c,i,J,P,Orb_NN_t)

                    ME1 = U.n[i,d] * O_NN_t1[P,J+1][1][Bra_ab,Ket_ci]
                    ME2 = V.n[i,d] * O_NN_t1[P,J+1][2][Bra_ab,Ket_ci]
                    ME3 = V.n[i,d] * O_NN_t1[P,J+1][3][Bra_ab,Ket_ci]
                    ME4 = U.n[i,d] * O_NN_t1[P,J+1][4][Bra_ab,Ket_ci]

                    Sum += (ME1 + ME2 + ME3 + ME4)
                end

                Q_NN.qp22.pn1111[P,J+1][Ind] = Sum
            end
        end
    end
    
    return Q_NN
end

function qpO2B_22_allocate_nn(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,Orb_NN_t::Orb2B_Temp,O_NN::O2B,F_NN::O2B_Pandya,Q_NN::qpO2B,O_NN_t1::Matrix{Vector{Matrix{Float64}}},O_NN_t2::Matrix{Vector{Matrix{Float64}}},U::O1B,V::O1B)
    # Preallocate arrays of allowed values of J & P ...
    JP_list = JP_initialize(Params.Calc.N2max + 1)

    # Precalculate the array of orbitals contributing for each combination of l & j ...
    Orb_lj = orbitals_lj_make(Params,Orb)

    println("\n\t\tAllocating the components of the 2-body neutron-neutron operator O^(22) ...")

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
                ME3 = U.n[i,a] * F_ibcd
                ME4 = V.n[i,a] * F_ibcd
                ME5 = Phase_cdJ * U.n[i,a] * F_ibdc
                ME6 = Phase_cdJ * V.n[i,a] * F_ibdc

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

                    ME1 = U.n[i,c] * O_NN_t2[P,J+1][1][Bra,Ket_id]
                    ME2 = V.n[i,c] * O_NN_t2[P,J+1][2][Bra,Ket_id]
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

    # Perform quasiparticle transformation in the 4th index & allocate the components of nnO^(22) ...
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

                    ME1 = U.n[i,d] * O_NN_t1[P,J+1][1][Bra_ab,Ket_ci]
                    ME2 = V.n[i,d] * O_NN_t1[P,J+1][2][Bra_ab,Ket_ci]
                    ME3 = V.n[i,d] * O_NN_t1[P,J+1][3][Bra_ab,Ket_ci]
                    ME4 = U.n[i,d] * O_NN_t1[P,J+1][4][Bra_ab,Ket_ci]
                    ME5 = U.n[i,d] * O_NN_t1[P,J+1][5][Bra_ab,Ket_ci]
                    ME6 = V.n[i,d] * O_NN_t1[P,J+1][6][Bra_ab,Ket_ci]

                    Sum += (ME1 + ME2 + ME3 + ME4 - ME5 - ME6)
                end

                Q_NN.qp22.nn[P,J+1][Ind] = Sum
            end
        end

    end
    
    return Q_NN
end

@inline function qpO2b_22_index(a::Int64,b::Int64,c::Int64,d::Int64,J::Int64,P::Int64,Orb_NN::Orb2B)
    N = Orb_NN.N[2,P,J+1]
    Bra = O2b_lookup(a,b,J,1,Orb_NN)
    Ket = O2b_lookup(c,d,J,1,Orb_NN)
    Braket_max, Bracket_min = max(Bra,Ket), min(Bra,Ket)
    Ind = Braket_max + (Bracket_min - 1) * N - div(Bracket_min * (Bracket_min - 1),2)
    return Ind
end

@inline function qpO2b_2002_index(a::Int64,b::Int64,c::Int64,d::Int64,J::Int64,P::Int64,Orb_NN::Orb2B)
    Bra = O2b_lookup(a,b,J,1,Orb_NN)
    Ket = O2b_lookup(c,d,J,1,Orb_NN)
    return Bra, Ket
end

@inline function qpO2b_1111_index(a::Int64,b::Int64,c::Int64,d::Int64,J::Int64,P::Int64,Orb_NN::Orb2B)
    N = Orb_NN.N[1,P,J+1]
    Bra = O2b_lookup(a,b,J,0,Orb_NN)
    Ket = O2b_lookup(c,d,J,0,Orb_NN)
    Braket_max, Bracket_min = max(Bra,Ket), min(Bra,Ket)
    Ind = Braket_max + (Bracket_min - 1) * N - div(Bracket_min * (Bracket_min - 1),2)
    return Ind
end

@inline function qpO2b_22_pp(a::Int64,b::Int64,c::Int64,d::Int64,J::Int64,P::Int64,O_NN::qpO2B,Orb::Vector{Orb1B},Orb_NN::Orb2B)
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
    @inbounds Ind = qpO2b_22_index(a_s,b_s,c_s,d_s,J,P,Orb_NN)
    return Amp * O_NN.qp22.pp[P,J+1][Ind]
end

@inline function qpO2b_2002_pn(a::Int64,b::Int64,c::Int64,d::Int64,J::Int64,P::Int64,O_NN::qpO2B,Orb::Vector{Orb1B},Orb_NN::Orb2B)
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
    @inbounds Bra, Ket = qpO2b_2002_index(a_s,b_s,c_s,d_s,J,P,Orb_NN)
    return Amp * O_NN.qp22.pn2002[P,J+1][Bra,Ket]
end

@inline function qpO2b_1111_pn(a::Int64,b::Int64,c::Int64,d::Int64,J::Int64,P::Int64,O_NN::qpO2B,Orb_NN::Orb2B)
    @inbounds Ind = qpO2b_1111_index(a,b,c,d,J,P,Orb_NN)
    @inbounds return O_NN.qp22.pn1111[P,J+1][Ind]
end

@inline function qpO2b_22_nn(a::Int64,b::Int64,c::Int64,d::Int64,J::Int64,P::Int64,O_NN::qpO2B,Orb::Vector{Orb1B},Orb_NN::Orb2B)
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
    @inbounds Ind = qpO2b_22_index(a_s,b_s,c_s,d_s,J,P,Orb_NN)
    return Amp * O_NN.qp22.nn[P,J+1][Ind]
end