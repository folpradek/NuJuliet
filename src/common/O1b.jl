function O1b_export(Params::Parameters,Orb::Vector{Orb1B},A::O1B,A_Export_Path::String)
    # Read parameters ...
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1)*(N_max + 2),2)

    # Export 1-body operator A under given path ...
    open(A_Export_Path, "w") do Export_File
        # Proton entries ...
        @inbounds for a in 1:a_max
            @inbounds for b in 1:a_max
                if Orb[a].l == Orb[b].l && Orb[a].j == Orb[b].j
                    ME = @views A.p[a,b]
                    write(Export_File, Float64(ME))
                end
            end
        end

        # Neutron entries ...
        @inbounds for a in 1:a_max
            @inbounds for b in 1:a_max
                if Orb[a].l == Orb[b].l && Orb[a].j == Orb[b].j
                    ME = @views A.n[a,b]
                    write(Export_File, Float64(ME))
                end
            end
        end
    end
    
    return
end

function O1b_import(Params::Parameters,Orb::Vector{Orb1B},Import_Path::String)
    # Read parameters ...
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1)*(N_max + 2),2)

    # Read the 1-body operator A matrix from given file ...
    pA, nA = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)
    open(Import_Path, "r") do Read_File
        # Proton entries ...
        @inbounds for a in 1:a_max
            @inbounds for b in 1:a_max
                if Orb[a].l == Orb[b].l && Orb[a].j == Orb[b].j
                    ME = read(Read_File,Float64)
                    pA[a,b] = ME
                end
            end
        end

        # Neutron entries ...
        @inbounds for a in 1:a_max
            @inbounds for b in 1:a_max
                if Orb[a].l == Orb[b].l && Orb[a].j == Orb[b].j
                    ME = read(Read_File,Float64)
                    nA[a,b] = ME
                end
            end
        end
    end
    
    return O1B(pA,nA)
end

function qpO1b_export(Params::Parameters,Orb::Vector{Orb1B},A::qpO1B,A_Export_Path::String)
    # Read parameters ...
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1)*(N_max + 2),2)

    # Export 1-body operator A under given path ...
    open(A_Export_Path, "w") do Export_File
            # Export of qp11 part ...
        # Proton entries ...
        @inbounds for a in 1:a_max
            @inbounds for b in 1:a_max
                if Orb[a].l == Orb[b].l && Orb[a].j == Orb[b].j
                    ME = A.qp11.p[a,b]
                    write(Export_File, Float64(ME))
                end
            end
        end

        # Neutron entries ...
        @inbounds for a in 1:a_max
            @inbounds for b in 1:a_max
                if Orb[a].l == Orb[b].l &&Orb[a].j == Orb[b].j
                    ME = A.qp11.n[a,b]
                    write(Export_File, Float64(ME))
                end
            end
        end

            # Export of qp20 part ...
        # Proton entries ...
        @inbounds for a in 1:a_max
            @inbounds for b in 1:a_max
                if Orb[a].l == Orb[b].l &&Orb[a].j == Orb[b].j
                    ME = A.qp20.p[a,b]
                    write(Export_File, Float64(ME))
                end
            end
        end

        # Neutron entries ...
        @inbounds for a in 1:a_max
            @inbounds for b in 1:a_max
                if Orb[a].l == Orb[b].l &&Orb[a].j == Orb[b].j
                    ME = A.qp20.n[a,b]
                    write(Export_File, Float64(ME))
                end
            end
        end

    end
    
    return
end

function qpO1b_import(Params::Parameters,Orb::Vector{Orb1B},Import_Path::String)
    # Read parameters ...
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1)*(N_max + 2),2)

    # Read the 1-body operator A matrix from given file ...
    pA11, pA20 = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)
    nA11, nA20 = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)
    open(Import_Path, "r") do Read_File
            # Read qp11 part ...
        # Proton entries ...
        @inbounds for a in 1:a_max
            @inbounds for b in 1:a_max
                if Orb[a].l == Orb[b].l && Orb[a].j == Orb[b].j
                    ME = read(Read_File,Float64)
                    pA11[a,b] = ME
                end
            end
        end

        # Neutron entries ...
        @inbounds for a in 1:a_max
            @inbounds for b in 1:a_max
                if Orb[a].l == Orb[b].l && Orb[a].j == Orb[b].j
                    ME = read(Read_File,Float64)
                    nA11[a,b] = ME
                end
            end
        end

            # Read qp20 part ...
        # Proton entries ...
        @inbounds for a in 1:a_max
            @inbounds for b in 1:a_max
                if Orb[a].l == Orb[b].l && Orb[a].j == Orb[b].j
                    ME = read(Read_File,Float64)
                    pA20[a,b] = ME
                end
            end
        end

        # Neutron entries ...
        @inbounds for a in 1:a_max
            @inbounds for b in 1:a_max
                if Orb[a].l == Orb[b].l && Orb[a].j == Orb[b].j
                    ME = read(Read_File,Float64)
                    nA20[a,b] = ME
                end
            end
        end
    end

    return qpO1B(O1B(pA11,nA11),O1B(pA20,nA20))
end

function qpN1b_make(Params::Parameters,U::O1B,V::O1B)
    # Read parameters ...
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1)*(N_max + 2),2)

    # Read the 1-body operator A matrix from given file ...
    pN11, pN20 = Matrix{Float64}(undef,a_max,a_max), Matrix{Float64}(undef,a_max,a_max)
    nN11, nN20 = Matrix{Float64}(undef,a_max,a_max), Matrix{Float64}(undef,a_max,a_max)

    # Allocate the components of 1-body particle number operator N in the qp space ...
    pN11 .= U.p' * U.p .- V.p' * V.p
    nN11 .= U.n' * U.n .- V.n' * V.n 
    pN20 .= U.p' * V.p
    nN20 .= U.n' * V.n

    return qpO1B(O1B(pN11,nN11),O1B(pN20,nN20))
end

function H1b(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,Orb_NNN::Orb3B,T::O1B,V_NN::O2B,V_NNN::Array{Vector{Vector{Float32}},4},Rho::O1B)
    # Read parameters ...
    A = Float64(Params.Calc.A)
    N_max = Params.Calc.Nmax
    N_2max = Params.Calc.N2max
    N_3max = Params.Calc.N3max
    a_max = div((N_max + 1)*(N_max + 2),2)
    CMS = Params.Calc.CMS

    println("\nEvaluating 1-body mean-field Hamiltonian h_N ...")

    function H1b_indices(Params::Parameters,Orb::Vector{Orb1B})
        # Read # initialize parameters ...
        N_max = Params.Calc.Nmax
        a_max = div((N_max + 1)*(N_max + 2),2)
        ab_count, de_count = 0, 0

        # Count how many ab pairs are there ...
        @inbounds for a in 1:a_max
            l_a = Orb[a].l
            j_a = Orb[a].j
            @inbounds for b in 1:a
                l_b = Orb[b].l
                j_b = Orb[b].j
                if j_a == j_b && l_a == l_b
                    ab_count += 1
                end
            end
        end

        # Count how many de pairs are there ...
        @inbounds for d in 1:a_max
            @inbounds for e in 1:a_max
                de_count += 1
            end
        end

        # Initialite the array ab ...
        ab, ab_count = zeros(Int64,2,ab_count), 0
        de, de_count = zeros(Int64,2,de_count), 0

        # Allocate the array ab ...
        @inbounds for a in 1:a_max
            l_a = Orb[a].l
            j_a = Orb[a].j
            @inbounds for b in 1:a
                l_b = Orb[b].l
                j_b = Orb[b].j
                if j_a == j_b && l_a == l_b
                    ab_count += 1
                    ab[1,ab_count] = a
                    ab[2,ab_count] = b
                end
            end
        end

        # Allocate the array de ...
        @inbounds for d in 1:a_max
            @inbounds for e in 1:a_max
                de_count += 1
                de[1,de_count] = d
                de[2,de_count] = e
            end
        end

        return ab, ab_count, de, de_count
    end

    # Allocate indices for iteration ...
    ad, ad_count, be, be_count = H1b_indices(Params,Orb)

    # Allocate new 1-body Hamiltonian matrices ...
    ph_N, nh_N = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)

    # Allocate the single-particle field H ...
    @inbounds for ad_i in 1:ad_count
        a, d = ad[1,ad_i], ad[2,ad_i]
        n_a, l_a, j_a = Orb[a].n, Orb[a].l, Orb[a].j
        n_d, l_d, j_d = Orb[d].n, Orb[d].l, Orb[d].j
        is_j_a_hat = 1.0 / Float64(j_a + 1)
        ph_N_local = zeros(Float64,Threads.maxthreadid())
        nh_N_local = zeros(Float64,Threads.maxthreadid())

        @inbounds Threads.@threads :static for be_i in 1:be_count
            b, e = be[1,be_i], be[2,be_i]
            n_b, l_b, j_b = Orb[b].n, Orb[b].l, Orb[b].j
            n_e, l_e, j_e = Orb[e].n, Orb[e].l, Orb[e].j
            P_ab = rem(l_a + l_b,2) + 1
            Tid = Threads.threadid()
            pSum_local, nSum_local = 0.0, 0.0

            if (2*(n_a + n_b) + l_a + l_b) <= N_2max
                if l_b == l_e && j_b == j_e && (2*(n_d + n_e) + l_d + l_e) <= N_2max
                    pRho_be, nRho_be = Rho.p[b,e], Rho.n[b,e]
                    @inbounds for J = div(abs(j_d - j_e),2):div(j_d + j_e,2)

                        # 2-body NN interaction part ...
                        if P_ab == (rem(l_d + l_e, 2) + 1)
                            J_j_a_hat = Float64(2*J + 1) * is_j_a_hat
                            pSum_local += J_j_a_hat * pRho_be * O2b_pp(a,b,d,e,J,P_ab,V_NN,Orb,Orb_NN)
                            pSum_local += J_j_a_hat * nRho_be * O2b_pn(a,b,d,e,J,P_ab,V_NN,Orb_NN)
                            nSum_local += J_j_a_hat * nRho_be * O2b_nn(a,b,d,e,J,P_ab,V_NN,Orb,Orb_NN)
                            nSum_local += J_j_a_hat * pRho_be * O2b_pn(b,a,e,d,J,P_ab,V_NN,Orb_NN)
                        end

                        # 3-body NNN interaction part ...
                        @inbounds for c in 1:a_max
                            n_c, l_c = Orb[c].n, Orb[c].l
                            if (2*(n_a + n_b + n_c) + l_a + l_b + l_c) <= N_3max
                                P_abc = rem(l_a + l_b + l_c, 2) + 1
                                j_c = Orb[c].j
                                @inbounds for f in 1:a_max
                                    n_f, l_f = Orb[f].n, Orb[f].l
                                    if (2*(n_d + n_e + n_f) + l_d + l_e + l_f) <= N_3max && l_c == l_f && P_abc == (rem(l_d + l_e + l_f,2) + 1)
                                        j_f = Orb[f].j
                                        if j_c == j_f
                                            pRho_cf, nRho_cf = Rho.p[c,f], Rho.n[c,f]

                                            ME001 = V3b_no2b(a,b,c,0,d,e,f,0,J,1,P_abc,V_NNN,Orb,Orb_NNN)
                                            ME101 = V3b_no2b(a,b,c,1,d,e,f,0,J,1,P_abc,V_NNN,Orb,Orb_NNN)
                                            ME011 = V3b_no2b(a,b,c,0,d,e,f,1,J,1,P_abc,V_NNN,Orb,Orb_NNN)
                                            ME111 = V3b_no2b(a,b,c,1,d,e,f,1,J,1,P_abc,V_NNN,Orb,Orb_NNN)
                                            ME113 = V3b_no2b(a,b,c,1,d,e,f,1,J,3,P_abc,V_NNN,Orb,Orb_NNN)

                                            pSum_local += is_j_a_hat * (0.5*ME113*pRho_be*pRho_cf + 0.25 * (ME001 +
                                                           sqrt(1.0/3.0)*ME101 + sqrt(1.0/3.0)*ME011 + ME111/3.0 + 2.0/3.0*ME113)*nRho_be*nRho_cf +
                                                           1.0/3.0 * (2.0*ME111 + ME113)*pRho_be*nRho_cf)

                                            nSum_local += is_j_a_hat * (0.5*ME113*nRho_be*nRho_cf + 0.25 * (ME001 +
                                                           sqrt(1.0/3.0)*ME101 + sqrt(1.0/3.0)*ME011 + ME111/3.0 + 2.0/3.0*ME113)*pRho_be*pRho_cf +
                                                           1.0/3.0 * (2.0*ME111 + ME113)*nRho_be*pRho_cf)
                                        end
                                    end
                                end
                            end
                        end
                    end
                end

            end
            ph_N_local[Tid] += phSum_local
            nh_N_local[Tid] += nhSum_local
        end

        # Sum over the Thread-local accumulators for H ...
        pSum = sum(ph_N_local)
        nSum = sum(nh_N_local)

        # Include the 1-body kinetic energy & inclusion of Center-of-Mass motion (CM) correction ...
            # Combined 1- + 2-body kinetic operator with CM correction ...
        if CMS == "CMS1+2B"
            pSum += T.p[a,d] * (1.0 - 1.0 / A)
            nSum += T.n[a,d] * (1.0 - 1.0 / A)
            # Pure 1-body kinetic operator with no CM correction ...
        elseif CMS != "CMS2B"
            pSum += T.p[a,d]
            nSum += T.n[a,d]
        end
            # No contribution for pure 2-body kinetic operator with CM correction ...

        # Allocate ph & nh ...
        if a != d
            ph_N[a,d], ph_N[d,a] = pSum, pSum
            nh_N[a,d], nh_N[d,a] = nSum, nSum
        elseif a == d
            ph_N[a,a], nh_N[a,a] = pSum, nSum
        end

    end

    println("\tCompleted the calculation of 1-body mean-field Hamiltonian h_N ...")

    return O1B(ph_N,nh_N)
end

function H1b(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,Orb_NNN::Orb3B,T::O1B,V_NN::O2B,V_NNN::Array{Vector{Vector{Float32}},4},Rho::O1B,Sigma_NN::O2B)
    # Read parameters ...
    A = Float64(Params.Calc.A)
    N_max = Params.Calc.Nmax
    N_2max = Params.Calc.N2max
    N_3max = Params.Calc.N3max
    a_max = div((N_max + 1)*(N_max + 2),2)
    CMS = Params.Calc.CMS

    println("\nEvaluating 1-body mean-field Hamiltonian h_N ...")

    function H1b_indices(Params::Parameters,Orb::Vector{Orb1B})
        # Read # initialize parameters ...
        N_max = Params.Calc.Nmax
        a_max = div((N_max + 1)*(N_max + 2),2)
        ab_count, de_count = 0, 0

        # Count how many ab pairs are there ...
        @inbounds for a in 1:a_max
            l_a = Orb[a].l
            j_a = Orb[a].j
            @inbounds for b in 1:a
                l_b = Orb[b].l
                j_b = Orb[b].j
                if j_a == j_b && l_a == l_b
                    ab_count += 1
                end
            end
        end

        # Count how many de pairs are there ...
        @inbounds for d in 1:a_max
            @inbounds for e in 1:a_max
                de_count += 1
            end
        end

        # Initialite the array ab ...
        ab, ab_count = zeros(Int64,2,ab_count), 0
        de, de_count = zeros(Int64,2,de_count), 0

        # Allocate the array ab ...
        @inbounds for a in 1:a_max
            l_a = Orb[a].l
            j_a = Orb[a].j
            @inbounds for b in 1:a
                l_b = Orb[b].l
                j_b = Orb[b].j
                if j_a == j_b && l_a == l_b
                    ab_count += 1
                    ab[1,ab_count] = a
                    ab[2,ab_count] = b
                end
            end
        end

        # Allocate the array de ...
        @inbounds for d in 1:a_max
            @inbounds for e in 1:a_max
                de_count += 1
                de[1,de_count] = d
                de[2,de_count] = e
            end
        end

        return ab, ab_count, de, de_count
    end

    # Allocate indices for iteration ...
    ad, ad_count, be, be_count = H1b_indices(Params,Orb)
        # Allocate auxiliary fields ...
    cf, cf_count = ad, ad_count
    ab, ab_count = be, be_count

    # Allocate new 1-body Hamiltonian matrices ...
    ph_N, nh_N = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)

    # Allocate the single-particle field H ...
        # Due to 1-body contractions with the 1-body density Rho ...
        println("\t\tCalculating 1-body contractions ... due to the 1-body density matrix Rho ...")
    @time @inbounds for ad_i in 1:ad_count
        a, d = ad[1,ad_i], ad[2,ad_i]
        n_a, l_a, j_a = Orb[a].n, Orb[a].l, Orb[a].j
        n_d, l_d, j_d = Orb[d].n, Orb[d].l, Orb[d].j
        is_j_a_hat = 1.0 / Float64(j_a + 1)
        ph_N_local = zeros(Float64,Threads.maxthreadid())
        nh_N_local = zeros(Float64,Threads.maxthreadid())
        @inbounds Threads.@threads :static for be_i in 1:be_count
            b, e = be[1,be_i], be[2,be_i]
            n_b, l_b, j_b = Orb[b].n, Orb[b].l, Orb[b].j
            n_e, l_e, j_e = Orb[e].n, Orb[e].l, Orb[e].j
            P_ab = rem(l_a + l_b,2) + 1
            Tid = Threads.threadid()
            phSum_local, nhSum_local = 0.0, 0.0

            if (2*(n_a + n_b) + l_a + l_b) <= N_2max
                if l_b == l_e && j_b == j_e && (2*(n_d + n_e) + l_d + l_e) <= N_2max
                    pRho_be, nRho_be = Rho.p[b,e], Rho.n[b,e]

                    @inbounds for J = div(abs(j_d - j_e),2):div(j_d + j_e,2)

                        # 2-body NN interaction part ...
                        if P_ab == (rem(l_d + l_e, 2) + 1)
                            J_j_a_hat = Float64(2*J + 1) * is_j_a_hat
                            phSum_local += J_j_a_hat * pRho_be * O2b_pp(a,b,d,e,J,P_ab,V_NN,Orb,Orb_NN)
                            phSum_local += J_j_a_hat * nRho_be * O2b_pn(a,b,d,e,J,P_ab,V_NN,Orb_NN)
                            nhSum_local += J_j_a_hat * nRho_be * O2b_nn(a,b,d,e,J,P_ab,V_NN,Orb,Orb_NN)
                            nhSum_local += J_j_a_hat * pRho_be * O2b_pn(b,a,e,d,J,P_ab,V_NN,Orb_NN)
                        end

                        # 3-body NNN interaction part ...
                        @inbounds for c in 1:a_max
                            n_c, l_c = Orb[c].n, Orb[c].l
                            if (2*(n_a + n_b + n_c) + l_a + l_b + l_c) <= N_3max
                                P_abc = rem(l_a + l_b + l_c, 2) + 1
                                j_c = Orb[c].j
                                @inbounds for f in 1:a_max
                                    n_f, l_f = Orb[f].n, Orb[f].l
                                    if (2*(n_d + n_e + n_f) + l_d + l_e + l_f) <= N_3max && l_c == l_f && P_abc == (rem(l_d + l_e + l_f,2) + 1)
                                        j_f = Orb[f].j
                                        if j_c == j_f
                                            pRho_cf, nRho_cf = Rho.p[c,f], Rho.n[c,f]

                                            ME001 = V3b_no2b(a,b,c,0,d,e,f,0,J,1,P_abc,V_NNN,Orb,Orb_NNN)
                                            ME101 = V3b_no2b(a,b,c,1,d,e,f,0,J,1,P_abc,V_NNN,Orb,Orb_NNN)
                                            ME011 = V3b_no2b(a,b,c,0,d,e,f,1,J,1,P_abc,V_NNN,Orb,Orb_NNN)
                                            ME111 = V3b_no2b(a,b,c,1,d,e,f,1,J,1,P_abc,V_NNN,Orb,Orb_NNN)
                                            ME113 = V3b_no2b(a,b,c,1,d,e,f,1,J,3,P_abc,V_NNN,Orb,Orb_NNN)

                                            phSum_local += is_j_a_hat * (0.5*ME113*pRho_be*pRho_cf + 0.25 * (ME001 +
                                                           sqrt(1.0/3.0)*ME101 + sqrt(1.0/3.0)*ME011 + ME111/3.0 + 2.0/3.0*ME113)*nRho_be*nRho_cf +
                                                           1.0/3.0 * (2.0*ME111 + ME113)*pRho_be*nRho_cf)

                                            nhSum_local += is_j_a_hat * (0.5*ME113*nRho_be*nRho_cf + 0.25 * (ME001 +
                                                           sqrt(1.0/3.0)*ME101 + sqrt(1.0/3.0)*ME011 + ME111/3.0 + 2.0/3.0*ME113)*pRho_be*pRho_cf +
                                                           1.0/3.0 * (2.0*ME111 + ME113)*nRho_be*pRho_cf)
                                        end
                                    end
                                end
                            end
                        end
                    end
                end

            end
            ph_N_local[Tid] += phSum_local
            nh_N_local[Tid] += nhSum_local
        end

        # Sum over the Thread-local accumulators for H ...
        phSum = sum(ph_N_local)
        nhSum = sum(nh_N_local)

        # Include the 1-body kinetic energy & inclusion of Center-of-Mass motion (CM) correction ...
            # Combined 1- + 2-body kinetic operator with CM correction ...
        if CMS == "CMS1+2B"
            phSum += T.p[a,d] * (1.0 - 1.0 / A)
            nhSum += T.n[a,d] * (1.0 - 1.0 / A)
            # Pure 1-body kinetic operator with no CM correction ...
        elseif CMS != "CMS2B"
            phSum += T.p[a,d]
            nhSum += T.n[a,d]
        end
            # No contribution for pure 2-body kinetic operator with CM correction ...

        # Allocate ph & nh ...
        if a != d
            ph_N[a,d], ph_N[d,a] = phSum, phSum
            nh_N[a,d], nh_N[d,a] = nhSum, nhSum
        elseif a == d
            ph_N[a,a], nh_N[a,a] = phSum, nhSum
        end

    end

        # Due to 2-body contractions with the 2-body correlation function Sigma ...
        println("\t\tCalculating 2-body contractions ... due to the 2-body correlation function matrix Sigma ...")
    @time @inbounds for cf_i in 1:cf_count
        c, f = cf[1,cf_i], cf[2,cf_i]
        n_c, l_c, j_c = Orb[c].n, Orb[c].l, Orb[c].j
        n_f, l_f, j_f = Orb[f].n, Orb[f].l, Orb[f].j

        j_c_hat = Float64(j_c + 1)

        ph_N_local = zeros(Float64,Threads.maxthreadid())
        nh_N_local = zeros(Float64,Threads.maxthreadid())

        @inbounds Threads.@threads :static for ab_i in 1:ab_count
            a, b = ab[1,ab_i], ab[2,ab_i]
            n_a, l_a, j_a = Orb[a].n, Orb[a].l, Orb[a].j
            n_b, l_b, j_b = Orb[b].n, Orb[b].l, Orb[b].j

            Tid = Threads.threadid()
            pSum, nSum = 0.0, 0.0

            if (2*(n_a + n_b) + l_a + l_b) <= N_2max && (2*(n_a + n_b + n_c) + l_a + l_b + l_c) <= N_3max
                P2B = rem(l_a + l_b, 2) + 1
                P3B = rem(l_a + l_b + l_c, 2) + 1
                @inbounds for J in div(abs(j_a - j_b),2):div((j_a + j_b),2)

                    @inbounds for d in 1:a_max
                        n_d, l_d, j_d = Orb[d].n, Orb[d].l, Orb[d].j

                        @inbounds @simd for e in 1:a_max
                            n_e, l_e, j_e = Orb[e].n, Orb[e].l, Orb[e].j

                            if (2*(n_d + n_e) + l_d + l_e) <= N_2max && (2*(n_d + n_e + n_f) + l_d + l_e + l_f) <= N_3max &&
                            rem(l_d + l_e, 2) + 1 == P2B && rem(l_d + l_e + l_f, 2) + 1 == P3B && abs(j_d - j_e) <= 2*J && 2*J <= (j_d + j_e)

                                ppSigma_abde = O2b_pp(a,b,d,e,J,P2B,Sigma_NN,Orb,Orb_NN)
                                pnSigma_abde = O2b_pn(a,b,d,e,J,P2B,Sigma_NN,Orb_NN)
                                pnSigma_baed = O2b_pn(b,a,e,d,J,P2B,Sigma_NN,Orb_NN)
                                nnSigma_abde = O2b_nn(a,b,d,e,J,P2B,Sigma_NN,Orb,Orb_NN)

                                ME001 = V3b_no2b(a,b,c,0,d,e,f,0,J,1,P3B,V_NNN,Orb,Orb_NNN)
                                ME101 = V3b_no2b(a,b,c,1,d,e,f,0,J,1,P3B,V_NNN,Orb,Orb_NNN)
                                ME011 = V3b_no2b(a,b,c,0,d,e,f,1,J,1,P3B,V_NNN,Orb,Orb_NNN)
                                ME111 = V3b_no2b(a,b,c,1,d,e,f,1,J,1,P3B,V_NNN,Orb,Orb_NNN)
                                ME113 = V3b_no2b(a,b,c,1,d,e,f,1,J,3,P3B,V_NNN,Orb,Orb_NNN)

                                npME001 = V3b_no2b(b,a,c,0,e,d,f,0,J,1,P3B,V_NNN,Orb,Orb_NNN)
                                npME101 = V3b_no2b(b,a,c,1,e,d,f,0,J,1,P3B,V_NNN,Orb,Orb_NNN)
                                npME011 = V3b_no2b(b,a,c,0,e,d,f,1,J,1,P3B,V_NNN,Orb,Orb_NNN)
                                npME111 = V3b_no2b(b,a,c,1,e,d,f,1,J,1,P3B,V_NNN,Orb,Orb_NNN)
                                npME113 = V3b_no2b(b,a,c,1,e,d,f,1,J,3,P3B,V_NNN,Orb,Orb_NNN)

                                @inbounds pSum += 0.25 / j_c_hat * (ME113 * ppSigma_abde + (2.0 * ME111 + ME113) / 3.0 * nnSigma_abde +
                                        2.0 * (ME001 - ME101 / sqrt(3.0) - ME011 / sqrt(3.0) + ME111 / 3.0 + 2.0 / 3.0 * ME113) * pnSigma_abde)
                                @inbounds nSum += 0.25 / j_c_hat * (ME113 * nnSigma_abde + (2.0 * ME111 + ME113) / 3.0 * ppSigma_abde +
                                        2.0 * (npME001 - npME101 / sqrt(3.0) - npME011 / sqrt(3.0) + npME111 / 3.0 + 2.0 / 3.0 * npME113) * pnSigma_baed)

                            end

                        end
                    end
                end
            end

            ph_N_local[Tid] += pSum
            nh_N_local[Tid] += nSum
        end

        @inbounds ph_N[c,f] += sum(ph_N_local)
        @inbounds nh_N[c,f] += sum(nh_N_local)

        if c != f
            @inbounds ph_N[f,c] += sum(ph_N_local)
            @inbounds nh_N[f,c] += sum(nh_N_local)
        end
    end

    println("\tCompleted the calculation of 1-body mean-field Hamiltonian h_N ...")

    return O1B(ph_N,nh_N)
end