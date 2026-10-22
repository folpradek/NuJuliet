function QRPA_OBDM(Params::Parameters,Orb::Vector{Orb1B},Orb_2qp::qpOrb2B,U::O1B,V::O1B,Y_QRPA::Matrix{Matrix{ComplexF64}},A_phonon::Matrix{Vector{Bool}})
    # Read calculation parameters ..
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1
    a_max = div((N_max + 1)*(N_max + 2),2)

    # Make the list of values of J & P for iteration ...
    JP_list = JP_initialize(J_max)

    # Initialize new OBDM Rhp ... corrected by QRPA ...
    pRho, nRho = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)

    # Initialize quasiparticle occupation numbers ...
    pN, nN = zeros(Float64,a_max), zeros(Float64,a_max)
        # Threaded formulation ...
    pN_thread, nN_thread = Vector{Vector{Float64}}(undef,Threads.maxthreadid()), Vector{Vector{Float64}}(undef,Threads.maxthreadid())
    @inbounds for Tid in 1:Threads.maxthreadid()
        pN_thread[Tid] = zeros(Float64,a_max)
        nN_thread[Tid] = zeros(Float64,a_max)
    end

    # Calculate the QRPA corrections to the reference mean-field OBDM ...
    @inbounds Threads.@threads for JP in JP_list
        Tid = Threads.threadid()
        J, P = JP[1], JP[2]
        N_qp = Orb_2qp.N[P,J+1]
        J_hat = Float64(2*J + 1)

        @inbounds for qp in 1:N_qp
            a, b, T_ab = Orb_2qp.i[P,J+1][qp].a, Orb_2qp.i[P,J+1][qp].b, Orb_2qp.i[P,J+1][qp].T
            j_a, j_b = Orb[a].j, Orb[b].j
            j_a_hat, j_b_hat = Float64(j_a + 1), Float64(j_b + 1)

            Amp_a = J_hat / j_a_hat / sqrt(Float64(1 + kronecker_delta(a,b)))
            Amp_b = J_hat / j_b_hat / sqrt(Float64(1 + kronecker_delta(a,b)))

            aSum, bSum = 0.0, 0.0

            @inbounds for nu in 1:N_qp
                if A_phonon[P,J+1][nu] == false
                    continue
                end

                aME = Amp_a * abs2(Y_QRPA[P,J+1][qp,nu])
                bME = Amp_b * abs2(Y_QRPA[P,J+1][qp,nu])

                if b <= a
                    aSum += aME
                end

                if a <= b
                    bSum += bME
                end
            end

            if T_ab == -1
                pN_thread[Tid][a] += aSum
                pN_thread[Tid][b] += bSum
            elseif T_ab == 1
                nN_thread[Tid][a] += aSum
                nN_thread[Tid][b] += bSum
            end

        end
    end

    # Sum the contributions from each thread to get the final quasiparticle occupation numbers ...
    @inbounds for a in 1:a_max
        pN[a] = sum(pN_thread[Tid][a] for Tid in 1:Threads.maxthreadid())
        nN[a] = sum(nN_thread[Tid][a] for Tid in 1:Threads.maxthreadid())
    end

    # Include the QRPA corrections to the OBDM ...
    @inbounds for a in 1:a_max
        pSum, nSum = 0.0, 0.0
        @inbounds for mu in 1:a_max
            pME = V.p[a,mu] * V.p[a,mu] * (1.0  - pN[mu]) + U.p[a,mu] * U.p[a,mu] * pN[mu]
            nME = V.n[a,mu] * V.n[a,mu] * (1.0  - nN[mu]) + U.n[a,mu] * U.n[a,mu] * nN[mu]
            pSum += pME
            nSum += nME
        end
        pRho[a,a] = pSum
        nRho[a,a] = nSum
    end

    return O1B(pRho,nRho)
end