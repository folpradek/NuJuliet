function QRPA_A_phonon_particle_number(Params::Parameters,Orb::Vector{Orb1B},Orb_2qp::qpOrb2B,U::O1B,V::O1B,X_QRPA::Matrix{Matrix{ComplexF64}},Y_QRPA::Matrix{Matrix{ComplexF64}})
    # Read parameters ...
    N_max = Params.Calc.Nmax
    J_max = 2*N_max + 1
    a_max = div((N_max + 1)*(N_max + 2),2)

    # Make the list of values of J & P for iteration ...
    JP_list = JP_initialize(J_max)

    # Initialize 1-body particle number operator (20) components for even & odd values of J ...
    pN_even, pN_odd = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)
    nN_even, nN_odd = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)


    # Evaluate the 1-body particle number operator (20) matrix elements in the QRPA phonon basis ...
    @inbounds Threads.@threads for a in 1:a_max
        l_a , j_a = Orb[a].l, Orb[a].j
        @inbounds for b in 1:a_max
            l_b, j_b = Orb[b].l, Orb[b].j

            pNEvenSum, pNOddSum = 0.0, 0.0
            nNEvenSum, nNOddSum = 0.0, 0.0

            @inbounds for k in 1:a_max
                l_k, j_k = Orb[k].l, Orb[k].j

                if l_a == l_k && j_a == j_k
                    @inbounds for l in 1:a_max
                        l_l, j_l = Orb[l].l, Orb[l].j

                        if l_l == l_b && j_l == j_b
                            # Read the quasiparticle amplitudes ...
                            pU_ka, pV_ka = U.p[k,a], V.p[k,a]
                            pU_lb, pV_lb = U.p[l,b], V.p[l,b]
                            nU_ka, nV_ka = U.n[k,a], V.n[k,a]
                            nU_lb, nV_lb = U.n[l,b], V.n[l,b]

                            # Case of Even J ...
                            pNEven =  (pV_ka * pU_lb + pU_ka * pV_lb)
                            nNEven =  (nV_ka * nU_lb + nU_ka * nV_lb)

                            pNEvenSum += pNEven
                            nNEvenSum += nNEven

                            # Case of Odd J ...
                            pNOdd =  (pV_ka * pU_lb - pU_ka * pV_lb)
                            nNOdd =  (nV_ka * nU_lb - nU_ka * nV_lb)

                            pNOddSum += pNOdd
                            nNOddSum += nNOdd
                        end
                    end
                end
            end

            pN_even[a,b], pN_odd[a,b] = pNEvenSum, pNOddSum
            nN_even[a,b], nN_odd[a,b] = nNEvenSum, nNOddSum
        end
    end

    # Initialize a field to mask non-A-body 1-phonon levels ...
    A_phonon = Matrix{Vector{Bool}}(undef,2,J_max+1)

    # Evaluate the mask of non-A-body 1-phonon levels ...
    @inbounds for JP in JP_list
        J, P = JP[1], JP[2]
        N_qp = Orb_2qp.N[P,J+1]
        A_phonon[P,J+1] = falses(N_qp)

        @inbounds for nu in 1:N_qp
            Sum = 0.0

            @inbounds for qp in 1:N_qp
                a, b, T_ab = Orb_2qp.i[P,J+1][qp].a, Orb_2qp.i[P,J+1][qp].b, Orb_2qp.i[P,J+1][qp].T
                Norm = ComplexF64(1.0 / sqrt(Float64(1 + kronecker_delta(a,b))))
                Amp = Norm * (X_QRPA[P,J+1][qp,nu] - Y_QRPA[P,J+1][qp,nu])

                if T_ab == -1
                    if rem(J,2) == 0
                        pNME = Amp * ComplexF64(pN_even[a,b])
                        Sum += pNME
                    elseif rem(J,2) == 1
                        pNME = Amp * ComplexF64(pN_odd[a,b])
                        Sum += pNME
                    end

                elseif T_ab == 1
                    if rem(J,2) == 0
                        nNME = Amp * ComplexF64(nN_even[a,b])
                        Sum += nNME
                    elseif rem(J,2) == 1
                        nNME = Amp * ComplexF64(nN_odd[a,b])
                        Sum += nNME
                    end
                end
            end

            if abs(Sum) > 1e-8
                A_phonon[P,J+1][nu] = true
            end

        end
    end

    return A_phonon
end