function QRPA_allocate(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,Orb_2qp::qpOrb2B,H_N::qpO1B,H_NN::qpO2B)
    # Read parameters ...
    N_max = Params.Calc.Nmax
    J_max = 2*N_max + 1

    # Make the list of values of J & P for iteration ...
    JP_list = JP_initialize(J_max)

    # Initialize the QTDA matrix A ...
    A = Matrix{Matrix{Float64}}(undef,2,J_max+1)
    B = Matrix{Matrix{Float64}}(undef,2,J_max+1)

    # Allocate the entries of the QRPA matrices A & B ...
    println("\nAllocating the QRPA matrices A & B ...")
    @inbounds Threads.@threads for JP in JP_list
        J, P = JP[1], JP[2]
        N_qp = Orb_2qp.N[P,J+1]
        A_JP = Matrix{Float64}(undef,N_qp,N_qp)
        B_JP = Matrix{Float64}(undef,N_qp,N_qp)
        @inbounds for i_qp in 1:N_qp
            a, b, T_ab = Orb_2qp.i[P,J+1][i_qp].a, Orb_2qp.i[P,J+1][i_qp].b, Orb_2qp.i[P,J+1][i_qp].T

            @inbounds for j_qp in 1:i_qp
                c, d, T_cd = Orb_2qp.i[P,J+1][j_qp].a, Orb_2qp.i[P,J+1][j_qp].b, Orb_2qp.i[P,J+1][j_qp].T

                ASum, BSum = 0.0, 0.0

                Norm = 1.0 / sqrt(Float64((1 + kronecker_delta(a,b)) * (1 + kronecker_delta(c,d))))

                # 1-body part of A ...
                if a == c && b == d && T_ab == T_cd
                    if T_ab == -1
                        E_a = H_N.qp11.p[a,a]
                        E_b = H_N.qp11.p[b,b]
                    elseif T_ab == 1
                        E_a = H_N.qp11.n[a,a]
                        E_b = H_N.qp11.n[b,b]
                    end
                    ASum += (E_a + E_b)
                end

                # 2-body part of A ...
                if T_ab == -1 && T_cd == -1
                    ME = Norm * qpO2b_22_pp(a,b,c,d,J,P,H_NN,Orb,Orb_NN)
                    ASum += ME
                elseif T_ab == -1 && T_cd == 1
                    ME = Norm * qpO2b_2002_pn(a,b,c,d,J,P,H_NN,Orb,Orb_NN)
                    ASum += ME
                elseif T_ab == 1 && T_cd == -1
                    ME = Norm * qpO2b_2002_pn(c,d,a,b,J,P,H_NN,Orb,Orb_NN)
                    ASum += ME
                elseif T_ab == 1 && T_cd == 1
                    ME = Norm * qpO2b_22_nn(a,b,c,d,J,P,H_NN,Orb,Orb_NN)
                    ASum += ME
                end

                # 2-body part of B ...
                if T_ab == -1 && T_cd == -1
                    ME = Norm * qpO2b_40_pp(a,b,c,d,J,P,H_NN,Orb,Orb_NN)
                    BSum += ME
                elseif T_ab == -1 && T_cd == 1
                    ME = Norm * qpO2b_40_pn(a,b,c,d,J,P,H_NN,Orb,Orb_NN)
                    BSum += ME
                elseif T_ab == 1 && T_cd == -1
                    ME = Norm * qpO2b_40_pn(c,d,a,b,J,P,H_NN,Orb,Orb_NN)
                    BSum += ME
                elseif T_ab == 1 && T_cd == 1
                    ME = Norm * qpO2b_40_nn(a,b,c,d,J,P,H_NN,Orb,Orb_NN)
                    BSum += ME
                end

                # Allocate the MEs ...
                A_JP[i_qp,j_qp] = ASum
                B_JP[i_qp,j_qp] = BSum

                # Case of h.c. terms ...
                if i_qp != j_qp
                    A_JP[j_qp,i_qp] = ASum
                    B_JP[j_qp,i_qp] = BSum
                end

            end

        end

        # Allocate JP block ...
        A[P,J+1] = A_JP
        B[P,J+1] = B_JP
    end

    println("\nThe QRPA matrices A & B succesfully allocated ...")

    return A, B
end