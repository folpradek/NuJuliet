function pnQTDA_allocate(Params::Parameters,Orb_NN::Orb2B,Orb_pn2qp::Orb2B_2qp,H_N::qpO1B,H_NN::qpO2B)
    # Read parameters ...
    N_max = Params.Calc.Nmax
    J_max = 2*N_max + 1

    # Make the list of values of J & P for iteration ...
    JP_list = JP_initialize(J_max)

    # Initialize the pnQTDA matrix A ...
    A = Matrix{Matrix{Float64}}(undef,2,J_max+1)

    # Allocate the entries of the pnQTDA matrix A ...
    println("\nAllocating the pnQTDA matrix A ...")
    @inbounds Threads.@threads for JP in JP_list
        J, P = JP[1], JP[2]
        N_qp = Orb_pn2qp.N[P,J+1]
        Orb_pn2qp_JP = Orb_pn2qp.Ind[P,J+1]

        A_JP = Matrix{Float64}(undef,N_qp,N_qp)

        @inbounds for i_qp in 1:N_qp
            a, b = Orb_pn2qp_JP[i_qp].a, Orb_pn2qp_JP[i_qp].b

            @inbounds for j_qp in 1:i_qp
                c, d = Orb_pn2qp_JP[j_qp].a, Orb_pn2qp_JP[j_qp].b

                ASum = 0.0

                # 1-body part of A ...
                if a == c && b == d
                    E_a = H_N.qp11.p[a,a]
                    E_b = H_N.qp11.n[b,b]
                    ASum += (E_a + E_b)
                end

                # 2-body part of A ... proton-neutron (11,11) component of H^(22) ...
                ME = qpO2b_1111_pn(a,b,c,d,J,P,H_NN,Orb_NN)
                ASum += ME

                # Allocate the MEs ...
                A_JP[i_qp,j_qp] = ASum

                # Case of h.c. terms ...
                if i_qp != j_qp
                    A_JP[j_qp,i_qp] = ASum
                end
            end
        end

        # Allocate given JP block of A ...
        A[P,J+1] = A_JP
    end

    println("\nThe pnQTDA matrix A succesfully allocated ...")

    return A
end
