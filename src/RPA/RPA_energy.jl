function RPA_energy(Params::Parameters,Orb::Vector{Orb1B},Orb_1p1h::Orb2B_1p1h,Orb_NN::Orb2B,E::Matrix{Vector{ComplexF64}},Y::Matrix{Matrix{ComplexF64}},Rho::O1B,h_N::O1B,Sigma::O2B,V_NN::O2B)
    # Read parameters ...
    N_max = Params.Calc.Nmax
    J_max = 2*N_max + 1
    Orthogon = Params.Calc.RPA.Ortho

    # Initialize the list of values of J & P for iteration ...
    JP_list = JP_initialize(J_max)

    println("\nCalculating RPA correlation energy ...")

    # Initialize variables for correlation energy ...
    E_corr, E_corr_threads = 0.0, zeros(Float64,Threads.maxthreadid())

    # Calculate the RPA correlation energy ...
    @inbounds Threads.@threads for JP in JP_list
        Sum, Tid = 0.0, Threads.threadid()
        J, P = JP[1], JP[2]

        N_ph = Orb_1p1h.N[P,J+1]
        Orb_1p1h_JP = Orb_1p1h.Ind[P,J+1]

        J_hat = Float64(2*J + 1)

        E_JP, Y_JP = E[P,J+1], Y[P,J+1]

        @inbounds for i_ph in 1:N_ph
            Orb_ph = Orb_1p1h_JP[i_ph]
            a_p, a_h, T_ph = Orb_ph.a, Orb_ph.b, Orb_ph.T
            O_p, O_h = 0.0, 0.0
            if T_ph == -1
                O_p, O_h = Orb[a_p].pO, Orb[a_h].pO
            elseif T_ph == 1
                O_p, O_h = Orb[a_p].nO, Orb[a_h].nO
            end
            @inbounds for i_nu in 1:N_ph
                ME = -J_hat * real(E_JP[i_nu]) * (O_h - O_p) * abs(Y_JP[i_ph,i_nu])^2
                Sum += ME
            end
        end

        E_corr_threads[Tid] += Sum
    end

    # Accumulate the total RPA correlation energy from all threads ...
    E_corr = sum(E_corr_threads)

    # Calculate RPA correlation energy from the 2-body correlation function ...
    E_corr_sigma = V2b_correlation_energy(Params,Orb,Orb_NN,Sigma,V_NN)

    # Calculate RPA correlation energy from the 1-body correlated density ...
            #=
            E_corr_rho, E_corr_rho_threads = 0.0, zeros(Float64,Threads.maxthreadid())
            @inbounds Threads.@threads for JP in JP_list
                Sum, Tid = 0.0, Threads.threadid()
                J, P = JP[1], JP[2]

                N_ph = Orb_1p1h.N[P,J+1]
                Orb_1p1h_JP = Orb_1p1h.Ind[P,J+1]

                J_hat = Float64(2*J + 1)

                E_JP, Y_JP = E[P,J+1], Y[P,J+1]

                @inbounds for i_ph in 1:N_ph
                    Orb_ph = Orb_1p1h_JP[i_ph]
                    a_p, a_h, T_ph = Orb_ph.a, Orb_ph.b, Orb_ph.T
                    E_p, E_h = 0.0, 0.0
                    if T_ph == -1
                        E_p, E_h = h_N.p[a_p,a_p], h_N.p[a_h,a_h]
                    elseif T_ph == 1
                        E_p, E_h = h_N.n[a_p,a_p], h_N.n[a_h,a_h]
                    end
                    @inbounds for i_nu in 1:N_ph
                        ME = J_hat * (E_p - E_h) * abs(Y_JP[i_ph,i_nu])^2
                        Sum += ME
                    end
                end

                E_corr_rho_threads[Tid] += Sum
            end

            # Accumulate the total RPA correlation energy from all threads ...
            E_corr_rho = sum(E_corr_rho_threads)
            =#

    E_corr_rho = 0.0
    @inbounds for Orb_a in Orb
        a, j_a = Orb_a.a, Orb_a.j
        j_a_hat = Float64(j_a + 1)
        E_corr_rho += j_a_hat * (h_N.p[a,a] * (Rho.p[a,a] - Orb_a.pO) + h_N.n[a,a] * (Rho.n[a,a] - Orb_a.nO))
    end

    println("\tRPA correlation energy (Algebraic identity)   ...         E_RPA = " * string(round(real(E_corr),digits=6)) * " MeV")
    println("\tRPA correlation energy (Correlation 2-body function) ...  E_RPA = " * string(round(real(E_corr_sigma),digits=6)) * " MeV")
    println("\tRPA correlation energy (Correlated 1-body density) ...    E_RPA = " * string(round(real(E_corr_rho),digits=6)) * " MeV\n")

    return E_corr
end