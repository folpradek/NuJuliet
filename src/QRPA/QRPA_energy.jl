function QRPA_energy(Params::Parameters,Orb_2qp::qpOrb2B,E_QRPA::Matrix{Vector{ComplexF64}},Y_QRPA::Matrix{Matrix{ComplexF64}},A_phonon::Matrix{Vector{Bool}})
    # Read parameters ...
    N_max = Params.Calc.Nmax
    J_max = 2*N_max + 1

    # Make the list of values of J & P for iteration ...
    JP_list = JP_initialize(J_max)

    # Initialize the accumulator for QRPA correlation energy ...
    E_corr_threads = zeros(Float64,Threads.maxthreadid())


    println("\nCalculating the total QRPA correlation energy ...")
    @inbounds Threads.@threads for JP in JP_list
        Sum, Tid = 0.0, Threads.threadid()
        J, P = JP[1], JP[2]
        N_qp = Orb_2qp.N[P,J+1]
        J_hat = Float64(2*J + 1)

        @inbounds for qp in 1:N_qp
            a, b = Orb_2qp.i[P,J+1][qp].a, Orb_2qp.i[P,J+1][qp].b
            N = -1.0 / Float64(1 + kronecker_delta(a,b))

            @inbounds for nu in 1:N_qp
                if A_phonon[P,J+1][nu] == false
                    continue
                end

                ME = N * real(E_QRPA[P,J+1][nu]) * abs(Y_QRPA[P,J+1][qp,nu])^2
                Sum += ME
            end
        end
        
        Sum = J_hat * Sum
        E_corr_threads[Tid] += Sum
    end

    # Accumulate the total QRPA correlation energy ...
    E_corr = sum(E_corr_threads)

    @printf("\n\tTotal QRPA correlation energy ...  E_QRPA = %12.6f MeV\n", E_corr)

    println("\tCalculation of the total QRPA correlation energy completed ...")

    return E_corr
end