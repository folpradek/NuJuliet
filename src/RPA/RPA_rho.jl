function RPA_OBDM(Params::Parameters,Orb::Vector{Orb1B},Orb_1p1h::Orb2B_1p1h,Y::Matrix{Matrix{ComplexF64}})
    # Read calculation parameters ..
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1
    a_max = div((N_max + 1)*(N_max + 2),2)

    # Initialize the list of values of J & P for iteration ...
    JP_list = JP_initialize(J_max)

    # Initialize density matrices ...
    pRho, nRho = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)

    @inbounds for a in 1:a_max
        pRho[a,a] = Orb[a].pO
        nRho[a,a] = Orb[a].nO
    end

    pRho_threads, nRho_threads = Vector{Matrix{Float64}}(undef,Threads.maxthreadid()), Vector{Matrix{Float64}}(undef,Threads.maxthreadid())
    @inbounds for Tid in 1:Threads.maxthreadid()
        pRho_threads[Tid] = zeros(Float64,a_max,a_max)
        nRho_threads[Tid] = zeros(Float64,a_max,a_max)
    end

    # Evaluate RPA OBDM ...
    @inbounds Threads.@threads for JP in JP_list
        J, P = JP[1], JP[2]
        N_ph = Orb_1p1h.N[P,J+1]
        Orb_1p1h_JP = Orb_1p1h.Ind[P,J+1]

        pRho_Tid = pRho_threads[Threads.threadid()]
        nRho_Tid = nRho_threads[Threads.threadid()]

        @inbounds for (i_ph,Orb_ph) in enumerate(Orb_1p1h_JP)
            a_p, a_h, T_ph = Orb_ph.a, Orb_ph.b, Orb_ph.T
            j_p, j_h = Orb[a_p].j, Orb[a_h].j

            Sum = 0.0
            @inbounds for i_nu in 1:N_ph
                Y_norm = @views norm(Y[P,J+1][:,i_nu])^2
                if abs(Y_norm) < 0.5
                    y = abs2(Y[P,J+1][i_ph,i_nu])
                    ME = y * Float64(2*J + 1)
                    Sum += ME
                end
            end

            if T_ph == -1
                pRho_Tid[a_h,a_h] -= Sum / Float64(j_h + 1)
                pRho_Tid[a_p,a_p] += Sum / Float64(j_p + 1)
            elseif T_ph == 1
                nRho_Tid[a_h,a_h] -= Sum / Float64(j_h + 1)
                nRho_Tid[a_p,a_p] += Sum / Float64(j_p + 1)
            end
        end
    end

    @inbounds for Tid in 1:Threads.maxthreadid()
        pRho .+= pRho_threads[Tid]
        nRho .+= nRho_threads[Tid]
    end    

    return O1B(pRho,nRho)
end