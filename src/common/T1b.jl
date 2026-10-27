function T1b(Params::Parameters,Orb::Vector{Orb1B})
    # Read parameters
    N_max, hw = Params.Int.Nmax, Params.Calc.hw
    a_max = div((N_max + 1)*(N_max + 2),2)

    # Allocate the 1-body kinetic operator ... LHO basis ...
    T = zeros(Float64,a_max,a_max)
    @inbounds for k = 1:a_max
        n_k, l_k, j_k = Orb[k].n, Orb[k].l, Orb[k].j
        N_k = Float64(2*n_k + l_k)
        T[k,k] += 0.5 * hw * (N_k + 1.5)
        @inbounds for l = 1:a_max
            n_l, l_l, j_l = Orb[l].n, Orb[l].l, Orb[l].j
            if l_k == l_l && j_k == j_l
                if n_k == (n_l + 1)
                    T[k,l] += 0.5 * hw * sqrt(0.5 * Float64(n_k * (2*n_k + 2*l_k + 1)))
                end
                if (n_k + 1) == n_l
                    T[k,l] += 0.5 * hw * sqrt(0.5 * Float64(n_l * (2*n_l + 2*l_l + 1)))
                end
            end
        end
    end
    return O1B(T,T)
end

function T1b_energy(Params::Parameters,Rho::O1B,Orb::Vector{Orb1B},T::O1B)
    # Read parameters ...
    A = Params.Calc.A
    hw = Params.Calc.hw
    N_max = Params.Calc.Nmax
    N_2max = Params.Calc.N2max
    CMS = Params.Calc.CMS
    a_max = div((N_max + 1)*(N_max + 2),2)

    println("\nCalculating the total kinetic energy of given ground-state reference state ...")
    # Calculate the total kinetic energy ...
    t_threads = zeros(Float64,Threads.maxthreadid())

    # Include the 2-body CM correction to the kinetic energy ...
    @inbounds Threads.@threads :static for a = 1:a_max
        Sum, Tid = 0.0, Threads.threadid()
        n_a = Orb[a].n
        l_a = Orb[a].l
        j_a = Orb[a].j
        @inbounds for d = 1:a_max
            n_d = Orb[d].n
            l_d = Orb[d].l
            j_d = Orb[d].j
            if l_a == l_d && j_a == j_d

                # 2-body CM correction to the kinetic energy ...
                @inbounds for b = 1:a_max
                    n_b = Orb[b].n
                    l_b = Orb[b].l
                    j_b = Orb[b].j
                    if (2*(n_a + n_b) + l_a + l_b) <= N_2max
                        @inbounds for e = 1:a_max
                            n_e = Orb[e].n
                            l_e = Orb[e].l
                            j_e = Orb[e].j
                            if l_b == l_e && j_b == j_e && (2*(n_d + n_e) + l_d + l_e) <= N_2max

                                if rem(l_a + l_b, 2) == rem(l_d + l_e, 2)
                                    @inbounds for J = div(abs(j_a - j_b),2):div((j_a + j_b),2)
                                        T_NN_Amp = Float64(2*J + 1) / Float64(A)

                                        if CMS == "CMS1+2B"
                                            TNN_sym =  T2b(Orb,a,b,d,e,J) * hw
    
                                            TNN_antisym = 1.0 / sqrt(Float64((1 + kronecker_delta(a,b))*(1 + kronecker_delta(d,e)))) * (T2b(Orb,a,b,d,e,J) -
                                                          Float64((-1)^(round(div(j_d + j_e,2) - J))) * T2b(Orb,a,b,e,d,J)) * hw
                                        elseif CMS == "CMS2B"
                                            Amp = hw / sqrt(Float64(1 + kronecker_delta(a,b)) * Float64(1 + kronecker_delta(d,e)))
                                            Amp_2 = 1.0 / sqrt(Float64(1 + kronecker_delta(a,b)) * Float64(1 + kronecker_delta(d,e)))
    
                                            TNN_sym = hw * T2b(Orb,a,b,d,e,J) + T.p[a,d] * kronecker_delta(b,e) + T.p[b,e] * kronecker_delta(a,d)

                                            TNN_antisym = (Amp * T2b(Orb,a,b,d,e,J) + Amp_2 * (kronecker_delta(b,e) * T.p[a,d] + kronecker_delta(a,d) * T.p[b,e])
                                                        - Float64((-1)^(div(j_d + j_e,2) - J)) * (Amp * T2b(Orb,a,b,e,d,J) + Amp_2 * (Float64(kronecker_delta(b,d)) *
                                                        T.p[a,e] + Float64(kronecker_delta(a,e)) * T.p[b,d]))) / Float64(A)
                                        else
                                            TNN_sym = 0.0
                                            TNN_antisym = 0.0
                                        end

                                        Sum += T_NN_Amp * TNN_antisym * Rho.p[b,e]
                                        Sum += T_NN_Amp * TNN_sym * Rho.n[b,e]
                                        Sum += T_NN_Amp * TNN_antisym * Rho.n[b,e]
                                        Sum += T_NN_Amp * TNN_sym * Rho.p[b,e]

                                    end
                                end

                            end
                        end
                    end
                end

                # 1-body kinetic operator & CMS correction ...
                if CMS == "CMS1+2B"
                    Sum += T.p[a,d] * (1.0 - 1.0 / Float64(A)) * Rho.p[a,d] * Float64(Orb[a].j + 1)
                    Sum += T.n[a,d] * (1.0 - 1.0 / Float64(A)) * Rho.n[a,d] * Float64(Orb[a].j + 1)
                elseif CMS != "CMS2B"
                    Sum += T.p[a,d] * Rho.p[a,d]
                    Sum += T.n[a,d] * Rho.n[a,d]
                end

            end
        end
        t_threads[Tid] += Sum
    end

    t = sum(t_threads)

    println("\tTotal kinetic energy reads:    <T> = " * string(t) * " MeV")

    return t
end