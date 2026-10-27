function HF_BCS_BMBPT_energy(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,H_N::qpO1B,H_NN::qpO2B)
    # Read parameters ...
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1)*(N_max + 2),2)
    IO = Params.Calc.Path

    # Initialize the thread accumulators ...
    dE, dE_threads = 0.0, zeros(Float64,Threads.maxthreadid())

    # Calculate the HF-BCS-BMBPT(2) energy correction ...
    println("\nCalculating HF-BCS-BMBPT(2) ground-state energy correction ... ")

    # Define a local function that prepares all the possible combinations of ab pairs ...
    function HF_BCS_BMBPT2_energy_indices(Params::Parameters)
        # Read parameters ...
        N_max = Params.Calc.Nmax
        a_max = div((N_max + 1)*(N_max + 2),2)

        # Initialite the counter for ab pairs ...
        ab_count = 0

        # Count the number of ab pairs ...
        @inbounds for T in -1:1
            @inbounds for a in 1:a_max
                @inbounds for b in 1:a_max
                    ab_count += 1
                end
            end
        end

        # Initialite the list of ab pairs ...
        ab_list = Vector{Tuple{Int64,Int64,Int64}}(undef,ab_count)

        # Reset the counter ...
        ab_count = 0

        # Allocate the list of ab pairs ...
        @inbounds for T in -1:1
            @inbounds for a in 1:a_max
                @inbounds for b in 1:a_max
                    ab_count += 1
                    ab_list[ab_count] = (a,b,T)
                end
            end
        end

        return ab_list
    end

    # Calculate the ab pairs for BMBPT(2) iteration ...
    ab_list =  HF_BCS_BMBPT2_energy_indices(Params)

    @inbounds Threads.@threads :static for ab in ab_list
        Sum, Tid = 0.0, Threads.threadid()
        a, b, T_ab = ab[1], ab[2], ab[3]
        l_a, j_a = Orb[a].l, Orb[a].j
        l_b, j_b = Orb[b].l, Orb[b].j
        P = rem(l_a + l_b,2) + 1

        # T = -1 proton-proton branch ...
        if T_ab == -1
            E_a, E_b = H_N.qp11.p[a,a], H_N.qp11.p[b,b]
            @inbounds for c in 1:a_max
                l_c, j_c, E_c = Orb[c].l, Orb[c].j, H_N.qp11.p[c,c]
                @inbounds for d in 1:a_max
                    l_d, j_d, E_d = Orb[d].l, Orb[d].j, H_N.qp11.p[d,d]
                    if (rem(l_c + l_d,2) + 1) == P
                        Denom = -1.0 / (E_a + E_b + E_c + E_d) / 24.0
                        @inbounds for J in max(div(abs(j_a - j_b),2),div(abs(j_c - j_d),2)):min(div(j_a + j_b,2),div(j_c + j_d,2))
                            ME = Float64(2*J + 1) * Denom * abs(qpO2b_40_pp(a,b,c,d,J,P,H_NN,Orb,Orb_NN))^2
                            Sum += ME
                        end
                    end
                end
            end

        # T = 0 proton-neutron branch ...
        elseif T_ab == 0
            E_a, E_b = H_N.qp11.p[a,a], H_N.qp11.p[b,b] 
            @inbounds for c in 1:a_max
                l_c, j_c, E_c = Orb[c].l, Orb[c].j, H_N.qp11.n[c,c]
                @inbounds for d in 1:a_max
                    l_d, j_d, E_d = Orb[d].l, Orb[d].j, H_N.qp11.n[d,d]
                    if (rem(l_c + l_d,2) + 1) == P
                        Denom = -1.0 / (E_a + E_b + E_c + E_d) / 4.0
                        @inbounds for J in max(div(abs(j_a - j_b),2),div(abs(j_c - j_d),2)):min(div(j_a + j_b,2),div(j_c + j_d,2))
                            ME = Float64(2*J + 1) * Denom * abs(qpO2b_40_pn(a,b,c,d,J,P,H_NN,Orb,Orb_NN))^2
                            Sum += ME
                        end
                    end
                end
            end

        # T = 1 neutron-neutron branch
        elseif T_ab == 1
            E_a, E_b = H_N.qp11.n[a,a], H_N.qp11.n[b,b]
            @inbounds for c in 1:a_max
                l_c, j_c, E_c = Orb[c].l, Orb[c].j, H_N.qp11.n[c,c]
                @inbounds for d in 1:a_max
                    l_d, j_d, E_d = Orb[d].l, Orb[d].j, H_N.qp11.n[d,d]
                    if (rem(l_c + l_d,2) + 1) == P
                        Denom = -1.0 / (E_a + E_b + E_c + E_d) / 24.0
                        @inbounds for J in max(div(abs(j_a - j_b),2),div(abs(j_c - j_d),2)):min(div(j_a + j_b,2),div(j_c + j_d,2))
                            ME = Float64(2*J + 1) * Denom * abs(qpO2b_40_nn(a,b,c,d,J,P,H_NN,Orb,Orb_NN))^2
                            Sum += ME
                        end
                    end
                end
            end
        end
        dE_threads[Tid] += Sum
    end

    # Get the total HF-BCS-BMBPT(2) energy from the thread accumulators ...
    dE = sum(dE_threads)

    # Print the total HF-BCS-BMBPT(2) energy correction ...
    println("\tHF-BCS-BMBPT(2) energy    ...   E^(2) = " * string(round(dE, sigdigits=9)) * " MeV\n")

    # Write the total HF-BCS-BMBPT(2) energy correction to the summary file ...
    Summary =  open(string("io/" * IO * "/BCS/BCS_summary.dat"), "a")
        println(Summary, "\nSpherical Hartree-Fock-BCS Leading Order Bogoliubov Many-Body Perturbation Theory (2) solution review:")
        @printf(Summary, "\nE_0^(2)     = %15.8f \t MeV \t\t ... \t LO HF-BCS-BMBPT(2) correction to ground-state energy", dE) 
        @printf(Summary, "\nE_0^(2) / A = %15.8f \t MeV / Nucleon \t\t ... \t LO HF-BCS-BMBPT(2) correction to ground-state energy", dE / Float64(Params.Calc.A)) 
    close(Summary)

    return
end