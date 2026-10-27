function HFB_density_operator_initialize(Params::Parameters,Orb::Vector{Orb1B})
    # Read parameters ...
    Z_target, N_target = Float64(Params.Calc.Z), Float64(Params.Calc.A - Params.Calc.Z)
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1) * (N_max + 2), 2)

    # Initialize the valence space shell number N ...
    pN_valence, nN_valence = 0, 0

    # Determine the valence space shell number N ...
        # Proton valence space shell ...
    @inbounds for N_a in 0:N_max
        N_tot = 0
        @inbounds for N_i in 0:N_a
            N_tot += (N_i + 1) * (N_i + 2)
        end
        if N_tot >= Z_target
            pN_valence = N_a
            break
        end
    end
        # Neutron valence space shell ...
    @inbounds for N_a in 0:N_max
        N_tot = 0
        @inbounds for N_i in 0:N_a
            N_tot += (N_i + 1) * (N_i + 2)
        end
        if N_tot >= N_target
            nN_valence = N_a
            break
        end
    end

    # Initialize density matrices ...
    pRho, nRho = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)
    pKappa, nKappa = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)


    # Read initial diffuseness a ...
    pa0, na0 = Params.Calc.HFB.pa0, Params.Calc.HFB.na0

    # Allocate density matrix Rho ...
    @inbounds for a in 1:a_max
        n_a, l_a = Orb[a].n, Orb[a].l
        N_a = 2 * n_a + l_a
        pRho[a,a] = exp(-(N_a - pN_valence) / pa0) / (1.0  + exp(-(N_a - pN_valence) / pa0))
        nRho[a,a] = exp(-(N_a - nN_valence) / na0) / (1.0  + exp(-(N_a - nN_valence) / na0))
    end

    # Calculate current particle numbers ...
    Z, N = HFB_particle_number(Params,O1B(pRho,nRho),Orb)

    # Perform adjustment of the density matrices to match target particle numbers ...
        # Proton density matrix Rho ...
    if Z > Z_target
        # Depletion of single-particle levels ...
        @inbounds for i in 1:500
            if (Z > Z_target)
                @inbounds for a in 1:a_max
                    n_a, l_a, j_a = Orb[a].n, Orb[a].l, Orb[a].j
                    N_a = 2 * n_a + l_a
                    if (pRho[a,a] - 0.001) >= 1e-7
                        dZ = 0.001 * Float64(j_a + 1)
                        if (Z - dZ) > Z_target
                            Z -= dZ
                            pRho[a,a] -= 0.001
                        end
                    end
                end
            else
                break
            end
        end

    elseif Z < Z_target
        # Increase occupation of a few shells above the Fermi sea ...
        @inbounds for i in 1:500
            if (Z < Z_target)
                @inbounds for a in 1:a_max
                    n_a, l_a, j_a = Orb[a].n, Orb[a].l, Orb[a].j
                    N_a = 2 * n_a + l_a
                    if (N_a >= pN_valence) && (N_a < (pN_valence + 3)) && ((pRho[a,a] + 0.001) < 1.0)
                        dZ = 0.001 * Float64(j_a + 1)
                        if (Z + dZ) < Z_target
                            Z += dZ
                            pRho[a,a] += 0.001
                        end
                    end
                end
            else
                break
            end
        end
    end
        # Neutron density matrix Rho ...
    if N > N_target
        # Depletion of single-particle levels ...
        @inbounds for i in 1:500
            if (N > N_target)
                @inbounds for a in 1:a_max
                    n_a, l_a, j_a = Orb[a].n, Orb[a].l, Orb[a].j
                    N_a = 2 * n_a + l_a
                    if (nRho[a,a] - 0.001) >= 1e-7
                        dN = 0.001 * Float64(j_a + 1)
                        if (N - dN) > N_target
                            N -= dN
                            nRho[a,a] -= 0.001
                        end
                    end
                end
            else
                break
            end
        end
        
    elseif N < N_target
        # Increase occupation of a few shells above the Fermi sea ...
        @inbounds for i in 1:500
            if (N < N_target)
                @inbounds for a in 1:a_max
                    n_a, l_a, j_a = Orb[a].n, Orb[a].l, Orb[a].j
                    N_a = 2 * n_a + l_a
                    if (N_a >= nN_valence) && (N_a < (nN_valence + 3)) && ((nRho[a,a] + 0.001) < 1.0)
                        dN = 0.001 * Float64(j_a + 1)
                        if (N + dN) < N_target
                            N += dN
                            nRho[a,a] += 0.001
                        end
                    end
                end
            else
                break
            end
        end
    end

    # Allocate the density matrix Kappa ... using the identity Kappa Kappa^dag = Rho - Rho^2 ...
    @inbounds for a in 1:a_max
        l_a = Orb[a].l
        pKappa[a,a] = Float64((-1)^(l_a)) * sqrt(pRho[a,a] - pRho[a,a]^2)
        nKappa[a,a] = Float64((-1)^(l_a)) * sqrt(nRho[a,a] - nRho[a,a]^2)
    end

    # Include small off-diagonal elements in Kappa ...
    @inbounds for a in 1:a_max
        l_a = Orb[a].l
        @inbounds for b in 1:a_max
            if Orb[a].j == Orb[b].j && Orb[a].l == Orb[b].l && a != b
                pKappa[a,b] += Float64((-1)^(l_a)) * sqrt(pKappa[a,a] * pKappa[b,b])
                nKappa[a,b] += Float64((-1)^(l_a)) * sqrt(nKappa[a,a] * nKappa[b,b])
            end
        end
    end

    return O1B(pRho,nRho), O1B(pKappa,nKappa)
end

function HFB_density_operator(Params::Parameters,U::O1B,V::O1B,Orb::Vector{Orb1B})
    # Read parameters ...
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1) * (N_max + 2), 2)

    # Preallocate matrices for Rho & Kappa density operators ...
    pRho, nRho = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)
    pKappa, nKappa = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)

    # Allocate density operators Rho & Kappa ...
    @inbounds for a in 1:a_max
        @inbounds for b in 1:a_max
            if (Orb[a].j == Orb[b].j) && (Orb[a].l == Orb[b].l)
                pRhoSum, pKappaSum = 0.0, 0.0
                nRhoSum, nKappaSum = 0.0, 0.0
                @inbounds for c in 1:a_max
                    if (Orb[a].j == Orb[c].j) && (Orb[a].l == Orb[c].l)
                        pMERho, pMEKappa = V.p[a,c] * V.p[b,c], V.p[a,c] * U.p[b,c]
                        nMERho, nMEKappa = V.n[a,c] * V.n[b,c], V.n[a,c] * U.n[b,c]
                        pRhoSum, pKappaSum = pRhoSum + pMERho, pKappaSum + pMEKappa
                        nRhoSum, nKappaSum = nRhoSum + nMERho, nKappaSum + nMEKappa
                    end
                end
                pRho[a,b], pKappa[a,b] = pRhoSum, pKappaSum
                nRho[a,b], nKappa[a,b] = nRhoSum, nKappaSum
            end
        end
    end

    return O1B(pRho,nRho), O1B(pKappa,nKappa)
end