function HFB_canonical_basis(Params::Parameters,Rho::O1B,Orb::Vector{Orb1B})
    # Read calculation parameters ...
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1) * (N_max + 2), 2)

    # Allocate density matrices ...
    pRho, nRho = deepcopy(Rho.p), deepcopy(Rho.n)

    # Symmetrize density matrices ...
    pRho .= 0.5 .* (pRho .+ pRho')
    nRho .= 0.5 .* (nRho .+ nRho')

    # Clean numerical noise & regularize the density matrix Rho ...
    @inbounds for a in 1:a_max
        l_a, j_a = Orb[a].l, Orb[a].j
        @inbounds for b in 1:a_max
            l_b, j_b = Orb[b].l, Orb[b].j
            if l_a != l_b || j_a != j_b
                pRho[a,b] = 0.0
                nRho[a,b] = 0.0
            end
        end
        R = 10.0 * Float64(l_a*(N_max+1) + div(j_a+1,2))
        pRho[a,a] += R
        nRho[a,a] += R
    end

    # Diagonalize 1-body HFB density matrix Rho ...
    _, pC = eigen(Symmetric(pRho),sortby=-)
    _, nC = eigen(Symmetric(nRho),sortby=-)

    # Clean numerical noise in C ...
    @inbounds Threads.@threads for a in 1:a_max
        @inbounds for b in 1:a_max
            pCME, nCME = abs(pC[a,b]), abs(nC[a,b])
            if pCME < 1e-13
                pC[a,b] = 0.0
            end
            if nCME < 1e-13
                nC[a,b] = 0.0
            end
        end
    end

    # Reorder the transformation matrix C ...
    C = HFB_canonical_basis_orbital_ordering(Params,O1B(pC,nC),Orb)

    return C
end

function HFB_canonical_basis_BCS(Params::Parameters,H::O1B,Orb::Vector{Orb1B})
    # Read calculation parameters ...
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1) * (N_max + 2), 2)

    # Allocate density matrices ...
    pH, nH = deepcopy(H.p), deepcopy(H.n)

    # Symmetrize density matrices ...
    pH .= 0.5 .* (pH .+ pH')
    nH .= 0.5 .* (nH .+ nH')

    # Eliminate any possible numerical noise spoiling block-diagonal structure ...
    @inbounds Threads.@threads for a in 1:a_max
        @inbounds for b in 1:a_max
            if (Orb[a].j != Orb[b].j) || (Orb[a].l != Orb[b].l)
                pH[a,b] = 0.0
                nH[a,b] = 0.0
            end
        end
    end

    # Diagonalize 1-body HFB density matrix Rho ...
    _, pC = eigen(Symmetric(pH),sortby=+)
    _, nC = eigen(Symmetric(nH),sortby=+)

    # Clean numerical noise in C ...
    @inbounds Threads.@threads for a in 1:a_max
        @inbounds for b in 1:a_max
            pCME, nCME = abs(pC[a,b]), abs(nC[a,b])
            if pCME < 1e-13
                pC[a,b] = 0.0
            end
            if nCME < 1e-13
                nC[a,b] = 0.0
            end
        end
    end

    # Reorder the transformation matrix C ...
    C = HFB_canonical_basis_orbital_ordering(Params,O1B(pC,nC),Orb)

    return C
end

function HFB_canonical_basis_diagonalize(Params::Parameters,Rho::O1B,Orb::Vector{Orb1B})
    # Read calculation parameters ...
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1) * (N_max + 2), 2)

    # The number of l & j combinations ...
    lj_max = (N_max+1)^2

    # Read the density matrix Rho ...
    pRho, nRho = Rho.p ,Rho.n

    # Initialize the transformation matrix C & occupation number Occ ...
    pC, pOcc = zeros(Float64,a_max,a_max), zeros(Float64,a_max)
    nC, nOcc = zeros(Float64,a_max,a_max), zeros(Float64,a_max)

    # Prepare the l & j blocks ...
    lj_a, lj_a_count = Vector{Vector{Int64}}(undef,lj_max), zeros(Int64,lj_max)
        # Note l & j numbers are evaluated in the direction as follows ...
        # for l in 0:N_max ... for j in 1:2:(2*N_max+1) ... hence the
        # analytic formula ... lj_ind = l_a*(N_max+1) + div(j_a+1,2)

    # Count the number of orbitals in each l & j block ...
    @inbounds for a in 1:a_max
        l_a, j_a = Orb[a].l, Orb[a].j
        lj_ind = l_a*(N_max+1) + div(j_a+1,2)
        lj_a_count[lj_ind] += 1
    end

    # Initialite the mapping lists for orbitals in each l & j block ...
    @inbounds for lj_ind in 1:lj_max
        lj_a[lj_ind] = Vector{Int64}(undef,lj_a_count[lj_ind])
    end

    # Reset the counter for orbitals in each l & j block ...
    lj_a_count .= 0

    # Allocate the mapping lists for orbitals in each l & j block ...
    @inbounds for a in 1:a_max
        l_a, j_a = Orb[a].l, Orb[a].j
        lj_ind = l_a*(N_max+1) + div(j_a+1,2)
        lj_a_count[lj_ind] += 1
        lj_a[lj_ind][lj_a_count[lj_ind]] = a
    end

    # Allocate & diagonalize l & j blocks of Rho ...
    @inbounds Threads.@threads for lj_ind in 1:lj_max
        a_lj_max = lj_a_count[lj_ind]

        if a_lj_max > 0
            # Initialite the density matrix Rho for given l & j ...
            pRho_lj = Matrix{Float64}(undef,a_lj_max,a_lj_max)
            nRho_lj = Matrix{Float64}(undef,a_lj_max,a_lj_max)

            # Allocate the density matrix Rho for given l & j ...
            @inbounds for a_lj in 1:a_lj_max
                a = lj_a[lj_ind][a_lj]
                @inbounds for b_lj in 1:a_lj_max
                    b = lj_a[lj_ind][b_lj]
                    pRho_lj[a_lj,b_lj] = pRho[a,b]
                    nRho_lj[a_lj,b_lj] = nRho[a,b]
                end
            end

            # Diagonalize the given l & j block of Rho ...
            pOcc_lj, pC_lj = eigen!(Symmetric(pRho_lj),sortby=-)
            nOcc_lj, nC_lj = eigen!(Symmetric(nRho_lj),sortby=-)

            # Allocate the transformation matrix C & occupation numbers Occ ...
            @inbounds for a_lj in 1:a_lj_max
                a = lj_a[lj_ind][a_lj]
                @inbounds for b_lj in 1:a_lj_max
                    b = lj_a[lj_ind][b_lj]
                    pC[b,a] = pC_lj[b_lj,a_lj]
                    nC[b,a] = nC_lj[b_lj,a_lj]
                end
                pOcc[a] = pOcc_lj[a_lj]
                nOcc[a] = nOcc_lj[a_lj]
            end

        end
    end

    return pOcc, pC, nOcc, nC
end