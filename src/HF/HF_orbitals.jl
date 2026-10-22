function HF_orbitals_make(Params::Parameters,SPE::pnVector,C::O1B,Orb::Vector{Orb1B})
    # Read parameters ...
    A, Z = Float64(Params.Calc.A), Float64(Params.Calc.Z)
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1) * (N_max + 2), 2)

    # Read HF orbitals & energies ...
    pC, nC =  C.p, C.n
    pSPE, nSPE = SPE.p, SPE.n

    # Initialize occupation numbers & masks vectors ...
    pOcc, pOcc_mask = zeros(Float64,a_max), falses(a_max)
    nOcc, nOcc_mask = zeros(Float64,a_max), falses(a_max)

    # Initialize arrays for ordering & masks ...
    pOrb_order, nOrb_order = Vector{Int}(undef,a_max), Vector{Int}(undef,a_max)
    pOrb_mask, nOrb_mask = falses(a_max), falses(a_max)

    # Make temporary HF densities ...
    pRho, nRho = @views [pC[:,a] .* pC[:,a]' for a in 1:a_max], @views [nC[:,a] .* nC[:,a]' for a in 1:a_max]

    # Allocate arrays with corresponding values of l & j numbers ...
    lp_values = [sum(pRho[a][i,i] * Orb[i].l for i in 1:a_max) for a in 1:a_max]
    jp_values = [sum(pRho[a][i,i] * Orb[i].j for i in 1:a_max) for a in 1:a_max]
    ln_values = [sum(nRho[a][i,i] * Orb[i].l for i in 1:a_max) for a in 1:a_max]
    jn_values = [sum(nRho[a][i,i] * Orb[i].j for i in 1:a_max) for a in 1:a_max]

    # Find ordering & performing masking ...
    @inbounds for a in 1:a_max
        lp, jp = lp_values[a], jp_values[a]
        ln, jn = ln_values[a], jn_values[a]

        @inbounds for c in 1:a_max
            if !pOrb_mask[c] && abs(Orb[c].l - lp) < 1e-8 && abs(Orb[c].j - jp) < 1e-8
                pOrb_order[c] = a
                pOrb_mask[c] = true
                break
            end
        end

        @inbounds for c in 1:a_max
            if !nOrb_mask[c] && abs(Orb[c].l - ln) < 1e-8 && abs(Orb[c].j - jn) < 1e-8
                nOrb_order[c] = a
                nOrb_mask[c] = true
                break
            end
        end
    end

    # Reshuffle SPE & C ...
    pSPE .= pSPE[pOrb_order]
    nSPE .= nSPE[nOrb_order]
    pC .= @views pC[:,pOrb_order]
    nC .= @views nC[:,nOrb_order]

    # Determine new proton occupation numbers ...
    Z_occ = 0.0
    while abs(Z - Z_occ) > 1e-8
        a_last, E_last = 1, 1e5

        @inbounds for a in 1:a_max
            E = pSPE[a]

            if (E_last > E) && (pOcc_mask[a] == false)
                E_last = E
                a_last = a
            else
                continue
            end
        end

        j_hat = Float64(Orb[a_last].j + 1)

        pOcc_mask[a_last] = true

        Occ = min(1.0, (Z - Z_occ) / j_hat)

        pOcc[a_last] = Occ
        Z_occ += j_hat * Occ
    end

    # Determine new neutron occupation numbers ...
    N_occ = 0.0
    while abs(A - Z - N_occ) > 1e-8
        a_last, E_last = 1, 1e5

        @inbounds for a in 1:a_max
            E = nSPE[a]

            if (E_last > E) && (nOcc_mask[a] == false)
                E_last = E
                a_last = a
            else
                continue
            end
        end

        j_hat = Float64(Orb[a_last].j + 1)

        nOcc_mask[a_last] = true

        Occ = min(1.0, (A - Z - N_occ) / j_hat)

        nOcc[a_last] = Occ
        N_occ += j_hat * Occ
    end

    # Initialize new single-particle orbitals ...
    Orb_new = Vector{Orb1B}(undef,a_max)

    # Allocate new single-particle orbitals ...
    @inbounds for a in 1:a_max
        a, n, l, j = Orb[a].a, Orb[a].n, Orb[a].l, Orb[a].j
        pO, nO = pOcc[a], nOcc[a]
        Orb_new[a] = Orb1B(a,n,l,j,pO,nO)
    end

    return Orb_new, pnVector(pSPE,nSPE), O1B(pC,nC)
end