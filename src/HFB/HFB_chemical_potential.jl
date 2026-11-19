function HFB_chemical_potential_secant(Params::Parameters,Lambda_1::pnFloat,dA_1::pnFloat,Lambda_2::pnFloat,dA_2::pnFloat,Lambda_lo::pnFloat,Lambda_hi::pnFloat)
    # Read parameters ...
    Tol = Params.Calc.HFB.Tol
    dLambda_max = Params.Calc.HFB.dLmax

    # Evaluate safeguarded secant iteration ... protons & neutrons are independent ...
        # Proton secant ...
    pLambda, pLambda_lo, pLambda_hi = HFB_chemical_potential_secant_step(Lambda_1.p,dA_1.p,Lambda_2.p,dA_2.p,Lambda_lo.p,Lambda_hi.p,dLambda_max,Tol)
        # Neutron secant ...
    nLambda, nLambda_lo, nLambda_hi = HFB_chemical_potential_secant_step(Lambda_1.n,dA_1.n,Lambda_2.n,dA_2.n,Lambda_lo.n,Lambda_hi.n,dLambda_max,Tol)

    return pnFloat(pLambda,nLambda), pnFloat(pLambda_lo,nLambda_lo), pnFloat(pLambda_hi,nLambda_hi)
end

function HFB_chemical_potential_secant_step(Lambda_1::Float64,dA_1::Float64,Lambda_2::Float64,dA_2::Float64,Lambda_lo::Float64,Lambda_hi::Float64,dLambda_max::Float64,Tol::Float64)
    # Safeguarded secant step for one kind of nucleons ...
    #   dA = A_target - A(Lambda) ... A(Lambda) is increasing in Lambda for fixed fields H & Delta,
    #   (Lambda_2,dA_2) is the latest point & Lambda_lo < Lambda_root < Lambda_hi is the bracket (NaN ... not known yet) ...

    # Particle number already converged ... keep Lambda ...
    if abs(dA_2) < Tol
        return Lambda_2, Lambda_lo, Lambda_hi
    end

    # Update the bracket with the latest point ...
    if dA_2 > 0.0
        # Too few particles ... root lies above Lambda_2 ...
        Lambda_lo = Lambda_2
    else
        # Too many particles ... root lies below Lambda_2 ...
        Lambda_hi = Lambda_2
    end

    # Secant step ...
    Denom = dA_2 - dA_1
    dLambda = 0.0
    if Denom != 0.0 && Lambda_1 != Lambda_2 && isfinite(Denom)
        dLambda = - dA_2 * (Lambda_2 - Lambda_1) / Denom
    end

    # The step must go towards the root ... otherwise (flat or stale secant) take the maximal step ...
    if !(dLambda * dA_2 > 0.0)
        dLambda = sign(dA_2) * dLambda_max
    end

    # Check if the step is not too large ...
    if abs(dLambda) > dLambda_max
        dLambda = sign(dLambda) * dLambda_max
    end

    # Update Lambda ...
    Lambda = Lambda_2 + dLambda

    # If the root is bracketed & the secant leaves the bracket ... use bisection instead ...
    if !isnan(Lambda_lo) && !isnan(Lambda_hi)
        if Lambda <= Lambda_lo || Lambda >= Lambda_hi
            Lambda = 0.5 * (Lambda_lo + Lambda_hi)
        end
    end

    return Lambda, Lambda_lo, Lambda_hi
end

function HFB_chemical_potential_gap(Params::Parameters,H::O1B,Lambda_H::pnFloat,Delta::O1B,Orb::Vector{Orb1B})
    # Chemical potential in the middle of the shell gap for closed-shell systems ...
    #   Closed shell ... pairing field Delta has collapsed & the particle number fills whole (l,j) levels of h = H + Lambda_H ...
    #   Otherwise NaN is returned & Lambda is determined by the secant method ...
    Z_target, N_target = Float64(Params.Calc.Z), Float64(Params.Calc.A - Params.Calc.Z)
    Tol = Params.Calc.HFB.Tol

    # Proton chemical potential ...
    pLambda = NaN
    if maximum(abs.(Delta.p)) < Tol
        pLambda = HFB_chemical_potential_gap_species(H.p,Lambda_H.p,Z_target,Orb)
    end

    # Neutron chemical potential ...
    nLambda = NaN
    if maximum(abs.(Delta.n)) < Tol
        nLambda = HFB_chemical_potential_gap_species(H.n,Lambda_H.n,N_target,Orb)
    end

    return pnFloat(pLambda,nLambda)
end

function HFB_chemical_potential_gap_species(H::Matrix{Float64},Lambda_H::Float64,A_target::Float64,Orb::Vector{Orb1B})
    # Collect single-particle energies & degeneracies (j + 1) ... h is block-diagonal in (l,j) ...
    SPE, Deg = Float64[], Float64[]
    @inbounds for a in eachindex(Orb)
        l_a, j_a = Orb[a].l, Orb[a].j

        # Only the first orbital of each (l,j) block starts the block ...
        First = true
        @inbounds for b in 1:(a-1)
            if Orb[b].l == l_a && Orb[b].j == j_a
                First = false
                break
            end
        end
        if First == false
            continue
        end

        # Diagonalize the (l,j) block of h ...
        Block = [b for b in eachindex(Orb) if Orb[b].l == l_a && Orb[b].j == j_a]
        @inbounds for E in eigvals(Symmetric(H[Block,Block]))
            push!(SPE, E + Lambda_H)
            push!(Deg, Float64(j_a + 1))
        end
    end

    # Fill the levels from the bottom ...
    Order = sortperm(SPE)
    A_filled = 0.0
    @inbounds for i in 1:(length(Order)-1)
        A_filled += Deg[Order[i]]

        # Target particle number is reached exactly ... the gap lies between levels i & i + 1 ...
        if abs(A_filled - A_target) < 1e-8
            return 0.5 * (SPE[Order[i]] + SPE[Order[i+1]])
        elseif A_filled > A_target
            break
        end
    end

    # Not a closed shell ...
    return NaN
end
