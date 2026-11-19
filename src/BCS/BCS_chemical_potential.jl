function BCS_chemical_potential(Params::Parameters,SPE::pnVector,Delta::pnVector,Lambda::pnFloat,dA::pnFloat,Orb::Vector{Orb1B})
    # Read parameters ...
    Z_target, N_target = Params.Calc.Z, Params.Calc.A - Params.Calc.Z
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1) * (N_max + 2), 2)
    pLambda, nLambda = Lambda.p, Lambda.n
    epsilon = Params.Calc.BCS.Tol

    # This inner lambda solver could be improved on ... using the Newton's method (???)

    # Solve for proton chemical potential ... with fixed Delta ...
    if abs(dA.p) > epsilon
        dZ = dA.p
        @inbounds for i in 1:500
            pLambda = pLambda + Params.Calc.BCS.q * dZ
            Z = 0.0
            @inbounds for a in 1:a_max
                j_a_hat = Float64(Orb[a].j + 1)
                V_a = sqrt(0.5 * (1.0 - (SPE.p[a] - pLambda) / sqrt((SPE.p[a] - pLambda)^2 + Delta.p[a]^2)))
                Z += j_a_hat * V_a^2
            end

            dZ = Z_target - Z

            if abs(dZ) < epsilon
                break
            end
        end
    end

    # Solve for neutron chemical potential ... with fixed Delta ...
    if abs(dA.n) > epsilon
        dN = dA.n
        @inbounds for i in 1:500
            nLambda = nLambda + Params.Calc.BCS.q * dN
            N = 0.0
            @inbounds for a in 1:a_max
                j_a_hat = Float64(Orb[a].j + 1)
                V_a = sqrt(0.5 * (1.0 - (SPE.n[a] - nLambda) / sqrt((SPE.n[a] - nLambda)^2 + Delta.n[a]^2)))
                N += j_a_hat * V_a^2
            end

            dN = N_target - N

            if abs(dN) < epsilon
                break
            end
        end
    end

    return pnFloat(pLambda,nLambda)
end

function HF_BCS_initialize_chemical_potential(Params::Parameters,SPE::pnVector,Orb::Vector{Orb1B})
    # Read parameters ...
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1) * (N_max + 2), 2)

    # Initialize Fermi level particle & hole SPEs ...
    pEps_particle, pEps_hole = 1000.0, -1000.0
    nEps_particle, nEps_hole = 1000.0, -1000.0

    # Find particle & hole SPEs near the Fermi level ...
    @inbounds for a in 1:a_max
        pE_a, nE_a = SPE.p[a], SPE.n[a]

        if Orb[a].pO == 1
            if pEps_hole < pE_a
                pEps_hole = pE_a
            end
        elseif Orb[a].pO == 0
            if pEps_particle > pE_a
                pEps_particle = pE_a
            end
        end

        if Orb[a].nO == 1
            if nEps_hole < nE_a
                nEps_hole = nE_a
            end
        elseif Orb[a].nO == 0
            if nEps_particle > nE_a
                nEps_particle = nE_a
            end
        end

    end

    # Determine the chemical potential
    Lambda = pnFloat(0.5 * (pEps_hole - pEps_particle), 0.5 * (nEps_hole - nEps_particle))

    return Lambda
end 