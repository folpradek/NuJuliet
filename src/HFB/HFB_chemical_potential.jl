function HFB_chemical_potential_secant(Params::Parameters,Lambda_1::pnFloat,dA_1::pnFloat,Lambda_2::pnFloat,dA_2::pnFloat)
    # Basic constants ...
        # Denominator epsilon ...
    Denom_eps = 1e-7
        # Maximum allowed Lambda step ...
    dLambda_max = Params.Calc.HFB.dLmax
    
    # Initialize Lambda ...
    pLambda, dpLambda = Lambda_2.p, 0.0
    nLambda, dnLambda = Lambda_2.n, 0.0

    # Evaluate secant iteration ...
        # Proton secant ...
    pDenom = (dA_2.p - dA_1.p)
    if abs(pDenom) > Denom_eps
        dpLambda = (Lambda_1.p * dA_2.p - Lambda_2.p * dA_1.p) / pDenom - pLambda
    elseif isfinite(dA_2.p)
        dpLambda = 0.1 * dA_2.p
    else
        dpLambda = 0.05 * pLambda
    end
        # Neutron secant ...
    nDenom = (dA_2.n - dA_1.n)
    if abs(nDenom) > Denom_eps
        dnLambda = (Lambda_1.n * dA_2.n - Lambda_2.n * dA_1.n) / nDenom - nLambda
    elseif isfinite(dA_2.n)
        dnLambda = 0.1 * dA_2.n
    else
        dnLambda = 0.05 * nLambda
    end

    # Check if steps are not too large ...
        # Proton step ...
    if abs(dpLambda) > dLambda_max
        dpLambda = sign(dpLambda) * dLambda_max
    end
        # Neutron step ...
    if abs(dnLambda) > dLambda_max
        dnLambda = sign(dnLambda) * dLambda_max
    end

    # Update Lambda ...
    pLambda += dpLambda
    nLambda += dnLambda

    return pnFloat(pLambda,nLambda)
end