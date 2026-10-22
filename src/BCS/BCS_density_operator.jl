function BCS_density_operator(Params::Parameters,U::pnVector,V::pnVector)
    # Read parameters ...
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1)*(N_max + 2),2)

    # Initialize density matrices ...
    pRho, pKappa = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)
    nRho, nKappa = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)

    # Compute the 1-body density matrix ... diagonal in canonical basis ...
    @inbounds for a in 1:a_max
        pRho[a,a], pKappa[a,a] = V.p[a]^2, V.p[a] * U.p[a]
        nRho[a,a], nKappa[a,a] = V.n[a]^2, V.n[a] * U.n[a]
    end

    return O1B(pRho,nRho), O1B(pKappa,nKappa)
end