function HF_density_operator(C::O1B,Orb::Vector{Orb1B})
    # Determine the number of single-particle orbitals ...
    a_max = length(Orb)

    # Initialize density matrix Rho ...
    pRho, nRho = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)

    # Allocate density matrix Rho ...
    @inbounds for Orb_a in Orb
        a, pO, nO = Orb_a.a, Orb_a.pO, Orb_a.nO

        pPhi, nPhi = @views C.p[:,a], @views C.n[:,a]

        pRho_a = pO .* (pPhi * pPhi')
        nRho_a = nO .* (nPhi * nPhi')

        pRho .+= pRho_a
        nRho .+= nRho_a
    end
    
    return O1B(pRho,nRho)
end