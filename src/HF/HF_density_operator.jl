function HF_density_operator(C::O1B,a_max::Int64,Orb::Vector{Orb1B})
    # Initialize density matrix Rho ...
    pRho, nRho = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)

    # Allocate density matrix Rho ...
    @inbounds for a in 1:a_max
        pOcc, nOcc = Orb[a].pO, Orb[a].nO

        pPhi, nPhi = @views C.p[:,a], @views C.n[:,a]

        pRho_a = pOcc .* (pPhi * pPhi')
        nRho_a = nOcc .* (nPhi * nPhi')

        pRho .+= pRho_a
        nRho .+= nRho_a
    end
    
    return O1B(pRho,nRho)
end