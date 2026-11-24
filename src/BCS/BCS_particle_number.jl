function BCS_particle_number(Params::Parameters,V::pnVector,Orb::Vector{Orb1B})
    # Read basis parameters ...
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1)*(N_max + 2),2)

    # Initialize the particle numbers Z & N ...
    Z, N = 0.0, 0.0

    # Evaluate <Z> & <N> from V amplitudes (Rho densities) ...
    @inbounds for a in 1:a_max
        z = Float64(Orb[a].j + 1) * V.p[a]^2
        Z += z
        n = Float64(Orb[a].j + 1) * V.n[a]^2
        N += n
    end

    return Z, N
end

function BCS_particle_number_dispersion(Params::Parameters,U::pnVector,V::pnVector,Orb::Vector{Orb1B})
    # Read parameters ...
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1)*(N_max + 2),2)
    dZ, dN = 0.0, 0.0

    # Calculate the particle number dispersion ...
    println("\nCalculating the BCS dispersion of proton & neutron particle numbers ...")

    @inbounds for a in 1:a_max
        j_a = Orb[a].j
        ja_hat = Float64(j_a + 1)
        dZ += 2.0 * ja_hat * U.p[a]^2 * V.p[a]^2
        dN += 2.0 * ja_hat * U.n[a]^2 * V.n[a]^2
    end

    # Calculate square roots of dispersion numbers ...
    dZ, dN = sqrt(dZ), sqrt(dN)

    println("\nBCS dispersion of nucleon numbers are ...")
    println("\tdZ = " * string(round(dZ,digits=5)))
    println("\tdN = " * string(round(dN,digits=5)))

    return pnFloat(dZ,dN)
end