function HFB_particle_number(Params::Parameters,Rho::O1B,Orb::Vector{Orb1B})
    # Read parameters ...
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1) * (N_max + 2), 2)

    # Initialize particle numbers ...
    Z, N = 0,0, 0,0

    # Evaluate the angular-momentum weighted trace of density matrices ...
    @inbounds for a in 1:a_max
        Z += Rho.p[a,a] * Float64(Orb[a].j + 1)
        N += Rho.n[a,a] * Float64(Orb[a].j + 1)
    end

    return Z, N
end

function HFB_particle_number_dispersion(Params::Parameters,Rho::O1B,Orb::Vector{Orb1B})
    # Read parameters ...
    Tol = min(1e-6,Params.Calc.HFB.Tol)
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1)*(N_max + 2),2)
    dZ, dN = 0.0, 0.0

    # Calculate the particle number dispersion ...
    println("\nCalculating the HFB dispersion of proton & neutron particle numbers ...")

    @inbounds for a in 1:a_max
        j_a, l_a = Orb[a].j, Orb[a].l
        j_a_hat = Float64(j_a + 1)
        @inbounds for b in 1:a_max
            if (j_a == Orb[b].j) && (l_a == Orb[b].l)
                dZ -= 2.0 * j_a_hat * Rho.p[a,b] * Rho.p[b,a]
                dN -= 2.0 * j_a_hat * Rho.n[a,b] * Rho.n[b,a]
            end
        end
        dZ += 2.0 * j_a_hat * Rho.p[a,a]
        dN += 2.0 * j_a_hat * Rho.n[a,a]
    end

    # Check that dZ & dN are non-negative ...
        # Case of proton number dispersion ...
    if dZ < 0.0 
        if abs(dZ) < Tol
            dZ = 0.0
        end
    end
        # Case of neutron number dispersion ...
    if dN < 0.0 
        if abs(dN) < Tol
            dN = 0.0
        end
    end

    # Calculate square roots of dispersion numbers ...
    dZ, dN = sqrt(abs(dZ)), sqrt(abs(dN))

    println("\nHFB dispersion of nucleon numbers are ...")
    println("\tdZ = " * string(round(dZ,digits=5)))
    println("\tdN = " * string(round(dN,digits=5)))

    return pnFloat(dZ,dN)
end