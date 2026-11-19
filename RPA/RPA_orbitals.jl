struct Orb1B_RPA_OBDM
    p::pnOrbVector
    h::pnOrbVector
    N_p::pnInteger
    N_h::pnInteger
end

function RPA_OBDM_orbitals_make(Params::Parameters,Orb::Vector{Orb1B})
    # Read parameters ...
    Tol_par = Params.Calc.RPA.Tol_par
    Tol_hol = Params.Calc.RPA.Tol_hol

    # Initialize particle number counters ...
    pN_particle, pN_hole = 0, 0
    nN_particle, nN_hole = 0, 0

    # Determine the number of particle & hole orbitals ...
    @inbounds for Orb_a in Orb
        if abs(Orb_a.pO) > Tol_hol
            pN_hole += 1
        end
        if abs(Orb_a.pO) < Tol_par
            pN_particle += 1
        end
        if abs(Orb_a.nO) > Tol_hol
            nN_hole += 1
        end
        if abs(Orb_a.nO) < Tol_par
            nN_particle += 1
        end
    end

    # Initialize the particle & hole orbitals ...
    pOrb_particle, pOrb_hole = Vector{Orb1B}(undef,pN_particle), Vector{Orb1B}(undef,pN_hole)
    nOrb_particle, nOrb_hole = Vector{Orb1B}(undef,nN_particle), Vector{Orb1B}(undef,nN_hole)

    # Reset particle number counters ...
    pN_particle, pN_hole = 0, 0
    nN_particle, nN_hole = 0, 0

    # Allocate the particle & hole orbitals ...
    @inbounds for Orb_a in Orb
        if abs(Orb_a.pO) < Tol_par
            pN_particle += 1
            pOrb_particle[pN_particle] = Orb_a
        end
        if abs(Orb_a.pO) > Tol_hol
            pN_hole += 1
            pOrb_hole[pN_hole] = Orb_a
        end
        if abs(Orb_a.nO) < Tol_par
            nN_particle += 1
            nOrb_particle[nN_particle] = Orb_a
        end
        if abs(Orb_a.nO) > Tol_hol
            nN_hole += 1
            nOrb_hole[nN_hole] = Orb_a
        end
    end

    # Allocate the 1-body partice & hole orbitals structure ...
    Orb_RPA = Orb1B_RPA_OBDM(pnOrbVector(pOrb_particle,nOrb_particle),
                             pnOrbVector(pOrb_hole,nOrb_hole),
                             pnInteger(pN_particle,nN_particle),
                             pnInteger(pN_hole,nN_hole)
                            )

    return Orb_RPA
end