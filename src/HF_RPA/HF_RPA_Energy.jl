function HF_RPA_energy(Params::Parameters,N_nu::Matrix{Int64},E_RPA::Matrix{Vector{ComplexF64}},Y_RPA::Matrix{Matrix{ComplexF64}})
    # Read parameters ...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1
    Orthogon = Params.Calc.RPA.Ortho

    println("\nCalculating RPA correlation energy ...")

    E_Corr_RPA = 0.0
    @inbounds for J in 0:J_max
        @inbounds for P in 1:2
            N_ph = N_nu[J+1,P]
            if !(J == 1 && P == 2)
                @inbounds for nu in 1:N_ph
                    @inbounds for ph in 1:N_ph
                        ME = -real(E_RPA[J+1,P][nu]) * abs(Y_RPA[J+1,P][ph,nu])^2 * Float64(2*J+ 1)
                        E_Corr_RPA += ME
                    end
                end
            else
                if Orthogon == true
                    @inbounds for nu in 2:N_ph
                        @inbounds for ph in 1:N_ph
                            ME = -real(E_RPA[J+1,P][nu]) * abs(Y_RPA[J+1,P][ph,nu])^2 * Float64(2*J+ 1)
                            E_Corr_RPA += ME
                        end
                    end
                else
                    @inbounds for nu in 1:N_ph
                        @inbounds for ph in 1:N_ph
                            ME = -real(E_RPA[J+1,P][nu]) * abs(Y_RPA[J+1,P][ph,nu])^2 * Float64(2*J+ 1)
                            E_Corr_RPA += ME
                        end
                    end
                end

            end
        end
    end

    println("\nCorrelation RPA energy is ...  E_RPA = " * string(round(real(E_Corr_RPA),digits=6)) * " MeV\n")

    return E_Corr_RPA
end

# Evaluation of RPA correlation energy ... for testing purposes!!!
function HF_RPA_energy_density(Params::Parameters,N_nu::Matrix{Int64},N_Particle::pnInteger,Particle::pnSVector,N_Hole::pnInteger,Hole::pnSVector,Orb::Vector{Orb1B},Orb_NN::Orb2B,V_NN::O2B,E_RPA::Matrix{Vector{ComplexF64}},Y_RPA::Matrix{Matrix{ComplexF64}},Rho::O1B)
    # Read calculation parameters ...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1

    println("\nCalculating RPA correlation energy ... Using the alternative 2-body density based formula ...")

    # Make Particle-Hole orbitals list ...
    ParticleHole = orbitals_ph_list(J_max,N_Particle,Particle,N_Hole,Hole)

    E_0 = 0.0
    E_0_Fermi = 0.0

    # Proton ph iteration ...
    @inbounds for h in 1:N_Hole.p
        a_h = Hole.p[h].a
        l_h = Hole.p[h].l
        j_h = Hole.p[h].j
        E_h = Hole.p[h].E
        @inbounds for p in 1:N_Particle.p
            a_p = Particle.p[p].a
            l_p = Particle.p[p].l
            j_p = Particle.p[p].j
            E_p = Particle.p[p].E
            P = rem(l_p + l_h,2) + 1
            @inbounds for J in div(abs(j_p - j_h),2):div(j_p + j_h,2)
                N_ph = N_nu[J+1,P]
                ph = ParticleHole.p[J+1,P,p,h]
                Amp_J = 0.5 * Float64(2*J + 1)

                @inbounds for nu in 1:N_ph
                    Y_ph_nu = Y_RPA[J+1,P][ph,nu]
                    ME = - Amp_J * (real(E_RPA[J+1,P][nu]) + (E_p - E_h)) * abs(Y_ph_nu)^2
                    E_0 += ME

                end

                ME = Amp_J * Rho.p[a_p,a_p] * O2b_pp(a_p,a_h,a_p,a_h,J,P,V_NN,Orb,Orb_NN)
                E_0_Fermi += ME
            end
        end
    end

    # Neutron ph iteration ...
    @inbounds for h in 1:N_Hole.n
        a_h = Hole.n[h].a
        l_h = Hole.n[h].l
        j_h = Hole.n[h].j
        E_h = Hole.n[h].E
        @inbounds for p in 1:N_Particle.n
            a_p = Particle.n[p].a
            l_p = Particle.n[p].l
            j_p = Particle.n[p].j
            E_p = Particle.n[p].E
            P = rem(l_p + l_h,2) + 1
            @inbounds for J in div(abs(j_p - j_h),2):div(j_p + j_h,2)
                N_ph = N_nu[J+1,P]
                ph = ParticleHole.n[J+1,P,p,h]
                Amp_J = 0.5 * Float64(2*J + 1)

                @inbounds for nu in 1:N_ph
                    Y_ph_nu = Y_RPA[J+1,P][ph,nu]
                    ME = - Amp_J * (real(E_RPA[J+1,P][nu]) + (E_p - E_h)) * abs(Y_ph_nu)^2
                    E_0 += ME

                end

                ME = Amp_J * Rho.n[a_p,a_p] * O2b_nn(a_p,a_h,a_p,a_h,J,P,V_NN,Orb,Orb_NN)
                E_0_Fermi += ME

            end
        end
    end

    println("RPA ground-state correlation energy E_0^RPA is ...     E_0_corr = " * string(round(real(E_0 + E_0_Fermi),digits=6)) * "\tMeV")
    println("RPA ground-state correlation energy due to Fermi sea depletion is ...     E_0_corr_dep = " * string(round(real(E_0_Fermi),digits=6)) * "\tMeV")
    println("To get the total ground-state energy add the mean-field energy ...")
    
    return
end

function HF_RPA_energy_bosonic(Params::Parameters,N_nu::Matrix{Int64},E_RPA::Matrix{Vector{ComplexF64}},Y_RPA::Matrix{Matrix{ComplexF64}})
    # Read parameters ...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1
    Orthogon = Params.Calc.RPA.Ortho

    println("\nCalculating RPA correlation energy ... Using the bosonic algebraic structure ...")

    E_Corr_RPA = 0.0
    @inbounds for J in 0:J_max
        @inbounds for P in 1:2
            N_ph = N_nu[J+1,P]
            if !(J == 1 && P == 2)
                @inbounds for nu in 1:N_ph
                    @inbounds for ph in 1:N_ph
                        ME = -real(E_RPA[J+1,P][nu]) * abs(Y_RPA[J+1,P][ph,nu])^2 * Float64(2*J+ 1)
                        E_Corr_RPA += ME
                    end
                end
            else
                if Orthogon == true
                    @inbounds for nu in 2:N_ph
                        @inbounds for ph in 1:N_ph
                            ME = -real(E_RPA[J+1,P][nu]) * abs(Y_RPA[J+1,P][ph,nu])^2 * Float64(2*J+ 1)
                            E_Corr_RPA += ME
                        end
                    end
                else
                    @inbounds for nu in 1:N_ph
                        @inbounds for ph in 1:N_ph
                            ME = -real(E_RPA[J+1,P][nu]) * abs(Y_RPA[J+1,P][ph,nu])^2 * Float64(2*J+ 1)
                            E_Corr_RPA += ME
                        end
                    end
                end

            end
        end
    end

    println("RPA ground-state correlation energy E_0^RPA is ...     E_0_corr = " * string(round(real(E_Corr_RPA),digits=6)) * "\tMeV")
    println("To get the total ground-state energy add the mean-field energy ...")

    return
end