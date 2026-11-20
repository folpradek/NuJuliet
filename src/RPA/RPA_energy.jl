#=
    RPA ground-state correlation energy ... two independent evaluations:

    1) Quasi-boson (QBA) energy ... RPA_energy_QBA()
           E_QBA = -sum_(J,P) (2J + 1) sum_nu w_nu Y_nu^+ N Y_nu = <H_QBA> = sum_nu [ Y_nu^+ A Y_nu + X_nu^+ B Y_nu ] ,     N = diag(D) ...
       If A & B contain the 2-body correlation function terms dA, dB (RPA_allocate_TBCF with Sigma_AB), their energy stays in E_QBA ...

    2) Density energy ... RPA_energy_rho() & RPA_energy_dress()
           E_dens = E_sigma + E_rho + E_dress ,     E_sigma = 1/4 sum V_NN Sigma ,     E_rho = sum_ab (2j_a + 1) h_N_ab (Rho - Rho_ref)_ba ,
           E_dress = sum_(J,P) (2J + 1) sum_nu [ Y_nu^+ dA Y_nu + X_nu^+ dB Y_nu ] ... energy of the Sigma terms of A & B,
       with the OBDM Rho of the RPA calculation (outer loop) & the reference OBDM Rho_ref, both in the current basis ...
       The Sigma_AB entering A & B is not contained in the recomputed Sigma (from X, Y), so its energy E_dress is added explicitly ...
       (E_sigma + E_rho -> MP2 at 2nd order only for a Pauli-consistent Rho, e.g. RPA_OBDM_sigma ... RPA_OBDM_oY2/oY4 give MP2(pn) only)

    Density = false (default TBCF = false) ... no Sigma, only the QBA energy is evaluated (E_dens = NaN) ... HF-ScRPA always passes Density = true ...

    All RPA solutions with a non-zero norm & a real energy enter ... (the zero-norm spurious solution has X = Y = 0, see RPA_diagonalize)
    Solutions with complex energies (instability) are excluded from the QBA & dressing energies & reported ...
=#
function RPA_energy(Params::Parameters,Orb::Vector{Orb1B},Orb_1p1h::Orb2B_1p1h,Orb_NN::Orb2B,E::Matrix{Vector{ComplexF64}},X::Matrix{Matrix{ComplexF64}},Y::Matrix{Matrix{ComplexF64}},
                    Rho::O1B,Rho_ref::O1B,h_N::O1B,Sigma::O2B,V_NN::O2B;Sigma_AB::Union{O2B,Nothing} = nothing,Density::Bool = Params.Calc.RPA.TBCF)
    # Sigma_AB ... 2-body correlation function used in the RPA matrices A & B (nothing ... A & B without Sigma terms)
    # Density  ... evaluate the density energy (needs Sigma; HF-ScRPA computes Sigma always, also for TBCF = false)
    println("\nCalculating RPA correlation energy ...")

    # Quasi-boson energy ...
    E_corr_QBA = RPA_energy_QBA(Params,Orb,Orb_1p1h,E,X,Y)

    # Density energy ... only with the 2-body correlation function ...
    E_corr_sigma, E_corr_rho, E_corr_dress = NaN64, NaN64, NaN64
    
    if Density == true
        E_corr_sigma, E_corr_rho = RPA_energy_rho(Params,Orb,Orb_NN,Rho,Rho_ref,h_N,Sigma,V_NN)

        # Energy of the 2-body correlation function terms of A & B ... (RPA_allocate_TBCF added to zero matrices)
        E_corr_dress = 0.0
        if Sigma_AB !== nothing
            Zero = [zeros(Float64,Orb_1p1h.N[P,J+1],Orb_1p1h.N[P,J+1]) for P in 1:2, J in 0:(2*Params.Calc.Nmax + 1)]
            @time dA, dB = RPA_allocate_TBCF(Params,Orb,Orb_1p1h,Orb_NN,V_NN,Sigma_AB,Zero,deepcopy(Zero))
            if Params.Calc.RPA.DEV
                @time dA = RPA_allocate_SE(Params,Orb,Orb_1p1h,Orb_NN,h_N,V_NN,dA, Mode = "OnShell")
            end
            E_corr_dress = RPA_energy_dress(Params,Orb,Orb_1p1h,E,X,Y,dA,dB)
        end
    end

    E_corr_dens = E_corr_sigma + E_corr_rho + E_corr_dress

    println("\tRPA correlation energy (Quasi-boson ... -sum w Y^+ N Y) ...                  E_QBA = " * string(round(E_corr_QBA,digits=6)) * " MeV")
    if Density == true
        println("\tRPA correlation energy (Correlated 2-body correlation function) ...          E_sig = " * string(round(E_corr_sigma,digits=6)) * " MeV")
        println("\tRPA correlation energy (Correlated 1-body density, h.drho) ...               E_rho = " * string(round(E_corr_rho,digits=6)) * " MeV")
        if Sigma_AB !== nothing
            println("\tRPA correlation energy (Sigma terms of A & B, Y^+ dA Y + X^+ dB Y) ...       D     = " * string(round(E_corr_dress,digits=6)) * " MeV")
        end
        println("\tRPA correlation energy (Density, E_sig + E_rho + D) ...                      E_RPA = " * string(round(E_corr_dens,digits=6)) * " MeV")
    end
    println()

    # All components of the correlation energy ... (NaN if not evaluated)
    E_corr = (E_QBA = E_corr_QBA, E_sigma = E_corr_sigma, E_rho = E_corr_rho, E_dress = E_corr_dress, E_dens = E_corr_dens)

    return E_corr
end

#=
    Quasi-boson energy ...
        E_QBA = -sum_(J,P) (2J + 1) sum_nu w_nu Y_nu^+ N Y_nu ,      N = diag(D), D_ph = n_h - n_p ...
    Note: at 2nd order E_QBA -> 2 MP2(pp + nn) + MP2(pn) ... the like-particle 2p-2h configurations are counted twice ...
=#
function RPA_energy_QBA(Params::Parameters,Orb::Vector{Orb1B},Orb_1p1h::Orb2B_1p1h,E::Matrix{Vector{ComplexF64}},X::Matrix{Matrix{ComplexF64}},Y::Matrix{Matrix{ComplexF64}})
    # Read parameters ...
    N_max = Params.Calc.Nmax
    J_max = 2*N_max + 1

    # Initialize the list of values of J & P for iteration ...
    JP_list = JP_initialize(J_max)
    N_JP = length(JP_list)

    # Results of individual J & P blocks ...
    E_QBA_JP = zeros(Float64,N_JP)
    N_drop_JP = zeros(Int64,N_JP)

    @inbounds Threads.@threads for i_JP in 1:N_JP
        J, P = JP_list[i_JP][1], JP_list[i_JP][2]

        N_ph = Orb_1p1h.N[P,J+1]
        if N_ph == 0
            continue
        end

        Orb_1p1h_JP = Orb_1p1h.Ind[P,J+1]
        J_hat = Float64(2*J + 1)

        E_JP, X_JP, Y_JP = E[P,J+1], X[P,J+1], Y[P,J+1]

        # Occupation differences D_ph = n_h - n_p ... (RPA metric N)
        D_JP = zeros(Float64,N_ph)
        @inbounds for i_ph in 1:N_ph
            a_p, a_h, T_ph = Orb_1p1h_JP[i_ph].a, Orb_1p1h_JP[i_ph].b, Orb_1p1h_JP[i_ph].T
            if T_ph == -1
                D_JP[i_ph] = Orb[a_h].pO - Orb[a_p].pO
            else
                D_JP[i_ph] = Orb[a_h].nO - Orb[a_p].nO
            end
        end

        SumE = 0.0
        @inbounds for i_nu in 1:N_ph
            X_nu, Y_nu = view(X_JP,:,i_nu), view(Y_JP,:,i_nu)

            # Physical solutions only ... non-zero norm & real energy ...
            RPA_norm = 0.0
            @inbounds for i_ph in 1:N_ph
                RPA_norm += D_JP[i_ph] * (abs2(X_nu[i_ph]) - abs2(Y_nu[i_ph]))
            end

            if abs(RPA_norm) < 1e-8
                continue
            end

            if abs(imag(E_JP[i_nu])) > 1e-8
                N_drop_JP[i_JP] += 1
                continue
            end

            # Quasi-boson energy ... -w_nu Y_nu^+ N Y_nu ...
            @inbounds for i_ph in 1:N_ph
                SumE -= J_hat * real(E_JP[i_nu]) * D_JP[i_ph] * abs2(Y_nu[i_ph])
            end
        end

        E_QBA_JP[i_JP] = SumE
    end

    if sum(N_drop_JP) > 0
        println("\tWarning! " * string(sum(N_drop_JP)) * " RPA solutions with complex energies (instability) are excluded from the quasi-boson energy ...")
        println("\t         (they are still contained in the 2-body correlation function Sigma ...)")
    end

    return sum(E_QBA_JP)
end

#=
    Energy of the 2-body correlation function terms dA, dB of A & B ... (part of the density energy, see RPA_energy)
        E_dress = sum_(J,P) (2J + 1) sum_nu [ Y_nu^+ dA Y_nu + X_nu^+ dB Y_nu ] ...
    Their contribution enters the RPA equations, but is not contained in the 2-body correlation function recomputed from X, Y ...
    Same physical solutions as in RPA_energy_QBA (non-zero norm & real energy) ...
=#
function RPA_energy_dress(Params::Parameters,Orb::Vector{Orb1B},Orb_1p1h::Orb2B_1p1h,E::Matrix{Vector{ComplexF64}},X::Matrix{Matrix{ComplexF64}},Y::Matrix{Matrix{ComplexF64}},
                          dA::Matrix{Matrix{Float64}},dB::Matrix{Matrix{Float64}})
    # Read parameters ...
    N_max = Params.Calc.Nmax
    J_max = 2*N_max + 1

    # Initialize the list of values of J & P for iteration ...
    JP_list = JP_initialize(J_max)
    N_JP = length(JP_list)

    # Results of individual J & P blocks ...
    E_dress_JP = zeros(Float64,N_JP)

    @inbounds Threads.@threads for i_JP in 1:N_JP
        J, P = JP_list[i_JP][1], JP_list[i_JP][2]

        N_ph = Orb_1p1h.N[P,J+1]
        if N_ph == 0
            continue
        end

        Orb_1p1h_JP = Orb_1p1h.Ind[P,J+1]
        J_hat = Float64(2*J + 1)

        E_JP, X_JP, Y_JP = E[P,J+1], X[P,J+1], Y[P,J+1]
        dA_JP, dB_JP = dA[P,J+1], dB[P,J+1]

        # Occupation differences D_ph = n_h - n_p ... (RPA metric N)
        D_JP = zeros(Float64,N_ph)
        @inbounds for i_ph in 1:N_ph
            a_p, a_h, T_ph = Orb_1p1h_JP[i_ph].a, Orb_1p1h_JP[i_ph].b, Orb_1p1h_JP[i_ph].T
            if T_ph == -1
                D_JP[i_ph] = Orb[a_h].pO - Orb[a_p].pO
            else
                D_JP[i_ph] = Orb[a_h].nO - Orb[a_p].nO
            end
        end

        SumD = 0.0
        @inbounds for i_nu in 1:N_ph
            X_nu, Y_nu = view(X_JP,:,i_nu), view(Y_JP,:,i_nu)

            # Physical solutions only ... non-zero norm & real energy ...
            RPA_norm = 0.0
            @inbounds for i_ph in 1:N_ph
                RPA_norm += D_JP[i_ph] * (abs2(X_nu[i_ph]) - abs2(Y_nu[i_ph]))
            end

            if abs(RPA_norm) < 1e-8
                continue
            end

            if abs(imag(E_JP[i_nu])) > 1e-8
                continue
            end

            # Energy of the Sigma terms of A & B ... Y_nu^+ dA Y_nu + X_nu^+ dB Y_nu ...
            SumD += J_hat * real(dot(Y_nu,dA_JP * Y_nu) + dot(X_nu,dB_JP * Y_nu))
        end

        E_dress_JP[i_JP] = SumD
    end

    return sum(E_dress_JP)
end

#=
    Density energy ... with the OBDM of the RPA calculation (outer loop) ...
        E_sigma = 1/4 sum V_NN Sigma          ... the full 2-body correlation function of RPA_sigma(),
        E_rho   = sum_ab (2j_a + 1) h_N_ab (Rho - Rho_ref)_ba ,
    Rho & Rho_ref in the current basis (full matrices, off-diagonal elements included) ...

    Note on the OBDM ... E_dens is correct at 2nd order (-> MP2) only with a Pauli-consistent OBDM (RPA_OBDM_sigma, built from the
    same Sigma_pphh). The quasi-boson OBDMs (RPA_OBDM_oY2, oY4) count the like-particle depletion twice, with them E_dens -> MP2(pn) ...
=#
function RPA_energy_rho(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,Rho::O1B,Rho_ref::O1B,h_N::O1B,Sigma::O2B,V_NN::O2B)
    # Calculate RPA correlation energy from the 2-body correlation function ...
    E_sigma = V2b_correlation_energy(Params,Orb,Orb_NN,Sigma,V_NN)

    # 1-body correlation energy E_rho = sum_ab (2j_a + 1) h_N_ab (Rho - Rho_ref)_ba ... over orbitals with equal l & j ...
    E_rho = 0.0
    @inbounds for Orb_a in Orb
        a, l_a, j_a = Orb_a.a, Orb_a.l, Orb_a.j
        @inbounds for Orb_b in Orb
            b = Orb_b.a
            if Orb_b.l != l_a || Orb_b.j != j_a
                continue
            end

            pdRho, ndRho = Rho.p[b,a] - Rho_ref.p[b,a], Rho.n[b,a] - Rho_ref.n[b,a]
            E_rho += hat(j_a) * (h_N.p[a,b] * pdRho + h_N.n[a,b] * ndRho)
        end
    end

    return E_sigma, E_rho
end
