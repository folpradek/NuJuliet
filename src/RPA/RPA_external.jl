function RPA_import_binary(Params::Parameters)
    # Parameters / dimensions
    N_max = Params.Calc.Nmax
    J_max = 2*N_max + 1
    bin_path = "io/" * Params.Calc.Path * "/" * Params.Calc.RPA.Name * "/bin/"

    # Initialite the list of values of J & P for iteration ...
    JP_list = JP_initialize(J_max)

    # Import 1-body single-particle orbitals ...
    Orb = orbitals_import(Params,bin_path * "orb.bin")

    # Construct 2-body 1p-1h orbitals ...
    Orb_1p1h = orbitals_1p1h_make(Params,Orb;Tol_par = Params.Calc.RPA.Tol_par,Tol_hol = Params.Calc.RPA.Tol_hol)

    println("\nImporting RPA solutions from binary format ...")

    # Initialize amplitudes & energies ...
    X_RPA = Matrix{Matrix{ComplexF64}}(undef,2,J_max+1)
    Y_RPA = Matrix{Matrix{ComplexF64}}(undef,2,J_max+1)
    E_RPA = Matrix{Vector{ComplexF64}}(undef,2,J_max+1)

    # Initialite each J & P block ...
    @inbounds for JP in JP_list
        J, P = JP[1], JP[2]
        N_ph = Orb_1p1h.N[P,J+1]
        X_RPA[P,J+1] = Matrix{ComplexF64}(undef,N_ph,N_ph)
        Y_RPA[P,J+1] = Matrix{ComplexF64}(undef,N_ph,N_ph)
        E_RPA[P,J+1] = Vector{ComplexF64}(undef,N_ph)
    end

    # Read RPA amplitudes X ...
    import_path_X = bin_path * "RPA_X.bin"
    open(import_path_X, "r") do Import_File
        @inbounds for JP in JP_list
            J, P = JP[1], JP[2]
            N_ph = Orb_1p1h.N[P,J+1]
            N_ph == 0 && continue
            Buffer = Vector{ComplexF64}(undef,N_ph*N_ph)
            read!(Import_File,Buffer)
            X_RPA[P,J+1] .= transpose(reshape(Buffer,N_ph,N_ph))
        end
    end

    # Read RPA amplitudes Y ...
    import_path_Y = bin_path * "RPA_Y.bin"
    open(import_path_Y, "r") do Import_File
        @inbounds for JP in JP_list
            J, P = JP[1], JP[2]
            N_ph = Orb_1p1h.N[P,J+1]
            N_ph == 0 && continue
            Buffer = Vector{ComplexF64}(undef,N_ph*N_ph)
            read!(Import_File,Buffer)
            Y_RPA[P,J+1] .= transpose(reshape(Buffer,N_ph,N_ph))
        end
    end

    # Read RPA energies E ...
    import_path_E = bin_path * "RPA_E.bin"
    open(import_path_E, "r") do Import_File
        @inbounds for JP in JP_list
            J, P = JP[1], JP[2]
            N_ph = Orb_1p1h.N[P,J+1]
            N_ph == 0 && continue
            Buffer = Vector{ComplexF64}(undef,N_ph)
            read!(Import_File,Buffer)
            E_RPA[P,J+1] .= Buffer
        end
    end

    println("\tRPA solution binaries have been succesufully imported ...")

    return X_RPA, Y_RPA, E_RPA
end

# Comment on the transition densities used by the external RPA routines below ...
    # All routines (radial transition densities, transition operator densities, the analysis of the phonon levels & the transition
    # currents) use the same reduced transition density matrix of the RPA state nu as RPA_rM (RPA_transitions.jl),
    #
    #     Gamma^t_ab(nu) = sum_ph (D_X[(a,b,t),ph] * X[ph,nu] + D_Y[(a,b,t),ph] * Y[ph,nu]) ,      <nu||M||0> = sum_(a,b,t) M^t_ab * Gamma^t_ab(nu) ,
    #
    # defined on the orbital pairs (a,b,t) of the multipole (J,P) ... see RPA_transition_kernel. The kernels D_X & D_Y are the coefficients
    # M_X & M_Y of RPA_rM_contract written before the contraction with the operator, hence
    #   ... Transitions = "Linear" ... <0|[Q_nu,M]|0> with the occupations of orb.bin (the 2-body correlation function does not enter) ,
    #   ... Transitions = "Exact"  ... <0|Q_nu M|0> with the RPA OBDM (rho.bin) & for TBCF = true the RPA 2-body correlation function (sigma.bin) ,
    # i.e. the external routines reproduce exactly the reduced transition matrix elements rM of RPA_solver / RPA_transitions_recalculate ...
    #
    # For any local 1-body operator F(r) (density, current, ...) the transition matrix element follows from the m-scheme transition density
    #     <nu J M|a^dagger_(a m_a) a_(b mu)|0> = (-1)^J * hat(J) * Gamma_ab(nu) * (j_b mu J M|j_a m_a) / hat(j_a)
    #                                         = (-1)^J * Gamma_ab(nu) * (-1)^(j_b - mu) (j_a m_a j_b -mu|J M) ,
    # i.e. F_nu(r) = (-1)^J * sum_(a,b,t) Gamma^t_ab(nu) * sum_(m_a mu) (-1)^(j_b - mu) (j_a m_a j_b -mu|J M) <a m_a|F(r)|b mu> ,
    # which for the 1p-1h pairs (p,h) & (h,p) reduces to the X (ph) & Y (hp) terms of the previous implementation ...
    #
    # Normalization of the exported densities & currents (common to both, the gauge g_nu makes pM_nu real & positive) ...
    #     density files ...  r^2 * Re(g_nu * (-1)^J * rho_nu^tr(r) / hat(J)) ,   rho_nu(r) = <nu J M|rho(r)|0> = rho_nu^tr(r) Y_JM(rhat) ,
    #     current files ...  Re(g_nu * (-1)^J * i * j_nu(r) / hat(J)) ,          j_nu(r) = <nu J M|j(r)|0> (convective & magnetization, units [c]) ,
    # so that the continuity equation div j_nu = -i (E_nu / hbar) rho_nu reads div J_map = (E_nu / hbar c) * (rho_map / r^2) * Y_JM between the files
    # (only the convective part, the magnetization current is divergence-free) ... the operator densities integrate to rM / (2J + 1) ...
    #
    # Note ... the bin files of the RPA directory have to be in the basis of the RPA amplitudes, i.e. RPA_Orb = false (see RPA_external_import) ...

function RPA_external_JP(JP::String)
    # Read the values of J & P ... JP = "J+" or "J-" ...
    P_char = string(JP[end])
    J = parse(Int64,JP[1:end-1])

    # Check input formating ...
    if (P_char != "+") && (P_char != "-")
        error("Invalid JP = \"$JP\"; use J+ or J- ...")
    end

    # Set the internal value of parity P ...
    P = (P_char == "-") ? 2 : 1

    return J, P
end

function RPA_external_import(Params::Parameters)
    # Import all RPA data needed by the external routines ... (as in RPA_transitions_recalculate)
    Transitions = Params.Calc.RPA.Transitions

    if (Transitions != "Exact") && (Transitions != "Linear")
        error("Invalid Params.Calc.RPA.Transitions = \"$Transitions\"; use \"Exact\" or \"Linear\".")
    end

    # For RPA_Orb = true the orbitals, basis & OBDM of the RPA directory are exported in the RPA natural-orbital basis (RPA_natural_orbitals_export),
    # which is not the basis of the RPA amplitudes X & Y (also the 1p-1h pairs would follow from the natural occupations) ...
    if Params.Calc.RPA.RPA_Orb == true
        error("RPA external routines: Params.Calc.RPA.RPA_Orb = true ... orb.bin, C.bin & rho.bin of the RPA directory are given in the RPA natural-orbital basis, " *
              "which does not match the basis of the RPA amplitudes X & Y ... use the RPA solution exported with RPA_Orb = false.")
    end

    # Set the path to the RPA input/output directory ...
    IO_RPA = "io/" * Params.Calc.Path * "/" * Params.Calc.RPA.Name

    # Import RPA single-particle orbitals ...
    Orb = orbitals_import(Params,IO_RPA * "/bin/orb.bin")

    # Construct 2-body 1p-1h orbitals ...
    Orb_1p1h = orbitals_1p1h_make(Params,Orb;Tol_par = Params.Calc.RPA.Tol_par,Tol_hol = Params.Calc.RPA.Tol_hol)

    # Import transformation matrix mapping LHO & reference RPA basis ...
    @time C = O1b_import(Params,Orb,IO_RPA * "/bin/C.bin")

    # Import RPA OBDM ... rho.bin is exported in the RPA (reference) basis (cf. RPA_orbitals_export), no transformation needed ...
    @time Rho = O1b_import(Params,Orb,IO_RPA * "/bin/rho.bin")

    # Import RPA 2-body correlation function ... only used by the exact transitions (as in RPA_solver & RPA_rM) ...
    Sigma, Orb_NN = O2b_initialize_empty(), nothing

    if Transitions == "Exact" && Params.Calc.RPA.TBCF == true
        Sigma_File = IO_RPA * "/bin/sigma.bin"

        if isfile(Sigma_File) == false
            error("RPA external routines: Transitions = \"Exact\" & TBCF = true, but the RPA 2-body correlation function $Sigma_File does not exist ...")
        end

        println("\nImporting RPA 2-body correlation function ...")
        @time Sigma, Orb_NN = O2b_import(Params,Orb,Sigma_File,Make_Orb_NN = true)
    end

    # Import RPA solutions ...
    @time X_RPA, Y_RPA, E_RPA = RPA_import_binary(Params)

    # Precalculate the reduced matrix elements of Electrogmanetic 1-body transition operators ... (charge radius of the RPA OBDM, as in RPA_solver)
    chR2 = rho_chR2(Params,Orb,C,Rho)
    TrOp = Tr1b_transformation(Params,Orb,C,Tr1b_initialize(Params,Orb,chR2))

    return (IO_RPA = IO_RPA, Orb = Orb, Orb_1p1h = Orb_1p1h, C = C, Rho = Rho, Sigma = Sigma, Orb_NN = Orb_NN,
            X = X_RPA, Y = Y_RPA, E = E_RPA, TrOp = TrOp)
end

function RPA_transition_kernel(Params::Parameters,J::Int64,P::Int64,Orb::Vector{Orb1B},Orb_1p1h::Orb2B_1p1h,Orb_NN,Rho::O1B,Sigma::O2B)
    # Evaluate the transition kernels D_X & D_Y of the RPA amplitudes X & Y for given J & P ... Gamma = D_X * X + D_Y * Y (see the comment above),
    # identical to the coefficients M_X & M_Y of RPA_rM_contract (RPA_transitions.jl), i.e. M_X[ph] = sum_(a,b,t) M^t_ab * D_X[(a,b,t),ph] ...
    #
    #   Linear ...  D_X[(p,h,t),ph] = (-1)^J * (n_h - n_p) ,                    D_Y[(h,p,t),ph] = (-1)^((j_p - j_h)/2) * (n_h - n_p) ,
    #   Exact  ...  1-body part ... G = (1 - Rho) * M * Rho ,
    #               D_X[(c,d,t),ph] += (-1)^J * (1 - Rho)_pc * Rho_dh ,         D_Y[(c,d,t),ph] += -(-1)^((j_p - j_h)/2) * (1 - Rho)_hc * Rho_dp ,
    #               (diagonal Rho ... D_X[(p,h,t),ph] = (-1)^J * rho_h * (1 - rho_p) & D_Y[(h,p,t),ph] = -(-1)^((j_p - j_h)/2) * rho_p * (1 - rho_h)) ,
    #               2-body part (TBCF = true) ... cross-coupled cumulant <ha;I|Sigma|pb;I> ,
    #               D_X[(a,b,+-t),ph] += (-1)^(1 + (j_p + j_h)/2) * K_(1,2) ,     D_Y[(b,a,+-t),ph] += (-1)^(J + (j_p + j_h)/2) * (-1)^((j_a - j_b)/2) * K_(1,2) ,
    #               (i.e. D_X^(2) = sum_I (-1)^(J + I + j_h + j_a) (2I+1) {...} <ha;I|Sigma|pb;I> ... the same corrected sign as in RPA_rM_contract)
    #               K = sum_I (-1)^(J + I + (j_p + j_a)/2) (2I + 1) {j_p j_h J; j_a j_b I} <ha;I|Sigma|pb;I> ,
    #               K_1 (pp/nn part of Sigma) for the pairs of the same isospin t as ph, K_2 (pn part) for the pairs of the opposite isospin -t ...
    #   Norm   ...  f_N = n_h - n_p ... metric of the RPA amplitudes, i.e. sum_ph f_N * (|X|^2 - |Y|^2) = 1 ...
    #
    # Only the pairs (a,b,t) with a non-vanishing kernel are returned (1p-1h & 1h-1p pairs for a diagonal Rho without Sigma) ...
    Transitions = Params.Calc.RPA.Transitions
    TBCF = (Transitions == "Exact") && (Params.Calc.RPA.TBCF == true)

    if (Transitions != "Exact") && (Transitions != "Linear")
        error("Invalid Params.Calc.RPA.Transitions = \"$Transitions\"; use \"Exact\" or \"Linear\".")
    end

    Orb_ph_list = Orb_1p1h.Ind[P,J+1]
    N_ph = Orb_1p1h.N[P,J+1]
    a_max = length(Orb)

    # All orbital pairs (a,b,t) of the multipole (J,P) & their lookup table ...
    Pairs_all = Ind2B[]
    @inbounds for t in (-1,1), Orb_a in Orb, Orb_b in Orb
        if parity(Orb_a.l + Orb_b.l) == P && triangle(Orb_a.j,Orb_b.j,2*J)
            push!(Pairs_all,Ind2B(Orb_a.a,Orb_b.a,t))
        end
    end

    Pair_Map = zeros(Int64,a_max,a_max,2)
    @inbounds for (i_pair,Pair) in enumerate(Pairs_all)
        Pair_Map[Pair.a,Pair.b,div(Pair.T + 3,2)] = i_pair
    end

    # (separate names of the full kernels, captured by the threaded closure below without boxing)
    D_X_all = zeros(Float64,length(Pairs_all),N_ph)
    D_Y_all = zeros(Float64,length(Pairs_all),N_ph)

    # Evaluate the kernels for each ph pair ... (each thread writes its own column)
    @inbounds Threads.@threads for i_ph in 1:N_ph
        Orb_ph = Orb_ph_list[i_ph]
        a_p, a_h, T_ph = Orb_ph.a, Orb_ph.b, Orb_ph.T

        (T_ph == -1 || T_ph == 1) || continue

        l_p, j_p = Orb[a_p].l, Orb[a_p].j
        l_h, j_h = Orb[a_h].l, Orb[a_h].j

        # Isospin indices of the pairs ... same (1) & opposite (2) isospin as the ph pair ...
        t_1, t_2 = div(T_ph + 3,2), div(-T_ph + 3,2)

        # Phase of the Y-term ... M[h,p] -> M[p,h] for the E-type operators ...
        Phase_hp = phase(div(j_p - j_h,2))

        # Linearized transition densities ...
        if Transitions == "Linear"
            D_ph = (T_ph == -1) ? (Orb[a_h].pO - Orb[a_p].pO) : (Orb[a_h].nO - Orb[a_p].nO)

            D_X_all[Pair_Map[a_p,a_h,t_1],i_ph] += phase(J) * D_ph
            D_Y_all[Pair_Map[a_h,a_p,t_1],i_ph] += Phase_hp * D_ph
        end

        # Exact transition densities ...
        if Transitions == "Exact"
            # 1-body part ... Rho is a scalar, i.e. it connects only the orbitals of the same (l,j) ...
            Rho_1 = (T_ph == -1) ? Rho.p : Rho.n

            @inbounds for c in 1:a_max, d in 1:a_max
                # X-term ... G[p,h] = sum_cd (1 - Rho)_pc * M_cd * Rho_dh ...
                if Orb[c].l == l_p && Orb[c].j == j_p && Orb[d].l == l_h && Orb[d].j == j_h
                    D_X_all[Pair_Map[c,d,t_1],i_ph] += phase(J) * (Float64(c == a_p) - Rho_1[a_p,c]) * Rho_1[d,a_h]
                end

                # Y-term ... G[h,p] = sum_cd (1 - Rho)_hc * M_cd * Rho_dp ... (opposite sign w.r.t. the linear branch, see RPA_rM_contract)
                if Orb[c].l == l_h && Orb[c].j == j_h && Orb[d].l == l_p && Orb[d].j == j_p
                    D_Y_all[Pair_Map[c,d,t_1],i_ph] -= Phase_hp * (Float64(c == a_h) - Rho_1[a_h,c]) * Rho_1[d,a_p]
                end
            end

            # 2-body part (2-body correlation function Sigma) ... as in RPA_rM_contract ...
            if TBCF == true
                Phase_X = phase(1 + div(j_p + j_h,2))
                Phase_Y = phase(J + div(j_p + j_h,2))

                @inbounds for Orb_a in Orb
                    a_a, l_a, j_a = Orb_a.a, Orb_a.l, Orb_a.j
                    P_ha = parity(l_h + l_a)

                    @inbounds for Orb_b in Orb
                        a_b, l_b, j_b = Orb_b.a, Orb_b.l, Orb_b.j

                        if triangle(j_a,j_b,2*J) == false || parity(l_p + l_b) != P_ha
                            continue
                        end

                        # Define 2-body kernels ... same (1) & opposite (2) isospin of the orbitals a & b ...
                        K_1, K_2 = 0.0, 0.0

                        I_min = div(max(abs(j_h - j_a),abs(j_p - j_b)),2)
                        I_max = div(min(j_h + j_a,j_p + j_b),2)

                        @inbounds for I in I_min:I_max
                            F = phase(J + I + div(j_p + j_a,2)) * hat(2*I) * w6j(j_p,j_h,2*J,j_a,j_b,2*I)

                            if T_ph == -1
                                K_1 += O2b_pp(a_h,a_a,a_p,a_b,I,P_ha,Sigma,Orb,Orb_NN) * F
                                K_2 += O2b_pn(a_h,a_a,a_p,a_b,I,P_ha,Sigma,Orb_NN) * F
                            else
                                # pn element with the proton first ... both pairs exchanged, phase (-1)^(j_h + j_a + j_p + j_b - 2I) ...
                                K_1 += O2b_nn(a_h,a_a,a_p,a_b,I,P_ha,Sigma,Orb,Orb_NN) * F
                                K_2 += O2b_pn(a_a,a_h,a_b,a_p,I,P_ha,Sigma,Orb_NN) * phase(div(j_p + j_a + j_h + j_b,2)) * F
                            end
                        end

                        Phase_ab = phase(div(j_a - j_b,2))

                        D_X_all[Pair_Map[a_a,a_b,t_1],i_ph] += Phase_X * K_1
                        D_X_all[Pair_Map[a_a,a_b,t_2],i_ph] += Phase_X * K_2
                        D_Y_all[Pair_Map[a_b,a_a,t_1],i_ph] += Phase_Y * Phase_ab * K_1
                        D_Y_all[Pair_Map[a_b,a_a,t_2],i_ph] += Phase_Y * Phase_ab * K_2
                    end
                end
            end
        end
    end

    # Keep only the pairs with a non-vanishing kernel ...
    Keep = [any(!iszero,view(D_X_all,i,:)) || any(!iszero,view(D_Y_all,i,:)) for i in eachindex(Pairs_all)]

    Pairs = Pairs_all[Keep]
    D_X = D_X_all[Keep,:]
    D_Y = D_Y_all[Keep,:]

    # Metric of the RPA amplitudes ...
    f_N = Vector{Float64}(undef,N_ph)

    @inbounds for (ph, Orb_ph) in enumerate(Orb_ph_list)
        a_p, a_h, t_ph = Orb_ph.a, Orb_ph.b, Orb_ph.T
        f_N[ph] = (t_ph == -1) ? (Orb[a_h].pO - Orb[a_p].pO) : (Orb[a_h].nO - Orb[a_p].nO)
    end

    return Pairs, D_X, D_Y, f_N
end

function RPA_transition_density_matrix(Pairs::Vector{Ind2B},D_X::Matrix{Float64},D_Y::Matrix{Float64},X::Matrix{Matrix{ComplexF64}},Y::Matrix{Matrix{ComplexF64}},J::Int64,P::Int64,nu_list::Vector{Int64})
    # Reduced transition density matrices Gamma[(a,b,t),nu] = D_X * X + D_Y * Y of the given phonon levels ...
    N_ph = size(D_X,2)
    Gamma = ComplexF64.(D_X) * X[P,J+1][1:N_ph,nu_list]
    mul!(Gamma,ComplexF64.(D_Y),Y[P,J+1][1:N_ph,nu_list],1.0,1.0)
    return Gamma
end

function RPA_external_multipoles(Params::Parameters,J::Int64,P::Int64,Pairs::Vector{Ind2B},Gamma::AbstractMatrix{ComplexF64},TrOp::Tr1B)
    # Reduced transition matrix elements <nu||M||0> / hat(J) of the physical (p, n), isoscalar & isovector EJ operators for the given phonon levels ...
    #   (same combinations as in RPA_rB ... isoscalar E1 = E1_C + E1_sC (centre-of-mass corrected), isovector E1 with the effective charges) ...
    A = Float64(Params.Calc.A)
    Z = Float64(Params.Calc.Z)
    e_p, e_n = (A - Z) / A, Z / A

    N_phonon = size(Gamma,2)
    pM, nM = zeros(ComplexF64,N_phonon), zeros(ComplexF64,N_phonon)
    isM, ivM = zeros(ComplexF64,N_phonon), zeros(ComplexF64,N_phonon)

    if ((J,P) in [(0,1),(1,2),(2,1),(3,2)]) == false
        return pM, nM, isM, ivM
    end

    Op = (J == 0) ? TrOp.E0 : (J == 1) ? TrOp.E1 : (J == 2) ? TrOp.E2 : TrOp.E3

    @inbounds for nu_ind in 1:N_phonon
        pMSum, nMSum, isMSum, ivMSum = 0.0im, 0.0im, 0.0im, 0.0im

        @inbounds for (ab, Pair) in enumerate(Pairs)
            a_a, a_b, t_ab = Pair.a, Pair.b, Pair.T

            Amp = Gamma[ab,nu_ind] / sqrt(Float64(2*J + 1))

            if t_ab == -1
                MME = Op.p[a_a,a_b] * Amp
                pMSum += MME

                if J == 1
                    isMSum += 0.5 * (TrOp.E1_C.p[a_a,a_b] + TrOp.E1_sC.p[a_a,a_b]) * Amp
                    ivMSum += e_p * MME
                else
                    isMSum += 0.5 * MME
                    ivMSum += 0.5 * MME
                end

            elseif t_ab == 1
                MME = Op.n[a_a,a_b] * Amp
                nMSum += MME

                if J == 1
                    isMSum += 0.5 * (TrOp.E1_C.n[a_a,a_b] + TrOp.E1_sC.n[a_a,a_b]) * Amp
                    ivMSum -= e_n * MME
                else
                    isMSum += 0.5 * MME
                    ivMSum -= 0.5 * MME
                end
            end
        end

        pM[nu_ind] = pMSum
        nM[nu_ind] = nMSum
        isM[nu_ind] = isMSum
        ivM[nu_ind] = ivMSum
    end

    return pM, nM, isM, ivM
end

@inline function RPA_external_gauge(M::ComplexF64)
    # Phase convention of the exported densities ... the proton transition matrix element is made real & positive (no change if it vanishes) ...
    return abs(M) > 0.0 ? conj(M) / abs(M) : 1.0 + 0.0im
end

@inline function RPA_external_normalize!(M::Vector{ComplexF64})
    # Renormalize M ... to reasonable scale averaged densities (no change if all vanish) ...
    Norm = sqrt(sum(abs2.(M)))
    M .= Norm > 0.0 ? abs.(M) ./ Norm : abs.(M)
    return M
end

@inline function RPA_external_YJ(Orb::Vector{Orb1B},a::Int64,b::Int64,J::Int64)
    # Angular & phase factor <a||Y_J||b> ... same convention as in Tr1b_EX_initialize, i.e. TrOp.EX[a,b] = <a||Y_J||b> * <a|r^X|b> (r^2 for E0) ...
    #   (vanishes for l_a + l_b + J odd, i.e. for the unnatural-parity multipoles) ...
    l_a, j_a = Orb[a].l, Orb[a].j
    l_b, j_b = Orb[b].l, Orb[b].j

    if isodd(l_a + l_b + J)
        return 0.0
    end

    return phase(J + div(j_a - 1,2)) * sqrt(Float64((j_a + 1) * (j_b + 1))) * cg(j_a,j_b,2*J,1,-1,0) / sqrt(4.0 * pi)
end

function RPA_external_orbitals_grid(Orb::Vector{Orb1B},C::O1B,r_grid::Vector{Float64},nu_proton::Float64,nu_neutron::Float64)
    # Radial representation of the reference basis orbitals ...
    a_max = length(Orb)
    N_grid = length(r_grid)

    pOrb_grid = Matrix{Float64}(undef,a_max,N_grid)
    nOrb_grid = Matrix{Float64}(undef,a_max,N_grid)

    @inbounds Threads.@threads for i in 1:N_grid
        r = r_grid[i]
        @inbounds for a in 1:a_max
            l_a, j_a = Orb[a].l, Orb[a].j

            pSum, nSum = 0.0, 0.0
            @inbounds for k in 1:a_max
                n_k, l_k, j_k = Orb[k].n, Orb[k].l, Orb[k].j

                if l_a != l_k || j_a != j_k
                    continue
                end

                pSum += C.p[k,a] * Psi_rad_LHO(r,n_k,l_k,nu_proton)
                nSum += C.n[k,a] * Psi_rad_LHO(r,n_k,l_k,nu_neutron)
            end

            pOrb_grid[a,i] = pSum
            nOrb_grid[a,i] = nSum
        end
    end

    return pOrb_grid, nOrb_grid
end

function RPA_external_radial_densities(Orb::Vector{Orb1B},Pairs::Vector{Ind2B},Gamma::AbstractMatrix{ComplexF64},pM::Vector{ComplexF64},
                                       J::Int64,P::Int64,r_grid::Vector{Float64},pOrb_grid::Matrix{Float64},nOrb_grid::Matrix{Float64},Power::Int64)
    # Radial transition densities r^2 / (2J + 1) * r^Power * sum_ab Re(Gauge * Gamma_ab) <a||Y_J||b> R_a(r) R_b(r) of the given phonon levels ...
    #   (Power = 0 ... transition densities, Power = X ... transition operator densities of r^X Y_J) ...
    N_phonon = size(Gamma,2)
    N_grid = length(r_grid)

    Amp_ab = [RPA_external_YJ(Orb,Pair.a,Pair.b,J) for Pair in Pairs]

    # Gauge of the given phonon levels ...
    Gauge = ones(ComplexF64,N_phonon)
    if ((J,P) in [(0,1),(1,2),(2,1),(3,2)]) == true
        Gauge .= RPA_external_gauge.(pM)
    end

    pRho_nu = Matrix{Float64}(undef,N_phonon,N_grid)
    nRho_nu = Matrix{Float64}(undef,N_phonon,N_grid)

    @inbounds Threads.@threads for i in 1:N_grid
        r = r_grid[i]
        Amp = r^(2 + Power) / Float64(2*J + 1)

        @inbounds for nu_ind in 1:N_phonon
            pSum, nSum = 0.0, 0.0

            @inbounds for (ab, Pair) in enumerate(Pairs)
                a_a, a_b, t_ab = Pair.a, Pair.b, Pair.T
                Amp_ab[ab] == 0.0 && continue

                Rho_ab = real(Gauge[nu_ind] * Gamma[ab,nu_ind]) * Amp_ab[ab]

                if t_ab == -1
                    pSum += Rho_ab * pOrb_grid[a_a,i] * pOrb_grid[a_b,i]
                elseif t_ab == 1
                    nSum += Rho_ab * nOrb_grid[a_a,i] * nOrb_grid[a_b,i]
                end
            end

            pRho_nu[nu_ind,i] = Amp * pSum
            nRho_nu[nu_ind,i] = Amp * nSum
        end
    end

    return pRho_nu, nRho_nu
end

function RPA_external_export_densities(IO_RPA::String,File_Name::String,JP::String,J::Int64,P::Int64,nu_list::Vector{Int64},E_RPA,
                                       r_grid::Vector{Float64},pRho_nu::Matrix{Float64},nRho_nu::Matrix{Float64},
                                       pM::Vector{ComplexF64},nM::Vector{ComplexF64},isM::Vector{ComplexF64},ivM::Vector{ComplexF64},Header::Vector{String})
    # Export of the individual & averaged radial densities ...
    N_grid = length(r_grid)
    N_phonon = length(nu_list)

    # Perform the export of indiviual RPA radial phonon densities ...
    @inbounds for (nu_ind, nu) in enumerate(nu_list)
        e = real(E_RPA[P,J+1][nu])
        sE = string(round(e,digits=3))

        Output_File = IO_RPA * "/densities/RPA_" * File_Name * "_JP$(JP)_nu$(nu)_E$(sE).dat"
        open(Output_File, "w") do Write_File
            @printf(Write_File, "%-20s %-20s %-20s %-20s %-20s\n", "r", Header...)
            @inbounds for i in 1:N_grid
                r, pRho, nRho = r_grid[i], pRho_nu[nu_ind,i], nRho_nu[nu_ind,i]
                isRho = 0.5 * (pRho + nRho)
                ivRho = 0.5 * (pRho - nRho)
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f %-20.8f\n", r, pRho, nRho, isRho, ivRho)
            end
        end
    end

    display("Export of individual phonon radial transition densities successfuly completed ...")

    # Check if averaging of transition densities is possible for given values of J & P ...
    if ((J,P) in [(0,1),(1,2),(2,1),(3,2)]) == false
        return
    end

    # Renormalize M ... to reasonable scale averaged densities ...
    RPA_external_normalize!(pM)
    RPA_external_normalize!(nM)
    RPA_external_normalize!(isM)
    RPA_external_normalize!(ivM)

    # Perform the export of of averaged RPA transition radial densities ...
    Output_File = IO_RPA * "/densities/RPA_" * File_Name * "_JP$(JP)_Averaged.dat"
    open(Output_File, "w") do Write_File
        @printf(Write_File, "%-20s %-20s %-20s %-20s %-20s\n", "r", "rho_p", "rho_n", "rho_is", "rho_iv")

        @inbounds for i in 1:N_grid
            r = r_grid[i]
            pRho, nRho, isRho, ivRho = 0.0, 0.0, 0.0, 0.0

            @inbounds for nu_ind in 1:N_phonon
                prho, nrho = pRho_nu[nu_ind,i], nRho_nu[nu_ind,i]
                pRho += real(pM[nu_ind]) * prho
                nRho += real(nM[nu_ind]) * nrho
                isRho += 0.5 * real(isM[nu_ind]) * (prho + nrho)
                ivRho += 0.5 * real(ivM[nu_ind]) * (prho - nrho)
            end
            @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f %-20.8f\n", r, pRho, nRho, isRho, ivRho)
        end
    end

    display("Export of averaged phonon radial transitions densities successfuly completed ...")

    return
end

function RPA_transition_densities(Params::Parameters;JP::String,nu_list::Vector{Int64},File_Name::String = "Phonon_Densities")
    # Initialize the angular momentum algebra ...
    wigner_init_float(75, "Jmax", 9)

    # Read parameters ...
    A = Float64(Params.Calc.A)
    hw = Params.Calc.hw

    # Define basic constants ...
    hc = 197.326980
    m_n = 939.565346
    m_p = 938.272013
    nu_proton = 0.5 * m_p * hw / hc^2
    nu_neutron = 0.5 * m_n * hw / hc^2

    # Define the properties of radial grid ...
    r_min, r_max = 0.0 + 1e-8, 2.5 * 1.2 * A^(1/3)
    N_grid = 2^13 + 1 # 8192 + 1 grid points ...

    # Read the values of J & P ...
    J, P = RPA_external_JP(JP)

    # Import RPA data ...
    Data = RPA_external_import(Params)

    # Evaluate the transition kernels & the transition density matrices of the given phonon levels ...
    Pairs, D_X, D_Y, f_N = RPA_transition_kernel(Params,J,P,Data.Orb,Data.Orb_1p1h,Data.Orb_NN,Data.Rho,Data.Sigma)
    Gamma = RPA_transition_density_matrix(Pairs,D_X,D_Y,Data.X,Data.Y,J,P,nu_list)

    # Reduced transition matrix elements of the given phonon levels ...
    pM, nM, isM, ivM = RPA_external_multipoles(Params,J,P,Pairs,Gamma,Data.TrOp)

    # Radial grid & radial representation of the reference basis orbitals ...
    r_grid = collect(range(r_min, stop = r_max, length = N_grid))
    pOrb_grid, nOrb_grid = RPA_external_orbitals_grid(Data.Orb,Data.C,r_grid,nu_proton,nu_neutron)

    # Radial transition densities of the given phonon levels ...
    pRho_nu, nRho_nu = RPA_external_radial_densities(Data.Orb,Pairs,Gamma,pM,J,P,r_grid,pOrb_grid,nOrb_grid,0)

    # Export ...
    RPA_external_export_densities(Data.IO_RPA,File_Name,JP,J,P,nu_list,Data.E,r_grid,pRho_nu,nRho_nu,pM,nM,isM,ivM,["rho_p","rho_n","rho_is","rho_iv"])

    return
end

function RPA_transition_operator_densities(Params::Parameters;JP::String,nu_list::Vector{Int64},File_Name::String = "Phonon_Transition_Operator_Densities")
    # Initialize the angular momentum algebra ...
    wigner_init_float(75, "Jmax", 9)

    # Read parameters ...
    A = Float64(Params.Calc.A)
    hw = Params.Calc.hw

    # Define basic constants ...
    hc = 197.326980
    m_n = 939.565346
    m_p = 938.272013
    nu_proton = 0.5 * m_p * hw / hc^2
    nu_neutron = 0.5 * m_n * hw / hc^2

    # Define the properties of radial grid ...
    r_min, r_max = 0.0 + 1e-8, 2.5 * 1.2 * A^(1/3)
    N_grid = 2^13 + 1 # 8192 + 1 grid points ...

    # Read the values of J & P ...
    J, P = RPA_external_JP(JP)

    # Check if the given JP is supported for transition density calculations ...
    if ((J,P) in [(0,1),(1,2),(2,1),(3,2)]) == false
        error("The given values of J & P are not supported for radial transition operator density calculations ...")
    end

    # Radial power of the transition operator ... E0 = r^2 Y_0, E1 = r Y_1, E2 = r^2 Y_2, E3 = r^3 Y_3 (as in Tr1b) ...
    Power = (J == 0) ? 2 : J

    # Import RPA data ...
    Data = RPA_external_import(Params)

    # Evaluate the transition kernels & the transition density matrices of the given phonon levels ...
    Pairs, D_X, D_Y, f_N = RPA_transition_kernel(Params,J,P,Data.Orb,Data.Orb_1p1h,Data.Orb_NN,Data.Rho,Data.Sigma)
    Gamma = RPA_transition_density_matrix(Pairs,D_X,D_Y,Data.X,Data.Y,J,P,nu_list)

    # Reduced transition matrix elements of the given phonon levels ...
    pM, nM, isM, ivM = RPA_external_multipoles(Params,J,P,Pairs,Gamma,Data.TrOp)

    # Radial grid & radial representation of the reference basis orbitals ...
    r_grid = collect(range(r_min, stop = r_max, length = N_grid))
    pOrb_grid, nOrb_grid = RPA_external_orbitals_grid(Data.Orb,Data.C,r_grid,nu_proton,nu_neutron)

    # Radial transition operator densities of the given phonon levels ... (integrate to rM / (2J + 1) = (-1)^J <nu||M||0> / (2J + 1) of the proton & neutron EJ operator) ...
    pRho_nu, nRho_nu = RPA_external_radial_densities(Data.Orb,Pairs,Gamma,pM,J,P,r_grid,pOrb_grid,nOrb_grid,Power)

    # Export ...
    RPA_external_export_densities(Data.IO_RPA,File_Name,JP,J,P,nu_list,Data.E,r_grid,pRho_nu,nRho_nu,pM,nM,isM,ivM,["pM","nM","isM","ivM"])

    return
end

function RPA_transition_analysis(Params::Parameters;JP::String,nu_list::Vector{Int64},File_Name::String = "Phonon_Transtion_Analysis", Cut::Float64 = 0.1)
    # Initialize the angular momentum algebra ...
    wigner_init_float(75, "Jmax", 9)

    # Read the values of J & P ...
    J, P = RPA_external_JP(JP)

    # Import RPA data ...
    Data = RPA_external_import(Params)
    IO_RPA, Orb, Orb_1p1h, X_RPA, Y_RPA, E_RPA, TrOp = Data.IO_RPA, Data.Orb, Data.Orb_1p1h, Data.X, Data.Y, Data.E, Data.TrOp

    # Make directory for export of information on structure of phonon levels ...
    if isdir(IO_RPA * "/levels") == false
        mkdir(IO_RPA * "/levels")
    end

    # Determine the number of phonon levels & the total number of phonons ...
    N_phonon = length(nu_list)
    N_ph = Orb_1p1h.N[P,J+1]

    # Evaluate the transition kernels of the RPA amplitudes X & Y ...
    Pairs, D_X, D_Y, f_N = RPA_transition_kernel(Params,J,P,Orb,Orb_1p1h,Data.Orb_NN,Data.Rho,Data.Sigma)

    # Contributions of the individual ph pairs to the reduced transition matrix elements of the EJ operator, separately for the proton & neutron operator ...
        # M^t_ph(nu) = W_X^t[ph] * X[ph,nu] + W_Y^t[ph] * Y[ph,nu] ,   W_X^t[ph] = sum_(a,b,t) M^t_ab * D_X[(a,b,t),ph] ... (sum over all ph = rM^t of RPA_rM)
    pW_X, pW_Y, nW_X, nW_Y = zeros(Float64,N_ph), zeros(Float64,N_ph), zeros(Float64,N_ph), zeros(Float64,N_ph)

    if ((J,P) in [(0,1),(1,2),(2,1),(3,2)]) == true
        Op = (J == 0) ? TrOp.E0 : (J == 1) ? TrOp.E1 : (J == 2) ? TrOp.E2 : TrOp.E3

        @inbounds for (ab, Pair) in enumerate(Pairs)
            a_a, a_b, t_ab = Pair.a, Pair.b, Pair.T
            @inbounds for ph in 1:N_ph
                if t_ab == -1
                    pW_X[ph] += Op.p[a_a,a_b] * D_X[ab,ph]
                    pW_Y[ph] += Op.p[a_a,a_b] * D_Y[ab,ph]
                else
                    nW_X[ph] += Op.n[a_a,a_b] * D_X[ab,ph]
                    nW_Y[ph] += Op.n[a_a,a_b] * D_Y[ab,ph]
                end
            end
        end
    end

    # Note: the amplitudes renormalized by sqrt(f_N) fulfil sum_ph (|X|^2 - |Y|^2) = 1 & are used for the Cut & the export,
    #       while the 1p-1h contributions M_ph to the reduced transition matrix element use the transition kernels D_X & D_Y,
    #       M_ph ... contribution to rM of the operator of the same isospin as the ph pair, M_ph(-t) ... of the opposite isospin (only for the pn part of Sigma) ...

    nu_compo = Vector{Vector{Tuple{Int64,Int64,Int64,Float64,Float64,Float64,Float64}}}(undef,N_phonon)

    # Iterate through the phonon solutions & determine the most dominant ph & hp components of given phonon levels ...
    @inbounds for (nu_ind, nu) in enumerate(nu_list)
        nu_compo[nu_ind] = Tuple{Int64,Int64,Int64,Float64,Float64,Float64,Float64}[]

        @inbounds for ph in 1:N_ph
            Orb_ph = Orb_1p1h.Ind[P,J+1][ph]
            a_p, a_h, t_ph = Orb_ph.a, Orb_ph.b, Orb_ph.T
            x_RPA, y_RPA = sqrt(f_N[ph]) * X_RPA[P,J+1][ph,nu], sqrt(f_N[ph]) * Y_RPA[P,J+1][ph,nu]

            Amp = (phase(J) * x_RPA + y_RPA)

            if abs2(Amp) > Cut
                X_ph, Y_ph = X_RPA[P,J+1][ph,nu], Y_RPA[P,J+1][ph,nu]

                pM_ph = real(pW_X[ph] * X_ph + pW_Y[ph] * Y_ph)
                nM_ph = real(nW_X[ph] * X_ph + nW_Y[ph] * Y_ph)

                M_1, M_2 = (t_ph == -1) ? (pM_ph, nM_ph) : (nM_ph, pM_ph)

                push!(nu_compo[nu_ind],(a_p,a_h,t_ph,real(x_RPA),real(y_RPA),M_1,M_2))
            end
        end
    end

    # Perform the export of information on individual RPA phonon levels ...
    @inbounds for (nu_ind, nu) in enumerate(nu_list)
        e = real(E_RPA[P,J+1][nu])
        sE = string(round(e,digits=3))

        Phonon_RPA = nu_compo[nu_ind]
        Output_File = IO_RPA * "/levels/RPA_" * File_Name * "_JP$(JP)_nu$(nu)_E$(sE).dat"
        open(Output_File, "w") do Write_File
            @printf(Write_File, "%-8s %-8s %-8s %-20s %-20s %-20s %-20s\n", "a_p", "a_h", "T_ph", "Re X_ph", "Re Y_ph", "Re M_ph", "Re M_ph(-t)")
            @inbounds for L in Phonon_RPA
                (a_p,a_h,t_ph,x_RPA,y_RPA,M_ph,M_ph_2) = L
                @printf(Write_File, "%-8d %-8d %-8d %-20.8f %-20.8f %-20.8f %-20.8f\n", a_p, a_h, t_ph, x_RPA, y_RPA, M_ph, M_ph_2)
            end
        end

    end

    display("Export of information on individual phonon levels successfuly completed ...")

    return
end

function RPA_transition_currents(Params::Parameters;JP::String,nu_list::Vector{Int64},File_Name::String = "Phonon_Currents",
                                     M::Int64=-1,xN_grid::Int64 = 40, zN_grid::Int64 = 40)
    # Note ... variables captured by the Threads.@threads closures below are assigned only once at the function level,
    #          otherwise Julia boxes them (Core.Box ... type-unstable access & a potential data race if ever written inside the threaded loops) ...

    # Initialize the angular momentum algebra ...
    wigner_init_float(75, "Jmax", 9)

    # Read parameters ...
    A = Float64(Params.Calc.A)
    hw = Params.Calc.hw
    N_max = Params.Calc.Nmax
    a_max = div((N_max+1)*(N_max+2),2)

    # Define basic constants ...
    hc = 197.326980
    m_n = 939.565346
    m_p = 938.272013
    g_n = -3.82608545
    g_p = 5.585694713
    nu_proton = 0.5 * m_p * hw / hc^2
    nu_neutron = 0.5 * m_n * hw / hc^2

    # Define the properties of radial grid ...
    r_min, r_max = 0.01, 3.0 * A^(1/3)
    rN_grid = 2^13 + 1 # 8192 + 1 grid points ...
    tN_grid = 256

    # Read the values of J & P ...
    J, P = RPA_external_JP(JP)

    # Set the value of projection M ... (M = -1 is the default for M = 0, keyword argument is not reassigned)
    M_proj = (M == -1) ? 0 : M

    # Import RPA data ...
    Data = RPA_external_import(Params)
    IO_RPA, Orb, E_RPA = Data.IO_RPA, Data.Orb, Data.E

    # Determine the number of phonon levels ...
    N_phonon = length(nu_list)

    # Evaluate the transition kernels & the transition density matrices of the given phonon levels ...
    Pairs, D_X, D_Y, f_N = RPA_transition_kernel(Params,J,P,Orb,Data.Orb_1p1h,Data.Orb_NN,Data.Rho,Data.Sigma)
    Gamma = RPA_transition_density_matrix(Pairs,D_X,D_Y,Data.X,Data.Y,J,P,nu_list)
    N_ab = length(Pairs)

    # Reduced transition matrix elements of the given phonon levels ...
    pM, nM, isM, ivM = RPA_external_multipoles(Params,J,P,Pairs,Gamma,Data.TrOp)

    # Initialize the radial & angular grid ...
        # Case of the radial grid ...
    r_grid = collect(range(r_min, stop = r_max, length = rN_grid))
        # Case of the angular theta grid ...
    theta_grid = collect(range(0.0, stop = 2.0*pi, length = tN_grid))

    # Radial representation of the reference basis orbitals ...
    pR_grid, nR_grid = RPA_external_orbitals_grid(Orb,Data.C,r_grid,nu_proton,nu_neutron)

    # Radial representation of the radial derivative of reference basis orbitals ...
    pdR_grid = Matrix{Float64}(undef,a_max,rN_grid)
    ndR_grid = Matrix{Float64}(undef,a_max,rN_grid)

    @inbounds Threads.@threads for a in 1:a_max
        pR_a = collect(@view pR_grid[a,:])
        nR_a = collect(@view nR_grid[a,:])
        pdR_a = differentiate_Oh6(r_grid,pR_a)
        ndR_a = differentiate_Oh6(r_grid,nR_a)
        @views pdR_grid[a,:] .= pdR_a
        @views ndR_grid[a,:] .= ndR_a
    end

    # Prepare the current map grid ...
        # Grid limits & step size ...
    x_min, x_max = -0.8 * r_max, 0.8 * r_max
    z_min, z_max = -0.8 * r_max, 0.8 * r_max
    dx, dz = (x_max - x_min) / Float64(xN_grid), (z_max - z_min) / Float64(zN_grid)
        # Grid size ...
    N_map = xN_grid * zN_grid
        # Map coordinate & index grid ...
    i_map = Matrix{Int64}(undef,2,N_map)
    x_map = Matrix{Float64}(undef,2,N_map)
        # Orbital pair current maps ... (the isospin of a pair is given by Pairs)
    J_ab_conv_map = zeros(Float64,3,N_ab,N_map)
    J_ab_spin_map = zeros(Float64,3,N_ab,N_map)
        # Phonon level current maps ...
    pJ_conv_map = Array{Float64}(undef,3,N_phonon,N_map)
    nJ_conv_map = Array{Float64}(undef,3,N_phonon,N_map)
    pJ_spin_map = Array{Float64}(undef,3,N_phonon,N_map)
    nJ_spin_map = Array{Float64}(undef,3,N_phonon,N_map)

    # Prepare the current map grid ...
        # Precalculate radial & angular grid step size ...
    dr = (r_max - r_min) / (rN_grid - 1)
    dtheta = 2.0 * pi / (tN_grid - 1)  # theta_grid spans 0..2*pi inclusive
        # Reset the grid size counter ...
    i_N = 0
        # Prepare the cartesian-like grid from initial radial x angular grid ...
    @inbounds for ind_x in 1:xN_grid
        x = x_min + Float64(ind_x) * dx
        @inbounds for ind_z in 1:zN_grid
            z = z_min + Float64(ind_z) * dz

            r = sqrt(x^2 + z^2)
            theta = atan(x,z)
            if theta < 0.0
                theta += 2.0*pi
            end

            n = Int64(round((r - r_min) / dr)) + 1
            m = Int64(round((theta - 0.0) / dtheta)) + 1

            n = clamp(n, 1, rN_grid)
            m = clamp(m, 1, tN_grid)

            i_N += 1
            i_map[1,i_N] = n
            i_map[2,i_N] = m
            x_map[1,i_N] = r_grid[n]
            x_map[2,i_N] = theta_grid[m]
        end
    end

    # Evaluate the grid of the 1-body transition current matrix elements of the orbital pairs ...
        # J_ab = sum_(m_a mu) (-1)^(j_b - mu) (j_a m_a j_b -mu|J M) <a m_a|j|b mu> , with mu = -m_b below, i.e. the phase (-1)^((j_b + m_b)/2) ...
        # for the pairs (p,h) the former ph term, for the pairs (h,p) the former hp term (up to the phase (-1)^(J + (j_p - j_h)/2) absorbed in the kernel D_Y) ...
        # J1b_convective = (grad psi_a)^+ psi_b - psi_a^+ grad psi_b = -2i j_conv & J1b_spin = -(i/2) curl(psi_a^+ sigma psi_b) , i.e. (g/2) * J1b_spin = -i j_mag
        # (j in units of hbar/m) ... the factor 1/2 of the convective current is applied with the units below ...
    @time @inbounds Threads.@threads for N in 1:N_map
        n, m = i_map[1,N], i_map[2,N]
        r, theta = r_grid[n], theta_grid[m]

        @inbounds for ab in 1:N_ab
            Pair = Pairs[ab]
            a_a, a_b, t_ab = Pair.a, Pair.b, Pair.T

            l_a, j_a = Orb[a_a].l, Orb[a_a].j
            l_b, j_b = Orb[a_b].l, Orb[a_b].j

            if t_ab == -1
                R_a, R_b = pR_grid[a_a,n], pR_grid[a_b,n]
                dR_a, dR_b = pdR_grid[a_a,n], pdR_grid[a_b,n]
                g = 0.5 * g_p
            else
                R_a, R_b = nR_grid[a_a,n], nR_grid[a_b,n]
                dR_a, dR_b = ndR_grid[a_a,n], ndR_grid[a_b,n]
                g = 0.5 * g_n
            end

            Jx_conv, Jy_conv, Jz_conv = 0.0, 0.0, 0.0
            Jx_spin, Jy_spin, Jz_spin = 0.0, 0.0, 0.0

            @inbounds for m_a in -j_a:2:j_a
                m_b = 2*M_proj - m_a

                abs(m_b) > j_b && continue

                Amp_ab = cg(j_a,j_b,2*J,m_a,m_b,2*M_proj) * phase(div(j_b + m_b,2))

                abs(Amp_ab) < 1e-14 && continue

                (jx_conv,jy_conv,jz_conv) = J1b_convective(r,theta,0.0,l_a,j_a,m_a,R_a,dR_a,l_b,j_b,-m_b,R_b,dR_b)
                (jx_spin,jy_spin,jz_spin) = g .* J1b_spin(r,theta,0.0,l_a,j_a,m_a,R_a,dR_a,l_b,j_b,-m_b,R_b,dR_b)

                Jx_conv += Amp_ab * real(jx_conv)
                Jy_conv += Amp_ab * real(jy_conv)
                Jz_conv += Amp_ab * real(jz_conv)

                Jx_spin += Amp_ab * real(jx_spin)
                Jy_spin += Amp_ab * real(jy_spin)
                Jz_spin += Amp_ab * real(jz_spin)
            end

            J_ab_conv_map[1,ab,N] = Jx_conv
            J_ab_conv_map[2,ab,N] = Jy_conv
            J_ab_conv_map[3,ab,N] = Jz_conv
            J_ab_spin_map[1,ab,N] = Jx_spin
            J_ab_spin_map[2,ab,N] = Jy_spin
            J_ab_spin_map[3,ab,N] = Jz_spin
        end
    end

    # Gauge of the given phonon levels ...
    Gauge = ones(ComplexF64,N_phonon)
    if ((J,P) in [(0,1),(1,2),(2,1),(3,2)]) == true
        Gauge .= RPA_external_gauge.(pM)
    end

    # Common normalization of the current maps & the density files ... 1 / hat(J) (see the comment at the top of this file) ...
    Norm_J = 1.0 / sqrt(Float64(2*J + 1))

    # Evaluate the grid of transition current matrix elements for given phonon levels ...
    @inbounds Threads.@threads for N in 1:N_map
        @inbounds for nu_ind in 1:N_phonon
            pJ_conv, nJ_conv = zeros(Float64,3), zeros(Float64,3)
            pJ_spin, nJ_spin = zeros(Float64,3), zeros(Float64,3)

            @inbounds for ab in 1:N_ab
                # Amplitude of the orbital pair ... the m-scheme current is j_nu = (-1)^J sum_ab Gamma_ab J_ab[j] , the pair maps hold J_ab[-i j] ,
                # the exported field (-1)^J i j_nu / hat(J) = sum_ab Gamma_ab (i * i) J_ab[-i j] / hat(J) ... hence the factor -1 / hat(J) ...
                Amp = -Norm_J * real(Gauge[nu_ind] * Gamma[ab,nu_ind])

                if Pairs[ab].T == -1
                    @inbounds for k in 1:3
                        pJ_conv[k] += Amp * J_ab_conv_map[k,ab,N]
                        pJ_spin[k] += Amp * J_ab_spin_map[k,ab,N]
                    end
                else
                    @inbounds for k in 1:3
                        nJ_conv[k] += Amp * J_ab_conv_map[k,ab,N]
                        nJ_spin[k] += Amp * J_ab_spin_map[k,ab,N]
                    end
                end
            end

            @inbounds for k in 1:3
                pJ_conv_map[k,nu_ind,N] = pJ_conv[k]
                nJ_conv_map[k,nu_ind,N] = nJ_conv[k]
                pJ_spin_map[k,nu_ind,N] = pJ_spin[k]
                nJ_spin_map[k,nu_ind,N] = nJ_spin[k]
            end
        end
    end

    # Scale the currents to correct physical units in [c] ... (convective current j = hbar/(2 m i) (psi^+ grad psi - grad psi^+ psi), i.e. the factor 1/2) ...
    pJ_conv_map .= pJ_conv_map .* 0.5 * hc / m_p
    nJ_conv_map .= nJ_conv_map .* 0.5 * hc / m_n
    pJ_spin_map .= pJ_spin_map .* hc / m_p
    nJ_spin_map .= nJ_spin_map .* hc / m_n

    # Perform the export of of individual RPA phonon level transition current maps ...
    @inbounds for (nu_ind, nu) in enumerate(nu_list)
        e = real(E_RPA[P,J+1][nu])
        sE = string(round(e,digits=3))

        for (Label, pJ_map, nJ_map) in (("Convective", pJ_conv_map, nJ_conv_map), ("Spin", pJ_spin_map, nJ_spin_map))
            Output_File = IO_RPA * "/densities/RPA_" * Label * "_" * File_Name * "_JP$(JP)_nu$(nu)_E$(sE).dat"
            open(Output_File, "w") do Write_File
                @printf(Write_File, "%-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s\n", "x", "z", "pJ_x", "pJ_z", "nJ_x", "nJ_z", "isJ_x", "isJ_z", "ivJ_x", "ivJ_z")
                @inbounds for N in 1:N_map
                    r, theta = x_map[1,N], x_map[2,N]
                    x = r * sin(theta)
                    z = r * cos(theta)
                    pJ_x = pJ_map[1,nu_ind,N]
                    pJ_z = pJ_map[3,nu_ind,N]
                    nJ_x = nJ_map[1,nu_ind,N]
                    nJ_z = nJ_map[3,nu_ind,N]
                    isJ_x = 0.5 * (pJ_x + nJ_x)
                    isJ_z = 0.5 * (pJ_z + nJ_z)
                    ivJ_x = 0.5 * (pJ_x - nJ_x)
                    ivJ_z = 0.5 * (pJ_z - nJ_z)
                    @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f\n", x, z, pJ_x, pJ_z, nJ_x, nJ_z, isJ_x, isJ_z, ivJ_x, ivJ_z)
                end
            end
        end
    end

    display("Export of individual phonon level transition current maps successfuly completed ...")

    if ((J,P) in [(0,1),(1,2),(2,1),(3,2)]) == false
        return
    end

    # Renormalize M ... to reasonable scale averaged current densities ...
    RPA_external_normalize!(pM)
    RPA_external_normalize!(nM)
    RPA_external_normalize!(isM)
    RPA_external_normalize!(ivM)

    # Perform the export of of averaged RPA transition current maps ... (convective & spin transition current densities)
    for (Label, pJ_map, nJ_map) in (("Convective", pJ_conv_map, nJ_conv_map), ("Spin", pJ_spin_map, nJ_spin_map))
        Output_File = IO_RPA * "/densities/RPA_" * Label * "_" * File_Name * "_JP$(JP)_Averaged.dat"
        open(Output_File, "w") do Write_File
            @printf(Write_File, "%-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s\n", "x", "z", "ppJ_x", "ppJ_z", "pnJ_x", "pnJ_z", "npJ_x", "npJ_z", "nnJ_x", "nnJ_z", "isJ_x", "isJ_z", "ivJ_x", "ivJ_z")
            @inbounds for N in 1:N_map
                r, theta = x_map[1,N], x_map[2,N]
                x = r * sin(theta)
                z = r * cos(theta)
                ppJ_x, pnJ_x, npJ_x, nnJ_x, isJ_x, ivJ_x = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                ppJ_z, pnJ_z, npJ_z, nnJ_z, isJ_z, ivJ_z = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

                @inbounds for nu_ind in 1:N_phonon
                    pJ_x, pJ_z = pJ_map[1,nu_ind,N], pJ_map[3,nu_ind,N]
                    nJ_x, nJ_z = nJ_map[1,nu_ind,N], nJ_map[3,nu_ind,N]

                    pm, nm = real(pM[nu_ind]), real(nM[nu_ind])
                    ism, ivm = real(isM[nu_ind]), real(ivM[nu_ind])

                    ppJ_x += pm * pJ_x
                    ppJ_z += pm * pJ_z
                    pnJ_x += pm * nJ_x
                    pnJ_z += pm * nJ_z
                    npJ_x += nm * pJ_x
                    npJ_z += nm * pJ_z
                    nnJ_x += nm * nJ_x
                    nnJ_z += nm * nJ_z
                    isJ_x += 0.5 * ism * (pJ_x + nJ_x)
                    isJ_z += 0.5 * ism * (pJ_z + nJ_z)
                    ivJ_x += 0.5 * ivm * (pJ_x - nJ_x)
                    ivJ_z += 0.5 * ivm * (pJ_z - nJ_z)
                end

                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f\n", x, z, ppJ_x, ppJ_z, pnJ_x, pnJ_z, npJ_x, npJ_z, nnJ_x, nnJ_z, isJ_x, isJ_z, ivJ_x, ivJ_z)
            end
        end
    end

    display("Export of averaged phonon level transition current maps successfuly completed ...")

    return
end

function RPA_transitions_recalculate(Params::Parameters)
    # Initialize the angular momentum algebra ...
    wigner_init_float(75, "Jmax", 9)

    # Import RPA data ... (orbitals, basis, RPA OBDM, for Transitions = "Exact" & TBCF = true the RPA 2-body correlation function, amplitudes & operators)
    Data = RPA_external_import(Params)
    IO_RPA, Orb, Orb_1p1h, C, Rho = Data.IO_RPA, Data.Orb, Data.Orb_1p1h, Data.C, Data.Rho

    # Check if the RPA transition directory exists, if not create it ...
    if !(isdir(IO_RPA * "/transitions"))
        mkdir(IO_RPA * "/transitions")
        mkdir(IO_RPA * "/transitions/E0")
        mkdir(IO_RPA * "/transitions/E1")
        mkdir(IO_RPA * "/transitions/E2")
        mkdir(IO_RPA * "/transitions/E3")
    end

    # Orbital pairs for the 2-body correlation function ... (Sigma & Orb_NN are used only for Transitions = "Exact" & TBCF = true)
    Orb_NN = Data.Orb_NN === nothing ? O2b_initialize(Params,Orb,Make_Orb_NN = true)[2] : Data.Orb_NN

    # Calculate the reduced transition matrix elements M_nu for each phonon level ... (as in RPA_solver)
    @time rM = RPA_rM(Params,Orb,Orb_1p1h,Orb_NN,Data.TrOp,Rho,Data.Sigma,Data.X,Data.Y)

    # Evaluate the reduced transition intensities B_nu for each phonon level ...
    @time rB = RPA_rB(Params,Orb_1p1h,rM,real.(Data.E))

    # Remove the previous transition moments & TRK values from RPA_summary.dat ... they are appended again by RPA_transitions_export ...
    Summary_File = IO_RPA * "/RPA_summary.dat"
    if isfile(Summary_File)
        Summary = read(Summary_File,String)
        i_moments = findfirst("\nReview of RPA electromagnetic transition moments",Summary)
        if i_moments !== nothing
            write(Summary_File,Summary[1:first(i_moments)-1])
        end
    end

    # Export of RPA electromagnetic transitions ... (RPA OBDM passed in the LHO basis, as in RPA_solver ... Rho_LHO = C * Rho * C')
    Rho_LHO = O1b_transformation(Rho,O1b_transpose(C))

    @time RPA_transitions_export(Params,Orb,Orb_1p1h,C,Rho_LHO,Data.E,rB)

    return
end
