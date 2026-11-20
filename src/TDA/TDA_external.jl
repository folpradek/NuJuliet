function TDA_import_binary(Params::Parameters)
    # Parameters / dimensions
    N_max = Params.Calc.Nmax
    J_max = 2*N_max + 1
    bin_path = "io/" * Params.Calc.Path * "/" * Params.Calc.TDA.Name * "/bin/"

    # Initialite the list of values of J & P for iteration ...
    JP_list = JP_initialize(J_max)

    # Import reference single-particle orbitals ... (the TDA does not export any ground-state data)
    Orb = orbitals_import(Params,"io/" * Params.Calc.Path * "/" * Params.Calc.TDA.Reference * "/bin/orb.bin")

    # Construct 2-body 1p-1h orbitals ... (as in TDA_solver_import)
    Orb_1p1h = orbitals_1p1h_make(Params,Orb;Tol_par = Params.Calc.TDA.Tol_par,Tol_hol = Params.Calc.TDA.Tol_hol)

    println("\nImporting TDA solutions from binary format ...")

    # Initialize amplitudes & energies ...
    X_TDA = Matrix{Matrix{Float64}}(undef,2,J_max+1)
    E_TDA = Matrix{Vector{Float64}}(undef,2,J_max+1)

    # Initialite each J & P block ...
    @inbounds for JP in JP_list
        J, P = JP[1], JP[2]
        N_ph = Orb_1p1h.N[P,J+1]
        X_TDA[P,J+1] = Matrix{Float64}(undef,N_ph,N_ph)
        E_TDA[P,J+1] = Vector{Float64}(undef,N_ph)
    end

    # Read TDA amplitudes X ...
    import_path_X = bin_path * "TDA_X.bin"
    open(import_path_X, "r") do Import_File
        @inbounds for JP in JP_list
            J, P = JP[1], JP[2]
            N_ph = Orb_1p1h.N[P,J+1]
            N_ph == 0 && continue
            Buffer = Vector{Float64}(undef,N_ph*N_ph)
            read!(Import_File,Buffer)
            X_TDA[P,J+1] .= transpose(reshape(Buffer,N_ph,N_ph))
        end
    end

    # Read TDA energies E ...
    import_path_E = bin_path * "TDA_E.bin"
    open(import_path_E, "r") do Import_File
        @inbounds for JP in JP_list
            J, P = JP[1], JP[2]
            N_ph = Orb_1p1h.N[P,J+1]
            N_ph == 0 && continue
            Buffer = Vector{Float64}(undef,N_ph)
            read!(Import_File,Buffer)
            E_TDA[P,J+1] .= Buffer
        end
    end

    println("\tTDA solution binaries have been succesufully imported ...")

    return X_TDA, E_TDA
end

function TDA_occupation_factors(Params::Parameters,Orb::Vector{Orb1B},Orb_1p1h::Orb2B_1p1h,Orb_NN::Orb2B,Rho::O1B,Sigma::O2B,J::Int64,P::Int64)
    # Evaluate the exact transition kernel of TDA amplitudes X for given J & P ... (cf. TDA_rM ... analog of RPA_transition_kernel)
        # Exact: Gamma_ab(nu) = sum_ph D[ab,ph] * X[ph,nu] ... transition density on the orbital pairs ab = (a,b,t) ...
        #        without Sigma ... ab = 1p-1h pairs (+ pairs coupled by a non-diagonal Rho), for diagonal Rho D[ph,ph] = (-1)^J * rho_h * (1 - rho_p) ,
        #        with Sigma (Reference_TBCF = true) ... all pairs ab of the multipole, including the Sigma contributions (TDA_transition_kernel) ...
        # Norm:  f_N = n_h - n_p ... metric of the TDA amplitudes, i.e. sum_ph f_N * |X|^2 = 1 ...
    N_ph = Orb_1p1h.N[P,J+1]

    Pairs, D = TDA_transition_kernel(Params,J,P,Orb,Orb_1p1h,Orb_NN,Rho,Sigma)

    f_N = Vector{Float64}(undef,N_ph)

    @inbounds for (ph, Orb_ph) in enumerate(Orb_1p1h.Ind[P,J+1])
        a_p, a_h, t_ph = Orb_ph.a, Orb_ph.b, Orb_ph.T

        n_p, n_h = 0.0, 0.0

        if t_ph == -1
            n_p, n_h = Orb[a_p].pO, Orb[a_h].pO
        elseif t_ph == 1
            n_p, n_h = Orb[a_p].nO, Orb[a_h].nO
        end

        f_N[ph] = n_h - n_p
    end

    return Pairs, D, f_N
end

function TDA_transition_densities(Params::Parameters;JP::String,nu_list::Vector{Int64},File_Name::String = "Phonon_Densities")
    # Initialize the angular momentum algebra ...
    wigner_init_float(75, "Jmax", 9)

    # Local angular & phase factor function ...
    @inline function YJ_reduced(Orb::Vector{Orb1B},a::Int64,b::Int64,J::Int64)
        l_a, j_a = Orb[a].l, Orb[a].j
        l_b, j_b = Orb[b].l, Orb[b].j

        if isodd(l_a + l_b + J)
            return 0.0
        end

        return phase(J + div(j_a - 1,2)) * sqrt(Float64((j_a + 1) * (j_b + 1))) * cg(j_a,j_b,2*J,1,-1,0) / sqrt(4.0 * pi)
    end

    # Read parameters ...
    A = Float64(Params.Calc.A)
    Z = Float64(Params.Calc.Z)
    hw = Params.Calc.hw
    N_max = Params.Calc.Nmax
    a_max = div((N_max+1)*(N_max+2),2)

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
    P_char = string(JP[end])
    J = parse(Int64,JP[1:end-1])

    # Check input formating ...
    if (P_char != "+") && (P_char != "-")
        error("Invalid JP = \"$JP\"; use J+ or J-.")
    end

    # Set the internal value of parity P ...
    P = (P_char == "-") ? 2 : 1

    # Set the path to the TDA input/output directory ...
    IO_TDA = "io/" * Params.Calc.Path * "/" * Params.Calc.TDA.Name

    # Set the path to the reference input directory ... (the TDA does not export any ground-state data)
    IO_ref = "io/" * Params.Calc.Path * "/" * Params.Calc.TDA.Reference

    # Import reference single-particle orbitals ...
    Orb = orbitals_import(Params,IO_ref * "/bin/orb.bin")

    # Construct 2-body 1p-1h orbitals ... (as in TDA_solver_import)
    Orb_1p1h = orbitals_1p1h_make(Params,Orb;Tol_par = Params.Calc.TDA.Tol_par,Tol_hol = Params.Calc.TDA.Tol_hol)

    # Import transformation matrix mapping LHO & reference basis ...
    C = O1b_import(Params,Orb,IO_ref * "/bin/C.bin")

    # Import reference OBDM ... reference basis ...
    Rho = O1b_import(Params,Orb,IO_ref * "/bin/rho.bin")

    # If used, import reference 2-body correlation function ...
    if Params.Calc.TDA.Reference_TBCF == true
        println("\nImporting reference 2-body correlation function ...")
        Sigma, Orb_NN = O2b_import(Params,Orb,IO_ref * "/bin/sigma.bin",Make_Orb_NN = true)
    else
        Sigma = O2b_initialize_empty()
        Orb_NN = O2b_initialize(Params,Orb,Make_Orb_NN = true)[2]
    end

    # Import TDA solutions ...
    X_TDA, E_TDA = TDA_import_binary(Params)

    # Determine the number of phonon levels & the total number of phonons ...
    N_phonon = length(nu_list)

    # Evaluate the exact transition kernel of TDA amplitudes X ...
    Pairs, D, f_N = TDA_occupation_factors(Params,Orb,Orb_1p1h,Orb_NN,Rho,Sigma,J,P)
    N_ab = length(Pairs)

    # Transition densities of given phonon levels on the orbital pairs ...
    Gamma = D * X_TDA[P,J+1][:,nu_list]

    # Precalculate the reduced matrix elements of Electrogmanetic 1-body transition operators ...
        # Evaluate the reference charge radius chR ... (as in TDA_solver)
    chR2 = rho_chR2(Params,Orb,C,Rho)
        # Initialize the 1-body Electromagnetic transition operators ...
    TrOp = Tr1b_initialize(Params,Orb,chR2)
         # Transform the 1-body transition operators to the reference basis ...
    TrOp = Tr1b_transformation(Params,Orb,C,TrOp)

    # Precalculate the matrix elements of 1-body transition operators ...
        # Initialize the matrix elements for given phonon levels ...
    pM, nM = zeros(Float64,N_phonon), zeros(Float64,N_phonon)
    isM, ivM = zeros(Float64,N_phonon), zeros(Float64,N_phonon)
        # Calculate the matrix elements for given phonon levels ...
    if (J,P) in [(0,1),(2,1),(3,2)]
        @inbounds for (nu_ind, nu) in enumerate(nu_list)
            pMSum, nMSum, isMSum, ivMSum = 0.0, 0.0, 0.0, 0.0

            @inbounds for ab in 1:N_ab
                Orb_ab = Pairs[ab]
                a_a, a_b, t_ab = Orb_ab.a, Orb_ab.b, Orb_ab.T

                MME = Gamma[ab,nu_ind] / sqrt(Float64(2*J + 1))

                if t_ab == -1
                    if J == 0 && P == 1
                        MME = TrOp.E0.p[a_a,a_b] * MME

                    elseif J == 2 && P == 1
                        MME = TrOp.E2.p[a_a,a_b] * MME

                    elseif J == 3 && P == 2
                        MME = TrOp.E3.p[a_a,a_b] * MME
                    end

                    pMSum += MME
                    isMSum += 0.5 * MME
                    ivMSum += 0.5 * MME

                elseif t_ab == 1
                    if J == 0 && P == 1
                        MME = TrOp.E0.n[a_a,a_b] * MME

                    elseif J == 2 && P == 1
                        MME = TrOp.E2.n[a_a,a_b] * MME

                    elseif J == 3 && P == 2
                        MME = TrOp.E3.n[a_a,a_b] * MME
                    end

                    nMSum += MME
                    isMSum += 0.5 * MME
                    ivMSum -= 0.5 * MME
                end
            end

            pM[nu_ind] = pMSum
            nM[nu_ind] = nMSum
            isM[nu_ind] = isMSum
            ivM[nu_ind] = ivMSum
        end

    elseif (J,P) == (1,2)
        @inbounds for (nu_ind, nu) in enumerate(nu_list)
            pMSum, nMSum, isMSum, ivMSum = 0.0, 0.0, 0.0, 0.0

            @inbounds for ab in 1:N_ab
                Orb_ab = Pairs[ab]
                a_a, a_b, t_ab = Orb_ab.a, Orb_ab.b, Orb_ab.T

                Amp = Gamma[ab,nu_ind] / sqrt(Float64(2*J + 1))

                if t_ab == -1
                    pMME = TrOp.E1.p[a_a,a_b] * Amp
                    isMME = 0.5 * (TrOp.E1_C.p[a_a,a_b] + TrOp.E1_sC.p[a_a,a_b]) * Amp
                    ivMME = TrOp.E1.p[a_a,a_b] * Amp * (A - Z) / (A)

                    pMSum += pMME
                    isMSum += isMME
                    ivMSum += ivMME

                elseif t_ab == 1
                    nMME = TrOp.E1.n[a_a,a_b] * Amp
                    isMME = 0.5 * (TrOp.E1_C.n[a_a,a_b] + TrOp.E1_sC.n[a_a,a_b]) * Amp
                    ivMME = - TrOp.E1.n[a_a,a_b] * Amp * (Z) / (A)

                    nMSum += nMME
                    isMSum += isMME
                    ivMSum += ivMME
                end
            end

            pM[nu_ind] = pMSum
            nM[nu_ind] = nMSum
            isM[nu_ind] = isMSum
            ivM[nu_ind] = ivMSum
        end
    end

    # Initialize the radial grid ...
    r_grid = collect(range(r_min, stop = r_max, length = N_grid))

    # Initialize the radial grid for single-particle orbitals ...
    pOrb_grid = Matrix{Float64}(undef,a_max,N_grid)
    nOrb_grid = Matrix{Float64}(undef,a_max,N_grid)

    # Calculate the radial representation of the reference basis orbitals ...
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

                pPsi = C.p[k,a] * Psi_rad_LHO(r,n_k,l_k,nu_proton)
                nPsi = C.n[k,a] * Psi_rad_LHO(r,n_k,l_k,nu_neutron)

                pSum += pPsi
                nSum += nPsi
            end

            pOrb_grid[a,i] = pSum
            nOrb_grid[a,i] = nSum
        end
    end

    # Precalculate the angular & phase factors of the orbital pairs, i.e. <a||Y_J||b> ... same convention as in Tr1b_EX_initialize (zero for unnatural parity) ...
    Amp_ab = Vector{Float64}(undef,N_ab)

    @inbounds for ab in 1:N_ab
        Amp_ab[ab] = YJ_reduced(Orb,Pairs[ab].a,Pairs[ab].b,J)
    end

    # Initialize radial grids for given phonon levels ...
    pRho_nu = Matrix{Float64}(undef,N_phonon,N_grid)
    nRho_nu = Matrix{Float64}(undef,N_phonon,N_grid)

    # For given phonon levels evaluate their radial representation ...
    @inbounds Threads.@threads for i in 1:N_grid
        r = r_grid[i]
        @inbounds for (nu_ind, nu) in enumerate(nu_list)
            pSum, nSum = 0.0, 0.0

            Gauge = 1.0

            if ((J,P) in [(0,1),(1,2),(2,1),(3,2)]) == true && pM[nu_ind] != 0.0
                Gauge = pM[nu_ind] / abs(pM[nu_ind])
            end

            @inbounds for ab in 1:N_ab
                Orb_ab = Pairs[ab]
                a_a, a_b, t_ab = Orb_ab.a, Orb_ab.b, Orb_ab.T

                if t_ab == -1
                    pRho = Gauge * Gamma[ab,nu_ind] * Amp_ab[ab] * pOrb_grid[a_a,i] * pOrb_grid[a_b,i]
                    pSum += pRho
                elseif t_ab == 1
                    nRho = Gauge * Gamma[ab,nu_ind] * Amp_ab[ab] * nOrb_grid[a_a,i] * nOrb_grid[a_b,i]
                    nSum += nRho
                end
            end
            Amp = r^2 / Float64(2*J + 1)
            pRho_nu[nu_ind,i] = Amp * pSum
            nRho_nu[nu_ind,i] = Amp * nSum
        end
    end

    # Perform the export of indiviual TDA radial phonon densities ...
    @inbounds for (nu_ind, nu) in enumerate(nu_list)
        e = E_TDA[P,J+1][nu]
        sE = string(round(e,digits=3))

        Output_File = IO_TDA * "/densities/TDA_" * File_Name * "_JP$(JP)_nu$(nu)_E$(sE).dat"
        open(Output_File, "w") do Write_File
            @printf(Write_File, "%-20s %-20s %-20s %-20s %-20s\n", "r", "rho_p", "rho_n", "rho_is", "rho_iv")
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
    phonon_weights!(pM)
    phonon_weights!(nM)
    phonon_weights!(isM)
    phonon_weights!(ivM)

    # Perform the export of of averaged TDA transition radial densities ...
    Output_File = IO_TDA * "/densities/TDA_" * File_Name * "_JP$(JP)_Averaged.dat"
    open(Output_File, "w") do Write_File
        @printf(Write_File, "%-20s %-20s %-20s %-20s %-20s\n", "r", "rho_p", "rho_n", "rho_is", "rho_iv")

        @inbounds for i in 1:N_grid
            r = r_grid[i]
            pRho, nRho, isRho, ivRho = 0.0, 0.0, 0.0, 0.0

            @inbounds for nu_ind in 1:N_phonon
                prho, nrho = pRho_nu[nu_ind,i], nRho_nu[nu_ind,i]
                pRho += pM[nu_ind] * prho
                nRho += nM[nu_ind] * nrho
                isRho += 0.5 * isM[nu_ind] * (prho + nrho)
                ivRho += 0.5 * ivM[nu_ind] * (prho - nrho)
            end
            @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f %-20.8f\n", r, pRho, nRho, isRho, ivRho)
        end
    end

    display("Export of averaged phonon radial transitions densities successfuly completed ...")

    return
end

function TDA_transition_operator_densities(Params::Parameters;JP::String,nu_list::Vector{Int64},File_Name::String = "Phonon_Transition_Operator_Densities")
    # Initialize the angular momentum algebra ...
    wigner_init_float(75, "Jmax", 9)

    # Read parameters ...
    A = Float64(Params.Calc.A)
    Z = Float64(Params.Calc.Z)
    hw = Params.Calc.hw
    N_max = Params.Calc.Nmax
    a_max = div((N_max+1)*(N_max+2),2)

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
    P_char = string(JP[end])
    J = parse(Int64,JP[1:end-1])

    # Check input formating ...
    if (P_char != "+") && (P_char != "-")
        error("Invalid JP = \"$JP\"; use J+ or J-.")
    end

    # Set internal value of parity P ...
    P = (P_char == "-") ? 2 : 1

    # Check if the given JP is supported for transition density calculations ...
    if ((J,P) in [(0,1),(1,2),(2,1),(3,2)]) == false
        error("The given values of J & P are not supported for radial transition operator density calculations ...")
    end

    # Set the path to the TDA input/output directory ...
    IO_TDA = "io/" * Params.Calc.Path * "/" * Params.Calc.TDA.Name

    # Set the path to the reference input directory ... (the TDA does not export any ground-state data)
    IO_ref = "io/" * Params.Calc.Path * "/" * Params.Calc.TDA.Reference

    # Import reference single-particle orbitals ...
    Orb = orbitals_import(Params,IO_ref * "/bin/orb.bin")

    # Construct 2-body 1p-1h orbitals ... (as in TDA_solver_import)
    Orb_1p1h = orbitals_1p1h_make(Params,Orb;Tol_par = Params.Calc.TDA.Tol_par,Tol_hol = Params.Calc.TDA.Tol_hol)

    # Import transformation matrix mapping LHO & reference basis ...
    C = O1b_import(Params,Orb,IO_ref * "/bin/C.bin")

    # Import reference OBDM ... reference basis ...
    Rho = O1b_import(Params,Orb,IO_ref * "/bin/rho.bin")

    # If used, import reference 2-body correlation function ...
    if Params.Calc.TDA.Reference_TBCF == true
        println("\nImporting reference 2-body correlation function ...")
        Sigma, Orb_NN = O2b_import(Params,Orb,IO_ref * "/bin/sigma.bin",Make_Orb_NN = true)
    else
        Sigma = O2b_initialize_empty()
        Orb_NN = O2b_initialize(Params,Orb,Make_Orb_NN = true)[2]
    end

    # Import TDA solutions ...
    X_TDA, E_TDA = TDA_import_binary(Params)

    # Determine the number of phonon levels & the total number of phonons ...
    N_phonon = length(nu_list)

    # Evaluate the exact transition kernel of TDA amplitudes X ...
    Pairs, D, f_N = TDA_occupation_factors(Params,Orb,Orb_1p1h,Orb_NN,Rho,Sigma,J,P)
    N_ab = length(Pairs)

    # Transition densities of given phonon levels on the orbital pairs ...
    Gamma = D * X_TDA[P,J+1][:,nu_list]

    # Precalculate the reduced matrix elements of Electrogmanetic 1-body transition operators ...
        # Evaluate the reference charge radius chR ... (as in TDA_solver)
    chR2 = rho_chR2(Params,Orb,C,Rho)
        # Iniztialize the 1-body Electromagnetic transition operators ...
    TrOp = Tr1b_initialize(Params,Orb,chR2)
         # Transform the 1-body transition operators to the reference basis ...
    TrOp = Tr1b_transformation(Params,Orb,C,TrOp)

    # Precalculate the matrix elements of 1-body transition operators ...
        # Initialize the matrix elements for given phonon levels ...
    pM, nM = zeros(Float64,N_phonon), zeros(Float64,N_phonon)
    isM, ivM = zeros(Float64,N_phonon), zeros(Float64,N_phonon)
        # Calculate the matrix elements for given phonon levels ...
    if (J,P) in [(0,1),(2,1),(3,2)]
        @inbounds for (nu_ind, nu) in enumerate(nu_list)
            pMSum, nMSum, isMSum, ivMSum = 0.0, 0.0, 0.0, 0.0

            @inbounds for ab in 1:N_ab
                Orb_ab = Pairs[ab]
                a_a, a_b, t_ab = Orb_ab.a, Orb_ab.b, Orb_ab.T

                MME = Gamma[ab,nu_ind] / sqrt(Float64(2*J + 1))

                if t_ab == -1
                    if J == 0 && P == 1
                        MME = TrOp.E0.p[a_a,a_b] * MME

                    elseif J == 2 && P == 1
                        MME = TrOp.E2.p[a_a,a_b] * MME

                    elseif J == 3 && P == 2
                        MME = TrOp.E3.p[a_a,a_b] * MME
                    end

                    pMSum += MME
                    isMSum += 0.5 * MME
                    ivMSum += 0.5 * MME

                elseif t_ab == 1
                    if J == 0 && P == 1
                        MME = TrOp.E0.n[a_a,a_b] * MME

                    elseif J == 2 && P == 1
                        MME = TrOp.E2.n[a_a,a_b] * MME

                    elseif J == 3 && P == 2
                        MME = TrOp.E3.n[a_a,a_b] * MME
                    end

                    nMSum += MME
                    isMSum += 0.5 * MME
                    ivMSum -= 0.5 * MME
                end
            end

            pM[nu_ind] = pMSum
            nM[nu_ind] = nMSum
            isM[nu_ind] = isMSum
            ivM[nu_ind] = ivMSum
        end

    elseif (J,P) == (1,2)
        @inbounds for (nu_ind, nu) in enumerate(nu_list)
            pMSum, nMSum, isMSum, ivMSum = 0.0, 0.0, 0.0, 0.0

            @inbounds for ab in 1:N_ab
                Orb_ab = Pairs[ab]
                a_a, a_b, t_ab = Orb_ab.a, Orb_ab.b, Orb_ab.T

                Amp = Gamma[ab,nu_ind] / sqrt(Float64(2*J + 1))

                if t_ab == -1
                    pMME = TrOp.E1.p[a_a,a_b] * Amp
                    isMME = 0.5 * (TrOp.E1_C.p[a_a,a_b] + TrOp.E1_sC.p[a_a,a_b]) * Amp
                    ivMME = TrOp.E1.p[a_a,a_b] * Amp * (A - Z) / (A)

                    pMSum += pMME
                    isMSum += isMME
                    ivMSum += ivMME

                elseif t_ab == 1
                    nMME = TrOp.E1.n[a_a,a_b] * Amp
                    isMME = 0.5 * (TrOp.E1_C.n[a_a,a_b] + TrOp.E1_sC.n[a_a,a_b]) * Amp
                    ivMME = - TrOp.E1.n[a_a,a_b] * Amp * (Z) / (A)

                    nMSum += nMME
                    isMSum += isMME
                    ivMSum += ivMME
                end
            end

            pM[nu_ind] = pMSum
            nM[nu_ind] = nMSum
            isM[nu_ind] = isMSum
            ivM[nu_ind] = ivMSum
        end
    end

    # Initialize the radial grid ...
    r_grid = collect(range(r_min, stop = r_max, length = N_grid))

    # Initialize the radial grid for single-particle orbitals ...
    pOrb_grid = Matrix{Float64}(undef,a_max,N_grid)
    nOrb_grid = Matrix{Float64}(undef,a_max,N_grid)

    # Calculate the radial representation of the reference basis orbitals ...
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

                pPsi = C.p[k,a] * Psi_rad_LHO(r,n_k,l_k,nu_proton)
                nPsi = C.n[k,a] * Psi_rad_LHO(r,n_k,l_k,nu_neutron)

                pSum += pPsi
                nSum += nPsi
            end

            pOrb_grid[a,i] = pSum
            nOrb_grid[a,i] = nSum
        end
    end

    # Initialize the radial representation of 1-body transition operator matrix elements ...
    pM_grid = zeros(Float64,a_max,a_max,N_grid)
    nM_grid = zeros(Float64,a_max,a_max,N_grid)

    # Calculate the radial representation of 1-body transition operator matrix elements ...
    println("Calculating the radial representation of 1-body transition operator matrix elements ...")

    @inbounds Threads.@threads for a in 1:a_max
        l_a, j_a = Orb[a].l, Orb[a].j
        @views pR_a = pOrb_grid[a,:]
        @views nR_a = nOrb_grid[a,:]

        @inbounds for b in 1:a
            l_b, j_b = Orb[b].l, Orb[b].j
            @views pR_b = pOrb_grid[b,:]
            @views nR_b = nOrb_grid[b,:]

            if rem(l_a + l_b + P + 1,2) != 0  || abs(j_b - 2*J) > j_a || j_a > (j_b + 2*J)
                continue
            end

            # Angular & phase factor <a||Y_J||b> ... same convention as in Tr1b_EX_initialize, i.e. TrOp.EX[a,b] = <a||Y_J||b> * <a|r^X|b> (r^2 for E0) ...
            Amp_ab = phase(J + div(j_a - 1,2)) * sqrt(Float64((j_a + 1) * (j_b + 1))) * cg(j_a,j_b,2*J,1,-1,0) / sqrt(4.0 * pi)

            @inbounds for i in 1:N_grid
                r = r_grid[i]

                if J == 0 && P == 1
                    pM_grid[a,b,i] = Amp_ab * r^2 * pR_a[i] * pR_b[i]
                    nM_grid[a,b,i] = Amp_ab * r^2 * nR_a[i] * nR_b[i]

                elseif J == 1 && P == 2
                    pM_grid[a,b,i] = Amp_ab * r * pR_a[i] * pR_b[i]
                    nM_grid[a,b,i] = Amp_ab * r * nR_a[i] * nR_b[i]

                elseif J == 2 && P == 1
                    pM_grid[a,b,i] = Amp_ab * r^2 * pR_a[i] * pR_b[i]
                    nM_grid[a,b,i] = Amp_ab * r^2 * nR_a[i] * nR_b[i]

                elseif J == 3 && P == 2
                    pM_grid[a,b,i] = Amp_ab * r^3 * pR_a[i] * pR_b[i]
                    nM_grid[a,b,i] = Amp_ab * r^3 * nR_a[i] * nR_b[i]

                end

                if a != b
                    Phase = phase(div(j_b - j_a,2))
                    pM_grid[b,a,i] = Phase * pM_grid[a,b,i]
                    nM_grid[b,a,i] = Phase * nM_grid[a,b,i]
                end

            end

        end
    end

    println("\tCalculation of the radial representation of 1-body transition operator matrix elements done ...")

    # Initialize radial grids for given phonon levels ...
    pRho_nu = Matrix{Float64}(undef,N_phonon,N_grid)
    nRho_nu = Matrix{Float64}(undef,N_phonon,N_grid)

    # For given phonon levels evaluate their radial representation ...
    @inbounds Threads.@threads for i in 1:N_grid
        r = r_grid[i]
        @inbounds for (nu_ind, nu) in enumerate(nu_list)
            pSum, nSum = 0.0, 0.0

            Gauge = 1.0

            if ((J,P) in [(0,1),(1,2),(2,1),(3,2)]) == true && pM[nu_ind] != 0.0
                Gauge = pM[nu_ind] / abs(pM[nu_ind])
            end

            @inbounds for ab in 1:N_ab
                Orb_ab = Pairs[ab]
                a_a, a_b, t_ab = Orb_ab.a, Orb_ab.b, Orb_ab.T

                if t_ab == -1
                    pm = pM_grid[a_a,a_b,i]
                    pRho = Gauge * Gamma[ab,nu_ind] * pm
                    pSum += pRho
                elseif t_ab == 1
                    nm = nM_grid[a_a,a_b,i]
                    nRho = Gauge * Gamma[ab,nu_ind] * nm
                    nSum += nRho
                end
            end

            Amp = r^2 / Float64(2*J + 1)

            pRho_nu[nu_ind,i] = Amp * pSum
            nRho_nu[nu_ind,i] = Amp * nSum
        end
    end

    # Perform the export of indiviual TDA radial phonon densities ...
    @inbounds for (nu_ind, nu) in enumerate(nu_list)
        e = E_TDA[P,J+1][nu]
        sE = string(round(e,digits=3))

        # Case of TDA densities ...
        Output_File = IO_TDA * "/densities/TDA_" * File_Name * "_JP$(JP)_nu$(nu)_E$(sE).dat"
        open(Output_File, "w") do Write_File
            @printf(Write_File, "%-20s %-20s %-20s %-20s %-20s\n", "r", "pM", "nM", "isM", "ivM")
            @inbounds for i in 1:N_grid
                r, pRho, nRho = r_grid[i], pRho_nu[nu_ind,i], nRho_nu[nu_ind,i]
                isRho = 0.5 * (pRho + nRho)
                ivRho = 0.5 * (pRho - nRho)
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f %-20.8f\n", r, pRho, nRho, isRho, ivRho)
            end
        end

    end

    display("Export of individual phonon radial transition densities successfuly completed ...")

    # Renormalize M ... to reasonable scale averaged densities ...
    phonon_weights!(pM)
    phonon_weights!(nM)
    phonon_weights!(isM)
    phonon_weights!(ivM)

    # Perform the export of of averaged TDA transition matrix radial densities ...
    Output_File = IO_TDA * "/densities/TDA_" * File_Name * "_JP$(JP)_Averaged.dat"
    open(Output_File, "w") do Write_File
        @printf(Write_File, "%-20s %-20s %-20s %-20s %-20s\n", "r", "rho_p", "rho_n", "rho_is", "rho_iv")
        @inbounds for i in 1:N_grid
            r = r_grid[i]
            pRho, nRho, isRho, ivRho = 0.0, 0.0, 0.0, 0.0
            @inbounds for nu_ind in 1:N_phonon
                prho, nrho = pRho_nu[nu_ind,i], nRho_nu[nu_ind,i]
                pRho += pM[nu_ind] * prho
                nRho += nM[nu_ind] * nrho
                isRho += 0.5 * isM[nu_ind] * (prho + nrho)
                ivRho += 0.5 * ivM[nu_ind] * (prho - nrho)
            end
            @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f %-20.8f\n", r, pRho, nRho, isRho, ivRho)
        end
    end

    display("Export of averaged phonon radial transitions densities successfuly completed ...")

    return
end

function TDA_transition_analysis(Params::Parameters;JP::String,nu_list::Vector{Int64},File_Name::String = "Phonon_Transtion_Analysis", Cut::Float64 = 0.1)
    # Initialize the angular momentum algebra ...
    wigner_init_float(75, "Jmax", 9)

    # Read the values of J & P ...
    P_char = string(JP[end])
    J = parse(Int64,JP[1:end-1])

    # Check input formating ...
    if (P_char != "+") && (P_char != "-")
        error("Invalid JP = \"$JP\"; use J+ or J-.")
    end

    # Set internal value of parity P ...
    P = (P_char == "-") ? 2 : 1

    # Set the path to the TDA input/output directory ...
    IO_TDA = "io/" * Params.Calc.Path * "/" * Params.Calc.TDA.Name

    # Set the path to the reference input directory ... (the TDA does not export any ground-state data)
    IO_ref = "io/" * Params.Calc.Path * "/" * Params.Calc.TDA.Reference

    # Import reference single-particle orbitals ...
    Orb = orbitals_import(Params,IO_ref * "/bin/orb.bin")

    # Construct 2-body 1p-1h orbitals ... (as in TDA_solver_import)
    Orb_1p1h = orbitals_1p1h_make(Params,Orb;Tol_par = Params.Calc.TDA.Tol_par,Tol_hol = Params.Calc.TDA.Tol_hol)

    # Import transformation matrix mapping LHO & reference basis ...
    C = O1b_import(Params,Orb,IO_ref * "/bin/C.bin")

    # Import reference OBDM ... reference basis ...
    Rho = O1b_import(Params,Orb,IO_ref * "/bin/rho.bin")

    # If used, import reference 2-body correlation function ...
    if Params.Calc.TDA.Reference_TBCF == true
        println("\nImporting reference 2-body correlation function ...")
        Sigma, Orb_NN = O2b_import(Params,Orb,IO_ref * "/bin/sigma.bin",Make_Orb_NN = true)
    else
        Sigma = O2b_initialize_empty()
        Orb_NN = O2b_initialize(Params,Orb,Make_Orb_NN = true)[2]
    end

    # Import TDA solutions ...
    X_TDA, E_TDA = TDA_import_binary(Params)

    # Make directory for export of information on structure of phonon levels ...
    if isdir(IO_TDA * "/levels") == false
        mkdir(IO_TDA * "/levels")
    end

    # Determine the number of phonon levels & the total number of phonons ...
    N_phonon = length(nu_list)
    N_ph = Orb_1p1h.N[P,J+1]

    # Evaluate the exact transition kernel of TDA amplitudes X ...
    Pairs, D, f_N = TDA_occupation_factors(Params,Orb,Orb_1p1h,Orb_NN,Rho,Sigma,J,P)
    N_ab = length(Pairs)

    # Precalculate the reduced matrix elements of Electrogmanetic 1-body transition operators ...
        # Evaluate the reference charge radius chR ... (as in TDA_solver)
    chR2 = rho_chR2(Params,Orb,C,Rho)
        # Iniztialize the 1-body Electromagnetic transition operators ...
    TrOp = Tr1b_initialize(Params,Orb,chR2)
         # Transform the 1-body transition operators to the reference basis ...
    TrOp = Tr1b_transformation(Params,Orb,C,TrOp)

    nu_count = zeros(Int64,N_phonon)

    # Note: the amplitudes renormalized by sqrt(f_N) fulfil sum_ph |X|^2 = 1 & are used for the Cut & the export,
    #       while the 1p-1h contributions M_ph to the reduced transition matrix element use the exact transition kernel D,
    #       i.e. M_ph = sum_ab M_ab * D[ab,ph] * X_ph over the pairs ab of the same isospin as ph & M_ph(-t) over the opposite isospin (pn part of Sigma),
    #       sum over all ph of M_ph & M_ph(-t) of the proton (neutron) operator = rM of TDA_rM, including Sigma for Reference_TBCF = true ...

    # Iterate through the phonon solutions & determine the most dominant ph components of given phonon levels ...
    @inbounds for (nu_ind, nu) in enumerate(nu_list)
        @inbounds for ph in 1:N_ph
            x_TDA = sqrt(f_N[ph]) * X_TDA[P,J+1][ph,nu]

            if abs2(x_TDA) > Cut
                nu_count[nu_ind] += 1
            end
        end
    end

    # EJ operator of the multipole (zero contributions for the other multipoles) ...
    Op = (J,P) == (0,1) ? TrOp.E0 : (J,P) == (1,2) ? TrOp.E1 : (J,P) == (2,1) ? TrOp.E2 : (J,P) == (3,2) ? TrOp.E3 : nothing

    nu_compo = Vector{Vector{Tuple{Int64,Int64,Int64,Float64,Float64,Float64}}}(undef,N_phonon)

    @inbounds for (nu_ind, nu) in enumerate(nu_list)
        nu_compo[nu_ind] = Vector{Tuple{Int64,Int64,Int64,Float64,Float64,Float64}}(undef,nu_count[nu_ind])

        count = 0

        @inbounds for ph in 1:N_ph
            Orb_ph = Orb_1p1h.Ind[P,J+1][ph]
            a_p, a_h, t_ph = Orb_ph.a, Orb_ph.b, Orb_ph.T
            x_TDA = sqrt(f_N[ph]) * X_TDA[P,J+1][ph,nu]

            if abs2(x_TDA) > Cut
                count += 1

                # Contributions to rM of the operator of the same (M_1) & of the opposite (M_2, pn part of Sigma only) isospin as the ph pair ...
                M_1, M_2 = 0.0, 0.0

                @inbounds for ab in 1:N_ab
                    (Op === nothing || D[ab,ph] == 0.0) && continue

                    Orb_ab = Pairs[ab]
                    a_a, a_b, t_ab = Orb_ab.a, Orb_ab.b, Orb_ab.T

                    Amp_M = D[ab,ph] * X_TDA[P,J+1][ph,nu] * (t_ab == -1 ? Op.p[a_a,a_b] : Op.n[a_a,a_b])

                    if t_ab == t_ph
                        M_1 += Amp_M
                    else
                        M_2 += Amp_M
                    end
                end

                nu_compo[nu_ind][count] = (a_p,a_h,t_ph,x_TDA,M_1,M_2)
            end

        end

    end

    # Perform the export of information on individual TDA phonon levels ...
    @inbounds for (nu_ind, nu) in enumerate(nu_list)
        e = E_TDA[P,J+1][nu]
        sE = string(round(e,digits=3))

        Phonon_TDA = nu_compo[nu_ind]
        Output_File = IO_TDA * "/levels/TDA_" * File_Name * "_JP$(JP)_nu$(nu)_E$(sE).dat"
        open(Output_File, "w") do Write_File
            @printf(Write_File, "%-8s %-8s %-8s %-20s %-20s %-20s\n", "a_p", "a_h", "T_ph", "X_ph", "M_ph", "M_ph(-t)")
            @inbounds for L in Phonon_TDA
                (a_p,a_h,t_ph,x_TDA,M_ph,M_ph_2) = L
                @printf(Write_File, "%-8d %-8d %-8d %-20.8f %-20.8f %-20.8f\n", a_p, a_h, t_ph, x_TDA, M_ph, M_ph_2)
            end
        end

    end

    display("Export of information on individual phonon levels successfuly completed ...")

    return
end

function TDA_transition_currents(Params::Parameters;JP::String,nu_list::Vector{Int64},File_Name::String = "Phonon_Currents",
                                 M::Int64=-1,xN_grid::Int64 = 40, zN_grid::Int64 = 40)
    # Note ... variables captured by the Threads.@threads closures below are assigned only once at the function level,
    #          otherwise Julia boxes them (Core.Box ... type-unstable access & a potential data race if ever written inside the threaded loops) ...

    # Initialize the angular momentum algebra ...
    wigner_init_float(75, "Jmax", 9)

    # Read parameters ...
    A = Float64(Params.Calc.A)
    Z = Float64(Params.Calc.Z)
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
    P_char = string(JP[end])
    J = parse(Int64,JP[1:end-1])

    # Check input formating ...
    if (P_char != "+") && (P_char != "-")
        error("Invalid JP = \"$JP\"; use J+ or J- ...")
    end

    # Set internal value of parity P ...
    P = (P_char == "-") ? 2 : 1

    # Set the value of projection M ... (M = -1 is the default for M = 0, keyword argument is not reassigned)
    M_proj = (M == -1) ? 0 : M

    # Set the path to the TDA input/output directory ...
    IO_TDA = "io/" * Params.Calc.Path * "/" * Params.Calc.TDA.Name

    # Set the path to the reference input directory ... (the TDA does not export any ground-state data)
    IO_ref = "io/" * Params.Calc.Path * "/" * Params.Calc.TDA.Reference

    # Import reference single-particle orbitals ...
    Orb = orbitals_import(Params,IO_ref * "/bin/orb.bin")

    # Construct 2-body 1p-1h orbitals ... (as in TDA_solver_import)
    Orb_1p1h = orbitals_1p1h_make(Params,Orb;Tol_par = Params.Calc.TDA.Tol_par,Tol_hol = Params.Calc.TDA.Tol_hol)

    # Import transformation matrix mapping LHO & reference basis ...
    C = O1b_import(Params,Orb,IO_ref * "/bin/C.bin")

    # Import reference OBDM ... reference basis ...
    Rho = O1b_import(Params,Orb,IO_ref * "/bin/rho.bin")

    # If used, import reference 2-body correlation function ...
    if Params.Calc.TDA.Reference_TBCF == true
        println("\nImporting reference 2-body correlation function ...")
        Sigma, Orb_NN = O2b_import(Params,Orb,IO_ref * "/bin/sigma.bin",Make_Orb_NN = true)
    else
        Sigma = O2b_initialize_empty()
        Orb_NN = O2b_initialize(Params,Orb,Make_Orb_NN = true)[2]
    end

    # Import TDA solutions ...
    X_TDA, E_TDA = TDA_import_binary(Params)

    # Determine the number of phonon levels & the total number of phonons ...
    N_phonon = length(nu_list)

    # Evaluate the exact transition kernel of TDA amplitudes X ...
    Pairs, D, f_N = TDA_occupation_factors(Params,Orb,Orb_1p1h,Orb_NN,Rho,Sigma,J,P)
    N_ab = length(Pairs)

    # Transition densities of given phonon levels on the orbital pairs ...
    Gamma = D * X_TDA[P,J+1][:,nu_list]

    # Precalculate the reduced matrix elements of Electrogmanetic 1-body transition operators ...
        # Evaluate the reference charge radius chR ... (as in TDA_solver)
    chR2 = rho_chR2(Params,Orb,C,Rho)
        # Iniztialize the 1-body Electromagnetic transition operators ...
    TrOp = Tr1b_initialize(Params,Orb,chR2)
         # Transform the 1-body transition operators to the reference basis ...
    TrOp = Tr1b_transformation(Params,Orb,C,TrOp)

    # Precalculate the matrix elements of 1-body transition operators ...
        # Initialize the matrix elements for given phonon levels ...
    pM, nM = zeros(Float64,N_phonon), zeros(Float64,N_phonon)
    isM, ivM = zeros(Float64,N_phonon), zeros(Float64,N_phonon)

        # Calculate the matrix elements for given phonon levels ...
    if (J,P) in [(0,1),(2,1),(3,2)]
        @inbounds for (nu_ind, nu) in enumerate(nu_list)
            pMSum, nMSum, isMSum, ivMSum = 0.0, 0.0, 0.0, 0.0

            @inbounds for ab in 1:N_ab
                Orb_ab = Pairs[ab]
                a_a, a_b, t_ab = Orb_ab.a, Orb_ab.b, Orb_ab.T

                MME = Gamma[ab,nu_ind] / sqrt(Float64(2*J + 1))

                if t_ab == -1
                    if J == 0 && P == 1
                        MME = TrOp.E0.p[a_a,a_b] * MME

                    elseif J == 2 && P == 1
                        MME = TrOp.E2.p[a_a,a_b] * MME

                    elseif J == 3 && P == 2
                        MME = TrOp.E3.p[a_a,a_b] * MME
                    end

                    pMSum += MME
                    isMSum += 0.5 * MME
                    ivMSum += 0.5 * MME

                elseif t_ab == 1
                    if J == 0 && P == 1
                        MME = TrOp.E0.n[a_a,a_b] * MME

                    elseif J == 2 && P == 1
                        MME = TrOp.E2.n[a_a,a_b] * MME

                    elseif J == 3 && P == 2
                        MME = TrOp.E3.n[a_a,a_b] * MME
                    end

                    nMSum += MME
                    isMSum += 0.5 * MME
                    ivMSum -= 0.5 * MME
                end
            end

            pM[nu_ind] = pMSum
            nM[nu_ind] = nMSum
            isM[nu_ind] = isMSum
            ivM[nu_ind] = ivMSum
        end

    elseif (J,P) == (1,2)
        @inbounds for (nu_ind, nu) in enumerate(nu_list)
            pMSum, nMSum, isMSum, ivMSum = 0.0, 0.0, 0.0, 0.0

            @inbounds for ab in 1:N_ab
                Orb_ab = Pairs[ab]
                a_a, a_b, t_ab = Orb_ab.a, Orb_ab.b, Orb_ab.T

                Amp = Gamma[ab,nu_ind] / sqrt(Float64(2*J + 1))

                if t_ab == -1
                    pMME = TrOp.E1.p[a_a,a_b] * Amp
                    isMME = 0.5 * (TrOp.E1_C.p[a_a,a_b] + TrOp.E1_sC.p[a_a,a_b]) * Amp
                    ivMME = TrOp.E1.p[a_a,a_b] * Amp * (A - Z) / (A)

                    pMSum += pMME
                    isMSum += isMME
                    ivMSum += ivMME

                elseif t_ab == 1
                    nMME = TrOp.E1.n[a_a,a_b] * Amp
                    isMME = 0.5 * (TrOp.E1_C.n[a_a,a_b] + TrOp.E1_sC.n[a_a,a_b]) * Amp
                    ivMME = - TrOp.E1.n[a_a,a_b] * Amp * (Z) / (A)

                    nMSum += nMME
                    isMSum += isMME
                    ivMSum += ivMME
                end
            end

            pM[nu_ind] = pMSum
            nM[nu_ind] = nMSum
            isM[nu_ind] = isMSum
            ivM[nu_ind] = ivMSum
        end
    end

    # Initialize the radial & angular grid ...
        # Case of the radial grid ...
    r_grid = collect(range(r_min, stop = r_max, length = rN_grid))
        # Case of the angular theta grid ...
    theta_grid = collect(range(0.0, stop = 2.0*pi, length = tN_grid))

    # Initialize the radial grid for single-particle orbitals ...
        # Case of wavefunctions ...
    pR_grid = Matrix{Float64}(undef,a_max,rN_grid)
    nR_grid = Matrix{Float64}(undef,a_max,rN_grid)
        # Case of wavefunction derivatives ...
    pdR_grid = Matrix{Float64}(undef,a_max,rN_grid)
    ndR_grid = Matrix{Float64}(undef,a_max,rN_grid)

    # Calculate the radial representation of the reference basis orbitals ...
    @inbounds Threads.@threads for i in 1:rN_grid
        r = r_grid[i]
        @inbounds for a in 1:a_max
            l_a, j_a = Orb[a].l, Orb[a].j
            pSum, nSum = 0.0, 0.0

            @inbounds for k in 1:a_max
                n_k, l_k, j_k = Orb[k].n, Orb[k].l, Orb[k].j

                if l_a != l_k || j_a != j_k
                    continue
                end

                pPsi = C.p[k,a] * Psi_rad_LHO(r,n_k,l_k,nu_proton)
                nPsi = C.n[k,a] * Psi_rad_LHO(r,n_k,l_k,nu_neutron)

                pSum += pPsi
                nSum += nPsi
            end

            pR_grid[a,i] = pSum
            nR_grid[a,i] = nSum
        end
    end

    # Calculate the radial representation of the radial derivative of reference basis orbitals ...
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
        # Single orbital pair current maps ... (only the X, i.e. a b^(-1), contributions in the TDA)
    pJ_ab_conv_map = Array{Float64}(undef,3,N_ab,N_map)
    nJ_ab_conv_map = Array{Float64}(undef,3,N_ab,N_map)
    pJ_ab_spin_map = Array{Float64}(undef,3,N_ab,N_map)
    nJ_ab_spin_map = Array{Float64}(undef,3,N_ab,N_map)
        # Phonon current maps ...
    pJ_conv_map = Array{Float64}(undef,3,N_phonon,N_map)
    nJ_conv_map = Array{Float64}(undef,3,N_phonon,N_map)
    pJ_spin_map = Array{Float64}(undef,3,N_phonon,N_map)
    nJ_spin_map = Array{Float64}(undef,3,N_phonon,N_map)

    # Prepare the current map grid ...
        # Precalculate radial & angular grid step size ...
    dr = (r_max - r_min) / (rN_grid - 1)
    dtheta = 2.0 * pi / (tN_grid - 1)  # if theta_grid spans 0..2*pi inclusive
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

    # Evaluate the grid of the 1-body transition current matrix elements ...
    @time @inbounds Threads.@threads for N in 1:N_map
        n, m = i_map[1,N], i_map[2,N]
        r, theta = r_grid[n], theta_grid[m]
        g = 1.0

        @inbounds for ab in 1:N_ab
            Orb_ab = Pairs[ab]
            a_a, a_b, t_ab = Orb_ab.a, Orb_ab.b, Orb_ab.T

            l_a, j_a = Orb[a_a].l, Orb[a_a].j
            l_b, j_b = Orb[a_b].l, Orb[a_b].j

            Jx_ab_conv, Jy_ab_conv, Jz_ab_conv = 0.0, 0.0, 0.0
            Jx_ab_spin, Jy_ab_spin, Jz_ab_spin = 0.0, 0.0, 0.0

            R_a, R_b, dR_a, dR_b = 0.0, 0.0, 0.0, 0.0

            if t_ab == -1
                R_a, R_b = pR_grid[a_a,n], pR_grid[a_b,n]
                dR_a, dR_b = pdR_grid[a_a,n], pdR_grid[a_b,n]
                g = 0.5 * g_p
            elseif t_ab == 1
                R_a, R_b = nR_grid[a_a,n], nR_grid[a_b,n]
                dR_a, dR_b = ndR_grid[a_a,n], ndR_grid[a_b,n]
                g = 0.5 * g_n
            end

            @inbounds for m_a in -j_a:2:j_a
                @inbounds for m_b in -j_b:2:j_b

                    # Case of a b^(-1) contributions ... (the hole-particle contributions of RPA_transition_currents are absent in the TDA, Y = 0)
                    if (m_a + m_b) == 2*M_proj
                        Amp_ab = cg(j_a,j_b,2*J,m_a,m_b,2*M_proj) * phase(div(j_b - m_b,2))

                        (jx_ab_conv,jy_ab_conv,jz_ab_conv) = J1b_convective(r,theta,0.0,l_a,j_a,m_a,R_a,dR_a,l_b,j_b,-m_b,R_b,dR_b)
                        (jx_ab_spin,jy_ab_spin,jz_ab_spin) = g .* J1b_spin(r,theta,0.0,l_a,j_a,m_a,R_a,dR_a,l_b,j_b,-m_b,R_b,dR_b)

                        Jx_ab_conv += Amp_ab * real(jx_ab_conv)
                        Jy_ab_conv += Amp_ab * real(jy_ab_conv)
                        Jz_ab_conv += Amp_ab * real(jz_ab_conv)

                        Jx_ab_spin += Amp_ab * real(jx_ab_spin)
                        Jy_ab_spin += Amp_ab * real(jy_ab_spin)
                        Jz_ab_spin += Amp_ab * real(jz_ab_spin)
                    end
                end
            end

            if t_ab == -1
                pJ_ab_conv_map[1,ab,N] = Jx_ab_conv
                pJ_ab_conv_map[2,ab,N] = Jy_ab_conv
                pJ_ab_conv_map[3,ab,N] = Jz_ab_conv
                pJ_ab_spin_map[1,ab,N] = Jx_ab_spin
                pJ_ab_spin_map[2,ab,N] = Jy_ab_spin
                pJ_ab_spin_map[3,ab,N] = Jz_ab_spin

            elseif t_ab == 1
                nJ_ab_conv_map[1,ab,N] = Jx_ab_conv
                nJ_ab_conv_map[2,ab,N] = Jy_ab_conv
                nJ_ab_conv_map[3,ab,N] = Jz_ab_conv
                nJ_ab_spin_map[1,ab,N] = Jx_ab_spin
                nJ_ab_spin_map[2,ab,N] = Jy_ab_spin
                nJ_ab_spin_map[3,ab,N] = Jz_ab_spin
            end

        end

    end

    # Evaluate the grid of transition current matrix elements for given phonon levels ...
    @inbounds Threads.@threads for N in 1:N_map
        @inbounds for (nu_ind, nu) in enumerate(nu_list)
            pJxConvSum, pJyConvSum, pJzConvSum = 0.0, 0.0, 0.0
            nJxConvSum, nJyConvSum, nJzConvSum = 0.0, 0.0, 0.0
            pJxSpinSum, pJySpinSum, pJzSpinSum = 0.0, 0.0, 0.0
            nJxSpinSum, nJySpinSum, nJzSpinSum = 0.0, 0.0, 0.0

            Gauge = 1.0

            if ((J,P) in [(0,1),(1,2),(2,1),(3,2)]) == true && pM[nu_ind] != 0.0
                Gauge = pM[nu_ind] / abs(pM[nu_ind])
            end

            @inbounds for ab in 1:N_ab
                t_ab = Pairs[ab].T

                # Exact TDA amplitude of the orbital pair ... Gamma includes (-1)^J, removed here (cf. f_X * X in RPA_transition_currents) ...
                x_TDA = Gauge * phase(J) * Gamma[ab,nu_ind]

                if t_ab == -1
                    pJxConvSum += x_TDA * pJ_ab_conv_map[1,ab,N]
                    pJyConvSum += x_TDA * pJ_ab_conv_map[2,ab,N]
                    pJzConvSum += x_TDA * pJ_ab_conv_map[3,ab,N]

                    pJxSpinSum += x_TDA * pJ_ab_spin_map[1,ab,N]
                    pJySpinSum += x_TDA * pJ_ab_spin_map[2,ab,N]
                    pJzSpinSum += x_TDA * pJ_ab_spin_map[3,ab,N]

                elseif t_ab == 1
                    nJxConvSum += x_TDA * nJ_ab_conv_map[1,ab,N]
                    nJyConvSum += x_TDA * nJ_ab_conv_map[2,ab,N]
                    nJzConvSum += x_TDA * nJ_ab_conv_map[3,ab,N]

                    nJxSpinSum += x_TDA * nJ_ab_spin_map[1,ab,N]
                    nJySpinSum += x_TDA * nJ_ab_spin_map[2,ab,N]
                    nJzSpinSum += x_TDA * nJ_ab_spin_map[3,ab,N]
                end

            end
            pJ_conv_map[1,nu_ind,N] = pJxConvSum
            pJ_conv_map[2,nu_ind,N] = pJyConvSum
            pJ_conv_map[3,nu_ind,N] = pJzConvSum
            nJ_conv_map[1,nu_ind,N] = nJxConvSum
            nJ_conv_map[2,nu_ind,N] = nJyConvSum
            nJ_conv_map[3,nu_ind,N] = nJzConvSum

            pJ_spin_map[1,nu_ind,N] = pJxSpinSum
            pJ_spin_map[2,nu_ind,N] = pJySpinSum
            pJ_spin_map[3,nu_ind,N] = pJzSpinSum
            nJ_spin_map[1,nu_ind,N] = nJxSpinSum
            nJ_spin_map[2,nu_ind,N] = nJySpinSum
            nJ_spin_map[3,nu_ind,N] = nJzSpinSum
        end

    end

    # Scale the currents to correct physical units in [c] ... (J1b_convective = -2i j_conv in units hbar/m, i.e. factor hbar/2m as in RPA_transition_currents)
    pJ_conv_map .= pJ_conv_map .* 0.5 * hc / m_p
    nJ_conv_map .= nJ_conv_map .* 0.5 * hc / m_n
    pJ_spin_map .= pJ_spin_map .* hc / m_p
    nJ_spin_map .= nJ_spin_map .* hc / m_n

    # Perform the export of of individual TDA phonon level transition current maps ...
    @inbounds for (nu_ind, nu) in enumerate(nu_list)
        e = E_TDA[P,J+1][nu]
        sE = string(round(e,digits=3))

        # Case of convective transition current densities ...
        Output_File_Convective = IO_TDA * "/densities/TDA_Convective_" * File_Name * "_JP$(JP)_nu$(nu)_E$(sE).dat"
        open(Output_File_Convective, "w") do Write_File
            @printf(Write_File, "%-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s\n", "x", "z", "pJ_x", "pJ_z", "nJ_x", "nJ_z", "isJ_x", "isJ_z", "ivJ_x", "ivJ_z")
            @inbounds for N in 1:N_map
                r, theta = x_map[1,N], x_map[2,N]
                x = r * sin(theta)
                z = r * cos(theta)
                pJ_x = pJ_conv_map[1,nu_ind,N]
                pJ_z = pJ_conv_map[3,nu_ind,N]
                nJ_x = nJ_conv_map[1,nu_ind,N]
                nJ_z = nJ_conv_map[3,nu_ind,N]
                isJ_x = 0.5 * (pJ_x + nJ_x)
                isJ_z = 0.5 * (pJ_z + nJ_z)
                ivJ_x = 0.5 * (pJ_x - nJ_x)
                ivJ_z = 0.5 * (pJ_z - nJ_z)
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f\n", x, z, pJ_x, pJ_z, nJ_x, nJ_z, isJ_x, isJ_z, ivJ_x, ivJ_z)
            end
        end

         # Case of spin transition current densities ...
        Output_File_Spin = IO_TDA * "/densities/TDA_Spin_" * File_Name * "_JP$(JP)_nu$(nu)_E$(sE).dat"
        open(Output_File_Spin, "w") do Write_File
            @printf(Write_File, "%-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s\n", "x", "z", "pJ_x", "pJ_z", "nJ_x", "nJ_z", "isJ_x", "isJ_z", "ivJ_x", "ivJ_z")
            @inbounds for N in 1:N_map
                r, theta = x_map[1,N], x_map[2,N]
                x = r * sin(theta)
                z = r * cos(theta)
                pJ_x = pJ_spin_map[1,nu_ind,N]
                pJ_z = pJ_spin_map[3,nu_ind,N]
                nJ_x = nJ_spin_map[1,nu_ind,N]
                nJ_z = nJ_spin_map[3,nu_ind,N]
                isJ_x = 0.5 * (pJ_x + nJ_x)
                isJ_z = 0.5 * (pJ_z + nJ_z)
                ivJ_x = 0.5 * (pJ_x - nJ_x)
                ivJ_z = 0.5 * (pJ_z - nJ_z)
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f\n", x, z, pJ_x, pJ_z, nJ_x, nJ_z, isJ_x, isJ_z, ivJ_x, ivJ_z)
            end
        end

    end

    display("Export of individual phonon level transition current maps successfuly completed ...")

    if ((J,P) in [(0,1),(1,2),(2,1),(3,2)]) == false
        return
    end

    # Renormalize M ... to reasonable scale averaged current densities ...
    phonon_weights!(pM)
    phonon_weights!(nM)
    phonon_weights!(isM)
    phonon_weights!(ivM)

    # Perform the export of of averaged TDA transition current maps ...
        # Case of convective transition current densities ...
    Output_File_Convective = IO_TDA * "/densities/TDA_Convective_" * File_Name * "_JP$(JP)_Averaged.dat"
    open(Output_File_Convective, "w") do Write_File
        @printf(Write_File, "%-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s\n", "x", "z", "ppJ_x", "ppJ_z", "pnJ_x", "pnJ_z", "npJ_x", "npJ_z", "nnJ_x", "nnJ_z", "isJ_x", "isJ_z", "ivJ_x", "ivJ_z")
        @inbounds for N in 1:N_map
            r, theta = x_map[1,N], x_map[2,N]
            x = r * sin(theta)
            z = r * cos(theta)
            ppJ_x, pnJ_x, npJ_x, nnJ_x, isJ_x, ivJ_x = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ppJ_z, pnJ_z, npJ_z, nnJ_z, isJ_z, ivJ_z = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            @inbounds for nu_ind in 1:N_phonon
                pJ_x, pJ_z = pJ_conv_map[1,nu_ind,N], pJ_conv_map[3,nu_ind,N]
                nJ_x, nJ_z = nJ_conv_map[1,nu_ind,N], nJ_conv_map[3,nu_ind,N]

                pm, nm = pM[nu_ind], nM[nu_ind]
                ism, ivm = isM[nu_ind], ivM[nu_ind]

                ppJ_x += pm * pJ_x
                ppJ_z += pm * pJ_z
                pnJ_x += pm * nJ_x
                pnJ_z += pm * nJ_z
                npJ_x += nm * pJ_x
                nnJ_x += nm * nJ_x
                npJ_z += nm * pJ_z
                nnJ_z += nm * nJ_z
                isJ_x += 0.5 * ism * (pJ_x + nJ_x)
                isJ_z += 0.5 * ism * (pJ_z + nJ_z)
                ivJ_x += 0.5 * ivm * (pJ_x - nJ_x)
                ivJ_z += 0.5 * ivm * (pJ_z - nJ_z)
            end

            @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f\n", x, z, ppJ_x, ppJ_z, pnJ_x, pnJ_z, npJ_x, npJ_z, nnJ_x, nnJ_z, isJ_x, isJ_z, ivJ_x, ivJ_z)
        end
    end
        # Case of spin transition current densities ...
    Output_File_Spin = IO_TDA * "/densities/TDA_Spin_" * File_Name * "_JP$(JP)_Averaged.dat"
    open(Output_File_Spin, "w") do Write_File
        @printf(Write_File, "%-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s\n", "x", "z", "ppJ_x", "ppJ_z", "pnJ_x", "pnJ_z", "npJ_x", "npJ_z", "nnJ_x", "nnJ_z", "isJ_x", "isJ_z", "ivJ_x", "ivJ_z")
        @inbounds for N in 1:N_map
            r, theta = x_map[1,N], x_map[2,N]
            x = r * sin(theta)
            z = r * cos(theta)
            ppJ_x, pnJ_x, npJ_x, nnJ_x, isJ_x, ivJ_x = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ppJ_z, pnJ_z, npJ_z, nnJ_z, isJ_z, ivJ_z = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            @inbounds for nu_ind in 1:N_phonon
                pJ_x, pJ_z = pJ_spin_map[1,nu_ind,N], pJ_spin_map[3,nu_ind,N]
                nJ_x, nJ_z = nJ_spin_map[1,nu_ind,N], nJ_spin_map[3,nu_ind,N]

                pm, nm = pM[nu_ind], nM[nu_ind]
                ism, ivm = isM[nu_ind], ivM[nu_ind]

                ppJ_x += pm * pJ_x
                ppJ_z += pm * pJ_z
                pnJ_x += pm * nJ_x
                pnJ_z += pm * nJ_z
                npJ_x += nm * pJ_x
                nnJ_x += nm * nJ_x
                npJ_z += nm * pJ_z
                nnJ_z += nm * nJ_z
                isJ_x += 0.5 * ism * (pJ_x + nJ_x)
                isJ_z += 0.5 * ism * (pJ_z + nJ_z)
                ivJ_x += 0.5 * ivm * (pJ_x - nJ_x)
                ivJ_z += 0.5 * ivm * (pJ_z - nJ_z)
            end

            @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f\n", x, z, ppJ_x, ppJ_z, pnJ_x, pnJ_z, npJ_x, npJ_z, nnJ_x, nnJ_z, isJ_x, isJ_z, ivJ_x, ivJ_z)
        end
    end

    display("Export of averaged phonon level transition current maps successfuly completed ...")

    return
end

function TDA_transitions_recalculate(Params::Parameters)
    # Set the path to the TDA input/output directory ...
    IO_TDA = "io/" * Params.Calc.Path * "/" * Params.Calc.TDA.Name

    # Check if the TDA transition directory exists, if not create it ...
    if !(isdir(IO_TDA * "/transitions"))
        mkdir(IO_TDA * "/transitions")
        mkdir(IO_TDA * "/transitions/E0")
        mkdir(IO_TDA * "/transitions/E1")
        mkdir(IO_TDA * "/transitions/E2")
        mkdir(IO_TDA * "/transitions/E3")
    end

    # Initialize the angular momentum algebra ...
    wigner_init_float(75, "Jmax", 9)

    # Set the path to the reference input directory ... (the TDA does not export any ground-state data)
    IO_ref = "io/" * Params.Calc.Path * "/" * Params.Calc.TDA.Reference

    # Import reference single-particle orbitals ...
    Orb = orbitals_import(Params,IO_ref * "/bin/orb.bin")

    # Construct 2-body 1p-1h orbitals ... (as in TDA_solver_import)
    Orb_1p1h = orbitals_1p1h_make(Params,Orb;Tol_par = Params.Calc.TDA.Tol_par,Tol_hol = Params.Calc.TDA.Tol_hol)

    # Import transformation matrix mapping LHO & reference basis ...
    C = O1b_import(Params,Orb,IO_ref * "/bin/C.bin")

    # Import reference OBDM ... reference basis ...
    Rho = O1b_import(Params,Orb,IO_ref * "/bin/rho.bin")

    # If used, import reference 2-body correlation function ...
    if Params.Calc.TDA.Reference_TBCF == true
        println("\nImporting reference 2-body correlation function ...")
        Sigma, Orb_NN = O2b_import(Params,Orb,IO_ref * "/bin/sigma.bin",Make_Orb_NN = true)
    else
        Sigma = O2b_initialize_empty()
        Orb_NN = O2b_initialize(Params,Orb,Make_Orb_NN = true)[2]
    end

    # Import TDA solutions ...
    @time X_TDA, E_TDA = TDA_import_binary(Params)

    # Evaluate the reference charge radius chR ... (as in TDA_solver)
    chR2 = rho_chR2(Params,Orb,C,Rho)

    # Precalculate the reduced matrix elements of Electrogmanetic 1-body transition operators ...
        # Iniztialize the 1-body Electromagnetic transition operators ...
    TrOp = Tr1b_initialize(Params,Orb,chR2)
         # Transform the 1-body transition operators to the reference basis ...
    TrOp = Tr1b_transformation(Params,Orb,C,TrOp)

    # Calculate the reduced transition matrix elements M_nu for each phonon level ... (exact transitions, as in TDA_solver)
    @time rM = TDA_rM(Params,Orb,Orb_1p1h,Orb_NN,TrOp,Rho,Sigma,X_TDA)

    # Evaluate the reduced transition intensities B_nu for each phonon level ...
    @time rB = TDA_rB(Params,Orb_1p1h,rM,E_TDA)

    # Remove the previous transition moments & TRK values from TDA_summary.dat ... they are appended again by TDA_transitions_export ...
    Summary_File = IO_TDA * "/TDA_summary.dat"
    if isfile(Summary_File)
        Summary = read(Summary_File,String)
        i_moments = findfirst("\nReview of TDA electromagnetic transition moments",Summary)
        if i_moments !== nothing
            write(Summary_File,Summary[1:first(i_moments)-1])
        end
    end

    # Export of TDA electromagnetic transitions ... (reference OBDM in the reference basis, as in TDA_solver)
    @time TDA_transitions_export(Params,Orb,Orb_1p1h,C,Rho,E_TDA,rB)

    return
end
