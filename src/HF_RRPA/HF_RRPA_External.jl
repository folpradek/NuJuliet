function HF_RRPA_import_binary(Params::Parameters,N_nu::Matrix{Int64})
    # Parameters / dimensions
    N_max = Params.Calc.Nmax
    J_max = 2*N_max + 1
    OutputFile = Params.Calc.Path

    # Initialite the list of values of J & P for iteration ...
    JP_list = JP_initialize(J_max)

    println("\nImporting RRPA solutions from binary format ...")

    # Initialize amplitudes & energies ...
    X_RRPA = Matrix{Matrix{ComplexF64}}(undef,J_max+1,2)
    Y_RRPA = Matrix{Matrix{ComplexF64}}(undef,J_max+1,2)
    E_RRPA = Matrix{Vector{ComplexF64}}(undef,J_max+1,2)

    # Initialite each J & P block ...
    @inbounds for JP in JP_list
        J, P = JP[1], JP[2]
        N_ph = N_nu[J+1,P]
        X_RRPA[J+1,P] = Matrix{ComplexF64}(undef,N_ph,N_ph)
        Y_RRPA[J+1,P] = Matrix{ComplexF64}(undef,N_ph,N_ph)
        E_RRPA[J+1,P] = Vector{ComplexF64}(undef,N_ph)
    end

    # Read RRPA amplitudes X ...
    Import_Path_RRPA_X = "IO/$(OutputFile)/Bin/RRPA_X.bin"
    open(Import_Path_RRPA_X, "r") do Import_File
        @inbounds for JP in JP_list
            J, P = JP[1], JP[2]
            N_ph = N_nu[J+1,P]
            N_ph == 0 && continue
            Buffer = Vector{ComplexF64}(undef,N_ph*N_ph)
            read!(Import_File,Buffer)
            X_RRPA[J+1,P] .= reshape(Buffer,N_ph,N_ph)
        end
    end

    # Read RRPA amplitudes Y ...
    Import_Path_RRPA_Y = "IO/$(OutputFile)/Bin/RRPA_Y.bin"
    open(Import_Path_RRPA_Y, "r") do Import_File
        @inbounds for JP in JP_list
            J, P = JP[1], JP[2]
            N_ph = N_nu[J+1,P]
            N_ph == 0 && continue
            Buffer = Vector{ComplexF64}(undef,N_ph*N_ph)
            read!(Import_File,Buffer)
            Y_RRPA[J+1,P] .= reshape(Buffer,N_ph,N_ph)
        end
    end

    # Read RRPA energies E ...
    Import_Path_RRPA_X = "IO/$(OutputFile)/Bin/RRPA_E.bin"
    open(Import_Path_RRPA_X, "r") do Import_File
        @inbounds for JP in JP_list
            J, P = JP[1], JP[2]
            N_ph = N_nu[J+1,P]
            N_ph == 0 && continue
            Buffer = Vector{ComplexF64}(undef,N_ph)
            read!(Import_File,Buffer)
            E_RRPA[J+1,P] .= Buffer
        end
    end

    println("\tRRPA solutions have been succesufully imported ...")

    return X_RRPA, Y_RRPA, E_RRPA
end

function HF_RRPA_transition_densities(Params::Parameters;JP::String,nu_list::Vector{Int64},File_Name::String = "Phonon_Densities")
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

    # Set the internal value of parity P ...
    P = 1
    if P_char == "-"
        P = 2
    end

    # Make single-particle orbitals ...
    Orb = orbitals_make(Params)

    # Prepare Particle & Hole orbitals ...
    N_Particle, Particle, N_Hole, Hole = orbitals_ph_make(Params,Orb)

    # Prepare 1p-1h phonon states ...
    N_Phonon, Phonon = orbitals_one_phonon_make(N_Particle,Particle,N_Hole,Hole)

    # Count & pre-index all phonon states in JP subspaces ...
    N_nu, Orb_Phonon = HF_RRPA_phonon_count(Params,N_Phonon,Phonon)

    # Import transformation matrix mapping LHO & reference RRPA basis ...
    @time C = O1b_import(Params,Orb,"IO/" * Params.Calc.Path * "/Bin/C_RRPA.bin")

    # Import RRPA OBDM ...
    @time Rho = O1b_import(Params,Orb,"IO/" * Params.Calc.Path * "/Bin/Rho_RRPA.bin")

    # Import RRPA solutions ...
    @time X_RRPA, Y_RRPA, E_RRPA = HF_RRPA_import_binary(Params,N_nu)

    # Determine the number of phonon levels & the total number of phonons ...
    N_phonon = length(nu_list)
    N_ph = N_nu[J+1,P]

    # Precalculate the reduced matrix elements of Electrogmanetic 1-body transition operators ...
        # Initialize the 1-body Electromagnetic transition operators ...
    TrOp = Tr1b_initialize(Params,Orb,1.0)
         # Transform the 1-body transition operators to the reference basis ...
    TrOp = Tr1b_transformation(Params,Orb,C,TrOp)

    # Precalculate the matrix elements of 1-body transition operators ...
        # Initialize the matrix elements for given phonon levels ...
    pM, nM = Vector{ComplexF64}(undef,N_phonon), Vector{ComplexF64}(undef,N_phonon)
    isM, ivM = Vector{ComplexF64}(undef,N_phonon), Vector{ComplexF64}(undef,N_phonon)
        # Calculate the matrix elements for given phonon levels ...
    if (J,P) in [(0,1),(2,1),(3,2)]
        @inbounds for (nu_ind, nu) in enumerate(nu_list)
            pMSum, nMSum, isMSum, ivMSum = 0.0im, 0.0im, 0.0im, 0.0im

            @inbounds for ph in 1:N_ph
                i_ph = Orb_Phonon[J+1,P][ph]
                p, h, t_ph = Phonon[i_ph].p, Phonon[i_ph].h, Phonon[i_ph].tz
                x_RRPA, y_RRPA = X_RRPA[J+1,P][ph,nu], Y_RRPA[J+1,P][ph,nu]

                MME = (phase(J) * x_RRPA  + y_RRPA) / sqrt(Float64(2*J + 1))

                if t_ph == -1
                    a_p, a_h = Particle.p[p].a, Hole.p[h].a

                    MME = sqrt(Rho.p[a_h,a_h] - Rho.p[a_p,a_p]) * MME

                    if J == 0 && P == 1
                        MME = TrOp.E0.p[a_p,a_h] * MME

                    elseif J == 2 && P == 1
                        MME = TrOp.E2.p[a_p,a_h] * MME

                    elseif J == 3 && P == 2
                        MME = TrOp.E3.p[a_p,a_h] * MME
                    end

                    pMSum += MME
                    isMSum += 0.5 * MME
                    ivMSum += 0.5 * MME

                elseif t_ph == 1
                    a_p, a_h = Particle.n[p].a, Hole.n[h].a

                    MME = sqrt(Rho.n[a_h,a_h] - Rho.n[a_p,a_p]) * MME

                    if J == 0 && P == 1
                        MME = TrOp.E0.n[a_p,a_h] * MME

                    elseif J == 2 && P == 1
                        MME = TrOp.E2.n[a_p,a_h] * MME

                    elseif J == 3 && P == 2
                        MME = TrOp.E3.n[a_p,a_h] * MME
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
            pMSum, nMSum, isMSum, ivMSum = 0.0im, 0.0im, 0.0im, 0.0im

            @inbounds for ph in 1:N_ph
                i_ph = Orb_Phonon[J+1,P][ph]
                p, h, t_ph = Phonon[i_ph].p, Phonon[i_ph].h, Phonon[i_ph].tz
                x_RRPA, y_RRPA = X_RRPA[J+1,P][ph,nu], Y_RRPA[J+1,P][ph,nu]

                Amp = (phase(J) * x_RRPA  + y_RRPA) / sqrt(Float64(2*J + 1))

                if t_ph == -1
                    a_p, a_h = Particle.p[p].a, Hole.p[h].a

                    Amp = sqrt(Rho.p[a_h,a_h] - Rho.p[a_p,a_p]) * Amp

                    pMME = TrOp.E1.p[a_p,a_h] * Amp
                    isMME = 0.5 * TrOp.E1_C.p[a_p,a_h] * Amp
                    ivMME = TrOp.E1.p[a_p,a_h] * Amp * (A - Z) / (A)

                    pMSum += pMME
                    isMSum += isMME
                    ivMSum += ivMME

                elseif t_ph == 1
                    a_p, a_h = Particle.n[p].a, Hole.n[h].a

                    Amp = sqrt(Rho.n[a_h,a_h] - Rho.n[a_p,a_p]) * Amp

                    nMME = TrOp.E1.n[a_p,a_h] * Amp
                    isMME = 0.5 * TrOp.E1_C.n[a_p,a_h] * Amp
                    ivMME = - TrOp.E1.n[a_p,a_h] * Amp * (Z) / (A)

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
    r_grid = range(r_min, stop = r_max, length = N_grid)
    r_grid = collect(r_grid)

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

    # Initialize radial grids for given phonon levels ...
    pRho_nu = Matrix{Float64}(undef,N_phonon,N_grid)
    nRho_nu = Matrix{Float64}(undef,N_phonon,N_grid)

    # For given phonon levels evaluate their radial representation ...
    @inbounds Threads.@threads for i in 1:N_grid
        r = r_grid[i]
        @inbounds for (nu_ind, nu) in enumerate(nu_list)
            pSum, nSum = 0.0, 0.0

            Gauge = 1.0

            if ((J,P) in [(0,1),(1,2),(2,1),(3,2)]) == true
                Gauge = real(pM[nu_ind]) / abs(pM[nu_ind])
            end

            @inbounds for ph in 1:N_ph
                i_ph = Orb_Phonon[J+1,P][ph]
                p, h, t_ph = Phonon[i_ph].p, Phonon[i_ph].h, Phonon[i_ph].tz
                x_RRPA, y_RRPA = X_RRPA[J+1,P][ph,nu], Y_RRPA[J+1,P][ph,nu]

                if t_ph == -1
                    a_p, a_h = Particle.p[p].a, Hole.p[h].a
                    pRho = Gauge * real(phase(J) * x_RRPA + y_RRPA) * pOrb_grid[a_p,i] * pOrb_grid[a_h,i]
                    pSum += pRho
                elseif t_ph == 1
                    a_p, a_h = Particle.n[p].a, Hole.n[h].a
                    nRho = Gauge * (phase(J) * x_RRPA + y_RRPA) * nOrb_grid[a_p,i] * nOrb_grid[a_h,i]
                    nSum += nRho
                end
            end
            Amp = r^2 / Float64(2*J + 1)
            pRho_nu[nu_ind,i] = Amp * pSum
            nRho_nu[nu_ind,i] = Amp * nSum
        end
    end

    # Perform the export of indiviual RRPA radial phonon densities ...
    @inbounds for (nu_ind, nu) in enumerate(nu_list)
        e = real(E_RRPA[J+1,P][nu])
        sE = string(round(e,digits=3))

        Output_File = "IO/" * Params.Calc.Path * "/RRPA/Densities/RRPA_" * File_Name * "_JP$(JP)_nu$(nu)_E$(sE).dat"
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
    pM .= abs.(pM) ./ sqrt(sum(pM.^2))
    nM .= abs.(nM) ./ sqrt(sum(nM.^2))
    isM .= abs.(isM) ./ sqrt(sum(isM.^2))
    ivM .= abs.(ivM) ./ sqrt(sum(ivM.^2))

    # Perform the export of of averaged RRPA transition radial densities ...
    Output_File = "IO/" * Params.Calc.Path * "/RRPA/Densities/RRPA_" * File_Name * "_JP$(JP)_Averaged.dat"
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

function HF_RRPA_transition_operator_densities(Params::Parameters;JP::String,nu_list::Vector{Int64},File_Name::String = "Phonon_Transition_Operator_Densities")
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
    P = 1
    if P_char == "-"
        P = 2
    end

    # Check if the given JP is supported for transition density calculations ...
    if ((J,P) in [(0,1),(1,2),(2,1),(3,2)]) == false
        error("The given values of J & P are not supported for radial transition operator density calculations ...")
    end

    # Make single-particle orbitals ...
    Orb = orbitals_make(Params)

    # Prepare Particle & Hole orbitals ...
    N_Particle, Particle, N_Hole, Hole = orbitals_ph_make(Params,Orb)

    # Prepare 1p-1h phonon states ...
    N_Phonon, Phonon = orbitals_one_phonon_make(N_Particle,Particle,N_Hole,Hole)

    # Count & pre-index all phonon states in JP subspaces ...
    N_nu, Orb_Phonon = HF_RPA_phonon_count(Params,N_Phonon,Phonon)

    # Import transformation matrix mapping LHO & reference RRPA basis ...
    @time C = O1b_import(Params,Orb,"IO/" * Params.Calc.Path * "/Bin/C_RRPA.bin")

    # Import RRPA OBDM ...
    @time Rho = O1b_import(Params,Orb,"IO/" * Params.Calc.Path * "/Bin/Rho_RRPA.bin")

    # Import RRPA solutions ...
    @time X_RRPA, Y_RRPA, E_RRPA = HF_RRPA_import_binary(Params,N_nu)

    # Determine the number of phonon levels & the total number of phonons ...
    N_phonon = length(nu_list)
    N_ph = N_nu[J+1,P]

    # Precalculate the reduced matrix elements of Electrogmanetic 1-body transition operators ...
        # Iniztialize the 1-body Electromagnetic transition operators ...
    TrOp = Tr1b_initialize(Params,Orb,1.0)
         # Transform the 1-body transition operators to the reference basis ...
    TrOp = Tr1b_transformation(Params,Orb,C,TrOp)

    # Precalculate the matrix elements of 1-body transition operators ...
        # Initialize the matrix elements for given phonon levels ...
    pM, nM = Vector{ComplexF64}(undef,N_phonon), Vector{ComplexF64}(undef,N_phonon)
    isM, ivM = Vector{ComplexF64}(undef,N_phonon), Vector{ComplexF64}(undef,N_phonon)
        # Calculate the matrix elements for given phonon levels ...
    if (J,P) in [(0,1),(2,1),(3,2)]
        @inbounds for (nu_ind, nu) in enumerate(nu_list)
            pMSum, nMSum, isMSum, ivMSum = 0.0im, 0.0im, 0.0im, 0.0im

            @inbounds for ph in 1:N_ph
                i_ph = Orb_Phonon[J+1,P][ph]
                p, h, t_ph = Phonon[i_ph].p, Phonon[i_ph].h, Phonon[i_ph].tz
                x_RRPA, y_RRPA = X_RRPA[J+1,P][ph,nu], Y_RRPA[J+1,P][ph,nu]

                MME = (phase(J) * x_RRPA  + y_RRPA) / sqrt(Float64(2*J + 1))

                if t_ph == -1
                    a_p, a_h = Particle.p[p].a, Hole.p[h].a

                    MME = sqrt(Rho.p[a_h,a_h] - Rho.p[a_p,a_p]) * MME

                    if J == 0 && P == 1
                        MME = TrOp.E0.p[a_p,a_h] * MME

                    elseif J == 2 && P == 1
                        MME = TrOp.E2.p[a_p,a_h] * MME

                    elseif J == 3 && P == 2
                        MME = TrOp.E3.p[a_p,a_h] * MME
                    end

                    pMSum += MME
                    isMSum += 0.5 * MME
                    ivMSum += 0.5 * MME

                elseif t_ph == 1
                    a_p, a_h = Particle.n[p].a, Hole.n[h].a

                    MME = sqrt(Rho.n[a_h,a_h] - Rho.n[a_p,a_p]) * MME

                    if J == 0 && P == 1
                        MME = TrOp.E0.n[a_p,a_h] * MME

                    elseif J == 2 && P == 1
                        MME = TrOp.E2.n[a_p,a_h] * MME

                    elseif J == 3 && P == 2
                        MME = TrOp.E3.n[a_p,a_h] * MME
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
            pMSum, nMSum, isMSum, ivMSum = 0.0im, 0.0im, 0.0im, 0.0im

            @inbounds for ph in 1:N_ph
                i_ph = Orb_Phonon[J+1,P][ph]
                p, h, t_ph = Phonon[i_ph].p, Phonon[i_ph].h, Phonon[i_ph].tz
                x_RRPA, y_RRPA = X_RRPA[J+1,P][ph,nu], Y_RRPA[J+1,P][ph,nu]

                Amp = (phase(J) * x_RRPA  + y_RRPA) / sqrt(Float64(2*J + 1))

                if t_ph == -1
                    a_p, a_h = Particle.p[p].a, Hole.p[h].a

                    Amp = sqrt(Rho.p[a_h,a_h] - Rho.p[a_p,a_p]) * Amp

                    pMME = TrOp.E1.p[a_p,a_h] * Amp
                    isMME = 0.5 * TrOp.E1_C.p[a_p,a_h] * Amp
                    ivMME = TrOp.E1.p[a_p,a_h] * Amp * (A - Z) / (A)

                    pMSum += pMME
                    isMSum += isMME
                    ivMSum += ivMME

                elseif t_ph == 1
                    a_p, a_h = Particle.n[p].a, Hole.n[h].a

                    Amp = sqrt(Rho.n[a_h,a_h] - Rho.n[a_p,a_p]) * Amp

                    nMME = TrOp.E1.n[a_p,a_h] * Amp
                    isMME = 0.5 * TrOp.E1_C.n[a_p,a_h] * Amp
                    ivMME = - TrOp.E1.n[a_p,a_h] * Amp * (Z) / (A)

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
    r_grid = range(r_min, stop = r_max, length = N_grid)
    r_grid = collect(r_grid)

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

            pN, nN = 1.0, 1.0

            if J == 0 && P == 1
                pN = 1.0 /  integrate_quadrature(r_grid, r_grid.^4 .* pR_a .* pR_b, M = 11)
                nN = 1.0 /  integrate_quadrature(r_grid, r_grid.^4 .* nR_a .* nR_b, M = 11)

            elseif J == 1 && P == 2
                pN = 1.0 /  integrate_quadrature(r_grid, r_grid.^3 .* pR_a .* pR_b, M = 11)
                nN = 1.0 /  integrate_quadrature(r_grid, r_grid.^3 .* nR_a .* nR_b, M = 11)

            elseif J == 2 && P == 1
                pN = 1.0 /  integrate_quadrature(r_grid, r_grid.^4 .* pR_a .* pR_b, M = 11)
                nN = 1.0 /  integrate_quadrature(r_grid, r_grid.^4 .* nR_a .* nR_b, M = 11)

            elseif J == 3 && P == 2
                pN = 1.0 /  integrate_quadrature(r_grid, r_grid.^5 .* pR_a .* pR_b, M = 11)
                nN = 1.0 /  integrate_quadrature(r_grid, r_grid.^5 .* nR_a .* nR_b, M = 11)
            end

            @inbounds for i in 1:N_grid
                r = r_grid[i]

                if J == 0 && P == 1
                    pM_grid[a,b,i] = TrOp.E0.p[a,b] * r^2 * pR_a[i] * pR_b[i] * pN
                    nM_grid[a,b,i] = TrOp.E0.n[a,b] * r^2 * nR_a[i] * nR_b[i] * nN

                elseif J == 1 && P == 2
                    pM_grid[a,b,i] = TrOp.E1.p[a,b] * r * pR_a[i] * pR_b[i] * pN
                    nM_grid[a,b,i] = TrOp.E1.n[a,b] * r * nR_a[i] * nR_b[i] * nN

                elseif J == 2 && P == 1
                    pM_grid[a,b,i] = TrOp.E2.p[a,b] * r^2 * pR_a[i] * pR_b[i] * pN
                    nM_grid[a,b,i] = TrOp.E2.n[a,b] * r^2 * nR_a[i] * nR_b[i] * nN

                elseif J == 3 && P == 2
                    pM_grid[a,b,i] = TrOp.E3.p[a,b] * r^3 * pR_a[i] * pR_b[i] * pN
                    nM_grid[a,b,i] = TrOp.E3.n[a,b] * r^3 * nR_a[i] * nR_b[i] * nN

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

            if ((J,P) in [(0,1),(1,2),(2,1),(3,2)]) == true
                Gauge = real(pM[nu_ind]) / abs(pM[nu_ind])
            end

            @inbounds for ph in 1:N_ph
                i_ph = Orb_Phonon[J+1,P][ph]
                p, h, t_ph = Phonon[i_ph].p, Phonon[i_ph].h, Phonon[i_ph].tz
                x_RRPA, y_RRPA = X_RRPA[J+1,P][ph,nu], Y_RRPA[J+1,P][ph,nu]

                if t_ph == -1
                    a_p, a_h = Particle.p[p].a, Hole.p[h].a
                    pm = pM_grid[a_p,a_h,i]
                    pRho = Gauge * real(phase(J) * x_RRPA + y_RRPA) * pm
                    pSum += pRho
                elseif t_ph == 1
                    a_p, a_h = Particle.n[p].a, Hole.n[h].a
                    nm = nM_grid[a_p,a_h,i]
                    nRho = Gauge * real(phase(J) * x_RRPA + y_RRPA) * nm
                    nSum += nRho
                end
            end

            Amp = r^2 / Float64(2*J + 1)

            pRho_nu[nu_ind,i] = Amp * pSum
            nRho_nu[nu_ind,i] = Amp * nSum
        end
    end

    # Perform the export of indiviual RRPA radial phonon densities ...
    @inbounds for (nu_ind, nu) in enumerate(nu_list)
        e = real(E_RRPA[J+1,P][nu])
        sE = string(round(e,digits=3))

        # Case of TDA densities ...
        Output_File = "IO/" * Params.Calc.Path * "/RRPA/Densities/RRPA_" * File_Name * "_JP$(JP)_nu$(nu)_E$(sE).dat"
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
    pM .= abs.(pM) ./ sqrt(sum(pM.^2))
    nM .= abs.(nM) ./ sqrt(sum(nM.^2))
    isM .= abs.(isM) ./ sqrt(sum(isM.^2))
    ivM .= abs.(ivM) ./ sqrt(sum(ivM.^2))

    # Perform the export of of averaged RRPA transition matrix radial densities ...
    Output_File = "IO/" * Params.Calc.Path * "/RRPA/Densities/RRPA_" * File_Name * "_JP$(JP)_Averaged.dat"
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

function HF_RRPA_transition_analysis(Params::Parameters;JP::String,nu_list::Vector{Int64},File_Name::String = "Phonon_Transtion_Analysis", Cut::Float64 = 0.1)
    # Initialize the angular momentum algebra ...
    wigner_init_float(75, "Jmax", 9)

    # Make directory for export of information on structure of phonon levels ...
    if isdir("IO/" * Params.Calc.Path * "/RRPA/Levels") == false
        mkdir("IO/" * Params.Calc.Path * "/RRPA/Levels")
    end

    # Read the values of J & P ...
    P_char = string(JP[end])
    J = parse(Int64,JP[1:end-1])

    # Check input formating ...
    if (P_char != "+") && (P_char != "-")
        error("Invalid JP = \"$JP\"; use J+ or J-.")
    end

    # Set internal value of parity P ...
    P = 1
    if P_char == "-"
        P = 2
    end

    # Make single-particle orbitals ...
    Orb = orbitals_make(Params)

    # Prepare Particle & Hole orbitals ...
    N_Particle, Particle, N_Hole, Hole = orbitals_ph_make(Params,Orb)

    # Prepare 1p-1h phonon states ...
    N_Phonon, Phonon = orbitals_one_phonon_make(N_Particle,Particle,N_Hole,Hole)

    # Count & pre-index all phonon states in JP subspaces ...
    N_nu, Orb_Phonon = HF_RPA_phonon_count(Params,N_Phonon,Phonon)

    # Import transformation matrix mapping LHO & reference RRPA basis ...
    @time C = O1b_import(Params,Orb,"IO/" * Params.Calc.Path * "/Bin/C_RRPA.bin")

    # Import RRPA OBDM ...
    @time Rho = O1b_import(Params,Orb,"IO/" * Params.Calc.Path * "/Bin/Rho_RRPA.bin")

    # Import RRPA solutions ...
    @time X_RRPA, Y_RRPA, E_RRPA = HF_RRPA_import_binary(Params,N_nu)

    # Determine the number of phonon levels & the total number of phonons ...
    N_phonon = length(nu_list)
    N_ph = N_nu[J+1,P]

    # Precalculate the reduced matrix elements of Electrogmanetic 1-body transition operators ...
        # Iniztialize the 1-body Electromagnetic transition operators ...
    TrOp = Tr1b_initialize(Params,Orb,1.0)
         # Transform the 1-body transition operators to the reference basis ...
    TrOp = Tr1b_transformation(Params,Orb,C,TrOp)

    nu_count = zeros(Int64,N_phonon)

    # Iterate through the phonon solutions & determine the most dominant ph & hp components of given phonon levels ...
    @inbounds for (nu_ind, nu) in enumerate(nu_list)
        @inbounds for ph in 1:N_ph
            x_RRPA, y_RRPA = X_RRPA[J+1,P][ph,nu], Y_RRPA[J+1,P][ph,nu]
            Amp = (phase(J) * x_RRPA + y_RRPA)

            if abs2(Amp) > Cut
                nu_count[nu_ind] += 1
            end
        end
    end

    nu_compo = Vector{Vector{Tuple{Int64,Int64,Int64,Float64,Float64,Float64}}}(undef,N_phonon)

    @inbounds for (nu_ind, nu) in enumerate(nu_list)
        nu_compo[nu_ind] = Vector{Tuple{Int64,Int64,Int64,Float64,Float64,Float64}}(undef,nu_count[nu_ind])
        
        count = 0

        @inbounds for ph in 1:N_ph
            i_ph = Orb_Phonon[J+1,P][ph]
            p, h, t_ph = Phonon[i_ph].p, Phonon[i_ph].h, Phonon[i_ph].tz
            x_RRPA, y_RRPA = X_RRPA[J+1,P][ph,nu], Y_RRPA[J+1,P][ph,nu]

            Amp = (phase(J) * x_RRPA + y_RRPA)

            if abs2(Amp) > Cut
                count += 1

                if t_ph == -1
                    a_p, a_h = Particle.p[p].a, Hole.p[h].a
                elseif t_ph == 1
                    a_p, a_h = Particle.n[p].a, Hole.n[h].a 
                end

                M = 0.0

                if J == 0 && P == 1
                    M = real(Amp) * TrOp.E0.p[a_p,a_h]

                elseif J == 1 && P == 2
                    M = real(Amp) * TrOp.E1.p[a_p,a_h]

                elseif J == 2 && P == 1
                    M = real(Amp) * TrOp.E2.p[a_p,a_h]

                elseif J == 3 && P == 2
                    M = real(Amp) * TrOp.E3.p[a_p,a_h]
                end

                nu_compo[nu_ind][count] = (a_p,a_h,t_ph,real(x_RRPA),real(y_RRPA),M)
            end

        end

    end

    # Perform the export of information on individual RRPA phonon levels ...
    @inbounds for (nu_ind, nu) in enumerate(nu_list)
        e = real(E_RRPA[J+1,P][nu])
        sE = string(round(e,digits=3))

        Phonon_RRPA = nu_compo[nu_ind]
        Output_File = "IO/" * Params.Calc.Path * "/RRPA/Levels/RRPA_" * File_Name * "_JP$(JP)_nu$(nu)_E$(sE).dat"
        open(Output_File, "w") do Write_File
            @printf(Write_File, "%-8s %-8s %-8s %-20s %-20s %-20s\n", "a_p", "a_h", "T_ph", "Re X_ph", "Re Y_ph", "Re M_ph")
            @inbounds for L in Phonon_RRPA
                (a_p,a_h,t_ph,x_RRPA,y_RRPA,M_ph) = L
                @printf(Write_File, "%-8d %-8d %-8d %-20.8f %-20.8f %-20.8f\n", a_p, a_h, t_ph, x_RRPA, y_RRPA, M_ph)
            end
        end

    end

    display("Export of information on individual phonon levels successfuly completed ...")

    return
end

function HF_RRPA_transition_currents(Params::Parameters;JP::String,nu_list::Vector{Int64},File_Name::String = "Phonon_Currents",
                                     M::Int64=-1,xN_grid::Int64 = 40, zN_grid::Int64 = 40)    
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
    P = 1
    if P_char == "-"
        P = 2
    end

    # Set the value of projection M ...
    if M == -1
        M = 0
    end

    # Make single-particle orbitals ...
    Orb = orbitals_make(Params)

    # Prepare Particle & Hole orbitals ...
    N_Particle, Particle, N_Hole, Hole = orbitals_ph_make(Params,Orb)

    # Prepare 1p-1h phonon states ...
    N_Phonon, Phonon = orbitals_one_phonon_make(N_Particle,Particle,N_Hole,Hole)

    # Count & pre-index all phonon states in JP subspaces ...
    N_nu, Orb_Phonon = HF_RPA_phonon_count(Params,N_Phonon,Phonon)

    # Import transformation matrix mapping LHO & reference RRPA basis ...
    @time C = O1b_import(Params,Orb,"IO/" * Params.Calc.Path * "/Bin/C_RRPA.bin")

    # Import RRPA OBDM ...
    @time Rho = O1b_import(Params,Orb,"IO/" * Params.Calc.Path * "/Bin/Rho_RRPA.bin")

    # Import RRPA solutions ...
    @time X_RRPA, Y_RRPA, E_RRPA = HF_RRPA_import_binary(Params,N_nu)

    # Determine the number of phonon levels & the total number of phonons ...
    N_phonon = length(nu_list)
    N_ph = N_nu[J+1,P]

    # Precalculate the reduced matrix elements of Electrogmanetic 1-body transition operators ...
        # Iniztialize the 1-body Electromagnetic transition operators ...
    TrOp = Tr1b_initialize(Params,Orb,1.0)
         # Transform the 1-body transition operators to the reference basis ...
    TrOp = Tr1b_transformation(Params,Orb,C,TrOp)

    # Precalculate the matrix elements of 1-body transition operators ...
        # Initialize the matrix elements for given phonon levels ...
    pM, nM = Vector{ComplexF64}(undef,N_phonon), Vector{ComplexF64}(undef,N_phonon)
    isM, ivM = Vector{ComplexF64}(undef,N_phonon), Vector{ComplexF64}(undef,N_phonon)

        # Calculate the matrix elements for given phonon levels ...
    if (J,P) in [(0,1),(2,1),(3,2)]
        @inbounds for (nu_ind, nu) in enumerate(nu_list)
            pMSum, nMSum, isMSum, ivMSum = 0.0im, 0.0im, 0.0im, 0.0im

            @inbounds for ph in 1:N_ph
                i_ph = Orb_Phonon[J+1,P][ph]
                p, h, t_ph = Phonon[i_ph].p, Phonon[i_ph].h, Phonon[i_ph].tz
                x_RRPA, y_RRPA = X_RRPA[J+1,P][ph,nu], Y_RRPA[J+1,P][ph,nu]

                MME = (phase(J) * x_RRPA - y_RRPA) / sqrt(Float64(2*J + 1))

                if t_ph == -1
                    a_p, a_h = Particle.p[p].a, Hole.p[h].a

                    MME = sqrt(Rho.p[a_h,a_h] - Rho.p[a_p,a_p]) * MME

                    if J == 0 && P == 1
                        MME = TrOp.E0.p[a_p,a_h] * MME

                    elseif J == 2 && P == 1
                        MME = TrOp.E2.p[a_p,a_h] * MME

                    elseif J == 3 && P == 2
                        MME = TrOp.E3.p[a_p,a_h] * MME
                    end

                    pMSum += MME
                    isMSum += 0.5 * MME
                    ivMSum += 0.5 * MME

                elseif t_ph == 1
                    a_p, a_h = Particle.n[p].a, Hole.n[h].a

                    MME = sqrt(Rho.n[a_h,a_h] - Rho.n[a_p,a_p]) * MME

                    if J == 0 && P == 1
                        MME = TrOp.E0.n[a_p,a_h] * MME

                    elseif J == 2 && P == 1
                        MME = TrOp.E2.n[a_p,a_h] * MME

                    elseif J == 3 && P == 2
                        MME = TrOp.E3.n[a_p,a_h] * MME
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
            pMSum, nMSum, isMSum, ivMSum = 0.0im, 0.0im, 0.0im, 0.0im

            @inbounds for ph in 1:N_ph
                i_ph = Orb_Phonon[J+1,P][ph]
                p, h, t_ph = Phonon[i_ph].p, Phonon[i_ph].h, Phonon[i_ph].tz
                x_RRPA, y_RRPA = X_RRPA[J+1,P][ph,nu], Y_RRPA[J+1,P][ph,nu]

                Amp = (phase(J) * x_RRPA - y_RRPA) / sqrt(Float64(2*J + 1))

                if t_ph == -1
                    a_p, a_h = Particle.p[p].a, Hole.p[h].a

                    Amp = sqrt(Rho.p[a_h,a_h] - Rho.p[a_p,a_p]) * Amp

                    pMME = TrOp.E1.p[a_p,a_h] * Amp
                    isMME = 0.5 * TrOp.E1_C.p[a_p,a_h] * Amp
                    ivMME = TrOp.E1.p[a_p,a_h] * Amp * (A - Z) / (A)

                    pMSum += pMME
                    isMSum += isMME
                    ivMSum += ivMME

                elseif t_ph == 1
                    a_p, a_h = Particle.n[p].a, Hole.n[h].a

                    Amp = sqrt(Rho.n[a_h,a_h] - Rho.n[a_p,a_p]) * Amp

                    nMME = TrOp.E1.n[a_p,a_h] * Amp
                    isMME = 0.5 * TrOp.E1_C.n[a_p,a_h] * Amp
                    ivMME = - TrOp.E1.n[a_p,a_h] * Amp * (Z) / (A)

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
    r_grid = range(r_min, stop = r_max, length = rN_grid)
    r_grid = collect(r_grid)
        # Case of the angular theta grid ...
    theta_grid = range(0.0, stop = 2.0*pi, length = tN_grid)
    theta_grid = collect(theta_grid)

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
        # Single state current maps ...
    pJ_ph_conv_map = Array{Float64}(undef,3,N_ph,N_map)
    nJ_ph_conv_map = Array{Float64}(undef,3,N_ph,N_map)
    pJ_ph_spin_map = Array{Float64}(undef,3,N_ph,N_map)
    nJ_ph_spin_map = Array{Float64}(undef,3,N_ph,N_map)
    pJ_hp_conv_map = Array{Float64}(undef,3,N_ph,N_map)
    nJ_hp_conv_map = Array{Float64}(undef,3,N_ph,N_map)
    pJ_hp_spin_map = Array{Float64}(undef,3,N_ph,N_map)
    nJ_hp_spin_map = Array{Float64}(undef,3,N_ph,N_map)

    pJ_conv_map = Array{Float64}(undef,3,N_phonon,N_map)
    nJ_conv_map = Array{Float64}(undef,3,N_phonon,N_map)
    pJ_spin_map = Array{Float64}(undef,3,N_phonon,N_map)
    nJ_spin_map = Array{Float64}(undef,3,N_phonon,N_map)

    # Prepare the current map grid ...
        # Precalculate radial & angular grid step size ...
    dr = (r_max - r_min) / (rN_grid - 1)
    dtheta = 2.0 * pi / (tN_grid - 1)  # if theta_grid spans 0..2*pi inclusive
        # Reset the grid size counter ...
    N_map = 0
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

            N_map += 1
            i_map[1,N_map] = n
            i_map[2,N_map] = m
            x_map[1,N_map] = r_grid[n]
            x_map[2,N_map] = theta_grid[m]
        end
    end

    # Evaluate the grid of the 1-body transition current matrix elements ...
    @time @inbounds Threads.@threads for N in 1:N_map
        n, m = i_map[1,N], i_map[2,N]
        r, theta = r_grid[n], theta_grid[m]
        g = 1.0

        @inbounds for ph in 1:N_ph
            i_ph = Orb_Phonon[J+1,P][ph]
            p, h, t_ph = Phonon[i_ph].p, Phonon[i_ph].h, Phonon[i_ph].tz

            if t_ph == -1
                a_p, a_h = Particle.p[p].a, Hole.p[h].a
            elseif t_ph == 1
                a_p, a_h = Particle.n[p].a, Hole.n[h].a
            end

            l_p, j_p = Orb[a_p].l, Orb[a_p].j
            l_h, j_h = Orb[a_h].l, Orb[a_h].j

            Jx_ph_conv, Jy_ph_conv, Jz_ph_conv = 0.0, 0.0, 0.0
            Jx_ph_spin, Jy_ph_spin, Jz_ph_spin = 0.0, 0.0, 0.0

            Jx_hp_conv, Jy_hp_conv, Jz_hp_conv = 0.0, 0.0, 0.0
            Jx_hp_spin, Jy_hp_spin, Jz_hp_spin = 0.0, 0.0, 0.0

            if t_ph == -1
                R_p, R_h = pR_grid[a_p,n], pR_grid[a_h,n]
                dR_p, dR_h = pdR_grid[a_p,n], pdR_grid[a_h,n]
                g = 0.5 * g_p
            elseif t_ph == 1
                R_p, R_h = nR_grid[a_p,n], nR_grid[a_h,n]
                dR_p, dR_h = ndR_grid[a_p,n], ndR_grid[a_h,n]
                g = 0.5 * g_n
            end

            @inbounds for m_p in -j_p:2:j_p
                @inbounds for m_h in -j_h:2:j_h

                    # Case of particle-hole contributions ...
                    if (m_p + m_h) == 2*M
                        Amp_ph = fCG(j_p,j_h,2*J,m_p,m_h,2*M) * phase(div(j_h - m_h,2))

                        (jx_ph_conv,jy_ph_conv,jz_ph_conv) = J1b_convective(r,theta,0.0,l_p,j_p,m_p,R_p,dR_p,l_h,j_h,-m_h,R_h,dR_h)
                        (jx_ph_spin,jy_ph_spin,jz_ph_spin) = g .* J1b_spin(r,theta,0.0,l_p,j_p,m_p,R_p,dR_p,l_h,j_h,-m_h,R_h,dR_h)

                        Jx_ph_conv += Amp_ph * real(jx_ph_conv)
                        Jy_ph_conv += Amp_ph * real(jy_ph_conv)
                        Jz_ph_conv += Amp_ph * real(jz_ph_conv)

                        Jx_ph_spin += Amp_ph * real(jx_ph_spin)
                        Jy_ph_spin += Amp_ph * real(jy_ph_spin)
                        Jz_ph_spin += Amp_ph * real(jz_ph_spin)
                    end


                    # Case of hole-particle contributions ...
                    if (m_p + m_h) == -2*M
                        Amp_hp = fCG(j_p,j_h,2*J,m_p,m_h,-2*M) * phase(J + M + div(j_h + m_h,2))

                        (jx_hp_conv,jy_hp_conv,jz_hp_conv) = J1b_convective(r,theta,0.0,l_h,j_h,-m_h,R_h,dR_h,l_p,j_p,m_p,R_p,dR_p)
                        (jx_hp_spin,jy_hp_spin,jz_hp_spin) = g .* J1b_spin(r,theta,0.0,l_h,j_h,-m_h,R_h,dR_h,l_p,j_p,m_p,R_p,dR_p)

                        Jx_hp_conv += Amp_hp * real(jx_hp_conv)
                        Jy_hp_conv += Amp_hp * real(jy_hp_conv)
                        Jz_hp_conv += Amp_hp * real(jz_hp_conv)

                        Jx_hp_spin += Amp_hp * real(jx_hp_spin)
                        Jy_hp_spin += Amp_hp * real(jy_hp_spin)
                        Jz_hp_spin += Amp_hp * real(jz_hp_spin)
                    end
                end
            end

            if t_ph == -1
                pJ_ph_conv_map[1,ph,N] = Jx_ph_conv
                pJ_ph_conv_map[2,ph,N] = Jy_ph_conv
                pJ_ph_conv_map[3,ph,N] = Jz_ph_conv
                pJ_ph_spin_map[1,ph,N] = Jx_ph_spin
                pJ_ph_spin_map[2,ph,N] = Jy_ph_spin
                pJ_ph_spin_map[3,ph,N] = Jz_ph_spin

                pJ_hp_conv_map[1,ph,N] = Jx_hp_conv
                pJ_hp_conv_map[2,ph,N] = Jy_hp_conv
                pJ_hp_conv_map[3,ph,N] = Jz_hp_conv
                pJ_hp_spin_map[1,ph,N] = Jx_hp_spin
                pJ_hp_spin_map[2,ph,N] = Jy_hp_spin
                pJ_hp_spin_map[3,ph,N] = Jz_hp_spin

            elseif t_ph == 1
                nJ_ph_conv_map[1,ph,N] = Jx_ph_conv
                nJ_ph_conv_map[2,ph,N] = Jy_ph_conv
                nJ_ph_conv_map[3,ph,N] = Jz_ph_conv
                nJ_ph_spin_map[1,ph,N] = Jx_ph_spin
                nJ_ph_spin_map[2,ph,N] = Jy_ph_spin
                nJ_ph_spin_map[3,ph,N] = Jz_ph_spin

                nJ_hp_conv_map[1,ph,N] = Jx_hp_conv
                nJ_hp_conv_map[2,ph,N] = Jy_hp_conv
                nJ_hp_conv_map[3,ph,N] = Jz_hp_conv
                nJ_hp_spin_map[1,ph,N] = Jx_hp_spin
                nJ_hp_spin_map[2,ph,N] = Jy_hp_spin
                nJ_hp_spin_map[3,ph,N] = Jz_hp_spin
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

            if ((J,P) in [(0,1),(1,2),(2,1),(3,2)]) == true
                Gauge = pM[nu_ind] / abs(pM[nu_ind])
            end

            @inbounds for ph in 1:N_ph
                i_ph = Orb_Phonon[J+1,P][ph]
                p, h, t_ph = Phonon[i_ph].p, Phonon[i_ph].h, Phonon[i_ph].tz

                if t_ph == -1
                    a_p, a_h = Particle.p[p].a, Hole.p[h].a
                elseif t_ph == 1
                    a_p, a_h = Particle.n[p].a, Hole.n[h].a
                end

                x_RRPA = real(Gauge * X_RRPA[J+1,P][ph,nu])
                y_RRPA = real(Gauge * Y_RRPA[J+1,P][ph,nu])

                if t_ph == -1
                    x_RRPA = sqrt(Rho.p[a_h,a_h] - Rho.p[a_p,a_p]) * x_RRPA
                    y_RRPA = sqrt(Rho.p[a_h,a_h] - Rho.p[a_p,a_p]) * y_RRPA

                    pJxConvSum += (x_RRPA * pJ_ph_conv_map[1,ph,N] - y_RRPA * pJ_hp_conv_map[1,ph,N])
                    pJyConvSum += (x_RRPA * pJ_ph_conv_map[2,ph,N] - y_RRPA * pJ_hp_conv_map[2,ph,N])
                    pJzConvSum += (x_RRPA * pJ_ph_conv_map[3,ph,N] - y_RRPA * pJ_hp_conv_map[3,ph,N])

                    pJxSpinSum += (x_RRPA * pJ_ph_spin_map[1,ph,N] - y_RRPA * pJ_hp_spin_map[1,ph,N])
                    pJySpinSum += (x_RRPA * pJ_ph_spin_map[2,ph,N] - y_RRPA * pJ_hp_spin_map[2,ph,N])
                    pJzSpinSum += (x_RRPA * pJ_ph_spin_map[3,ph,N] - y_RRPA * pJ_hp_spin_map[3,ph,N])

                elseif t_ph == 1
                    x_RRPA = sqrt(Rho.n[a_h,a_h] - Rho.n[a_p,a_p]) * x_RRPA
                    y_RRPA = sqrt(Rho.n[a_h,a_h] - Rho.n[a_p,a_p]) * y_RRPA

                    nJxConvSum += (x_RRPA * nJ_ph_conv_map[1,ph,N] - y_RRPA * nJ_hp_conv_map[1,ph,N])
                    nJyConvSum += (x_RRPA * nJ_ph_conv_map[2,ph,N] - y_RRPA * nJ_hp_conv_map[2,ph,N])
                    nJzConvSum += (x_RRPA * nJ_ph_conv_map[3,ph,N] - y_RRPA * nJ_hp_conv_map[3,ph,N])

                    nJxSpinSum += (x_RRPA * nJ_ph_spin_map[1,ph,N] - y_RRPA * nJ_hp_spin_map[1,ph,N])
                    nJySpinSum += (x_RRPA * nJ_ph_spin_map[2,ph,N] - y_RRPA * nJ_hp_spin_map[2,ph,N])
                    nJzSpinSum += (x_RRPA * nJ_ph_spin_map[3,ph,N] - y_RRPA * nJ_hp_spin_map[3,ph,N])
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

    # Scale the currents to correct physical units in [c] ...
    pJ_conv_map .= pJ_conv_map .* hc / m_p
    nJ_conv_map .= nJ_conv_map .* hc / m_p
    pJ_spin_map .= pJ_spin_map .* hc / m_p
    nJ_spin_map .= nJ_spin_map .* hc / m_p

    # Perform the export of of individual RRPA phonon level transition current maps ...
    @inbounds for (nu_ind, nu) in enumerate(nu_list)
        e =real(E_RRPA[J+1,P][nu])
        sE = string(round(e,digits=3))

        # Case of convective transition current densities ...
        Output_File_Convective = "IO/" * Params.Calc.Path * "/RRPA/Densities/RRPA_Convective_" * File_Name * "_JP$(JP)_nu$(nu)_E$(sE).dat"
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
        Output_File_Spin = "IO/" * Params.Calc.Path * "/RRPA/Densities/RRPA_Spin_" * File_Name * "_JP$(JP)_nu$(nu)_E$(sE).dat"
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
    pM .= abs.(pM) ./ sqrt(sum(pM.^2))
    nM .= abs.(nM) ./ sqrt(sum(nM.^2))
    isM .= abs.(isM) ./ sqrt(sum(isM.^2))
    ivM .= abs.(ivM) ./ sqrt(sum(ivM.^2))

    # Perform the export of of averaged RRPA transition current maps ...
        # Case of convective transition current densities ...
    Output_File_Convective = "IO/" * Params.Calc.Path * "/RRPA/Densities/RRPA_Convective_" * File_Name * "_JP$(JP)_Averaged.dat"
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

                pm, nm = real(pM[nu_ind]), real(nM[nu_ind])
                ism, ivm = real(isM[nu_ind]), real(ivM[nu_ind])

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
    Output_File_Spin = "IO/" * Params.Calc.Path * "/RRPA/Densities/RRPA_Spin_" * File_Name * "_JP$(JP)_Averaged.dat"
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

                pm, nm = real(pM[nu_ind]), real(nM[nu_ind])
                ism, ivm = real(isM[nu_ind]), real(ivM[nu_ind])

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

function HF_RRPA_transitions_recalculate(Params::Parameters)
    # Check if the RRPA transition directory exists, if not create it ...
    if !(isdir("IO/" * Params.Calc.Path * "/RRPA/Transitions"))
        mkdir("IO/" * Params.Calc.Path * "/RRPA/Transitions")
        mkdir("IO/" * Params.Calc.Path * "/RRPA/Transitions/E0")
        mkdir("IO/" * Params.Calc.Path * "/RRPA/Transitions/E1")
        mkdir("IO/" * Params.Calc.Path * "/RRPA/Transitions/E2")
        mkdir("IO/" * Params.Calc.Path * "/RRPA/Transitions/E3")
    end

    # Initialize the angular momentum algebra ...
    wigner_init_float(75, "Jmax", 9)

    # Make single-particle orbitals ...
    Orb = orbitals_make(Params)

    # Prepare Particle & Hole orbitals ...
    N_Particle, Particle, N_Hole, Hole = orbitals_ph_make(Params,Orb)

    # Prepare 1p-1h phonon states ...
    N_Phonon, Phonon = orbitals_one_phonon_make(N_Particle,Particle,N_Hole,Hole)

    # Count & pre-index all phonon states in JP subspaces ...
    N_nu, Orb_Phonon = HF_RPA_phonon_count(Params,N_Phonon,Phonon)

    # Import transformation matrix mapping LHO & reference RRPA basis ...
    @time C = O1b_import(Params,Orb,"IO/" * Params.Calc.Path * "/Bin/C_RRPA.bin")

    # Import RRPA OBDM ...
    @time Rho = O1b_import(Params,Orb,"IO/" * Params.Calc.Path * "/Bin/Rho_RRPA.bin")

    # Import RRPA solutions ...
    @time X_RRPA, Y_RRPA, E_RRPA = HF_RRPA_import_binary(Params,N_nu)

    # Evaluate the RRPA charge radius chR ...
    chR2 = OBDM_chR2(Params,Orb,C,Rho)

    # Precalculate the reduced matrix elements of Electrogmanetic 1-body transition operators ...
        # Iniztialize the 1-body Electromagnetic transition operators ...
    TrOp = Tr1b_initialize(Params,Orb,chR2)
         # Transform the 1-body transition operators to the reference basis ...
    TrOp = Tr1b_transformation(Params,Orb,C,TrOp)

    # Calculate the reduced transition matrix elements M_nu for each phonon level ...
    rM = HF_RRPA_rM(Params,N_nu,Orb_Phonon,Phonon,Particle,Hole,Rho,TrOp,X_RRPA,Y_RRPA)

    # Evaluate the reduced transition intensities B_nu for each phonon level ...
    rB = HF_RRPA_rB(Params,N_nu,rM,E_RRPA)

    # Export of RRPA electromagnetic transitions ...
    HF_RRPA_export_transitions(Params,Orb,N_nu,C,Rho,E_RRPA,rB)

    return
end

function HF_RRPA_phonon_collectivity(Params::Parameters)
    # Check if the RRPA collectivity directory exists, if not create it ...
    if !(isdir("IO/" * Params.Calc.Path * "/RRPA/Collectivity"))
        mkdir("IO/" * Params.Calc.Path * "/RRPA/Collectivity")
    end

    # Read calculation parameters ...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1

    # Make the list of values of J & P ...
    JP_list = JP_initialize(J_max)

    # Make single-particle orbitals ...
    Orb = orbitals_make(Params)

    # Prepare Particle & Hole orbitals ...
    N_Particle, Particle, N_Hole, Hole = orbitals_ph_make(Params,Orb)

    # Prepare 1p-1h phonon states ...
    N_Phonon, Phonon = orbitals_one_phonon_make(N_Particle,Particle,N_Hole,Hole)

    # Count & pre-index all phonon states in JP subspaces ...
    N_nu, Orb_Phonon = HF_RPA_phonon_count(Params,N_Phonon,Phonon)

    # Import transformation matrix mapping LHO & reference RRPA basis ...
    @time C = O1b_import(Params,Orb,"IO/" * Params.Calc.Path * "/Bin/C_RRPA.bin")

    # Import RRPA OBDM ...
    @time Rho = O1b_import(Params,Orb,"IO/" * Params.Calc.Path * "/Bin/Rho_RRPA.bin")

    # Import RRPA solutions ...
    @time X_RRPA, Y_RRPA, E_RRPA = HF_RRPA_import_binary(Params,N_nu)

    # Initialize fields for many-body collectivity of RRPA phonons ...
    n_phonon = Matrix{Vector{Float64}}(undef,J_max+1,2)
    S_phonon = Matrix{Vector{Float64}}(undef,J_max+1,2)

    # Evaluate the many-body collectivity of RRPA phonons ...
    @inbounds Threads.@threads for JP in JP_list
        J, P = JP[1], JP[2]
        N_ph = N_nu[J+1,P]
        n_JP = zeros(Float64,N_ph)
        S_JP = zeros(Float64,N_ph)

        X_JP, Y_JP = X_RRPA[J+1,P], Y_RRPA[J+1,P]

        @inbounds for nu in 1:N_ph
            nSum, SSum = 0.0, 0.0

            @inbounds for ph in 1:N_ph
                w = abs2(X_JP[ph,nu]) - abs2(Y_JP[ph,nu])

                nSum += w^2
                
                if w > 1e-15
                    SSum -= w * log(w)
                else
                    SSum += 0.0
                end
            end

            n_JP[nu] = 1.0 / sqrt(nSum)
            S_JP[nu] = SSum
        end

        n_phonon[J+1,P] = n_JP
        S_phonon[J+1,P] = S_JP
    end

    # Export the measures of many-body collectivity of RRPA phonons ...
    @inbounds Threads.@threads for JP in JP_list
        J, P = JP[1], JP[2]
        N_ph = N_nu[J+1,P]
        E_JP = E_RRPA[J+1,P]
        n_JP = n_phonon[J+1,P]
        S_JP = S_phonon[J+1,P]

        sP = (P == 2) ? "-" : "+"
        
        Output_File = "IO/" * Params.Calc.Path * "/RRPA/Collectivity/RRPA_Phonon_Collectivity_J$(J)_P$(sP).dat"
        open(Output_File, "w") do Write_File
            @printf(Write_File, "%-10s %-12s %-12s %-12s\n", "nu", "E_nu", "n_nu", "S_nu")
            @inbounds for nu in 1:N_ph
                E_nu, n_nu, S_nu = E_JP[nu], n_JP[nu], S_JP[nu]
                @printf(Write_File, "%-10d %-12.8f %-12.8f %-12.8f\n", nu, E_nu, n_nu, S_nu)
            end
        end
    end

    return
end