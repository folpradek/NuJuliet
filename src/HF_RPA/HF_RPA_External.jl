function HF_RPA_import_binary(Params::Parameters,N_nu::Matrix{Int64})
    # Parameters / dimensions
    N_max = Params.Calc.Nmax
    J_max = 2*N_max + 1
    OutputFile = Params.Calc.Path

    # Initialite the list of values of J & P for iteration ...
    JP_list = JP_initialize(J_max)

    println("\nImporting TDA & RPA solutions from binary format ...")

    # Initialize amplitudes & energies ...
        # Case of TDA ...
    X_TDA = Matrix{Matrix{Float64}}(undef,J_max+1,2)
    E_TDA = Matrix{Vector{Float64}}(undef,J_max+1,2)
        # Case of RPA ...
    X_RPA = Matrix{Matrix{ComplexF64}}(undef,J_max+1,2)
    Y_RPA = Matrix{Matrix{ComplexF64}}(undef,J_max+1,2)
    E_RPA = Matrix{Vector{ComplexF64}}(undef,J_max+1,2)

    # Initialite each J & P block ...
    @inbounds for JP in JP_list
        J, P = JP[1], JP[2]
        N_ph = N_nu[J+1,P]
        X_TDA[J+1,P] = Matrix{Float64}(undef,N_ph,N_ph)
        E_TDA[J+1,P] = Vector{Float64}(undef,N_ph)
        X_RPA[J+1,P] = Matrix{ComplexF64}(undef,N_ph,N_ph)
        Y_RPA[J+1,P] = Matrix{ComplexF64}(undef,N_ph,N_ph)
        E_RPA[J+1,P] = Vector{ComplexF64}(undef,N_ph)
    end

    # Read TDA amplitudes X ...
    Import_Path_TDA_X = "IO/$(OutputFile)/Bin/TDA_X.bin"
    open(Import_Path_TDA_X, "r") do Import_File
        @inbounds for JP in JP_list
            J, P = JP[1], JP[2]
            N_ph = N_nu[J+1,P]
            N_ph == 0 && continue
            Buffer = Vector{Float64}(undef,N_ph*N_ph)
            read!(Import_File,Buffer)
            X_TDA[J+1,P] .= reshape(Buffer,N_ph,N_ph)
        end
    end

    # Read TDA energies E ...
    Import_Path_TDA_E = "IO/$(OutputFile)/Bin/TDA_E.bin"
    open(Import_Path_TDA_E, "r") do Import_File
        @inbounds for JP in JP_list
            J, P = JP[1], JP[2]
            N_ph = N_nu[J+1,P]
            N_ph == 0 && continue
            Buffer = Vector{Float64}(undef,N_ph)
            read!(Import_File,Buffer)
            E_TDA[J+1,P] .= Buffer
        end
    end

    # Read RPA amplitudes X ...
    Import_Path_RPA_X = "IO/$(OutputFile)/Bin/RPA_X.bin"
    open(Import_Path_RPA_X, "r") do Import_File
        @inbounds for JP in JP_list
            J, P = JP[1], JP[2]
            N_ph = N_nu[J+1,P]
            N_ph == 0 && continue
            Buffer = Vector{ComplexF64}(undef,N_ph*N_ph)
            read!(Import_File,Buffer)
            X_RPA[J+1,P] .= reshape(Buffer,N_ph,N_ph)
        end
    end

    # Read RPA amplitudes Y ...
    Import_Path_RPA_Y = "IO/$(OutputFile)/Bin/RPA_Y.bin"
    open(Import_Path_RPA_Y, "r") do Import_File
        @inbounds for JP in JP_list
            J, P = JP[1], JP[2]
            N_ph = N_nu[J+1,P]
            N_ph == 0 && continue
            Buffer = Vector{ComplexF64}(undef,N_ph*N_ph)
            read!(Import_File,Buffer)
            Y_RPA[J+1,P] .= reshape(Buffer,N_ph,N_ph)
        end
    end

    # Read RPA energies E ...
    Import_Path_RPA_E = "IO/$(OutputFile)/Bin/RPA_E.bin"
    open(Import_Path_RPA_E, "r") do Import_File
        @inbounds for JP in JP_list
            J, P = JP[1], JP[2]
            N_ph = N_nu[J+1,P]
            N_ph == 0 && continue
            Buffer = Vector{ComplexF64}(undef,N_ph)
            read!(Import_File,Buffer)
            E_RPA[J+1,P] .= Buffer
        end
    end

    println("\tTDA & RPA solutions have been succesufully imported ...")

    return X_TDA, E_TDA, X_RPA, Y_RPA, E_RPA
end

function HF_RPA_transition_densities(Params::Parameters;JP::String,nu_list::Vector{Int64},File_Name::String = "Phonon_Densities")
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

    # Make single-particle orbitals ...
    Orb = orbitals_make(Params)

    # Prepare Particle & Hole orbitals ...
    N_Particle, Particle, N_Hole, Hole = orbitals_ph_make(Params,Orb)

    # Prepare 1p-1h phonon states ...
    N_Phonon, Phonon = orbitals_one_phonon_make(N_Particle,Particle,N_Hole,Hole)

    # Count & pre-index all phonon states in JP subspaces ...
    N_nu, Orb_Phonon = HF_RPA_phonon_count(Params,N_Phonon,Phonon)

    # Import transformation matrix mapping LHO & reference HF basis ...
    @time C = O1b_import(Params,Orb,"IO/" * Params.Calc.Path * "/Bin/C_HF.bin")

    # Import TDA & RPA solutions ...
    @time X_TDA, E_TDA, X_RPA, Y_RPA, E_RPA = HF_RPA_import_binary(Params,N_nu)

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
            # Case of TDA transition matrix elements ...
    pM_TDA, nM_TDA = Vector{ComplexF64}(undef,N_phonon), Vector{ComplexF64}(undef,N_phonon)
    isM_TDA, ivM_TDA = Vector{ComplexF64}(undef,N_phonon), Vector{ComplexF64}(undef,N_phonon)
            # Case of RPA transition matrix elements ...
    pM_RPA, nM_RPA = Vector{ComplexF64}(undef,N_phonon), Vector{ComplexF64}(undef,N_phonon)
    isM_RPA, ivM_RPA = Vector{ComplexF64}(undef,N_phonon), Vector{ComplexF64}(undef,N_phonon)

        # Calculate the matrix elements for given phonon levels ...
    if (J,P) in [(0,1),(2,1),(3,2)]
        @inbounds for (nu_ind, nu) in enumerate(nu_list)
            pMTDASum, nMTDASum, isMTDASum, ivMTDASum = 0.0im, 0.0im, 0.0im, 0.0im
            pMRPASum, nMRPASum, isMRPASum, ivMRPASum = 0.0im, 0.0im, 0.0im, 0.0im

            @inbounds for ph in 1:N_ph
                i_ph = Orb_Phonon[J+1,P][ph]
                p, h, t_ph = Phonon[i_ph].p, Phonon[i_ph].h, Phonon[i_ph].tz
                x_TDA, x_RPA, y_RPA = X_TDA[J+1,P][ph,nu], X_RPA[J+1,P][ph,nu], Y_RPA[J+1,P][ph,nu]

                MTDAME = phase(J) * x_TDA / sqrt(Float64(2*J + 1))
                MRPAME = (phase(J) * x_RPA + y_RPA) / sqrt(Float64(2*J + 1))

                if t_ph == -1
                    a_p, a_h = Particle.p[p].a, Hole.p[h].a

                    if J == 0 && P == 1
                        MTDAME = TrOp.E0.p[a_p,a_h] * MTDAME
                        MRPAME = TrOp.E0.p[a_p,a_h] * MRPAME

                    elseif J == 2 && P == 1
                        MTDAME = TrOp.E2.p[a_p,a_h] * MTDAME
                        MRPAME = TrOp.E2.p[a_p,a_h] * MRPAME

                    elseif J == 3 && P == 2
                        MTDAME = TrOp.E3.p[a_p,a_h] * MTDAME
                        MRPAME = TrOp.E3.p[a_p,a_h] * MRPAME
                    end

                    pMTDASum += MTDAME
                    isMTDASum += 0.5 * MTDAME
                    ivMTDASum += 0.5 * MTDAME

                    pMRPASum += MRPAME
                    isMRPASum += 0.5 * MRPAME
                    ivMRPASum += 0.5 * MRPAME

                elseif t_ph == 1
                    a_p, a_h = Particle.n[p].a, Hole.n[h].a

                    if J == 0 && P == 1
                        MTDAME = TrOp.E0.n[a_p,a_h] * MTDAME
                        MRPAME = TrOp.E0.n[a_p,a_h] * MRPAME

                    elseif J == 2 && P == 1
                        MTDAME = TrOp.E2.n[a_p,a_h] * MTDAME
                        MRPAME = TrOp.E2.n[a_p,a_h] * MRPAME

                    elseif J == 3 && P == 2
                        MTDAME = TrOp.E3.n[a_p,a_h] * MTDAME
                        MRPAME = TrOp.E3.n[a_p,a_h] * MRPAME
                    end

                    nMTDASum += MTDAME
                    isMTDASum += 0.5 * MTDAME
                    ivMTDASum -= 0.5 * MTDAME

                    nMRPASum += MRPAME
                    isMRPASum += 0.5 * MRPAME
                    ivMRPASum -= 0.5 * MRPAME
                end

            end

            pM_TDA[nu_ind] = pMTDASum
            nM_TDA[nu_ind] = nMTDASum
            isM_TDA[nu_ind] = isMTDASum
            ivM_TDA[nu_ind] = ivMTDASum

            pM_RPA[nu_ind] = pMRPASum
            nM_RPA[nu_ind] = nMRPASum
            isM_RPA[nu_ind] = isMRPASum
            ivM_RPA[nu_ind] = ivMRPASum
        end

    elseif (J,P) == (1,2)
        @inbounds for (nu_ind, nu) in enumerate(nu_list)
            pMTDASum, nMTDASum, isMTDASum, ivMTDASum = 0.0im, 0.0im, 0.0im, 0.0im
            pMRPASum, nMRPASum, isMRPASum, ivMRPASum = 0.0im, 0.0im, 0.0im, 0.0im

            @inbounds for ph in 1:N_ph
                i_ph = Orb_Phonon[J+1,P][ph]
                p, h, t_ph = Phonon[i_ph].p, Phonon[i_ph].h, Phonon[i_ph].tz
                x_TDA, x_RPA, y_RPA = X_TDA[J+1,P][ph,nu], X_RPA[J+1,P][ph,nu], Y_RPA[J+1,P][ph,nu]

                Amp_TDA = phase(J) * x_TDA / sqrt(Float64(2*J + 1))
                Amp_RPA = (phase(J) * x_RPA + y_RPA) / sqrt(Float64(2*J + 1))

                if t_ph == -1
                    a_p, a_h = Particle.p[p].a, Hole.p[h].a

                    pMTDAME = TrOp.E1.p[a_p,a_h] * Amp_TDA
                    isMTDAME = 0.5 * TrOp.E1_C.p[a_p,a_h] * Amp_TDA
                    ivMTDAME = TrOp.E1.p[a_p,a_h] * Amp_TDA * (A - Z) / A

                    pMRPAME = TrOp.E1.p[a_p,a_h] * Amp_RPA
                    isMRPAME = 0.5 * TrOp.E1_C.p[a_p,a_h] * Amp_RPA
                    ivMRPAME = TrOp.E1.p[a_p,a_h] * Amp_RPA * (A - Z) / A

                    pMTDASum += pMTDAME
                    isMTDASum += isMTDAME
                    ivMTDASum += ivMTDAME

                    pMRPASum += pMRPAME
                    isMRPASum += isMRPAME
                    ivMRPASum += ivMRPAME

                elseif t_ph == 1
                    a_p, a_h = Particle.n[p].a, Hole.n[h].a

                    nMTDAME = TrOp.E1.n[a_p,a_h] * Amp_TDA
                    isMTDAME = 0.5 * TrOp.E1_C.n[a_p,a_h] * Amp_TDA
                    ivMTDAME = - TrOp.E1.n[a_p,a_h] * Amp_TDA * Z / A

                    nMRPAME = TrOp.E1.n[a_p,a_h] * Amp_RPA
                    isMRPAME = 0.5 * TrOp.E1_C.n[a_p,a_h] * Amp_RPA
                    ivMRPAME = - TrOp.E1.n[a_p,a_h] * Amp_RPA * Z / A

                    nMTDASum += nMTDAME
                    isMTDASum += isMTDAME
                    ivMTDASum += ivMTDAME

                    nMRPASum += nMRPAME
                    isMRPASum += isMRPAME
                    ivMRPASum += ivMRPAME
                end
            end

            pM_TDA[nu_ind] = pMTDASum
            nM_TDA[nu_ind] = nMTDASum
            isM_TDA[nu_ind] = isMTDASum
            ivM_TDA[nu_ind] = ivMTDASum

            pM_RPA[nu_ind] = pMRPASum
            nM_RPA[nu_ind] = nMRPASum
            isM_RPA[nu_ind] = isMRPASum
            ivM_RPA[nu_ind] = ivMRPASum
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
        # Case of TDA ...
    pRho_nu_TDA = Matrix{Float64}(undef,N_phonon,N_grid)
    nRho_nu_TDA = Matrix{Float64}(undef,N_phonon,N_grid)
        # Case of RPA ...
    pRho_nu_RPA = Matrix{Float64}(undef,N_phonon,N_grid)
    nRho_nu_RPA = Matrix{Float64}(undef,N_phonon,N_grid)

    # For given phonon levels evaluate their radial representation ...
    @inbounds Threads.@threads for i in 1:N_grid
        r = r_grid[i]
        @inbounds for (nu_ind, nu) in enumerate(nu_list)
            pTDASum, nTDASum = 0.0, 0.0
            pRPASum, nRPASum = 0.0, 0.0

            Gauge_TDA, Gauge_RPA = 1.0, 1.0

            if ((J,P) in [(0,1),(1,2),(2,1),(3,2)]) == true
                Gauge_TDA = pM_TDA[nu_ind] / abs(pM_TDA[nu_ind])
                Gauge_RPA = real(pM_RPA[nu_ind]) / abs(pM_RPA[nu_ind])
            end

            @inbounds for ph in 1:N_ph
                i_ph = Orb_Phonon[J+1,P][ph]
                p, h, t_ph = Phonon[i_ph].p, Phonon[i_ph].h, Phonon[i_ph].tz
                x_TDA, x_RPA, y_RPA = X_TDA[J+1,P][ph,nu], X_RPA[J+1,P][ph,nu], Y_RPA[J+1,P][ph,nu]

                if t_ph == -1
                    a_p, a_h = Particle.p[p].a, Hole.p[h].a
                    pRho_TDA = Gauge_TDA * phase(J) * x_TDA * pOrb_grid[a_p,i] * pOrb_grid[a_h,i]
                    pRho_RPA = Gauge_RPA * real(phase(J) * x_RPA + y_RPA) * pOrb_grid[a_p,i] * pOrb_grid[a_h,i]
                    pTDASum += pRho_TDA
                    pRPASum += pRho_RPA
                elseif t_ph == 1
                    a_p, a_h = Particle.n[p].a, Hole.n[h].a
                    nRho_TDA = Gauge_TDA * phase(J) * x_TDA * nOrb_grid[a_p,i] * nOrb_grid[a_h,i]
                    nRho_RPA = Gauge_RPA * real(phase(J) * x_RPA + y_RPA) * nOrb_grid[a_p,i] * nOrb_grid[a_h,i]
                    nTDASum += nRho_TDA
                    nRPASum += nRho_RPA
                end
            end

            Amp = r^2 / Float64(2*J + 1)

            pRho_nu_TDA[nu_ind,i] = Amp * pTDASum
            nRho_nu_TDA[nu_ind,i] = Amp * nTDASum
            pRho_nu_RPA[nu_ind,i] = Amp * pRPASum
            nRho_nu_RPA[nu_ind,i] = Amp * nRPASum
        end
    end

    # Perform the export of indiviual radial phonon densities ...
    @inbounds for (nu_ind, nu) in enumerate(nu_list)
        e_TDA, e_RPA = E_TDA[J+1,P][nu], real(E_RPA[J+1,P][nu])
        sE_TDA, sE_RPA = string(round(e_TDA,digits=3)), string(round(e_RPA,digits=3))

        # Case of TDA densities ...
        Output_File = "IO/" * Params.Calc.Path * "/RPA/Densities/TDA_" * File_Name * "_JP$(JP)_nu$(nu)_E$(sE_TDA).dat"
        open(Output_File, "w") do Write_File
            @printf(Write_File, "%-20s %-20s %-20s %-20s %-20s\n", "r", "rho_p", "rho_n", "rho_is", "rho_iv")
            @inbounds for i in 1:N_grid
                r, pRho, nRho = r_grid[i], pRho_nu_TDA[nu_ind,i], nRho_nu_TDA[nu_ind,i]
                isRho = 0.5 * (pRho + nRho)
                ivRho = 0.5 * (pRho - nRho)
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f %-20.8f\n", r, pRho, nRho, isRho, ivRho)
            end
        end

        # Case of RPA densities ...
        Output_File = "IO/" * Params.Calc.Path * "/RPA/Densities/RPA_" * File_Name * "_JP$(JP)_nu$(nu)_E$(sE_RPA).dat"
        open(Output_File, "w") do Write_File
            @printf(Write_File, "%-20s %-20s %-20s %-20s %-20s\n", "r", "rho_p", "rho_n", "rho_is", "rho_iv")
            @inbounds for i in 1:N_grid
                r, pRho, nRho = r_grid[i], pRho_nu_RPA[nu_ind,i], nRho_nu_RPA[nu_ind,i]
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
        # Case of TDA transition matrix elements ...
    pM_TDA .= abs.(pM_TDA) ./ sqrt(sum(pM_TDA.^2))
    nM_TDA .= abs.(nM_TDA) ./ sqrt(sum(nM_TDA.^2))
    isM_TDA .= abs.(isM_TDA) ./ sqrt(sum(isM_TDA.^2))
    ivM_TDA .= abs.(ivM_TDA) ./ sqrt(sum(ivM_TDA.^2))
        # Case of RPA transition matrix elements ...
    pM_RPA .= abs.(pM_RPA) ./ sqrt(sum(pM_RPA.^2))
    nM_RPA .= abs.(nM_RPA) ./ sqrt(sum(nM_RPA.^2))
    isM_RPA .= abs.(isM_RPA) ./ sqrt(sum(isM_RPA.^2))
    ivM_RPA .= abs.(ivM_RPA) ./ sqrt(sum(ivM_RPA.^2))

    # Perform the export of of averaged transition current maps ...
        # Export TDA averaged transition radial densities ...
    Output_File = "IO/" * Params.Calc.Path * "/RPA/Densities/TDA_" * File_Name * "_JP$(JP)_Averaged.dat"
    open(Output_File, "w") do Write_File
        @printf(Write_File, "%-20s %-20s %-20s %-20s %-20s\n", "r", "rho_p", "rho_n", "rho_is", "rho_iv")
        @inbounds for i in 1:N_grid
            r = r_grid[i]
            pRho, nRho, isRho, ivRho = 0.0, 0.0, 0.0, 0.0
            @inbounds for nu_ind in 1:N_phonon
                prho, nrho = pRho_nu_TDA[nu_ind,i], nRho_nu_TDA[nu_ind,i]
                pRho += pM_TDA[nu_ind] * prho
                nRho += nM_TDA[nu_ind] * nrho
                isRho += 0.5 * isM_TDA[nu_ind] * (prho + nrho)
                ivRho += 0.5 * ivM_TDA[nu_ind] * (prho - nrho)
            end
            @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f %-20.8f\n", r, pRho, nRho, isRho, ivRho)
        end
    end
        # Export RPA averaged transition radial densities ...
    Output_File_RPA = "IO/" * Params.Calc.Path * "/RPA/Densities/RPA_" * File_Name * "_JP$(JP)_Averaged.dat"
    open(Output_File_RPA, "w") do Write_File
        @printf(Write_File, "%-20s %-20s %-20s %-20s %-20s\n", "r", "rho_p", "rho_n", "rho_is", "rho_iv")
        @inbounds for i in 1:N_grid
            r = r_grid[i]
            pRho, nRho, isRho, ivRho = 0.0, 0.0, 0.0, 0.0
            @inbounds for nu_ind in 1:N_phonon
                prho, nrho = pRho_nu_RPA[nu_ind,i], nRho_nu_RPA[nu_ind,i]
                pRho += real(pM_RPA[nu_ind]) * prho
                nRho += real(nM_RPA[nu_ind]) * nrho
                isRho += 0.5 * real(isM_RPA[nu_ind]) * (prho + nrho)
                ivRho += 0.5 * real(ivM_RPA[nu_ind]) * (prho - nrho)
            end
            @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f %-20.8f\n", r, pRho, nRho, isRho, ivRho)
        end
    end


    display("Export of averaged phonon radial transitions densities successfuly completed ...")

    return
end

function HF_RPA_transition_operator_densities(Params::Parameters;JP::String,nu_list::Vector{Int64},File_Name::String = "Phonon_Transition_Operator_Densities")
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

    # Import transformation matrix mapping LHO & reference HF basis ...
    @time C = O1b_import(Params,Orb,"IO/" * Params.Calc.Path * "/Bin/C_HF.bin")

    # Import TDA & RPA solutions ...
    @time X_TDA, E_TDA, X_RPA, Y_RPA, E_RPA = HF_RPA_import_binary(Params,N_nu)

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
            # Case of TDA transition matrix elements ...
    pM_TDA, nM_TDA = Vector{ComplexF64}(undef,N_phonon), Vector{ComplexF64}(undef,N_phonon)
    isM_TDA, ivM_TDA = Vector{ComplexF64}(undef,N_phonon), Vector{ComplexF64}(undef,N_phonon)
            # Case of RPA transition matrix elements ...
    pM_RPA, nM_RPA = Vector{ComplexF64}(undef,N_phonon), Vector{ComplexF64}(undef,N_phonon)
    isM_RPA, ivM_RPA = Vector{ComplexF64}(undef,N_phonon), Vector{ComplexF64}(undef,N_phonon)

        # Calculate the matrix elements for given phonon levels ...
    if (J,P) in [(0,1),(2,1),(3,2)]
        @inbounds for (nu_ind, nu) in enumerate(nu_list)
            pMTDASum, nMTDASum, isMTDASum, ivMTDASum = 0.0im, 0.0im, 0.0im, 0.0im
            pMRPASum, nMRPASum, isMRPASum, ivMRPASum = 0.0im, 0.0im, 0.0im, 0.0im

            @inbounds for ph in 1:N_ph
                i_ph = Orb_Phonon[J+1,P][ph]
                p, h, t_ph = Phonon[i_ph].p, Phonon[i_ph].h, Phonon[i_ph].tz
                x_TDA, x_RPA, y_RPA = X_TDA[J+1,P][ph,nu], X_RPA[J+1,P][ph,nu], Y_RPA[J+1,P][ph,nu]

                MTDAME = phase(J) * x_TDA / sqrt(Float64(2*J + 1))
                MRPAME = (phase(J) * x_RPA + y_RPA) / sqrt(Float64(2*J + 1))

                if t_ph == -1
                    a_p, a_h = Particle.p[p].a, Hole.p[h].a

                    if J == 0 && P == 1
                        MTDAME = TrOp.E0.p[a_p,a_h] * MTDAME
                        MRPAME = TrOp.E0.p[a_p,a_h] * MRPAME

                    elseif J == 2 && P == 1
                        MTDAME = TrOp.E2.p[a_p,a_h] * MTDAME
                        MRPAME = TrOp.E2.p[a_p,a_h] * MRPAME

                    elseif J == 3 && P == 2
                        MTDAME = TrOp.E3.p[a_p,a_h] * MTDAME
                        MRPAME = TrOp.E3.p[a_p,a_h] * MRPAME
                    end

                    pMTDASum += MTDAME
                    isMTDASum += 0.5 * MTDAME
                    ivMTDASum += 0.5 * MTDAME

                    pMRPASum += MRPAME
                    isMRPASum += 0.5 * MRPAME
                    ivMRPASum += 0.5 * MRPAME

                elseif t_ph == 1
                    a_p, a_h = Particle.n[p].a, Hole.n[h].a

                    if J == 0 && P == 1
                        MTDAME = TrOp.E0.n[a_p,a_h] * MTDAME
                        MRPAME = TrOp.E0.n[a_p,a_h] * MRPAME

                    elseif J == 2 && P == 1
                        MTDAME = TrOp.E2.n[a_p,a_h] * MTDAME
                        MRPAME = TrOp.E2.n[a_p,a_h] * MRPAME

                    elseif J == 3 && P == 2
                        MTDAME = TrOp.E3.n[a_p,a_h] * MTDAME
                        MRPAME = TrOp.E3.n[a_p,a_h] * MRPAME
                    end

                    nMTDASum += MTDAME
                    isMTDASum += 0.5 * MTDAME
                    ivMTDASum -= 0.5 * MTDAME

                    nMRPASum += MRPAME
                    isMRPASum += 0.5 * MRPAME
                    ivMRPASum -= 0.5 * MRPAME
                end

            end

            pM_TDA[nu_ind] = pMTDASum
            nM_TDA[nu_ind] = nMTDASum
            isM_TDA[nu_ind] = isMTDASum
            ivM_TDA[nu_ind] = ivMTDASum

            pM_RPA[nu_ind] = pMRPASum
            nM_RPA[nu_ind] = nMRPASum
            isM_RPA[nu_ind] = isMRPASum
            ivM_RPA[nu_ind] = ivMRPASum
        end

    elseif (J,P) == (1,2)
        @inbounds for (nu_ind, nu) in enumerate(nu_list)
            pMTDASum, nMTDASum, isMTDASum, ivMTDASum = 0.0im, 0.0im, 0.0im, 0.0im
            pMRPASum, nMRPASum, isMRPASum, ivMRPASum = 0.0im, 0.0im, 0.0im, 0.0im

            @inbounds for ph in 1:N_ph
                i_ph = Orb_Phonon[J+1,P][ph]
                p, h, t_ph = Phonon[i_ph].p, Phonon[i_ph].h, Phonon[i_ph].tz
                x_TDA, x_RPA, y_RPA = X_TDA[J+1,P][ph,nu], X_RPA[J+1,P][ph,nu], Y_RPA[J+1,P][ph,nu]

                Amp_TDA = phase(J) * x_TDA / sqrt(Float64(2*J + 1))
                Amp_RPA = (phase(J) * x_RPA + y_RPA) / sqrt(Float64(2*J + 1))

                if t_ph == -1
                    a_p, a_h = Particle.p[p].a, Hole.p[h].a

                    pMTDAME = TrOp.E1.p[a_p,a_h] * Amp_TDA
                    isMTDAME = 0.5 * TrOp.E1_C.p[a_p,a_h] * Amp_TDA
                    ivMTDAME = TrOp.E1.p[a_p,a_h] * Amp_TDA * (A - Z) / A

                    pMRPAME = TrOp.E1.p[a_p,a_h] * Amp_RPA
                    isMRPAME = 0.5 * TrOp.E1_C.p[a_p,a_h] * Amp_RPA
                    ivMRPAME = TrOp.E1.p[a_p,a_h] * Amp_RPA * (A - Z) / A

                    pMTDASum += pMTDAME
                    isMTDASum += isMTDAME
                    ivMTDASum += ivMTDAME

                    pMRPASum += pMRPAME
                    isMRPASum += isMRPAME
                    ivMRPASum += ivMRPAME

                elseif t_ph == 1
                    a_p, a_h = Particle.n[p].a, Hole.n[h].a

                    nMTDAME = TrOp.E1.n[a_p,a_h] * Amp_TDA
                    isMTDAME = 0.5 * TrOp.E1_C.n[a_p,a_h] * Amp_TDA
                    ivMTDAME = - TrOp.E1.n[a_p,a_h] * Amp_TDA * Z / A

                    nMRPAME = TrOp.E1.n[a_p,a_h] * Amp_RPA
                    isMRPAME = 0.5 * TrOp.E1_C.n[a_p,a_h] * Amp_RPA
                    ivMRPAME = - TrOp.E1.n[a_p,a_h] * Amp_RPA * Z / A

                    nMTDASum += nMTDAME
                    isMTDASum += isMTDAME
                    ivMTDASum += ivMTDAME

                    nMRPASum += nMRPAME
                    isMRPASum += isMRPAME
                    ivMRPASum += ivMRPAME
                end
            end

            pM_TDA[nu_ind] = pMTDASum
            nM_TDA[nu_ind] = nMTDASum
            isM_TDA[nu_ind] = isMTDASum
            ivM_TDA[nu_ind] = ivMTDASum

            pM_RPA[nu_ind] = pMRPASum
            nM_RPA[nu_ind] = nMRPASum
            isM_RPA[nu_ind] = isMRPASum
            ivM_RPA[nu_ind] = ivMRPASum
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
        # Case of TDA ...
    pRho_nu_TDA = Matrix{Float64}(undef,N_phonon,N_grid)
    nRho_nu_TDA = Matrix{Float64}(undef,N_phonon,N_grid)
        # Case of RPA ...
    pRho_nu_RPA = Matrix{Float64}(undef,N_phonon,N_grid)
    nRho_nu_RPA = Matrix{Float64}(undef,N_phonon,N_grid)

    # For given phonon levels evaluate their radial representation ...
    @inbounds Threads.@threads for i in 1:N_grid
        r = r_grid[i]
        @inbounds for (nu_ind, nu) in enumerate(nu_list)
            pTDASum, nTDASum = 0.0, 0.0
            pRPASum, nRPASum = 0.0, 0.0

            Gauge_TDA, Gauge_RPA = 1.0, 1.0

            if ((J,P) in [(0,1),(1,2),(2,1),(3,2)]) == true
                Gauge_TDA = pM_TDA[nu_ind] / abs(pM_TDA[nu_ind])
                Gauge_RPA = real(pM_RPA[nu_ind]) / abs(pM_RPA[nu_ind])
            end

            @inbounds for ph in 1:N_ph
                i_ph = Orb_Phonon[J+1,P][ph]
                p, h, t_ph = Phonon[i_ph].p, Phonon[i_ph].h, Phonon[i_ph].tz
                x_TDA, x_RPA, y_RPA = X_TDA[J+1,P][ph,nu], X_RPA[J+1,P][ph,nu], Y_RPA[J+1,P][ph,nu]

                if t_ph == -1
                    a_p, a_h = Particle.p[p].a, Hole.p[h].a
                    pM = pM_grid[a_p,a_h,i]
                    pRho_TDA = Gauge_TDA * phase(J) * x_TDA * pM 
                    pRho_RPA = Gauge_RPA * real(phase(J) * x_RPA + y_RPA) * pM
                    pTDASum += pRho_TDA
                    pRPASum += pRho_RPA
                elseif t_ph == 1
                    a_p, a_h = Particle.n[p].a, Hole.n[h].a
                    nM = nM_grid[a_p,a_h,i]
                    nRho_TDA = Gauge_TDA * phase(J) * x_TDA * nM
                    nRho_RPA = Gauge_RPA * real(phase(J) * x_RPA + y_RPA) * nM
                    nTDASum += nRho_TDA
                    nRPASum += nRho_RPA
                end
            end

            Amp = r^2 / Float64(2*J + 1)

            pRho_nu_TDA[nu_ind,i] = Amp * pTDASum
            nRho_nu_TDA[nu_ind,i] = Amp * nTDASum
            pRho_nu_RPA[nu_ind,i] = Amp * pRPASum
            nRho_nu_RPA[nu_ind,i] = Amp * nRPASum
        end
    end

    # Perform the export of indiviual radial phonon densities ...
    @inbounds for (nu_ind, nu) in enumerate(nu_list)
        e_TDA, e_RPA = E_TDA[J+1,P][nu], real(E_RPA[J+1,P][nu])
        sE_TDA, sE_RPA = string(round(e_TDA,digits=3)), string(round(e_RPA,digits=3))

        # Case of TDA densities ...
        Output_File = "IO/" * Params.Calc.Path * "/RPA/Densities/TDA_" * File_Name * "_JP$(JP)_nu$(nu)_E$(sE_TDA).dat"
        open(Output_File, "w") do Write_File
            @printf(Write_File, "%-20s %-20s %-20s %-20s %-20s\n", "r", "pM", "nM", "isM", "ivM")
            @inbounds for i in 1:N_grid
                r, pRho, nRho = r_grid[i], pRho_nu_TDA[nu_ind,i], nRho_nu_TDA[nu_ind,i]
                isRho = 0.5 * (pRho + nRho)
                ivRho = 0.5 * (pRho - nRho)
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f %-20.8f\n", r, pRho, nRho, isRho, ivRho)
            end
        end

        # Case of RPA densities ...
        Output_File = "IO/" * Params.Calc.Path * "/RPA/Densities/RPA_" * File_Name * "_JP$(JP)_nu$(nu)_E$(sE_RPA).dat"
        open(Output_File, "w") do Write_File
            @printf(Write_File, "%-20s %-20s %-20s %-20s %-20s\n", "r", "pM", "nM", "isM", "ivM")
            @inbounds for i in 1:N_grid
                r, pRho, nRho = r_grid[i], pRho_nu_RPA[nu_ind,i], nRho_nu_RPA[nu_ind,i]
                isRho = 0.5 * (pRho + nRho)
                ivRho = 0.5 * (pRho - nRho)
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f %-20.8f\n", r, pRho, nRho, isRho, ivRho)
            end
        end

    end

    display("Export of individual phonon radial transition densities successfuly completed ...")

    # Renormalize M ... to reasonable scale averaged densities ...
        # Case of TDA transition matrix elements ...
    pM_TDA .= abs.(pM_TDA) ./ sqrt(sum(pM_TDA.^2))
    nM_TDA .= abs.(nM_TDA) ./ sqrt(sum(nM_TDA.^2))
    isM_TDA .= abs.(isM_TDA) ./ sqrt(sum(isM_TDA.^2))
    ivM_TDA .= abs.(ivM_TDA) ./ sqrt(sum(ivM_TDA.^2))
        # Case of RPA transition matrix elements ...
    pM_RPA .= abs.(pM_RPA) ./ sqrt(sum(pM_RPA.^2))
    nM_RPA .= abs.(nM_RPA) ./ sqrt(sum(nM_RPA.^2))
    isM_RPA .= abs.(isM_RPA) ./ sqrt(sum(isM_RPA.^2))
    ivM_RPA .= abs.(ivM_RPA) ./ sqrt(sum(ivM_RPA.^2))

    # Perform the export of of averaged transition matrix current maps ...
        # Export TDA averaged transition matrix radial densities ...
    Output_File = "IO/" * Params.Calc.Path * "/RPA/Densities/TDA_" * File_Name * "_JP$(JP)_Averaged.dat"
    open(Output_File, "w") do Write_File
        @printf(Write_File, "%-20s %-20s %-20s %-20s %-20s\n", "r", "rho_p", "rho_n", "rho_is", "rho_iv")
        @inbounds for i in 1:N_grid
            r = r_grid[i]
            pRho, nRho, isRho, ivRho = 0.0, 0.0, 0.0, 0.0
            @inbounds for nu_ind in 1:N_phonon
                prho, nrho = pRho_nu_TDA[nu_ind,i], nRho_nu_TDA[nu_ind,i]
                pRho += pM_TDA[nu_ind] * prho
                nRho += nM_TDA[nu_ind] * nrho
                isRho += 0.5 * isM_TDA[nu_ind] * (prho + nrho)
                ivRho += 0.5 * ivM_TDA[nu_ind] * (prho - nrho)
            end
            @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f %-20.8f\n", r, pRho, nRho, isRho, ivRho)
        end
    end
        # Export RPA averaged transition matrix radial densities ...
    Output_File_RPA = "IO/" * Params.Calc.Path * "/RPA/Densities/RPA_" * File_Name * "_JP$(JP)_Averaged.dat"
    open(Output_File_RPA, "w") do Write_File
        @printf(Write_File, "%-20s %-20s %-20s %-20s %-20s\n", "r", "rho_p", "rho_n", "rho_is", "rho_iv")
        @inbounds for i in 1:N_grid
            r = r_grid[i]
            pRho, nRho, isRho, ivRho = 0.0, 0.0, 0.0, 0.0
            @inbounds for nu_ind in 1:N_phonon
                prho, nrho = pRho_nu_RPA[nu_ind,i], nRho_nu_RPA[nu_ind,i]
                pRho += real(pM_RPA[nu_ind]) * prho
                nRho += real(nM_RPA[nu_ind]) * nrho
                isRho += 0.5 * real(isM_RPA[nu_ind]) * (prho + nrho)
                ivRho += 0.5 * real(ivM_RPA[nu_ind]) * (prho - nrho)
            end
            @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f %-20.8f\n", r, pRho, nRho, isRho, ivRho)
        end
    end


    display("Export of averaged phonon radial transitions densities successfuly completed ...")

    return
end

function HF_RPA_transition_analysis(Params::Parameters;JP::String,nu_list::Vector{Int64},File_Name::String = "Phonon_Transtion_Analysis", Cut::Float64 = 0.1)
    # Initialize the angular momentum algebra ...
    wigner_init_float(75, "Jmax", 9)

    # Make directory for export of information on structure of phonon levels ...
    if isdir("IO/" * Params.Calc.Path * "/RPA/Levels") == false
        mkdir("IO/" * Params.Calc.Path * "/RPA/Levels")
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

    # Import transformation matrix mapping LHO & reference HF basis ...
    @time C = O1b_import(Params,Orb,"IO/" * Params.Calc.Path * "/Bin/C_HF.bin")

    # Import TDA & RPA solutions ...
    @time X_TDA, E_TDA, X_RPA, Y_RPA, E_RPA = HF_RPA_import_binary(Params,N_nu)

    # Determine the number of phonon levels & the total number of phonons ...
    N_phonon = length(nu_list)
    N_ph = N_nu[J+1,P]

    # Precalculate the reduced matrix elements of Electrogmanetic 1-body transition operators ...
        # Iniztialize the 1-body Electromagnetic transition operators ...
    TrOp = Tr1b_initialize(Params,Orb,1.0)
         # Transform the 1-body transition operators to the reference basis ...
    TrOp = Tr1b_transformation(Params,Orb,C,TrOp)

    nu_count_TDA = zeros(Int64,N_phonon)
    nu_count_RPA = zeros(Int64,N_phonon)

    # Iterate through the phonon solutions & determine the most dominant ph & hp components of given phonon levels ...
    @inbounds for (nu_ind, nu) in enumerate(nu_list)
        @inbounds for ph in 1:N_ph
            x_TDA, x_RPA, y_RPA = X_TDA[J+1,P][ph,nu], X_RPA[J+1,P][ph,nu], Y_RPA[J+1,P][ph,nu]
            Amp_TDA = phase(J) * x_TDA
            Amp_RPA = (phase(J) * x_RPA + y_RPA)

            if abs2(Amp_TDA) > Cut
                nu_count_TDA[nu_ind] += 1
            end

            if abs2(Amp_RPA) > Cut
                nu_count_RPA[nu_ind] += 1
            end
        end
    end

    nu_compo_TDA = Vector{Vector{Tuple{Int64,Int64,Int64,Float64,Float64}}}(undef,N_phonon)
    nu_compo_RPA = Vector{Vector{Tuple{Int64,Int64,Int64,Float64,Float64,Float64}}}(undef,N_phonon)

    @inbounds for (nu_ind, nu) in enumerate(nu_list)
        nu_compo_TDA[nu_ind] = Vector{Tuple{Int64,Int64,Int64,Float64,Float64}}(undef,nu_count_TDA[nu_ind])
        nu_compo_RPA[nu_ind] = Vector{Tuple{Int64,Int64,Int64,Float64,Float64,Float64}}(undef,nu_count_RPA[nu_ind])
        
        count_TDA, count_RPA = 0, 0

        @inbounds for ph in 1:N_ph
            i_ph = Orb_Phonon[J+1,P][ph]
            p, h, t_ph = Phonon[i_ph].p, Phonon[i_ph].h, Phonon[i_ph].tz
            x_TDA, x_RPA, y_RPA = X_TDA[J+1,P][ph,nu], X_RPA[J+1,P][ph,nu], Y_RPA[J+1,P][ph,nu]

            Amp_TDA = phase(J) * x_TDA
            Amp_RPA = (phase(J) * x_RPA + y_RPA)

            if abs2(Amp_TDA) > Cut
                count_TDA += 1

                if t_ph == -1
                    a_p, a_h = Particle.p[p].a, Hole.p[h].a
                elseif t_ph == 1
                    a_p, a_h = Particle.n[p].a, Hole.n[h].a 
                end
                M_TDA = 0.0

                if J == 0 && P == 1
                    M_TDA = real(Amp_TDA) * TrOp.E0.p[a_p,a_h]

                elseif J == 1 && P == 2
                    M_TDA = real(Amp_TDA) * TrOp.E1.p[a_p,a_h]

                elseif J == 2 && P == 1
                    M_TDA = real(Amp_TDA) * TrOp.E2.p[a_p,a_h]

                elseif J == 3 && P == 2
                    M_TDA = real(Amp_TDA) * TrOp.E3.p[a_p,a_h]
                end

                nu_compo_TDA[nu_ind][count_TDA] = (a_p,a_h,t_ph,real(x_TDA),M_TDA)

            end


            if abs2(Amp_RPA) > Cut
                count_RPA += 1

                if t_ph == -1
                    a_p, a_h = Particle.p[p].a, Hole.p[h].a
                elseif t_ph == 1
                    a_p, a_h = Particle.n[p].a, Hole.n[h].a 
                end

                Amp_RPA_2 = abs2(Amp_RPA)
                M_RPA = 0.0

                if J == 0 && P == 1
                    M_RPA = real(Amp_RPA) * TrOp.E0.p[a_p,a_h]

                elseif J == 1 && P == 2
                    M_RPA = real(Amp_RPA) * TrOp.E1.p[a_p,a_h]

                elseif J == 2 && P == 1
                    M_RPA = real(Amp_RPA) * TrOp.E2.p[a_p,a_h]

                elseif J == 3 && P == 2
                    M_RPA = real(Amp_RPA) * TrOp.E3.p[a_p,a_h]
                end

                nu_compo_RPA[nu_ind][count_RPA] = (a_p,a_h,t_ph,real(x_RPA),real(y_RPA),M_RPA)
            end

        end

    end

    # Perform the export of information on individual phonon levels ...
    @inbounds for (nu_ind, nu) in enumerate(nu_list)
        e_TDA, e_RPA = E_TDA[J+1,P][nu], real(E_RPA[J+1,P][nu])
        sE_TDA, sE_RPA = string(round(e_TDA,digits=3)), string(round(e_RPA,digits=3))

        Phonon_TDA, Phonon_RPA = nu_compo_TDA[nu_ind], nu_compo_RPA[nu_ind]

        # Case of TDA levels ...
        Output_File = "IO/" * Params.Calc.Path * "/RPA/Levels/TDA_" * File_Name * "_JP$(JP)_nu$(nu)_E$(sE_TDA).dat"
        open(Output_File, "w") do Write_File
            @printf(Write_File, "%-8s %-8s %-8s %-20s %-20s\n", "a_p", "a_h", "T_ph", "X_ph", "M_ph")
            @inbounds for L in Phonon_TDA
                (a_p,a_h,t_ph,x_TDA,M_ph) = L
                @printf(Write_File, "%-8d %-8d %-8d %-20.8f %-20.8f\n", a_p, a_h, t_ph, x_TDA, M_ph)
            end
        end

        # Case of RPA levels ...
        Output_File = "IO/" * Params.Calc.Path * "/RPA/Levels/RPA_" * File_Name * "_JP$(JP)_nu$(nu)_E$(sE_RPA).dat"
        open(Output_File, "w") do Write_File
            @printf(Write_File, "%-8s %-8s %-8s %-20s %-20s %-20s\n", "a_p", "a_h", "T_ph", "Re X_ph", "Re Y_ph", "Re M_ph")
            @inbounds for L in Phonon_RPA
                (a_p,a_h,t_ph,x_RPA,y_RPA,M_ph) = L
                @printf(Write_File, "%-8d %-8d %-8d %-20.8f %-20.8f %-20.8f\n", a_p, a_h, t_ph, x_RPA, y_RPA, M_ph)
            end
        end

    end

    display("Export of information on individual phonon levels successfuly completed ...")

    return
end

function HF_RPA_transition_currents(Params::Parameters;JP::String,nu_list::Vector{Int64},File_Name::String = "Phonon_Currents",
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
    r_min, r_max = 0.01, 2.5 * 1.2 * A^(1/3)
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

    # Import transformation matrix mapping LHO & reference basis ...
    @time C = O1b_import(Params,Orb,"IO/" * Params.Calc.Path * "/Bin/C_HF.bin")

    # Import TDA & RPA solutions ...
    @time X_TDA, E_TDA, X_RPA, Y_RPA, E_RPA = HF_RPA_import_binary(Params,N_nu)

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
            # Case of TDA transition matrix elements ...
    pM_TDA, nM_TDA = Vector{Float64}(undef,N_phonon), Vector{Float64}(undef,N_phonon)
    isM_TDA, ivM_TDA = Vector{Float64}(undef,N_phonon), Vector{Float64}(undef,N_phonon)
            # Case of RPA transition matrix elements ...
    pM_RPA, nM_RPA = Vector{ComplexF64}(undef,N_phonon), Vector{ComplexF64}(undef,N_phonon)
    isM_RPA, ivM_RPA = Vector{ComplexF64}(undef,N_phonon), Vector{ComplexF64}(undef,N_phonon)

        # Calculate the matrix elements for given phonon levels ...
    if (J,P) in [(0,1),(2,1),(3,2)]
        @inbounds for (nu_ind, nu) in enumerate(nu_list)
            pMTDASum, nMTDASum, isMTDASum, ivMTDASum = 0.0, 0.0, 0.0, 0.0
            pMRPASum, nMRPASum, isMRPASum, ivMRPASum = 0.0, 0.0, 0.0, 0.0

            @inbounds for ph in 1:N_ph
                i_ph = Orb_Phonon[J+1,P][ph]
                p, h, t_ph = Phonon[i_ph].p, Phonon[i_ph].h, Phonon[i_ph].tz
                x_TDA, x_RPA, y_RPA = X_TDA[J+1,P][ph,nu], X_RPA[J+1,P][ph,nu], Y_RPA[J+1,P][ph,nu]

                MME_TDA = phase(J) * x_TDA / sqrt(Float64(2*J + 1))
                MME_RPA = (phase(J) * x_RPA - y_RPA) / sqrt(Float64(2*J + 1))

                if t_ph == -1
                    a_p, a_h = Particle.p[p].a, Hole.p[h].a

                    if J == 0 && P == 1
                        MME_TDA = TrOp.E0.p[a_p,a_h] * MME_TDA
                        MME_RPA = TrOp.E0.p[a_p,a_h] * MME_RPA

                    elseif J == 2 && P == 1
                        MME_TDA = TrOp.E2.p[a_p,a_h] * MME_TDA
                        MME_RPA = TrOp.E2.p[a_p,a_h] * MME_RPA

                    elseif J == 3 && P == 2
                        MME_TDA = TrOp.E3.p[a_p,a_h] * MME_TDA
                        MME_RPA = TrOp.E3.p[a_p,a_h] * MME_RPA
                    end

                    pMTDASum += MME_TDA
                    isMTDASum += 0.5 * MME_TDA
                    ivMTDASum += 0.5 * MME_TDA

                    pMRPASum += MME_RPA
                    isMRPASum += 0.5 * MME_RPA
                    ivMRPASum += 0.5 * MME_RPA

                elseif t_ph == 1
                    a_p, a_h = Particle.n[p].a, Hole.n[h].a

                    if J == 0 && P == 1
                        MME_TDA = TrOp.E0.n[a_p,a_h] * MME_TDA
                        MME_RPA = TrOp.E0.n[a_p,a_h] * MME_RPA

                    elseif J == 2 && P == 1
                        MME_TDA = TrOp.E2.n[a_p,a_h] * MME_TDA
                        MME_RPA = TrOp.E2.n[a_p,a_h] * MME_RPA

                    elseif J == 3 && P == 2
                        MME_TDA = TrOp.E3.n[a_p,a_h] * MME_TDA
                        MME_RPA = TrOp.E3.n[a_p,a_h] * MME_RPA
                    end

                    nMTDASum += MME_TDA
                    isMTDASum += 0.5 * MME_TDA
                    ivMTDASum -= 0.5 * MME_TDA

                    nMRPASum += MME_RPA
                    isMRPASum += 0.5 * MME_RPA
                    ivMRPASum -= 0.5 * MME_RPA
                end
            end

            pM_TDA[nu_ind] = pMTDASum
            nM_TDA[nu_ind] = nMTDASum
            isM_TDA[nu_ind] = isMTDASum
            ivM_TDA[nu_ind] = ivMTDASum

            pM_RPA[nu_ind] = pMRPASum
            nM_RPA[nu_ind] = nMRPASum
            isM_RPA[nu_ind] = isMRPASum
            ivM_RPA[nu_ind] = ivMRPASum
        end

    elseif (J,P) == (1,2)
        @inbounds for (nu_ind, nu) in enumerate(nu_list)
            pMTDASum, nMTDASum, isMTDASum, ivMTDASum = 0.0, 0.0, 0.0, 0.0
            pMRPASum, nMRPASum, isMRPASum, ivMRPASum = 0.0, 0.0, 0.0, 0.0

            @inbounds for ph in 1:N_ph
                i_ph = Orb_Phonon[J+1,P][ph]
                p, h, t_ph = Phonon[i_ph].p, Phonon[i_ph].h, Phonon[i_ph].tz
                x_TDA, x_RPA, y_RPA = X_TDA[J+1,P][ph,nu], X_RPA[J+1,P][ph,nu], Y_RPA[J+1,P][ph,nu]

                Amp_TDA = phase(J) / sqrt(Float64(2*J + 1)) * x_TDA
                Amp_RPA = (phase(J) * x_RPA - y_RPA) / sqrt(Float64(2*J + 1))

                if t_ph == -1
                    a_p, a_h = Particle.p[p].a, Hole.p[h].a

                    pMTDAME = TrOp.E1.p[a_p,a_h] * Amp_TDA
                    isMTDAME = 0.5 * TrOp.E1_C.p[a_p,a_h] * Amp_TDA
                    ivMTDAME = TrOp.E1.p[a_p,a_h] * Amp_TDA * (A - Z) / A

                    pMRPAME = TrOp.E1.p[a_p,a_h] * Amp_RPA
                    isMRPAME = 0.5 * TrOp.E1_C.p[a_p,a_h] * Amp_RPA
                    ivMRPAME = TrOp.E1.p[a_p,a_h] * Amp_RPA * (A - Z) / A

                    pMTDASum += pMTDAME
                    isMTDASum += isMTDAME
                    ivMTDASum += ivMTDAME

                    pMRPASum += pMRPAME
                    isMRPASum += isMRPAME
                    ivMRPASum += ivMRPAME

                elseif t_ph == 1
                    a_p, a_h = Particle.n[p].a, Hole.n[h].a

                    nMTDAME = TrOp.E1.n[a_p,a_h] * Amp_TDA
                    isMTDAME = 0.5 * TrOp.E1_C.n[a_p,a_h] * Amp_TDA
                    ivMTDAME = - TrOp.E1.n[a_p,a_h] * Amp_TDA * Z / A

                    nMRPAME = TrOp.E1.n[a_p,a_h] * Amp_RPA
                    isMRPAME = 0.5 * TrOp.E1_C.n[a_p,a_h] * Amp_RPA
                    ivMRPAME = - TrOp.E1.n[a_p,a_h] * Amp_RPA * Z / A

                    nMTDASum += nMTDAME
                    isMTDASum += isMTDAME
                    ivMTDASum += ivMTDAME

                    nMRPASum += nMRPAME
                    isMRPASum += isMRPAME
                    ivMRPASum += ivMRPAME
                end
            end

            pM_TDA[nu_ind] = pMTDASum
            nM_TDA[nu_ind] = nMTDASum
            isM_TDA[nu_ind] = isMTDASum
            ivM_TDA[nu_ind] = ivMTDASum

            pM_RPA[nu_ind] = pMRPASum
            nM_RPA[nu_ind] = nMRPASum
            isM_RPA[nu_ind] = isMRPASum
            ivM_RPA[nu_ind] = ivMRPASum
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

    pJ_TDA_conv_map = Array{Float64}(undef,3,N_phonon,N_map)
    nJ_TDA_conv_map = Array{Float64}(undef,3,N_phonon,N_map)
    pJ_TDA_spin_map = Array{Float64}(undef,3,N_phonon,N_map)
    nJ_TDA_spin_map = Array{Float64}(undef,3,N_phonon,N_map)

    pJ_RPA_conv_map = Array{Float64}(undef,3,N_phonon,N_map)
    nJ_RPA_conv_map = Array{Float64}(undef,3,N_phonon,N_map)
    pJ_RPA_spin_map = Array{Float64}(undef,3,N_phonon,N_map)
    nJ_RPA_spin_map = Array{Float64}(undef,3,N_phonon,N_map)

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
            pJxConvTDASum, pJyConvTDASum, pJzConvTDASum = 0.0, 0.0, 0.0
            nJxConvTDASum, nJyConvTDASum, nJzConvTDASum = 0.0, 0.0, 0.0
            pJxSpinTDASum, pJySpinTDASum, pJzSpinTDASum = 0.0, 0.0, 0.0
            nJxSpinTDASum, nJySpinTDASum, nJzSpinTDASum = 0.0, 0.0, 0.0

            pJxConvRPASum, pJyConvRPASum, pJzConvRPASum = 0.0, 0.0, 0.0
            nJxConvRPASum, nJyConvRPASum, nJzConvRPASum = 0.0, 0.0, 0.0
            pJxSpinRPASum, pJySpinRPASum, pJzSpinRPASum = 0.0, 0.0, 0.0
            nJxSpinRPASum, nJySpinRPASum, nJzSpinRPASum = 0.0, 0.0, 0.0

            Gauge_TDA = 1.0
            Gauge_RPA = 1.0

            if ((J,P) in [(0,1),(1,2),(2,1),(3,2)]) == true
                Gauge_TDA = pM_TDA[nu_ind] / abs(pM_TDA[nu_ind])
                Gauge_RPA = pM_RPA[nu_ind] / abs(pM_RPA[nu_ind])
            end

            @inbounds for ph in 1:N_ph
                i_ph = Orb_Phonon[J+1,P][ph]
                p, h, t_ph = Phonon[i_ph].p, Phonon[i_ph].h, Phonon[i_ph].tz

                if t_ph == -1
                    a_p, a_h = Particle.p[p].a, Hole.p[h].a
                elseif t_ph == 1
                    a_p, a_h = Particle.n[p].a, Hole.n[h].a
                end

                x_TDA = Gauge_TDA * X_TDA[J+1,P][ph,nu]
                x_RPA = real(Gauge_RPA * X_RPA[J+1,P][ph,nu])
                y_RPA = real(Gauge_RPA * Y_RPA[J+1,P][ph,nu])

                if t_ph == -1
                    pJxConvTDASum += x_TDA * pJ_ph_conv_map[1,ph,N]
                    pJyConvTDASum += x_TDA * pJ_ph_conv_map[2,ph,N]
                    pJzConvTDASum += x_TDA * pJ_ph_conv_map[3,ph,N]

                    pJxSpinTDASum += x_TDA * pJ_ph_spin_map[1,ph,N]
                    pJySpinTDASum += x_TDA * pJ_ph_spin_map[2,ph,N]
                    pJzSpinTDASum += x_TDA * pJ_ph_spin_map[3,ph,N]

                    pJxConvRPASum += (x_RPA * pJ_ph_conv_map[1,ph,N] - y_RPA * pJ_hp_conv_map[1,ph,N])
                    pJyConvRPASum += (x_RPA * pJ_ph_conv_map[2,ph,N] - y_RPA * pJ_hp_conv_map[2,ph,N])
                    pJzConvRPASum += (x_RPA * pJ_ph_conv_map[3,ph,N] - y_RPA * pJ_hp_conv_map[3,ph,N])

                    pJxSpinRPASum += (x_RPA * pJ_ph_spin_map[1,ph,N] - y_RPA * pJ_hp_spin_map[1,ph,N])
                    pJySpinRPASum += (x_RPA * pJ_ph_spin_map[2,ph,N] - y_RPA * pJ_hp_spin_map[2,ph,N])
                    pJzSpinRPASum += (x_RPA * pJ_ph_spin_map[3,ph,N] - y_RPA * pJ_hp_spin_map[3,ph,N])

                elseif t_ph == 1
                    nJxConvTDASum += x_TDA * nJ_ph_conv_map[1,ph,N]
                    nJyConvTDASum += x_TDA * nJ_ph_conv_map[2,ph,N]
                    nJzConvTDASum += x_TDA * nJ_ph_conv_map[3,ph,N]

                    nJxSpinTDASum += x_TDA * nJ_ph_spin_map[1,ph,N]
                    nJySpinTDASum += x_TDA * nJ_ph_spin_map[2,ph,N]
                    nJzSpinTDASum += x_TDA * nJ_ph_spin_map[3,ph,N]

                    nJxConvRPASum += (x_RPA * nJ_ph_conv_map[1,ph,N] - y_RPA * nJ_hp_conv_map[1,ph,N])
                    nJyConvRPASum += (x_RPA * nJ_ph_conv_map[2,ph,N] - y_RPA * nJ_hp_conv_map[2,ph,N])
                    nJzConvRPASum += (x_RPA * nJ_ph_conv_map[3,ph,N] - y_RPA * nJ_hp_conv_map[3,ph,N])

                    nJxSpinRPASum += (x_RPA * nJ_ph_spin_map[1,ph,N] - y_RPA * nJ_hp_spin_map[1,ph,N])
                    nJySpinRPASum += (x_RPA * nJ_ph_spin_map[2,ph,N] - y_RPA * nJ_hp_spin_map[2,ph,N])
                    nJzSpinRPASum += (x_RPA * nJ_ph_spin_map[3,ph,N] - y_RPA * nJ_hp_spin_map[3,ph,N])
                end

            end

            pJ_TDA_conv_map[1,nu_ind,N] = pJxConvTDASum
            pJ_TDA_conv_map[2,nu_ind,N] = pJyConvTDASum
            pJ_TDA_conv_map[3,nu_ind,N] = pJzConvTDASum
            nJ_TDA_conv_map[1,nu_ind,N] = nJxConvTDASum
            nJ_TDA_conv_map[2,nu_ind,N] = nJyConvTDASum
            nJ_TDA_conv_map[3,nu_ind,N] = nJzConvTDASum

            pJ_TDA_spin_map[1,nu_ind,N] = pJxSpinTDASum
            pJ_TDA_spin_map[2,nu_ind,N] = pJySpinTDASum
            pJ_TDA_spin_map[3,nu_ind,N] = pJzSpinTDASum
            nJ_TDA_spin_map[1,nu_ind,N] = nJxSpinTDASum
            nJ_TDA_spin_map[2,nu_ind,N] = nJySpinTDASum
            nJ_TDA_spin_map[3,nu_ind,N] = nJzSpinTDASum

            pJ_RPA_conv_map[1,nu_ind,N] = pJxConvRPASum
            pJ_RPA_conv_map[2,nu_ind,N] = pJyConvRPASum
            pJ_RPA_conv_map[3,nu_ind,N] = pJzConvRPASum
            nJ_RPA_conv_map[1,nu_ind,N] = nJxConvRPASum
            nJ_RPA_conv_map[2,nu_ind,N] = nJyConvRPASum
            nJ_RPA_conv_map[3,nu_ind,N] = nJzConvRPASum

            pJ_RPA_spin_map[1,nu_ind,N] = pJxSpinRPASum
            pJ_RPA_spin_map[2,nu_ind,N] = pJySpinRPASum
            pJ_RPA_spin_map[3,nu_ind,N] = pJzSpinRPASum
            nJ_RPA_spin_map[1,nu_ind,N] = nJxSpinRPASum
            nJ_RPA_spin_map[2,nu_ind,N] = nJySpinRPASum
            nJ_RPA_spin_map[3,nu_ind,N] = nJzSpinRPASum
        end
    
    end

    # Scale the currents to correct physical units in [c] ...
    pJ_TDA_conv_map .= pJ_TDA_conv_map .* hc / m_p
    nJ_TDA_conv_map .= nJ_TDA_conv_map .* hc / m_p
    pJ_TDA_spin_map .= pJ_TDA_spin_map .* hc / m_p
    nJ_TDA_spin_map .= nJ_TDA_spin_map .* hc / m_p

    pJ_RPA_conv_map .= pJ_RPA_conv_map .* hc / m_p
    nJ_RPA_conv_map .= nJ_RPA_conv_map .* hc / m_p
    pJ_RPA_spin_map .= pJ_RPA_spin_map .* hc / m_p
    nJ_RPA_spin_map .= nJ_RPA_spin_map .* hc / m_p

    # Perform the export of of individual phonon level transition current maps ...
    @inbounds for (nu_ind, nu) in enumerate(nu_list)
        e_TDA, e_RPA = E_TDA[J+1,P][nu], real(E_RPA[J+1,P][nu])
        sE_TDA, sE_RPA = string(round(e_TDA,digits=3)), string(round(e_RPA,digits=3))

        # Export TDA convective transition current densities ...
        Output_File_TDA = "IO/" * Params.Calc.Path * "/RPA/Densities/TDA_Convective_" * File_Name * "_JP$(JP)_nu$(nu)_E$(sE_TDA).dat"
        open(Output_File_TDA, "w") do Write_File
            @printf(Write_File, "%-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s\n", "x", "z", "pJ_x", "pJ_z", "nJ_x", "nJ_z", "isJ_x", "isJ_z", "ivJ_x", "ivJ_z")
            
            @inbounds for N in 1:N_map
                r, theta = x_map[1,N], x_map[2,N]
                x = r * sin(theta)
                z = r * cos(theta)
                pJ_x = pJ_TDA_conv_map[1,nu_ind,N]
                pJ_z = pJ_TDA_conv_map[3,nu_ind,N]
                nJ_x = nJ_TDA_conv_map[1,nu_ind,N]
                nJ_z = nJ_TDA_conv_map[3,nu_ind,N]
                isJ_x = 0.5 * (pJ_x + nJ_x)
                isJ_z = 0.5 * (pJ_z + nJ_z)
                ivJ_x = 0.5 * (pJ_x - nJ_x)
                ivJ_z = 0.5 * (pJ_z - nJ_z)
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f\n", x, z, pJ_x, pJ_z, nJ_x, nJ_z, isJ_x, isJ_z, ivJ_x, ivJ_z)
            end
        end

         # Export TDA spin transition current densities ...
        Output_File_TDA = "IO/" * Params.Calc.Path * "/RPA/Densities/TDA_Spin_" * File_Name * "_JP$(JP)_nu$(nu)_E$(sE_TDA).dat"
        open(Output_File_TDA, "w") do Write_File
            @printf(Write_File, "%-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s\n", "x", "z", "pJ_x", "pJ_z", "nJ_x", "nJ_z", "isJ_x", "isJ_z", "ivJ_x", "ivJ_z")
            
            @inbounds for N in 1:N_map
                r, theta = x_map[1,N], x_map[2,N]
                x = r * sin(theta)
                z = r * cos(theta)
                pJ_x = pJ_TDA_spin_map[1,nu_ind,N]
                pJ_z = pJ_TDA_spin_map[3,nu_ind,N]
                nJ_x = nJ_TDA_spin_map[1,nu_ind,N]
                nJ_z = nJ_TDA_spin_map[3,nu_ind,N]
                isJ_x = 0.5 * (pJ_x + nJ_x)
                isJ_z = 0.5 * (pJ_z + nJ_z)
                ivJ_x = 0.5 * (pJ_x - nJ_x)
                ivJ_z = 0.5 * (pJ_z - nJ_z)
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f\n", x, z, pJ_x, pJ_z, nJ_x, nJ_z, isJ_x, isJ_z, ivJ_x, ivJ_z)
            end
        end

        # Export RPA convective transition current densities ...
        Output_File_RPA = "IO/" * Params.Calc.Path * "/RPA/Densities/RPA_Convective_" * File_Name * "_JP$(JP)_nu$(nu)_E$(sE_RPA).dat"
        open(Output_File_RPA, "w") do Write_File
            @printf(Write_File, "%-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s\n", "x", "z", "pJ_x", "pJ_z", "nJ_x", "nJ_z", "isJ_x", "isJ_z", "ivJ_x", "ivJ_z")
            
            @inbounds for N in 1:N_map
                r, theta = x_map[1,N], x_map[2,N]
                x = r * sin(theta)
                z = r * cos(theta)
                pJ_x = pJ_RPA_conv_map[1,nu_ind,N]
                pJ_z = pJ_RPA_conv_map[3,nu_ind,N]
                nJ_x = nJ_RPA_conv_map[1,nu_ind,N]
                nJ_z = nJ_RPA_conv_map[3,nu_ind,N]
                isJ_x = 0.5 * (pJ_x + nJ_x)
                isJ_z = 0.5 * (pJ_z + nJ_z)
                ivJ_x = 0.5 * (pJ_x - nJ_x)
                ivJ_z = 0.5 * (pJ_z - nJ_z)
                @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f\n", x, z, pJ_x, pJ_z, nJ_x, nJ_z, isJ_x, isJ_z, ivJ_x, ivJ_z)
            end
        end

         # Export RPA spin transition current densities ...
        Output_File_RPA = "IO/" * Params.Calc.Path * "/RPA/Densities/RPA_Spin_" * File_Name * "_JP$(JP)_nu$(nu)_E$(sE_RPA).dat"
        open(Output_File_RPA, "w") do Write_File
            @printf(Write_File, "%-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s\n", "x", "z", "pJ_x", "pJ_z", "nJ_x", "nJ_z", "isJ_x", "isJ_z", "ivJ_x", "ivJ_z")
            
            @inbounds for N in 1:N_map
                r, theta = x_map[1,N], x_map[2,N]
                x = r * sin(theta)
                z = r * cos(theta)
                pJ_x = pJ_RPA_spin_map[1,nu_ind,N]
                pJ_z = pJ_RPA_spin_map[3,nu_ind,N]
                nJ_x = nJ_RPA_spin_map[1,nu_ind,N]
                nJ_z = nJ_RPA_spin_map[3,nu_ind,N]
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

    # Renormalize M ... to reasonable scale averaged densities ...
        # Case of TDA transition matrix elements ...
    pM_TDA .= abs.(pM_TDA) ./ sqrt(sum(pM_TDA.^2))
    nM_TDA .= abs.(nM_TDA) ./ sqrt(sum(nM_TDA.^2))
    isM_TDA .= abs.(isM_TDA) ./ sqrt(sum(isM_TDA.^2))
    ivM_TDA .= abs.(ivM_TDA) ./ sqrt(sum(ivM_TDA.^2))
        # Case of RPA transition matrix elements ...
    pM_RPA .= abs.(pM_RPA) ./ sqrt(sum(pM_RPA.^2))
    nM_RPA .= abs.(nM_RPA) ./ sqrt(sum(nM_RPA.^2))
    isM_RPA .= abs.(isM_RPA) ./ sqrt(sum(isM_RPA.^2))
    ivM_RPA .= abs.(ivM_RPA) ./ sqrt(sum(ivM_RPA.^2))

    # Perform the export of of averaged transition current maps ...
        # Export of TDA averaged convective transition current densities ...
    Output_File_TDA_M = "IO/" * Params.Calc.Path * "/RPA/Densities/TDA_Convective_" * File_Name * "_JP$(JP)_Averaged.dat"
    open(Output_File_TDA_M, "w") do Write_File
        @printf(Write_File, "%-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s\n", "x", "z", "ppJ_x", "ppJ_z", "pnJ_x", "pnJ_z", "npJ_x", "npJ_z", "nnJ_x", "nnJ_z", "isJ_x", "isJ_z", "ivJ_x", "ivJ_z")
        
        @inbounds for N in 1:N_map
            r, theta = x_map[1,N], x_map[2,N]
            x = r * sin(theta)
            z = r * cos(theta)
            ppJ_x, pnJ_x, npJ_x, nnJ_x, isJ_x, ivJ_x = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ppJ_z, pnJ_z, npJ_z, nnJ_z, isJ_z, ivJ_z = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            @inbounds for nu_ind in 1:N_phonon
                pJ_x, pJ_z = pJ_TDA_conv_map[1,nu_ind,N], pJ_TDA_conv_map[3,nu_ind,N]
                nJ_x, nJ_z = nJ_TDA_conv_map[1,nu_ind,N], nJ_TDA_conv_map[3,nu_ind,N]

                pM, nM = real(pM_TDA[nu_ind]), real(nM_TDA[nu_ind])
                isM, ivM = real(isM_TDA[nu_ind]), real(ivM_TDA[nu_ind])

                ppJ_x += pM * pJ_x
                ppJ_z += pM * pJ_z
                pnJ_x += pM * nJ_x
                pnJ_z += pM * nJ_z
                npJ_x += nM * pJ_x
                nnJ_x += nM * nJ_x
                npJ_z += nM * pJ_z
                nnJ_z += nM * nJ_z
                isJ_x += 0.5 * isM * (pJ_x + nJ_x)
                isJ_z += 0.5 * isM * (pJ_z + nJ_z)
                ivJ_x += 0.5 * ivM * (pJ_x - nJ_x)
                ivJ_z += 0.5 * ivM * (pJ_z - nJ_z)
            end

            @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f\n", x, z, ppJ_x, ppJ_z, pnJ_x, pnJ_z, npJ_x, npJ_z, nnJ_x, nnJ_z, isJ_x, isJ_z, ivJ_x, ivJ_z)
        end
    end
        # Export of TDA averaged spin transition current densities ...
    Output_File_TDA_M = "IO/" * Params.Calc.Path * "/RPA/Densities/TDA_Spin_" * File_Name * "_JP$(JP)_Averaged.dat"
    open(Output_File_TDA_M, "w") do Write_File
        @printf(Write_File, "%-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s\n", "x", "z", "ppJ_x", "ppJ_z", "pnJ_x", "pnJ_z", "npJ_x", "npJ_z", "nnJ_x", "nnJ_z", "isJ_x", "isJ_z", "ivJ_x", "ivJ_z")
        
        @inbounds for N in 1:N_map
            r, theta = x_map[1,N], x_map[2,N]
            x = r * sin(theta)
            z = r * cos(theta)
            ppJ_x, pnJ_x, npJ_x, nnJ_x, isJ_x, ivJ_x = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ppJ_z, pnJ_z, npJ_z, nnJ_z, isJ_z, ivJ_z = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            @inbounds for nu_ind in 1:N_phonon
                pJ_x, pJ_z = pJ_TDA_spin_map[1,nu_ind,N], pJ_TDA_spin_map[3,nu_ind,N]
                nJ_x, nJ_z = nJ_TDA_spin_map[1,nu_ind,N], nJ_TDA_spin_map[3,nu_ind,N]

                pM, nM = real(pM_TDA[nu_ind]), real(nM_TDA[nu_ind])
                isM, ivM = real(isM_TDA[nu_ind]), real(ivM_TDA[nu_ind])

                ppJ_x += pM * pJ_x
                ppJ_z += pM * pJ_z
                pnJ_x += pM * nJ_x
                pnJ_z += pM * nJ_z
                npJ_x += nM * pJ_x
                nnJ_x += nM * nJ_x
                npJ_z += nM * pJ_z
                nnJ_z += nM * nJ_z
                isJ_x += 0.5 * isM * (pJ_x + nJ_x)
                isJ_z += 0.5 * isM * (pJ_z + nJ_z)
                ivJ_x += 0.5 * ivM * (pJ_x - nJ_x)
                ivJ_z += 0.5 * ivM * (pJ_z - nJ_z)
            end

            @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f\n", x, z, ppJ_x, ppJ_z, pnJ_x, pnJ_z, npJ_x, npJ_z, nnJ_x, nnJ_z, isJ_x, isJ_z, ivJ_x, ivJ_z)
        end
    end
        # Export of RPA averaged convective transition current densities ...
    Output_File_RPA_M = "IO/" * Params.Calc.Path * "/RPA/Densities/RPA_Convective_" * File_Name * "_JP$(JP)_Averaged.dat"
    open(Output_File_RPA_M, "w") do Write_File
        @printf(Write_File, "%-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s\n", "x", "z", "ppJ_x", "ppJ_z", "pnJ_x", "pnJ_z", "npJ_x", "npJ_z", "nnJ_x", "nnJ_z", "isJ_x", "isJ_z", "ivJ_x", "ivJ_z")
        
        @inbounds for N in 1:N_map
            r, theta = x_map[1,N], x_map[2,N]
            x = r * sin(theta)
            z = r * cos(theta)
            ppJ_x, pnJ_x, npJ_x, nnJ_x, isJ_x, ivJ_x = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ppJ_z, pnJ_z, npJ_z, nnJ_z, isJ_z, ivJ_z = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            @inbounds for nu_ind in 1:N_phonon
                pJ_x, pJ_z = pJ_RPA_conv_map[1,nu_ind,N], pJ_RPA_conv_map[3,nu_ind,N]
                nJ_x, nJ_z = nJ_RPA_conv_map[1,nu_ind,N], nJ_RPA_conv_map[3,nu_ind,N]

                pM, nM = real(pM_RPA[nu_ind]), real(nM_RPA[nu_ind])
                isM, ivM = real(isM_RPA[nu_ind]), real(ivM_RPA[nu_ind])

                ppJ_x += pM * pJ_x
                ppJ_z += pM * pJ_z
                pnJ_x += pM * nJ_x
                pnJ_z += pM * nJ_z
                npJ_x += nM * pJ_x
                nnJ_x += nM * nJ_x
                npJ_z += nM * pJ_z
                nnJ_z += nM * nJ_z
                isJ_x += 0.5 * isM * (pJ_x + nJ_x)
                isJ_z += 0.5 * isM * (pJ_z + nJ_z)
                ivJ_x += 0.5 * ivM * (pJ_x - nJ_x)
                ivJ_z += 0.5 * ivM * (pJ_z - nJ_z)
            end

            @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f\n", x, z, ppJ_x, ppJ_z, pnJ_x, pnJ_z, npJ_x, npJ_z, nnJ_x, nnJ_z, isJ_x, isJ_z, ivJ_x, ivJ_z)
        end
    end
        # Export of RPA averaged spin transition current densities ...
    Output_File_RPA_M = "IO/" * Params.Calc.Path * "/RPA/Densities/RPA_Spin_" * File_Name * "_JP$(JP)_Averaged.dat"
    open(Output_File_RPA_M, "w") do Write_File
        @printf(Write_File, "%-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s\n", "x", "z", "ppJ_x", "ppJ_z", "pnJ_x", "pnJ_z", "npJ_x", "npJ_z", "nnJ_x", "nnJ_z", "isJ_x", "isJ_z", "ivJ_x", "ivJ_z")
        @inbounds for N in 1:N_map
            r, theta = x_map[1,N], x_map[2,N]
            x = r * sin(theta)
            z = r * cos(theta)
            ppJ_x, pnJ_x, npJ_x, nnJ_x, isJ_x, ivJ_x = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ppJ_z, pnJ_z, npJ_z, nnJ_z, isJ_z, ivJ_z = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            @inbounds for nu_ind in 1:N_phonon
                pJ_x, pJ_z = pJ_RPA_spin_map[1,nu_ind,N], pJ_RPA_spin_map[3,nu_ind,N]
                nJ_x, nJ_z = nJ_RPA_spin_map[1,nu_ind,N], nJ_RPA_spin_map[3,nu_ind,N]

                pM, nM = real(pM_RPA[nu_ind]), real(nM_RPA[nu_ind])
                isM, ivM = real(isM_RPA[nu_ind]), real(ivM_RPA[nu_ind])

                ppJ_x += pM * pJ_x
                ppJ_z += pM * pJ_z
                pnJ_x += pM * nJ_x
                pnJ_z += pM * nJ_z
                npJ_x += nM * pJ_x
                nnJ_x += nM * nJ_x
                npJ_z += nM * pJ_z
                nnJ_z += nM * nJ_z
                isJ_x += 0.5 * isM * (pJ_x + nJ_x)
                isJ_z += 0.5 * isM * (pJ_z + nJ_z)
                ivJ_x += 0.5 * ivM * (pJ_x - nJ_x)
                ivJ_z += 0.5 * ivM * (pJ_z - nJ_z)
            end

            @printf(Write_File, "%-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f %-20.8f\n", x, z, ppJ_x, ppJ_z, pnJ_x, pnJ_z, npJ_x, npJ_z, nnJ_x, nnJ_z, isJ_x, isJ_z, ivJ_x, ivJ_z)
        end
    end

    display("Export of averaged phonon level transition current maps successfuly completed ...")

    return
end

function HF_RPA_transitions_recalculate(Params::Parameters)
    # Check if RPA transition directories exist, if not create them ...
    if !(isdir("IO/" * Params.Calc.Path * "/RPA/Transitions"))
        mkdir("IO/" * Params.Calc.Path * "/RPA/Transitions")
        mkdir("IO/" * Params.Calc.Path * "/RPA/Transitions/E0")
        mkdir("IO/" * Params.Calc.Path * "/RPA/Transitions/E1")
        mkdir("IO/" * Params.Calc.Path * "/RPA/Transitions/E2")
        mkdir("IO/" * Params.Calc.Path * "/RPA/Transitions/E3")
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

    # Import LHO -> reference OBDM transformation matrices ...
    @time C = O1b_import(Params,Orb,"IO/" * Params.Calc.Path * "/Bin/C_HF.bin")

    # Import TDA & RPA solutions ...
    @time X_TDA, E_TDA, X_RPA, Y_RPA, E_RPA = HF_RPA_import_binary(Params,N_nu)

    # Calculate RPA One-Body Density Matrix (OBDM) ...
    @time Rho_RPA = HF_RPA_OBDM(Params,Orb,Orb_Phonon,Phonon,Particle,Hole,N_nu,Y_RPA)

    # Evaluate the RPA charge radius chR ...
    chR2 = OBDM_chR2(Params,Orb,C,Rho_RPA)

    # Precalculate the reduced matrix elements of Electrogmanetic 1-body transition operators ...
        # Iniztialize the 1-body Electromagnetic transition operators ...
    TrOp = Tr1b_initialize(Params,Orb,chR2)
         # Transform the 1-body transition operators to the reference basis ...
    TrOp = Tr1b_transformation(Params,Orb,C,TrOp)

    # Calculate the reduced transition matrix elements M_nu for each phonon level ...
    rM_TDA, rM_RPA = HF_RPA_rM(Params,N_nu,Orb_Phonon,Phonon,Particle,Hole,X_TDA,X_RPA,Y_RPA,TrOp)

    # Evaluate the reduced transition intensities B_nu for each phonon level ...
    rB_TDA = HF_RPA_rB(Params,N_nu,rM_TDA,E_TDA)
    rB_RPA = HF_RPA_rB(Params,N_nu,rM_RPA,real.(E_RPA))

    # Export of RRPA electromagnetic transitions ...
    HF_RPA_transitions_export(Params,Orb,N_nu,C,Rho_RPA,E_TDA,E_RPA,rB_TDA,rB_RPA)

    return
end