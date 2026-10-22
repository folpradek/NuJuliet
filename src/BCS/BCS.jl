function BCS(Params::Parameters)
    # Initialize Wigner symbols ...
    wigner_init_float(Params.Int.N2max + 2, "Jmax", 6)

    # Calculation parameters ...
    println("Starting BCS calculation with NO2B NN+NNN interaction ...")

    println("\nInteraction data:")
    println("hw = " * string(Params.Int.hw) * " , N_max = " * string(Params.Int.Nmax) * " , N_2max = " * string(Params.Int.N2max) *
            ", N_3max = " * string(Params.Int.N3max) * ", J-scheme basis size = " * string(div((Params.Int.Nmax+1)*(Params.Int.Nmax+2),2)) *
            ", M-scheme basis size = " * string(div((Params.Int.Nmax+1)*(Params.Int.Nmax+2)*(Params.Int.Nmax+3),6)))

    println("\nCalculation data:")
    println("A = " * string(Params.Calc.A) * " , Z = " * string(Params.Calc.Z) * " , N_max = " * string(Params.Calc.Nmax) *
            ", N_2max = " * string(Params.Calc.N2max) * " , N_3max = " * string(Params.Calc.N3max) * ", J-scheme basis size = " *
            string(div((Params.Calc.Nmax+1)*(Params.Calc.Nmax+2),2)) * ", M-scheme basis size = " *
            string(div((Params.Calc.Nmax+1)*(Params.Calc.Nmax+2)*(Params.Calc.Nmax+3),6)))

    println("Center of mass correction option is set to:    " * Params.Calc.CMS)
    if Params.Calc.CMS == "CMS1+2B"
        println("\nCombined 1-body + 2-body center of mass motion correction is included ...")
    elseif Params.Calc.CMS == "CMS2B"
        println("\nOnly pure 2-body center of mass motion correction is included ...")
    else
        println("\nNo center of mass motion correction is included ...")
    end

    if Threads.nthreads() > 1
        println("\n_________________________________________________________")
        println("Multiple active threads detected ...")
        println("Parallelization report:    Number of active threads = " * string(Threads.nthreads()))
        println("_________________________________________________________")
    end

    # Make new directory for results ...
    if isdir("io/" * Params.Calc.Path)
        rm("io/" * Params.Calc.Path, recursive = true)
    end
    mkdir("io/" * Params.Calc.Path)
    mkdir("io/" * Params.Calc.Path * "/BCS")
    mkdir("io/" * Params.Calc.Path * "/BCS/bin")
    mkdir("io/" * Params.Calc.Path * "/BCS/densities")

    # Make Parameters structure for reference mean-field calculation ...
    Params_Ref = BCS_reference(Params)
    println("\nReference closed-shell nucleus selected for BCS calculation is:")
    println("     A = " * string(Params_Ref.Calc.A) * ",     Z = " * string(Params_Ref.Calc.Z))

    # Start HF-BCS calculation ...
    println("\nStarting BCS calculation ...")

    @time BCS_solver(Params,Params_Ref)

    # Before exiting perform the Garbage Collection ...
    GC.gc()
    
    println("\nAll calculations have finished ...\n")

    return
end

function BCS_reference(Params::Parameters)
    # User specified reference closed-(sub)shell nucleus for BCS calculation ...
    if Params.Calc.BCS.A0 > 0 && Params.Calc.BCS.Z0 > 0
        # Set the parameters for reference mean-field calculation ...
        CalcParams = Calculation_Parameters(
                    A = Params.Calc.BCS.A0,
                    Z = Params.Calc.BCS.Z0,
                    hw = Params.Calc.hw,
                    Nmax = Params.Calc.Nmax,
                    N2max = Params.Calc.N2max,
                    N3max = Params.Calc.N3max,                                                         
                    CMS = Params.Calc.CMS,
                    Path = Params.Calc.Path,
                    HF = Params.Calc.HF,
                    RPA = Params.Calc.RPA,
                    RRPA = Params.Calc.RRPA,
                    BCS = Params.Calc.BCS,
                    HFB = Params.Calc.HFB,
                    )

        return Parameters(Params.Int,CalcParams)

    # Otherwise the closest closed-shell nucleus is picked ...
    else
        # Initialize basic variables ...
        Magic_Numbers = pnVector([2,8,20,28,50,82],[2,8,20,28,50,82,126])
        N_target, Z_target = Params.Calc.A - Params.Calc.Z, Params.Calc.Z
        N_ref, Z_ref = 0, 0
        dN, dZ = 1000, 1000

        # Find the largest closed-shell (magic) proton number not exceeding Z_target ...
        @inbounds for Z in Magic_Numbers.p 
            if Z <= Z_target && (Z_target - Z) < dZ
                Z_ref, dZ = Z, (Z_target - Z)
            end
        end

        # Find the largest closed-shell (magic) neutron number not exceeding N_target ...
        @inbounds for N in Magic_Numbers.n
            if N <= N_target && (N_target - N) < dN
                N_ref, dN = N, (N_target - N)
            end
        end

        # Set the parameters for reference mean-field calculation ...
        CalcParams = Calculation_Parameters(
                    A = N_ref + Z_ref,
                    Z = Z_ref,
                    hw = Params.Calc.hw,
                    Nmax = Params.Calc.Nmax,
                    N2max = Params.Calc.N2max,
                    N3max = Params.Calc.N3max,                                                         
                    CMS = Params.Calc.CMS,
                    Path = Params.Calc.Path,
                    HF = Params.Calc.HF,
                    RPA = Params.Calc.RPA,
                    RRPA = Params.Calc.RRPA,
                    BCS = Params.Calc.BCS,
                    HFB = Params.Calc.HFB,
                    )

        return Parameters(Params.Int,CalcParams)
    end
end