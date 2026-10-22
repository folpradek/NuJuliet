function HFB(Params::Parameters)
    # Initialize Wigner symbols ...
    wigner_init_float(Params.Int.N2max + 2, "Jmax", 6)

    # Calculation parameters ...
    println("Starting spherical HFB calculation with NO2B NN+NNN interaction ...")

    println("\nInteraction data:")
    println("hw = " * string(Params.Int.hw) * " , N_max = " * string(Params.Int.Nmax) * " , N_2max = " * string(Params.Int.N2max) *
            " , N_3max = " * string(Params.Int.N3max) * ", J-scheme basis size = " * string(div((Params.Int.Nmax+1)*(Params.Int.Nmax+2),2)) *
            ", M-scheme basis size = " * string(div((Params.Int.Nmax+1)*(Params.Int.Nmax+2)*(Params.Int.Nmax+3),6)))

    println("\nCalculation data:")
    println("A = " * string(Params.Calc.A) * " , Z = " * string(Params.Calc.Z) * " , N_max = " * string(Params.Calc.Nmax) *
            " , N_2max = " * string(Params.Calc.N2max) * " , N_3max = " * string(Params.Calc.N3max) * ", J-scheme basis size = " *
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
    mkdir("io/" * Params.Calc.Path * "/HFB")
    mkdir("io/" * Params.Calc.Path * "/HFB/bin")
    mkdir("io/" * Params.Calc.Path * "/HFB/densities")

    # Start HFB calculation ...
    println("\nStarting spherical Hartree-Fock-Bogoliubov calculation ...")

    @time HFB_solver(Params)

    # Before exiting perform the Garbage Collection ...
    GC.gc()
    
    println("\nAll calculations have finished ...\n")

    return
end