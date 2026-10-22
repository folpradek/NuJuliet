function QTDA(Params::Parameters)
    # Initialize Wigner symbols ...
    wigner_init_float(75,"Jmax",9)

    # Calculation parameters ...
    println("Starting spherical QTDA calculation with residual NO2B NN+NNN interaction based on available reference quasiparticle mean-field calculation")
    println("\nCalculation data:")
    println("A = " * string(Params.Calc.A) * " , Z = " * string(Params.Calc.Z) * " , hw = " * string(Params.Calc.hw) *
            " MeV , N_max = " * string(Params.Calc.Nmax) * " , J-scheme LHO basis size = " * string(div((Params.Calc.Nmax+1)*(Params.Calc.Nmax+2),2)) *
            " , M-scheme LHO basis size = " * string(div((Params.Calc.Nmax+1)*(Params.Calc.Nmax+2)*(Params.Calc.Nmax+3),6)))

    if Threads.nthreads() > 1
        println("\n_________________________________________________________")
        println("Multiple active threads detected ...")
        println("Parallelization report:    Number of active threads = " * string(Threads.nthreads()))
        println("_________________________________________________________")
    end

    # Make new directories for results ...
    println(Params.Calc.Path)
    if !(isfile("io/" * Params.Calc.Path * "/" * Params.Calc.QTDA.Reference * "/bin/C.bin")) ||
       !(isfile("io/" * Params.Calc.Path * "/" * Params.Calc.QTDA.Reference * "/bin/U.bin")) ||
       !(isfile("io/" * Params.Calc.Path * "/" * Params.Calc.QTDA.Reference * "/bin/V.bin"))
        println("\nError! ... No precomputed reference quasiparticle mean-field (HFB, BCS) solution is available in given Input_File path ... run HFB/BCS solver first ...")
        return
    end
    if (isdir("io/" * Params.Calc.Path * "/QTDA"))
        rm("io/" * Params.Calc.Path * "/QTDA", recursive = true)
    end
    mkdir("io/" * Params.Calc.Path * "/QTDA")
    if !(isdir("io/" * Params.Calc.Path * "/QTDA/bin"))
        mkdir("io/" * Params.Calc.Path * "/QTDA/bin")
    end
    if !(isdir("io/" * Params.Calc.Path * "/QTDA/densities"))
        mkdir("io/" * Params.Calc.Path * "/QTDA/densities")
    end
    if !(isdir("io/" * Params.Calc.Path * "/QTDA/transitions"))
        mkdir("io/" * Params.Calc.Path * "/QTDA/transitions")
        mkdir("io/" * Params.Calc.Path * "/QTDA/transitions/E0")
        mkdir("io/" * Params.Calc.Path * "/QTDA/transitions/E1")
        mkdir("io/" * Params.Calc.Path * "/QTDA/transitions/E2")
        mkdir("io/" * Params.Calc.Path * "/QTDA/transitions/E3")
    end
    if !(isdir("io/" * Params.Calc.Path * "/QTDA/spectra"))
        mkdir("io/" * Params.Calc.Path * "/QTDA/spectra")
    end
    
    println("\nStarting QTDA calculation ...")

    # Start QTDA calculation ...
    @time QTDA_solver(Params)
    
    println("\nSpherical QTDA solver run completed ...\n")

    return 
end


