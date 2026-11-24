function QRPA(Params::Parameters)
    # Initialize Wigner symbols ...
    wigner_init_float(75,"Jmax",9)

    # Calculation parameters ...
    println("Starting spherical QRPA calculation with residual NO2B NN+NNN interaction based on available reference quasiparticle mean-field calculation")
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
    if !(isfile("io/" * Params.Calc.Path * "/" * Params.Calc.QRPA.Reference * "/bin/C.bin")) ||
       !(isfile("io/" * Params.Calc.Path * "/" * Params.Calc.QRPA.Reference * "/bin/U.bin")) ||
       !(isfile("io/" * Params.Calc.Path * "/" * Params.Calc.QRPA.Reference * "/bin/V.bin"))
        println("\nError! ... No precomputed reference quasiparticle mean-field (HFB, BCS) solution is available in given Input_File path ... run HFB/BCS solver first ...")
        return
    end
    if (isdir("io/" * Params.Calc.Path * "/QRPA"))
        rm("io/" * Params.Calc.Path * "/QRPA", recursive = true)
    end
    mkdir("io/" * Params.Calc.Path * "/QRPA")
    if !(isdir("io/" * Params.Calc.Path * "/QRPA/bin"))
        mkdir("io/" * Params.Calc.Path * "/QRPA/bin")
    end
    if !(isdir("io/" * Params.Calc.Path * "/QRPA/densities"))
        mkdir("io/" * Params.Calc.Path * "/QRPA/densities")
    end
    if !(isdir("io/" * Params.Calc.Path * "/QRPA/transitions"))
        mkdir("io/" * Params.Calc.Path * "/QRPA/transitions")
        mkdir("io/" * Params.Calc.Path * "/QRPA/transitions/E0")
        mkdir("io/" * Params.Calc.Path * "/QRPA/transitions/E1")
        mkdir("io/" * Params.Calc.Path * "/QRPA/transitions/E2")
        mkdir("io/" * Params.Calc.Path * "/QRPA/transitions/E3")
    end
    if !(isdir("io/" * Params.Calc.Path * "/QRPA/spectra"))
        mkdir("io/" * Params.Calc.Path * "/QRPA/spectra")
    end
    if !(isdir("io/" * Params.Calc.Path * "/QRPA/amplitudes"))
        mkdir("io/" * Params.Calc.Path * "/QRPA/amplitudes")
    end
    
    println("\nStarting QRPA calculation ...")

    # Start QRPA calculation ...
    @time QRPA_solver(Params)
    
    println("\nSpherical QRPA solver run completed ...\n")

    return 
end

