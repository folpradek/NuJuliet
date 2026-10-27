function RPA(Params::Parameters)
    # Initialize Wigner symbols ...
    wigner_init_float(75, "Jmax", 9)

    # Calculation parameters
    println("Starting RPA calculations with residual NO2B NN+NNN interaction & for given reference solutions ...")
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

    # Check if reference solution exists ...
    if !(isdir("io/" * Params.Calc.Path * "/" * Params.Calc.RPA.Reference))
        println("\nError! ... No precomputed reference solutions available in IO directory ''$("io/" * Params.Calc.Path)'' for given input path ''$(Params.Calc.RPA.Reference)''... e.g. run HF solver first ...")
        return
    end

    # Make new directories for results ...
    if !(isdir("io/" * Params.Calc.Path))
        mkdir("io/" * Params.Calc.Path)
        return
    end

    IO_RPA = "io/" * Params.Calc.Path * "/" * Params.Calc.RPA.Name

    if (isdir(IO_RPA))
        rm(IO_RPA, recursive = true)
    end
    
    mkdir(IO_RPA)
    if !(isdir(IO_RPA * "/bin"))
        mkdir(IO_RPA * "/bin")
    end

    if !(isdir(IO_RPA * "/densities"))
        mkdir(IO_RPA * "/densities")
    end

    if !(isdir(IO_RPA * "/transitions"))
        mkdir(IO_RPA * "/transitions")
        mkdir(IO_RPA * "/transitions/E0")
        mkdir(IO_RPA * "/transitions/E1")
        mkdir(IO_RPA * "/transitions/E2")
        mkdir(IO_RPA * "/transitions/E3")
    end

    if !(isdir(IO_RPA * "/spectra"))
        mkdir(IO_RPA * "/spectra")
    end

    if !(isdir(IO_RPA * "/amplitudes"))
        mkdir(IO_RPA * "/amplitudes")
    end
    
    println("\nStarting RPA calculation ...")
    
    # Run the RPA calculation ...
    @time RPA_solver(Params)
    
    println("\nRPA calculation has finished ...\n")

    return
end