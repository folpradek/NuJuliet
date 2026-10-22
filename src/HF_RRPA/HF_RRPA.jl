function HF_RRPA(Params::Parameters)
    # Initialize Wigner symbols ...
    wigner_init_float(75,"Jmax",9)

    # Calculation parameters
    println("Starting Renormalized-RPA calculations with residual NO2B NN+NNN interaction")
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
    println("\nInput/Output path to the calculation reads ... ''IO/" * Params.Calc.Path * "''")
    if !(isdir("IO/" * Params.Calc.Path))
        println("\nError! ... No precomputed HF solution is available in given Input_File path ... run HF solver first ...")
        return
    end
    if (isdir("IO/" * Params.Calc.Path * "/RRPA"))
        rm("IO/" * Params.Calc.Path * "/RRPA", recursive = true)
    end
    mkdir("IO/" * Params.Calc.Path * "/RRPA")
    if !(isdir("IO/" * Params.Calc.Path * "/RRPA/Densities"))
        mkdir("IO/" * Params.Calc.Path * "/RRPA/Densities")
    end
    if !(isdir("IO/" * Params.Calc.Path * "/RRPA/Transitions"))
        mkdir("IO/" * Params.Calc.Path * "/RRPA/Transitions")
        mkdir("IO/" * Params.Calc.Path * "/RRPA/Transitions/E0")
        mkdir("IO/" * Params.Calc.Path * "/RRPA/Transitions/E1")
        mkdir("IO/" * Params.Calc.Path * "/RRPA/Transitions/E2")
        mkdir("IO/" * Params.Calc.Path * "/RRPA/Transitions/E3")
    end
    if !(isdir("IO/" * Params.Calc.Path * "/RRPA/Spectra"))
        mkdir("IO/" * Params.Calc.Path * "/RRPA/Spectra")
    end
    if !(isdir("IO/" * Params.Calc.Path * "/RRPA/Amplitudes"))
        mkdir("IO/" * Params.Calc.Path * "/RRPA/Amplitudes")
    end
    
    # Start RRPA calculation ...
    println("\nStarting RRPA calculation ...")
    @time HF_RRPA_solver(Params)

    println("\nAll calculations have finished ...\n")

    return
end