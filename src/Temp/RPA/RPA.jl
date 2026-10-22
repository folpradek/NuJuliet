function RPA(Params::Parameters)
    # Initialize Wigner symbols ...
    wigner_init_float(75, "Jmax", 9)

    # Calculation parameters
    println("Starting RPA calculations with residual NO2B NN+NNN interaction & given reference ground state ...")
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
    if !(isdir("IO/" * Params.Calc.Path))
        println("\nError! ... No precomputed reference solution is available in given Input_File path ... e.g. run HF solver first ...")
        return
    end
    if (isdir("IO/" * Params.Calc.Path * "/RPA"))
        rm("IO/" * Params.Calc.Path * "/RPA", recursive = true)
    end
    mkdir("IO/" * Params.Calc.Path * "/RPA")
    if !(isdir("IO/" * Params.Calc.Path * "/RPA/Densities"))
        mkdir("IO/" * Params.Calc.Path * "/RPA/Densities")
    end
    if !(isdir("IO/" * Params.Calc.Path * "/RPA/Transitions"))
        mkdir("IO/" * Params.Calc.Path * "/RPA/Transitions")
        mkdir("IO/" * Params.Calc.Path * "/RPA/Transitions/E0")
        mkdir("IO/" * Params.Calc.Path * "/RPA/Transitions/E1")
        mkdir("IO/" * Params.Calc.Path * "/RPA/Transitions/E2")
        mkdir("IO/" * Params.Calc.Path * "/RPA/Transitions/E3")
    end
    if !(isdir("IO/" * Params.Calc.Path * "/RPA/Spectra"))
        mkdir("IO/" * Params.Calc.Path * "/RPA/Spectra")
    end
    if !(isdir("IO/" * Params.Calc.Path * "/RPA/Amplitudes"))
        mkdir("IO/" * Params.Calc.Path * "/RPA/Amplitudes")
    end
    
    println("\nStarting RPA calculation ...")
    
    # Start RPA calculation ...
    @time RPA_solver(Params)
    
    println("\nAll calculations have finished ...\n")

    return
end