function pnQTDA(Params::Parameters)
    # Initialize Wigner symbols ...
    wigner_init_float(75,"Jmax",9)

    # Calculation parameters ...
    println("Starting spherical pnQTDA calculation with residual NO2B NN+NNN interaction based on available reference quasiparticle mean-field calculation")
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
    if !(isfile("io/" * Params.Calc.Path * "/" * Params.Calc.pnQTDA.Reference * "/bin/qpH1b.bin")) ||
       !(isfile("io/" * Params.Calc.Path * "/" * Params.Calc.pnQTDA.Reference * "/bin/qpH2b.bin"))
        println("\nError! ... No precomputed reference quasiparticle mean-field (HFB, BCS) solution is available in given Input_File path ... run HFB/BCS solver first ...")
        return
    end
    if (isdir("io/" * Params.Calc.Path * "/pnQTDA"))
        rm("io/" * Params.Calc.Path * "/pnQTDA", recursive = true)
    end
    mkdir("io/" * Params.Calc.Path * "/pnQTDA")
    mkdir("io/" * Params.Calc.Path * "/pnQTDA/bin")
    mkdir("io/" * Params.Calc.Path * "/pnQTDA/spectra")

    println("\nStarting pnQTDA calculation ...")

    # Start pnQTDA calculation ...
    @time pnQTDA_solver(Params)

    println("\nSpherical pnQTDA solver run completed ...\n")

    return
end
