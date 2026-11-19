function TDA(Params::Parameters)
    # Read the TDA parameters ...
    TDA_Params = Params.Calc.TDA

    # Initialize Wigner symbols ...
    wigner_init_float(75, "Jmax", 9)

    # Calculation parameters
    println("Starting TDA calculations with residual NO2B NN+NNN interaction & for given reference solution ...")
    println("\nCalculation data:")
    println("A = " * string(Params.Calc.A) * " , Z = " * string(Params.Calc.Z) * " , hw = " * string(Params.Calc.hw) *
            " MeV , N_max = " * string(Params.Calc.Nmax) * " , J-scheme LHO basis size = " * string(div((Params.Calc.Nmax+1)*(Params.Calc.Nmax+2),2)) *
            " , M-scheme LHO basis size = " * string(div((Params.Calc.Nmax+1)*(Params.Calc.Nmax+2)*(Params.Calc.Nmax+3),6)))
    println("\nTDA parameters:    Name = ''" * TDA_Params.Name * "'' , Reference = ''" * TDA_Params.Reference *
            "'' , Reference_TBCF = " * string(TDA_Params.Reference_TBCF) * " , Ortho = " * string(TDA_Params.Ortho) *
            " , Tol_hol = " * string(TDA_Params.Tol_hol) * " , Tol_par = " * string(TDA_Params.Tol_par))

    if Threads.nthreads() > 1
        println("\n_________________________________________________________")
        println("Multiple active threads detected ...")
        println("Parallelization report:    Number of active threads = " * string(Threads.nthreads()))
        println("_________________________________________________________")
    end

    # Check if reference solution exists ...
    IO_ref = "io/" * Params.Calc.Path * "/" * TDA_Params.Reference
    if !(isdir(IO_ref))
        println("\nError! ... No precomputed reference solutions available in IO directory ''$("io/" * Params.Calc.Path)'' for given input path ''$(TDA_Params.Reference)''... e.g. run HF solver first ...")
        return
    end

    # Check if the reference 2-body correlation function exists ... (Reference_TBCF = true)
    if TDA_Params.Reference_TBCF == true && !isfile(IO_ref * "/bin/sigma.bin")
        println("\nError! ... Reference_TBCF = true, but no reference 2-body correlation function ''" * IO_ref * "/bin/sigma.bin'' is available ...")
        return
    end

    # Make new directories for results ...
    IO_TDA = "io/" * Params.Calc.Path * "/" * TDA_Params.Name

    # Safety checks ... the output directory is removed before the run, hence it must not be the reference directory,
    # nor any other existing non-TDA output directory (e.g. RPA results) ...
    if TDA_Params.Name == TDA_Params.Reference
        println("\nError! ... TDA output directory ''$(IO_TDA)'' coincides with the reference directory ... choose a different TDA_Parameters.Name ...")
        return
    end

    if isdir(IO_TDA)
        if isfile(IO_TDA * "/.TDA_output") || isempty(readdir(IO_TDA))
            rm(IO_TDA, recursive = true)
        else
            println("\nError! ... Output directory ''$(IO_TDA)'' exists & does not contain a previous TDA run ... it is NOT overwritten ... choose a different TDA_Parameters.Name ...")
            return
        end
    end

    mkdir(IO_TDA)

    mkdir(IO_TDA * "/bin")
    mkdir(IO_TDA * "/densities")
    mkdir(IO_TDA * "/transitions")
    mkdir(IO_TDA * "/transitions/E0")
    mkdir(IO_TDA * "/transitions/E1")
    mkdir(IO_TDA * "/transitions/E2")
    mkdir(IO_TDA * "/transitions/E3")
    mkdir(IO_TDA * "/spectra")
    mkdir(IO_TDA * "/amplitudes")

    println("\nStarting TDA calculation ...")

    # Run the TDA calculation ...
    @time TDA_solver(Params)

    println("\nTDA calculation has finished ...\n")

    return
end
