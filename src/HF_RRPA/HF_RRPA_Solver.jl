function HF_RRPA_solver(Params::Parameters)
    # Make s.p. orbitals ...
    Orb = orbitals_make(Params)

    # 1-body kinetic operator ...
    @time T = T1b(Params,Orb)

    # 2-body NN interaction & Orbitals ...
    @time V_NN_Bare, Orb_NN_Bare = V2b_read(Params,Orb)

    # 3-body NNN interaction & Orbitals ...
    @time V_NNN_Bare, Orb_NNN_Bare = V3b_no2b_read(Params,Orb)

    # Prepare Particle & Hole orbitals ...
    N_Particle, Particle, N_Hole, Hole = orbitals_ph_make(Params,Orb)

    # Prepare 1p-1h phonon states ...
    N_Phonon, Phonon = orbitals_one_phonon_make(N_Particle,Particle,N_Hole,Hole)

    # Count & pre-index all phonon states in JP subspaces ...
    N_nu, Orb_Phonon = HF_RRPA_phonon_count(Params,N_Phonon,Phonon)

    # Start RRPA iteration ...
    @time E_RPA, X_RPA, Y_RPA, Rho, C, h = HF_RRPA_solve(Params,N_nu,Orb_Phonon,Phonon,N_Particle,Particle,N_Hole,Hole,Orb,Orb_NN_Bare,Orb_NNN_Bare,T,V_NN_Bare,V_NNN_Bare)

    # Evaluate the RRPA charge radius chR ...
    chR2 = OBDM_chR2(Params,Orb,C,Rho)

    # Initialize transition operators ...
    TrOp = Tr1b_initialize(Params,Orb,chR2)

    TrOp = Tr1b_transformation(Params,Orb,C,TrOp)

    # Electromagnetic reduced multipole operators ...
    rM = HF_RRPA_rM(Params,N_nu,Orb_Phonon,Phonon,Particle,Hole,Rho,TrOp,X_RPA,Y_RPA)

    # Electromagnetic reduced transition intensities ...
    rB = HF_RRPA_rB(Params,N_nu,rM,E_RPA)

    # RRPA ground state energy ...
    E_corr = HF_RRPA_energy(Params,N_nu,N_Particle,Particle,N_Hole,Hole,E_RPA,Y_RPA)

    # RRPA solutions export ...
    @time HF_RRPA_export(Params,Orb,N_nu,E_corr,C,Rho,h,E_RPA,X_RPA,Y_RPA,rB)

        # RRPA transition radial densities export ...
        # Requires manual control in corresponding function in HF_RRPA_Transitions.jl file
        #   !!! One has to manually choose the phonon transition to export !!!
        #@time HF_RRPA_Transition_Densities_Export(Params,Orb,N_nu,Orb_Phonon,Phonon,Particle,Hole,X_RPA,Y_RPA,TrOp)

    # Deallocate the NN & NNN interaction & perform the Garbace Collection ...
    V_NN_Bare, V_NNN_Bare = nothing, nothing
    GC.gc()

    return
end

function HF_RRPA_solve(Params::Parameters,N_nu::Matrix{Int64},Orb_Phonon::Matrix{Vector{Int64}},Phonon::Vector{PhState},N_Particle::pnInteger,Particle::pnSVector,N_Hole::pnInteger,Hole::pnSVector,Orb::Vector{Orb1B},Orb_NN_Bare::Orb2B,Orb_NNN_Bare::Orb3B,T::O1B,V_NN_Bare::O2B,V_NNN_Bare::Array{Vector{Vector{Float32}},4})
    # Read parameters ...
    N_max = Params.Calc.Nmax
    a_max = div((N_max+1)*(N_max+2),2)

    # Iteration parameters ...
    eta, epsilon = 1.0, Params.Calc.RRPA.Tol
    Iteration, Iteration_max = 0, Params.Calc.RRPA.Imax
    E_old, E_new, dE = 2.0, 1.0, 1.0

    # Import residual 2-body interaction ... HF basis ...
    println("\nImporting Residual 2-body interaction ...")
    @time V_NN_Res, Orb_NN_Res = O2b_import(Params,Orb,"IO/" * Params.Calc.Path * "/Bin/V2B_HF.bin",Make_Orb_NN=true)

    # Transformation matrices LHO -> HF bases ...
    C_LHO_HF = O1b_import(Params,Orb,"IO/" * Params.Calc.Path * "/Bin/C_HF.bin")
          
    C_HF_RRPA = O1B(diagm(ones(Float64,a_max)),diagm(ones(Float64,a_max)))

    # Initialize density matrices ...
    Rho = O1B(zeros(Float64,a_max,a_max),zeros(Float64,a_max,a_max))

    Rho_LHO = O1B(zeros(Float64,a_max,a_max),zeros(Float64,a_max,a_max))

    pRho_HF, nRho_HF = zeros(Float64,a_max,a_max),zeros(Float64,a_max,a_max)
    @inbounds for a in 1:a_max
        pRho_HF[a,a] = Orb[a].pO
        nRho_HF[a,a] = Orb[a].nO
    end
    Rho_HF = O1B(pRho_HF,nRho_HF)

    Rho_0 = O1B(deepcopy(pRho_HF), deepcopy(nRho_HF))

    # Initialize 1-body Hamiltonian matrix ...
    H = O1B(zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max))
    
    # Standard RPA calculation - RRPA(0) ...

    # Allocate matrices A & B ...
    println("\nAllocating RPA matrices A & B ...")
    @time A, B = HF_RRPA0_allocate(Params,N_nu,Orb_Phonon,Phonon,Particle,Hole,Orb,Orb_NN_Res,V_NN_Res)

    # Solve RPA generalized-eigenvalue problem ...
    println("\nSolving the RPA Generalized-Eigenvalue Problem ...")
    @time E_RPA, X_RPA, Y_RPA = HF_RRPA0_diagonalize(Params,A,B,N_nu)

    # RRPA iteration ...
    @time while (eta > epsilon) && (Iteration < Iteration_max)

        @time Rho, C = HF_RRPA_OBDM_iteration(Params,N_nu,N_Particle,Particle,N_Hole,Hole,X_RPA,Y_RPA,Rho_HF,Rho_0)

        C_HF_RRPA = O1B(C_HF_RRPA.p * C.p, C_HF_RRPA.n * C.n)

        Rho_0 = O1B(C.p' * Rho_0.p * C.p, C.n' * Rho_0.n * C.n)

        pRho_LHO = C_LHO_HF.p * C_HF_RRPA.p * Rho.p * C_HF_RRPA.p' * C_LHO_HF.p'
        nRho_LHO = C_LHO_HF.n * C_HF_RRPA.n * Rho.n * C_HF_RRPA.n' * C_LHO_HF.n'

        pRho_LHO .= 0.5 .* (pRho_LHO .+ pRho_LHO')
        nRho_LHO .= 0.5 .* (nRho_LHO .+ nRho_LHO')

        Rho_LHO = O1B(pRho_LHO,nRho_LHO)

        # Make 1-body Mean-field Hamiltonian ... in the LHO basis ...
        if Params.Calc.RRPA.ScOBH == true
            H = HF_RRPA_h1b(Params,Rho_LHO,Orb,Orb_NN_Bare,Orb_NNN_Bare,T,V_NN_Bare,V_NNN_Bare)
        elseif Params.Calc.RRPA.ScOBH == false
            H = HF_RRPA_h1b(Params,Rho_HF,Orb,Orb_NN_Bare,Orb_NNN_Bare,T,V_NN_Bare,V_NNN_Bare)
        end

        # Transformation to the HF-RRPA basis ...
        H = O1B(deepcopy(C_HF_RRPA.p' * C_LHO_HF.p' * H.p * C_LHO_HF.p * C_HF_RRPA.p), deepcopy(C_HF_RRPA.n' * C_LHO_HF.n' * H.n * C_LHO_HF.n * C_HF_RRPA.n))
        H = O1B(0.5 .* (H.p + H.p'), 0.5 .* (H.n + H.n'))

        println("\nUpdating the residual interaction ...")
        if Params.Calc.RRPA.ScV3N== true
            @time V_NN_Res = HF_RRPA_V2b(Params,Orb,Orb_NN_Bare,Orb_NNN_Bare,V_NN_Bare,V_NNN_Bare,Rho_LHO,O1B(C_LHO_HF.p * C_HF_RRPA.p, C_LHO_HF.n * C_HF_RRPA.n))
        elseif Params.Calc.RRPA.ScV3N== false && Params.Calc.RRPA.OBDM == "Full"
            @time V_NN_Res = HF_RRPA_V2b_transform(Params,Orb,Orb_NN_Res,V_NN_Res,O1B(C.p, C.n))
        end

        # Allocate A & B matrices ...
        println("\nAllocating RPA matrices A & B ...")
        @time A, B = HF_RRPA_allocate(Params,N_nu,Orb_Phonon,Phonon,Particle,Hole,Orb,Orb_NN_Res,V_NN_Res,Rho,H)

        # Solve RPA eqs. ...
        println("\nSolving the RPA Generalized-Eigenvalue Problem ...")
        @time E_RPA, X_RPA, Y_RPA = HF_RRPA_iteration_diagonalize(Params,A,B,N_nu)

        Sum = 0.0
        @inbounds for a in 1:a_max
            @inbounds for b in 1:a_max
                ME = (abs(Rho.p[a,b] - Rho_0.p[a,b]) + abs(Rho.n[a,b] - Rho_0.n[a,b])) / Float64(2*a_max^2)
                Sum += ME
            end
        end

        # Evaluate the RRPA ground-state energy ...
            # To avoid spam, Terminal Output is supressed for this call ...
        Out = "/dev/null"
        if Sys.iswindows()
            Out = "NUL"
        end
        open(Out, "w") do devnull_io
            redirect_stdout(devnull_io) do
                redirect_stderr(devnull_io) do
                    E_new = HF_RRPA_energy(Params,N_nu,N_Particle,Particle,N_Hole,Hole,E_RPA,Y_RPA)
                    return E_new
                end
            end
        end

        dE = abs(E_new - E_old)
        
        if abs(Sum - eta) < epsilon && (Iteration > 50) && (E_new < E_old)
            println("\nHF-RRPA iteration finished - cycled degenerate solution found ...")
            println("\tHThe solution with lower total energy was picked ...")
            break
        elseif (abs(Sum) < 1e2 * epsilon) && (Iteration > 75) && (E_new < E_old)
            println("\nRRPA iteration has finished ... Solution converged with lesser precisions 100.0 * Tol ...")
            println("\tHThe solution with lower total energy was picked ...")
            break
        elseif (abs(Sum) < 1e3 * epsilon) && (Iteration > 100) && (E_new < E_old)
            println("\nRRPA iteration has finished ... Solution converged with lesser precisions 1000.0 * Tol ...")
            println("\tHThe solution with lower total energy was picked ...")
            break
        end

        eta = Sum
        Iteration += 1

        println("\tRRPA iteration:   Iteration Number = " * string(Iteration) * ",\teta = " * string(eta))

        Rho_0 = O1B(deepcopy(Rho.p),deepcopy(Rho.n))
    end

    println("\nHF-RRPA iteration has terminated ...")

    # Final calculation with/without Orthogonalization ...

    # Allocate A & B matrices ...
    println("\nAllocating RPA matrices A & B ...")
    @time A, B = HF_RRPA_allocate(Params,N_nu,Orb_Phonon,Phonon,Particle,Hole,Orb,Orb_NN_Res,V_NN_Res,Rho,H)

    # Initialize transition operators ...
    @time TrOp = Tr1b_initialize(Params,Orb,1.0)

    @time TrOp = Tr1b_transformation(Params,Orb,O1B(C_LHO_HF.p * C_HF_RRPA.p, C_LHO_HF.n * C_HF_RRPA.n),TrOp)

    # Solve RPA eqs. ...
    println("\nSolving the RPA Generalized-Eigenvalue Problem ...")
    @time E_RPA, X_RPA, Y_RPA = HF_RRPA_diagonalize(Params,A,B,N_nu,Orb_Phonon,Phonon,Particle,Hole,TrOp,Rho,X_RPA,Y_RPA)

    # Intermediate evaluation of RRPA correlation energy ... testing purposes!
    HF_RRPA_energy_bosonic(Params,N_nu,N_Particle,Particle,N_Hole,Hole,E_RPA,Y_RPA)

    HF_RRPA_energy_density(Params,N_nu,N_Particle,Particle,N_Hole,Hole,Orb,H,Orb_NN_Res,V_NN_Res,A,E_RPA,Y_RPA,Rho)

    #E_HF = HF_Energy(Params,Rho_LHO,Orb,Orb_NN_Bare,Orb_NNN_Bare,T,V_NN_Bare,V_NNN_Bare)

    # Deallocate the NN interaction & perform garbace collection ...
    V_NN_Res = nothing
    GC.gc()

    return E_RPA, X_RPA, Y_RPA, Rho, O1B(C_LHO_HF.p * C_HF_RRPA.p, C_LHO_HF.n * C_HF_RRPA.n), H
end