function HFB_Broyden_initialize_mapping(Params::Parameters,m::Int64,Orb::Vector{Orb1B})
    # Read calculation parameters ...
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1) * (N_max + 2), 2)

    # Initialize the counter of linearized indices & Broyden mapping ...
    M_count, Broyden_Map = 0, zeros(Int64,a_max,a_max)

    # Enumerate M_count & allocate Broyden mapping ...
    @inbounds for a in 1:a_max
        l_a, j_a = Orb[a].l, Orb[a].j
        @inbounds for b in 1:a_max
            l_b, j_b = Orb[b].l, Orb[b].j
            if (l_a == l_b) && (j_a == j_b)
                M_count += 1
                Broyden_Map[a,b] = M_count 
            end
        end
    end

    # Initialize the total number of linearized indices & Broyden key ...
    M, Broyden_Key = 0, Vector{HFB_Broyden_Key}(undef,M_count)

    # Enumerate M & allocate Broyden key ...
    @inbounds for a in 1:a_max
        l_a, j_a = Orb[a].l, Orb[a].j
        @inbounds for b in 1:a_max
            l_b, j_b = Orb[b].l, Orb[b].j
            if (l_a == l_b) && (j_a == j_b)
                M += 1
                Broyden_Key[M] = HFB_Broyden_Key(a,b) 
            end
        end
    end

    return HFB_Broyden(Broyden_Map,Broyden_Key,M,m), M
end

function HFB_Broyden_initialize(Params::Parameters,Rho::O1B,Kappa::O1B,Orb::Vector{Orb1B})
    # Number of history vectors ... m = 8 by default ...
    m = Params.Calc.HFB.Broy_m

    # Prepare Broyden ... Map, Key & calculate the number of linearized indices M ...
    Broyden, M = HFB_Broyden_initialize_mapping(Params,m,Orb)

    # Initialize arrays x & r ...
    x_Rho, x_Kappa = HFB_Broyden_linearize(Rho,Kappa,Broyden;Initialize = true)
    y_Rho, y_Kappa = pnVector(zeros(Float64,M),zeros(Float64,M)), pnVector(zeros(Float64,M),zeros(Float64,M))
    z_Rho, z_Kappa = pnVector(zeros(Float64,M),zeros(Float64,M)), pnVector(zeros(Float64,M),zeros(Float64,M))
    r_Rho, r_Kappa = pnVector(zeros(Float64,M),zeros(Float64,M)), pnVector(zeros(Float64,M),zeros(Float64,M))
    s_Rho, s_Kappa = pnVector(zeros(Float64,M),zeros(Float64,M)), pnVector(zeros(Float64,M),zeros(Float64,M))

    # Initialize history vectors X & R ...
    X_Rho, X_Kappa = O1B(zeros(Float64,M,m),zeros(Float64,M,m)), O1B(zeros(Float64,M,m),zeros(Float64,M,m))
    R_Rho, R_Kappa = O1B(zeros(Float64,M,m),zeros(Float64,M,m)), O1B(zeros(Float64,M,m),zeros(Float64,M,m))

    # Initialize Broyden vector ...
    Broyden_Vector = HFB_Broyden_Vector(x_Rho,y_Rho,z_Rho,X_Rho,r_Rho,s_Rho,R_Rho,x_Kappa,y_Kappa,z_Kappa,X_Kappa,r_Kappa,s_Kappa,R_Kappa)

    return Broyden, Broyden_Vector
end

function HFB_Broyden_linearize(Rho::O1B,Kappa::O1B,Broyden::HFB_Broyden;Initialize::Bool=false)
    # Initialize linearized Rho & Kappa ...
    pRho, pKappa = zeros(Float64,Broyden.M), zeros(Float64,Broyden.M)
    nRho, nKappa = zeros(Float64,Broyden.M), zeros(Float64,Broyden.M)

    # Allocate linearized Rho & Kappa ...
        # Paralelize using threads (???)
    @inbounds for i in 1:Broyden.M
        a, b = Broyden.Key[i].a, Broyden.Key[i].b
        pRho[i], pKappa[i] = Rho.p[a,b], Kappa.p[a,b]
        nRho[i], nKappa[i] = Rho.n[a,b], Kappa.n[a,b]
    end

    if Initialize == false
        return pRho, pKappa, nRho, nKappa
    elseif Initialize == true
        return pnVector(pRho,nRho), pnVector(pKappa,nKappa)
    end
end

function HFB_Broyden_reconstruct(Params::Parameters,Rho::pnVector,Kappa::pnVector,Broyden::HFB_Broyden)
    # Read calculation parameters ...
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1) * (N_max + 2), 2)

    # Initialize density matrices Rho & Kappa ...
    pRho, pKappa = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)
    nRho, nKappa = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)

    # Allocate density matrices Rho & Kappa ...
        # Paralelize using threads (???)
    @inbounds for a in 1:a_max
        @inbounds for b in 1:a_max
            i = Broyden.Map[a,b]
            j = Broyden.Map[b,a]
            if i != 0 && j != 0
                pRho[a,b], pKappa[a,b] = 0.5 * (Rho.p[i] + Rho.p[j]), 0.5 * (Kappa.p[i] + Kappa.p[j])
                nRho[a,b], nKappa[a,b] = 0.5 * (Rho.n[i] + Rho.n[j]), 0.5 * (Kappa.n[i] + Kappa.n[j])
            end
        end
    end

    # Clean numerical noise in Rho & Kappa ...
    @inbounds for a in 1:a_max
        @inbounds for b in 1:a_max
            if abs(pRho[a,b]) < 1e-14
                pRho[a,b] = 0.0
            end
            if abs(pKappa[a,b]) < 1e-14
                pKappa[a,b] = 0.0
            end
           if abs(nRho[a,b]) < 1e-14
                nRho[a,b] = 0.0
            end
            if abs(nKappa[a,b]) < 1e-14
                nKappa[a,b] = 0.0
            end
        end
    end

    return O1B(pRho,nRho), O1B(pKappa,nKappa)
end

function HFB_Broyden_history_update(Iteration::Int64,Broyden::HFB_Broyden,BroyVec::HFB_Broyden_Vector)
    # Check if there is enough history vectors to perform update ...
        # Not enough history vectors ... beta is trivial ...
    if Iteration < (Broyden.m + 3)
        return zeros(Float64,Broyden.m), zeros(Float64,Broyden.m), zeros(Float64,Broyden.m), zeros(Float64,Broyden.m)

        # There is enough history vectors ... beta is non-trivial ...
    else
        # Allocate temporary R vectors ... to be rescaled ...
        R_pRho, R_pKappa = copy(BroyVec.R_Rho.p), copy(BroyVec.R_Kappa.p)
        R_nRho, R_nKappa = copy(BroyVec.R_Rho.n), copy(BroyVec.R_Kappa.n)

        # Evaluate the current history vector index m ...
        m = rem(Iteration,Broyden.m) + 1
        m_current = m

        # Rescale R vectors ...
        @inbounds for n in 1:Broyden.m
            # History decay constant ...
            f = exp(-0.3 * (n - 1))

            # Rescalling of each history vector ...
            @views begin
                c = @views R_pRho[:, m_current]
                d = norm(BroyVec.r_Rho.p)
                if d > 0
                    c .*= f * exp(-(norm(c) / d - 1.0))
                else
                    c .*= f
                end
            end

            @views begin
                c = @views R_nRho[:, m_current]
                d = norm(BroyVec.r_Rho.n)
                if d > 0
                    c .*= f * exp(-(norm(c) / d - 1.0))
                else
                    c .*= f
                end
            end

            @views begin
                c = @views R_pKappa[:, m_current]
                d = norm(BroyVec.r_Kappa.p)
                if d > 0
                    c .*= f * exp(-(norm(c) / d - 1.0))
                else
                    c .*= f
                end
            end

            @views begin
                c = @views R_nKappa[:, m_current]
                d = norm(BroyVec.r_Kappa.n)
                if d > 0
                    c .*= f * exp(-(norm(c) / d - 1.0))
                else
                    c .*= f
                end
            end

            # Evaluatthe current history vector index m ...
            m_current += 1
            if m_current > Broyden.m
                m_current = 1
            end
        end

        # Define the residual overlap matrix N = R' * R ...
        N_pRho, N_pKappa = R_pRho' * R_pRho, R_pKappa' * R_pKappa
        N_nRho, N_nKappa = R_nRho' * R_nRho, R_nKappa' * R_nKappa

        # Calculate the regularizing factors epsilon ...
        Eps_pRho = max(1e-4 * opnorm(N_pRho,2), 1e-7) .* diagm(ones(Float64,Broyden.m))
        Eps_nRho = max(1e-4 * opnorm(N_nRho,2), 1e-7) .* diagm(ones(Float64,Broyden.m))
        Eps_pKappa = max(1e-4 * opnorm(N_pKappa,2), 1e-7) .* diagm(ones(Float64,Broyden.m))
        Eps_nKappa = max(1e-4 * opnorm(N_nKappa,2), 1e-7) .* diagm(ones(Float64,Broyden.m))

        # Calculate & allocate beta ...
        beta_pRho = (N_pRho .+ Eps_pRho) \ (R_pRho' * BroyVec.r_Rho.p)
        beta_nRho = (N_nRho .+ Eps_nRho) \ (R_nRho' * BroyVec.r_Rho.n)
        beta_pKappa = (N_pKappa .+ Eps_pKappa) \ (R_pKappa' * BroyVec.r_Kappa.p)
        beta_nKappa = (N_nKappa .+ Eps_nKappa) \ (R_nKappa' * BroyVec.r_Kappa.n)

        # Set beta to zero if any NaN is present
        if any(isnan, beta_pRho); beta_pRho .= 0.0; end
        if any(isnan, beta_nRho); beta_nRho .= 0.0; end
        if any(isnan, beta_pKappa); beta_pKappa .= 0.0; end
        if any(isnan, beta_nKappa); beta_nKappa .= 0.0; end

        return beta_pRho, beta_pKappa, beta_nRho, beta_nKappa
    end
end

function HFB_Broyden_update(Params::Parameters,Iteration::Int64,Rho::O1B,Kappa::O1B,Broyden::HFB_Broyden,BroyVec::HFB_Broyden_Vector,Orb::Vector{Orb1B})
    # Read the parameters for Broyden update ...
        # Quenching parameters alpha ... by default alpha = 0.9 ...
    alpha_pRho, alpha_pKappa = Params.Calc.HFB.Broy_Amax, Params.Calc.HFB.Broy_Amax
    alpha_nRho, alpha_nKappa = Params.Calc.HFB.Broy_Amax, Params.Calc.HFB.Broy_Amax
        # Maximal relative contribution of beta ... by default beta_max = 0.3 ...
    beta_max = Params.Calc.HFB.Broy_Bmax
        # Maximal number of damping quenches ... by default Quench_max = 3 ...
    Quench_max = Params.Calc.HFB.Broy_Qmax
        # Maximal relative norm of trial change ... by default T_max = 0.975 ...
    trial_max = Params.Calc.HFB.Broy_Tmax

    # Initialize quenching flags ...
    Q_pRho, Q_pKappa = false, false
    Q_nRho, Q_nKappa = false, false

    # History vector index m ...
    m = rem(Iteration,Broyden.m) + 1

    # Linearize Rho & Kappa ...
    pRho, pKappa, nRho, nKappa = HFB_Broyden_linearize(Rho,Kappa,Broyden)

    # Update & allocate vectors r & s ...
        # Case of s = r^(n-1) ...
    BroyVec.s_Rho.p .= BroyVec.r_Rho.p
    BroyVec.s_Rho.n .= BroyVec.r_Rho.n
    BroyVec.s_Kappa.p .= BroyVec.r_Kappa.p
    BroyVec.s_Kappa.n .= BroyVec.r_Kappa.n
        # Case of r = r^(n) ...
    BroyVec.r_Rho.p .= pRho .- BroyVec.x_Rho.p
    BroyVec.r_Rho.n .= nRho .- BroyVec.x_Rho.n
    BroyVec.r_Kappa.p .= pKappa .- BroyVec.x_Kappa.p
    BroyVec.r_Kappa.n .= nKappa .- BroyVec.x_Kappa.n

    # Update history vectors X & R ...
        # Case of X = X^(n) ...
    @views BroyVec.X_Rho.p[:,m] .= BroyVec.x_Rho.p[:] .- BroyVec.y_Rho.p[:]
    @views BroyVec.X_Rho.n[:,m] .= BroyVec.x_Rho.n[:] .- BroyVec.y_Rho.n[:]
    @views BroyVec.X_Kappa.p[:,m] .= BroyVec.x_Kappa.p[:] .- BroyVec.y_Kappa.p[:]
    @views BroyVec.X_Kappa.n[:,m] .= BroyVec.x_Kappa.n[:] .- BroyVec.y_Kappa.n[:]
        # Case of R = R^(n) ...
    @views BroyVec.R_Rho.p[:,m] .= BroyVec.r_Rho.p[:] .- BroyVec.s_Rho.p[:]
    @views BroyVec.R_Rho.n[:,m] .= BroyVec.r_Rho.n[:] .- BroyVec.s_Rho.n[:]
    @views BroyVec.R_Kappa.p[:,m] .= BroyVec.r_Kappa.p[:] .- BroyVec.s_Kappa.p[:]
    @views BroyVec.R_Kappa.n[:,m] .= BroyVec.r_Kappa.n[:] .- BroyVec.s_Kappa.n[:]

    # Allocate finite difference D for previous residual r ...
    D_pRho, D_pKappa = norm(BroyVec.r_Rho.p), norm(BroyVec.r_Kappa.p)
    D_nRho, D_nKappa = norm(BroyVec.r_Rho.n), norm(BroyVec.r_Kappa.n)

    # Evaluate the Broyden's fit matrix beta ...
    beta_pRho, beta_pKappa, beta_nRho, beta_nKappa = HFB_Broyden_history_update(Iteration,Broyden,BroyVec)

    # Perform several quenching iterations for densities ...
    @inbounds for Quench in 1:Quench_max

        # Quench pRho ...
        if Q_pRho == false
            alpha_r = alpha_pRho .* BroyVec.r_Rho.p
            X_beta = BroyVec.X_Rho.p * beta_pRho
            dX_beta_max = beta_max * norm(alpha_r)
            if dX_beta_max > 1e-8 && norm(X_beta) > dX_beta_max
                X_beta .= X_beta .* (dX_beta_max / (norm(X_beta) + 1e-10))
            end

            #pRho_trial = BroyVec.x_Rho.p .+ alpha_r #.- X_beta
            pRho_trial = pRho
            D_pRho_trial = norm(pRho .- pRho_trial)

            if ((D_pRho_trial / D_pRho) < trial_max) || (Quench == Quench_max)
                BroyVec.z_Rho.p .= pRho_trial
                Q_pRho = true
            else
                alpha_pRho = 0.5 * alpha_pRho
            end
        end

        # Quench pKappa ...
        if Q_pKappa == false
            alpha_r = alpha_pKappa .* BroyVec.r_Kappa.p
            X_beta = BroyVec.X_Kappa.p * beta_pKappa
            dX_beta_max = beta_max * norm(alpha_r)
            if dX_beta_max > 1e-8 && norm(X_beta) > dX_beta_max
                X_beta .= X_beta .* (dX_beta_max / (norm(X_beta) + 1e-10))
            end

            #pKappa_trial = BroyVec.x_Kappa.p .+ alpha_r #.- X_beta
            pKappa_trial = pKappa

            D_pKappa_trial = norm(pKappa .- pKappa_trial)

            if ((D_pKappa_trial / D_pKappa) < trial_max) || (Quench == Quench_max)
                BroyVec.z_Kappa.p .= pKappa_trial
                Q_pKappa = true
            else 
                alpha_pKappa = 0.5 * alpha_pKappa
            end
        end

        # Quench nRho ...
        if Q_nRho == false
            alpha_r = alpha_nRho .* BroyVec.r_Rho.n
            X_beta = BroyVec.X_Rho.n * beta_nRho
            dX_beta_max = beta_max * norm(alpha_r)
            if dX_beta_max > 1e-8 && norm(X_beta) > dX_beta_max
                X_beta .= X_beta .* (dX_beta_max / (norm(X_beta) + 1e-10))
            end

            #nRho_trial = BroyVec.x_Rho.n .+ alpha_r #.- X_beta
            nRho_trial = nRho

            D_nRho_trial = norm(nRho .- nRho_trial)

            if ((D_nRho_trial / D_nRho) < trial_max) || (Quench == Quench_max)
                BroyVec.z_Rho.n .= nRho_trial
                Q_nRho = true
            else
                alpha_nRho = 0.5 * alpha_nRho
            end
        end

        # Quench nKappa ...
        if Q_nKappa == false
            alpha_r = alpha_nKappa .* BroyVec.r_Kappa.n
            X_beta = BroyVec.X_Kappa.n * beta_nKappa
            dX_beta_max = beta_max * norm(alpha_r)
            if dX_beta_max > 1e-8 && norm(X_beta) > dX_beta_max
                X_beta .= X_beta .* (dX_beta_max / (norm(X_beta) + 1e-10))
            end

            #nKappa_trial = BroyVec.x_Kappa.n .+ alpha_r #.- X_beta
            nKappa_trial = nKappa

            D_nKappa_trial = norm(nKappa .- nKappa_trial)

            if ((D_nKappa_trial / D_nKappa) < trial_max) || (Quench == Quench_max)
                BroyVec.z_Kappa.n .= nKappa_trial
                Q_nKappa = true
            else
                alpha_nKappa = 0.5 * alpha_nKappa
            end
        end

        # Break loop if update is done ...
        if Q_pRho == true && Q_pKappa == true && Q_nRho == true && Q_nKappa == true 
            break
        end

    end

    # Update previous step vector y ...
    BroyVec.y_Rho.p .= BroyVec.x_Rho.p
    BroyVec.y_Rho.n .= BroyVec.x_Rho.n
    BroyVec.y_Kappa.p .= BroyVec.x_Kappa.p
    BroyVec.y_Kappa.n .= BroyVec.x_Kappa.n

    # Update current step vector x ...
    BroyVec.x_Rho.p .= BroyVec.z_Rho.p
    BroyVec.x_Rho.n .= BroyVec.z_Rho.n
    BroyVec.x_Kappa.p .= BroyVec.z_Kappa.p
    BroyVec.x_Kappa.n .= BroyVec.z_Kappa.n

    # Update densities Rho & Kappa ...
    Rho, Kappa = HFB_Broyden_reconstruct(Params,BroyVec.z_Rho,BroyVec.z_Kappa,Broyden)

    return Rho, Kappa, BroyVec
end