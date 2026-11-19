# Modified Broyden mixing of HFB densities Rho & Kappa ...
#   D. D. Johnson, Phys. Rev. B 38, 12807 (1988) ... A. Baran et al., Phys. Rev. C 78, 014318 (2008) ...
#
#   The HFB iteration is a fixed-point problem x = F(x) for the vector x = (pRho, nRho, pKappa, nKappa) ...
#   F(x) are the densities obtained by the diagonalization of the HFB matrix built from the input densities x ...
#   The residual is r = F(x) - x & the new input densities are given by the Broyden update
#
#       x_new = x + alpha * r - sum_k gamma_k * (dX_k + alpha * dR_k) ,
#
#   where dX_k = (x_k+1 - x_k) / |r_k+1 - r_k| & dR_k = (r_k+1 - r_k) / |r_k+1 - r_k| are the last m (normalized)
#   differences of inputs & residuals ... the coefficients gamma = (w0^2 I + dR' dR)^(-1) dR' r are the least-squares
#   fit of the current residual r by the residual history dR ... (alpha = 1 & gamma = 0 is the plain HFB iteration) ...
#
#   Proton & neutron Rho & Kappa share ONE Broyden Jacobian ... the proton-neutron & Rho-Kappa couplings are included ...
#
#   Meaning of the fields of HFB_Broyden_Vector ... (Rho & Kappa analogously) ...
#       x ... current input density                y ... previous input density               z ... new (mixed) input density
#       r ... current residual F(x) - x            s ... previous residual
#       X ... history of normalized dX (M x m)     R ... history of normalized dR (M x m)
#
#   Used parameters of HFB_Parameters ...
#       Broy_m    ... number of history vectors m                          (default 8)
#       Broy_Amax ... linear mixing alpha                                  (default 0.9)
#       Broy_Tmax ... history is restarted if |r_new| > |r_old| / Broy_Tmax (default 0.9)

function HFB_Broyden_initialize_mapping(Params::Parameters,m::Int64,Orb::Vector{Orb1B})
    # Read calculation parameters ...
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1) * (N_max + 2), 2)

    # Initialize the number of linearized indices M & Broyden mapping ...
    M, Broyden_Map = 0, zeros(Int64,a_max,a_max)

    # Enumerate M & allocate Broyden mapping ... Rho & Kappa are symmetric (l,j)-blocks ... only a <= b is stored ...
    @inbounds for a in 1:a_max
        l_a, j_a = Orb[a].l, Orb[a].j
        @inbounds for b in a:a_max
            l_b, j_b = Orb[b].l, Orb[b].j
            if (l_a == l_b) && (j_a == j_b)
                M += 1
                Broyden_Map[a,b] = M
                Broyden_Map[b,a] = M
            end
        end
    end

    # Allocate Broyden key ... the inverse of the Broyden mapping ...
    Broyden_Key = Vector{HFB_Broyden_Key}(undef,M)
    @inbounds for a in 1:a_max
        @inbounds for b in a:a_max
            i = Broyden_Map[a,b]
            if i != 0
                Broyden_Key[i] = HFB_Broyden_Key(a,b)
            end
        end
    end

    return HFB_Broyden(Broyden_Map,Broyden_Key,M,m), M
end

function HFB_Broyden_weights(Broyden::HFB_Broyden,Orb::Vector{Orb1B})
    # Weights w of linearized elements ... |x|^2 = sum_i (w_i x_i)^2 is the M-scheme Frobenius norm of the density matrix ...
    #   (2j + 1) ... degeneracy of the (l,j)-block ... 2 ... the off-diagonal elements (a,b) & (b,a) are stored only once ...
    w = zeros(Float64,Broyden.M)
    @inbounds for i in 1:Broyden.M
        a, b = Broyden.Key[i].a, Broyden.Key[i].b
        w[i] = sqrt(Float64(Orb[a].j + 1))
        if a != b
            w[i] *= sqrt(2.0)
        end
    end

    return w
end

function HFB_Broyden_initialize(Params::Parameters,Rho::O1B,Kappa::O1B,Orb::Vector{Orb1B})
    # Number of history vectors ... m = 8 by default ...
    m = Params.Calc.HFB.Broy_m

    # Prepare Broyden ... Map, Key & calculate the number of linearized indices M ...
    Broyden, M = HFB_Broyden_initialize_mapping(Params,m,Orb)

    # Initialize the input vector x by the initial guess on densities ... other vectors are zero ...
    x_Rho, x_Kappa = HFB_Broyden_linearize(Rho,Kappa,Broyden,Orb)
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

function HFB_Broyden_linearize(Rho::O1B,Kappa::O1B,Broyden::HFB_Broyden,Orb::Vector{Orb1B})
    # Weights of linearized elements ...
    w = HFB_Broyden_weights(Broyden,Orb)

    # Initialize linearized Rho & Kappa ...
    pRho, pKappa = zeros(Float64,Broyden.M), zeros(Float64,Broyden.M)
    nRho, nKappa = zeros(Float64,Broyden.M), zeros(Float64,Broyden.M)

    # Allocate linearized & weighted Rho & Kappa ... symmetrized over (a,b) & (b,a) ...
    @inbounds for i in 1:Broyden.M
        a, b = Broyden.Key[i].a, Broyden.Key[i].b
        pRho[i], pKappa[i] = 0.5 * w[i] * (Rho.p[a,b] + Rho.p[b,a]), 0.5 * w[i] * (Kappa.p[a,b] + Kappa.p[b,a])
        nRho[i], nKappa[i] = 0.5 * w[i] * (Rho.n[a,b] + Rho.n[b,a]), 0.5 * w[i] * (Kappa.n[a,b] + Kappa.n[b,a])
    end

    return pnVector(pRho,nRho), pnVector(pKappa,nKappa)
end

function HFB_Broyden_reconstruct(Params::Parameters,Rho::pnVector,Kappa::pnVector,Broyden::HFB_Broyden,Orb::Vector{Orb1B})
    # Read calculation parameters ...
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1) * (N_max + 2), 2)

    # Weights of linearized elements ...
    w = HFB_Broyden_weights(Broyden,Orb)

    # Initialize density matrices Rho & Kappa ...
    pRho, pKappa = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)
    nRho, nKappa = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)

    # Allocate symmetric density matrices Rho & Kappa ... remove the weights ...
    @inbounds for i in 1:Broyden.M
        a, b = Broyden.Key[i].a, Broyden.Key[i].b
        pRho[a,b], pKappa[a,b] = Rho.p[i] / w[i], Kappa.p[i] / w[i]
        nRho[a,b], nKappa[a,b] = Rho.n[i] / w[i], Kappa.n[i] / w[i]
        pRho[b,a], pKappa[b,a] = pRho[a,b], pKappa[a,b]
        nRho[b,a], nKappa[b,a] = nRho[a,b], nKappa[a,b]
    end

    return O1B(pRho,nRho), O1B(pKappa,nKappa)
end

function HFB_Broyden_norm(Rho::pnVector,Kappa::pnVector)
    # Norm of the full vector (pRho, nRho, pKappa, nKappa) ...
    return sqrt(sum(abs2,Rho.p) + sum(abs2,Rho.n) + sum(abs2,Kappa.p) + sum(abs2,Kappa.n))
end

function HFB_Broyden_history_clear(BroyVec::HFB_Broyden_Vector)
    # Forget all history vectors ... the next Broyden step is the linear mixing ...
    BroyVec.X_Rho.p .= 0.0
    BroyVec.X_Rho.n .= 0.0
    BroyVec.X_Kappa.p .= 0.0
    BroyVec.X_Kappa.n .= 0.0
    BroyVec.R_Rho.p .= 0.0
    BroyVec.R_Rho.n .= 0.0
    BroyVec.R_Kappa.p .= 0.0
    BroyVec.R_Kappa.n .= 0.0

    return
end

function HFB_Broyden_history_update(Iteration::Int64,Broyden::HFB_Broyden,BroyVec::HFB_Broyden_Vector)
    # Store the newest history vectors dX = x - y & dR = r - s ... both normalized by |dR| ...
    #   The history is cyclic ... the oldest vector is overwritten by the newest one ...
    k = rem(Iteration,Broyden.m) + 1

    # Residual difference dR & its norm |dR| ...
    dR_Rho = pnVector(BroyVec.r_Rho.p .- BroyVec.s_Rho.p, BroyVec.r_Rho.n .- BroyVec.s_Rho.n)
    dR_Kappa = pnVector(BroyVec.r_Kappa.p .- BroyVec.s_Kappa.p, BroyVec.r_Kappa.n .- BroyVec.s_Kappa.n)
    dR_norm = HFB_Broyden_norm(dR_Rho,dR_Kappa)

    # Residual has not changed at all ... nothing new to learn (& avoid division by zero) ...
    if dR_norm < 1e-14
        return
    end

    # Allocate history vectors X & R ...
    @views BroyVec.X_Rho.p[:,k] .= (BroyVec.x_Rho.p .- BroyVec.y_Rho.p) ./ dR_norm
    @views BroyVec.X_Rho.n[:,k] .= (BroyVec.x_Rho.n .- BroyVec.y_Rho.n) ./ dR_norm
    @views BroyVec.X_Kappa.p[:,k] .= (BroyVec.x_Kappa.p .- BroyVec.y_Kappa.p) ./ dR_norm
    @views BroyVec.X_Kappa.n[:,k] .= (BroyVec.x_Kappa.n .- BroyVec.y_Kappa.n) ./ dR_norm
    @views BroyVec.R_Rho.p[:,k] .= dR_Rho.p ./ dR_norm
    @views BroyVec.R_Rho.n[:,k] .= dR_Rho.n ./ dR_norm
    @views BroyVec.R_Kappa.p[:,k] .= dR_Kappa.p ./ dR_norm
    @views BroyVec.R_Kappa.n[:,k] .= dR_Kappa.n ./ dR_norm

    return
end

function HFB_Broyden_gamma(Broyden::HFB_Broyden,BroyVec::HFB_Broyden_Vector)
    # Regularization weight w0 ... (Johnson's choice w0 = 0.01) ...
    w0 = 0.01

    # Overlap matrix of history vectors A = dR' dR ... summed over pRho, nRho, pKappa & nKappa ...
    A = BroyVec.R_Rho.p' * BroyVec.R_Rho.p .+ BroyVec.R_Rho.n' * BroyVec.R_Rho.n .+
        BroyVec.R_Kappa.p' * BroyVec.R_Kappa.p .+ BroyVec.R_Kappa.n' * BroyVec.R_Kappa.n

    # Projection of the current residual on history vectors c = dR' r ...
    c = BroyVec.R_Rho.p' * BroyVec.r_Rho.p .+ BroyVec.R_Rho.n' * BroyVec.r_Rho.n .+
        BroyVec.R_Kappa.p' * BroyVec.r_Kappa.p .+ BroyVec.R_Kappa.n' * BroyVec.r_Kappa.n

    # Regularization ... also makes empty (zero) history columns harmless ... their gamma_k = 0 ...
    @inbounds for k in 1:Broyden.m
        A[k,k] += w0^2
    end

    # Solve the least-squares problem for gamma ...
    gamma = Symmetric(A) \ c

    # Fall back to the linear mixing in case of numerical problems ...
    if any(!isfinite, gamma)
        gamma .= 0.0
    end

    return gamma
end

function HFB_Broyden_update(Params::Parameters,Iteration::Int64,Rho::O1B,Kappa::O1B,Broyden::HFB_Broyden,BroyVec::HFB_Broyden_Vector,Orb::Vector{Orb1B})
    # Read the parameters for Broyden update ...
        # Linear mixing alpha ... by default alpha = 0.9 ...
    alpha = Params.Calc.HFB.Broy_Amax
        # Restart threshold ... history is cleared if the residual grows by more than 1 / T_max ... by default T_max = 0.9 ...
    T_max = Params.Calc.HFB.Broy_Tmax

    # The first Broyden iteration ... plain HFB iteration (output densities are used as they are) before it ...
    #   The initial guess is only a rough estimate ... plain iteration at Iteration = 0 also guarantees that the mixed
    #   densities keep the exact particle numbers Z & N in all following iterations ...
    #   The Lipkin-Nogami correction is switched on in HFB_solve after the update with Iteration = 16 (Iteration > 15) ...
    #   its first output densities arrive at Iteration = 17 ... the Broyden starts there with an empty history ...
    Iteration_Broyden = 1
    if Params.Calc.HFB.LNT == true
        Iteration_Broyden = 17
    end

    # Linearize the output densities F(x) = Rho & Kappa ...
    F_Rho, F_Kappa = HFB_Broyden_linearize(Rho,Kappa,Broyden,Orb)

    # Update residual vectors s = r^(n-1) & r = F(x^(n)) - x^(n) ...
    BroyVec.s_Rho.p .= BroyVec.r_Rho.p
    BroyVec.s_Rho.n .= BroyVec.r_Rho.n
    BroyVec.s_Kappa.p .= BroyVec.r_Kappa.p
    BroyVec.s_Kappa.n .= BroyVec.r_Kappa.n
    BroyVec.r_Rho.p .= F_Rho.p .- BroyVec.x_Rho.p
    BroyVec.r_Rho.n .= F_Rho.n .- BroyVec.x_Rho.n
    BroyVec.r_Kappa.p .= F_Kappa.p .- BroyVec.x_Kappa.p
    BroyVec.r_Kappa.n .= F_Kappa.n .- BroyVec.x_Kappa.n

    # Evaluate the new input densities z ...
    if Iteration < Iteration_Broyden
        # Plain HFB iteration ... z = F(x) ...
        BroyVec.z_Rho.p .= F_Rho.p
        BroyVec.z_Rho.n .= F_Rho.n
        BroyVec.z_Kappa.p .= F_Kappa.p
        BroyVec.z_Kappa.n .= F_Kappa.n

    else
        # Update the history ...
        r_norm = HFB_Broyden_norm(BroyVec.r_Rho,BroyVec.r_Kappa)
        s_norm = HFB_Broyden_norm(BroyVec.s_Rho,BroyVec.s_Kappa)
        if (Iteration == Iteration_Broyden) && (Iteration_Broyden > 1)
            # The map F(x) has changed (Lipkin-Nogami switched on) ... old history is not valid ...
            HFB_Broyden_history_clear(BroyVec)
        elseif r_norm > s_norm / T_max
            # The residual has grown ... history is misleading ... restart with the newest vectors only ...
            HFB_Broyden_history_clear(BroyVec)
            HFB_Broyden_history_update(Iteration,Broyden,BroyVec)
        else
            # Regular update ... store the newest vectors dX & dR ...
            HFB_Broyden_history_update(Iteration,Broyden,BroyVec)
        end

        # Evaluate the Broyden coefficients gamma ...
        gamma = HFB_Broyden_gamma(Broyden,BroyVec)

        # Broyden update ... z = x + alpha * r - (X + alpha * R) * gamma ...
        BroyVec.z_Rho.p .= BroyVec.x_Rho.p .+ alpha .* BroyVec.r_Rho.p .- (BroyVec.X_Rho.p .+ alpha .* BroyVec.R_Rho.p) * gamma
        BroyVec.z_Rho.n .= BroyVec.x_Rho.n .+ alpha .* BroyVec.r_Rho.n .- (BroyVec.X_Rho.n .+ alpha .* BroyVec.R_Rho.n) * gamma
        BroyVec.z_Kappa.p .= BroyVec.x_Kappa.p .+ alpha .* BroyVec.r_Kappa.p .- (BroyVec.X_Kappa.p .+ alpha .* BroyVec.R_Kappa.p) * gamma
        BroyVec.z_Kappa.n .= BroyVec.x_Kappa.n .+ alpha .* BroyVec.r_Kappa.n .- (BroyVec.X_Kappa.n .+ alpha .* BroyVec.R_Kappa.n) * gamma
    end

    # Collapsed pairing ... the output Kappa vanishes ... keep the mixed Kappa exactly zero as well ...
    #   (a tiny extrapolated Kappa would generate a tiny Delta ~ Tol, which confuses the closed-shell treatment of Lambda)
    if norm(F_Kappa.p) < 1e-12
        BroyVec.z_Kappa.p .= 0.0
    end
    if norm(F_Kappa.n) < 1e-12
        BroyVec.z_Kappa.n .= 0.0
    end

    # Update previous step vector y = x ...
    BroyVec.y_Rho.p .= BroyVec.x_Rho.p
    BroyVec.y_Rho.n .= BroyVec.x_Rho.n
    BroyVec.y_Kappa.p .= BroyVec.x_Kappa.p
    BroyVec.y_Kappa.n .= BroyVec.x_Kappa.n

    # Update current step vector x = z ...
    BroyVec.x_Rho.p .= BroyVec.z_Rho.p
    BroyVec.x_Rho.n .= BroyVec.z_Rho.n
    BroyVec.x_Kappa.p .= BroyVec.z_Kappa.p
    BroyVec.x_Kappa.n .= BroyVec.z_Kappa.n

    # Update densities Rho & Kappa ...
    Rho, Kappa = HFB_Broyden_reconstruct(Params,BroyVec.z_Rho,BroyVec.z_Kappa,Broyden,Orb)

    return Rho, Kappa, BroyVec
end
