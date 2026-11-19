function HF_radial_potential(Params::Parameters,Orb::Vector{Orb1B},C::O1B,T::O1B,h_N::O1B)
    # Read parameters ...
    hw, A = Params.Calc.hw, Float64(Params.Calc.A)
    N_max = Params.Calc.Nmax
    CMS = Params.Calc.CMS
    a_max = div((N_max + 1)*(N_max + 2),2)

    # Basic constants ...
    hc, m_p, m_n = 197.326980, 938.272013, 939.565346

    println("\nPreparing non-local radial HF mean-field potentials ...")

    # HF potential in the LHO basis ... remove the 1-body kinetic term contained in h_N ...
    #   (T is allocated with the interaction N_max, Params.Int.Nmax >= Params.Calc.Nmax ... use its leading a_max x a_max block) ...
    pV, nV = C.p * h_N.p * C.p', C.n * h_N.n * C.n'
    pT, nT = T.p[1:a_max,1:a_max], T.n[1:a_max,1:a_max]
    if CMS == "CMS1+2B"
        pV .-= pT .* (1.0 - 1.0 / A)
        nV .-= nT .* (1.0 - 1.0 / A)
    elseif CMS != "CMS2B"
        pV .-= pT
        nV .-= nT
    end

    # Radial grid ... subsampled rho_export grid (every 16th point, so the points coincide with rho_export) ...
    r1, r2 = 0.0 + 1e-8, 2.5 * 1.2 * A^(1/3)
    r_grid = collect(range(r1, stop = r2, length = 2^13 + 1))[1:16:end]
    N_r = length(r_grid)

    # Radial LHO wave functions R_a(r) on the grid ...
    nu_p, nu_n = 0.5 * m_p * hw / hc^2, 0.5 * m_n * hw / hc^2
    pR, nR = zeros(Float64,a_max,N_r), zeros(Float64,a_max,N_r)
    @inbounds Threads.@threads for i in 1:N_r
        @inbounds for a in 1:a_max
            pR[a,i] = Psi_rad_LHO(r_grid[i],Orb[a].n,Orb[a].l,nu_p)
            nR[a,i] = Psi_rad_LHO(r_grid[i],Orb[a].n,Orb[a].l,nu_n)
        end
    end

    # List of partial waves (l,j) present in the basis ... ordered by l, then j ...
    lj_list = sort(unique([(Orb[a].l, Orb[a].j) for a in 1:a_max]))
    N_lj = length(lj_list)

    # Partial-wave potentials V_lj(r,r') = R_lj^T V_lj R_lj & the total potential ...
    pV_lj, nV_lj = Vector{Matrix{Float64}}(undef,N_lj), Vector{Matrix{Float64}}(undef,N_lj)
    pV_tot, nV_tot = zeros(Float64,N_r,N_r), zeros(Float64,N_r,N_r)
    @inbounds for (i_lj, (l, j)) in enumerate(lj_list)
        Block = [a for a in 1:a_max if Orb[a].l == l && Orb[a].j == j]
        pV_lj[i_lj] = pR[Block,:]' * pV[Block,Block] * pR[Block,:]
        nV_lj[i_lj] = nR[Block,:]' * nV[Block,Block] * nR[Block,:]
        pV_tot .+= Float64(j + 1) / (4.0 * π) .* pV_lj[i_lj]
        nV_tot .+= Float64(j + 1) / (4.0 * π) .* nV_lj[i_lj]
    end

    # Spectroscopic labels of the partial waves, e.g. p3/2 ...
    l_label = "spdfghiklmnoqrtuvwxyz"
    lj_label = [string(l_label[l+1]) * string(j) * "/2" for (l, j) in lj_list]

    # Export ... columns r, r', pV_tot, nV_tot, then pV & nV of every partial wave ...
    Export_Path = "io/" * Params.Calc.Path * "/HF/densities/HF_radial_potential.dat"
    open(Export_Path, "w") do Export_File
        @printf(Export_File, "# Non-local HF potential V(r,r') = h_HF - T_1b (CMS = %s) ... r, r' in fm, V in MeV fm^-3\n", CMS)
        @printf(Export_File, "# V_lj(r,r') = sum_ab R_a(r) V_ab R_b(r') per partial wave, V_tot = sum_lj (2j+1)/(4 pi) V_lj\n")
        @printf(Export_File, "%-14s %-14s %-16s %-16s", "r", "r'", "pV_tot", "nV_tot")
        for Label in lj_label
            @printf(Export_File, " %-16s %-16s", "pV_" * Label, "nV_" * Label)
        end
        @printf(Export_File, "\n")

        @inbounds for i in 1:N_r
            @inbounds for k in 1:N_r
                @printf(Export_File, "%-14.8f %-14.8f %-16.8e %-16.8e", r_grid[i], r_grid[k], pV_tot[i,k], nV_tot[i,k])
                @inbounds for i_lj in 1:N_lj
                    @printf(Export_File, " %-16.8e %-16.8e", pV_lj[i_lj][i,k], nV_lj[i_lj][i,k])
                end
                @printf(Export_File, "\n")
            end
            # Blank line between blocks of constant r ... (gnuplot splot / pm3d friendly) ...
            @printf(Export_File, "\n")
        end
    end

    println("\tNon-local radial HF mean-field potentials exported to " * Export_Path * " ...")

    return
end
