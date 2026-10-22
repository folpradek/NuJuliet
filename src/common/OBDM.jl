# Note the Radial_Density() function assumes Rho expressed in the LHO basis, C transformation matrix from LHO to reference basis ...
    # Summary_File = String with direct access to the output summary file, e.g. Summary_File = "IO/Sample_Calculation/HF/HF_Summary.dat"
    # Densities_File = String with direct access to the output folder for radial densities, e.g. Densities_File = "IO/Sample_Calculation/HF/Densities/HF_Radial_Densities.dat"

function OBDM_export(Params::Parameters,Orb::Vector{Orb1B},Summary_File::String,Densities_File::String,Rho::O1B,C::O1B)
    # Transform Rho from LHO to the reference basis ...
    Rho = O1B(C.p' * Rho.p * C.p, C.n' * Rho.n * C.n)

    # Calculation of 1-body & 2-body CMS...
    if Params.Calc.CMS == "CMS1+2B" || Params.Calc.CMS  == "CMS2B"
        R_CMS = OBDM_radial_cms(Params,Rho,C,Orb)
    else
        R_CMS = [0.0, 0.0, 0.0, 0.0]
    end

    # Evaluate nucleon densities on a grid ...
    Rho_grid = OBDM_radial_density_grid(Params,Rho,C,Orb)

    # Evaluate charge density on a grid & calculate anomalous magnetic moment correction ...
    Rho_grid, kR2 = OBDM_radial_chdensity_grid(Params,Rho,C,Orb,R_CMS,Rho_grid[1],Rho_grid[2],Rho_grid[3])

    # Calculate radii from the radial densities ...
    pR2, nR2, chR2 = OBDM_radial_radii(Rho_grid[1],Rho_grid[2],Rho_grid[3],Rho_grid[4])

    # Export radial densities ...
    OBDM_radial_density_export(Densities_File,Rho_grid)

    # Export radii to summary file ...
    OBDM_radial_summary(Params,Summary_File,pR2,nR2,chR2,R_CMS,kR2)

    println("\nRadial densities and corresponding radii sucessfully exported ...")

    return
end

function OBDM_radial_cms(Params::Parameters,Rho::O1B,C::O1B,Orb::Vector{Orb1B})
    # Read parameters ...
    hw, A, Z = Params.Calc.hw, Float64(Params.Calc.A), Float64(Params.Calc.Z)
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1)*(N_max + 2),2)

    # Calculate 1st & 2nd radial moments in LHO basis ...
    Radial_r1_LHO = radial_moment_matrix_LHO(1,hw,Orb)
    Radial_r2_LHO = radial_moment_matrix_LHO(2,hw,Orb)

    # Initialize arrays for the radial moments ...
    pRadial_r1, pRadial_r2 = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)
    nRadial_r1, nRadial_r2 = zeros(Float64,a_max,a_max), zeros(Float64,a_max,a_max)

    # Transform Radial Moment matrices to the reference basis ...
    @inbounds Threads.@threads for a in 1:a_max
        l_a = Orb[a].l
        j_a = Orb[a].j
        @inbounds for b in 1:a_max
            l_b = Orb[b].l
            j_b = Orb[b].j
            pR1Sum, pR2Sum = 0.0, 0.0
            nR1Sum, nR2Sum = 0.0, 0.0
            @inbounds for k in 1:a_max
                l_k = Orb[k].l
                j_k = Orb[k].j
                if (l_k == l_a) && (j_k == j_a)
                    @inbounds for l in 1:a_max
                        l_l = Orb[l].l
                        j_l = Orb[l].j
                        if (l_l == l_b) && (j_l == j_b)
                            pME1 = C.p[k,a] * C.p[l,b] * Radial_r1_LHO[k,l]
                            pME2 = C.p[k,a] * C.p[l,b] * Radial_r2_LHO[k,l]
                            pR1Sum += pME1
                            pR2Sum += pME2

                            nME1 = C.n[k,a] * C.n[l,b] * Radial_r1_LHO[k,l]
                            nME2 = C.n[k,a] * C.n[l,b] * Radial_r2_LHO[k,l]
                            nR1Sum += nME1
                            nR2Sum += nME2
                        end
                    end
                end
            end
            pRadial_r1[a,b], pRadial_r2[a,b] = pR1Sum, pR2Sum
            nRadial_r1[a,b], nRadial_r2[a,b] = nR1Sum, nR2Sum
        end
    end

    # Initialize CM corrections to radii ...
    pR2_CMS1, pR2_CMS2 = 0.0, 0.0
    nR2_CMS1, nR2_CMS2 = 0.0, 0.0

    # Calculate 2-body CM corrections to proton & neutron radii ...
    @inbounds for a in 1:a_max
        l_a = Orb[a].l
        j_a = Orb[a].j
        @inbounds for b in 1:a_max
            l_b = Orb[b].l
            j_b = Orb[b].j
            Amp = fCG(j_a,j_b,2,1,-1,0) * fCG(j_b,j_a,2,1,-1,0)
            pME = Amp * ((Rho.p[a,a] * Rho.p[b,b]  * abs(pRadial_r1[a,b])^2) * (2.0 / (Z*A) - 1.0 / A^2) - 1.0 / A^2 * Rho.n[a,a] * Rho.n[b,b] * abs(nRadial_r1[a,b])^2)
            nME = Amp * ((Rho.n[a,a] * Rho.n[b,b]  * abs(nRadial_r1[a,b])^2) * (2.0 / (Z*A) - 1.0 / A^2) - 1.0 / A^2 * Rho.p[a,a] * Rho.p[b,b] * abs(pRadial_r1[a,b])^2)
            pR2_CMS2 += pME
            nR2_CMS2 += nME
        end
        pME = ((1.0 / A^2 - 2.0 / (Z*A)) * Rho.p[a,a] * pRadial_r2[a,a] + Rho.n[a,a] * nRadial_r2[a,a] / A^2) * Float64(j_a + 1)
        nME = ((1.0 / A^2 - 2.0 / ((A-Z)*A)) * Rho.n[a,a] * nRadial_r2[a,a] + Rho.p[a,a] * pRadial_r2[a,a] / A^2) * Float64(j_a + 1)
        pR2_CMS1 += pME
        nR2_CMS1 += nME
    end

    return [pR2_CMS1, pR2_CMS2, nR2_CMS1 ,nR2_CMS2]
end

function OBDM_radial_density_grid(Params::Parameters,Rho::O1B,C::O1B,Orb::Vector{Orb1B})
    # Read parameters ...
    hw, A = Params.Calc.hw, Float64(Params.Calc.A)
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1)*(N_max + 2),2)

    # Basic constants ...
    hc, m_p, m_n = 197.326980, 938.272013, 939.565346
    nu_proton = 0.5 * m_p * hw / hc^2
    nu_neutron = 0.5 * m_n * hw / hc^2

    # Preallocate grid ...
    r1, r2 = 0.0 + 1e-8, 2.5 * 1.2 * A^(1/3)
    N_Sampling = 2^13 + 1 # 8192 + 1 grid points ...
    r_grid, pRho_grid, nRho_grid = collect(range(r1, stop = r2, length = N_Sampling)), zeros(Float64,N_Sampling), zeros(Float64,N_Sampling)

    # Evaluate the radial grid ...
    @inbounds Threads.@threads for i in 1:N_Sampling
        r = r_grid[i]
        pSum, nSum = 0.0, 0.0
        @inbounds for a in 1:a_max
            l_a = Orb[a].l
            j_a = Orb[a].j
            pRad, nRad = 0.0, 0.0
            @inbounds for k in 1:a_max
                l_k = Orb[k].l
                j_k = Orb[k].j
                n_k = Orb[k].n
                if (j_a == j_k) && (l_a == l_k)
                    pPsi = Psi_rad_LHO(r,n_k,l_k,nu_proton) * C.p[k,a]
                    nPsi = Psi_rad_LHO(r,n_k,l_k,nu_neutron) * C.n[k,a]
                    pRad += pPsi
                    nRad += nPsi
                end
            end
            pSum += Rho.p[a,a] * pRad^2 * Float64(j_a + 1)
            nSum += Rho.n[a,a] * nRad^2 * Float64(j_a + 1)
        end
        pRho_grid[i] = pSum / (4.0 * π)
        nRho_grid[i] = nSum / (4.0 * π)
    end
    
    return [r_grid, pRho_grid, nRho_grid]
end

function OBDM_radial_chdensity_grid(Params::Parameters,Rho::O1B,C::O1B,Orb::Vector{Orb1B},R_CMS::Vector{Float64},r_grid::Vector{Float64},pRho_grid::Vector{Float64},nRho_grid::Vector{Float64})
    # Read parameters ...
    hw, A, Z = Params.Calc.hw, Float64(Params.Calc.A), Float64(Params.Calc.Z)
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1)*(N_max + 2),2)

    # Basic constants ...
    hc, m_p, m_n = 197.326980, 938.272013, 939.565346
    k_p, k_n = 1.793, 1.913
    nu_proton = 0.5 * m_p * hw / hc^2
    nu_neutron = 0.5 * m_n * hw / hc^2

    # CMS radius corrections ...
    pR_CMS = R_CMS[1] + R_CMS[2]
    nR_CMS = R_CMS[3] + R_CMS[4]

    # Preallocate charged densitiy grid ...
    N_Sampling = 2^13 + 1
    chRho_grid = zeros(Float64,N_Sampling)

    # Preallocate anomalous magnetic moment correction ...
    kR2 = 0.0
    
    # Evaluate the charged density ...
    @inbounds Threads.@threads for i in 1:N_Sampling
        r = r_grid[i]
        chSum = 0.0
        @inbounds for a in 1:a_max
            l_a = Orb[a].l
            j_a = Orb[a].j
            pRad, nRad = 0.0, 0.0
            @inbounds for k in 1:a_max
                l_k = Orb[k].l
                j_k = Orb[k].j
                n_k = Orb[k].n
                if (j_a == j_k) && (l_a == l_k)
                    pPsi = Psi_rad_LHO(r,n_k,l_k,nu_proton) * C.p[k,a]
                    nPsi = Psi_rad_LHO(r,n_k,l_k,nu_neutron) * C.n[k,a]
                    pRad += pPsi
                    nRad += nPsi
                end
            end 
            # Spin-Orbit interaction matrix element ...
            SO = Float64(l_a * kronecker_delta(j_a,2*l_a+1) - (l_a+1) * kronecker_delta(j_a,2*l_a-1))
            chME = (k_n / m_n^2 * Rho.n[a,a] * nRad^2 + k_p / m_p^2 * Rho.p[a,a] * pRad^2) *
                    hc^2 * SO * Float64(j_a + 1) / Z
            chSum += chME
        end
        chME = fold_pRho(r,r_grid,pRho_grid,pR_CMS)
        chSum += chME
        chME = fold_nRho(r,r_grid,nRho_grid,nR_CMS)
        chSum += chME
        chRho_grid[i] += chSum
    end

    # Separately we also evaluate the anomalous magnetic moment contribution to the charged radius ...
    @inbounds for a in 1:a_max
        l_a, j_a = Orb[a].l, Orb[a].j
        SO = Float64(l_a * kronecker_delta(j_a,2*l_a+1) - (l_a+1) * kronecker_delta(j_a,2*l_a-1))
        kME = hc^2 * (k_p / m_p^2 * Rho.p[a,a] + k_n / m_n^2 * Rho.n[a,a]) * Float64(j_a + 1)^2 * SO / (4.0 * π * Z)
        kR2 += kME
    end

    return [r_grid, pRho_grid, nRho_grid, chRho_grid], kR2
end

function OBDM_radial_radii(r_grid::Vector{Float64},pRho_grid::Vector{Float64},nRho_grid::Vector{Float64},chRho_grid::Vector{Float64})
    pR2 = integrate_trap(r_grid, r_grid.^4 .* pRho_grid) / integrate_trap(r_grid, r_grid.^2 .* pRho_grid)
    nR2 = integrate_trap(r_grid, r_grid.^4 .* nRho_grid) / integrate_trap(r_grid, r_grid.^2 .* nRho_grid)
    chR2 = integrate_trap(r_grid, r_grid.^4 .* chRho_grid) / integrate_trap(r_grid, r_grid.^2 .* chRho_grid)
    return pR2, nR2, chR2
end

function OBDM_radial_density_export(Densities_File::String,Rho_grid::Vector{Vector{Float64}})
    # Export radial grid & densities to file ...
    r_grid, pRho, nRho, chRho = Rho_grid
    N = minimum((length(r_grid), length(pRho), length(nRho), length(chRho)))
    open(Densities_File, "w") do Export_File
        @printf(Export_File, "%-14s %-14s %-14s %-14s\n", "r", "pRho", "nRho", "chRho")
        @inbounds for i in 1:N
            @printf(Export_File, "%-14.8f %-14.8f %-14.8f %-14.8f\n",r_grid[i], pRho[i], nRho[i], chRho[i])
        end
    end

    return
end

function OBDM_radial_summary(Params::Parameters,Summary_File::String,pR2::Float64,nR2::Float64,chR2::Float64,R_CMS::Vector{Float64},kR2::Float64)
    # Read parameters ...
    A, Z, N = Float64(Params.Calc.A), Float64(Params.Calc.Z), Float64(Params.Calc.A - Params.Calc.Z)

    # Print basis information regarding the charge radii ...
    println("\nResulting point-proton, charged & point-neutron radii ...")
    println("\tr_p  = " * string(round(sqrt(pR2 + R_CMS[1] + R_CMS[2] + 0.5 * (197.326980 / 938.272013)^2), sigdigits=6)) * " fm \t\t ... \t Point-proton radius")
    println("\tr_n  = " * string(round(sqrt(nR2 + R_CMS[3] + R_CMS[4] + 0.5 * (197.326980 / 939.565346)^2), sigdigits=6)) * " fm \t\t ... \t Point-neutron radius")
    println("\tr_ch = " * string(round(sqrt(pR2 + R_CMS[1] + R_CMS[2] - 0.106 * N / Z + 0.8414 - 0.143), sigdigits=6)) * " fm \t\t ... \t Standard charge radius")
    println("\tr_ch = " * string(round(sqrt(chR2 + R_CMS[1] + R_CMS[2] + 0.5 * (197.326980 / 938.272013)^2), sigdigits=6)) * " fm \t\t ... \t Folded charge radius")
    println("For more details see the summary file ... ''" * Summary_File * "'' ...")

    # Export resulsts & contributions to radii ...
    println("\nExporting information on radii ...")
    Summary =  open(Summary_File, "a")
        println(Summary, "\n\nResulting nuclear radii ...")
        @printf(Summary, "\nr_p   = %12.6f fm \t\t ... \t Point-proton radius\n", sqrt(abs(pR2 + R_CMS[1] + R_CMS[2] + 0.5 * (197.326980 / 938.272013)^2)))
        @printf(Summary, "r_n   = %12.6f fm \t\t ... \t Point-neutron radius\n", sqrt(abs(nR2 + R_CMS[3] + R_CMS[4] + 0.5 * (197.326980 / 939.565346)^2)))
        @printf(Summary, "r_ch  = %12.6f fm \t\t ... \t Standard charge radius\n", sqrt(abs(pR2 + R_CMS[1] + R_CMS[2] - 0.106 * N / Z + 0.8414 - 0.143)))
        @printf(Summary, "r_ch  = %12.6f fm \t\t ... \t Folded charge radius\n", sqrt(abs(chR2 + R_CMS[1] + R_CMS[2] + 0.5 * (197.326980 / 938.272013)^2)))
        @printf(Summary, "\nr_p  = %12.6f fm \t\t ... \t Uncorrected point-proton radius\n", sqrt(abs(pR2)))
        @printf(Summary, "r_ch = %12.6f fm \t\t ... \t Uncorrected folded charge radius\n", sqrt(abs(chR2)))
        @printf(Summary, "r_n  = %12.6f fm \t\t ... \t Uncorrected point-neutron radius\n", sqrt(abs(nR2)))
        println(Summary, "\nDetails on radii corrections ...\n")
        @printf(Summary, "r_p2_cm1^2 = %12.6f fm^2\n", R_CMS[1])
        @printf(Summary, "r_p2_cm2^2 = %12.6f fm^2\n", R_CMS[2])
        @printf(Summary, "r_n2_cm1^2 = %12.6f fm^2\n", R_CMS[3])
        @printf(Summary, "r_n2_cm2^2 = %12.6f fm^2\n", R_CMS[4])
        @printf(Summary, "r_ch2_k^2  = %12.6f fm^2\n", kR2)
        @printf(Summary, "r_pDF2^2   = %12.6f fm^2\n", 0.5 * (197.326980 / 938.272013)^2)
        @printf(Summary, "r_nDF2^2   = %12.6f fm^2\n", 0.5 * (197.326980 / 939.565346)^2)
        println(Summary, "\n\nResulting nuclear radii normalized by the scale factor 1/A^1/3 ...")
        @printf(Summary, "\nr_p   = %12.6f fm \t\t ... \t Point-proton radius\n", sqrt(abs(pR2 + R_CMS[1] + R_CMS[2] + 0.5 * (197.326980 / 938.272013)^2) / A^(1/3)))
        @printf(Summary, "r_n   = %12.6f fm \t\t ... \t Point-neutron radius\n", sqrt(abs(nR2 + R_CMS[3] + R_CMS[4] + 0.5 * (197.326980 / 939.565346)^2) / A^(1/3)))
        @printf(Summary, "r_ch  = %12.6f fm \t\t ... \t Standard charge radius\n", sqrt(abs(pR2 + R_CMS[1] + R_CMS[2] - 0.106 * N / Z + 0.8414 - 0.143) / A^(1/3)))
        @printf(Summary, "r_ch  = %12.6f fm \t\t ... \t Folded charge radius\n", sqrt(abs(chR2 + R_CMS[1] + R_CMS[2] + 0.5 * (197.326980 / 938.272013)^2) / A^(1/3)))
        @printf(Summary, "\nr_p  = %12.6f fm \t\t ... \t Uncorrected point-proton radius\n", sqrt(abs(pR2) / A^(1/3)))
        @printf(Summary, "r_ch = %12.6f fm \t\t ... \t Uncorrected folded charge radius\n", sqrt(abs(chR2) / A^(1/3)))
        @printf(Summary, "r_n  = %12.6f fm \t\t ... \t Uncorrected point-neutron radius\n", sqrt(abs(nR2) / A^(1/3)))
        println(Summary, "\nDetails on radii corrections ...\n")
        @printf(Summary, "r_p2_cm1^2 = %12.6f fm^2\n", R_CMS[1])
        @printf(Summary, "r_p2_cm2^2 = %12.6f fm^2\n", R_CMS[2])
        @printf(Summary, "r_n2_cm1^2 = %12.6f fm^2\n", R_CMS[3])
        @printf(Summary, "r_n2_cm2^2 = %12.6f fm^2\n", R_CMS[4])
        @printf(Summary, "r_ch2_k^2  = %12.6f fm^2\n", kR2)
        @printf(Summary, "r_pDF2^2   = %12.6f fm^2\n", 0.5 * (197.326980 / 938.272013)^2)
        @printf(Summary, "r_nDF2^2   = %12.6f fm^2\n", 0.5 * (197.326980 / 939.565346)^2)
    close(Summary)

    return
end

# Auxiliary functions for contributions of finite proton & neutron sizes to charge densities ...
@inline function fold_pRho(r::Float64,r_grid::Vector{Float64},pRho_grid::Vector{Float64},pR_CMS::Float64)
    N_Sampling = 2^13 + 1
    Rho = Vector{Float64}(undef,N_Sampling)
    @inbounds for i in 1:N_Sampling
        Rho[i] = convolution_p(r_grid[i],r,pR_CMS) * pRho_grid[i]
    end
    chRho = integrate_trap(r_grid,Rho)
    return chRho
end

@inline function fold_nRho(r::Float64,r_grid::Vector{Float64},nRho_grid::Vector{Float64},nR_CMS::Float64)
    N_Sampling = 2^13 + 1
    Rho = Vector{Float64}(undef,N_Sampling)
    @inbounds for i in 1:N_Sampling
        Rho[i] = convolution_n(r_grid[i],r,nR_CMS) * nRho_grid[i]
    end
    chRho = integrate_trap(r_grid,Rho)
    return chRho
end

@inline function convolution_p(x::Float64,r::Float64,pR_CMS::Float64)
    a_1, a_2, a_3 = 0.506373, 0.327922, 0.165705
    r_1 = sqrt(abs(0.431566 + pR_CMS + 0.5 * (197.326980 / 938.272013)^2))
    r_2 = sqrt(abs(0.139140 + pR_CMS + 0.5 * (197.326980 / 938.272013)^2))
    r_3 = sqrt(abs(1.525540 + pR_CMS + 0.5 * (197.326980 / 938.272013)^2))
    rho = x / (r * sqrt(pi)) * (a_1 / r_1 * (exp(-((r-x)/r_1)^2) - exp(-((r+x)/r_1)^2)) +
                                a_2 / r_2 * (exp(-((r-x)/r_2)^2) - exp(-((r+x)/r_2)^2)) +
                                a_3 / r_3 * (exp(-((r-x)/r_3)^2) - exp(-((r+x)/r_3)^2)))
    return rho
end

@inline function convolution_n(x::Float64,r::Float64,nR_CMS::Float64)
    r_p = sqrt(abs(0.4828 - 0.038664 + nR_CMS + 0.5 * (197.326980 / 939.565346)^2))
    r_m = sqrt(abs(0.4828 + 0.038664 + nR_CMS + 0.5 * (197.326980 / 939.565346)^2))
    rho = x / (r * sqrt(pi)) * ((exp(-((r-x)/r_p)^2) - exp(-((r+x)/r_p)^2)) / r_p -
                                (exp(-((r-x)/r_m)^2) - exp(-((r+x)/r_m)^2)) / r_m)
    return rho
end

function OBDM_rN(Params::Parameters,N::Int64,Rho::O1B,C::O1B,Orb::Vector{Orb1B})
    # Read parameters ...
    hw, A = Params.Calc.hw, Float64(Params.Calc.A)
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1)*(N_max + 2),2)

    # Basic constants ...
    hc, m_p = 197.326980, 938.272013
    nu_proton = 0.5 * m_p * hw / hc^2

    # Preallocate grid ...
    r1, r2 = 0.0 + 1e-8, 2.5 * 1.2 * A^(1/3)
    N_Sampling = 2^13 + 1 # 8192 + 1 grid points ...
    r_grid, pRho_grid = collect(range(r1, stop = r2, length = N_Sampling)), zeros(Float64,N_Sampling)

    # Evaluate the radial grid ...
    @inbounds Threads.@threads for i in 1:N_Sampling
        r = r_grid[i]
        pSum = 0.0
        @inbounds for a in 1:a_max
            l_a, j_a = Orb[a].l, Orb[a].j
            pRad = 0.0
            @inbounds for k in 1:a_max
                l_k, j_k, n_k = Orb[k].l, Orb[k].j, Orb[k].n
                if (j_a == j_k) && (l_a == l_k)
                    pPsi = Psi_rad_LHO(r,n_k,l_k,nu_proton) * C.p[k,a]
                    pRad += pPsi
                end
            end
            pSum += Float64(j_a + 1) * Rho.p[a,a] * pRad^2
        end
        pRho_grid[i] = pSum / (4.0 * π)
    end

    rN = integrate_trap(r_grid, r_grid.^(N+2) .* pRho_grid) / integrate_trap(r_grid, r_grid.^2 .* pRho_grid)

    return rN
end

function OBDM_chR2(Params::Parameters,Orb::Vector{Orb1B},C::O1B,Rho::O1B)
    # Read parameters ...
    Z, N = Float64(Params.Calc.Z), Float64(Params.Calc.A - Params.Calc.Z)

    # Calculation of 1-body & 2-body CMS...
    if Params.Calc.CMS == "CMS1+2B" || Params.Calc.CMS  == "CMS2B"
        R_CMS = OBDM_radial_cms(Params,Rho,C,Orb)
    else
        R_CMS = [0.0, 0.0, 0.0, 0.0]
    end

    println("\nCalculating the charge-radius squared chR2 ...")

    # Evaluate nucleon densities on a grid ...
    Rho_grid = OBDM_radial_density_grid(Params,Rho,C,Orb)

        #Evaluate charge density on a grid & calculate anomalous magnetic moment correction ...
        #Rho_grid, kR2 = OBDM_radial_chdensity_grid(Params,Rho,C,Orb,R_CMS,Rho_grid[1],Rho_grid[2],Rho_grid[3])

    # Calculate radii from the radial densities ...
    pR2, nR2, chR2 = OBDM_radial_radii(Rho_grid[1],Rho_grid[2],Rho_grid[3],Rho_grid[2])

    # Calculate the chR2 charge-radius squared
    chR2 = pR2 + R_CMS[1] + R_CMS[2] - 0.106 * N / Z + 0.8414 - 0.143

    println("\tThe charge-radius squared chR2 computed ...")

    return chR2
end