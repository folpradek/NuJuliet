function S0_Export(Params::Parameters,N_nu::Matrix{Int64},E_TDA::Matrix{Vector{Float64}},E_RPA::Matrix{Vector{ComplexF64}},rB_TDA::ReducedTransition,rB_RPA::ReducedTransition)
    # Read parameters ...
    Orthogon = Params.Calc.Ortho
    E_min = Params[6][1]
    E_max = Params[6][2]
    Delta = Params[6][3]
    Output_File = Params[8]

    println("\nPreparing export of RPA & TDA S0 electric strength functions ...")

    # Prepare grid ...
    N_grid = 25000
    E_grid = range(E_min, stop = E_max, length = N_grid)

    E0_TDA_grid = Vector{Float64}(undef,N_grid)
    E1_TDA_grid = Vector{Float64}(undef,N_grid)
    E2_TDA_grid = Vector{Float64}(undef,N_grid)
    E3_TDA_grid = Vector{Float64}(undef,N_grid)
    E0is_TDA_grid = Vector{Float64}(undef,N_grid)
    E1is_TDA_grid = Vector{Float64}(undef,N_grid)
    E2is_TDA_grid = Vector{Float64}(undef,N_grid)
    E3is_TDA_grid = Vector{Float64}(undef,N_grid)
    E0iv_TDA_grid = Vector{Float64}(undef,N_grid)
    E1iv_TDA_grid = Vector{Float64}(undef,N_grid)
    E2iv_TDA_grid = Vector{Float64}(undef,N_grid)
    E3iv_TDA_grid = Vector{Float64}(undef,N_grid)

    E0_RPA_grid = Vector{Float64}(undef,N_grid)
    E1_RPA_grid = Vector{Float64}(undef,N_grid)
    E2_RPA_grid = Vector{Float64}(undef,N_grid)
    E3_RPA_grid = Vector{Float64}(undef,N_grid)
    E0is_RPA_grid = Vector{Float64}(undef,N_grid)
    E1is_RPA_grid = Vector{Float64}(undef,N_grid)
    E2is_RPA_grid = Vector{Float64}(undef,N_grid)
    E3is_RPA_grid = Vector{Float64}(undef,N_grid)
    E0iv_RPA_grid = Vector{Float64}(undef,N_grid)
    E1iv_RPA_grid = Vector{Float64}(undef,N_grid)
    E2iv_RPA_grid = Vector{Float64}(undef,N_grid)
    E3iv_RPA_grid = Vector{Float64}(undef,N_grid)

    @inbounds for (i, E) in enumerate(E_grid)
        physE0Sum_TDA = 0.0
        physE1Sum_TDA = 0.0
        physE2Sum_TDA = 0.0
        physE3Sum_TDA = 0.0
        isE0Sum_TDA = 0.0
        isE1Sum_TDA = 0.0
        isE2Sum_TDA = 0.0
        isE3Sum_TDA = 0.0
        ivE0Sum_TDA = 0.0
        ivE1Sum_TDA = 0.0
        ivE2Sum_TDA = 0.0
        ivE3Sum_TDA = 0.0

        physE0Sum_RPA = 0.0
        physE1Sum_RPA = 0.0
        physE2Sum_RPA = 0.0
        physE3Sum_RPA = 0.0
        isE0Sum_RPA = 0.0
        isE1Sum_RPA = 0.0
        isE2Sum_RPA = 0.0
        isE3Sum_RPA = 0.0
        ivE0Sum_RPA = 0.0
        ivE1Sum_RPA = 0.0
        ivE2Sum_RPA = 0.0
        ivE3Sum_RPA = 0.0

        # E0
        J = 0
        P = 1
        N_ph = N_nu[J+1,P]
        @inbounds for nu in 1:N_ph
            E_nu = E_TDA[J+1,P][nu]

            physME = rB_TDA.E0.ph[nu] * lorentzian(E_nu-E,Delta) / Float64(2*J + 1)
            isME = rB_TDA.E0.is[nu] * lorentzian(E_nu-E,Delta) / Float64(2*J + 1)
            ivME = rB_TDA.E0.iv[nu] * lorentzian(E_nu-E,Delta) / Float64(2*J + 1)

            physE0Sum_TDA += physME
            isE0Sum_TDA += isME
            ivE0Sum_TDA += ivME

            E_nu = real(E_RPA[J+1,P][nu])

            physME = rB_RPA.E0.ph[nu] * lorentzian(E_nu-E,Delta) / Float64(2*J + 1)
            isME = rB_RPA.E0.is[nu] * lorentzian(E_nu-E,Delta) / Float64(2*J + 1)
            ivME = rB_RPA.E0.iv[nu] * lorentzian(E_nu-E,Delta) / Float64(2*J + 1)

            physE0Sum_RPA+= physME
            isE0Sum_RPA += isME
            ivE0Sum_RPA += ivME
        end

        # E1
        J = 1
        P = 2
        N_ph = N_nu[J+1,P]
        @inbounds for nu in 1:N_ph
            E_nu = E_TDA[J+1,P][nu]

            physME = rB_TDA.E1.ph[nu] * lorentzian(E_nu-E,Delta) / Float64(2*J + 1)
            isME = rB_TDA.E1.is[nu] * lorentzian(E_nu-E,Delta) / Float64(2*J + 1)
            ivME = rB_TDA.E1.iv[nu] * lorentzian(E_nu-E,Delta) / Float64(2*J + 1)

            physE1Sum_TDA += physME
            isE1Sum_TDA += isME
            ivE1Sum_TDA += ivME

            E_nu = real(E_RPA[J+1,P][nu])

            physME = rB_RPA.E1.ph[nu] * lorentzian(E_nu-E,Delta) / Float64(2*J + 1)
            isME = rB_RPA.E1.is[nu] * lorentzian(E_nu-E,Delta) / Float64(2*J + 1)
            ivME = rB_RPA.E1.iv[nu] * lorentzian(E_nu-E,Delta) / Float64(2*J + 1)

            physE1Sum_RPA+= physME
            isE1Sum_RPA += isME
            ivE1Sum_RPA += ivME
        end

        # E2
        J = 2
        P = 1
        N_ph = N_nu[J+1,P]
        @inbounds for nu in 1:N_ph
            E_nu = E_TDA[J+1,P][nu]

            physME = rB_TDA.E2.ph[nu] * lorentzian(E_nu-E,Delta) / Float64(2*J + 1)
            isME = rB_TDA.E2.is[nu] * lorentzian(E_nu-E,Delta) / Float64(2*J + 1)
            ivME = rB_TDA.E2.iv[nu] * lorentzian(E_nu-E,Delta) / Float64(2*J + 1)

            physE2Sum_TDA += physME
            isE2Sum_TDA += isME
            ivE2Sum_TDA += ivME

            E_nu = real(E_RPA[J+1,P][nu])

            physME = rB_RPA.E2.ph[nu] * lorentzian(E_nu-E,Delta) / Float64(2*J + 1)
            isME = rB_RPA.E2.is[nu] * lorentzian(E_nu-E,Delta) / Float64(2*J + 1)
            ivME = rB_RPA.E2.iv[nu] * lorentzian(E_nu-E,Delta) / Float64(2*J + 1)

            physE2Sum_RPA+= physME
            isE2Sum_RPA += isME
            ivE2Sum_RPA += ivME
        end

        # E3
        J = 3
        P = 2
        N_ph = N_nu[J+1,P]
        @inbounds for nu in 1:N_ph
            E_nu = E_TDA[J+1,P][nu]

            physME = rB_TDA.E3.ph[nu] * lorentzian(E_nu-E,Delta) / Float64(2*J + 1)
            isME = rB_TDA.E3.is[nu] * lorentzian(E_nu-E,Delta) / Float64(2*J + 1)
            ivME = rB_TDA.E3.iv[nu] * lorentzian(E_nu-E,Delta) / Float64(2*J + 1)

            physE3Sum_TDA += physME
            isE3Sum_TDA += isME
            ivE3Sum_TDA += ivME

            E_nu = real(E_RPA[J+1,P][nu])

            physME = rB_RPA.E3.ph[nu] * lorentzian(E_nu-E,Delta) / Float64(2*J + 1)
            isME = rB_RPA.E3.is[nu] * lorentzian(E_nu-E,Delta) / Float64(2*J + 1)
            ivME = rB_RPA.E3.iv[nu] * lorentzian(E_nu-E,Delta) / Float64(2*J + 1)

            physE3Sum_RPA+= physME
            isE3Sum_RPA += isME
            ivE3Sum_RPA += ivME
        end

        E0_TDA_grid[i] = physE0Sum_TDA
        E1_TDA_grid[i] = physE1Sum_TDA
        E2_TDA_grid[i] = physE2Sum_TDA
        E3_TDA_grid[i] = physE3Sum_TDA
        E0is_TDA_grid[i] = isE0Sum_TDA
        E1is_TDA_grid[i] = isE1Sum_TDA
        E2is_TDA_grid[i] = isE2Sum_TDA
        E3is_TDA_grid[i] = isE3Sum_TDA
        E0iv_TDA_grid[i] = ivE0Sum_TDA
        E1iv_TDA_grid[i] = ivE1Sum_TDA
        E2iv_TDA_grid[i] = ivE2Sum_TDA
        E3iv_TDA_grid[i] = ivE3Sum_TDA

        E0_RPA_grid[i] = physE0Sum_RPA
        E1_RPA_grid[i] = physE1Sum_RPA
        E2_RPA_grid[i] = physE2Sum_RPA
        E3_RPA_grid[i] = physE3Sum_RPA
        E0is_RPA_grid[i] = isE0Sum_RPA
        E1is_RPA_grid[i] = isE1Sum_RPA
        E2is_RPA_grid[i] = isE2Sum_RPA
        E3is_RPA_grid[i] = isE3Sum_RPA
        E0iv_RPA_grid[i] = ivE0Sum_RPA
        E1iv_RPA_grid[i] = ivE1Sum_RPA
        E2iv_RPA_grid[i] = ivE2Sum_RPA
        E3iv_RPA_grid[i] = ivE3Sum_RPA

    end

    println("\nExporting RPA & TDA E0, E1, E2 & E3 stregth function S0 ...")

    open(Output_File * "/RPA/Transitions/E0/TDA_E0_S0.dat", "w") do Write_File
        @inbounds for (i, E) in enumerate(E_grid)
            Row = string(round(E,digits=6)) * "\t" * string(round(E0_TDA_grid[i],digits=6)) * "\t" *
                    string(round(E0is_TDA_grid[i],digits=6)) * "\t" * string(round(E0iv_TDA_grid[i],digits=6))
            println(Write_File, Row)
        end
    end

    open(Output_File * "/RPA/Transitions/E0/RPA_E0_S0.dat", "w") do Write_File
        @inbounds for (i, E) in enumerate(E_grid)
            Row = string(round(E,digits=6)) * "\t" * string(round(E0_RPA_grid[i],digits=6)) * "\t" *
                    string(round(E0is_RPA_grid[i],digits=6)) * "\t" * string(round(E0iv_RPA_grid[i],digits=6))
            println(Write_File, Row)
        end
    end

    if Orthogon == true
        open(Output_File * "/RPA/Transitions/E1/TDA_E1_S0_Ortho.dat", "w") do Write_File
            @inbounds for (i, E) in enumerate(E_grid)
                Row = string(round(E,digits=6)) * "\t" * string(round(E1_TDA_grid[i],digits=6)) * "\t" *
                        string(round(E1is_TDA_grid[i],digits=6)) * "\t" * string(round(E1iv_TDA_grid[i],digits=6))
                println(Write_File, Row)
            end
        end

        open(Output_File * "/RPA/Transitions/E1/RPA_E1_S0_Ortho.dat", "w") do Write_File
            @inbounds for (i, E) in enumerate(E_grid)
                Row = string(round(E,digits=6)) * "\t" * string(round(E1_RPA_grid[i],digits=6)) * "\t" *
                        string(round(E1is_RPA_grid[i],digits=6)) * "\t" * string(round(E1iv_RPA_grid[i],digits=6))
                println(Write_File, Row)
            end
        end
    else
        open(Output_File * "/RPA/Transitions/E1/TDA_E1_S0_Spur.dat", "w") do Write_File
            @inbounds for (i, E) in enumerate(E_grid)
                Row = string(round(E,digits=6)) * "\t" * string(round(E1_TDA_grid[i],digits=6)) * "\t" *
                        string(round(E1is_TDA_grid[i],digits=6)) * "\t" * string(round(E1iv_TDA_grid[i],digits=6))
                println(Write_File, Row)
            end
        end

        open(Output_File * "/RPA/Transitions/E1/RPA_E1_S0_Spur.dat", "w") do Write_File
            @inbounds for (i, E) in enumerate(E_grid)
                Row = string(round(E,digits=6)) * "\t" * string(round(E1_RPA_grid[i],digits=6)) * "\t" *
                        string(round(E1is_RPA_grid[i],digits=6)) * "\t" * string(round(E1iv_RPA_grid[i],digits=6))
                println(Write_File, Row)
            end
        end
    end

    open(Output_File * "/RPA/Transitions/E2/TDA_E2_S0.dat", "w") do Write_File
        @inbounds for (i, E) in enumerate(E_grid)
            Row = string(round(E,digits=6)) * "\t" * string(round(E2_TDA_grid[i],digits=6)) * "\t" *
                    string(round(E2is_TDA_grid[i],digits=6)) * "\t" * string(round(E2iv_TDA_grid[i],digits=6))
            println(Write_File, Row)
        end
    end

    open(Output_File * "/RPA/Transitions/E2/RPA_E2_S0.dat", "w") do Write_File
        @inbounds for (i, E) in enumerate(E_grid)
            Row = string(round(E,digits=6)) * "\t" * string(round(E2_RPA_grid[i],digits=6)) * "\t" *
                    string(round(E2is_RPA_grid[i],digits=6)) * "\t" * string(round(E2iv_RPA_grid[i],digits=6))
            println(Write_File, Row)
        end
    end

    open(Output_File * "/RPA/Transitions/E3/TDA_E3_S0.dat", "w") do Write_File
        @inbounds for (i, E) in enumerate(E_grid)
            Row = string(round(E,digits=6)) * "\t" * string(round(E3_TDA_grid[i],digits=6)) * "\t" *
                    string(round(E3is_TDA_grid[i],digits=6)) * "\t" * string(round(E3iv_TDA_grid[i],digits=6))
            println(Write_File, Row)
        end
    end

    open(Output_File * "/RPA/Transitions/E3/RPA_E3_S0.dat", "w") do Write_File
        @inbounds for (i, E) in enumerate(E_grid)
            Row = string(round(E,digits=6)) * "\t" * string(round(E3_RPA_grid[i],digits=6)) * "\t" *
                    string(round(E3is_RPA_grid[i],digits=6)) * "\t" * string(round(E3iv_RPA_grid[i],digits=6))
            println(Write_File, Row)
        end
    end

    println("\nRPA & TDA S0 strength functions of E0, E1, E2 & E3 transitions exported ...")

    return
end

function Sigma_Export(Params::Vector{Any},N_nu::Matrix{Int64},E_TDA::Matrix{Vector{Float64}},E_RPA::Matrix{Vector{ComplexF64}},rB_TDA::ReducedTransition,rB_RPA::ReducedTransition)
    # Read calculation parameters ...
    Orthogon = Params[5]
    E_min = Params[7][1]
    E_max = Params[7][2]
    Delta = Params[7][3]
    Output_File = Params[8]

    # Physical constants ...
    alpha = 1.0 / 137.035999177
    HbarC = 197.326980

    println("\nPreparing export of RPA & TDA photoabsorbtion cross-sections ...")
    
    # Preallocate grid ...
    N_grid = 25000
    E_grid = range(E_min, stop = E_max, length = N_grid)

    SigmaE1TDA_grid = Vector{Float64}(undef,N_grid)
    SigmaE1RPA_grid = Vector{Float64}(undef,N_grid)
    SigmaE2TDA_grid = Vector{Float64}(undef,N_grid)
    SigmaE2RPA_grid = Vector{Float64}(undef,N_grid)
    SigmaE3TDA_grid = Vector{Float64}(undef,N_grid)
    SigmaE3RPA_grid = Vector{Float64}(undef,N_grid)


    @inbounds for (i, E) in enumerate(E_grid)
        SigmaE1TDASum = 0.0
        SigmaE1RPASum = 0.0
        SigmaE2TDASum = 0.0
        SigmaE2RPASum = 0.0
        SigmaE3TDASum = 0.0
        SigmaE3RPASum = 0.0

        # E1
        # Based on formula by Kvasil, eq. (7.388) p. 484 in mega lecture notes ... https://ipnp.cz/~kvasil/valya/NS+NR.pdf
        J = 1
        P = 2
        N_ph = N_nu[J+1,P]
        @inbounds for nu in 1:N_ph
            E_nu = E_TDA[J+1,P][nu]
            Amp = 16.0 * pi^3 * alpha / 9.0 * E_nu * lorentzian(E_nu-E,Delta)
            ME = rB_TDA.E1.ph[nu] * Amp
            SigmaE1TDASum += ME
        end

        if Orthogon == true
            @inbounds for nu in 1:N_ph
                E_nu = real(E_RPA[J+1,P][nu])
                Amp = 16.0 * pi^3 * alpha / 9.0 * E_nu * lorentzian(E_nu-E,Delta)
                ME = rB_RPA.E1.ph[nu] * Amp
                SigmaE1RPASum += ME
            end
        else
            @inbounds for nu in 1:N_ph
                E_nu = real(E_RPA[J+1,P][nu])
                Amp = 16.0 * pi^3 * alpha / 9.0 * E_nu * lorentzian(E_nu-E,Delta)
                ME = rB_RPA.E1.iv[nu] * Amp
                SigmaE1RPASum += ME
            end
        end

        # E2
        J = 2
        P = 1
        N_ph = N_nu[J+1,P]
        @inbounds for nu in 1:N_ph
            E_nu = E_TDA[J+1,P][nu]
            Amp = 4.0 * pi^3 * alpha / (75.0 * HbarC^2) * E_nu^3 * lorentzian(E_nu-E,Delta)
            ME = rB_TDA.E2.ph[nu] * Amp
            SigmaE2TDASum += ME
        end

        @inbounds for nu in 1:N_ph
            E_nu = real(E_RPA[J+1,P][nu])
            Amp = 4.0 * pi^3 * alpha / (75.0 * HbarC^2) * E_nu^3 * lorentzian(E_nu-E,Delta)
            ME = rB_RPA.E2.ph[nu] * Amp
            SigmaE2RPASum += ME
        end

        # E3
        J = 3
        P = 2
        N_ph = N_nu[J+1,P]
        @inbounds for nu in 1:N_ph
            E_nu = E_TDA[J+1,P][nu]
            Amp = 16.0 * pi^3 * alpha / (11025.0 * HbarC^4) * E_nu^5 * lorentzian(E_nu-E,Delta)
            ME = rB_TDA.E3.ph[nu] * Amp
            SigmaE3TDASum += ME
        end

        @inbounds for nu in 1:N_ph
            E_nu = real(E_RPA[J+1,P][nu])
            Amp = 16.0 * pi^3 * alpha / (11025.0 * HbarC^4) * E_nu^5 * lorentzian(E_nu-E,Delta)
            ME = rB_RPA.E3.ph[nu] * Amp
            SigmaE3RPASum += ME
        end

        # In milibarn units ... 1 b = 10^2 fm^2 ...
        SigmaE1TDA_grid[i] = SigmaE1TDASum * 10.0
        SigmaE1RPA_grid[i] = SigmaE1RPASum * 10.0
        SigmaE2TDA_grid[i] = SigmaE2TDASum * 10.0
        SigmaE2RPA_grid[i] = SigmaE2RPASum * 10.0
        SigmaE3TDA_grid[i] = SigmaE3TDASum * 10.0
        SigmaE3RPA_grid[i] = SigmaE3RPASum * 10.0

    end

    @views E_grid = round.(E_grid, digits=6)

    if Orthogon == true
        open(Output_File * "/RPA/Transitions/E1/TDA_Sigma_E1_Ortho.dat", "w") do Write_File
            @inbounds for (i, E) in enumerate(E_grid)
                Row = string(E) * "\t" * string(round(SigmaE1TDA_grid[i],digits=6))
                println(Write_File, Row)
            end
        end

        open(Output_File * "/RPA/Transitions/E1/RPA_Sigma_E1_Ortho.dat", "w") do Write_File
            @inbounds for (i, E) in enumerate(E_grid)
                Row = string(E) * "\t" * string(round(SigmaE1RPA_grid[i],digits=6))
                println(Write_File, Row)
            end
        end
    else
        open(Output_File * "/RPA/Transitions/E1/TDA_Sigma_E1_Spur.dat", "w") do Write_File
            @inbounds for (i, E) in enumerate(E_grid)
                Row = string(E) * "\t" * string(round(SigmaE1TDA_grid[i],digits=6))
                println(Write_File, Row)
            end
        end

        open(Output_File * "/RPA/Transitions/E1/RPA_Sigma_E1_Spur.dat", "w") do Write_File
            @inbounds for (i, E) in enumerate(E_grid)
                Row = string(E) * "\t" * string(round(SigmaE1RPA_grid[i],digits=6))
                println(Write_File, Row)
            end
        end
    end

    open(Output_File * "/RPA/Transitions/E2/TDA_Sigma_E2.dat", "w") do Write_File
        @inbounds for (i, E) in enumerate(E_grid)
            Row = string(E) * "\t" * string(round(SigmaE2TDA_grid[i],digits=6))
            println(Write_File, Row)
        end
    end

    open(Output_File * "/RPA/Transitions/E2/RPA_Sigma_E2.dat", "w") do Write_File
        @inbounds for (i, E) in enumerate(E_grid)
            Row = string(E) * "\t" * string(round(SigmaE2RPA_grid[i],digits=6))
            println(Write_File, Row)
        end
    end

    open(Output_File * "/RPA/Transitions/E3/TDA_Sigma_E3.dat", "w") do Write_File
        @inbounds for (i, E) in enumerate(E_grid)
            Row = string(E) * "\t" * string(round(SigmaE3TDA_grid[i],digits=6))
            println(Write_File, Row)
        end
    end

    open(Output_File * "/RPA/Transitions/E3/RPA_Sigma_E3.dat", "w") do Write_File
        @inbounds for (i, E) in enumerate(E_grid)
            Row = string(E) * "\t" * string(round(SigmaE3RPA_grid[i],digits=6))
            println(Write_File, Row)
        end
    end

    println("\nRPA & TDA photoabsorbtion cross-sections export done ...")

    return
end