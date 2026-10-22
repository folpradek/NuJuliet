function HF_RPA_collectivity(Params::Parameters,N_nu::Matrix{Int64},X_TDA::Matrix{Matrix{Float64}},X_RPA::Matrix{Matrix{ComplexF64}},Y_RPA::Matrix{Matrix{ComplexF64}})
    # Read calculation parameters ...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1

    println("\nEvaluating collectivity of RPA & TDA phonon states ...")

    CI_TDA = Matrix{Vector{Float64}}(undef,J_max+1,2)
    CI_RPA = Matrix{Vector{Float64}}(undef,J_max+1,2)

    # "Phonon-states collectivy" - meassure of collectivity?
    @inbounds for J in 0:J_max
        @inbounds for P in 1:2
            N_ph = N_nu[J+1,P]
            CI_TDA_JP = Vector{Float64}(undef,N_ph)
            CI_RPA_JP = Vector{Float64}(undef,N_ph)
            X_TDA_JP = X_TDA[J+1,P]
            X_RPA_JP = X_RPA[J+1,P]
            Y_RPA_JP = Y_RPA[J+1,P]
            @inbounds for ph in 1:N_ph
                dSum_TDA = 0.0
                nSum_TDA = 0.0
                dSum_RPA = 0.0
                nSum_RPA = 0.0
                @inbounds for qg in 1:N_ph
                    ME = abs(X_TDA_JP[qg,ph])^2
                    dSum_TDA += ME
                    ME = abs(X_TDA_JP[qg,ph])
                    nSum_TDA += ME
                    ME = abs(X_RPA_JP[qg,ph])^2
                    dSum_RPA += ME
                    ME = abs(X_RPA_JP[qg,ph])
                    nSum_RPA += ME
                end
                CI_TDA_JP[ph] = nSum_TDA^2 / dSum_TDA
                CI_RPA_JP[ph] = nSum_RPA^2 / dSum_RPA
            end
            CI_TDA[J+1,P] = CI_TDA_JP
            CI_RPA[J+1,P] = CI_RPA_JP
        end
    end

    return CI_TDA, CI_RPA
end