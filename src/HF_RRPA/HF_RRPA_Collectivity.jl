function HF_RRPA_collectivity(Params::Parameters,N_nu::Matrix{Int64},X_RPA::Matrix{Matrix{ComplexF64}},Y_RPA::Matrix{Matrix{ComplexF64}})
    # Read calculation parameters ...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1

    println("\nEvaluating collectivity of RPA & TDA phonon states ...")

    CI = Matrix{Vector{Float64}}(undef,J_max+1,2)

    # "Phonon-states collectivy" - meassure of collectivity?
    @inbounds for J in 0:J_max
        @inbounds for P in 1:2
            N_ph = N_nu[J+1,P]
            CI_JP = Vector{Float64}(undef,N_ph)
            X_RPA_JP = X_RPA[J+1,P]
            #Y_RPA_JP = Y_RPA[J+1,P]
            @inbounds for ph in 1:N_ph
                dSum = 0.0
                nSum = 0.0
                @inbounds for qg in 1:N_ph
                    ME = abs(X_RPA_JP[qg,ph])^2
                    dSum += ME
                    ME = abs(X_RPA_JP[qg,ph])
                    nSum += ME
                end
                CI_JP[ph] = nSum^2 / dSum
            end
            CI[J+1,P] = CI_JP
        end
    end

    return CI
end