function RPA_energy(Params::Parameters,N_nu::Matrix{Int64},E_RPA::Matrix{Vector{ComplexF64}},Y_RPA::Matrix{Matrix{ComplexF64}},N::Matrix{Matrix{Float64}})
    # Read parameters ...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1
    Orthogon = Params.Calc.RPA.Ortho

    println("\nCalculating RPA correlation energy ...")

    E_corr_RPA = 0.0
    @inbounds for J in 0:J_max
        J_hat = Float64(2*J + 1)
        @inbounds for P in 1:2
            N_ph = N_nu[J+1,P]
            if !(J == 1 && P == 2)
                @inbounds for nu in 1:N_ph
                    @inbounds for ph in 1:N_ph
                        ME = -real(E_RPA[J+1,P][nu]) * abs(Y_RPA[J+1,P][ph,nu])^2 * J_hat * N[J+1,P][ph,ph]
                        E_corr_RPA += ME
                    end
                end
            else
                if Orthogon == true
                    @inbounds for nu in 2:N_ph
                        @inbounds for ph in 1:N_ph
                            ME = -real(E_RPA[J+1,P][nu]) * abs(Y_RPA[J+1,P][ph,nu])^2 * J_hat * N[J+1,P][ph,ph]
                            E_corr_RPA += ME
                        end
                    end
                else
                    @inbounds for nu in 1:N_ph
                        @inbounds for ph in 1:N_ph
                            ME = -real(E_RPA[J+1,P][nu]) * abs(Y_RPA[J+1,P][ph,nu])^2 * J_hat * N[J+1,P][ph,ph]
                            E_corr_RPA += ME
                        end
                    end
                end

            end
        end
    end

    println("\tCorrelation RPA energy is ...  E_RPA = " * string(round(real(E_corr_RPA),digits=6)) * " MeV\n")

    return E_corr_RPA
end