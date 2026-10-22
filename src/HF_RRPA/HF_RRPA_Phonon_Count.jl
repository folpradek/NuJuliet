function HF_RRPA_phonon_count(Params::Parameters,N_Phonon::Int64,Phonon::Vector{PhState})
    # Read calculation parameters ...
    N_max = Params.Calc.Nmax
    N_2max = 2*N_max
    J_max = N_2max + 1

    println("\nCounting all 1p-1h phonon states ...")

    N_nu = Matrix{Int64}(undef,J_max+1,2)
    Orb_Phonon = Matrix{Vector{Int64}}(undef,J_max+1,2)
    @inbounds for J in 0:J_max
        @inbounds for P in 1:2
            # Count the number of phonon states ...
            N_ph = 0
            @inbounds for ph in 1:N_Phonon
                J_ph, P_ph = Phonon[ph].J, Phonon[ph].P
                if J_ph == J && P == P_ph
                    N_ph += 1
                end
            end
            N_nu[J+1,P] = N_ph

            # For given J & P preallocate contributing phonon indices ...
            Phonon_Orb = Vector{Int64}(undef,N_ph)
            N_ph = 0
            @inbounds for ph in 1:N_Phonon
                J_ph, P_ph = Phonon[ph].J, Phonon[ph].P
                if J_ph == J && P == P_ph
                    N_ph += 1
                    Phonon_Orb[N_ph] = ph
                end
            end
            Orb_Phonon[J+1,P] = Phonon_Orb
        end
    end

    println("\nTotal number of phonons ... J_scheme phonon basis size = " * string(sum(N_nu)))

    return N_nu, Orb_Phonon
end