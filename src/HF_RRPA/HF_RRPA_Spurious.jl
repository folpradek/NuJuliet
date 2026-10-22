function HF_RRPA_spurious_initialize(N_ph::Int64,Orb_Phonon::Vector{Int64},Phonon::Vector{PhState},Particle::pnSVector,Hole::pnSVector,TrOp::Tr1B,Rho::O1B,X_RPA::Matrix{Matrix{ComplexF64}},Y_RPA::Matrix{Matrix{ComplexF64}})
    ph_CMS = Vector{ComplexF64}(undef,N_ph)
    @inbounds for ph in 1:N_ph
        Ind_ph = Orb_Phonon[ph]
        p, h = Phonon[Ind_ph].p, Phonon[Ind_ph].h
        t_ph, J_ph, P_ph = Phonon[Ind_ph].tz, Phonon[Ind_ph].J, Phonon[Ind_ph].P
        if t_ph == -1
            a_p = Particle.p[p].a
            a_h = Hole.p[h].a
        elseif t_ph == 1
            a_p = Particle.n[p].a
            a_h = Hole.n[h].a
        end
        
        if t_ph == 1
            Sum = 0.0
            @inbounds for nu in 1:N_ph
                @inbounds for qg in 1:N_ph
                    Ind_qg = Orb_Phonon[qg]
                    q, g = Phonon[Ind_qg].p, Phonon[Ind_qg].h
                    t_qg, J_qg, P_qg = Phonon[Ind_qg].tz, Phonon[Ind_qg].J, Phonon[Ind_qg].P
                    if t_qg == -1
                        a_q = Particle.p[q].a
                        a_g = Hole.p[g].a
                    elseif t_qg == 1
                        a_q = Particle.n[q].a
                        a_g = Hole.n[g].a
                    end

                    if t_qg == t_ph && J_ph == J_qg && P_ph == P_qg
                        ME_E1 = TrOp.E1.n[a_q,a_g] * sqrt(Rho.n[a_g,a_g] - Rho.n[a_q,a_q]) * (conj(Y_RPA[J_ph+1,P_ph][qg,nu]) - conj(X_RPA[J_ph+1,P_ph][qg,nu])) * (X_RPA[J_ph+1,P_ph][ph,nu] - Y_RPA[J_ph+1,P_ph][ph,nu])
                        Sum += ME_E1
                    end
                end
            end
            ph_CMS[ph] = Sum
        elseif t_ph == -1
            Sum = 0.0
            @inbounds for nu in 1:N_ph
                @inbounds for qg in 1:N_ph
                    Ind_qg = Orb_Phonon[qg]
                    q, g = Phonon[Ind_qg].p, Phonon[Ind_qg].h
                    t_qg, J_qg, P_qg = Phonon[Ind_qg].tz, Phonon[Ind_qg].J, Phonon[Ind_qg].P
                    if t_qg == -1
                        a_q = Particle.p[q].a
                        a_g = Hole.p[g].a
                    elseif t_qg == 1
                        a_q = Particle.n[q].a
                        a_g = Hole.n[g].a
                    end

                    if t_qg == t_ph && J_ph == J_qg && P_ph == P_qg
                        ME_E1 = TrOp.E1.p[a_q,a_g] * sqrt(Rho.p[a_g,a_g] - Rho.p[a_q,a_q]) *  (conj(Y_RPA[J_ph+1,P_ph][qg,nu]) - conj(X_RPA[J_ph+1,P_ph][qg,nu])) * (X_RPA[J_ph+1,P_ph][ph,nu] - Y_RPA[J_ph+1,P_ph][ph,nu])
                        Sum += ME_E1
                    end
                end
            end
            ph_CMS[ph] = Sum
        end

    end

    @inbounds for ph in 1:N_ph
        ME = ph_CMS[ph]
        if abs(ME) < 1e-6
            ph_CMS[ph] = 0.0
        end
    end

    ph_CMS = ph_CMS ./ norm(ph_CMS)
    return ph_CMS
    
end

function HF_RRPA_spurious_orthogonalize(N_ph::Int64,Spur_State::Vector{ComplexF64})
    # Orthogonalization done by Householder reflection algorithm
    # For explanation see ...
    #
    # https://blogs.mathworks.com/cleve/2016/07/25/compare-gram-schmidt-and-householder-orthogonalization-algorithms/?s_tid=answers_rc2-1_p4_BOTH
    #
    # "More numerically stable than modified Gramm-Schmidt with same numerical cost"

    Spur_Ind = 0

    I = diagm(ones(ComplexF64,N_ph))

    @inbounds for a in 1:N_ph
        proj = 0.0
        @inbounds for b in 1:N_ph
            ME = Spur_State[b] * I[b,a]
            proj += ME
        end
        @inbounds for b in 1:N_ph
            ME = I[b,a] - proj * Spur_State[b]
            I[b,a] = ME
        end
    end

    @inbounds for a in 1:N_ph
        b = @views I[:,a] ./ norm(I[:,a]) 
        @views I[:,a] = b
    end

    U = zeros(ComplexF64, N_ph, N_ph)

    @views U[:,1] = I[:,1]

    @inbounds for nu in 2:N_ph
        u = @views I[:,nu]
        @inbounds for mu in 1:(nu - 1)
            u -= @views (U[:, mu]' * u) * U[:, mu]
        end
        if abs(norm(u)) < 1e-4
            Spur_Ind = nu
        end
        @views U[:, nu] = u ./ norm(u)
    end
    @views U[:,Spur_Ind] = Spur_State

    # Check orthonormality ... precision in order of 1e-15 or smaller is thanks to the Householder algorithm ... very precise
    #println("Orthogonality check (close to identity?) ...")
    #orthogonality_check = U' * U
    #display(orthogonality_check)

    return U, Spur_Ind
end