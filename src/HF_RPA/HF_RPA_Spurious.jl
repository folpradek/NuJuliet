function Spur_Ini(N_ph::Int64,Orb_Phonon::Vector{Int64},Phonon::Vector{PhState},Particle::pnSVector,Hole::pnSVector,TrOp::Tr1B)
    ph_CMS = Vector{Float64}(undef,N_ph)
    @inbounds for ph in 1:N_ph
        Ind_ph = Orb_Phonon[ph]
        p, h = Phonon[Ind_ph].p, Phonon[Ind_ph].h
        t_ph = Phonon[Ind_ph].tz
        if t_ph == -1
            a_p = Particle.p[p].a
            a_h = Hole.p[h].a
        elseif t_ph == 1
            a_p = Particle.n[p].a
            a_h = Hole.n[h].a
        end

        if t_ph == 1
            ME_E1 = TrOp.E1.n[a_p,a_h]
            ph_CMS[ph] = ME_E1
        elseif t_ph == -1
            ME_E1 = TrOp.E1.p[a_p,a_h]
            ph_CMS[ph] = ME_E1
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

function Spur_Ortho(N_ph::Int64,Spur_State::Vector{Float64})
    # Orthogonalization done by Householder reflection algorithm
    # For explanation see ...
    #
    # https://blogs.mathworks.com/cleve/2016/07/25/compare-gram-schmidt-and-householder-orthogonalization-algorithms/?s_tid=answers_rc2-1_p4_BOTH
    #
    # "More numerically stable than modified Gramm-Schmidt with same numerical cost"

    Spur_Ind = 0

    I = diagm(ones(Float64,N_ph))

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

    U = zeros(Float64, N_ph, N_ph)

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