function RPA_spurious_ini(Orb_1p1h::Orb2B_1p1h,TrOp::Tr1B)
    # Recover 1p-1h orbitals ...
    N_ph = Orb_1p1h.N[2,2]
    Orb_ph_list = Orb_1p1h.Ind[2,2]

    # Initialize spurious center-of-mass motion vector ...
    ph_CMS = Vector{Float64}(undef,N_ph)

    # Allocate the components of the spurious center-of-mass motion vector ...
    @inbounds for (i_ph,Orb_ph) in enumerate(Orb_ph_list)
        a_p, a_h, T_ph = Orb_ph.a, Orb_ph.b, Orb_ph.T

        if T_ph == 1
            ME_E1 = TrOp.E1.n[a_p,a_h]
            ph_CMS[i_ph] = ME_E1
        elseif T_ph == -1
            ME_E1 = TrOp.E1.p[a_p,a_h]
            ph_CMS[i_ph] = ME_E1
        end
    end

    # Remove numerical noise from the spurious center-of-mass motion vector ...
    @inbounds for i_ph in 1:N_ph
        ME = ph_CMS[i_ph]
        if abs(ME) < 1e-6
            ph_CMS[i_ph] = 0.0
        end
    end

    # Renormalize the spurious center-of-mass motion vector ...
    ph_CMS = ph_CMS ./ norm(ph_CMS)

    return ph_CMS
end

function RPA_spurious_ortho(N_ph::Int64,spurious_level::Vector{Float64})
    # Orthogonalization done by Householder reflection algorithm
    # For explanation see ...
    #
    # https://blogs.mathworks.com/cleve/2016/07/25/compare-gram-schmidt-and-householder-orthogonalization-algorithms/?s_tid=answers_rc2-1_p4_BOTH
    #
    # "More numerically stable than modified Gramm-Schmidt with same numerical cost"

    spurious_ind = 0

    I = diagm(ones(Float64,N_ph))

    @inbounds for a in 1:N_ph
        proj = 0.0
        @inbounds for b in 1:N_ph
            ME = spurious_level[b] * I[b,a]
            proj += ME
        end
        @inbounds for b in 1:N_ph
            ME = I[b,a] - proj * spurious_level[b]
            I[b,a] = ME
        end
    end

    @inbounds for a in 1:N_ph
        b = @views I[:,a] ./ norm(I[:,a]) 
        @views I[:,a] = b
    end

    U = zeros(Float64,N_ph,N_ph)

    @views U[:,1] = I[:,1]

    @inbounds for nu in 2:N_ph
        u = @views I[:,nu]
        @inbounds for mu in 1:(nu - 1)
            u -= @views (U[:, mu]' * u) * U[:, mu]
        end
        if abs(norm(u)) < 1e-4
            spurious_ind = nu
        end
        @views U[:, nu] = u ./ norm(u)
    end
    @views U[:,spurious_ind] = spurious_level

    return U, spurious_ind
end