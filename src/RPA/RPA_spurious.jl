function RPA_spurious_ini(Orb::Vector{Orb1B},Orb_1p1h::Orb2B_1p1h,TrOp::Tr1B)
    # Recover 1p-1h orbitals ...
    N_ph = Orb_1p1h.N[2,2]
    Orb_ph_list = Orb_1p1h.Ind[2,2]

    # Initialize spurious center-of-mass motion vector ...
    Spur_CMS = Vector{Float64}(undef,N_ph)

    # Allocate the components of the spurious center-of-mass motion vector ...
    @inbounds for (i_ph,Orb_ph) in enumerate(Orb_ph_list)
        a_p, a_h, T_ph = Orb_ph.a, Orb_ph.b, Orb_ph.T

        if T_ph == 1
            O_p, O_h = Orb[a_p].nO, Orb[a_h].nO
            D_ph = (O_h - O_p)
            ME_E1 = D_ph * TrOp.E1.n[a_p,a_h]
            Spur_CMS[i_ph] = ME_E1
        elseif T_ph == -1
            O_p, O_h = Orb[a_p].pO, Orb[a_h].pO
            D_ph = (O_h - O_p)
            ME_E1 = D_ph * TrOp.E1.p[a_p,a_h]
            Spur_CMS[i_ph] = ME_E1
        end
    end

    # Remove numerical noise from the spurious center-of-mass motion vector ...
    @inbounds for i_ph in 1:N_ph
        ME = Spur_CMS[i_ph]
        if abs(ME) < 1e-14
            Spur_CMS[i_ph] = 0.0
        end
    end

    # Renormalize the spurious center-of-mass motion vector ...
    Spur_CMS = Spur_CMS ./ norm(Spur_CMS)

    return Spur_CMS
end

function RPA_spurious_ortho!(U::AbstractMatrix{Float64},Spur::Vector{Float64})
    # Comment on optimization by Claude ...
        # Construct an orthonormal basis U of the 1p-1h space such that the column U[:,Spur_ind] is the spurious
        # state Spur & the remaining N_ph - 1 columns span the subspace orthogonal to Spur ...
        #
        # Input  ... U    ... N_ph x N_ph working array, overwritten by the orthonormal basis
        #            Spur ... spurious state normalized to unity, |Spur| = 1
        # Output ... Spur_ind ... index of the column of U containing Spur
        #
        # The orthogonal complement is defined as the Gram-Schmidt orthonormalization (in the natural order a = 1, ..., N_ph)
        # of the unit vectors e_a with the spurious state projected-out,
        #
        #     v_a = (1 - Spur * Spur') * e_a .
        #
        # Since the Gram-Schmidt procedure of v_a has the closed form solution
        #
        #     u_a = (e_a - c_a * Spur_a) / sqrt(1 - c_a * Spur[a]) ,
        #
        #     c_a = Spur[a] / N_a ,      N_a = sum_{b >= a} Spur[b]^2 ,      1 - c_a * Spur[a] = N_(a+1) / N_a ,
        #
        # where Spur_a denotes the vector Spur with components b < a set to zero, the basis is constructed directly
        # in O(N_ph^2) operations without any temporary arrays. Note that one can directly verify Spur' * u_a = 0
        # & v_b' * u_a = 0 for all b < a, i.e. u_a is exactly the a-th Gram-Schmidt vector. The squared norm
        # 1 - c_a * Spur[a] is evaluated as the ratio N_(a+1) / N_a to avoid numerical cancellation.
        #
        # The vector v_a becomes linearly dependent on v_1, ..., v_(a-1) for the last non-zero component of Spur,
        # where N_(a+1) = 0. Following the previous implementation, the dependence is detected by the norm of the
        # Gram-Schmidt residual of the normalized v_a falling below 1e-4 (the last such index is taken) & the
        # corresponding column is replaced by the spurious state Spur ...

    # Read the dimension of 1p-1h space ...
    N_ph = length(Spur)

    # Initialize the spurious index ...
    Spur_ind = 0

    U .= 0.0

    # Construct the Gram-Schmidt vectors u_a in the reversed order, accumulating N_a on the fly ...
    N_a = 0.0
    @inbounds for a in N_ph:-1:1
        Spur_a = Spur[a]
        N_next = N_a
        N_a += Spur_a^2

        # Squared norms of the Gram-Schmidt residual u_a & of the projected vector v_a ...
        # (if N_a = 0, i.e. Spur vanishes for all b >= a, then simply u_a = e_a)
        c_a = 0.0
        u2_a = 1.0

        if N_a > 0.0
            c_a = Spur_a / N_a
            u2_a = N_next / N_a
        end

        v2_a = 1.0 - Spur_a^2

        # Linearly dependent vector ... to be replaced by the spurious state ...
        if Spur_ind == 0 && (v2_a <= 0.0 || sqrt(u2_a / v2_a) < 1e-4)
            Spur_ind = a
            continue
        end

        # Normalized Gram-Schmidt vector u_a ...
        norm_a = 1.0 / sqrt(u2_a)

        @inbounds for b in (a+1):N_ph
            U[b,a] = -norm_a * c_a * Spur[b]
        end

        U[a,a] = sqrt(u2_a)
    end

    # Store the spurious state ...
    @views U[:,Spur_ind] .= Spur

    return Spur_ind
end

@inline function RPA_spurious(Ortho::Bool,J::Int64,P::Int64)
    return Ortho && J == 1 && P == 2
end