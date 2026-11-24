function orbitals_pn2qp_make(Params::Parameters,Orb::Vector{Orb1B})
    # Read parameters ...
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1)*(N_max + 2),2)
    J_max = 2*N_max + 1

    # Count the total number of pn 2qp orbitals for each J & P ... a = proton, b = neutron, no ordering restriction ...
    N_2qp = zeros(Int64,2,J_max+1)

    @inbounds for a in 1:a_max
        l_a ,j_a = Orb[a].l , Orb[a].j
        @inbounds for b in 1:a_max
            l_b ,j_b = Orb[b].l , Orb[b].j
            P = parity(l_a + l_b)
            @inbounds for J in div(abs(j_a - j_b),2):div(j_a + j_b,2)
                N_2qp[P,J+1] += 1
            end
        end
    end

    # Initialize the matrix storage for pn 2qp orbitals ...
    Orb_2qp_list = Matrix{Vector{Ind2B}}(undef,2,J_max+1)

    # Initialize mapping dictionary for pn 2qp orbitals ...
    Dic_2qp = Dict{UInt64,Int64}()

    # Initialize the entries of the storage matrix for pn 2qp orbitals ...
    @inbounds for P in 1:2
        @inbounds for J in 0:J_max
            Orb_2qp_list[P,J+1] = Vector{Ind2B}(undef,N_2qp[P,J+1])
            N_2qp[P,J+1] = 0
        end
    end

    # Allocate the indices for pn 2qp orbitals ... T = 0 labels the pn pair ...
    @inbounds for a in 1:a_max
        l_a ,j_a = Orb[a].l, Orb[a].j
        @inbounds for b in 1:a_max
            l_b ,j_b = Orb[b].l, Orb[b].j
            P = parity(l_a + l_b)
            @inbounds for J in div(abs(j_a - j_b),2):div(j_a + j_b,2)
                N_2qp[P,J+1] += 1
                Orb_2qp_list[P,J+1][N_2qp[P,J+1]] = Ind2B(a,b,0)

                Orb_2qp_ab = orb2b_2qp_key(a,b,J,P,0)
                Dic_2qp[Orb_2qp_ab] = N_2qp[P,J+1]
            end
        end
    end

    Orb_pn2qp = Orb2B_2qp(N_2qp,Orb_2qp_list,Dic_2qp)

    return Orb_pn2qp
end
