# Functions for 2-body 1p1h orbitals ...

# Move the structures to corresponding file ...
    struct Ind2B
        a::Int64
        b::Int64
        T::Int64
    end

    struct Orb2B_1p1h
        N::Matrix{Int64}
        Ind::Matrix{Vector{Ind2B}}
        Dic::Dict{UInt64,Int64}
    end

@inline function orb2b_1p1h_key(a::Int64,b::Int64,J::Int64,P::Int64,T::Int64)
    return UInt64(P) |
          (UInt64(T + 1) << 2) |
          (UInt64(J) << 4) |
          (UInt64(a) << 10) |
          (UInt64(b) << 18)
end

@inline function orb2b_1p1h_index(a::Int64,b::Int64,J::Int64,P::Int64,T::Int64,Orb_1p1h::Orb2B_1p1h)
    key = orb2b_1p1h_key(a,b,J,P,T)
    dict = Orb_1p1h.Dic
    orb = get(dict,key,0)
    return orb
end

function orbitals_1p1h_make(Params::Parameters,Orb::Vector{Orb1B})
    # Read parameters ...
    N_max = Params.Calc.Nmax
    J_max = 2*N_max + 1
    dn = Params.Calc.RPA.Tol

    # Count the number of particle & hole orbitals ...
    N_ph = zeros(Int64,2,J_max+1)

    @inbounds for J in 0:J_max
        @inbounds for P in 1:2
            N_ph_JP = 0

            @inbounds for Orb_a in Orb
                l_a, j_a, pO_a, nO_a = Orb_a.l, Orb_a.j, Orb_a.pO, Orb_a.nO

                @inbounds for Orb_b in Orb
                    l_b, j_b, pO_b, nO_b = Orb_b.l, Orb_b.j, Orb_b.pO, Orb_b.nO

                    P_ab = rem(l_a + l_b,2) + 1

                    if (P_ab != P) || (triangle(j_a,j_b,2*J) == false)
                        continue
                    end

                    if (pO_b - pO_a) > dn
                        N_ph_JP += 1
                    end

                    if (nO_b - nO_a) > dn
                        N_ph_JP += 1
                    end
                end
            end

            N_ph[P,J+1] = N_ph_JP
        end
    end

    # Initialize index array for 1p-1h orbitals ...
    Ind_ph = Matrix{Vector{Ind2B}}(undef,2,J_max+1)

    @inbounds for P in 1:2
        @inbounds for J in 0:J_max
            N_ph_PJ = N_ph[P,J+1]
            Ind_ph[P,J+1] = Vector{Ind2B}(undef,N_ph_PJ)
        end
    end

    # Initialize mapping dictionary for 1p-1h orbitals ...
    Dic_ph = Dict{UInt64,Int64}()

    # Allocate indices of 1p-1h orbitals ...
    @inbounds for J in 0:J_max
        @inbounds for P in 1:2
            N_ph_JP = 0

            @inbounds for Orb_a in Orb
                a, l_a, j_a, pO_a, nO_a = Orb_a.a, Orb_a.l, Orb_a.j, Orb_a.pO, Orb_a.nO
                
                @inbounds for Orb_b in Orb
                    b, l_b, j_b, pO_b, nO_b = Orb_b.a, Orb_b.l, Orb_b.j, Orb_b.pO, Orb_b.nO

                    P_ab = rem(l_a + l_b,2) + 1

                    if (P_ab != P) || (triangle(j_a,j_b,2*J) == false)
                        continue
                    end

                    if (pO_b - pO_a) > dn
                        N_ph_JP += 1

                        Ind_ph[P,J+1][N_ph_JP] = Ind2B(a,b,-1)

                        Orb_ph_ab = orb2b_1p1h_key(a,b,J,P,-1)
                        Dic_ph[Orb_ph_ab] = N_ph_JP
                    end

                    if (nO_b - nO_a) > dn
                        N_ph_JP += 1

                        Ind_ph[P,J+1][N_ph_JP] = Ind2B(a,b,1)

                        Orb_ph_ab = orb2b_1p1h_key(a,b,J,P,1)
                        Dic_ph[Orb_ph_ab] = N_ph_JP
                    end
                end
            end
        end
    end

    # Allocate the structure for 2-body 1p-1h orbitals ...
    Orb_ph = Orb2B_1p1h(N_ph,Ind_ph,Dic_ph)

    return Orb_ph
end