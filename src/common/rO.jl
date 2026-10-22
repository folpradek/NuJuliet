@inline function rGrad(a::Int64,b::Int64,Orb::Vector{Orb1B})
    n_a, l_a, j_a = Orb[a].n, Orb[a].l, Orb[a].j
    n_b, l_b, j_b = Orb[b].n, Orb[b].l, Orb[b].j
    Amp = f6j(j_a,j_b,2,2*l_b,2*l_a,1) * sqrt(Float64((j_a + 1) * (j_b + 1))) * Float64((-1)^(l_a + div(j_b + 1,2)))
    rME = Amp * (sqrt(Float64(l_b+1)*(Float64(n_b+l_b)+3.0/2.0)) * Float64(kronecker_delta(l_a,l_b+1)*kronecker_delta(n_a,n_b)) +
                sqrt(Float64((l_b+1)*n_b)) * Float64(kronecker_delta(l_a,l_b+1)*kronecker_delta(n_a,n_b-1)) +
                sqrt(Float64(l_b)*(Float64(n_b+l_b)+1.0/2.0)) * Float64(kronecker_delta(l_a,l_b-1)*kronecker_delta(n_a,n_b)) +
                sqrt(Float64(l_b*(n_b+1))) * Float64(kronecker_delta(l_a,l_b-1)*kronecker_delta(n_a,n_b+1)))
    return rME
end

@inline function rL(a::Int64,b::Int64,Orb::Vector{Orb1B})
    l_a, l_b = Orb[a].l, Orb[b].l
    if l_a == l_b
        j_a, j_b = Orb[a].j, Orb[b].j
        rME = sqrt(Float64((j_a + 1) * (j_b + 1) * l_a * (l_a + 1) * (2*l_a + 1))) *
            Float64((-1)^(l_a + div(j_b + 3,2))) *f6j(2*l_a,2*l_a,2,j_b,j_a,1)
        return rME
    else
        return 0.0
    end
end

@inline function rY_scalar(a::Int64,b::Int64,L::Int64,Orb::Vector{Orb1B})
    l_a, l_b = Orb[a].l, Orb[b].l
    if rem(l_a + l_b + L,2) == 0
        j_a, j_b = Orb[a].j, Orb[b].j
        rME = Float64((-1)^(L + div(j_b - 1,2))) * sqrt(Float64((j_a + 1) * (j_b + 1)) / (4.0 * pi)) * fCG(j_a,j_b,2*L,1,-1,0)
        return rME
    else
        return 0.0
    end
end

@inline function rY_vector(a::Int64,b::Int64,L::Int64,J::Int64,a_max::Int64,Orb::Vector{Orb1B})
    j_a, j_b = Orb[a].j, Orb[b].j
    Amp = Float64((-1)^(j_a + j_b + J)) * sqrt(Float64(2*J + 1))
    rME = 0.0
    @inbounds for c in 1:a_max
        j_c = Orb[c].j
        ME = Amp * f6j(2*L,2,2*J,j_b,j_a,j_c) * rY_scalar(a,c,L,Orb) * rL(c,b,Orb)
        rME += ME
    end
    return rME
end

@inline function rS(a::Int64,b::Int64,Orb::Vector{Orb1B})
    l_a, l_b = Orb[a].l, Orb[b].l
    if l_a == l_b
        j_a, j_b = Orb[a].j, Orb[b].j
        rME = 0.5 * Float64((-1)^(l_a + div(j_b + 3,2))) * sqrt(Float64(6 * (j_a + 1) * (j_b + 1))) * f6j(1,1,2,j_b,j_a,2*l_a)
        return rME
    else
        return 0.0
    end
end

@inline function rRN(a::Int64,b::Int64,N::Int64,hw::Float64,Orb::Vector{Orb1B})
    n_a, l_a = Orb[a].n, Orb[a].l
    n_b, l_b = Orb[b].n, Orb[b].l
    b_osc = sqrt(0.5 * (939.565346 + 938.272013) * hw) / 197.326980
    rME = radial_moment_LHO(N,n_a,l_a,n_b,l_b,b_osc)
    return rME
end

# Note the reduced matrix element corresponds to [Grad x S]^(1) rather than curl S ... -i sqrt(2) is absent ...
@inline function rGrad_cross_S(a::Int64,b::Int64,a_max::Int64,Orb::Vector{Orb1B})
    j_a, j_b = Orb[a].j, Orb[b].j
    Amp = Float64((-1)^(1 + div(j_a + j_b,2))) * sqrt(3.0)
    rME = 0.0
    @inbounds for c in 1:a_max
        j_c = Orb[c].j
        ME = Amp * f6j(2,2,2,j_b,j_a,j_c) * rGrad(a,c,Orb) * rS(c,b,Orb)
        rME += ME
    end
    return rME
end

@inline function rRN_Y_scalar(a::Int64,b::Int64,N::Int64,L::Int64,hw::Float64,Orb::Vector{Orb1B})
    rME = rRN(a,b,N,hw,Orb) * rY_scalar(a,b,L,Orb)
    return rME
end

@inline function rRN_Y_vector(a::Int64,b::Int64,N::Int64,L::Int64,J::Int64,hw::Float64,a_max::Int64,Orb::Vector{Orb1B})
    rME = rRN(a,b,N,hw,Orb) * rY_vector(a,b,L,J,a_max,Orb)
    return rME
end

@inline function rGrad_dot_Y_vector(a::Int64,b::Int64,L::Int64,J::Int64,a_max::Int64,Orb::Vector{Orb1B})
    j_a, j_b = Orb[a].j, Orb[b].j
    if j_a == j_b
        Amp = Float64(j_a + 1)
        rME = 0.0
        @inbounds for c in 1:a_max
            j_c = Orb[c].j
            ME = Float64((-1)^(div(j_c - j_a,2))) / Amp * rGrad(a,c,Orb) * rY_vector(c,b,L,J,a_max,Orb)
            rME += ME
        end
        return rME
    else
        return 0.0
    end
end

@inline function rGrad_dot_RN_Y_vector(a::Int64,b::Int64,N::Int64,L::Int64,J::Int64,hw::Float64,a_max::Int64,Orb::Vector{Orb1B})
    j_a, j_b = Orb[a].j, Orb[b].j
    if j_a == j_b
        Amp = Float64(j_a + 1)
        rME = 0.0
        @inbounds for c in 1:a_max
            j_c = Orb[c].j
            ME = Float64((-1)^(div(j_c - j_a,2))) / Amp * rGrad(a,c,Orb) * rRN_Y_vector(c,b,N,L,J,hw,a_max,Orb)
            rME += ME
        end
        return rME
    else
        return 0.0
    end
end

@inline function rGrad_cross_S_dot_Y_vector(a::Int64,b::Int64,L::Int64,J::Int64,a_max::Int64,Orb::Vector{Orb1B})
    j_a, j_b = Orb[a].j, Orb[b].j
    if j_a == j_b
        Amp = Float64(j_a + 1)
        rME = 0.0
        @inbounds for c in 1:a_max
            j_c = Orb[c].j
            ME = Float64((-1)^(div(j_c - j_a,2))) / Amp * rGrad_cross_S(a,c,a_max,Orb) * rY_vector(c,b,L,J,a_max,Orb)
            rME += ME
        end
        return rME
    else
        return 0.0
    end
end

@inline function rGrad_cross_S_dot_RN_Y_vector(a::Int64,b::Int64,N::Int64,L::Int64,J::Int64,hw::Float64,a_max::Int64,Orb::Vector{Orb1B})
    j_a, j_b = Orb[a].j, Orb[b].j
    if j_a == j_b
        Amp = Float64(j_a + 1)
        rME = 0.0
        @inbounds for c in 1:a_max
            j_c = Orb[c].j
            ME = Float64((-1)^(div(j_c - j_a,2))) / Amp * rGrad_cross_S(a,c,a_max,Orb) * rRN_Y_vector(c,b,N,L,J,hw,a_max,Orb)
            rME += ME
        end
        return rME
    else
        return 0.0
    end
end