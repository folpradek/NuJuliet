function T2b(Orb::Vector{Orb1B},a::Int64,b::Int64,c::Int64,d::Int64,J::Int64)
    Grad1 = rGrad(a,c,Orb)
    Grad2 = rGrad(b,d,Orb)
    j_a = Orb[a].j
    j_b = Orb[b].j
    j_c = Orb[c].j
    j_d = Orb[d].j
    T_NN = Float64((-1)^(div(j_b + j_c,2) + J)) * Grad1 * Grad2 * f6j(j_a, j_b, 2*J, j_d, j_c, 2)
    return T_NN
end

function rGrad(a::Int64,b::Int64,Orb::Vector{Orb1B})
    n_a = Orb[a].n
    l_a = Orb[a].l
    j_a = Orb[a].j
    n_b = Orb[b].n
    l_b = Orb[b].l
    j_b = Orb[b].j
    Amp = f6j(j_a,j_b,2,2*l_b,2*l_a,1) * sqrt(Float64((j_a + 1) * (j_b + 1))) * Float64((-1)^(l_a + div(j_b + 1,2)))
    Grad = Amp * (sqrt(Float64(l_b+1)*(Float64(n_b+l_b)+3.0/2.0)) * Float64(kronecker_delta(l_a,l_b+1)*kronecker_delta(n_a,n_b)) +
                  sqrt(Float64((l_b+1)*n_b)) * Float64(kronecker_delta(l_a,l_b+1)*kronecker_delta(n_a,n_b-1)) +
                  sqrt(Float64(l_b)*(Float64(n_b+l_b)+1.0/2.0)) * Float64(kronecker_delta(l_a,l_b-1)*kronecker_delta(n_a,n_b)) +
                  sqrt(Float64(l_b*(n_b+1))) * Float64(kronecker_delta(l_a,l_b-1)*kronecker_delta(n_a,n_b+1)))
    return Grad
end