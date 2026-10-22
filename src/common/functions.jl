function KroneckerDelta(a::Int,b::Int)
    if a == b
        return 1
    else
        return 0
    end
end

@inline function kronecker_delta(a::Int,b::Int)
    if a == b
        return 1
    else
        return 0
    end
end

@inline function phase(Arg::Int)
    Phase = isodd(Arg) ? -1.0 : 1.0
    return Phase
end

# Angular momentum algebra wrappers
@inline function cg(j_1::Int,j_2::Int,j_3::Int,m_1::Int,m_2::Int,m_3::Int)
    return fCG(j_1,j_2,j_3,m_1,m_2,m_3)
end

@inline function w3j(j_1::Int,j_2::Int,j_3::Int,m_1::Int,m_2::Int,m_3::Int)
    return f3j(j_1,j_2,j_3,m_1,m_2,m_3)
end

@inline function w6j(j_1::Int64,j_2::Int64,j_3::Int64,j_4::Int64,j_5::Int64,j_6::Int64)
    return f6j(j_1,j_2,j_3,j_4,j_5,j_6)
end

@inline function w9j(j_1::Int,j_2::Int,j_3::Int,j_4::Int,j_5::Int,j_6::Int,j_7::Int,j_8::Int,j_9::Int)
    return f9j(j_1,j_2,j_3,j_4,j_5,j_6,j_7,j_8,j_9)
end

@inline function triangle(j_1::Int64,j_2::Int64,j_3::Int64)
    if (abs(j_1 - j_2) <= j_3) && (j_3 <= (j_1 + j_2))
        return true
    else
        return false
    end
end

@inline function hat(J::Int64)
    return Float64(2*J + 1)
end

pretty_summarysize(x) = Base.format_bytes(Base.summarysize(x))

function lorentzian(x::Float64,w::Float64)
    return w / (x^2 + w^2 / 4.0) / (2.0 * pi)
end

function GeneralizedLaguerre(n::Int,k::Number,x::AbstractArray)
    if n == 0
        return ones(eltype(x), size(x))
    elseif n == 1
        return 1 .+ k .- x
    else
        GL1 = ones(eltype(x), size(x))
        GL2 = 1 .+ k .- x
        GL = GL2
        @inbounds for j in 2:n
            GL = ((k + 2*j - 1 .- x) .* GL2 .+ (-k - j + 1) .* GL1) ./ j
            GL1 = GL2
            GL2 = GL
        end
        return GL
    end
end

function GeneralizedLaguerre(n::Int,k::Number,x::Number)
    if n == 0
        return 1
    elseif n == 1
        return 1 + k - x
    else
        GL1 = 1
        GL2 = 1 + k - x
        GL = GL2
        @inbounds for j in 2:n
            GL = ((k + 2*j - 1 - x) * GL2 + (-k - j + 1) * GL1) / j
            GL1 = GL2
            GL2 = GL
        end
        return GL
    end
end

function doublefactorial(number::Integer)
    fact = one(number)
    @inbounds for m in iseven(number)+1:2:number
        fact *= m
    end
    return fact
end

function Psi_rad_LHO(r::Float64,n::Int64,l::Int64,nu::Float64)
    N = sqrt(sqrt(2 * nu^3 / π) * 2^(n + 2*l + 3) * factorial(n) * nu^l / doublefactorial(2*n + 2*l + 1))
    Psi =  N * exp(-nu * r^2) * r^l * GeneralizedLaguerre(n, l + 0.5, 2 * nu * r^2)
    return Psi
end

function integrate_trap(x::Vector{Float64},y::Vector{Float64})
    n = length(x) - 1
    I = 0.0
    @inbounds for i in 1:n
        dx = x[i+1] - x[i]
        I += 0.5 * (y[i] + y[i+1]) * dx
    end
    return I
end

function radial_moment_matrix_LHO(lambda::Int64,hw::Float64,Orb::Vector{Orb1B})
    HbarC = 197.326980
    m_n = 939.565346
    m_p = 938.272013
    b_osc = 1.0/ (HbarC / sqrt(0.5 * (m_p + m_n) * hw))
    
    a_max = length(Orb)
    Radial_lambda = zeros(Float64,a_max,a_max)

    @inbounds Threads.@threads for a in 1:a_max
        n_a = Orb[a].n
        l_a = Orb[a].l
        @inbounds for b in 1:a_max
            n_b = Orb[b].n
            l_b = Orb[b].l
            ME = radial_moment_LHO(lambda,n_a,l_a,n_b,l_b,b_osc)
            Radial_lambda[a,b] = ME
        end
    end

    return Radial_lambda
end

function radial_moment_LHO(k::Int64,n_a::Int64,l_a::Int64,n_b::Int64,l_b::Int64,b_osc::Float64)
    gamma = Vector{Float64}(undef,171)
    gamma[1] = 1.0
    gamma[2] = sqrt(pi)
    gamma[3] = 1.0
    @inbounds for n in 3:170
        gamma[n+1] = (Float64(n)/2.0 - 1.0)*gamma[n-1]
    end

    dfactorial = Vector{Float64}(undef,171)
    dfactorial[1] = 1.0
    dfactorial[2] = 1.0
    @inbounds for n in 2:170
        dfactorial[n+1] = Float64(n)*dfactorial[n-1]
    end

    gi = Vector{Float64}(undef,171)
    gi[1] = sqrt(pi) / 2.0
    gi[2] = 0.5
    @inbounds for n in 2:170
        if rem(n,2) == 0
            gi[n+1] = sqrt(pi) * dfactorial[n] / Float64(2^(1+div(n,2)))
        elseif rem(n,2) == 1
            gi[n+1] = dfactorial[n] / Float64(2^(1+div(n-1,2)))
        end
    end

    Amp = 4.0 / (b_osc^(k+0) * sqrt(pi)) * sqrt(Float64(factorial(n_a) * factorial(n_b) * 2^(n_a + n_b + l_a + l_b)) /
                dfactorial[2*n_a + 2*l_a + 2] / dfactorial[2*n_b + 2*l_b + 2])

    Sum = 0.0

    @inbounds for m_a in 0:n_a
        @inbounds for m_b in 0:n_b
            Sum += Float64((-1)^(m_a + m_b) * gamma[2*n_a + 2*l_a + 3 + 1] * gamma[2*n_b + 2*l_b + 3 + 1] *
                    gi[k + 0 + 2 + l_a + l_b + 2*m_a + 2*m_b + 1] / factorial(m_a) / factorial(m_b) /
                    factorial(n_a - m_a) / factorial(n_b - m_b) / gamma[2*l_a + 2*m_a + 3 + 1] / gamma[2*l_b + 2*m_b + 3 + 1])
        end
    end

    I = Sum * Amp

    return I
end

function logm(A::Matrix{Float64})
    D, U = eigen(A)
    l = length(D)
    @inbounds for i in 1:l
        if D[i] < 0.0
            D[i] = -1.0 * D[i]
        end
    end
    D = diagm(log.(D))
    A_log = U * D * U'
    return A_log
end

function JP_initialize(J_max::Int64)
    JP = Vector{Vector{Int64}}(undef,2*(J_max+1))
    JP_count = 0
    @inbounds for J in 0:J_max
        @inbounds for P in 1:2
            JP_count += 1
            JP[JP_count] = [J, P]
        end
    end
    return JP
end

function differentiate_Oh6(x::Vector{Float64},f::Vector{Float64})
    # Central, forward & backward 6th-order differences with 7-point stencils ...
    # See numerical coefficients at https://en.wikipedia.org/wiki/Finite_difference_coefficient#

    # Determine the grid size ...
    n = length(x)

    # Check the minimal grid size for the 7-point difference ...
    if n < 7
        throw("The numerical 6th order O(h^6) numerical differentiation requires at least 7 grid points ...")
    end

    # Check the grid dimensions match ...
    if length(f) != n
        throw("Grid dimensions do not matchh ... numerical differentation is not possible ...")
    end

    # Determine numerical step h ... assuming a uniform grid ...
    dx = x[2] - x[1]

    # Initialize the numerical derivative grid ...
    df = Vector{Float64}(undef,n)

    # Perform the 6th-order central difference approximation at interior points ...
    @inbounds for j in 4:(n-3)
        df[j] = (f[j+3] - 9f[j+2] + 45f[j+1] - 45f[j-1] + 9f[j-2] - f[j-3]) / (60dx)
    end

    # Perform the 6th-order difference approximation at boundary points ...
    @inbounds begin
        # 6th-order forward difference ...
        df[1] = (- 147f[1] + 360f[2] - 450f[3] + 400f[4] - 225f[5] + 72f[6] -10f[7]) / (60dx)
        df[2] = (- 147f[2] + 360f[3] - 450f[4] + 400f[5] - 225f[6] + 72f[7] -10f[8]) / (60dx)
        df[3] = (- 147f[3] + 360f[4] - 450f[5] + 400f[6] - 225f[7] + 72f[8] -10f[9]) / (60dx)

        # 6th-order backward difference ...
        df[n]   = (147f[n] - 360f[n-1] + 450f[n-2] - 400f[n-3] + 225f[n-4] - 72f[n-5] + 10f[n-6]) / (60dx)
        df[n-1] = (147f[n-1] - 360f[n-2] + 450f[n-3] - 400f[n-4] + 225f[n-5] - 72f[n-6] + 10f[n-7]) / (60dx)
        df[n-2] = (147f[n-2] - 360f[n-3] + 450f[n-4] - 400f[n-5] + 225f[n-6] - 72f[n-7] + 10f[n-8]) / (60dx)
    end

    return df
end

function integrate_quadrature(x::Vector{Float64},f::Vector{Float64};M::Int64=0)
    # Read size of field x ...
    N = length(x)

    # Check if the Romberg quadrature is possible ...
    if N < 2
        error("Romberg integration requires at least two points.")
    end

    if N != length(f)
        error("Romberg integration requires arrays of equal sizes!")
    end

    if mod((N-1),2) != 0
        error("Romberg integration requires array size to be of form (2^k + 1)!")
    end

    if M > 0 && M < Int(floor(log2(N-1)))
        n = M
    else
        n = Int(floor(log2(N-1)))
    end

    # Check if at least one Richardson's extrapolation for the Romberg iteration is possible
    # if not, performs trapezoidal rule ...
    if n < 1
        h = x[end] - x[1]
        return h * (f[1] + f[end]) / 2.0
    end

    # Initialize the Romberg table collumn ...
    R = zeros(Float64,n+1)

    # Calculate the Romberg table ... R_j,0 ...
    @inbounds for j in 0:n
        t_j = 2^j
        h_j = (x[end] - x[1]) / t_j

        # Trapezoidal rule ...
        I = (f[1] + f[end]) / 2.0

        s = div((N - 1), t_j)
        @inbounds for k in 1:(t_j - 1)
            I += f[1+k*s]
        end
        R[j+1] = I * h_j

    end

    # Extrapolation for the rest of the Romberg's table ...
    @inbounds for k in 1:n
        @inbounds for j in n:-1:k
            f_j = 4.0^j
            R[j+1] = (f_j * R[j + 1] - R[j]) / (f_j - 1.0)
        end
    end

    return R[n+1]
end

@inline function legendre_polynomial(n::Int,x::Float64)
    # Check the validity of input arguments ...
    if n < 0
        throw(ArgumentError("Degree n must be non-negative (got n=$n) ..."))
    end
    if abs(x) > 1.0 + 1e-15
        throw(DomainError(x, "x must be in the range [-1, 1] for Legendre polynomials ..."))
    end

    # Initial values for iteration ... L_0(x) = 1 & L_1(x) = x ...
    if n == 0
        return 1.0
    elseif n == 1
        return x
    else
        Ln1 = 1.0
        Ln2 = x
        @inbounds for i in 2:n
            Ln3 = (Float64(2*i - 1) * x * Ln2 - Float64(i - 1) * Ln1) / Float64(i)
            Ln1 = Ln2
            Ln2 = Ln3
        end
        return Ln2
    end
end

@inline function associated_legendre_polynomial(n::Int,m::Int,x::Float64)
    # Check the validity of input arguments ...
    if n < 0
        throw(ArgumentError("Degree n must be non-negative (got n=$n) ..."))
    end
    if m < 0 || m > n
        throw(ArgumentError("Order m must satisfy 0 <= m <= n (got m=$m, n=$n) ..."))
    end
    if abs(x) > 1.0 + 1e-15
        throw(DomainError(x, "x must be in the range [-1, 1] for Legendre polynomials ..."))
    end

    # Calculate L_m^m ...
    Lmm = 1.0
    if m > 0
        #somx2 = sqrt((1.0 - x) * (1.0 + x))
        somx2 = sqrt(max(0.0, 1.0 - x^2))
        f = 1.0
        @inbounds for i in 1:m
            Lmm *= -f * somx2
            f += 2.0
        end
    end

    if n == m
        return Lmm
    end

    # Calculate L_m+1^m ...
    Lmmp1 = x * (2m + 1) * Lmm

    if n == m + 1
        return Lmmp1
    end

    # Calculate L_n^m ...
    Ln = 0.0
    @inbounds for i in (m + 2):n
        two_i_minus_1 = 2i - 1
        i_plus_m_minus_1 = i + m - 1
        i_minus_m = i - m
        
        Ln = (two_i_minus_1 * x * Lmmp1 - i_plus_m_minus_1 * Lmm) / i_minus_m
        Lmm = Lmmp1
        Lmmp1 = Ln
    end

    return Ln
end

@inline function normalize_spherical_angles(theta::Float64,phi::Float64)
    theta_norm = mod(theta, 2.0 * pi)
    phi_norm = mod(phi, 2.0 * pi)

    if theta_norm > pi
        theta_norm = 2.0 * pi - theta_norm
        phi_norm = mod(phi_norm + pi, 2.0 * pi)
    end

    return theta_norm, phi_norm
end

@inline function scalar_spherical_harmonic_function(l::Int,m::Int,theta::Float64,phi::Float64)
    # Check the validity of input arguments ...
    if abs(m) > l
        throw(ArgumentError("Order |m| cannot be greater than degree n (got l=$l, m=$m) ..."))
    end
    #=
    if theta < 0 || theta > pi + 1e-15
        throw(DomainError(theta, "theta must be in the range [0, pi] ..."))
    end
    =#

    # Proper treatment of projection m ...
    m_abs = abs(m)
    if m_abs > l
        return 0.0im
    end

    theta, phi = normalize_spherical_angles(theta, phi)

    # Ensure x in range of [-1,1] ...
    x = clamp(cos(theta), -1.0, 1.0)
    
    # Call the Associated Legendre Polynomial value ... at x = cos(theta) ...
    Llm = associated_legendre_polynomial(l,m_abs,x)
    
    # Calculate the normalization factor sqrt((2l+1)/4pi * (l-m)!/(l+m)!) ... see Varshalovich ...
    # Factorial is computed iteratively to avoid overflow ...
    r = 1.0
    @inbounds for i in (l - m_abs + 1):(l + m_abs)
        r /= i
    end
    N = sqrt((2*l + 1) * r / (4.0 * pi))
    
    # Calculate the azimuthal phase exp(i m phi) ...
        # Case of m non-negative ...
    if m >= 0
        Phase = ComplexF64(cos(m * phi), sin(m * phi))
        return Phase * N * Llm

        # Case of m negative ...
    else
        Phase_pm = ComplexF64(cos(m_abs * phi), sin(m_abs * phi))
        Ylpm = Phase_pm * N * Llm 
        
        # Apply Y_L-m = (-1)^m * (Y_n^m)* ...
        Ylm = (m_abs % 2 == 0) ? conj(Ylpm) : -conj(Ylpm)
        return Ylm
    end
end

function vector_spherical_harmonic_function(J::Int,L::Int,M::Int,theta::Float64,phi::Float64)
    # Check the validity of input arguments ...
    if abs(M) > J
        throw(ArgumentError("Total projection |M| cannot exceed total angular momentum J ..."))
    end
    if abs(L - J) > 1
        throw(ArgumentError("Triangle inequality violated: |L - J| must be <= 1 (got L=$L, J=$J) ..."))
    end
    if J < 0 || L < 0
        throw(ArgumentError("Angular momentum indices J & L must be non-negative ..."))
    end
    #=
    if theta < 0 || theta > pi + 1e-15
        throw(DomainError(theta, "theta must be in the range [0, pi] ..."))
    end
    =#

    theta, phi = normalize_spherical_angles(theta, phi)

    # Initialize components of Y_JLM ...
    Yx, Yy, Yz = 0.0im, 0.0im, 0.0im

    # Precompite the square root ...
    isqrt2 = 1.0 / sqrt(2.0)
    
    # Sum over all possible projections mu of unit vector...
    @inbounds for mu in -1:1
        # Determine scalar spherical harmonic projection number m ...
        m = M - mu
        
        # Selection rule |m| <= L ...
        if abs(m) <= L
            # Calculate the Clebsch-Gordan coefficient ...
            CG = fCG(2*L,2,2*J,2*m,2*mu,2*M)
            
            if abs(CG) > 1e-12
                # Calculate the scalar harmonic Y_Lm ...
                Ylm = scalar_spherical_harmonic_function(L,m,theta,phi)
                CG_Ylm = CG * Ylm
                
                # Determine the cartesian components of Y_JLM ...
                if mu == 1
                    Yx -= CG_Ylm * isqrt2
                    Yy -= CG_Ylm * isqrt2 * 1im
                elseif mu == 0
                    Yz += CG_Ylm
                elseif mu == -1
                    Yx += CG_Ylm * isqrt2
                    Yy -= CG_Ylm * isqrt2 * 1im
                end
            end
        end
    end
    
    return (Yx,Yy,Yz)
end



@inline function J1b_convective(r::Float64,theta::Float64,phi::Float64,
                                l_a::Int64,j_a::Int64,m_a::Int64,R_a::Float64,dR_a::Float64,
                                l_b::Int64,j_b::Int64,m_b::Int64,R_b::Float64,dR_b::Float64)

    # Initialize the vlue of 1-body current J ...
    J_x, J_y, J_z = 0.0, 0.0, 0.0

    # Bra term ...
    Amp_1 = fCG(2*l_a,1,j_a,m_a-1,1,m_a) * fCG(2*l_b,1,j_b,m_b-1,1,m_b)

    if abs(Amp_1) > 1e-12

        # First bracket ...
        if l_a > 0
            Y = scalar_spherical_harmonic_function(l_b,div(m_b-1,2),theta,phi)

            # 1st term ...
            (Y_x,Y_y,Y_z) = vector_spherical_harmonic_function(l_a,l_a-1,-div(m_a-1,2),theta,phi)

            J = sqrt(Float64(l_a) / Float64(2*l_a + 1)) * (dR_a + Float64(l_a + 1) / r * R_a) * R_b * Y * Amp_1 * phase(div(m_a-1,2))

            J_x += J * Y_x
            J_y += J * Y_y
            J_z += J * Y_z

            # 2nd term ...
            (Y_x,Y_y,Y_z) = vector_spherical_harmonic_function(l_a,l_a+1,-div(m_a-1,2),theta,phi)

            J = -sqrt(Float64(l_a - 1) / Float64(2*l_a + 1)) * (dR_a - Float64(l_a) / r * R_a) * R_b * Y * Amp_1 * phase(div(m_a-1,2))

            J_x += J * Y_x
            J_y += J * Y_y
            J_z += J * Y_z
        end


        # Second bracket ...
        if l_b > 0
            Y = phase(div(m_a-1,2)) * scalar_spherical_harmonic_function(l_a,-div(m_a-1,2),theta,phi)

            # 1st term ...
            (Y_x,Y_y,Y_z) = vector_spherical_harmonic_function(l_b,l_b-1,div(m_b-1,2),theta,phi)

            J = -sqrt(Float64(l_b) / Float64(2*l_b + 1)) * (dR_b + Float64(l_b + 1) / r * R_b) * R_a * Y * Amp_1

            J_x += J * Y_x
            J_y += J * Y_y
            J_z += J * Y_z

            # 2nd term ...
            (Y_x,Y_y,Y_z) = vector_spherical_harmonic_function(l_b,l_b+1,div(m_b-1,2),theta,phi)

            J = sqrt(Float64(l_b - 1) / Float64(2*l_b + 1)) * (dR_b - Float64(l_b) / r * R_b) * R_a * Y * Amp_1

            J_x += J * Y_x
            J_y += J * Y_y
            J_z += J * Y_z
        end

    end

    # Ket term ...
    Amp_2 = fCG(2*l_a,1,j_a,m_a+1,-1,m_a) * fCG(2*l_b,1,j_b,m_b+1,-1,m_b)

    if abs(Amp_2) > 1e-12

        # First bracket ...
        if l_a > 0
            Y = scalar_spherical_harmonic_function(l_b,div(m_b+1,2),theta,phi)

            # 1st term ...
            (Y_x,Y_y,Y_z) = vector_spherical_harmonic_function(l_a,l_a-1,-div(m_a+1,2),theta,phi)

            J = sqrt(Float64(l_a) / Float64(2*l_a + 1)) * (dR_a + Float64(l_a + 1) / r * R_a) * R_b * Y * Amp_2 * phase(div(m_a+1,2))

            J_x += J * Y_x
            J_y += J * Y_y
            J_z += J * Y_z

            # 2nd term ...
            (Y_x,Y_y,Y_z) = vector_spherical_harmonic_function(l_a,l_a+1,-div(m_a+1,2),theta,phi)

            J = -sqrt(Float64(l_a - 1) / Float64(2*l_a + 1)) * (dR_a - Float64(l_a) / r * R_a) * R_b * Y * Amp_2 * phase(div(m_a+1,2))

            J_x += J * Y_x
            J_y += J * Y_y
            J_z += J * Y_z
        end


        # Second bracket ...
        if l_b > 0
            Y = phase(div(m_a+1,2)) * scalar_spherical_harmonic_function(l_a,-div(m_a+1,2),theta,phi)

            # 1st term ...
            (Y_x,Y_y,Y_z) = vector_spherical_harmonic_function(l_b,l_b-1,div(m_b+1,2),theta,phi)

            J = -sqrt(Float64(l_b) / Float64(2*l_b + 1)) * (dR_b + Float64(l_b + 1) / r * R_b) * R_a * Y * Amp_2

            J_x += J * Y_x
            J_y += J * Y_y
            J_z += J * Y_z

            # 2nd term ...
            (Y_x,Y_y,Y_z) = vector_spherical_harmonic_function(l_b,l_b+1,div(m_b+1,2),theta,phi)

            J = sqrt(Float64(l_b - 1) / Float64(2*l_b + 1)) * (dR_b - Float64(l_b) / r * R_b) * R_a * Y * Amp_2

            J_x += J * Y_x
            J_y += J * Y_y
            J_z += J * Y_z
        end

    end

    return (J_x,J_y,J_z)
end

@inline function J1b_spin(r::Float64,theta::Float64,phi::Float64,
                          l_a::Int64,j_a::Int64,m_a::Int64,R_a::Float64,dR_a::Float64,
                          l_b::Int64,j_b::Int64,m_b::Int64,R_b::Float64,dR_b::Float64)

    # Initialize the vlue of 1-body spin current J ...
    J_x, J_y, J_z = 0.0, 0.0, 0.0

    # Calculate the difference of input projection numbers ...
    M = div(m_b - m_a,2)

    # Calculate the outer common amplitudes factor ...
    Amp = phase(l_a + div(j_b + m_a,2)) * sqrt(Float64((j_a + 1) * (j_b + 1) * (2*l_a + 1) * (2*l_b + 1)) * 3.0 / (8.0 * pi))

    # Evaluate the sum over L, J  & individual contributions ...
    @inbounds for L in abs(l_a - l_b):(l_a + l_b)
        C_la_lb_L = fCG(2*l_a,2*l_b,2*L,0,0,0)
        @inbounds for J in div(abs(j_a - j_b),2):div(j_a + j_b,2)

            if abs(M) > J
                continue
            end

            Amp_JL = Amp * C_la_lb_L * phase(J) * fCG(j_a,j_b,2*J,-m_a,m_b,2*M) * f9j(2*l_a,j_a,1,2*l_b,j_b,1,2*L,2*J,2)

            if abs(Amp_JL) < 1e-12
                continue
            end

            # Term 1 ... J = L - 1
            if J == (L - 1) && J > 0
                (Y_x,Y_y,Y_z) = vector_spherical_harmonic_function(J,J,M,theta,phi)

                ME = Amp_JL * sqrt(Float64(J) / Float64(2*J + 1)) * (dR_a * R_b + R_a * dR_b + Float64(J + 2) / r * R_a * R_b)
                
                J_x += ME * Y_x
                J_y += ME * Y_y
                J_z += ME * Y_z
            end

            # Term 2 ... J = L
            if J == L && J > 0
                (Y_x,Y_y,Y_z) = vector_spherical_harmonic_function(J,J+1,M,theta,phi)

                ME = Amp_JL * sqrt(Float64(J) / Float64(2*J + 1)) * (dR_a * R_b + R_a * dR_b - Float64(J) / r * R_a * R_b)

                J_x += ME * Y_x
                J_y += ME * Y_y
                J_z += ME * Y_z
            end

            # Term 3 ... J = L
            if J == L
                (Y_x,Y_y,Y_z) = vector_spherical_harmonic_function(J,J-1,M,theta,phi)

                ME = Amp_JL * sqrt(Float64(J + 1) / Float64(2*J + 1)) * (dR_a * R_b + R_a * dR_b + Float64(J + 1) / r * R_a * R_b)

                J_x += ME * Y_x
                J_y += ME * Y_y
                J_z += ME * Y_z
            end

            # Term 4 ... J = L + 1
            if J == (L + 1)
                (Y_x,Y_y,Y_z) = vector_spherical_harmonic_function(J,J,M,theta,phi)

                ME = Amp_JL * sqrt(Float64(J + 1) / Float64(2*J + 1)) * (dR_a * R_b + R_a * dR_b - Float64(J - 1) / r * R_a * R_b)

                J_x += ME * Y_x
                J_y += ME * Y_y
                J_z += ME * Y_z
            end

        end
    end

    return (J_x,J_y,J_z)
end