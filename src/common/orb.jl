function orbitals_make(Params::Parameters;Int::Bool = false)
    # Read parameters ...
    A, Z = Float64(Params.Calc.A), Float64(Params.Calc.Z)
    N_max = Params.Calc.Nmax
    if Int; N_max = Params.Int.Nmax; end
    a_max = div((N_max + 1)*(N_max + 2),2)

    println("\nConstructing Linear-Harmonic Oscillator basis single-particle orbitals ...")

    # Make LHO single-particle orbitals ...
    Orb = Vector{Orb1B}(undef,a_max)
    Orb_arr = zeros(Int64,a_max,4)
    Obc_occ = zeros(Float64,a_max,2)
    a = 0

    @inbounds for N = 0:N_max
        @inbounds for l in 0:N
            n = Int64(floor((N - l)/2))
            if (2*n + l == N)
                @inbounds for j in Int64(abs(2*l - 1)):2:Int64(2*l + 1)
                    a += 1
                    Orb_arr[a,1] = a
                    Orb_arr[a,2] = n
                    Orb_arr[a,3] = l
                    Orb_arr[a,4] = j
                end
            end
        end
    end

    if abs(A) > 1e-8 || abs(Z) > 1e-8
        Z_occ, N_occ = 0.0, 0.0
        Z_fill, N_fill = true, true
        Z_mask, N_mask = falses(a_max), falses(a_max)

        while N_fill
            @inbounds for N = 0:N_max
                @inbounds for n in 0:div(N, 2)
                    l = N - 2*n
                    @inbounds for j = (2*l + 1):-2:abs(2*l-1)
                        j_hat = Float64(j + 1)
                        @inbounds for a = 1:a_max
                            if (Orb_arr[a,2] == n) && (Orb_arr[a,3] == l) && (Orb_arr[a,4] == j) && (N_mask[a] == false)
                                Occ = min(1.0, (A - Z - N_occ) / j_hat)
                                Obc_occ[a,2] = Occ
                                N_occ += j_hat * Occ
                                N_mask[a] = true
                                if abs(A - Z - N_occ) < 1e-8
                                    N_fill = false
                                end
                            end
                        end
                    end
                end
            end
        end

        while Z_fill
            @inbounds for N in 0:N_max
                @inbounds for n = 0:div(N, 2)
                    l = N - 2*n
                    @inbounds for j = (2*l + 1):-2:abs(2*l-1)
                        j_hat = Float64(j + 1)
                        @inbounds for a = 1:a_max
                            if (Orb_arr[a,2] == n) && (Orb_arr[a,3] == l) && (Orb_arr[a,4] == j) && (Z_mask[a] == false)
                                Occ = min(1.0, (Z - Z_occ) / j_hat)
                                Obc_occ[a,1] = Occ
                                Z_occ += j_hat * Occ
                                Z_mask[a] = true
                                if abs(Z - Z_occ) < 1e-8
                                    Z_fill = false
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    # Allocate the single-particle orbitals ...
    @inbounds for b = 1:a_max
        Orb[b] = Orb1B(Orb_arr[b,1],Orb_arr[b,2],Orb_arr[b,3],Orb_arr[b,4],Obc_occ[b,1],Obc_occ[b,2])
    end

    println("\nLinear-Harmonic Oscillator basis single-particle orbitals initialized ...")

    return Orb
end

function orbitals_update(Orb::Vector{Orb1B},Rho::O1B)
    # Determine the number of single-particle orbitals ...
    a_max = size(Orb)

    # Initialize new array for single-particle orbitals ...
    Orb_new = Vector{Orb1B}(undef,a_max)

    # Construct new updated single-particle orbitals ...
    @inbounds for Orb_a in Orb
        a_a, n_a, l_a, j_a = Orb_a.a, Orb_a.n, Orb_a.l, Orb_a.j
        pO_a, nO_a = Rho.p[a_a,a_a], Rho.n[a_a,a_a]
        Orb_new[a_a] = Orb1B(a_a,n_a,l_a,j_a,pO_a,nO_a)
    end

    # Update the single-particle orbitals ...
    Orb .= Orb_new

    return Orb
end

function orbitals_export(Params::Parameters,Orb::Vector{Orb1B},export_path::String)
    # Read parameters ...
    N_max = Params.Calc.Nmax
    a_max = div((N_max+1)*(N_max+2),2)

    # Export the given table of single-particle orbitals ...
    open(export_path, "w") do export_file
        @inbounds for i in 1:a_max
            a, n, l, j, pO, nO = Orb[i].a, Orb[i].n, Orb[i].l, Orb[i].j, Orb[i].pO, Orb[i].nO

            write(export_file, Int64(a))
            write(export_file, Int64(n))
            write(export_file, Int64(l))
            write(export_file, Int64(j))
            write(export_file, Float64(pO))
            write(export_file, Float64(nO))
        end
    end

    return
end

function orbitals_import(Params::Parameters,import_path::String)
    # Read parameters ...
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1)*(N_max + 2),2)

    # Initialize vector field for table of single-particle orbitals ...
    Orb = Vector{Orb1B}(undef,a_max)

    # Read the given table of single-particle orbitals ...
    open(import_path, "r") do read_file
        @inbounds for i in 1:a_max
            a = read(read_file,Int64)
            n = read(read_file,Int64)
            l = read(read_file,Int64)
            j = read(read_file,Int64)
            pO = read(read_file,Float64)
            nO = read(read_file,Float64)

            Orb[i] = Orb1B(a,n,l,j,pO,nO)
        end
    end

    return Orb
end

function orbitals_lj_make(Params::Parameters,Orb::Vector{Orb1B})
    # Parameter initialization...
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1)*(N_max + 2),2)

    Orb_lj = Matrix{Vector{Int64}}(undef,N_max+1,2*N_max+1)

    # Count the number of orbitals for each l,j pair ...
    @inbounds for l in 0:N_max
        @inbounds for j in 1:(2*N_max+1)
            lj_count = 0
            @inbounds for a in 1:a_max
                l_a, j_a = Orb[a].l, Orb[a].j
                if l == l_a && j == j_a
                    lj_count += 1
                end
            end
            Orb_lj[l+1,j] = zeros(Int64,lj_count)
        end
    end

    # Fill the Orb_lj array with orbital indices ...
    @inbounds for l in 0:N_max
        @inbounds for j in 1:(2*N_max+1)
            lj_count = 0
            @inbounds for a in 1:a_max
                l_a, j_a = Orb[a].l, Orb[a].j
                if l == l_a && j == j_a
                    lj_count += 1
                    Orb_lj[l+1,j][lj_count] = a
                end
            end
        end
    end

    return Orb_lj
end

function orbitals_export(Params::Parameters,Orb::Vector{Orb1B})
    N_max = Params.Calc.Nmax
    Output_File = Params.Calc.Path
    a_max = div((N_max+1)*(N_max+2),2)

    Orbitals_LHO_Export = "IO/" * Output_File * "/Orbitals_LHO.dat"
    Orbitals_HF_Export = "IO/" * Output_File * "/Orbitals_HF.dat"

    # Import HF s.p. energies ...
    pE_Import = "IO/" * Output_File * "/Bin/pE_HF.bin"
    pSPE = Vector{Float64}(undef,a_max)
    open(pE_Import, "r") do Read_File
        @inbounds for a in 1:a_max
            ME = read(Read_File, Float64)
            pSPE[a] = ME
        end
    end

    nE_Import = "IO/" * Output_File * "/Bin/nE_HF.bin"
    nSPE = Vector{Float64}(undef,a_max)
    open(nE_Import, "r") do Read_File
        @inbounds for a in 1:a_max
            ME = read(Read_File, Float64)
            nSPE[a] = ME
        end
    end

    # Export Single-Particle State Orbitals in the LHO basis ...
    println("\nExporting LHO basis s.p. orbitals in human-readable format ...")

    open(Orbitals_LHO_Export, "w") do Export_File
        println(Export_File, "a\tn\tl\t2j\tpOc\tnOc")
        @inbounds for a in 1:a_max
            Row = string(Orb[a].a) * "\t" * string(Orb[a].n) *
                  "\t" * string(Orb[a].l) * "\t" * string(Orb[a].j) *
                  "\t " * string(Orb[a].pO) * "\t " * string(Orb[a].nO)
            println(Export_File, Row)
        end
    
    end
    
    println("\nExported LHO basis s.p. orbitals in human-readable format ...")

    # Export Single-Particle State Orbitals in the HF basis ...
    println("\nExporting HF basis s.p. orbitals in human-readable format ...")

    open(Orbitals_HF_Export, "w") do Export_File
        println(Export_File, "a\tl\t2j\tpOc\tnOc\tE_p\t\t\tE_n")
        @inbounds for a in 1:a_max
            Row = string(Orb[a].a) * "\t" * string(Orb[a].l) * "\t" * string(Orb[a].j) *
                  "\t " * string(Orb[a].pO) * "\t " * string(Orb[a].nO) *
                  "\t" * string(pSPE[a]) * "\t" * string(nSPE[a])
            println(Export_File, Row)
        end
    end
    
    println("\nExported HF basis s.p. orbitals in human-readable format ...")

    return
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

function orbitals_1p1h_make(Params::Parameters,Orb::Vector{Orb1B};Tol_par::Float64=1e-4,Tol_hol::Float64=0.999)
    # Read parameters ...
    N_max = Params.Calc.Nmax
    J_max = 2*N_max + 1

    # Count the number of particle & hole orbitals ...
    N_ph = zeros(Int64,2,J_max+1)

    @inbounds for J in 0:J_max
        @inbounds for P in 1:2
            N_ph_JP = 0

            @inbounds for Orb_a in Orb
                l_a, j_a, pO_a, nO_a = Orb_a.l, Orb_a.j, Orb_a.pO, Orb_a.nO

                @inbounds for Orb_b in Orb
                    l_b, j_b, pO_b, nO_b = Orb_b.l, Orb_b.j, Orb_b.pO, Orb_b.nO

                    P_ab = parity(l_a + l_b)

                    if (P_ab != P) || (triangle(j_a,j_b,2*J) == false)
                        continue
                    end

                    if pO_b > Tol_hol && pO_a < Tol_par && (pO_b - pO_a) > 1e-8
                        N_ph_JP += 1
                    end

                    if nO_b > Tol_hol && nO_a < Tol_par && (nO_b - nO_a) > 1e-8
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

                    P_ab = parity(l_a + l_b)

                    if (P_ab != P) || (triangle(j_a,j_b,2*J) == false)
                        continue
                    end

                    if pO_b > Tol_hol && pO_a < Tol_par && (pO_b - pO_a) > 1e-8
                        N_ph_JP += 1

                        Ind_ph[P,J+1][N_ph_JP] = Ind2B(a,b,-1)

                        Orb_ph_ab = orb2b_1p1h_key(a,b,J,P,-1)
                        Dic_ph[Orb_ph_ab] = N_ph_JP
                    end

                    if nO_b > Tol_hol && nO_a < Tol_par && (nO_b - nO_a) > 1e-8
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

@inline function orb2b_2qp_key(a::Int64,b::Int64,J::Int64,P::Int64,T::Int64)
    return UInt64(P) |
          (UInt64(T + 1) << 2) |
          (UInt64(J) << 4) |
          (UInt64(a) << 10) |
          (UInt64(b) << 18)
end

@inline function orb2b_2qp_index(a::Int64,b::Int64,J::Int64,P::Int64,T::Int64,Orb_2qp::Orb2B_2qp)
    key = orb2b_2qp_key(a,b,J,P,T)
    dict = Orb_2qp.Dic
    orb = get(dict,key,0)
    return orb
end

function orbitals_2qp_make(Params::Parameters,Orb::Vector{Orb1B})
    # Read parameters ...
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1)*(N_max + 2),2)
    J_max = 2*N_max + 1

    # Count the total number of 2qp orbitals for each J & P ... a => b convention ...
    N_2qp = zeros(Int64,2,J_max+1)

    # pp counts ...
    @inbounds for a in 1:a_max
        l_a ,j_a = Orb[a].l , Orb[a].j
        @inbounds for b in 1:a
            l_b ,j_b = Orb[b].l , Orb[b].j
            @inbounds for J in div(abs(j_a - j_b),2):div(j_a + j_b,2)
                if a != b || (a == b && rem(J,2) == 0)
                    P = parity(l_a + l_b)
                    N_2qp[P,J+1] += 1
                end
            end
        end
    end

    # nn counts ...
    @inbounds for a in 1:a_max
        l_a ,j_a = Orb[a].l , Orb[a].j
        @inbounds for b in 1:a
            l_b ,j_b = Orb[b].l , Orb[b].j
            @inbounds for J in div(abs(j_a - j_b),2):div(j_a + j_b,2)
                if a != b || (a == b && rem(J,2) == 0)
                    P = parity(l_a + l_b)
                    N_2qp[P,J+1] += 1
                end
            end
        end
    end

    # Initialize the matrix storage for 2qp orbitals ...
    Orb_2qp_list = Matrix{Vector{Ind2B}}(undef,2,J_max+1)

    # Initialize mapping dictionary for 2qp orbitals ...
    Dic_2qp = Dict{UInt64,Int64}()

    # Initialize the entries of the storage matrix for 2qp orbitals ...
    @inbounds for P in 1:2
        @inbounds for J in 0:J_max
            Orb_2qp_list[P,J+1] = Vector{Ind2B}(undef,N_2qp[P,J+1])
            N_2qp[P,J+1] = 0
        end
    end

    # Allocate the indices for 2qp orbitals ...
        # pp entries ...
    @inbounds for a in 1:a_max
        l_a ,j_a = Orb[a].l, Orb[a].j
        @inbounds for b in 1:a
            l_b ,j_b = Orb[b].l, Orb[b].j
            @inbounds for J in div(abs(j_a - j_b),2):div(j_a + j_b,2)
                if a != b || (a == b && rem(J,2) == 0)
                    P = parity(l_a + l_b)
                    N_2qp[P,J+1] += 1
                    Orb_2qp_list[P,J+1][N_2qp[P,J+1]] = Ind2B(a,b,-1)

                    Orb_2qp_ab = orb2b_2qp_key(a,b,J,P,-1)
                    Dic_2qp[Orb_2qp_ab] = N_2qp[P,J+1]
                end
            end
        end
    end
        # nn entries ...
    @inbounds for a in 1:a_max
        l_a ,j_a = Orb[a].l, Orb[a].j
        @inbounds for b in 1:a
            l_b ,j_b = Orb[b].l, Orb[b].j
            @inbounds for J in div(abs(j_a - j_b),2):div(j_a + j_b,2)
                if a != b || (a == b && rem(J,2) == 0)
                    P = parity(l_a + l_b)
                    N_2qp[P,J+1] += 1
                    Orb_2qp_list[P,J+1][N_2qp[P,J+1]] = Ind2B(a,b,+1)

                    Orb_2qp_ab = orb2b_2qp_key(a,b,J,P,+1)
                    Dic_2qp[Orb_2qp_ab] = N_2qp[P,J+1]
                end
            end
        end
    end

    Orb_2qp = Orb2B_2qp(N_2qp,Orb_2qp_list,Dic_2qp)

    return Orb_2qp
end

# Move to OBDM.jl ?
function rho_natural_orbitals(Params::Parameters,Orb::Vector{Orb1B},Rho::O1B)
    # Read parameters ...
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1)*(N_max + 2),2)

    # Initialize Rho ...
    pRho = zeros(Float64,a_max,a_max)
    nRho = zeros(Float64,a_max,a_max)

    pRho .= Rho.p
    nRho .= Rho.n

    # Symmetrize, clean & regularize Rho ...
        # Symmetrize density matrices ...
    pRho .= 0.5 .* (pRho .+ pRho')
    nRho .= 0.5 .* (nRho .+ nRho')
        # Eliminate any possible numerical noise spoiling block-diagonal structure ...
    @inbounds for a in 1:a_max
        @inbounds for b in 1:a_max
            if (Orb[a].j != Orb[b].j) || (Orb[a].l != Orb[b].l)
                pRho[a,b] = 0.0
                nRho[a,b] = 0.0
            end
        end
    end
        # Add a tiny deterministic diagonal splitting to lift degeneracies ...
    @inbounds for a in 1:a_max
        pRho[a,a] += 1e-10 * Float64(a)
        nRho[a,a] += 1e-10 * Float64(a)
    end

    # Diagonalize Rho ...
    pOcc, pC = eigen(Symmetric(pRho))
    nOcc, nC = eigen(Symmetric(nRho))

    # Hungarian Algorithm based reordering of the Natural Orbitals basis ...
        # Initialize the cost matrix for Hungarian algorithm ...
        pCost = ones(Float64,a_max,a_max) .- abs.(pC)
        nCost = ones(Float64,a_max,a_max) .- abs.(nC)

        # Apply the Hungarian algorithm ....
        pReordering, pTemp = hungarian(pCost)
        nReordering, nTemp = hungarian(nCost)

        # Apply reordering ...
        pC .= @views pC[:,pReordering]
        nC .= @views nC[:,nReordering]
        pOcc .= pOcc[pReordering]
        nOcc .= nOcc[nReordering]

        # For convenience fix positive phases ...
        @inbounds for a in 1:a_max
            if pC[a,a] < 0.0
                pC[:,a] .*= -1.0
            end
            if nC[a,a] < 0.0
                nC[:,a] .*= -1.0
            end
        end

    # Update the density matrix in the new natural orbitals basis ...
    pRho .= diagm(pOcc)
    nRho .= diagm(nOcc)

    # Allocate orbitals matrix C & density matrix Rho as 1-body operator structures ...
    C , Rho = O1B(pC,nC), O1B(pRho,nRho)

    # Initialize new single-particle orbitals ...
    Orb_new = Vector{Orb1B}(undef,a_max)

    # Allocate new single-particle orbitals ...
    @inbounds for a in 1:a_max
        # Determine thel l & j quantum numbers of the new natural orbital ...
        plSum, pjSum, nlSum, njSum = 0.0, 0.0, 0.0, 0.0
        @inbounds for b in 1:a_max
            l_b, j_b = Float64(Orb[b].l), Float64(Orb[b].j)
            plSum += abs2(pC[b,a]) * l_b
            pjSum += abs2(pC[b,a]) * j_b
            nlSum += abs2(nC[b,a]) * l_b
            njSum += abs2(nC[b,a]) * j_b
        end
        l_a, j_a = round(Int64,plSum), round(Int64,pjSum)
        nl_a, nj_a = round(Int64,nlSum), round(Int64,njSum)
        if (l_a != nl_a) || (j_a != nj_a)
            error("rho_natural_orbitals: proton & neutron natural orbitals in column $a differ in (l,j) = ($l_a,$j_a) vs ($nl_a,$nj_a)!")
        end

        # Determine the radial quantum number n of the new natural orbital ...
        n_a = 0
        @inbounds for b in 1:1:(a-1)
            l_b, j_b = Orb[b].l, Orb[b].j
            if l_a == l_b && j_a == j_b
                n_a += 1
            end
        end

        # Read the new occupation numbers ...
        pO_a, nO_a = pOcc[a], nOcc[a]

        # Allocate the natural orbital ...
        Orb_new[a] = Orb1B(a,n_a,l_a,j_a,pO_a,nO_a)
    end

    return Orb_new, C, Rho
end