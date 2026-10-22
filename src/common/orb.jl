function orbitals_make(Params::Parameters)
    # Read parameters ...
    A, Z = Float64(Params.Calc.A), Float64(Params.Calc.Z)
    N_max = Params.Int.Nmax
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

function orbitals_ph_list(J_max::Int64,N_Particle::pnInteger,Particle::pnSVector,N_Hole::pnInteger,Hole::pnSVector)
    N_Phonon = zeros(Int64,J_max+1,2)

    pParticleHole = zeros(Int64,J_max+1,2,N_Particle.p,N_Hole.p)
    nParticleHole = zeros(Int64,J_max+1,2,N_Particle.n,N_Hole.n)

    @inbounds for p in 1:N_Particle.p
        @inbounds for h in 1:N_Hole.p
            P = rem(Particle.p[p].l + Hole.p[h].l,2) + 1
            @inbounds for J in div(abs(Particle.p[p].j - Hole.p[h].j),2):div(Particle.p[p].j + Hole.p[h].j,2)
                N_Phonon[J+1,P] += 1
                pParticleHole[J+1,P,p,h] = N_Phonon[J+1,P]
            end
        end
    end
    
    @inbounds for p in 1:N_Particle.n
        @inbounds for h in 1:N_Hole.n
            P = rem(Particle.n[p].l + Hole.n[h].l,2) + 1
            @inbounds for J in div(abs(Particle.n[p].j - Hole.n[h].j),2):div(Particle.n[p].j + Hole.n[h].j,2)
                N_Phonon[J+1,P] += 1
                nParticleHole[J+1,P,p,h] = N_Phonon[J+1,P]
            end
        end
    end

    ParticleHole = pnArray(pParticleHole,nParticleHole)

    return ParticleHole
end

function orbitals_one_phonon_make(N_Particle::pnInteger,Particle::pnSVector,N_Hole::pnInteger,Hole::pnSVector)
    N_Phonon = 0

    for p in 1:N_Particle.p
        for h in 1:N_Hole.p
            for J in div(abs(Particle.p[p].j - Hole.p[h].j),2):div(Particle.p[p].j + Hole.p[h].j,2)
                N_Phonon += 1
            end
        end
    end
    
    for p in 1:N_Particle.n
        for h in 1:N_Hole.n
            for J in div(abs(Particle.n[p].j - Hole.n[h].j),2):div(Particle.n[p].j + Hole.n[h].j,2)
                N_Phonon += 1
            end
        end
    end

    Phonon = Vector{PhState}(undef,N_Phonon)

    N_ph = 0
    for p in 1:N_Particle.p
        for h in 1:N_Hole.p
            for J in div(abs(Particle.p[p].j - Hole.p[h].j),2):div(Particle.p[p].j + Hole.p[h].j,2)
                N_ph += 1
                Phonon[N_ph] = PhState(p,h,-1,J,rem(Particle.p[p].l + Hole.p[h].l,2)+1)
            end
        end
    end

    for p in 1:N_Particle.n
        for h in 1:N_Hole.n
            for J in div(abs(Particle.n[p].j - Hole.n[h].j),2):div(Particle.n[p].j + Hole.n[h].j,2)
                N_ph += 1
                Phonon[N_ph] = PhState(p,h,1,J,rem(Particle.n[p].l + Hole.n[h].l,2)+1)
            end
        end
    end

    return N_Phonon, Phonon
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
                    P = rem(l_a + l_b,2) + 1
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
                    P = rem(l_a + l_b,2) + 1
                    N_2qp[P,J+1] += 1
                end
            end
        end
    end

    # Initialize the matrix storage for 2qp orbitals ...
    Orb_2qp_list = Matrix{Vector{Ind2qp}}(undef,2,J_max+1)

    # Initialize the entries of the storage matrix for 2qp orbitals ...
    @inbounds for P in 1:2
        @inbounds for J in 0:J_max
            Orb_2qp_list[P,J+1] = Vector{Ind2qp}(undef,N_2qp[P,J+1])
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
                    P = rem(l_a + l_b,2) + 1
                    N_2qp[P,J+1] += 1
                    Orb_2qp_list[P,J+1][N_2qp[P,J+1]] = Ind2qp(a,b,-1)
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
                    P = rem(l_a + l_b,2) + 1
                    N_2qp[P,J+1] += 1
                    Orb_2qp_list[P,J+1][N_2qp[P,J+1]] = Ind2qp(a,b,1)
                end
            end
        end
    end

    Orb_2qp = qpOrb2B(N_2qp,Orb_2qp_list)

    return Orb_2qp
end

function OBDM_natural_orbitals(Params::Parameters,Orb::Vector{Orb1B},Rho::O1B)
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
        lSum, jSum = 0.0, 0.0
        @inbounds for b in 1:a_max
            l_b, j_b = Float64(Orb[b].l), Float64(Orb[b].j)
            lSum += abs2(pC[b,a]) * l_b
            jSum += abs2(pC[b,a]) * j_b
        end
        l_a, j_a = round(Int64,lSum), round(Int64,jSum)

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