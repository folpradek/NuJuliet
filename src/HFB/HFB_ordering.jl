function HFB_orbital_ordering(Params::Parameters,SQE::pnVector,U::O1B,V::O1B,Orb::Vector{Orb1B};Final_Ordering::Bool=false,C::O1B=O1B(zeros(Float64,1,1),zeros(Float64,1,1)))
    # Read parameters ...
    N_max = Params.Calc.Nmax
    a_max = div((N_max + 1) * (N_max + 2), 2)

    # Read input arrays ...
    pU, pV, pSQE =  deepcopy(U.p), deepcopy(V.p), deepcopy(SQE.p)
    nU, nV, nSQE =  deepcopy(U.n), deepcopy(V.n), deepcopy(SQE.n)

    # Final reordering by occupation numbers ...
    if Final_Ordering == true && Params.Calc.HFB.Pairing != "BCS"
        # Reorder ... descending in U ...
        pU_norm, nU_norm = zeros(Float64,a_max), zeros(Float64,a_max)

        @inbounds for a in 1:a_max
            pU_norm[a] = @views norm(pU[:,a])^2
            nU_norm[a] = @views norm(nU[:,a])^2
        end

        pOrbs_sort = sortperm(pU_norm)
        nOrbs_sort = sortperm(nU_norm)

        # Proton single-quasiparticle orbitals ...
        @views pSQE .= pSQE[pOrbs_sort]
        @views pU .= pU[:,pOrbs_sort]
        @views pV .= pV[:,pOrbs_sort]

        # Neutron single-quasiparticle orbitals ...
        @views nSQE .= nSQE[nOrbs_sort]
        @views nU .= nU[:,nOrbs_sort]
        @views nV .= nV[:,nOrbs_sort]
    end

    # Find the ordering for single-quasiparticle orbitals ... vectors (U,V) are matched to (l,j)-blocks ...
    pOrb_order, pUV = HFB_orbital_matching(vcat(pU,pV),Orb)
    nOrb_order, nUV = HFB_orbital_matching(vcat(nU,nV),Orb)

    # Perform reordering of single-quasiparticle orbitals ...
    pSQE .= pSQE[pOrb_order]
    pU .= pUV[1:a_max,pOrb_order]
    pV .= pUV[(a_max+1):(2*a_max),pOrb_order]

    nSQE .= nSQE[nOrb_order]
    nU .= nUV[1:a_max,nOrb_order]
    nV .= nUV[(a_max+1):(2*a_max),nOrb_order]

    return pnVector(pSQE,nSQE), O1B(pU,nU), O1B(pV,nV)
end

function HFB_canonical_orbital_ordering(Params::Parameters,Occ::pnVector,C::O1B,Orb::Vector{Orb1B})
    # Read needed arrays ...
    pOcc, nOcc = deepcopy(Occ.p), deepcopy(Occ.n)

    # Find the ordering for single-particle orbitals ... vectors C are matched to (l,j)-blocks ...
    pOrb_order, pC = HFB_orbital_matching(C.p,Orb)
    nOrb_order, nC = HFB_orbital_matching(C.n,Orb)

    # Perform reordering of single-particle orbitals ...
    pOcc .= pOcc[pOrb_order]
    nOcc .= nOcc[nOrb_order]
    pC .= pC[:,pOrb_order]
    nC .= nC[:,nOrb_order]

    return pC, nC, pOcc, nOcc
end

function HFB_canonical_basis_orbital_ordering(Params::Parameters,C::O1B,Orb::Vector{Orb1B})
    # Find the ordering for single-particle orbitals ... vectors C are matched to (l,j)-blocks ...
    pOrb_order, pC = HFB_orbital_matching(C.p,Orb)
    nOrb_order, nC = HFB_orbital_matching(C.n,Orb)

    # Perform reordering of single-particle orbitals ...
    pC .= pC[:,pOrb_order]
    nC .= nC[:,nOrb_order]

    return O1B(pC,nC)
end

function HFB_orbital_matching(X::Matrix{Float64},Orb::Vector{Orb1B})
    # Match the eigenvectors X[:,a] (columns) to the LHO orbitals b ... Order[b] = a ...
    #   Rows of X are LHO components ... X = C (a_max rows) or X = [U; V] (2 * a_max rows) ...
    #   Within each (l,j)-block the vectors keep their input order ... the 1st vector of the block goes to the 1st orbital
    #   of the block, the 2nd vector to the 2nd orbital, ... (the same convention as before) ...
    #
    #   The matrices (Rho, H, HFB matrix) are block-diagonal in (l,j), but if two eigenvalues from different blocks are
    #   (numerically) degenerate, the dense eigensolver returns arbitrary mixtures of the two block states ... such vectors
    #   are assigned to blocks by the maximal total weight (Hungarian algorithm) & projected back to their (l,j)-block ...
    a_max = length(Orb)
    Row_max = size(X,1)
    Tol_pure = 1e-10

    # Define local function for l & j blocks ...
    function HFB_orbital_blocks(Orb::Vector{Orb1B})
        # Label the (l,j)-blocks of the LHO basis ... Block[a] is the block of orbital a ... blocks numbered by first appearance ...
        a_max = length(Orb)
        Block, B_max = zeros(Int64,a_max), 0

        @inbounds for a in 1:a_max
            # Check if the (l,j)-block of orbital a was already labeled ...
            @inbounds for b in 1:(a-1)
                if (Orb[b].l == Orb[a].l) && (Orb[b].j == Orb[a].j)
                    Block[a] = Block[b]
                    break
                end
            end

            # New (l,j)-block ...
            if Block[a] == 0
                B_max += 1
                Block[a] = B_max
            end
        end

        return Block, B_max
    end

    # Label the (l,j)-blocks ...
    Block, B_max = HFB_orbital_blocks(Orb)

    # Evaluate weights W[B,a] of vectors a in (l,j)-blocks B ...
    W = zeros(Float64,B_max,a_max)
    @inbounds for a in 1:a_max
        @inbounds for r in 1:Row_max
            b = rem(r - 1,a_max) + 1
            W[Block[b],a] += X[r,a]^2
        end
        @views W[:,a] ./= sum(W[:,a])
    end

    # Assign each vector to the (l,j)-block with its largest weight ...
    Vector_block = zeros(Int64,a_max)
    @inbounds for a in 1:a_max
        Vector_block[a] = argmax(W[:,a])
    end

    # Check that each block got exactly as many vectors as it has orbitals ... otherwise use the Hungarian algorithm ...
    Block_size = [count(==(B),Block) for B in 1:B_max]
    Block_count = [count(==(B),Vector_block) for B in 1:B_max]
    if Block_size != Block_count
        # Cost of putting vector a to orbital b ...
        Cost = zeros(Float64,a_max,a_max)
        @inbounds for b in 1:a_max
            @inbounds for a in 1:a_max
                Cost[b,a] = 1.0 - W[Block[b],a]
            end
        end

        # Apply the Hungarian algorithm ... Assignment[b] = a ...
        Assignment, _ = hungarian(Cost)
        @inbounds for b in 1:a_max
            Vector_block[Assignment[b]] = Block[b]
        end
    end

    # Find the ordering ... vectors of block B (in input order) go to orbitals of block B (in LHO order) ...
    Order = zeros(Int64,a_max)
    @inbounds for B in 1:B_max
        Block_orbitals = findall(==(B),Block)
        Block_vectors = findall(==(B),Vector_block)
        Order[Block_orbitals] .= Block_vectors
    end

    # Project the mixed vectors back to their (l,j)-block ... pure vectors are kept untouched ...
    X_new = copy(X)
    @inbounds for B in 1:B_max
        Block_vectors = findall(==(B),Vector_block)

        # All vectors of the block are pure ... nothing to do ...
        if all(W[B,a] > (1.0 - Tol_pure) for a in Block_vectors)
            continue
        end

        # Rows of the block B & the rest ...
        Block_rows = [r for r in 1:Row_max if Block[rem(r - 1,a_max) + 1] == B]
        Other_rows = [r for r in 1:Row_max if Block[rem(r - 1,a_max) + 1] != B]

        # Remove components outside the block & orthonormalize (Lowdin) ... P_new = P * (P' * P)^(-1/2) ...
        P = X[Block_rows,Block_vectors]
        S = Symmetric(P' * P)
        X_new[Block_rows,Block_vectors] .= P * inv(sqrt(S))
        X_new[Other_rows,Block_vectors] .= 0.0
    end

    return Order, X_new
end
