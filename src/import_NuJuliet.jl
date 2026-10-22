module NuJuliet
    # Import all needed packages ...
    using BenchmarkTools, DelimitedFiles, LinearAlgebra, CGcoefficient, Hungarian, Mmap, Printf

    # Matrix elements & general functions ...
    include("import_structures.jl")
    include("common/functions.jl")
    include("common/orb.jl")
    include("common/O1b.jl")
    include("common/T1b.jl")
    include("common/Tr1b.jl")
    include("common/OBDM.jl")
    include("common/T2b.jl")
    include("common/O2b.jl")
    include("common/V2b.jl")
    include("common/V3b_no2b.jl")
    include("common/qpO2b.jl")
    include("common/qpO2b_40.jl")
    include("common/qpO2b_31.jl")
    include("common/qpO2b_22.jl")

    # HF + MBPT ...
    include("HF/HF.jl")
    include("HF/HF_solver.jl")
    include("HF/HF_allocate.jl")
    include("HF/HF_density_operator.jl")
    include("HF/HF_orbitals.jl")
    include("HF/HF_potential.jl")
    include("HF/HF_energy.jl")
    include("HF/HF_export.jl")
    include("HF/HF_MBPT.jl")

    # HF-RPA ...
    include("HF_RPA/HF_RPA_Solver.jl")
    include("HF_RPA/HF_RPA.jl")
    include("HF_RPA/HF_RPA_Phonon_Count.jl") 
    include("HF_RPA/HF_RPA_Allocate.jl")
    include("HF_RPA/HF_RPA_Spurious.jl")
    include("HF_RPA/HF_RPA_Diagonalize.jl")
    include("HF_RPA/HF_RPA_Energy.jl")
    include("HF_RPA/HF_RPA_OBDM.jl")
    include("HF_RPA/HF_RPA_Transitions.jl")
    include("HF_RPA/HF_RPA_Export.jl")
    include("HF_RPA/HF_RPA_External.jl")

    # HF-Renormalized-RPA ...
    include("HF_RRPA/HF_RRPA.jl")
    include("HF_RRPA/HF_RRPA_Solver.jl")
    include("HF_RRPA/HF_RRPA_Phonon_Count.jl")
    include("HF_RRPA/HF_RRPA_Hamiltonian.jl")
    include("HF_RRPA/HF_RRPA_OBDM.jl")
    include("HF_RRPA/HF_RRPA_Allocate.jl")
    include("HF_RRPA/HF_RRPA_Diagonalize.jl")
    include("HF_RRPA/HF_RRPA_Spurious.jl")
    include("HF_RRPA/HF_RRPA_Transitions.jl")
    include("HF_RRPA/HF_RRPA_Energy.jl")
    include("HF_RRPA/HF_RRPA_Export.jl")
    include("HF_RRPA/HF_RRPA_External.jl")

    # BCS ...
    include("BCS/BCS.jl")
    include("BCS/BCS_solver.jl")
    include("BCS/BCS_V2b.jl")
    include("BCS/BCS_density_operator.jl")
    include("BCS/BCS_chemical_potential.jl")
    include("BCS/BCS_particle_number.jl")
    include("BCS/BCS_allocate.jl")
    include("BCS/BCS_energy.jl")
    include("BCS/BCS_BMBPT.jl")
    include("BCS/BCS_export.jl")

    # HFB solver ...
    include("HFB/HFB.jl")
    include("HFB/HFB_solver.jl")
    include("HFB/HFB_density_operator.jl")
    include("HFB/HFB_chemical_potential.jl")
    include("HFB/HFB_particle_number.jl")
    include("HFB/HFB_Lipkin_Nogami.jl")
    include("HFB/HFB_Broyden.jl")
    include("HFB/HFB_ordering.jl")
    include("HFB/HFB_allocate.jl")
    include("HFB/HFB_canonical_basis.jl")
    include("HFB/HFB_energy.jl")
    include("HFB/HFB_BMBPT.jl")
    include("HFB/HFB_export.jl")

    # QTDA solver ...
    include("QTDA/QTDA.jl")
    include("QTDA/QTDA_solver.jl")
    include("QTDA/QTDA_allocate.jl")
    include("QTDA/QTDA_spurious.jl")
    include("QTDA/QTDA_diagonalize.jl")
    include("QTDA/QTDA_transitions.jl")
    include("QTDA/QTDA_export.jl")

    # QRPA solver ...
    include("QRPA/QRPA.jl")
    include("QRPA/QRPA_solver.jl")
    include("QRPA/QRPA_allocate.jl")
    include("QRPA/QRPA_spurious.jl")
    include("QRPA/QRPA_diagonalize.jl")
    include("QRPA/QRPA_A_Phonon.jl")
    include("QRPA/QRPA_energy.jl")
    include("QRPA/QRPA_OBDM.jl")
    include("QRPA/QRPA_transitions.jl")
    include("QRPA/QRPA_export.jl")

    for n in names(@__MODULE__; all=true)
        if Base.isidentifier(n) && n ∉ (Symbol(@__MODULE__), :eval, :include)
            @eval export $n
        end
    end
    
end