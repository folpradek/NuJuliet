module NuJuliet
    # Import all needed packages ...
    using BenchmarkTools, DelimitedFiles, LinearAlgebra, CGcoefficient, Printf

    # Matrix elements & general functions ...
    include("import_structures.jl")
    include("common/functions.jl")
    include("common/orb.jl")
    include("common/O1b.jl")
    include("common/T1b.jl")
    include("common/Tr1b.jl")
    include("common/rho.jl")
    include("common/T2b.jl")
    include("common/O2b.jl")
    include("common/V2b.jl")
    include("common/V3b_no2b.jl")
    include("common/qpO2b.jl")
    include("common/qpO2b_40.jl")
    include("common/qpO2b_31.jl")
    include("common/qpO2b_22.jl")
    include("common/Hamiltonian.jl")

    # HF + HF-MBPT ...
    include("HF/HF.jl")
    include("HF/HF_solver.jl")
    include("HF/HF_allocate.jl")
    include("HF/HF_density_operator.jl")
    include("HF/HF_orbitals.jl")
    include("HF/HF_potential.jl")
    include("HF/HF_energy.jl")
    include("HF/HF_export.jl")
    include("HF/HF_MBPT.jl")

    # TDA ...
    include("TDA/TDA.jl")
    include("TDA/TDA_solver.jl")
    include("TDA/TDA_allocate.jl")
    include("TDA/TDA_spurious.jl")
    include("TDA/TDA_diagonalize.jl")
    include("TDA/TDA_transitions.jl")
    include("TDA/TDA_external.jl")
    include("TDA/TDA_export.jl")

    # RPA ...
    include("RPA/RPA.jl")
    include("RPA/RPA_solver.jl")
    include("RPA/RPA_orbitals.jl") 
    include("RPA/RPA_Hamiltonian.jl")
    include("RPA/RPA_allocate.jl")
    include("RPA/RPA_spurious.jl")
    include("RPA/RPA_diagonalize.jl")
    include("RPA/RPA_rho.jl")
    include("RPA/RPA_sigma.jl")
    include("RPA/RPA_energy.jl")
    include("RPA/RPA_transitions.jl")
    include("RPA/RPA_external.jl")
    include("RPA/RPA_export.jl")

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

    # HFB ...
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

    # QTDA ...
    include("QTDA/QTDA.jl")
    include("QTDA/QTDA_solver.jl")
    include("QTDA/QTDA_allocate.jl")
    include("QTDA/QTDA_spurious.jl")
    include("QTDA/QTDA_diagonalize.jl")
    include("QTDA/QTDA_transitions.jl")
    include("QTDA/QTDA_external.jl")
    include("QTDA/QTDA_export.jl")

    # pnQTDA ...
    include("pnQTDA/pnQTDA.jl")
    include("pnQTDA/pnQTDA_solver.jl")
    include("pnQTDA/pnQTDA_orbitals.jl")
    include("pnQTDA/pnQTDA_allocate.jl")
    include("pnQTDA/pnQTDA_diagonalize.jl")
    include("pnQTDA/pnQTDA_external.jl")
    include("pnQTDA/pnQTDA_export.jl")

    # QRPA ...
    include("QRPA/QRPA.jl")
    include("QRPA/QRPA_solver.jl")
    include("QRPA/QRPA_allocate.jl")
    include("QRPA/QRPA_spurious.jl")
    include("QRPA/QRPA_diagonalize.jl")
    include("QRPA/QRPA_A_Phonon.jl")
    include("QRPA/QRPA_energy.jl")
    include("QRPA/QRPA_OBDM.jl")
    include("QRPA/QRPA_transitions.jl")
    include("QRPA/QRPA_external.jl")
    include("QRPA/QRPA_export.jl")

    for n in names(@__MODULE__; all=true)
        if Base.isidentifier(n) && n ∉ (Symbol(@__MODULE__), :eval, :include)
            @eval export $n
        end
    end
end