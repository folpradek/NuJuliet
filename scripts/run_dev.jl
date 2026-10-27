using DelimitedFiles, LinearAlgebra, CGcoefficient, BenchmarkTools, Hungarian, Mmap, Printf, MKL

include("../src/import_structures.jl")
include("../src/common/functions.jl")
include("../src/common/orb.jl")
include("../src/common/O1b.jl")
include("../src/common/T1b.jl")
include("../src/common/Tr1b.jl")
include("../src/common/OBDM.jl")
include("../src/common/T2b.jl")
include("../src/common/O2b.jl")
include("../src/common/V2b.jl")
include("../src/common/V3b_no2b.jl")
include("../src/common/qpO2b.jl")
include("../src/common/qpO2b_40.jl")
include("../src/common/qpO2b_31.jl")
include("../src/common/qpO2b_22.jl")

 # HF + MBPT ...
include("../src/HF/HF.jl")
include("../src/HF/HF_solver.jl")
include("../src/HF/HF_allocate.jl")
include("../src/HF/HF_density_operator.jl")
include("../src/HF/HF_orbitals.jl")
include("../src/HF/HF_potential.jl")
include("../src/HF/HF_energy.jl")
include("../src/HF/HF_export.jl")
include("../src/HF/HF_MBPT.jl")

# HF-RPA ...
include("../src/RPA/RPA.jl")
include("../src/RPA/RPA_solver.jl")
include("../src/RPA/RPA_orbitals.jl") 
include("../src/RPA/RPA_allocate.jl")
include("../src/RPA/RPA_spurious.jl")
include("../src/RPA/RPA_diagonalize.jl")
include("../src/RPA/RPA_rho.jl")
include("../src/RPA/RPA_sigma.jl")
include("../src/RPA/RPA_energy.jl")
include("../src/RPA/RPA_transitions.jl")
include("../src/RPA/RPA_export.jl")

# HF-RRPA
include("../src/HF_RRPA/HF_RRPA.jl")
include("../src/HF_RRPA/HF_RRPA_Solver.jl")
include("../src/HF_RRPA/HF_RRPA_Phonon_Count.jl")
include("../src/HF_RRPA/HF_RRPA_Hamiltonian.jl")
include("../src/HF_RRPA/HF_RRPA_OBDM.jl")
include("../src/HF_RRPA/HF_RRPA_Allocate.jl")
include("../src/HF_RRPA/HF_RRPA_Diagonalize.jl")
include("../src/HF_RRPA/HF_RRPA_Spurious.jl")
include("../src/HF_RRPA/HF_RRPA_Transitions.jl")
include("../src/HF_RRPA/HF_RRPA_Energy.jl")
include("../src/HF_RRPA/HF_RRPA_Export.jl")

# HF-ScRPA ...
#=
include("../src/Temp/HF_ScRPA/HF_ScRPA_TBDM.jl")
include("../src/Temp/HF_ScRPA/HF_ScRPA_Solver.jl")
include("../src/Temp/HF_ScRPA/HF_ScRPA.jl")
include("../src/Temp/HF_ScRPA/HF_ScRPA_Allocate.jl")
include("../src/Temp/HF_ScRPA/HF_ScRPA_Diagonalize.jl")
=#


# BCS
include("../src/BCS/BCS.jl")
include("../src/BCS/BCS_solver.jl")
include("../src/BCS/BCS_V2b.jl")
include("../src/BCS/BCS_density_operator.jl")
include("../src/BCS/BCS_chemical_potential.jl")
include("../src/BCS/BCS_particle_number.jl")
include("../src/BCS/BCS_allocate.jl")
include("../src/BCS/BCS_energy.jl")
include("../src/BCS/BCS_export.jl")
include("../src/BCS/BCS_BMBPT.jl")

# HFB
include("../src/HFB/HFB.jl")
include("../src/HFB/HFB_solver.jl")
include("../src/HFB/HFB_density_operator.jl")
include("../src/HFB/HFB_chemical_potential.jl")
include("../src/HFB/HFB_particle_number.jl")
include("../src/HFB/HFB_Lipkin_Nogami.jl")
include("../src/HFB/HFB_Broyden.jl")
include("../src/HFB/HFB_ordering.jl")
include("../src/HFB/HFB_allocate.jl")
include("../src/HFB/HFB_canonical_basis.jl")
include("../src/HFB/HFB_energy.jl")
include("../src/HFB/HFB_export.jl")
include("../src/HFB/HFB_BMBPT.jl")

# QTDA solver ...
include("../src/QTDA/QTDA.jl")
include("../src/QTDA/QTDA_solver.jl")
include("../src/QTDA/QTDA_allocate.jl")
include("../src/QTDA/QTDA_spurious.jl")
include("../src/QTDA/QTDA_diagonalize.jl")
include("../src/QTDA/QTDA_transitions.jl")
include("../src/QTDA/QTDA_export.jl")

# QRPA solver ...
include("../src/QRPA/QRPA.jl")
include("../src/QRPA/QRPA_solver.jl")
include("../src/QRPA/QRPA_allocate.jl")
include("../src/QRPA/QRPA_spurious.jl")
include("../src/QRPA/QRPA_diagonalize.jl")
include("../src/QRPA/QRPA_A_Phonon.jl")
include("../src/QRPA/QRPA_transitions.jl")
include("../src/QRPA/QRPA_energy.jl")
include("../src/QRPA/QRPA_OBDM.jl")
include("../src/QRPA/QRPA_export.jl")

include("../src/HF_RPA/HF_RPA_External.jl")
include("../src/HF_RRPA/HF_RRPA_External.jl")


function main_run_dev()

    IntParams = Interaction_Parameters(
                NN_File = "io/NN.bin",
                NNN_File = "io/NNN.bin",
                hw = 16.0,
                Nmax = 3,
                N2max = 6,
                N3max = 9,
                #cRes = 1.0,
                #cP2N = 1.0,
                #cP3N = 1.0,
                nG2N = 0.0,
                pG2N = 0.0
                )

    # Calculation parameters structure ...
    CalcParams = Calculation_Parameters(
                A = 16,
                Z = 8,
                hw = 16.0,
                Nmax = 3,
                N2max = 6,
                N3max = 9,
                CMS = "CMS1+2B",
                Path = "HF_RPA_A16_Z8_hw16.0_Nmax3_N2max6_N3max9",
                HF = HF_Parameters(Tol = 1e-8, Imax = 150, BMF = true, ESH = true, MBPT = HF_MBPT_Parameters(Export = true, TBCF = true, MFP = true)),
                RPA = RPA_Parameters(Reference = "HF-MBPT", Name = "HF-MBPT-RPA", Transitions = "Linear", Ortho = true, TBCF = true, cB = 1.0)
                )

    #HF(Parameters(IntParams,CalcParams))

    RPA(Parameters(IntParams,CalcParams))

    #HF_RPA(Parameters(IntParams,CalcParams))


    CalcParams = Calculation_Parameters(
                A = 18,
                Z = 8,
                hw = 16.0,
                Nmax = 3,
                N2max = 6,
                N3max = 9,
                Path = "HFB_A18_Z8_hw16.0_Nmax3_N2max6_N3max9",
                CMS = "CMS1+2B",
                HFB = HFB_Parameters(Tol = 1e-8, Imax = 500, Pairing = "Full", dLmax = 0.5, BMF = true, LNT = false, HRF = false),
                QTDA = QTDA_Parameters(Ortho = true, Reference = "HFB"),
                QRPA = QRPA_Parameters(Ortho = true, Reference = "HFB"),
                )

    # Calculation call ...
    #HFB(Parameters(IntParams,CalcParams))

    # QTDA calculation call ...
    #QTDA(Parameters(IntParams,CalcParams))
    
    # QRPA calculation call ...
    #QRPA(Parameters(IntParams,CalcParams))


    CalcParams = Calculation_Parameters(
                A = 18,
                Z = 8,
                hw = 16.0,
                Nmax = 3,
                N2max = 6,
                N3max = 9,
                CMS = "CMS1+2B",
                Path = "ScBCS_A18_Z8_hw16.0_Nmax3_N2max6_N3max9",
                #HF = HF_Parameters(Tol = 1e-7, IMax = 100),
                BCS = BCS_Parameters(ScBCS = true, Tol = 1e-8, Imax = 500, q = 0.05, pD0 = 0.5, nD0 = 0.5, BMF = true, HRF = false),
                QTDA = QTDA_Parameters(Ortho = true, Reference = "BCS"),
                QRPA = QRPA_Parameters(Ortho = true, Reference = "BCS"),
                )

    # HF-BCS calculation call ...
    #BCS(Parameters(IntParams,CalcParams))

    # QTDA calculation call ...
    #QTDA(Parameters(IntParams,CalcParams))
    
    # QRPA calculation call ...
    #QRPA(Parameters(IntParams,CalcParams))

    return
end

@time main_run_dev()
