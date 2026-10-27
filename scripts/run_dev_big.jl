using DelimitedFiles, LinearAlgebra, CGcoefficient, BenchmarkTools, Hungarian, Mmap, Printf

include("../Source/Import_Structures.jl")
include("../Source/Functions/Functions.jl")
include("../Source/MatrixElements/Orb.jl")
include("../Source/MatrixElements/O1B.jl")
include("../Source/MatrixElements/T1B.jl")
include("../Source/MatrixElements/Tr1B.jl")
include("../Source/MatrixElements/OBDM.jl")
include("../Source/MatrixElements/T2B.jl")
include("../Source/MatrixElements/O2B.jl")
include("../Source/MatrixElements/V2B.jl")
include("../Source/MatrixElements/V3B_NO2B.jl")
include("../Source/MatrixElements/qpO2B.jl")
include("../Source/MatrixElements/qpO2B_40.jl")
include("../Source/MatrixElements/qpO2B_31.jl")
include("../Source/MatrixElements/qpO2B_22.jl")

 # HF + MBPT ...
include("../Source/HF/HF.jl")
include("../Source/HF/HF_solver.jl")
include("../Source/HF/HF_allocate.jl")
include("../Source/HF/HF_density_operator.jl")
include("../Source/HF/HF_orbitals.jl")
include("../Source/HF/HF_potential.jl")
include("../Source/HF/HF_energy.jl")
include("../Source/HF/HF_export.jl")
include("../Source/HF/HF_MBPT.jl")

# HF-RPA ...
include("../Source/RPA/RPA.jl")
include("../Source/RPA/RPA_solver.jl")
include("../Source/RPA/RPA_orbitals.jl") 
include("../Source/RPA/RPA_allocate.jl")
include("../Source/RPA/RPA_spurious.jl")
include("../Source/RPA/RPA_diagonalize.jl")
include("../Source/RPA/RPA_sigma.jl")
include("../Source/RPA/RPA_energy.jl")
include("../Source/RPA/RPA_OBDM.jl")
include("../Source/RPA/RPA_transitions.jl")
include("../Source/RPA/RPA_export.jl")

# HF-RRPA
include("../Source/HF_RRPA/HF_RRPA.jl")
include("../Source/HF_RRPA/HF_RRPA_Solver.jl")
include("../Source/HF_RRPA/HF_RRPA_Phonon_Count.jl")
include("../Source/HF_RRPA/HF_RRPA_Hamiltonian.jl")
include("../Source/HF_RRPA/HF_RRPA_OBDM.jl")
include("../Source/HF_RRPA/HF_RRPA_Allocate.jl")
include("../Source/HF_RRPA/HF_RRPA_Diagonalize.jl")
include("../Source/HF_RRPA/HF_RRPA_Spurious.jl")
include("../Source/HF_RRPA/HF_RRPA_Transitions.jl")
include("../Source/HF_RRPA/HF_RRPA_Energy.jl")
include("../Source/HF_RRPA/HF_RRPA_Export.jl")

# HF-ScRPA ...
#=
include("../Source/Temp/HF_ScRPA/HF_ScRPA_TBDM.jl")
include("../Source/Temp/HF_ScRPA/HF_ScRPA_Solver.jl")
include("../Source/Temp/HF_ScRPA/HF_ScRPA.jl")
include("../Source/Temp/HF_ScRPA/HF_ScRPA_Allocate.jl")
include("../Source/Temp/HF_ScRPA/HF_ScRPA_Diagonalize.jl")
=#

#=
# BCS
include("../Source/BCS/BCS.jl")
include("../Source/BCS/BCS_Solver.jl")
include("../Source/BCS/BCS_Residual_Interaction.jl")
include("../Source/BCS/BCS_Density_Operator.jl")
include("../Source/BCS/BCS_Chemical_Potential.jl")
include("../Source/BCS/BCS_Particle_Number.jl")
include("../Source/BCS/BCS_Allocate.jl")
include("../Source/BCS/BCS_Energy.jl")
include("../Source/BCS/BCS_Export.jl")
include("../Source/BCS/BCS_BMBPT.jl")

# HFB
include("../Source/HFB/HFB.jl")
include("../Source/HFB/HFB_Solver.jl")
include("../Source/HFB/HFB_Density_Operator.jl")
include("../Source/HFB/HFB_Chemical_Potential.jl")
include("../Source/HFB/HFB_Particle_Number.jl")
include("../Source/HFB/HFB_Lipkin_Nogami.jl")
include("../Source/HFB/HFB_Broyden.jl")
include("../Source/HFB/HFB_Ordering.jl")
include("../Source/HFB/HFB_Allocate.jl")
include("../Source/HFB/HFB_Canonical_Basis.jl")
include("../Source/HFB/HFB_Energy.jl")
include("../Source/HFB/HFB_Export.jl")
include("../Source/HFB/HFB_BMBPT.jl")

# QTDA solver ...
include("../Source/QTDA/QTDA.jl")
include("../Source/QTDA/QTDA_Solver.jl")
include("../Source/QTDA/QTDA_Allocate.jl")
include("../Source/QTDA/QTDA_Spurious.jl")
include("../Source/QTDA/QTDA_Diagonalize.jl")
include("../Source/QTDA/QTDA_Transitions.jl")
include("../Source/QTDA/QTDA_Export.jl")

# QRPA solver ...
include("../Source/QRPA/QRPA.jl")
include("../Source/QRPA/QRPA_Solver.jl")
include("../Source/QRPA/QRPA_Allocate.jl")
include("../Source/QRPA/QRPA_Spurious.jl")
include("../Source/QRPA/QRPA_Diagonalize.jl")
include("../Source/QRPA/QRPA_A_Phonon.jl")
include("../Source/QRPA/QRPA_Transitions.jl")
include("../Source/QRPA/QRPA_Energy.jl")
include("../Source/QRPA/QRPA_OBDM.jl")
include("../Source/QRPA/QRPA_Export.jl")
=#

include("../Source/HF_RPA/HF_RPA_External.jl")
include("../Source/HF_RRPA/HF_RRPA_External.jl")


function main_run_dev()

    IntParams = Interaction_Parameters(
                NN_File = "IO/NN.bin",
                NNN_File = "IO/NNN.bin",
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
                HF = HF_Parameters(Tol = 1e-7, Imax = 150, BMF = true,
                                   PCGS = HF_PCGS_Parameters(TBCF = true, MFP = true, Export = true)),
                RPA = RPA_Parameters(Ortho = false, Reference = "HF-MBPT", Tol = 0.6, TBCF = false)
                )

    #HF(Parameters(IntParams,CalcParams))

    RPA(Parameters(IntParams,CalcParams))


    #HF_RPA(Parameters(IntParams,CalcParams))

    return




    IntParams = Interaction_Parameters(
                NN_File = "../../../../storage/interaction_files/NuHamil_NN_NNN_files/DN2LOGO394/hw_16/NN-DN2LOGO394_vbare2.0_hw16_emax16_e2max32.me2j.bin",
                NNN_File = "../../../../storage/interaction_files/NuHamil_NN_NNN_files/DN2LOGO394/hw_16/NNN_DN2LOGO394_hw16_hwt16_emax16_e2max32_e3max28_jmax3_59.stream.bin",
                hw = 16.0,
                Nmax = 16,
                N2max = 32,
                N3max = 28
                )

    CalcParams = Calculation_Parameters(
                A = 4,
                Z = 2,
                hw = 16.0,
                Nmax = 14,
                N2max = 28,
                N3max = 28,
                CMS = "CMS1+2B",
                Path = "HF_MBPT3_DN2LOGO394_A4_Z2_hw16.0_Nmax14_N2max28_N3max28",
                HF = HF_Parameters(Tol = 1e-8, Imax = 150, BMF = true),
                RPA = RPA_Parameters(Ortho = true)
                )

    #HF(Parameters(IntParams,CalcParams))

    #return

    CalcParams = Calculation_Parameters(
                A = 16,
                Z = 8,
                hw = 16.0,
                Nmax = 6,
                N2max = 12,
                N3max = 10,
                CMS = "CMS1+2B",
                Path = "HF_MBPT3_DN2LOGO394_A16_Z8_hw16.0_Nmax6_N2max12_N3max10",
                HF = HF_Parameters(Tol = 1e-8, Imax = 150, BMF = true),
                RPA = RPA_Parameters(Ortho = true)
                )

    HF(Parameters(IntParams,CalcParams))

    #return

    #=

    IntParams = Interaction_Parameters(
                NN_File = "../../../../storage/folprecht/ChiPots/NN/NN-DN2LOGO450_vbare2.0_hw16_emax14_e2max28.me2j.bin",
                NNN_File = "../../../../storage/folprecht/ChiPots/NNN/NNN_DN2LOGO450_hw16_Nmax14_N2max28_N3max26_no2b.stream.bin",
                hw = 16.0,
                Nmax = 14,
                N2max = 28,
                N3max = 26
                )

    CalcParams = Calculation_Parameters(
                A = 16,
                Z = 8,
                hw = 16.0,
                Nmax = 10,
                N2max = 20,
                N3max = 18,
                CMS = "CMS1+2B",
                Path = "HF_A16_Z8_hw16.0_Nmax10_N2max20_N3max18",
                HF = HF_Parameters(Tol = 1e-8, Imax = 150, BMF = true),
                RPA = RPA_Parameters(Ortho = true)
                )

    HF(Parameters(IntParams,CalcParams))

    CalcParams = Calculation_Parameters(
                A = 40,
                Z = 20,
                hw = 16.0,
                Nmax = 12,
                N2max = 24,
                N3max = 20,
                CMS = "CMS1+2B",
                Path = "HF_A40_Z20_hw16.0_Nmax12_N2max24_N3max20",
                HF = HF_Parameters(Tol = 1e-8, Imax = 150, BMF = true),
                RPA = RPA_Parameters(Ortho = true)
                )

    HF(Parameters(IntParams,CalcParams))

    CalcParams = Calculation_Parameters(
                A = 48,
                Z = 20,
                hw = 16.0,
                Nmax = 14,
                N2max = 28,
                N3max = 26,
                CMS = "CMS1+2B",
                Path = "HF_A48_Z20_hw16.0_Nmax14_N2max28_N3max26",
                HF = HF_Parameters(Tol = 1e-8, Imax = 150, BMF = true),
                RPA = RPA_Parameters(Ortho = true)
                )

    HF(Parameters(IntParams,CalcParams))
    =#

    IntParams = Interaction_Parameters(
                NN_File = "../../../../storage/interaction_files/NuHamil_NN_NNN_files/DN2LOGO394/hw_16/NN-DN2LOGO394_vbare2.0_hw16_emax16_e2max32.me2j.bin",
                NNN_File = "../../../../storage/interaction_files/NuHamil_NN_NNN_files/DN2LOGO394/hw_16/NNN_DN2LOGO394_hw16_hwt16_emax16_e2max32_e3max28_jmax3_59.stream.bin",
                hw = 16.0,
                Nmax = 16,
                N2max = 32,
                N3max = 28
                )

    CalcParams = Calculation_Parameters(
                A = 4,
                Z = 2,
                hw = 16.0,
                Nmax = 10,
                N2max = 20,
                N3max = 16,
                CMS = "CMS1+2B",
                Path = "HF_MBPT3_DN2LOGO394_A4_Z2_hw16.0_Nmax10_N2max20_N3max16",
                HF = HF_Parameters(Tol = 1e-8, Imax = 150, BMF = true),
                RPA = RPA_Parameters(Ortho = true)
                )

    HF(Parameters(IntParams,CalcParams))


    CalcParams = Calculation_Parameters(
                A = 40,
                Z = 20,
                hw = 16.0,
                Nmax = 10,
                N2max = 20,
                N3max = 16,
                CMS = "CMS1+2B",
                Path = "HF_MBPT3_DN2LOGO394_A40_Z22_hw16.0_Nmax10_N2max20_N3max16",
                HF = HF_Parameters(Tol = 1e-8, Imax = 150, BMF = true),
                RPA = RPA_Parameters(Ortho = true)
                )

    HF(Parameters(IntParams,CalcParams))


    CalcParams = Calculation_Parameters(
                A = 16,
                Z = 8,
                hw = 16.0,
                Nmax = 10,
                N2max = 20,
                N3max = 16,
                CMS = "CMS1+2B",
                Path = "HF_MBPT3_DN2LOGO394_A16_Z8_hw16.0_Nmax10_N2max20_N3max16",
                HF = HF_Parameters(Tol = 1e-8, Imax = 150, BMF = true),
                RPA = RPA_Parameters(Ortho = true)
                )

    HF(Parameters(IntParams,CalcParams))

    CalcParams = Calculation_Parameters(
                A = 16,
                Z = 8,
                hw = 16.0,
                Nmax = 14,
                N2max = 28,
                N3max = 28,
                CMS = "CMS1+2B",
                Path = "HF_MBPT3_DN2LOGO394_A16_Z8_hw16.0_Nmax14_N2max28_N3max28",
                HF = HF_Parameters(Tol = 1e-8, Imax = 150, BMF = true),
                RPA = RPA_Parameters(Ortho = true)
                )

    HF(Parameters(IntParams,CalcParams))

    CalcParams = Calculation_Parameters(
                A = 40,
                Z = 20,
                hw = 16.0,
                Nmax = 14,
                N2max = 28,
                N3max = 28,
                CMS = "CMS1+2B",
                Path = "HF_MBPT3_DN2LOGO394_A40_Z20_hw16.0_Nmax14_N2max28_N3max28",
                HF = HF_Parameters(Tol = 1e-8, Imax = 150, BMF = true),
                RPA = RPA_Parameters(Ortho = true)
                )

    HF(Parameters(IntParams,CalcParams))

    CalcParams = Calculation_Parameters(
                A = 48,
                Z = 20,
                hw = 16.0,
                Nmax = 14,
                N2max = 28,
                N3max = 28,
                CMS = "CMS1+2B",
                Path = "HF_MBPT3_DN2LOGO394_A48_Z20_hw16.0_Nmax14_N2max28_N3max28",
                HF = HF_Parameters(Tol = 1e-8, Imax = 150, BMF = true),
                RPA = RPA_Parameters(Ortho = true)
                )

    HF(Parameters(IntParams,CalcParams))

    CalcParams = Calculation_Parameters(
                A = 90,
                Z = 40,
                hw = 16.0,
                Nmax = 14,
                N2max = 28,
                N3max = 28,
                CMS = "CMS1+2B",
                Path = "HF_MBPT3_DN2LOGO394_A90_Z40_hw16.0_Nmax14_N2max28_N3max28",
                HF = HF_Parameters(Tol = 1e-8, Imax = 150, BMF = true),
                RPA = RPA_Parameters(Ortho = true)
                )

    HF(Parameters(IntParams,CalcParams))

    CalcParams = Calculation_Parameters(
                A = 132,
                Z = 50,
                hw = 16.0,
                Nmax = 14,
                N2max = 28,
                N3max = 28,
                CMS = "CMS1+2B",
                Path = "HF_MBPT3_DN2LOGO394_A132_Z50_hw16.0_Nmax14_N2max28_N3max28",
                HF = HF_Parameters(Tol = 1e-8, Imax = 150, BMF = true),
                RPA = RPA_Parameters(Ortho = true)
                )

    HF(Parameters(IntParams,CalcParams))

    CalcParams = Calculation_Parameters(
                A = 208,
                Z = 82,
                hw = 16.0,
                Nmax = 14,
                N2max = 28,
                N3max = 28,
                CMS = "CMS1+2B",
                Path = "HF_MBPT3_DN2LOGO394_A208_Z82_hw16.0_Nmax14_N2max28_N3max28",
                HF = HF_Parameters(Tol = 1e-8, Imax = 150, BMF = true),
                RPA = RPA_Parameters(Ortho = true)
                )

    HF(Parameters(IntParams,CalcParams))

    return
end

@time main_run_dev()
