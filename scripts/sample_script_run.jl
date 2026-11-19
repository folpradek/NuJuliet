include("../src/import_NuJuliet.jl")
using .NuJuliet

function sample_script_run()

    # Interaction parameters structure ...
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
                #pG2N = 0.0,
                #nG2N = 0.0
                )

    # Calculation parameters structure ...
    CalcParams = Calculation_Parameters(
                A = 16,
                Z = 8,
                hw = 16.0,
                Nmax = 3,
                N2max = 6,
                N3max = 9,
                #CMS = "CMS1+2B",
                Path = "HF_RPA_TDA_A16_Z8_hw16.0_Nmax3_N2max6_N3max9",
                #HF = HF_Parameters(Tol = 1e-8, Imax = 150, BMF = true, ESH = true, MBPT = HF_MBPT_Parameters(Export = true, TBCF = true, ScH = true)),
                #TDA = TDA_Parameters(Reference = "HF", Name = "HF-TDA", Reference_TBCF = true, Ortho = true, Tol_hol = 0.3, Tol_par = 0.7),
                #RPA = RPA_Parameters(Reference = "HF", Name = "HF-RPA", Reference_TBCF = false,
                #                     Ortho = true, Transitions = "Linear", Tol_hol = 0.25, Tol_par = 0.85,
                #                     Tol = 1e-5, Imax = 100, OBDM_digonal = false, ScH = true),
                )

    Params = Parameters(IntParams,CalcParams)
    # HF calculation call ...
    HF(Params)

    # TDA calculation call ...
    TDA(Params)

    # RPA calculation call ...
    RPA(Params)

    # Function for export of transition radial densities ... 
        # JP specifies spin & parity, nu_list the list of phonon to plot,
        # note phonons are ordered by energy, so nu = 1 is the lowest phonon of given JP, nu = 2 the second lowest ...,
        # File_Name is the name of the output files in .dat format
    RPA_transition_densities(Params; JP = "1-", nu_list=[5,6,7], File_Name ="Phonon_Densities")
    TDA_transition_densities(Params; JP = "1-", nu_list=[5,6,7], File_Name ="Phonon_Densities")
    
    # Function for export of transition currents ...
        # JP specifies spin & parity, M specifies projection, I suggest using M = 0, nu_list the list of phonon to plot,
        # note phonons are ordered by energy, so nu = 1 is the lowest phonon of given JP, nu = 2 the second lowest ...,
        # File_Name is the name of the output files in .dat format, x & z grid is okay with 40x40 points ... Cartesian
    TDA_transition_currents(Params; JP = "1-", M = 0, nu_list=[5,6,7],
                            File_Name ="Phonon_Currents", xN_grid = 40, zN_grid = 40)
    RPA_transition_currents(Params; JP = "1-", M = 0, nu_list=[5,6,7],
                            File_Name ="Phonon_Currents", xN_grid = 40, zN_grid = 40)


    # Calculation parameters structure ...
    CalcParams = Calculation_Parameters(
                A = 4,
                Z = 2,
                hw = 16.0,
                Nmax = 3,
                N2max = 6,
                N3max = 9,
                #CMS = "CMS1+2B",
                Path = "HF_ScRPA_A4_Z2_hw16.0_Nmax3_N2max6_N3max9",
                #HF = HF_Parameters(Tol = 1e-8, Imax = 150, BMF = true, ESH = true, MBPT = HF_MBPT_Parameters(Export = true, TBCF = true, ScH = true)),
                #TDA = TDA_Parameters(Reference = "HF", Name = "HF-TDA", Reference_TBCF = true, Ortho = true, Tol_hol = 0.3, Tol_par = 0.7),
                RPA = RPA_Parameters(Reference = "HF", Name = "HF-ScRPA", Reference_TBCF = false,
                                     Ortho = true, Transitions = "Linear", TBCF = false, Tol_hol = 0.3, Tol_par = 0.85,
                                     Tol = 1e-5, Imax = 100, OBDM_diagonal = false, ScH = true),
                )

    Params = Parameters(IntParams,CalcParams)

    # HF calculation call ...
    HF(Params)

    # HF-ScRPA calculation call ...
    HF_ScRPA(Params)

    CalcParams = Calculation_Parameters(
                A = 18,
                Z = 8,
                hw = 16.0,
                Nmax = 3,
                N2max = 6,
                N3max = 9,
                #CMS = "CMS1+2B",
                Path = "BCS_A18_Z8_hw16.0_Nmax3_N2max6_N3max9",
                #HF = HF_Parameters(Tol = 1e-7, IMax = 100),
                #BCS = BCS_Parameters(ScBCS = true, Tol = 1e-7, Imax = 500, q = 0.05, pD0 = 0.5, nD0 = 0.5, BMF = false, HRF = false),
                QTDA = QTDA_Parameters(Ortho = true, Reference = "BCS"),
                QRPA = QRPA_Parameters(Ortho = true, Reference = "BCS")
                )

    Params = Parameters(IntParams,CalcParams)

    # HF-BCS calculation call ...
    BCS(Params)

    # QTDA calculation call ...
    QTDA(Params)
    
    # QRPA calculation call ...
    QRPA(Params)

    CalcParams = Calculation_Parameters(
                A = 18,
                Z = 8,
                hw = 16.0,
                Nmax = 3,
                N2max = 6,
                N3max = 9,
                #CMS = "CMS1+2B",
                Path = "HFB_A18_Z8_hw16.0_Nmax3_N2max6_N3max9",
                #HFB = HFB_Parameters(Tol = 1e-8, Imax = 150, Pairing = "Full", dLmax = 0.5, BMF = true, LNT = false, LNRes = false, HRF = false),
                QTDA = QTDA_Parameters(Ortho = true, Reference = "HFB"),
                QRPA = QRPA_Parameters(Ortho = true, Reference = "HFB")
                )
    
    Params = Parameters(IntParams,CalcParams)

    # HFB calculation call ...
    HFB(Params)

    # QTDA calculation call ...
    QTDA(Params)

    # pnQTDA calculation call ...
    pnQTDA(Params)
    
    # QRPA calculation call ...
    QRPA(Params)

    # Sample call for interaction file binaries parsing ...
        # Interaction parameters structure ...
    IntParams = Interaction_Parameters(
                NN_File = "io/NN.bin",
                NNN_File = "io/NNN.bin",
                hw = 16.0,
                Nmax = 3,
                N2max = 6,
                N3max = 9,
                )

    # Calculation parameters structure ...
    CalcParams = Calculation_Parameters(
                A = 16,
                Z = 8,
                hw = 16.0,
                Nmax = 3,
                N2max = 6,
                N3max = 9
                )

    Params = Parameters(IntParams,CalcParams)

    N_max_Out = 2
    N_2max_Out = 4
    N_3max_Out = 6

    V2b_parse(Params,N_max_Out,N_2max_Out,"io/NN_parsed.bin")
    V3b_no2b_parse(Params,N_max_Out,N_2max_Out,N_3max_Out,"io/NNN_parsed.bin")

    return
end

sample_script_run()