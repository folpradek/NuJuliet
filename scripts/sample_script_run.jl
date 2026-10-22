include("../src/import_NuJuliet.jl")
using .NuJuliet

# Note all the commented parameter options are optional and
# can be safely omitted for solver calls - default values
# will be used instead ...
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
                #Path = "A16_Z8_hw16.0_Nmax3_N2max6_N3max9",
                HF = HF_Parameters(Tol = 1e-7, Imax = 100, BMF = true, HRF = false, MBPT = HF_MBPT_Parameters(Export = true, TBCF = true, MFP = true)),
                #RPA = RPA_Parameters(Ortho = true),
                #RRPA = RRPA_Parameters(Ortho = true, Tol = 1e-7, Imax = 100, ScV3N = true, ScOBH = true, dOBDM = false)
                )

    Params = Parameters(IntParams,CalcParams)

    # HF calculation call ...
    HF(Params)

    # Currently being reworked ...
    #=
    # HF-RPA calculation call ...
    HF_RPA(Params)

    # Function for export of transition radial densities ... 
        # JP specifies spin & parity, nu_list the list of phonon to plot,
        # note phonons are ordered by energy, so nu = 1 is the lowest phonon of given JP, nu = 2 the second lowest ...,
        # File_Name is the name of the output files in .dat format
    HF_RPA_transition_densities(Params; JP = "1-", nu_list=[5,6,7], File_Name ="Phonon_Densities")
    
    # Function for export of transition currents ...
        # JP specifies spin & partiy, M specifies projection, I suggest using M = 0, nu_list the list of phonon to plot,
        # note phonons are ordered by energy, so nu = 1 is the lowest phonon of given JP, nu = 2 the second lowest ...,
        # File_Name is the name of the output files in .dat format, x & z grid is okay with 40x40 points ... Cartesian
    HF_RPA_transition_currents(Params; JP = "1-", M = 0, nu_list=[5,6,7],
                               File_Name ="Phonon_Currents", xN_grid = 40, zN_grid = 40)

    # HF-RRPA calculation call ...
    HF_RRPA(Params)

    # Function for export of transition radial densities ... as for RPA ...
    HF_RRPA_transition_densities(Params; JP = "1-", nu_list=[5,6,7], File_Name ="Phonon_Densities")
    
    # Function for export of transition currents ... as for RPA ...
    HF_RRPA_transition_currents(Params; JP = "1-", M = 0, nu_list=[5,6,7],
                                File_Name ="Phonon_Currents", xN_grid = 40, zN_grid = 40)
    =#

    CalcParams = Calculation_Parameters(
                A = 18,
                Z = 8,
                hw = 16.0,
                Nmax = 3,
                N2max = 6,
                N3max = 9,
                #CMS = "CMS1+2B",
                #Path = "ScBCS_A18_Z8_hw16.0_Nmax3_N2max6_N3max9",
                #HF = HF_Parameters(Tol = 1e-7, IMax = 100),
                #BCS = BCS_Parameters(ScBCS = true, Tol = 1e-7, Imax = 500, q = 0.05, pD0 = 0.5, nD0 = 0.5, BMF = false, HRF = false),
                #QTDA = QTDA_Parameters(Ortho = true, Reference = "HFB"),
                #QRPA = QRPA_Parameters(Ortho = true, Reference = "HFB")
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
                #Path = "HFB_A18_Z8_hw16.0_Nmax3_N2max6_N3max9",
                #HFB = HFB_Parameters(Tol = 1e-8, Imax = 150, Pairing = "Full", dLmax = 0.5, BMF = true, LNT = false, LNRes = false, HRF = false),
                #QTDA = QTDA_Parameters(Ortho = true, Reference = "HFB"),
                #QRPA = QRPA_Parameters(Ortho = true, Reference = "HFB")
                )
    
    Params = Parameters(IntParams,CalcParams)

    # HFB calculation call ...
    HFB(Params)

    # QTDA calculation call ...
    QTDA(Params)
    
    # QRPA calculation call ...
    QRPA(Params)

end

sample_script_run()
