function RPA_Hamiltonian(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,Orb_NNN::Orb3B,
                         C::O1B,Rho::O1B,Sigma::O2B,T_bare::O1B,V_NN_bare::O2B,V_NNN_bare::Array{Vector{Vector{Float32}},4})
    println("\nConstructing Hamiltonian for RPA calculation ...")

    if Params.Calc.RPA.ScH == true
        Rho = O1b_transformation(Rho,O1b_transpose(C))

        if Params.Calc.RPA.TBCF == true
            Sigma = O2b_transformation(Params,Orb,Orb_NN,Sigma,O1b_transpose(C),Report=false)

            h_N = H1b(Params,Orb,Orb_NN,Orb_NNN,T_bare,V_NN_bare,V_NNN_bare,Rho;Sigma=Sigma,Report=false)

            V_NN = H2b(Params,Orb,Orb_NN,Orb_NNN,V_NN_bare,V_NNN_bare,Rho,Report=false)

            h_N = O1b_transformation(h_N,C)
            V_NN = O2b_transformation(Params,Orb,Orb_NN,V_NN,C,Report=false)

            # O2b_transformation works in place ... transform the caller's Sigma back to the NAT basis ...
            Sigma = O2b_transformation(Params,Orb,Orb_NN,Sigma,C,Report=false)

        else
            h_N = H1b(Params,Orb,Orb_NN,Orb_NNN,T_bare,V_NN_bare,V_NNN_bare,Rho)
            V_NN = H2b(Params,Orb,Orb_NN,Orb_NNN,V_NN_bare,V_NNN_bare,Rho,Report=false)

            h_N = O1b_transformation(h_N,C)
            V_NN = O2b_transformation(Params,Orb,Orb_NN,V_NN,C,Report=false)
        end

    else
        h_N, V_NN = RPA_Hamiltonian_HF_reference(Params,Orb,Orb_NN,C)
    end

    println("\tFinished construction of RPA Hamiltonian ...")

    return h_N, V_NN
end

function RPA_Hamiltonian_HF_reference(Params::Parameters,Orb::Vector{Orb1B},Orb_NN::Orb2B,C::O1B)
    # Import the 1-body & 2-body reference HF Hamiltonian ... (HF basis)
    H_N = O1b_import(Params,Orb,"io/" * Params.Calc.Path * "/" * Params.Calc.RPA.Reference * "/bin/H1b.bin")
    H_NN = O2b_import(Params,Orb,"io/" * Params.Calc.Path * "/" * Params.Calc.RPA.Reference * "/bin/H2b.bin")

    # Transformation HF -> NAT ... (O1b_transformation(O,C) = C' * O * C, C_HF = C_LHO_HF, C = C_LHO_NAT)
    C_HF = O1b_import(Params,Orb,"io/" * Params.Calc.Path * "/" * Params.Calc.RPA.Reference * "/bin/C.bin")
    C_HF_NAT = O1B(C_HF.p' * C.p, C_HF.n' * C.n)

    # Transform the HF Hamiltonian from the HF to the NAT basis ... (H_NN is a fresh import, its in-place transformation is harmless)
    H_N = O1b_transformation(H_N,C_HF_NAT)
    H_NN = O2b_transformation(Params,Orb,Orb_NN,H_NN,C_HF_NAT,Report = false)

    return H_N, H_NN
end
