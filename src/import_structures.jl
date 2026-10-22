# Parameter structures ...
    Base.@kwdef struct Interaction_Parameters
        NN_File::String = ""
        NNN_File::String = ""
        hw::Float64 = 0.0
        Nmax::Int64 = 0
        N2max::Int64 = 0
        N3max::Int64 = 0
        cP2N::Float64 = 1.0
        cP3N::Float64 = 1.0
        cRes::Float64 = 1.0
        pG2N::Float64 = 0.0
        nG2N::Float64 = 0.0
    end

    Base.@kwdef struct HF_MBPT_Parameters
        Export::Bool = false
        TBCF::Bool = false
        MFP::Bool = false
    end

    Base.@kwdef struct HF_Parameters
        Tol::Float64 = 1e-7
        Imax::Int64 = 100
        BMF::Bool = true
        HRF::Bool = false
        MBPT::HF_MBPT_Parameters = HF_MBPT_Parameters()
    end

    Base.@kwdef struct RPA_Parameters
        Name::String = "RPA"
        Reference::String = "HF"
        Tol::Float64 = 1e-5
        Ortho::Bool = true
        TBCF::Bool = false
        Transitions::String = "Exact"
    end

    Base.@kwdef struct RRPA_Parameters
        Ortho::Bool = true
        Tol::Float64 = 1e-7
        Imax::Int64 = 150
        ScV3N::Bool = true
        ScOBH::Bool = true
        dOBDM::Bool = false
    end

    Base.@kwdef struct BCS_Parameters
        ScBCS::Bool = false
        Tol::Float64 = 1e-7
        Imax::Int64 = 500
        q::Float64 = 0.05
        A0::Int64 = 0
        Z0::Int64 = 0
        pD0::Float64 = 0.5
        nD0::Float64 = 0.5
        BMF::Bool = true
        HRF::Bool = false
    end

    Base.@kwdef struct HFB_Parameters
        Tol::Float64 = 1e-7
        Imax::Int64 = 150
        Pairing::String = "Full"
        dLmax::Float64 = 0.5
        pL0::Float64 = -3.0
        nL0::Float64 = -3.0
        pa0::Float64 = 0.25
        na0::Float64 = 0.25
        Broy_m::Int64 = 8
        Broy_Qmax::Int64 = 3
        Broy_Amax::Float64 = 0.9
        Broy_Bmax::Float64 = 0.3
        Broy_Tmax::Float64 = 0.9
        LNT::Bool = false
        LNRes::Bool = true
        BMF::Bool = true
        HRF::Bool = false
    end

    Base.@kwdef struct QTDA_Parameters
        Ortho::Bool = true
        Reference::String = "HFB"
    end

    Base.@kwdef struct QRPA_Parameters
        Ortho::Bool = true
        Reference::String = "HFB"
    end

    Base.@kwdef struct Calculation_Parameters
        A::Int64 = 0
        Z::Int64 = 0
        hw::Float64 = 0.0
        Nmax::Int64 = 0
        N2max::Int64 = 0
        N3max::Int64 = 0
        CMS::String = "CMS1+2B"
        Path::String = "A" * string(A) * "_Z" * string(Z) *
                       "_hw" * string(hw) * "_Nmax" * string(Nmax) *
                       "_N2max" * string(N2max) * "_N3max" * string(N3max) * "_" * CMS
        HF::HF_Parameters = HF_Parameters()
        RPA::RPA_Parameters = RPA_Parameters()
        RRPA::RRPA_Parameters = RRPA_Parameters()
        BCS::BCS_Parameters = BCS_Parameters()
        HFB::HFB_Parameters = HFB_Parameters()
        QTDA::QTDA_Parameters = QTDA_Parameters()
        QRPA::QRPA_Parameters = QRPA_Parameters()
    end

    struct Parameters
        Int::Interaction_Parameters
        Calc::Calculation_Parameters
    end

# Orbitals, Single-Particle, Particle-Hole & Phonon State structures & 1-body operators ...
    struct Orb1B
        a::Int64
        n::Int64
        l::Int64
        j::Int64
        pO::Float64
        nO::Float64
    end

    struct SPState
        a::Int64
        l::Int64
        j::Int64
        E::Float64
    end

    struct pnSVector
        p::Vector{SPState}
        n::Vector{SPState}
    end

    struct pnInteger
        p::Int64
        n::Int64
    end

    struct pnFloat
        p::Float64
        n::Float64
    end

    struct pnVector
        p::Vector{Float64}
        n::Vector{Float64}
    end

    struct pnMatrix
        p::Matrix{Float64}
        n::Matrix{Float64}
    end

    struct pnArray
        p::Array
        n::Array
    end

    struct pnCVector
        p::Vector{ComplexF64}
        n::Vector{ComplexF64}
    end

    struct pnOrbVector
        p::Vector{Orb1B}
        n::Vector{Orb1B}
    end

    struct PhState
        p::Int64
        h::Int64
        tz::Int64
        J::Int64
        P::Int64
    end

    struct Ind2qp
        a::Int64
        b::Int64
        T::Int64
    end

    struct qpOrb2B
        N::Matrix{Int64}
        i::Matrix{Vector{Ind2qp}}
    end

    struct O1B
        p::Matrix{Float64}
        n::Matrix{Float64}
    end

    struct qpO1B
        qp11::O1B
        qp20::O1B
    end

# Interaction array structures ...
    # Standard 2-body NN operator structures ...
    struct O2B
        pp::Matrix{Vector{Float64}}
        pn::Matrix{Vector{Float64}}
        nn::Matrix{Vector{Float64}}
    end

    struct Orb2B
        Dic::Dict{UInt64,Int64}
        N::Array{Int64}
        Ind::Array{Vector{Vector{Int64}},3}
    end

    # Standard 3-body NNN operator orbitals structure ...
    struct Orb3B
        Dic::Dict{UInt64,Int64}
        N::Array{Vector{Int64}}
    end

    # Temporary 2-body NN operator structures ...
    struct O2B_Temp
        pp::Matrix{Matrix{Float64}}
        pn::Matrix{Matrix{Float64}}
        nn::Matrix{Matrix{Float64}}
    end

    struct Orb2B_Temp
        Dic::Dict{UInt64,Int64}
        N::Matrix{Int64}
        Ind::Matrix{Vector{Vector{Int64}}}
    end

    # Quasiparticle 2-body NN operator structures ...
    struct qpO40
        pp::Matrix{Vector{Float64}}
        pn::Matrix{Matrix{Float64}}
        nn::Matrix{Vector{Float64}}
    end

    struct qpO31
        pp::Matrix{Matrix{Float64}}
        pn2011::Matrix{Matrix{Float64}}
        pn1120::Matrix{Matrix{Float64}}
        nn::Matrix{Matrix{Float64}}
    end

    struct qpO22
        pp::Matrix{Vector{Float64}}
        pn2002::Matrix{Matrix{Float64}}
        pn1111::Matrix{Vector{Float64}}
        nn::Matrix{Vector{Float64}}
    end

    struct qpO2B
        qp40::qpO40
        qp31::qpO31
        qp22::qpO22
    end

    struct O2B_31_Temp
        pp::Vector{Matrix{Matrix{Float64}}}
        pn2011::Vector{Matrix{Matrix{Float64}}}
        pn1120::Vector{Matrix{Matrix{Float64}}}
        nn::Vector{Matrix{Matrix{Float64}}}
    end

# Electromagnetic transition operator structures ...
    struct Tr1B
        E0::O1B
        E1::O1B
        E2::O1B
        E3::O1B
        E1_VC::O1B
        E1_VS::O1B
        E1_TC::O1B
        E1_TS::O1B
        E1_sTC::O1B
        E1_sTS::O1B
        E1_C::O1B
        E1_sC::O1B
        M1::O1B
        M2::O1B
        M3::O1B
    end

    struct qpTr1B
        E0::qpO1B
        E1::qpO1B
        E2::qpO1B
        E3::qpO1B
        E1_VC::qpO1B
        E1_VS::qpO1B
        E1_TC::qpO1B
        E1_TS::qpO1B
        E1_sTC::qpO1B
        E1_sTS::qpO1B
        E1_C::qpO1B
        E1_sC::qpO1B
        M1::qpO1B
        M2::qpO1B
        M3::qpO1B
    end

    struct ReducedMultipole
        E0::pnCVector
        E1::pnCVector
        E2::pnCVector
        E3::pnCVector
        E1_VC::pnCVector
        E1_VS::pnCVector
        E1_TC::pnCVector
        E1_TS::pnCVector
        E1_sTC::pnCVector
        E1_sTS::pnCVector
        E1_C::pnCVector
        E1_sC::pnCVector
    end

    struct Transition
        ph::Vector{Float64}
        is::Vector{Float64}
        iv::Vector{Float64}
    end

    struct ReducedTransition
        E0::Transition
        E1::Transition
        E2::Transition
        E3::Transition
        E1_V::Transition
        E1_VC::Transition
        E1_VS::Transition
        E1_T::Transition
        E1_TC::Transition
        E1_TS::Transition
        E1_C::Transition
        E1_CC::Transition
        E1_CS::Transition
        E1_c::Transition
        E1_NLO_LWA::Transition
    end

# HF-MBPT auxilliary structures ...
    struct Orb1B_HF_MBPT
        p::pnOrbVector
        h::pnOrbVector
        N_p::pnInteger
        N_h::pnInteger
    end

    struct Orb_HF_MBPT3
        a::Int64
        l::Int64
        j::Int64
        t::Int64
        E::Float64
    end

# HF-RRPA auxilliary structures ...
    struct OBDM_Iteration_Pairs
        N::Int64
        T::Vector{Int64}
        a::Vector{Int64}
        b::Vector{Int64}
    end

# HFB structures ...
    # Structures for the modified Broyden's algorithm ...
    struct HFB_Broyden_Key
        a::Int64
        b::Int64
    end

    struct HFB_Broyden
        Map::Matrix{Int64}
        Key::Vector{HFB_Broyden_Key}
        M::Int64
        m::Int64
    end

    mutable struct HFB_Broyden_Vector
        x_Rho::pnVector
        y_Rho::pnVector
        z_Rho::pnVector
        X_Rho::O1B
        r_Rho::pnVector
        s_Rho::pnVector
        R_Rho::O1B
        x_Kappa::pnVector
        y_Kappa::pnVector
        z_Kappa::pnVector
        X_Kappa::O1B
        r_Kappa::pnVector
        s_Kappa::pnVector
        R_Kappa::O1B
    end