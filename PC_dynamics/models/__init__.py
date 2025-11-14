import numpy as np
from modelbase.ode import Model


def get_model():
    def mass_action_1(S1: float, k_fwd: float) -> float:
        return k_fwd * S1

    def mass_action_2(S1: float, S2: float, k_fwd: float) -> float:
        return k_fwd * S1 * S2

    def reversible_mass_action_1_1(
            S1: float,
            P1: float,
            k_fwd: float,
            k_bwd: float,
    ) -> float:
        return k_fwd * S1 - k_bwd * P1

    def reversible_mass_action_2_1(
            S1: float,
            S2: float,
            P1: float,
            k_fwd: float,
            k_bwd: float,
    ) -> float:
        return k_fwd * S1 * S2 - k_bwd * P1

    def reversible_mass_action_2_2(
            S1: float,
            S2: float,
            P1: float,
            P2: float,
            k_fwd: float,
            k_bwd: float,
    ) -> float:
        return k_fwd * S1 * S2 - k_bwd * P1 * P2

    def vB6f(PCo, PCr, PQ, PQtot, k_b6f, Keq_b6f):
        k_b6f_reverse = k_b6f / Keq_b6f
        f_PQ = (
                PQ / PQtot
        )  # want to keep the rates in terms of fraction of PQHs, not total number
        f_PQH2 = 1 - f_PQ
        return f_PQH2 * PCo ** 2 * k_b6f - f_PQ * PCr ** 2 * k_b6f_reverse

    def vPSII(PQ, PQtot, k2_exc, PSII):
        f_PQ = PQ / PQtot
        return k2_exc * f_PQ * PSII

    def Keq_FAFd(E0_FA, F, E0_Fd, RT):
        DG1 = -E0_FA * F
        DG2 = -E0_Fd * F
        DG = -DG1 + DG2
        K = np.exp(-DG / RT)
        return K

    def dg_ph(r, t):
        return np.log(10) * r * t

    def excitation_PS(
            PFD: float, Na: float, Nb: float, phi, f_antennae: float, f_NPQ: float = 0.0
    ) -> float:
        """
        Returns rate constant for excitation of PSI or PSII, excitation events per ms.
        Based on Kroon et al. 2006

        Parameters:
            PFD : Irradiance,umol m-2 s-1
            Na  : Number of chla molecules per photosystem
            Nb  : Number of chlb molecules per photosystem
            phi : Quantum efficiency of photosystem
            f_antennae  : fraction of light absorbed by antennae
            f_NPQ  : fraction of excitation lost due to NPQ

        Returns:
            float: rate constant, excitation events per s.
        """
        # # # Excitation of PSI, Kroon et al. 2006
        rho_chl_a = 8.6749e-21  # m2/chla, optical cross-section of chl a
        rho_chl_b = 9.1222e-21  # m2/chlb, optical cross section of chl b

        av_No = 6.022e23  # molecules/mol (Avogadro’s number)

        cfI = av_No * 1e-6  # micro mol/s to quanta/s

        rho_PSI = rho_chl_a * Na + rho_chl_b * Nb  # m2, optical cross section of PSI
        exc_I = rho_PSI * PFD * cfI  # quanta/s, rate of excitation of PSI

        k_exc = exc_I * phi * f_antennae * (1 - f_NPQ)  # excitations/s,

        return k_exc

    def get_binding_rate_constants(k: float, chl: float) -> float:
        """
        Calculate binding rate constants on [Chl] basis.

        Parameters:
            k: Excitation rate co nstant for P700.FB (M^-1 s^-1)
            chl: Chlorophyll conc: M
        Returns:
            float: Binding rate constant, molChl/mmol/s
        """

        return k * chl / 1000

    def keq_PQred(E0_QA, F, E0_PQ, pHstroma, dG_pH, RT):
        DG1 = -E0_QA * F
        DG2 = -2 * E0_PQ * F
        DG = -2 * DG1 + DG2 + 2 * pHstroma * dG_pH
        K = np.exp(-DG / RT)
        return K

    def proportional(R, T):
        return R * T

    def ratio(kfd, keq):
        return kfd / keq

    m = Model()

    m.add_parameters(
        parameters={
            "kI_off": 3200.0,
            "Chl": 0.05,
            "kI_on_": 110000000.0,
            "PSI_tot": 2.5,
            "PC_tot": 4,
            "kII_off": 8000.0,
            "kII_on_": 40000000.0,
            "kI_+_on_": 600000000.0,
            "kI_+_off": 13000.0,
            "Light_intensity": 0,
            "k_recomb": 2.0,
            "Na_PSI": 254,
            "Nb_PSI": 85,
            "phi_PSI": 0.97,
            "f_antennae": 0.45,
            "E0_FA": -0.55,
            "E0_Fd": -0.43,
            "F": 96.485,
            "R": 0.0083,
            "T": 298.0,
            "kFd": 250000.0,
            "Fd_tot": 5,
            "ket": 58000.0,
            "kbet": 4500.0,
            "E0_QA": -0.14,
            "E0_PQ": 0.354,
            "kb6f": 2.5,
            "PQtot": 17.5,
            "pHstroma": 7.9,
            "Na_PSII": 261,
            "Nb_PSII": 158,
            "f_NPQ": 0.3,
            "PSII": 2.5,
            "kcbc": 100.0,
        }
    )
    m.add_derived_parameter(
        parameter_name="kI_on",
        function=get_binding_rate_constants,
        parameters=["Chl", "kI_on_"],
    )
    m.add_derived_parameter(
        parameter_name="kII_on",
        function=get_binding_rate_constants,
        parameters=["Chl", "kII_on_"],
    )
    m.add_derived_parameter(
        parameter_name="kI_+_on",
        function=get_binding_rate_constants,
        parameters=["Chl", "kI_+_on_"],
    )
    m.add_derived_parameter(
        parameter_name="k_exc",
        function=excitation_PS,
        parameters=["Light_intensity", "Na_PSI", "Nb_PSI", "phi_PSI", "f_antennae"],
    )
    m.add_derived_parameter(
        parameter_name="RT",
        function=proportional,
        parameters=["R", "T"],
    )
    m.add_derived_parameter(
        parameter_name="keqFd",
        function=Keq_FAFd,
        parameters=["E0_FA", "F", "E0_Fd", "RT"],
    )
    m.add_derived_parameter(
        parameter_name="kbwd",
        function=ratio,
        parameters=["kFd", "keqFd"],
    )
    m.add_derived_parameter(
        parameter_name="dG_pH",
        function=dg_ph,
        parameters=["R", "T"],
    )
    m.add_derived_parameter(
        parameter_name="Keqb6f",
        function=keq_PQred,
        parameters=["E0_QA", "F", "E0_PQ", "pHstroma", "dG_pH", "RT"],
    )
    m.add_derived_parameter(
        parameter_name="k2_exc",
        function=excitation_PS,
        parameters=[
            "Light_intensity",
            "Na_PSII",
            "Nb_PSII",
            "phi_PSI",
            "f_antennae",
            "f_NPQ",
        ],
    )
    m.add_compounds(
        compounds=[
            "PCr",
            "FBoP700r",
            "FBoP700rPCr",
            "PCo",
            "FBoP700rPCo",
            "FBrP700o",
            "FBrP700oPCr",
            "FBrP700oPCo",
            "FBoP700o",
            "FBoP700oPCr",
            "FBoP700oPCo",
            "FBrP700r",
            "FBrP700rPCr",
            "FBrP700rPCo",
            "FDo",
            "FDr",
            "PQH2",
            "PQ",
            "P",
        ]
    )
    m.add_rate(
        rate_name="V_bind_FBo.P700r.PCr",
        function=reversible_mass_action_2_1,
        substrates=["FBoP700r", "PCr"],
        products=["FBoP700rPCr"],
        modifiers=[],
        parameters=["kI_on", "kI_off"],
        reversible=True,
        args=["FBoP700r", "PCr", "FBoP700rPCr", "kI_on", "kI_off"],
    )
    m.add_rate(
        rate_name="V_bind_FBo.P700r.PCo",
        function=reversible_mass_action_2_1,
        substrates=["FBoP700r", "PCo"],
        products=["FBoP700rPCo"],
        modifiers=[],
        parameters=["kII_on", "kII_off"],
        reversible=True,
        args=["FBoP700r", "PCo", "FBoP700rPCo", "kII_on", "kII_off"],
    )
    m.add_rate(
        rate_name="V_bind_FBr.P700o.PCr",
        function=reversible_mass_action_2_1,
        substrates=["FBrP700o", "PCr"],
        products=["FBrP700oPCr"],
        modifiers=[],
        parameters=["kI_+_on", "kI_+_off"],
        reversible=True,
        args=["FBrP700o", "PCr", "FBrP700oPCr", "kI_+_on", "kI_+_off"],
    )
    m.add_rate(
        rate_name="V_bind_FBr.P700o.PCo",
        function=reversible_mass_action_2_1,
        substrates=["FBrP700o", "PCo"],
        products=["FBrP700oPCo"],
        modifiers=[],
        parameters=["kI_+_on", "kI_+_off"],
        reversible=True,
        args=["FBrP700o", "PCo", "FBrP700oPCo", "kI_+_on", "kI_+_off"],
    )
    m.add_rate(
        rate_name="V_bind_FBo.P700o.PCr",
        function=reversible_mass_action_2_1,
        substrates=["FBoP700o", "PCr"],
        products=["FBoP700oPCr"],
        modifiers=[],
        parameters=["kI_+_on", "kI_+_off"],
        reversible=True,
        args=["FBoP700o", "PCr", "FBoP700oPCr", "kI_+_on", "kI_+_off"],
    )
    m.add_rate(
        rate_name="V_bind_FBo.P700o.PCo",
        function=mass_action_2,
        substrates=["FBoP700o", "PCo"],
        products=["FBoP700oPCo"],
        modifiers=[],
        parameters=["kI_+_on"],
        reversible=False,
        args=["FBoP700o", "PCo", "kI_+_on"],
    )
    m.add_rate(
        rate_name="V_rel_FBo.P700o.PCo",
        function=mass_action_1,
        substrates=["FBoP700oPCo"],
        products=["FBoP700o", "PCo"],
        modifiers=[],
        parameters=["kI_+_off"],
        reversible=False,
        args=["FBoP700oPCo", "kI_+_off"],
    )
    m.add_rate(
        rate_name="V_bind_FBr.P700r.PCr",
        function=reversible_mass_action_2_1,
        substrates=["FBrP700r", "PCr"],
        products=["FBrP700rPCr"],
        modifiers=[],
        parameters=["kI_on", "kI_off"],
        reversible=True,
        args=["FBrP700r", "PCr", "FBrP700rPCr", "kI_on", "kI_off"],
    )
    m.add_rate(
        rate_name="V_bind_FBr.P700r.PCo",
        function=mass_action_2,
        substrates=["FBrP700r", "PCo"],
        products=["FBrP700rPCo"],
        modifiers=[],
        parameters=["kII_on"],
        reversible=False,
        args=["FBrP700r", "PCo", "kII_on"],
    )
    m.add_rate(
        rate_name="V_rel_FBr.P700r.PCo",
        function=mass_action_1,
        substrates=["FBrP700rPCo"],
        products=["FBrP700r", "PCo"],
        modifiers=[],
        parameters=["kII_off"],
        reversible=False,
        args=["FBrP700rPCo", "kII_off"],
    )
    m.add_rate(
        rate_name="V_ex_FBo.P700r",
        function=reversible_mass_action_1_1,
        substrates=["FBoP700r"],
        products=["FBrP700o"],
        modifiers=[],
        parameters=["k_exc", "k_recomb"],
        reversible=True,
        args=["FBoP700r", "FBrP700o", "k_exc", "k_recomb"],
    )
    m.add_rate(
        rate_name="V_ex_FBo.P700r.PCr",
        function=reversible_mass_action_1_1,
        substrates=["FBoP700rPCr"],
        products=["FBrP700oPCr"],
        modifiers=[],
        parameters=["k_exc", "k_recomb"],
        reversible=True,
        args=["FBoP700rPCr", "FBrP700oPCr", "k_exc", "k_recomb"],
    )
    m.add_rate(
        rate_name="V_ex_FBo.P700r.PCo",
        function=reversible_mass_action_1_1,
        substrates=["FBoP700rPCo"],
        products=["FBrP700oPCo"],
        modifiers=[],
        parameters=["k_exc", "k_recomb"],
        reversible=True,
        args=["FBoP700rPCo", "FBrP700oPCo", "k_exc", "k_recomb"],
    )
    m.add_rate(
        rate_name="V_FD_red_P700o",
        function=reversible_mass_action_2_2,
        substrates=["FBrP700o", "FDo"],
        products=["FBoP700o", "FDr"],
        modifiers=[],
        parameters=["kFd", "kbwd"],
        reversible=True,
        args=["FBrP700o", "FDo", "FBoP700o", "FDr", "kFd", "kbwd"],
    )
    m.add_rate(
        rate_name="V_FD_red_P700o.PCr",
        function=reversible_mass_action_2_2,
        substrates=["FBrP700oPCr", "FDo"],
        products=["FBoP700oPCr", "FDr"],
        modifiers=[],
        parameters=["kFd", "kbwd"],
        reversible=True,
        args=["FBrP700oPCr", "FDo", "FBoP700oPCr", "FDr", "kFd", "kbwd"],
    )
    m.add_rate(
        rate_name="V_FD_red_P700o.PCo",
        function=reversible_mass_action_2_2,
        substrates=["FBrP700oPCo", "FDo"],
        products=["FBoP700oPCo", "FDr"],
        modifiers=[],
        parameters=["kFd", "kbwd"],
        reversible=True,
        args=["FBrP700oPCo", "FDo", "FBoP700oPCo", "FDr", "kFd", "kbwd"],
    )
    m.add_rate(
        rate_name="V_FD_red_P700r.PCo",
        function=reversible_mass_action_2_2,
        substrates=["FBrP700rPCo", "FDo"],
        products=["FBoP700rPCo", "FDr"],
        modifiers=[],
        parameters=["kFd", "kbwd"],
        reversible=True,
        args=["FBrP700rPCo", "FDo", "FBoP700rPCo", "FDr", "kFd", "kbwd"],
    )
    m.add_rate(
        rate_name="V_FD_red_P700r.PCr",
        function=reversible_mass_action_2_2,
        substrates=["FBrP700rPCr", "FDo"],
        products=["FBoP700rPCr", "FDr"],
        modifiers=[],
        parameters=["kFd", "kbwd"],
        reversible=True,
        args=["FBrP700rPCr", "FDo", "FBoP700rPCr", "FDr", "kFd", "kbwd"],
    )
    m.add_rate(
        rate_name="V_FD_red_P700r",
        function=reversible_mass_action_2_2,
        substrates=["FBrP700r", "FDo"],
        products=["FBoP700r", "FDr"],
        modifiers=[],
        parameters=["kFd", "kbwd"],
        reversible=True,
        args=["FBrP700r", "FDo", "FBoP700r", "FDr", "kFd", "kbwd"],
    )
    m.add_rate(
        rate_name="V_et_FBr.P700o.PCr",
        function=reversible_mass_action_1_1,
        substrates=["FBrP700oPCr"],
        products=["FBrP700rPCo"],
        modifiers=[],
        parameters=["ket", "kbet"],
        reversible=True,
        args=["FBrP700oPCr", "FBrP700rPCo", "ket", "kbet"],
    )
    m.add_rate(
        rate_name="V_et_FBo.P700o.PCr",
        function=reversible_mass_action_1_1,
        substrates=["FBoP700oPCr"],
        products=["FBoP700rPCo"],
        modifiers=[],
        parameters=["ket", "kbet"],
        reversible=True,
        args=["FBoP700oPCr", "FBoP700rPCo", "ket", "kbet"],
    )
    m.add_rate(
        rate_name="vB6f",
        function=vB6f,
        substrates=["PQH2", "PCo"],
        products=["PQ", "PCr"],
        modifiers=[],
        parameters=["PQtot", "kb6f", "Keqb6f"],
        reversible=True,
        args=["PCo", "PCr", "PQ", "PQtot", "kb6f", "Keqb6f"],
    )
    m.add_rate(
        rate_name="VPSII",
        function=vPSII,
        substrates=["PQ"],
        products=["PQH2"],
        modifiers=[],
        parameters=["PQtot", "k2_exc", "PSII"],
        reversible=False,
        args=["PQ", "PQtot", "k2_exc", "PSII"],
    )
    m.add_rate(
        rate_name="VCBC",
        function=mass_action_1,
        substrates=["FDr"],
        products=["FDo", "P"],
        modifiers=[],
        parameters=["kcbc"],
        reversible=False,
        args=["FDr", "kcbc"],
    )

    m.add_stoichiometries(
        rate_stoichiometries={
            "V_bind_FBo.P700r.PCr": {"FBoP700r": -1, "PCr": -1, "FBoP700rPCr": 1},
            "V_bind_FBo.P700r.PCo": {"FBoP700r": -1, "PCo": -1, "FBoP700rPCo": 1},
            "V_bind_FBr.P700o.PCr": {"FBrP700o": -1, "PCr": -1, "FBrP700oPCr": 1},
            "V_bind_FBr.P700o.PCo": {"FBrP700o": -1, "PCo": -1, "FBrP700oPCo": 1},
            "V_bind_FBo.P700o.PCr": {"FBoP700o": -1, "PCr": -1, "FBoP700oPCr": 1},
            "V_bind_FBo.P700o.PCo": {"FBoP700o": -1, "PCo": -1, "FBoP700oPCo": 1},
            "V_rel_FBo.P700o.PCo": {"FBoP700o": 1, "PCo": 1, "FBoP700oPCo": -1},
            "V_bind_FBr.P700r.PCr": {"FBrP700r": -1, "PCr": -1, "FBrP700rPCr": 1},
            "V_bind_FBr.P700r.PCo": {"FBrP700r": -1, "PCo": -1, "FBrP700rPCo": 1},
            "V_rel_FBr.P700r.PCo": {"FBrP700r": 1, "PCo": 1, "FBrP700rPCo": -1},
            "V_ex_FBo.P700r": {"FBoP700r": -1, "FBrP700o": 1},
            "V_ex_FBo.P700r.PCr": {"FBoP700rPCr": -1, "FBrP700oPCr": 1},
            "V_ex_FBo.P700r.PCo": {"FBoP700rPCo": -1, "FBrP700oPCo": 1},
            "V_FD_red_P700o": {"FBrP700o": -1, "FDo": -1, "FBoP700o": 1, "FDr": 1},
            "V_FD_red_P700o.PCr": {
                "FBrP700oPCr": -1,
                "FDo": -1,
                "FBoP700oPCr": 1,
                "FDr": 1,
            },
            "V_FD_red_P700o.PCo": {
                "FBrP700oPCo": -1,
                "FDo": -1,
                "FBoP700oPCo": 1,
                "FDr": 1,
            },
            "V_FD_red_P700r.PCo": {
                "FBrP700rPCo": -1,
                "FDo": -1,
                "FBoP700rPCo": 1,
                "FDr": 1,
            },
            "V_FD_red_P700r.PCr": {
                "FBrP700rPCr": -1,
                "FDo": -1,
                "FBoP700rPCr": 1,
                "FDr": 1,
            },
            "V_FD_red_P700r": {"FBrP700r": -1, "FDo": -1, "FBoP700r": 1, "FDr": 1},
            "V_et_FBr.P700o.PCr": {"FBrP700oPCr": -1, "FBrP700rPCo": 1},
            "V_et_FBo.P700o.PCr": {"FBoP700oPCr": -1, "FBoP700rPCo": 1},
            "vB6f": {"PQH2": -1, "PCo": -2, "PQ": 1, "PCr": 2},
            "VPSII": {"PQ": -1, "PQH2": 1},
            "VCBC": {"FDr": -1, "FDo": 1, "P": 1},
        }
    )

    return m
