import numpy as np


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
    return f_PQH2 * PCo**2 * k_b6f - f_PQ * PCr**2 * k_b6f_reverse


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



