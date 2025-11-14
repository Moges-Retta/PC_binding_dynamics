# helper functions for plotting
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

dicts = {
    'fb_p700_indices': ["FBoP700r", "FBoP700rPCr", "FBoP700rPCo", "FBrP700o", "FBrP700oPCr", "FBrP700oPCo", "FBrP700r",
                        "FBrP700rPCr", "FBoP700o"
        , "FBoP700oPCr", "FBoP700oPCo"],  # rows contributing to FB·P700
    'fd_cbc_indices': ["FDr", "FDo"],  # Fdo, Fdr, CBC
    'pq_indices': ["PQ", "PQH2"],  # PQ and PQH2
    'pc_indices': ["PCr", "PCo", "FBoP700rPCr", "FBoP700rPCo", "FBrP700oPCr", "FBrP700oPCo",
                   "FBoP700oPCo", "FBoP700oPCr", "FBrP700rPCr", "FBrP700rPCo"]  # PCox and PCred
}

cols = [item for sublist in dicts.values() for item in sublist]

# Separate lists
cols_red = [x for x in cols if ("P700r" in x) or (x in ["FDr", "PCr", "PQH2"])]
cols_ox = [x for x in cols if ("P700o" in x) or (x in ["PCo", "PQ", "FDo"])]

print("Reduced:", cols_red)
print("Oxidized:", cols_ox)


def state_v_indices():
    # Define species indices for each group

    dicts = {
        'fb_p700_indices': ["FBoP700r", "FBoP700rPCr", "FBoP700rPCo", "FBrP700o", "FBrP700oPCr", "FBrP700oPCo",
                            "FBrP700r", "FBrP700rPCr", "FBoP700o"
            , "FBoP700oPCr", "FBoP700oPCo"],  # rows contributing to FB·P700
        'fd_cbc_indices': ["FDr", "FDo"],  # Fdo, Fdr, CBC
        'pq_indices': ["PQ", "PQH2"],  # PQ and PQH2
        'pc_indices': ["PCr", "PCo", "FBoP700rPCr", "FBoP700rPCo", "FBrP700oPCr", "FBrP700oPCo",
                       "FBoP700oPCo", "FBoP700oPCr", "FBrP700rPCr", "FBrP700rPCo"]  # PCox and PCred
    }

    return dicts


def calculate_total_mass(df):
    """
    Calculate the total mass of species to check for conservation
    """
    dict_indices = state_v_indices()

    fb_p700_indices = dict_indices['fb_p700_indices']
    pq_indices = dict_indices['pq_indices']
    pc_indices = dict_indices['pc_indices']
    fd_cbc_indices = dict_indices['fd_cbc_indices']

    fb_p700_sum = df[fb_p700_indices].sum(axis=1)
    pq_sum = df[pq_indices].sum(axis=1)
    pc_sum = df[pc_indices].sum(axis=1)
    fd_sum = df[fd_cbc_indices].sum(axis=1)

    return fb_p700_sum, pq_sum, pc_sum, fd_sum


def compute_electrons(df):
    """
    Returns the fraction of P700+

    Parameters:
        sol.y (array): concentration of species

    Returns:
        array: fraction of P700+.
    """
    dict_indices = state_v_indices()
    cols = [item for sublist in dict_indices.values() for item in sublist]

    cols_red = [x for x in cols if ("P700r" in x) or (x in ["FDr", "PCr", "PQH2"])]
    cols_oxd = [x for x in cols if ("P700o" in x) or (x in ["PCo", "PQ", "FDo"])]

    red = df[cols_red].sum(axis=1)
    ox = df[cols_oxd].sum(axis=1)

    return ox, red


# Define custom labels for species
custom_labels = {
    "FBoP700r": r"$FB{^o}.P700{^r}$",  # FB.P700 concentration
    "FBoP700rPCr": r"$FB{^o}.P700{^r}.PC{^r}$",  # FB.P700.PCred concentration (PCred-bound state)
    "FBoP700rPCo": r"$FB{^o}.P700{^r}.PC{^o}$",  # FB.P700.PCox concentration (PCox-bound state)
    "FBrP700oPCr": r"$FB{^r}.P700{^o}.PC{^r}$",  # FB-.P700+.PCox concentration (excited PCox-bound state)
    "FBrP700o": r"$FB{^r}.P700{^o}$",  # FB-.P700+ concentration (excited unbound state)
    "FBrP700oPCo": r"$FB{^r}.P700{^o}.PC{^o}$",  # FB-.P700+.PCred concentration (excited PCred-bound state)
    "FBoP700oPCo": r"$FB{^o}.P700{^o}.PC{^o}$",  # FB-.P700 concentration (reduced unbound state)
    "FBoP700oPCr": r"$FB{^o}.P700{^o}.PC{^r}$",  # FFB.P700+.PCox concentration (oxidised P700 PCox-bound state)
    "FBoP700o": r"$FB{^o}.P700{^o}$",  # FB.P700+ concentration (oxidised P700)
    "FBrP700rPCo": r"$FB{^r}.P700{^r}.PC{^o}$",  # FB.P700+.PCred concentration (oxidized PCred-bound state)
    "PCo": r"$PC{^o}$",  # PCox concentration (oxidized plastocyanin)
    "PCr": r"$PC{^r}$",  # PCred concentration (reduced plastocyanin)
    "FDo": r"$FD{^o}$",  # Fdo concentration (oxidised ferredoxin)
    "FDr": r"$FD{^r}$",  # Fdr concentration (reduced ferredoxin)
    "PQ": r"$PQ{^o}$",  # PQ+
    "PQH2": r"$PQH_{2}$",  # PQH2
    "P": "CBC",  # product from Fd- consumption in CBC cycle
    "FBrP700r": r"$FB{^r}.P700{^r}$",  # FBr.P700r
    "FBrP700rPCr": r"$FB{^r}.P700{^r}.PC{^r}$"  # FBr.P700r.PCr

}

species_colors = {
    'FBoP700r': '#1f77b4',  # blue
    'FBoP700rPCr': '#ff7f0e',  # orange
    'FBoP700rPCo': '#2ca02c',  # green
    'FBrP700o': '#d62728',  # red
    'FBrP700oPCr': '#9467bd',  # purple
    'FBrP700oPCo': '#8c564b',  # brown
    'FBoP700o': '#e377c2',  # pink
    'FBoP700oPCr': '#7f7f7f',  # gray
    'FBoP700oPCo': '#bcbd22',  # lime
    'FBrP700rPCo': '#17becf',  # cyan
    'PCo': '#1a55FF',  # vibrant blue
    'PCr': '#F28500',  # tangerine
    'FDo': '#00B945',  # jade green
    'FDr': '#DC267F',  # magenta
    'PQ': '#FF6DB6',  # hot pink
    'PQH2': '#785EF0',  # indigo
    'P': '#FE6100',  # orange-red
    'FBrP700r': '#FFB000',  # golden yellow
    'FBrP700rPCr': '#648FFF',  # sky blue
}

species_groups = [

    ["FBoP700r", "FBoP700rPCr", "FBoP700rPCo"],  # P700 binding and excitation
    ["FBrP700o", "FBrP700oPCr", "FBrP700oPCo"],  # FB-.P700+ bounded or free and FB-.P700
    ["FBoP700o",  # FB-.P700.PCox bounded or free
     "FBoP700oPCr", "FBoP700oPCo"],  # FB.P700+ bounded or free, after e transfer to Fd
    ["PCo", "PCr"],  # PCox, PCred
    ["FDo", "FDr"],  # Fd and Fd-
    ["PQ", "PQH2"],  # PQ+, PQH2
    ["P"],  # product from Fd- consumption in CBC cycle
    ["FBrP700r", "FBrP700rPCr", "FBrP700rPCo"]  # FBr.P700r bounded or free and FBr.P700r.PCr

]


def plot_conservation(df:pd.DataFrame):
    """
    Plots the conservation of mass.

    Parameters:
        df (DataFrame): Solution of ODE.
    """
    fb_p700_sum, pq_sum, pc_sum, fd_cbc_indices = calculate_total_mass(df)

    # Plot each balance over time
    plt.figure()

    plt.plot(pq_sum, label='PQ + PQH2', marker='^')
    plt.axhline(17.5, color='gray', linestyle='--', label='Expected total  = 17.5')

    plt.plot(fd_cbc_indices, label='Fdo + Fdr', marker='^')
    plt.axhline(5, color='gray', linestyle='--', label='Expected total  = 5')

    plt.plot(pc_sum, label='Pco + PCr', marker='^')
    plt.axhline(4, color='gray', linestyle='--', label='Expected total = 4')

    plt.plot(fb_p700_sum, label='FB·P700 Total', marker='o')
    plt.axhline(2.5, color='gray', linestyle='--', label='Expected total  = 2.5')

    plt.title("Species pool balance over time")
    plt.xlabel("Time Step")
    plt.ylabel("Total Concentration")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_conc_electrons(df):
    plt.figure()
    ox, red = compute_electrons(df)
    plt.plot(ox, label='ox')
    plt.plot(red, label='red')
    plt.plot(red + ox, label='total')
    plt.title("Balance of electrons")
    plt.xlabel("Time Step (ms)")
    plt.ylabel("Total Concentration (mol/molChl)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_single_solution(df, species_group:list) -> None:
    """
    Helper function to plot species concentrations from a single solution object.

    Parameters:
        df (data frame): Solution object from scipy.integrate.solve_ivp
        species_group (List[List[str]]): List of groups, where each group is a list of species names
    """
    for group in species_group:
        plt.figure(figsize=(8, 5))

        # Calculate base index for this group
        base_index = sum(len(g) for g in species_group[:species_group.index(group)])

        # Plot each species in the group
        for i, species in enumerate(group):
            species_index = base_index + i
            color = species_colors.get(species, None)

            plt.plot(
                df[species],
                label=custom_labels.get(species, species),
                color=color
            )

        # Set labels and show plot
        plt.xlabel("Time (ms)")
        plt.ylabel("Concentration (mmol/mol Chl)")
        plt.legend(loc="best")
        plt.show()


