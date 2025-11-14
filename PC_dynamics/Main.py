from importlib.metadata import version
from matplotlib import pyplot as plt
from modelbase.ode import Simulator
from PC_dynamics.protcols import DIRK, PIRK
from PC_dynamics.utils import plot_conservation, plot_conc_electrons, plot_single_solution, species_groups

for pkg in ("modelbase",):
    print(f"{pkg:<10} {version(pkg)}")


from models import get_model
m = get_model()

# dark adapted state
y0 = dict(FBoP700r=2.5, FBoP700rPCr=0.0, FBoP700rPCo=0.0, FBrP700o=0.0, FBrP700oPCr=0.0, FBrP700oPCo=0.0, FBoP700o=0.0,
          FBoP700oPCr=0.0, FBoP700oPCo=0.0, FBrP700rPCo=0.0, PCo=4, PCr=0.0, FDo=5.0, FDr=0.0, PQ=17.5, PQH2=0.0, P=0.0,
          FBrP700r=0.0, FBrP700rPCr=0.0)

# test simulation and plotting

pfd = 10
m.update_parameter("Light_intensity",pfd)

fig, ax = (
    Simulator(m)
    .initialise(y0)
    .simulate_and(t_end=5)
    .plot(
        xlabel="time (s)",
        ylabel="concentration (mmol/mol chl)",
        title="Concentrations over time ",
        figure_kwargs={"figsize": (10, 5)},
    )
)
plt.show()

result = (
    Simulator(m)
    .initialise(y0)
    .simulate_and(t_end=5)
)
df = result.get_results_df()

# plot simulation results

plot_conservation(df)

plot_conc_electrons(df)

plot_single_solution(df,species_groups)

# simulate DIRK experiment
s1 = Simulator(m)
s1.initialise(y0)
s1.clear_results()
DIRK(s1, time_relax=5, relax_pfd=0, pfd_illumination=300,
     pre_dark_time=10, post_dark_time=10)

c = s1.get_full_results_df()
v = s1.get_fluxes_df()

a0 = c["FBrP700o"]
a1 = c["FBrP700oPCr"]
a2 = c["FBrP700oPCo"]
a3 = c["FBoP700o"]
a4 = c["FBoP700oPCr"]
a5 = c["FBoP700oPCo"]

aT = m.get_parameter('PSI_tot')

p700_ox = a0 + a1 + a2 + a3 + a4 + a5
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(p700_ox / aT, label="P700$^{+}$", color="darkblue")
# ax.plot(c["FDo"]/get_pool_size()['Fd'], label = "Fd", color="red")
# ax.plot(c["PCo"]/get_pool_size()['PC'], label = "PC", color="cyan")

ax.set(
    xlim=(9.5, 16),
    xlabel=("Time (s)"),
    ylabel=("P700 oxidation level"))

ax.axvspan(10, 15, color='grey', alpha=0.5, lw=0)
ax.legend()

plt.show()

# simulate PIRK experiment

s1 = Simulator(m)
s1.initialise(y0)
s1.clear_results()

PIRK(s1, time_relax=0.05, ss_pfd=300, ss_time=160,time_pulse=0.001,
     pfd_dark=0, pfd_pulse=1180, time_delay=True)

c= s1.get_full_results_df()
v = s1.get_fluxes_df()

a0 = c["FBrP700o"]
a1 = c["FBrP700oPCr"]
a2 = c["FBrP700oPCo"]
a3 = c["FBoP700o"]
a4 = c["FBoP700oPCr"]
a5 = c["FBoP700oPCo"]

aT = m.get_parameter('PSI_tot')

p700_ox = a0 + a1 + a2 + a3 + a4 + a5
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(p700_ox / aT, label="P700$^{+}$", color="darkblue")
# ax.plot(c["FDo"]/get_pool_size()['Fd'], label = "Fd", color="red")
# ax.plot(c["PCo"]/get_pool_size()['PC'], label = "PC", color="cyan")

ax.set(
    xlim=(159.95, 160.5),
    # ylim=(0.16,0.5),
    xlabel=("time (s))"),
    ylabel=(r"P700$^{+}$ fraction (-)"))
ax.axvspan(160, 160.39, color='grey', alpha=0.5, lw=0)
ax.legend(loc='upper right')  # slightly outside the plot

plt.show()