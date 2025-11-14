from modelbase.ode import Simulator
from tqdm.notebook import tqdm
import itertools as it


def DIRK(s: Simulator, time_relax: float, relax_pfd: float,
         pfd_illumination: float, pre_dark_time: float, post_dark_time):
    t = it.accumulate([pre_dark_time, time_relax, post_dark_time])
    for time, pfd in tqdm(zip(t
            , [pfd_illumination, relax_pfd, pfd_illumination])):
        s.update_parameter("Light_intensity", pfd)
        print(pfd)
        print(time)
        s.simulate(time)


def PIRK(s: Simulator, time_relax: float, ss_pfd: float,ss_time:float, time_pulse: float,
         pfd_dark: float, pfd_pulse: float, time_delay=bool):

    s.update_parameter("Light_intensity", ss_pfd)
    s.simulate(ss_time)

    pfds = list([pfd_pulse, pfd_dark] * 4)
    if time_delay == True:
        t = list(it.accumulate(it.chain.from_iterable((time_pulse, time_relax * i) for i in [1, 2, 4, 0.5])))
    if time_delay == False:
        t = list(it.accumulate(it.chain.from_iterable((time_pulse, time_relax) for i in range(4))))

    for i in range(len(t)):
        t[i] = t[i] + 160

    for t, pfd in tqdm(zip(t, pfds)):
        s.update_parameter("Light_intensity", pfd)
        s.simulate(t)
        print(t, pfd)



