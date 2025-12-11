import subprocess
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r'/home/torsten/Work/Software/BDNNsim')
from bdnn_simulator import *

rnd_seed = int(np.random.choice(np.arange(1, 1e8), 1)[0])
# rnd_seed = 42069


bd_sim = bdnn_simulator(s_species = 3,  # number of starting species
                        rangeSP = [100, 1000],  # min/max size data set
                        minExtant_SP = 3, # minimum number of extant lineages
                        root_r = [20., 20.],  # range root ages
                        rangeL = [0.3, 0.3],  # range of birth rates
                        rangeM = [0.05, 0.05],  # range of death rates
                        scale = 20.,
                        # fixed_Mtt = np.array([[35., 0.01], [0.0, 0.1]]),
                        n_cont_traits = [0, 0],  # number of continuous traits
                        cont_traits_sigma_clado = [0, 0],
                        cont_traits_sigma = [0, 0], # evolutionary rates for continuous traits
                        n_cat_traits = [1, 1],
                        n_cat_traits_states = [3, 3], # range number of states for categorical trait
                        cat_traits_diag = None,
                        cat_traits_evolve = False,
                        # cat_traits_Q = np.array([[0, 0, 0], [0, 0, 0]]),
                        cat_traits_effect = np.array([[1, 1, 1], [1, 1, 1]]),
                        cat_traits_effect_decr_incr = np.array([[False, False],[False, False]]),
                        cat_traits_min_freq = [0.1],
                        # sp_env_file = "./temp_series.csv", # Path to environmental file influencing speciation
                        # sp_env_eff = [1.2, 1.2],  # range environmental effect on speciation rate
                        # env_effect_cat_trait = [[1, 1, 1], [None]],
                        seed = rnd_seed)

# Birth-death simulation
res_bd = bd_sim.run_simulation(verbose = True)

ts_te_out = pd.DataFrame(res_bd['ts_te'])
ts_te_out.to_csv("./ts_te.csv")

species_id_out = pd.DataFrame(res_bd['species_trait_list'])
species_id_out.to_csv("./species_id_list.csv")



## Environment effect

import subprocess
import sys
import numpy as np
import pandas as pd

from bdnn_simulator import *

sim_white = EnvironmentSimulator(root=20, scale=20, model="white", mean=0, sd=1)
sim_bm = EnvironmentSimulator(root=20, scale=20, model="BM", mean=0, sd=1)
sim_bmean = EnvironmentSimulator(root=20, scale=20, model="bmean", mean=0, sd=1)
sim_trend = EnvironmentSimulator(root=20, scale=20, model="trend", mean=0, sd=1, slope=0.001)
sim_shift = EnvironmentSimulator(root=20, scale=20, model="shift", mean=0, sd=0.1, shift=[200, 350])

df_white = sim_white.simulate_env()
sim_white.plot()   # plots stored simulation

df_bm = sim_bm.simulate_env()
sim_bm.plot()   # plots stored simulation

df_bmean = sim_bmean.simulate_env()
sim_bmean.plot()   # plots stored simulation

df_trend = sim_trend.simulate_env()
sim_trend.plot()   # plots stored simulation

df_shift = sim_shift.simulate_env()
sim_shift.plot()   # plots stored simulation





sys.path.insert(0, r'/home/torsten/Work/Software/BDNNsim')

import sys
import numpy as np
import pandas as pd
from bdnn_simulator import *

rnd_seed = int(np.random.choice(np.arange(1, 1e8), 1)[0])
# rnd_seed = 42069


bd_sim_trend = bdnn_simulator(s_species = 3,  # number of starting species
                        rangeSP = [100, 1000],  # min/max size data set
                        minExtant_SP = 3, # minimum number of extant lineages
                        root_r = [20., 20.],  # range root ages
                        rangeL = [0.35, 0.35],  # range of birth rates
                        rangeM = [0.05, 0.05],  # range of death rates
                        scale = 20.,
                        # fixed_Mtt = np.array([[35., 0.01], [0.0, 0.1]]),
                        n_cont_traits = [0, 0],  # number of continuous traits
                        cont_traits_sigma_clado = [0, 0],
                        cont_traits_sigma = [0, 0], # evolutionary rates for continuous traits
                        n_cat_traits = [1, 1],
                        n_cat_traits_states = [3, 3], # range number of states for categorical trait
                        cat_traits_diag = None,
                        cat_traits_evolve = False,
                        # cat_traits_Q = np.array([[0, 0, 0], [0, 0, 0]]),
                        cat_traits_effect = np.array([[1, 1, 1], [1, 1, 1]]),
                        cat_traits_effect_decr_incr = np.array([[False, False],[False, False]]),
                        cat_traits_min_freq = [0.1],
                        env_sim = True,
                        env_sim_model = "trend",
                        env_sim_trend_slope = 0.001,
                        env_sim_mean = 0,
                        env_sim_sd = 1,
                        # sp_env_file = "./temp_series.csv", # Path to environmental file influencing speciation
                        sp_env_eff = [1.2, 1.2],  # range environmental effect on speciation rate
                        env_effect_cat_trait = [[1, 0, 0], [None]],
                        seed = rnd_seed)

# Birth-death simulation
res_bd_env_trend = bd_sim_trend.run_simulation(verbose = True)

ts_te_out_env = pd.DataFrame(res_bd_env['ts_te'])
ts_te_out_env.to_csv("./ts_te_env.csv")

species_id_out_env = pd.DataFrame(res_bd_env['species_trait_list'])
species_id_out_env.to_csv("./species_id_list_env.csv")
