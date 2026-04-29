### Environmental dependency in speciation

import subprocess
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r'/home/torsten/Work/Software/BDNNsim')
from bdnn_simulator import *
from analysis.plots import *

rnd_seed = int(np.random.choice(np.arange(1, 1e8), 1)[0])
# rnd_seed = 42006789


bd_sim_env_dep_lamb = bdnn_simulator(s_species = 2,  # number of starting species
                        rangeSP = [100, 1000],  # min/max size data set
                        minExtant_SP = 3, # minimum number of extant lineages
                        root_r = [20., 20.],  # range root ages
                        rangeL = [0.2, 0.2],  # range of birth rates
                        rangeM = [0.05, 0.05],  # range of death rates
                        scale = 20.,
                        # fixed_Mtt = np.array([[35., 0.01], [0.0, 0.1]]),
                        n_cont_traits = [0, 0],  # number of continuous traits
                        cont_traits_sigma_clado = [0, 0],
                        cont_traits_sigma = [0, 0], # evolutionary rates for continuous traits
                        n_cat_traits=[1, 1],
                        n_cat_traits_states=[2, 2],  # range number of states for categorical trait
                        cat_traits_diag=None,
                        cat_traits_evolve=False,
                        # cat_traits_Q = np.array([[0, 0], [0, 0]]),
                        cat_traits_effect=np.array([[1., 1.], [1., 1.]]),
                        cat_traits_effect_decr_incr=np.array([[False, False], [False, False]]),
                        cat_traits_min_freq=[0.1],
                        env_sim=True,
                        env_sim_model="BM",
                        env_sim_trend_slope=0.001,
                        env_sim_mean=0,
                        env_sim_sd=0.01,
                        env_sim_shift=[200, 350],
                        env_sim_shift_mag=20,
                        # sp_env_file = "./temp_series.csv", # Path to environmental file influencing speciation
                        sp_env_eff=[1.5, 1.5],  # range environmental effect on speciation rate
                        ex_env_eff=[1., 1.],  # range environmental effect on extinction rate
                        env_effect_by_state_sp=[1.0, 0.0],
                        env_effect_by_state_ex=[0.0, 0.0],
                        divdep_by_state=True,
                        divdep_target_trait_idx=0,
                        divdep_sp_mode="linear",
                        divdep_ex_mode="logistic",
                        divdep_sp_eff=[1.0, 1.0],
                        divdep_ex_eff=[1.0, 1.0],
                        divdep_effect_by_state_sp=[1.0, 1.0],
                        divdep_effect_by_state_ex=[1.0, 1.0],
                        divdep_state_matrix_sp=np.array([
                            [0.0, 0.0],
                            [0.0, 0.0]
                        ]),
                        divdep_state_matrix_ex=np.array([
                            [0.0, 0.0],
                            [0.0, 0.0]
                        ]),
                        # env_effect_cat_trait=[[1, 0, 0], [None]],
                        # causal_matrix_l = np.array([[0., 1., 0.], [0., 0., 0.], [0., 0., 0.]]),
                        # causal_matrix_m = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),
                        # causal_matrix_m = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),
                        # sp_env_file = "./temp_series.csv", # Path to environmental file influencing speciation
                        # sp_env_eff = [1.2, 1.2],  # range environmental effect on speciation rate
                        # env_effect_cat_trait = [[1, 1, 1], [None]],
                        seed = rnd_seed)

# Birth-death simulation
res_bd_env_dep_lamb = bd_sim_env_dep_lamb.run_simulation(verbose = True)

out_env_dep_lamb = plot_combined_diversification_figure(
    res_bd_env_dep_lamb,
    trait_idx=0,
    env_which="sp",     # or "ex"
    summary="mean",     # or "median"
    stacked_diversity=False,
    save_path="figs/env_dep_lamb.png",
    dpi=300
)





### Environmental dependency in extinction

import subprocess
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r'/home/torsten/Work/Software/BDNNsim')
from bdnn_simulator import *
from analysis.plots import *

rnd_seed = int(np.random.choice(np.arange(1, 1e8), 1)[0])
# rnd_seed = 42006789


bd_sim_env_dep_mu = bdnn_simulator(s_species = 2,  # number of starting species
                        rangeSP = [100, 1000],  # min/max size data set
                        minExtant_SP = 3, # minimum number of extant lineages
                        root_r = [20., 20.],  # range root ages
                        rangeL = [0.2, 0.2],  # range of birth rates
                        rangeM = [0.05, 0.05],  # range of death rates
                        scale = 20.,
                        # fixed_Mtt = np.array([[35., 0.01], [0.0, 0.1]]),
                        n_cont_traits = [0, 0],  # number of continuous traits
                        cont_traits_sigma_clado = [0, 0],
                        cont_traits_sigma = [0, 0], # evolutionary rates for continuous traits
                        n_cat_traits=[1, 1],
                        n_cat_traits_states=[2, 2],  # range number of states for categorical trait
                        cat_traits_diag=None,
                        cat_traits_evolve=False,
                        # cat_traits_Q = np.array([[0, 0], [0, 0]]),
                        cat_traits_effect=np.array([[1., 1.], [1., 1.]]),
                        cat_traits_effect_decr_incr=np.array([[False, False], [False, False]]),
                        cat_traits_min_freq=[0.1],
                        env_sim=True,
                        env_sim_model="BM",
                        env_sim_trend_slope=0.001,
                        env_sim_mean=0,
                        env_sim_sd=0.01,
                        env_sim_shift=[200, 350],
                        env_sim_shift_mag=20,
                        # sp_env_file = "./temp_series.csv", # Path to environmental file influencing speciation
                        sp_env_eff=[1., 1.],  # range environmental effect on speciation rate
                        ex_env_eff=[1.5, 1.5],  # range environmental effect on extinction rate
                        env_effect_by_state_sp=[0.0, 0.0],
                        env_effect_by_state_ex=[1.0, 0.0],
                        divdep_by_state=True,
                        divdep_target_trait_idx=0,
                        divdep_sp_mode="linear",
                        divdep_ex_mode="logistic",
                        divdep_sp_eff=[1.0, 1.0],
                        divdep_ex_eff=[1.0, 1.0],
                        divdep_effect_by_state_sp=[1.0, 1.0],
                        divdep_effect_by_state_ex=[1.0, 1.0],
                        divdep_state_matrix_sp=np.array([
                            [0.0, 0.0],
                            [0.0, 0.0]
                        ]),
                        divdep_state_matrix_ex=np.array([
                            [0.0, 0.0],
                            [0.0, 0.0]
                        ]),
                        # env_effect_cat_trait=[[1, 0, 0], [None]],
                        # causal_matrix_l = np.array([[0., 1., 0.], [0., 0., 0.], [0., 0., 0.]]),
                        # causal_matrix_m = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),
                        # causal_matrix_m = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),
                        # sp_env_file = "./temp_series.csv", # Path to environmental file influencing speciation
                        # sp_env_eff = [1.2, 1.2],  # range environmental effect on speciation rate
                        # env_effect_cat_trait = [[1, 1, 1], [None]],
                        seed = rnd_seed)

# Birth-death simulation
res_bd_env_dep_mu = bd_sim_env_dep_mu.run_simulation(verbose = True)

out_env_dep_mu = plot_combined_diversification_figure(
    res_bd_env_dep_mu,
    trait_idx=0,
    env_which="sp",     # or "ex"
    summary="mean",     # or "median"
    stacked_diversity=False,
    save_path="figs/env_dep_mu.png",
    dpi=300
)


### Environmental dependency in both speciation and extinction

import subprocess
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r'/home/torsten/Work/Software/BDNNsim')
from bdnn_simulator import *
from analysis.plots import *

rnd_seed = int(np.random.choice(np.arange(1, 1e8), 1)[0])
# rnd_seed = 42006789


bd_sim_env_dep_lamb_mu = bdnn_simulator(s_species = 2,  # number of starting species
                        rangeSP = [100, 1000],  # min/max size data set
                        minExtant_SP = 3, # minimum number of extant lineages
                        root_r = [20., 20.],  # range root ages
                        rangeL = [0.2, 0.2],  # range of birth rates
                        rangeM = [0.05, 0.05],  # range of death rates
                        scale = 20.,
                        # fixed_Mtt = np.array([[35., 0.01], [0.0, 0.1]]),
                        n_cont_traits = [0, 0],  # number of continuous traits
                        cont_traits_sigma_clado = [0, 0],
                        cont_traits_sigma = [0, 0], # evolutionary rates for continuous traits
                        n_cat_traits=[1, 1],
                        n_cat_traits_states=[2, 2],  # range number of states for categorical trait
                        cat_traits_diag=None,
                        cat_traits_evolve=False,
                        # cat_traits_Q = np.array([[0, 0], [0, 0]]),
                        cat_traits_effect=np.array([[1., 1.], [1., 1.]]),
                        cat_traits_effect_decr_incr=np.array([[False, False], [False, False]]),
                        cat_traits_min_freq=[0.1],
                        env_sim=True,
                        env_sim_model="BM",
                        env_sim_trend_slope=0.001,
                        env_sim_mean=0,
                        env_sim_sd=0.01,
                        env_sim_shift=[200, 350],
                        env_sim_shift_mag=20,
                        # sp_env_file = "./temp_series.csv", # Path to environmental file influencing speciation
                        sp_env_eff=[1.5, 1.5],  # range environmental effect on speciation rate
                        ex_env_eff=[1.5, 1.5],  # range environmental effect on extinction rate
                        env_effect_by_state_sp=[1.0, 0.0],
                        env_effect_by_state_ex=[1.0, 0.0],
                        divdep_by_state=True,
                        divdep_target_trait_idx=0,
                        divdep_sp_mode="linear",
                        divdep_ex_mode="logistic",
                        divdep_sp_eff=[1.0, 1.0],
                        divdep_ex_eff=[1.0, 1.0],
                        divdep_effect_by_state_sp=[1.0, 1.0],
                        divdep_effect_by_state_ex=[1.0, 1.0],
                        divdep_state_matrix_sp=np.array([
                            [0.0, 0.0],
                            [0.0, 0.0]
                        ]),
                        divdep_state_matrix_ex=np.array([
                            [0.0, 0.0],
                            [0.0, 0.0]
                        ]),
                        # env_effect_cat_trait=[[1, 0, 0], [None]],
                        # causal_matrix_l = np.array([[0., 1., 0.], [0., 0., 0.], [0., 0., 0.]]),
                        # causal_matrix_m = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),
                        # causal_matrix_m = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),
                        # sp_env_file = "./temp_series.csv", # Path to environmental file influencing speciation
                        # sp_env_eff = [1.2, 1.2],  # range environmental effect on speciation rate
                        # env_effect_cat_trait = [[1, 1, 1], [None]],
                        seed = rnd_seed)

# Birth-death simulation
res_bd_env_dep_lamb_mu = bd_sim_env_dep_lamb_mu.run_simulation(verbose = True)

out_env_dep_lamb_mu = plot_combined_diversification_figure(
    res_bd_env_dep_lamb_mu,
    trait_idx=0,
    env_which="sp",     # or "ex"
    summary="mean",     # or "median"
    stacked_diversity=False,
    save_path="figs/env_dep_lamb_mu.png",
    dpi=300
)



### Self diversity dependency in speciation

import subprocess
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r'/home/torsten/Work/Software/BDNNsim')
from bdnn_simulator import *
from analysis.plots import *

rnd_seed = int(np.random.choice(np.arange(1, 1e8), 1)[0])
# rnd_seed = 42006789


bd_sim_div_dep_self_lamb = bdnn_simulator(s_species = 2,  # number of starting species
                        rangeSP = [100, 1000],  # min/max size data set
                        minExtant_SP = 3, # minimum number of extant lineages
                        root_r = [20., 20.],  # range root ages
                        rangeL = [0.2, 0.2],  # range of birth rates
                        rangeM = [0.05, 0.05],  # range of death rates
                        scale = 20.,
                        # fixed_Mtt = np.array([[35., 0.01], [0.0, 0.1]]),
                        n_cont_traits = [0, 0],  # number of continuous traits
                        cont_traits_sigma_clado = [0, 0],
                        cont_traits_sigma = [0, 0], # evolutionary rates for continuous traits
                        n_cat_traits=[1, 1],
                        n_cat_traits_states=[2, 2],  # range number of states for categorical trait
                        cat_traits_diag=None,
                        cat_traits_evolve=False,
                        # cat_traits_Q = np.array([[0, 0], [0, 0]]),
                        cat_traits_effect=np.array([[1., 1.], [1., 1.]]),
                        cat_traits_effect_decr_incr=np.array([[False, False], [False, False]]),
                        cat_traits_min_freq=[0.1],
                        env_sim=True,
                        env_sim_model="BM",
                        env_sim_trend_slope=0.001,
                        env_sim_mean=0,
                        env_sim_sd=0.01,
                        env_sim_shift=[200, 350],
                        env_sim_shift_mag=20,
                        # sp_env_file = "./temp_series.csv", # Path to environmental file influencing speciation
                        sp_env_eff=[1., 1.],  # range environmental effect on speciation rate
                        ex_env_eff=[1., 1.],  # range environmental effect on extinction rate
                        env_effect_by_state_sp=[0.0, 0.0],
                        env_effect_by_state_ex=[0.0, 0.0],
                        divdep_by_state=True,
                        divdep_target_trait_idx=0,
                        divdep_sp_mode="logistic",
                        divdep_ex_mode="logistic",
                        divdep_sp_eff=[1.0, 1.0],
                        divdep_ex_eff=[1.0, 1.0],
                        divdep_effect_by_state_sp=[1.0, 1.0],
                        divdep_effect_by_state_ex=[1.0, 1.0],
                        divdep_state_matrix_sp=np.array([
                            [0.5, 0.0], # [self dep on 0, lineage 1 on 0]
                            [0.0, 0.0] # [lineage 0 on 1, self dep on 1]
                        ]),
                        divdep_state_matrix_ex=np.array([
                            [0.0, 0.0], # [self dep on 0, lineage 1 on 0]
                            [0.0, 0.0] # [lineage 0 on 1, self dep on 1]
                        ]),
                        # env_effect_cat_trait=[[1, 0, 0], [None]],
                        # causal_matrix_l = np.array([[0., 1., 0.], [0., 0., 0.], [0., 0., 0.]]),
                        # causal_matrix_m = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),
                        # causal_matrix_m = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),
                        # sp_env_file = "./temp_series.csv", # Path to environmental file influencing speciation
                        # sp_env_eff = [1.2, 1.2],  # range environmental effect on speciation rate
                        # env_effect_cat_trait = [[1, 1, 1], [None]],
                        seed = rnd_seed)

# Birth-death simulation
res_bd_div_dep_self_lamb = bd_sim_div_dep_self_lamb.run_simulation(verbose = True)

out_div_dep_self_lamb = plot_combined_diversification_figure(
    res_bd_div_dep_self_lamb,
    trait_idx=0,
    env_which="sp",     # or "ex"
    summary="mean",     # or "median"
    stacked_diversity=False,
    save_path="figs/div_dep_self_lamb.png",
    dpi=300
)


### Self diversity dependency in extinction

import subprocess
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r'/home/torsten/Work/Software/BDNNsim')
from bdnn_simulator import *
from analysis.plots import *

rnd_seed = int(np.random.choice(np.arange(1, 1e8), 1)[0])
# rnd_seed = 42006789


bd_sim_div_dep_self_mu = bdnn_simulator(s_species = 2,  # number of starting species
                        rangeSP = [100, 1000],  # min/max size data set
                        minExtant_SP = 3, # minimum number of extant lineages
                        root_r = [20., 20.],  # range root ages
                        rangeL = [0.2, 0.2],  # range of birth rates
                        rangeM = [0.05, 0.05],  # range of death rates
                        scale = 20.,
                        # fixed_Mtt = np.array([[35., 0.01], [0.0, 0.1]]),
                        n_cont_traits = [0, 0],  # number of continuous traits
                        cont_traits_sigma_clado = [0, 0],
                        cont_traits_sigma = [0, 0], # evolutionary rates for continuous traits
                        n_cat_traits=[1, 1],
                        n_cat_traits_states=[2, 2],  # range number of states for categorical trait
                        cat_traits_diag=None,
                        cat_traits_evolve=False,
                        # cat_traits_Q = np.array([[0, 0], [0, 0]]),
                        cat_traits_effect=np.array([[1., 1.], [1., 1.]]),
                        cat_traits_effect_decr_incr=np.array([[False, False], [False, False]]),
                        cat_traits_min_freq=[0.1],
                        env_sim=True,
                        env_sim_model="BM",
                        env_sim_trend_slope=0.001,
                        env_sim_mean=0,
                        env_sim_sd=0.01,
                        env_sim_shift=[200, 350],
                        env_sim_shift_mag=20,
                        # sp_env_file = "./temp_series.csv", # Path to environmental file influencing speciation
                        sp_env_eff=[1., 1.],  # range environmental effect on speciation rate
                        ex_env_eff=[1., 1.],  # range environmental effect on extinction rate
                        env_effect_by_state_sp=[0.0, 0.0],
                        env_effect_by_state_ex=[0.0, 0.0],
                        divdep_by_state=True,
                        divdep_target_trait_idx=0,
                        divdep_sp_mode="logistic",
                        divdep_ex_mode="logistic",
                        divdep_sp_eff=[1.0, 1.0],
                        divdep_ex_eff=[1.0, 1.0],
                        divdep_effect_by_state_sp=[1.0, 1.0],
                        divdep_effect_by_state_ex=[1.0, 1.0],
                        divdep_state_matrix_sp=np.array([
                            [0.0, 0.0], # [self dep on 0, lineage 1 on 0]
                            [0.0, 0.0] # [lineage 0 on 1, self dep on 1]
                        ]),
                        divdep_state_matrix_ex=np.array([
                            [0.0, 0.0], # [self dep on 0, lineage 1 on 0]
                            [0.0, -1.0] # [lineage 0 on 1, self dep on 1]
                        ]),
                        # env_effect_cat_trait=[[1, 0, 0], [None]],
                        # causal_matrix_l = np.array([[0., 1., 0.], [0., 0., 0.], [0., 0., 0.]]),
                        # causal_matrix_m = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),
                        # causal_matrix_m = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),
                        # sp_env_file = "./temp_series.csv", # Path to environmental file influencing speciation
                        # sp_env_eff = [1.2, 1.2],  # range environmental effect on speciation rate
                        # env_effect_cat_trait = [[1, 1, 1], [None]],
                        seed = rnd_seed)

# Birth-death simulation
res_bd_div_dep_self_mu = bd_sim_div_dep_self_mu.run_simulation(verbose = True)

out_div_dep_self_mu = plot_combined_diversification_figure(
    res_bd_div_dep_self_mu,
    trait_idx=0,
    env_which="sp",     # or "ex"
    summary="mean",     # or "median"
    stacked_diversity=False,
    save_path="figs/div_dep_self_mu.png",
    dpi=300
)



### Self diversity dependency in both speciation and extinction

import subprocess
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r'/home/torsten/Work/Software/BDNNsim')
from bdnn_simulator import *
from analysis.plots import *

rnd_seed = int(np.random.choice(np.arange(1, 1e8), 1)[0])
# rnd_seed = 42006789


bd_sim_div_dep_self_lambda_mu = bdnn_simulator(s_species = 2,  # number of starting species
                        rangeSP = [100, 1000],  # min/max size data set
                        minExtant_SP = 3, # minimum number of extant lineages
                        root_r = [20., 20.],  # range root ages
                        rangeL = [0.2, 0.2],  # range of birth rates
                        rangeM = [0.05, 0.05],  # range of death rates
                        scale = 20.,
                        # fixed_Mtt = np.array([[35., 0.01], [0.0, 0.1]]),
                        n_cont_traits = [0, 0],  # number of continuous traits
                        cont_traits_sigma_clado = [0, 0],
                        cont_traits_sigma = [0, 0], # evolutionary rates for continuous traits
                        n_cat_traits=[1, 1],
                        n_cat_traits_states=[2, 2],  # range number of states for categorical trait
                        cat_traits_diag=None,
                        cat_traits_evolve=False,
                        # cat_traits_Q = np.array([[0, 0], [0, 0]]),
                        cat_traits_effect=np.array([[1., 1.], [1., 1.]]),
                        cat_traits_effect_decr_incr=np.array([[False, False], [False, False]]),
                        cat_traits_min_freq=[0.1],
                        env_sim=True,
                        env_sim_model="BM",
                        env_sim_trend_slope=0.001,
                        env_sim_mean=0,
                        env_sim_sd=0.01,
                        env_sim_shift=[200, 350],
                        env_sim_shift_mag=20,
                        # sp_env_file = "./temp_series.csv", # Path to environmental file influencing speciation
                        sp_env_eff=[1., 1.],  # range environmental effect on speciation rate
                        ex_env_eff=[1., 1.],  # range environmental effect on extinction rate
                        env_effect_by_state_sp=[0.0, 0.0],
                        env_effect_by_state_ex=[0.0, 0.0],
                        divdep_by_state=True,
                        divdep_target_trait_idx=0,
                        divdep_sp_mode="logistic",
                        divdep_ex_mode="logistic",
                        divdep_sp_eff=[1.0, 1.0],
                        divdep_ex_eff=[1.0, 1.0],
                        divdep_effect_by_state_sp=[1.5, 1.5],
                        divdep_effect_by_state_ex=[1.5, 1.5],
                        divdep_state_matrix_sp=np.array([
                            [0.0, 0.0], # [self dep on 0, state 0 depends on diversity of state 1]
                            [0.0, 0.5] # [state 1 depends on diversity of state 0, self dep on 1]
                        ]),
                        divdep_state_matrix_ex=np.array([
                            [0.0, 0.0], # [self dep on 0, state 0 depends on diversity of state 1]
                            [0.0, -1.0] # [state 1 depends on diversity of state 0, self dep on 1]
                        ]),
                        # env_effect_cat_trait=[[1, 0, 0], [None]],
                        # causal_matrix_l = np.array([[0., 1., 0.], [0., 0., 0.], [0., 0., 0.]]),
                        # causal_matrix_m = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),
                        # causal_matrix_m = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),
                        # sp_env_file = "./temp_series.csv", # Path to environmental file influencing speciation
                        # sp_env_eff = [1.2, 1.2],  # range environmental effect on speciation rate
                        # env_effect_cat_trait = [[1, 1, 1], [None]],
                        seed = rnd_seed)

# Birth-death simulation
res_bd_div_dep_self_lambda_mu = bd_sim_div_dep_self_lambda_mu.run_simulation(verbose = True)

out_div_dep_self_lambda_mu = plot_combined_diversification_figure(
    res_bd_div_dep_self_lambda_mu,
    trait_idx=0,
    env_which="sp",     # or "ex"
    summary="mean",     # or "median"
    stacked_diversity=False,
    save_path="figs/div_dep_self_lambda_mu.png",
    dpi=300
)




### Cross diversity dependency in speciation

import subprocess
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r'/home/torsten/Work/Software/BDNNsim')
from bdnn_simulator import *
from analysis.plots import *

rnd_seed = int(np.random.choice(np.arange(1, 1e8), 1)[0])
# rnd_seed = 42006789


bd_sim_div_dep_cross_lambda = bdnn_simulator(s_species = 2,  # number of starting species
                        rangeSP = [100, 1000],  # min/max size data set
                        minExtant_SP = 3, # minimum number of extant lineages
                        root_r = [20., 20.],  # range root ages
                        rangeL = [0.2, 0.2],  # range of birth rates
                        rangeM = [0.05, 0.05],  # range of death rates
                        scale = 20.,
                        # fixed_Mtt = np.array([[35., 0.01], [0.0, 0.1]]),
                        n_cont_traits = [0, 0],  # number of continuous traits
                        cont_traits_sigma_clado = [0, 0],
                        cont_traits_sigma = [0, 0], # evolutionary rates for continuous traits
                        n_cat_traits=[1, 1],
                        n_cat_traits_states=[2, 2],  # range number of states for categorical trait
                        cat_traits_diag=None,
                        cat_traits_evolve=False,
                        # cat_traits_Q = np.array([[0, 0], [0, 0]]),
                        cat_traits_effect=np.array([[1., 1.], [1., 1.]]),
                        cat_traits_effect_decr_incr=np.array([[False, False], [False, False]]),
                        cat_traits_min_freq=[0.1],
                        env_sim=True,
                        env_sim_model="BM",
                        env_sim_trend_slope=0.001,
                        env_sim_mean=0,
                        env_sim_sd=0.01,
                        env_sim_shift=[200, 350],
                        env_sim_shift_mag=20,
                        # sp_env_file = "./temp_series.csv", # Path to environmental file influencing speciation
                        sp_env_eff=[1., 1.],  # range environmental effect on speciation rate
                        ex_env_eff=[1., 1.],  # range environmental effect on extinction rate
                        env_effect_by_state_sp=[0.0, 0.0],
                        env_effect_by_state_ex=[0.0, 0.0],
                        divdep_by_state=True,
                        divdep_target_trait_idx=0,
                        divdep_sp_mode="logistic",
                        divdep_ex_mode="logistic",
                        divdep_sp_eff=[1.0, 1.0],
                        divdep_ex_eff=[1.0, 1.0],
                        divdep_effect_by_state_sp=[1.5, 1.5],
                        divdep_effect_by_state_ex=[1.5, 1.5],
                        divdep_state_matrix_sp=np.array([
                            [0.0, 1.0], # [self dep on 0, state 0 depends on diversity of state 1], positive values mean inverse relationship, needs fixing
                            [0.0, 0.0] # [state 1 depends on diversity of state 0, self dep on 1], positive values mean inverse relationship, needs fixing
                        ]),
                        divdep_state_matrix_ex=np.array([
                            [0.0, 0.0], # [self dep on 0, state 0 depends on diversity of state 1], positive values mean inverse relationship, needs fixing
                            [0.0, 0.0] # [state 1 depends on diversity of state 0, self dep on 1], positive values mean inverse relationship, needs fixing
                        ]),
                        # env_effect_cat_trait=[[1, 0, 0], [None]],
                        # causal_matrix_l = np.array([[0., 1., 0.], [0., 0., 0.], [0., 0., 0.]]),
                        # causal_matrix_m = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),
                        # causal_matrix_m = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),
                        # sp_env_file = "./temp_series.csv", # Path to environmental file influencing speciation
                        # sp_env_eff = [1.2, 1.2],  # range environmental effect on speciation rate
                        # env_effect_cat_trait = [[1, 1, 1], [None]],
                        seed = rnd_seed)

# Birth-death simulation
res_bd_div_dep_cross_lambda = bd_sim_div_dep_cross_lambda.run_simulation(verbose = True)

out_div_dep_cross_lambda = plot_combined_diversification_figure(
    res_bd_div_dep_cross_lambda,
    trait_idx=0,
    env_which="sp",     # or "ex"
    summary="mean",     # or "median"
    stacked_diversity=False,
    save_path="figs/div_dep_cross_lambda.png",
    dpi=300
)



### Cross diversity dependency in extinction

import subprocess
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r'/home/torsten/Work/Software/BDNNsim')
from bdnn_simulator import *
from analysis.plots import *

rnd_seed = int(np.random.choice(np.arange(1, 1e8), 1)[0])
# rnd_seed = 42006789


bd_sim_div_dep_cross_mu = bdnn_simulator(s_species = 2,  # number of starting species
                        rangeSP = [100, 1000],  # min/max size data set
                        minExtant_SP = 3, # minimum number of extant lineages
                        root_r = [20., 20.],  # range root ages
                        rangeL = [0.2, 0.2],  # range of birth rates
                        rangeM = [0.05, 0.05],  # range of death rates
                        scale = 20.,
                        # fixed_Mtt = np.array([[35., 0.01], [0.0, 0.1]]),
                        n_cont_traits = [0, 0],  # number of continuous traits
                        cont_traits_sigma_clado = [0, 0],
                        cont_traits_sigma = [0, 0], # evolutionary rates for continuous traits
                        n_cat_traits=[1, 1],
                        n_cat_traits_states=[2, 2],  # range number of states for categorical trait
                        cat_traits_diag=None,
                        cat_traits_evolve=False,
                        # cat_traits_Q = np.array([[0, 0], [0, 0]]),
                        cat_traits_effect=np.array([[1., 1.], [1., 1.]]),
                        cat_traits_effect_decr_incr=np.array([[False, False], [False, False]]),
                        cat_traits_min_freq=[0.1],
                        env_sim=True,
                        env_sim_model="BM",
                        env_sim_trend_slope=0.001,
                        env_sim_mean=0,
                        env_sim_sd=0.01,
                        env_sim_shift=[200, 350],
                        env_sim_shift_mag=20,
                        # sp_env_file = "./temp_series.csv", # Path to environmental file influencing speciation
                        sp_env_eff=[1., 1.],  # range environmental effect on speciation rate
                        ex_env_eff=[1., 1.],  # range environmental effect on extinction rate
                        env_effect_by_state_sp=[0.0, 0.0],
                        env_effect_by_state_ex=[0.0, 0.0],
                        divdep_by_state=True,
                        divdep_target_trait_idx=0,
                        divdep_sp_mode="logistic",
                        divdep_ex_mode="logistic",
                        divdep_sp_eff=[1.0, 1.0],
                        divdep_ex_eff=[1.0, 1.0],
                        divdep_effect_by_state_sp=[1.5, 1.5],
                        divdep_effect_by_state_ex=[1.5, 1.5],
                        divdep_state_matrix_sp=np.array([
                            [0.0, 0.0], # [self dep on 0, state 0 depends on diversity of state 1], positive values mean inverse relationship, needs fixing
                            [0.0, 0.0] # [state 1 depends on diversity of state 0, self dep on 1], positive values mean inverse relationship, needs fixing
                        ]),
                        divdep_state_matrix_ex=np.array([
                            [0.0, 0.0], # [self dep on 0, state 0 depends on diversity of state 1], positive values mean inverse relationship, needs fixing
                            [-1.0, 0.0] # [state 1 depends on diversity of state 0, self dep on 1], positive values mean inverse relationship, needs fixing
                        ]),
                        # env_effect_cat_trait=[[1, 0, 0], [None]],
                        # causal_matrix_l = np.array([[0., 1., 0.], [0., 0., 0.], [0., 0., 0.]]),
                        # causal_matrix_m = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),
                        # causal_matrix_m = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),
                        # sp_env_file = "./temp_series.csv", # Path to environmental file influencing speciation
                        # sp_env_eff = [1.2, 1.2],  # range environmental effect on speciation rate
                        # env_effect_cat_trait = [[1, 1, 1], [None]],
                        seed = rnd_seed)

# Birth-death simulation
res_bd_div_dep_cross_mu = bd_sim_div_dep_cross_mu.run_simulation(verbose = True)

out_div_dep_cross_mu = plot_combined_diversification_figure(
    res_bd_div_dep_cross_mu,
    trait_idx=0,
    env_which="sp",     # or "ex"
    summary="mean",     # or "median"
    stacked_diversity=False,
    save_path="figs/div_dep_cross_mu.png",
    dpi=300
)



### Cross diversity dependency in both speciation and extinction

import subprocess
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r'/home/torsten/Work/Software/BDNNsim')
from bdnn_simulator import *
from analysis.plots import *

rnd_seed = int(np.random.choice(np.arange(1, 1e8), 1)[0])
# rnd_seed = 42006789


bd_sim_div_dep_cross_lambda_mu = bdnn_simulator(s_species = 2,  # number of starting species
                        rangeSP = [100, 1000],  # min/max size data set
                        minExtant_SP = 3, # minimum number of extant lineages
                        root_r = [20., 20.],  # range root ages
                        rangeL = [0.2, 0.2],  # range of birth rates
                        rangeM = [0.05, 0.05],  # range of death rates
                        scale = 20.,
                        # fixed_Mtt = np.array([[35., 0.01], [0.0, 0.1]]),
                        n_cont_traits = [0, 0],  # number of continuous traits
                        cont_traits_sigma_clado = [0, 0],
                        cont_traits_sigma = [0, 0], # evolutionary rates for continuous traits
                        n_cat_traits=[1, 1],
                        n_cat_traits_states=[2, 2],  # range number of states for categorical trait
                        cat_traits_diag=None,
                        cat_traits_evolve=False,
                        # cat_traits_Q = np.array([[0, 0], [0, 0]]),
                        cat_traits_effect=np.array([[1., 1.], [1., 1.]]),
                        cat_traits_effect_decr_incr=np.array([[False, False], [False, False]]),
                        cat_traits_min_freq=[0.1],
                        env_sim=True,
                        env_sim_model="BM",
                        env_sim_trend_slope=0.001,
                        env_sim_mean=0,
                        env_sim_sd=0.01,
                        env_sim_shift=[200, 350],
                        env_sim_shift_mag=20,
                        # sp_env_file = "./temp_series.csv", # Path to environmental file influencing speciation
                        sp_env_eff=[1., 1.],  # range environmental effect on speciation rate
                        ex_env_eff=[1., 1.],  # range environmental effect on extinction rate
                        env_effect_by_state_sp=[0.0, 0.0],
                        env_effect_by_state_ex=[0.0, 0.0],
                        divdep_by_state=True,
                        divdep_target_trait_idx=0,
                        divdep_sp_mode="logistic",
                        divdep_ex_mode="logistic",
                        divdep_sp_eff=[1.0, 1.0],
                        divdep_ex_eff=[1.0, 1.0],
                        divdep_effect_by_state_sp=[1.5, 1.5],
                        divdep_effect_by_state_ex=[1.5, 1.5],
                        divdep_state_matrix_sp=np.array([
                            [0.0, 0.0], # [self dep on 0, state 0 depends on diversity of state 1], positive values mean inverse relationship, needs fixing
                            [0.5, 0.0] # [state 1 depends on diversity of state 0, self dep on 1], positive values mean inverse relationship, needs fixing
                        ]),
                        divdep_state_matrix_ex=np.array([
                            [0.0, 0.0], # [self dep on 0, state 0 depends on diversity of state 1], positive values mean inverse relationship, needs fixing
                            [-0.5, 0.0] # [state 1 depends on diversity of state 0, self dep on 1], positive values mean inverse relationship, needs fixing
                        ]),
                        # env_effect_cat_trait=[[1, 0, 0], [None]],
                        # causal_matrix_l = np.array([[0., 1., 0.], [0., 0., 0.], [0., 0., 0.]]),
                        # causal_matrix_m = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),
                        # causal_matrix_m = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),
                        # sp_env_file = "./temp_series.csv", # Path to environmental file influencing speciation
                        # sp_env_eff = [1.2, 1.2],  # range environmental effect on speciation rate
                        # env_effect_cat_trait = [[1, 1, 1], [None]],
                        seed = rnd_seed)

# Birth-death simulation
res_bd_div_dep_cross_lambda_mu = bd_sim_div_dep_cross_lambda_mu.run_simulation(verbose = True)

out_div_dep_cross_lambda_mu = plot_combined_diversification_figure(
    res_bd_div_dep_cross_lambda_mu,
    trait_idx=0,
    env_which="sp",     # or "ex"
    summary="mean",     # or "median"
    stacked_diversity=False,
    save_path="figs/div_dep_cross_lambda_mu.png",
    dpi=300
)


### Feedback diversity dependency in speciation

import subprocess
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r'/home/torsten/Work/Software/BDNNsim')
from bdnn_simulator import *
from analysis.plots import *

rnd_seed = int(np.random.choice(np.arange(1, 1e8), 1)[0])
# rnd_seed = 42006789


bd_sim_div_dep_fb_lambda = bdnn_simulator(s_species = 2,  # number of starting species
                        rangeSP = [100, 1000],  # min/max size data set
                        minExtant_SP = 3, # minimum number of extant lineages
                        root_r = [20., 20.],  # range root ages
                        rangeL = [0.2, 0.2],  # range of birth rates
                        rangeM = [0.05, 0.05],  # range of death rates
                        scale = 20.,
                        # fixed_Mtt = np.array([[35., 0.01], [0.0, 0.1]]),
                        n_cont_traits = [0, 0],  # number of continuous traits
                        cont_traits_sigma_clado = [0, 0],
                        cont_traits_sigma = [0, 0], # evolutionary rates for continuous traits
                        n_cat_traits=[1, 1],
                        n_cat_traits_states=[2, 2],  # range number of states for categorical trait
                        cat_traits_diag=None,
                        cat_traits_evolve=False,
                        # cat_traits_Q = np.array([[0, 0], [0, 0]]),
                        cat_traits_effect=np.array([[1., 1.], [1., 1.]]),
                        cat_traits_effect_decr_incr=np.array([[False, False], [False, False]]),
                        cat_traits_min_freq=[0.1],
                        env_sim=True,
                        env_sim_model="BM",
                        env_sim_trend_slope=0.001,
                        env_sim_mean=0,
                        env_sim_sd=0.01,
                        env_sim_shift=[200, 350],
                        env_sim_shift_mag=20,
                        # sp_env_file = "./temp_series.csv", # Path to environmental file influencing speciation
                        sp_env_eff=[1., 1.],  # range environmental effect on speciation rate
                        ex_env_eff=[1., 1.],  # range environmental effect on extinction rate
                        env_effect_by_state_sp=[0.0, 0.0],
                        env_effect_by_state_ex=[0.0, 0.0],
                        divdep_by_state=True,
                        divdep_target_trait_idx=0,
                        divdep_sp_mode="logistic",
                        divdep_ex_mode="logistic",
                        divdep_sp_eff=[1.0, 1.0],
                        divdep_ex_eff=[1.0, 1.0],
                        divdep_effect_by_state_sp=[1.5, 1.5],
                        divdep_effect_by_state_ex=[1.5, 1.5],
                        divdep_state_matrix_sp=np.array([
                            [0.0, 0.5], # [self dep on 0, state 0 depends on diversity of state 1], positive values mean inverse relationship, needs fixing
                            [0.5, 0.0] # [state 1 depends on diversity of state 0, self dep on 1], positive values mean inverse relationship, needs fixing
                        ]),
                        divdep_state_matrix_ex=np.array([
                            [0.0, 0.0], # [self dep on 0, state 0 depends on diversity of state 1], positive values mean inverse relationship, needs fixing
                            [0.0, 0.0] # [state 1 depends on diversity of state 0, self dep on 1], positive values mean inverse relationship, needs fixing
                        ]),
                        # env_effect_cat_trait=[[1, 0, 0], [None]],
                        # causal_matrix_l = np.array([[0., 1., 0.], [0., 0., 0.], [0., 0., 0.]]),
                        # causal_matrix_m = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),
                        # causal_matrix_m = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),
                        # sp_env_file = "./temp_series.csv", # Path to environmental file influencing speciation
                        # sp_env_eff = [1.2, 1.2],  # range environmental effect on speciation rate
                        # env_effect_cat_trait = [[1, 1, 1], [None]],
                        seed = rnd_seed)

# Birth-death simulation
res_bd_div_dep_fb_lambda = bd_sim_div_dep_fb_lambda.run_simulation(verbose = True)

out_div_dep_fb_lambda = plot_combined_diversification_figure(
    res_bd_div_dep_fb_lambda,
    trait_idx=0,
    env_which="sp",     # or "ex"
    summary="mean",     # or "median"
    stacked_diversity=False,
    save_path="figs/div_dep_fb_lambda.png",
    dpi=300
)



### Feedback diversity dependency in extinction

import subprocess
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r'/home/torsten/Work/Software/BDNNsim')
from bdnn_simulator import *
from analysis.plots import *

rnd_seed = int(np.random.choice(np.arange(1, 1e8), 1)[0])
# rnd_seed = 42006789


bd_sim_div_dep_fb_mu = bdnn_simulator(s_species = 2,  # number of starting species
                        rangeSP = [100, 1000],  # min/max size data set
                        minExtant_SP = 3, # minimum number of extant lineages
                        root_r = [20., 20.],  # range root ages
                        rangeL = [0.2, 0.2],  # range of birth rates
                        rangeM = [0.05, 0.05],  # range of death rates
                        scale = 20.,
                        # fixed_Mtt = np.array([[35., 0.01], [0.0, 0.1]]),
                        n_cont_traits = [0, 0],  # number of continuous traits
                        cont_traits_sigma_clado = [0, 0],
                        cont_traits_sigma = [0, 0], # evolutionary rates for continuous traits
                        n_cat_traits=[1, 1],
                        n_cat_traits_states=[2, 2],  # range number of states for categorical trait
                        cat_traits_diag=None,
                        cat_traits_evolve=False,
                        # cat_traits_Q = np.array([[0, 0], [0, 0]]),
                        cat_traits_effect=np.array([[1., 1.], [1., 1.]]),
                        cat_traits_effect_decr_incr=np.array([[False, False], [False, False]]),
                        cat_traits_min_freq=[0.1],
                        env_sim=True,
                        env_sim_model="BM",
                        env_sim_trend_slope=0.001,
                        env_sim_mean=0,
                        env_sim_sd=0.01,
                        env_sim_shift=[200, 350],
                        env_sim_shift_mag=20,
                        # sp_env_file = "./temp_series.csv", # Path to environmental file influencing speciation
                        sp_env_eff=[1., 1.],  # range environmental effect on speciation rate
                        ex_env_eff=[1., 1.],  # range environmental effect on extinction rate
                        env_effect_by_state_sp=[0.0, 0.0],
                        env_effect_by_state_ex=[0.0, 0.0],
                        divdep_by_state=True,
                        divdep_target_trait_idx=0,
                        divdep_sp_mode="logistic",
                        divdep_ex_mode="logistic",
                        divdep_sp_eff=[1.0, 1.0],
                        divdep_ex_eff=[1.0, 1.0],
                        divdep_effect_by_state_sp=[1.5, 1.5],
                        divdep_effect_by_state_ex=[1.5, 1.5],
                        divdep_state_matrix_sp=np.array([
                            [0.0, 0.0], # [self dep on 0, state 0 depends on diversity of state 1], positive values mean inverse relationship, needs fixing
                            [0.0, 0.0] # [state 1 depends on diversity of state 0, self dep on 1], positive values mean inverse relationship, needs fixing
                        ]),
                        divdep_state_matrix_ex=np.array([
                            [0.0, -0.5], # [self dep on 0, state 0 depends on diversity of state 1], positive values mean inverse relationship, needs fixing
                            [-0.5, 0.0] # [state 1 depends on diversity of state 0, self dep on 1], positive values mean inverse relationship, needs fixing
                        ]),
                        # env_effect_cat_trait=[[1, 0, 0], [None]],
                        # causal_matrix_l = np.array([[0., 1., 0.], [0., 0., 0.], [0., 0., 0.]]),
                        # causal_matrix_m = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),
                        # causal_matrix_m = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),
                        # sp_env_file = "./temp_series.csv", # Path to environmental file influencing speciation
                        # sp_env_eff = [1.2, 1.2],  # range environmental effect on speciation rate
                        # env_effect_cat_trait = [[1, 1, 1], [None]],
                        seed = rnd_seed)

# Birth-death simulation
res_bd_div_dep_fb_mu = bd_sim_div_dep_fb_mu.run_simulation(verbose = True)

out_div_dep_fb_mu = plot_combined_diversification_figure(
    res_bd_div_dep_fb_mu,
    trait_idx=0,
    env_which="sp",     # or "ex"
    summary="mean",     # or "median"
    stacked_diversity=False,
    save_path="figs/div_dep_fb_mu.png",
    dpi=300
)




### Feedback diversity dependency in both speciation and extinction

import subprocess
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r'/home/torsten/Work/Software/BDNNsim')
from bdnn_simulator import *
from analysis.plots import *

rnd_seed = int(np.random.choice(np.arange(1, 1e8), 1)[0])
# rnd_seed = 42006789


bd_sim_div_dep_fb_lambda_mu = bdnn_simulator(s_species = 2,  # number of starting species
                        rangeSP = [100, 1000],  # min/max size data set
                        minExtant_SP = 3, # minimum number of extant lineages
                        root_r = [20., 20.],  # range root ages
                        rangeL = [0.2, 0.2],  # range of birth rates
                        rangeM = [0.05, 0.05],  # range of death rates
                        scale = 20.,
                        # fixed_Mtt = np.array([[35., 0.01], [0.0, 0.1]]),
                        n_cont_traits = [0, 0],  # number of continuous traits
                        cont_traits_sigma_clado = [0, 0],
                        cont_traits_sigma = [0, 0], # evolutionary rates for continuous traits
                        n_cat_traits=[1, 1],
                        n_cat_traits_states=[2, 2],  # range number of states for categorical trait
                        cat_traits_diag=None,
                        cat_traits_evolve=False,
                        # cat_traits_Q = np.array([[0, 0], [0, 0]]),
                        cat_traits_effect=np.array([[1., 1.], [1., 1.]]),
                        cat_traits_effect_decr_incr=np.array([[False, False], [False, False]]),
                        cat_traits_min_freq=[0.1],
                        env_sim=True,
                        env_sim_model="BM",
                        env_sim_trend_slope=0.001,
                        env_sim_mean=0,
                        env_sim_sd=0.01,
                        env_sim_shift=[200, 350],
                        env_sim_shift_mag=20,
                        # sp_env_file = "./temp_series.csv", # Path to environmental file influencing speciation
                        sp_env_eff=[1., 1.],  # range environmental effect on speciation rate
                        ex_env_eff=[1., 1.],  # range environmental effect on extinction rate
                        env_effect_by_state_sp=[0.0, 0.0],
                        env_effect_by_state_ex=[0.0, 0.0],
                        divdep_by_state=True,
                        divdep_target_trait_idx=0,
                        divdep_sp_mode="logistic",
                        divdep_ex_mode="logistic",
                        divdep_sp_eff=[1.0, 1.0],
                        divdep_ex_eff=[1.0, 1.0],
                        divdep_effect_by_state_sp=[1.2, 1.2],
                        divdep_effect_by_state_ex=[1.2, 1.2],
                        divdep_state_matrix_sp=np.array([
                            [0.0, 0.25], # [self dep on 0, state 0 depends on diversity of state 1], positive values mean inverse relationship, needs fixing
                            [0.25, 0.0] # [state 1 depends on diversity of state 0, self dep on 1], positive values mean inverse relationship, needs fixing
                        ]),
                        divdep_state_matrix_ex=np.array([
                            [0.0, -0.25], # [self dep on 0, state 0 depends on diversity of state 1], positive values mean inverse relationship, needs fixing
                            [-0.25, 0.0] # [state 1 depends on diversity of state 0, self dep on 1], positive values mean inverse relationship, needs fixing
                        ]),
                        # env_effect_cat_trait=[[1, 0, 0], [None]],
                        # causal_matrix_l = np.array([[0., 1., 0.], [0., 0., 0.], [0., 0., 0.]]),
                        # causal_matrix_m = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),
                        # causal_matrix_m = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),
                        # sp_env_file = "./temp_series.csv", # Path to environmental file influencing speciation
                        # sp_env_eff = [1.2, 1.2],  # range environmental effect on speciation rate
                        # env_effect_cat_trait = [[1, 1, 1], [None]],
                        seed = rnd_seed)

# Birth-death simulation
res_bd_div_dep_fb_lambda_mu = bd_sim_div_dep_fb_lambda_mu.run_simulation(verbose = True)

out_div_dep_fb_lambda_mu = plot_combined_diversification_figure(
    res_bd_div_dep_fb_lambda_mu,
    trait_idx=0,
    env_which="sp",     # or "ex"
    summary="mean",     # or "median"
    stacked_diversity=False,
    save_path="figs/div_dep_fb_lambda_mu.png",
    dpi=300
)



### Full dependency in both speciation and extinction

import subprocess
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r'/home/torsten/Work/Software/BDNNsim')
from bdnn_simulator import *
from analysis.plots import *

rnd_seed = int(np.random.choice(np.arange(1, 1e8), 1)[0])
# rnd_seed = 42006789


bd_sim_full_dep_lambda_mu = bdnn_simulator(s_species = 2,  # number of starting species
                        rangeSP = [100, 1000],  # min/max size data set
                        minExtant_SP = 3, # minimum number of extant lineages
                        root_r = [20., 20.],  # range root ages
                        rangeL = [0.2, 0.2],  # range of birth rates
                        rangeM = [0.05, 0.05],  # range of death rates
                        scale = 20.,
                        # fixed_Mtt = np.array([[35., 0.01], [0.0, 0.1]]),
                        n_cont_traits = [0, 0],  # number of continuous traits
                        cont_traits_sigma_clado = [0, 0],
                        cont_traits_sigma = [0, 0], # evolutionary rates for continuous traits
                        n_cat_traits=[1, 1],
                        n_cat_traits_states=[2, 2],  # range number of states for categorical trait
                        cat_traits_diag=None,
                        cat_traits_evolve=False,
                        # cat_traits_Q = np.array([[0, 0], [0, 0]]),
                        cat_traits_effect=np.array([[1., 1.], [1., 1.]]),
                        cat_traits_effect_decr_incr=np.array([[False, False], [False, False]]),
                        cat_traits_min_freq=[0.1],
                        env_sim=True,
                        env_sim_model="BM",
                        env_sim_trend_slope=0.001,
                        env_sim_mean=0,
                        env_sim_sd=0.01,
                        env_sim_shift=[200, 350],
                        env_sim_shift_mag=20,
                        # sp_env_file = "./temp_series.csv", # Path to environmental file influencing speciation
                        sp_env_eff=[1., 1.],  # range environmental effect on speciation rate
                        ex_env_eff=[-1., -1.],  # range environmental effect on extinction rate
                        env_effect_by_state_sp=[0.5, 0.5],
                        env_effect_by_state_ex=[0.5, 0.5],
                        divdep_by_state=True,
                        divdep_target_trait_idx=0,
                        divdep_sp_mode="logistic",
                        divdep_ex_mode="linear",
                        divdep_sp_eff=[1.0, 1.0],
                        divdep_ex_eff=[1.0, 1.0],
                        divdep_effect_by_state_sp=[1.2, 1.2],
                        divdep_effect_by_state_ex=[1.2, 1.2],
                        divdep_state_matrix_sp=np.array([
                            [0.0, 0.25], # [self dep on 0, state 0 depends on diversity of state 1], positive values mean inverse relationship, needs fixing
                            [0.25, 0.0] # [state 1 depends on diversity of state 0, self dep on 1], positive values mean inverse relationship, needs fixing
                        ]),
                        divdep_state_matrix_ex=np.array([
                            [0.0, -0.25], # [self dep on 0, state 0 depends on diversity of state 1], positive values mean inverse relationship, needs fixing
                            [-0.25, 0.0] # [state 1 depends on diversity of state 0, self dep on 1], positive values mean inverse relationship, needs fixing
                        ]),
                        # env_effect_cat_trait=[[1, 0, 0], [None]],
                        # causal_matrix_l = np.array([[0., 1., 0.], [0., 0., 0.], [0., 0., 0.]]),
                        # causal_matrix_m = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),
                        # causal_matrix_m = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),
                        # sp_env_file = "./temp_series.csv", # Path to environmental file influencing speciation
                        # sp_env_eff = [1.2, 1.2],  # range environmental effect on speciation rate
                        # env_effect_cat_trait = [[1, 1, 1], [None]],
                        seed = rnd_seed)

# Birth-death simulation
res_bd_full_dep_fb_lambda_mu = bd_sim_full_dep_lambda_mu.run_simulation(verbose = True)

out_full_dep_fb_lambda_mu = plot_combined_diversification_figure(
    res_bd_full_dep_fb_lambda_mu,
    trait_idx=0,
    env_which="sp",     # or "ex"
    summary="mean",     # or "median"
    stacked_diversity=False,
    save_path="figs/div_dep_full_lambda_mu.png",
    dpi=300
)

## Testing if response to environment is bell shaped

import subprocess
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r'/home/torsten/Work/Software/BDNNsim')
from bdnn_simulator import *
from analysis.plots import *

rnd_seed = int(np.random.choice(np.arange(1, 1e8), 1)[0])
# rnd_seed = 42006789


bd_sim_env_test = bdnn_simulator(s_species = 2,  # number of starting species
                        rangeSP = [100, 1000],  # min/max size data set
                        minExtant_SP = 3, # minimum number of extant lineages
                        root_r = [20., 20.],  # range root ages
                        rangeL = [0.2, 0.2],  # range of birth rates
                        rangeM = [0.05, 0.05],  # range of death rates
                        scale = 20.,
                        # fixed_Mtt = np.array([[35., 0.01], [0.0, 0.1]]),
                        n_cont_traits = [0, 0],  # number of continuous traits
                        cont_traits_sigma_clado = [0, 0],
                        cont_traits_sigma = [0, 0], # evolutionary rates for continuous traits
                        n_cat_traits=[1, 1],
                        n_cat_traits_states=[2, 2],  # range number of states for categorical trait
                        cat_traits_diag=None,
                        cat_traits_evolve=False,
                        # cat_traits_Q = np.array([[0, 0], [0, 0]]),
                        cat_traits_effect=np.array([[1., 1.], [1., 1.]]),
                        cat_traits_effect_decr_incr=np.array([[False, False], [False, False]]),
                        cat_traits_min_freq=[0.1],
                        env_sim=False,
                        env_sim_model="BM",
                        env_sim_trend_slope=0.001,
                        env_sim_mean=0,
                        env_sim_sd=0.01,
                        env_sim_shift=[200, 350],
                        env_sim_shift_mag=20,
                        sp_env_file="temp_series.tsv",
                        ex_env_file="temp_series.tsv",
                        # sp_env_file = "./temp_series.csv", # Path to environmental file influencing speciation
                        sp_env_eff=[1.2, 1.2],  # range environmental effect on speciation rate
                        ex_env_eff=[1.2, 1.2],  # range environmental effect on extinction rate
                        env_effect_by_state_sp=[1.0, 0.0],
                        env_effect_by_state_ex=[0.0, 0.5],
                        divdep_by_state=False,
                        divdep_target_trait_idx=0,
                        divdep_sp_mode="linear",
                        divdep_ex_mode="logistic",
                        divdep_sp_eff=[1.0, 1.0],
                        divdep_ex_eff=[1.0, 1.0],
                        divdep_effect_by_state_sp=[1.0, 1.0],
                        divdep_effect_by_state_ex=[1.0, 1.0],
                        divdep_state_matrix_sp=np.array([
                            [0.0, 0.0],
                            [0.0, 0.0]
                        ]),
                        divdep_state_matrix_ex=np.array([
                            [0.0, 0.0],
                            [0.0, 0.0]
                        ]),
                        # env_effect_cat_trait=[[1, 0, 0], [None]],
                        # causal_matrix_l = np.array([[0., 1., 0.], [0., 0., 0.], [0., 0., 0.]]),
                        # causal_matrix_m = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),
                        # causal_matrix_m = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),
                        # sp_env_file = "./temp_series.csv", # Path to environmental file influencing speciation
                        # sp_env_eff = [1.2, 1.2],  # range environmental effect on speciation rate
                        # env_effect_cat_trait = [[1, 1, 1], [None]],
                        seed = rnd_seed)

# Birth-death simulation
res_bd_env_test = bd_sim_env_test.run_simulation(verbose = True)

out_env_dep_lamb = plot_combined_diversification_figure(
    res_bd_env_test,
    trait_idx=0,
    env_which="sp",     # or "ex"
    summary="mean",     # or "median"
    stacked_diversity=False,
    save_path="figs/test_envir_bell.png",
    dpi=300
)


## Running 100 simulations with varying environmental effect


from analysis.parallel_simulations import run_prior_simulations_parallel_with_hard_timeouts


if __name__ == "__main__":
    sampled_values, failed_values = run_prior_simulations_parallel_with_hard_timeouts(
        n_sims=100,
        n_processes=10,
        scale_rate=2,
        scale_effect=0.5,
        output_dir="simulation_outputs",
        timeout_per_simulation=60,

        # your usual simulator settings here
        rangeSP = [100, 1000],  # min/max size data set
        minExtant_SP = 3, # minimum number of extant lineages
        root_r = [20., 20.],  # range root ages
        scale = 20.,
        n_cat_traits=[1, 1],
        n_cat_traits_states=[2, 2],
        env_sim=True,
        env_sim_model="BM",
        env_sim_trend_slope=0.001,
        env_sim_mean=0,
        env_sim_sd=0.01,
        env_sim_shift=[200, 350],
        env_sim_shift_mag=20,
        sp_env_eff=[1.2, 1.2],  # range environmental effect on speciation rate
        ex_env_eff=[1.2, 1.2],  # range environmental effect on extinction rate
        divdep_by_state=False,
        divdep_sp_mode="exponential",
        divdep_ex_mode="exponential",
)


