from rl.utils import renderpolicy
from cassie import CassieEnv
from rl.policies import GaussianMLP, BetaMLP
from cassie.slipik_env import CassieIKEnv
from cassie.no_delta_env import CassieEnv_nodelta
from cassie.speed_env import CassieEnv_speed
from cassie.speed_no_delta_env import CassieEnv_speed_no_delta
from cassie.speed_double_freq_env import CassieEnv_speed_dfreq

import torch

import numpy as np
import os
import time

# cassie_env = CassieEnv("walking", clock_based=True, state_est=False)
# cassie_env = CassieEnv_speed("walking", clock_based=True, state_est=False)
# cassie_env = CassieEnv_speed_no_delta("walking", clock_based=True, state_est=False)
cassie_env = CassieEnv_speed_dfreq("walking", clock_based=True, state_est=False)

# env = cassieRLEnvMirror()
# env.phase = 0
# env.counter = 0
# num_inputs = env.observation_space.shape[0]
# num_outputs = env.action_space.shape[0]
# cassie_env = CassieIKEnv(clock_based=True)
obs_dim = cassie_env.observation_space.shape[0] # TODO: could make obs and ac space static properties
action_dim = cassie_env.action_space.shape[0]

# policy = torch.load("./trained_models/stiff_spring.pt")
# policy2 = GaussianMLP(obs_dim, action_dim, nonlinearity="tanh", init_std=np.exp(-1), learn_std=False)
# policy2 = ""
policy = torch.load("./trained_models/speed_input_double_freq_max_speed_orient.pt")
policy.eval()
renderpolicy(cassie_env, policy, deterministic=False, dt=0.05, speedup = 1)

# policies = []
# policy = torch.load("./trained_models/stiff_StateEst_step.pt")
# policy.eval()
# policies.append(policy)
# for i in range(2, 5):
#     policy = torch.load("./trained_models/stiff_StateEst_step{}.pt".format(i))
#     policy.eval()
#     policies.append(policy)

# rendermultipolicy(cassie_env, policies, deterministic=False, dt=0.05)
