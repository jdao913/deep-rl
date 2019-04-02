from rl.utils import renderpolicy
from cassie import CassieEnv
from rl.policies import GaussianMLP, BetaMLP
from cassie.slipik_env import CassieIKEnv

import torch

import numpy as np
import os
import time

cassie_env = CassieEnv("walking", clock_based=True)
# cassie_env = CassieIKEnv(clock_based=True)
obs_dim = cassie_env.observation_space.shape[0] # TODO: could make obs and ac space static properties
action_dim = cassie_env.action_space.shape[0]

# policy = torch.load("./trained_models/stiff_spring.pt")
# policy2 = GaussianMLP(obs_dim, action_dim, nonlinearity="tanh", init_std=np.exp(-1), learn_std=False)
# policy2 = ""
policy = torch.load("./trained_models/regular_spring7.pt")
policy.eval()
# policy = torch.load("IKEnvParallelGradClipBigBatch.pt")
policy.train(0)
renderpolicy(cassie_env, policy, deterministic=False, dt=0.05)
# policy = GaussianMLP(obs_dim, action_dim, nonlinearity="tanh", init_std=np.exp(-1), learn_std=False)