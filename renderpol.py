from rl.utils import renderpolicy, rendermultipolicy, renderpolicy_speedinput, rendermultipolicy_speedinput
from cassie import CassieEnv
from rl.policies import GaussianMLP, BetaMLP
from cassie.slipik_env import CassieIKEnv
from cassie.no_delta_env import CassieEnv_nodelta
from cassie.speed_env import CassieEnv_speed
from cassie.speed_double_freq_env import CassieEnv_speed_dfreq
from cassie.speed_no_delta_env import CassieEnv_speed_no_delta
from cassie.speed_no_delta_neutral_foot_env import CassieEnv_speed_no_delta_neutral_foot
from cassie.standing_env import CassieEnv_stand
from cassie.speed_sidestep_env import CassieEnv_speed_sidestep

import torch

import numpy as np
import os
import time

# cassie_env = CassieEnv("walking", clock_based=True, state_est=True)
# cassie_env = CassieEnv_nodelta("walking", clock_based=True, state_est=False)
# cassie_env = CassieEnv_speed("walking", clock_based=True, state_est=True)
# cassie_env = CassieEnv_speed_dfreq("walking", clock_based=True, state_est=False)
# cassie_env = CassieEnv_speed_no_delta("walking", clock_based=True, state_est=False)
cassie_env = CassieEnv_speed_no_delta_neutral_foot("walking", clock_based=True, state_est=True)
# cassie_env = CassieEnv_speed_sidestep("walking", clock_based=True, state_est=True)
# cassie_env = CassieEnv_stand(state_est=False)

# policy = torch.load("./trained_models/stiff_spring.pt")
# policies = []
# policy = torch.load("./trained_models/stiff_StateEst_step.pt")
# policy.eval()
# policies.append(policy)
# for i in [1, 2, 3, 5]:
#     policy = torch.load("./trained_models/stiff_StateEst_speed{}.pt".format(i))
#     policy.eval()
#     policies.append(policy)
# rendermultipolicy(cassie_env, policies, deterministic=True, dt=0.05, speedup = 1)
# torch.nn.Module.dump_patches = True
# print("phaselen: ", cassie_env.phaselen)
# torch.nn.Module.dump_patches = True
# policy = torch.load("./trained_models/stiff_spring/stiff_StateEst_speed2.pt")
# policy = torch.load("./trained_models/nodelta_neutral_StateEst_speedreward.pt")
# torch.save(policy, "./trained_models/sidestep_StateEst_footxypenaltysmall_forcepenalty_hipyaw_limittargs_pelpenalty_h0001_speed-05-1_side03_freq1.pt")
policy = torch.load("./trained_models/nodelta_neutral_StateEst_symmetry_speed0-3_freq1-2.pt")
policy.eval()
renderpolicy_speedinput(cassie_env, policy, deterministic=True, dt=0.05, speedup = 2)

# model = ActorCriticNet(num_inputs, num_outputs, [256, 256])
# model.load_state_dict(torch.load("torch_model/SupervisedMultiDirectionMar6.pt"))
# max_speed = 0.5
# min_speed = 0
# max_y_speed = 0.5
# min_y_speed = -0.5
# symmetry = False


# policies = []
# policy = torch.load("./trained_models/stiff_StateEst_step.pt")
# policy.eval()
# policies.append(policy)
# for i in range(2, 5):
#     policy = torch.load("./trained_models/stiff_StateEst_step{}.pt".format(i))
#     policy.eval()
#     policies.append(policy)

# rendermultipolicy(cassie_env, policies, deterministic=False, dt=0.05)
