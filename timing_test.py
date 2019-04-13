from rl.utils import renderpolicy
from cassie import CassieEnv
from rl.policies import GaussianMLP, BetaMLP

import torch
import numpy as np
import os
import time

cassie_env = CassieEnv("stepping", clock_based=True, state_est=True)
# cassie_env = CassieIKEnv(clock_based=True)
obs_dim = cassie_env.observation_space.shape[0] # TODO: could make obs and ac space static properties
action_dim = cassie_env.action_space.shape[0]

# policy = torch.load("./trained_models/stiff_spring.pt")

policy = torch.load("./trained_models/stiff_StateEst_step3.pt")
policy.eval()

# policies = []
# for i in range(5, 12):
#    policy = torch.load("./trained_models/regular_spring{}.pt".format(i))
#    policy.eval()
#    policies.append(policy)
# policy = torch.load("./trained_models/Normal.pt")
# policy.eval()
# policies.append(policy)

iters = 200
state = torch.Tensor(cassie_env.reset_for_test())
total_time = 0
for i in range(iters):
   start_t = time.process_time()
   _, action = policy.act(state, False)
   
   
   # total_act = np.zeros(10)
   # for policy in policies:
   #    _, action = policy.act(state, False)
   #    total_act += action.data[0].numpy()
   # action = total_act / len(policies)
   total_time += time.process_time() - start_t

   action = action.data[0].numpy()
   state, reward, done, _ = cassie_env.step(action)
   if done:
         state = cassie_env.reset()

   state = torch.Tensor(state)

print("Average total time:", total_time / iters, "sec")
# print("Average policy computation time:", total_time / iters / len(policies))


