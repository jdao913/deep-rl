import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import time

from rl.utils import renderpolicy
from cassie import CassieEnv
from rl.policies import GaussianMLP

# Load environment and policy
cassie_env = CassieEnv("walking", clock_based=True)
obs_dim = cassie_env.observation_space.shape[0] # TODO: could make obs and ac space static properties
action_dim = cassie_env.action_space.shape[0]

file_prefix = "limit_test"
policy = torch.load("./trained_models/regular_spring7.pt")
policy.eval()

num_steps = 10
torques = np.zeros((num_steps*60, 10))
GRFs = np.zeros((num_steps*60, 2))
targets = np.zeros((num_steps*60, 10))
# Execute policy and save torques
with torch.no_grad():
    state = torch.Tensor(cassie_env.reset_for_test())
    for i in range(num_steps):
        _, action = policy.act(state, False)
        # targets[i, :] = action
        for j in range(60):
            ref_pos, ref_vel = cassie_env.get_ref_state(cassie_env.phase + 1)
            target = action.data[0].numpy() + ref_pos[cassie_env.pos_idx]
            h = 0.0001
            Tf = 1.0 / 300.0
            alpha = h / (Tf + h)
            real_action = (1-alpha)*cassie_env.prev_action + alpha*target            
            targets[i*60+j, :] = real_action
            # print(target)

            cassie_env.step_simulation(action.data[0].numpy())
            torques[i*60+j, :] = cassie_env.cassie_state.motor.torque[:]
            GRFs[i*60+j, :] = cassie_env.sim.get_foot_forces()
        
        cassie_env.time  += 1
        cassie_env.phase += 1

        if cassie_env.phase > cassie_env.phaselen:
            cassie_env.phase = 0
            cassie_env.counter += 1

        state = cassie_env.get_full_state()
        state = torch.Tensor(state)

# Graph torque data
fig, ax = plt.subplots(2, 5, figsize=(15, 5))
t = np.linspace(0, num_steps-1, num_steps*60)
titles = ["Hip Roll", "Hip Yaw", "Hip Pitch", "Knee", "Foot"]
ax[0][0].set_ylabel("Torque")
ax[1][0].set_ylabel("Torque")
for i in range(5):
    ax[0][i].plot(t, torques[:, i])
    ax[0][i].set_title("Left " + titles[i])
    ax[1][i].plot(t, torques[:, i+5])
    ax[1][i].set_title("Right " + titles[i])
    ax[1][i].set_xlabel("Timesteps (0.0005 sec)")

plt.tight_layout()
plt.savefig("./plots/"+file_prefix+"_torques.png")

# Graph GRF data
# fig, ax = plt.subplots(2, figsize=(10, 5))
# t = np.linspace(0, num_steps-1, num_steps*60)
# ax[0].set_ylabel("GRFs")

# ax[0].plot(t, GRFs[:, 0])
# ax[0].set_title("Left Foot")
# ax[0].set_xlabel("Timesteps (0.0005 sec)")
# ax[1].plot(t, GRFs[:, 1])
# ax[1].set_title("Right Foot")
# ax[1].set_xlabel("Timesteps (0.0005 sec)")

# plt.tight_layout()
# plt.savefig("./plots/"+file_prefix+"_GRFs.png")

# Graph PD target data
fig, ax = plt.subplots(2, 5, figsize=(15, 5))
t = np.linspace(0, num_steps-1, num_steps*60)
titles = ["Hip Roll", "Hip Yaw", "Hip Pitch", "Knee", "Foot"]
ax[0][0].set_ylabel("PD Target")
ax[1][0].set_ylabel("PD Target")
for i in range(5):
    ax[0][i].plot(t, targets[:, i])
    ax[0][i].set_title("Left " + titles[i])
    ax[1][i].plot(t, targets[:, i+5])
    ax[1][i].set_title("Right " + titles[i])
    ax[1][i].set_xlabel("Timesteps (0.0005 sec)")

plt.tight_layout()
plt.savefig("./plots/"+file_prefix+"_targets.png")