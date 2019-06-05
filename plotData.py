import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import time

from rl.utils import renderpolicy
from cassie import CassieEnv
from cassie.no_delta_env import CassieEnv_nodelta
from cassie.speed_env import CassieEnv_speed
from cassie.speed_double_freq_env import CassieEnv_speed_dfreq
from cassie.speed_no_delta_neutral_foot_env import CassieEnv_speed_no_delta_neutral_foot
from cassie.standing_env import CassieEnv_stand

from rl.policies import GaussianMLP

def avg_pols(policies, state):
    total_act = np.zeros(10)
    for policy in policies:
        _, action = policy.act(state, False)
        total_act += action.data[0].numpy()
    return total_act / len(policies)

# Load environment and policy
# cassie_env = CassieEnv("walking", clock_based=True, state_est=False)
# cassie_env = CassieEnv_nodelta("walking", clock_based=True, state_est=False)
# cassie_env = CassieEnv_speed("walking", clock_based=True, state_est=True)
# cassie_env = CassieEnv_speed_dfreq("walking", clock_based=True, state_est=False)
cassie_env = CassieEnv_speed_no_delta_neutral_foot("walking", clock_based=True, state_est=True)
# cassie_env = CassieEnv_stand(state_est=False)

obs_dim = cassie_env.observation_space.shape[0] # TODO: could make obs and ac space static properties
action_dim = cassie_env.action_space.shape[0]

do_multi = False
no_delta = True
offset = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])

file_prefix = "nodelta_neutral_StateEst_freq1_symmetry_forward_speedmatch_randfreq1-2"
policy = torch.load("./trained_models/nodelta_neutral_StateEst_freq1_symmetry_forward_speedmatch_randfreq1-2.pt")
policy.eval()

policies = []
if do_multi:
    # for i in range(5, 12):
    #     policy = torch.load("./trained_models/regular_spring"+str(i)+".pt")
    #     policy.eval()
    #     policies.append(policy)
    # policy = torch.load("./trained_models/Normal.pt")
    # policy.eval()
    # policies.append(policy)
    # policy = torch.load("./trained_models/stiff_StateEst_step.pt")
    # policy.eval()
    # policies.append(policy)
    for i in [1, 2, 3, 5]:
        policy = torch.load("./trained_models/stiff_StateEst_speed{}.pt".format(i))
        policy.eval()
        policies.append(policy)

num_steps = 150
pre_steps = 0
torques = np.zeros((num_steps*60, 10))
GRFs = np.zeros((num_steps*60, 2))
targets = np.zeros((num_steps*60, 10))
heights = np.zeros(num_steps*60)
speeds = np.zeros(num_steps*60)
foot_pos = np.zeros((num_steps*60, 2))
actions = np.zeros((num_steps*60, 10))
# Execute policy and save torques
with torch.no_grad():
    state = torch.Tensor(cassie_env.reset_for_test())
    cassie_env.speed = .3
    cassie_env.phase_add = 2
    for i in range(pre_steps):
        if not do_multi:
            _, action = policy.act(state, True)
            state, reward, done, _ = cassie_env.step(action.data.numpy())
        else:
            action = avg_pols(policies, state)
            state, reward, done, _ = cassie_env.step(action)
        state = torch.Tensor(state)
    for i in range(num_steps):
        if not do_multi:
            _, action = policy.act(state, True)
            action = action.data.numpy()
        else:
            action = avg_pols(policies, state)
            # state, reward, done, _ = cassie_env.step(action)
        # targets[i, :] = action
        for j in range(60):
            if no_delta:
                target = action + offset
            else:
                ref_pos, ref_vel = cassie_env.get_ref_state(cassie_env.phase + cassie_env.phase_add)
                target = action + ref_pos[cassie_env.pos_idx]
            h = 0.0001
            Tf = 1.0 / 300.0
            alpha = h / (Tf + h)
            # real_action = (1-alpha)*cassie_env.prev_action + alpha*target            
            targets[i*60+j, :] = target
            actions[i*60+j, :] = action
            # print(target)

            cassie_env.step_simulation(action)
            torques[i*60+j, :] = cassie_env.cassie_state.motor.torque[:]
            GRFs[i*60+j, :] = cassie_env.sim.get_foot_forces()
            heights[i*60+j] = cassie_env.sim.qpos()[2]
            speeds[i*60+j] = cassie_env.sim.qvel()[0]
            curr_foot = np.zeros(6)
            cassie_env.sim.foot_pos(curr_foot)
            foot_pos[i*60+j, :] = curr_foot[[2, 5]]
        
        cassie_env.time  += 1
        cassie_env.phase += cassie_env.phase_add

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
fig, ax = plt.subplots(2, figsize=(10, 5))
t = np.linspace(0, num_steps-1, num_steps*60)
ax[0].set_ylabel("GRFs")

ax[0].plot(t, GRFs[:, 0])
ax[0].set_title("Left Foot")
ax[0].set_xlabel("Timesteps (0.0005 sec)")
ax[1].plot(t, GRFs[:, 1])
ax[1].set_title("Right Foot")
ax[1].set_xlabel("Timesteps (0.0005 sec)")

plt.tight_layout()
plt.savefig("./plots/"+file_prefix+"_GRFs.png")

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

# Graph action data
fig, ax = plt.subplots(2, 5, figsize=(15, 5))
t = np.linspace(0, num_steps-1, num_steps*60)
titles = ["Hip Roll", "Hip Yaw", "Hip Pitch", "Knee", "Foot"]
ax[0][0].set_ylabel("Action")
ax[1][0].set_ylabel("Action")
for i in range(5):
    ax[0][i].plot(t, actions[:, i])
    ax[0][i].set_title("Left " + titles[i])
    ax[1][i].plot(t, actions[:, i+5])
    ax[1][i].set_title("Right " + titles[i])
    ax[1][i].set_xlabel("Timesteps (0.0005 sec)")

plt.tight_layout()
plt.savefig("./plots/"+file_prefix+"_actions.png")

# Graph state data
fig, ax = plt.subplots(2, 2, figsize=(10, 5))
t = np.linspace(0, num_steps-1, num_steps*60)
ax[0][0].set_ylabel("meters")
ax[0][0].plot(t, heights[:])
ax[0][0].set_title("Height")
ax[0][1].set_ylabel("m/s")
ax[0][1].plot(t, speeds[:])
ax[0][1].set_title("Speed")
titles = ["Left", "Right"]
for i in range(2):
    ax[1][i].plot(t, foot_pos[:, i])
    ax[1][i].set_title(titles[i] + " Foot")
    ax[1][i].set_xlabel("Timesteps (0.0005 sec)")

plt.tight_layout()
plt.savefig("./plots/"+file_prefix+"_state.png")