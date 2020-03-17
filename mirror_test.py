"""Python file for automatically running experiments from command line."""
import argparse

#from baselines import bench

from rl.utils import run_experiment
from rl.policies import GaussianMLP, BetaMLP
from rl.algos import PPO, MirrorPPO, PPO_ADAM_adapt, Mirror_PPO_ADAM_adapt

from rl.envs.normalize import PreNormalizer
from rl.envs.wrappers import SymmetricEnv

# NOTE: importing cassie for some reason breaks openai gym, BUG ?
from cassie import CassieEnv, CassieTSEnv
from cassie.no_delta_env import CassieEnv_nodelta
from cassie.speed_env import CassieEnv_speed
from cassie.speed_double_freq_env import CassieEnv_speed_dfreq
from cassie.speed_no_delta_neutral_foot_env import CassieEnv_speed_no_delta_neutral_foot
from cassie.standing_env import CassieEnv_stand
from cassie.speed_sidestep_env import CassieEnv_speed_sidestep
from cassie.cassiemujoco import pd_in_t, state_out_t, CassieSim, CassieVis

import numpy as np
import os
import time
import functools
import copy
import select
import tty
import termios
import sys
import torch

def get_u(targets):
    u = pd_in_t()
    P = np.array([100,  100,  88,  96,  50]) 
    D = np.array([10.0, 10.0, 8.0, 9.6, 5.0])
    for i in range(5):
        # TODO: move setting gains out of the loop?
        # maybe write a wrapper for pd_in_t ?
        u.leftLeg.motorPd.pGain[i]  = P[i]
        u.rightLeg.motorPd.pGain[i] = P[i]

        u.leftLeg.motorPd.dGain[i]  = D[i]
        u.rightLeg.motorPd.dGain[i] = D[i]

        u.leftLeg.motorPd.torque[i]  = 0 # Feedforward torque
        u.rightLeg.motorPd.torque[i] = 0 

        u.leftLeg.motorPd.pTarget[i]  = targets[i]
        u.rightLeg.motorPd.pTarget[i] = targets[i + 5]

        u.leftLeg.motorPd.dTarget[i]  = 0
        u.rightLeg.motorPd.dTarget[i] = 0
    return u

def isData():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

def set_qpos_from_obs(qpos, obs):
    obs_ind1 = [2, 3, 4, 5, 6, 7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
    obs_ind2 = [15, 16, 20, 29, 30, 34]
    qpos[obs_ind1] = obs[0:15]
    qpos[obs_ind2] = obs[34:40]
    return qpos

env_fn = functools.partial(CassieEnv_speed_no_delta_neutral_foot, "walking", clock_based=True, state_est=True)
env = env_fn()
obs_dim = env_fn().observation_space.shape[0] # TODO: could make obs and ac space static properties
action_dim = env_fn().action_space.shape[0]
mirror_obs = [0.1, 1, 2, 3, 4, -10, -11, 12, 13, 14, -5, -6, 7, 8, 9, 15,
                        16, 17, 18, 19, 20, -26, -27, 28, 29, 30, -21, -22, 23, 24,
                        25, 31, 32, 33, 37, 38, 39, 34, 35, 36, 43, 44, 45, 40, 41, 42]
mirror_obs += [i for i in range(obs_dim - env_fn().ext_size, obs_dim)]
sym_env = SymmetricEnv(env_fn, mirrored_obs=mirror_obs, mirrored_act = [-5, -6, 7, 8, 9, -0.1, -1, 2, 3, 4])

env.sim.set_qpos(env.trajectory.qpos[0, :])
env.render()
pos_idx = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]

# orig_action = env.trajectory.qpos[0, pos_idx]
# mir_action = sym_env.mirror_action(torch.Tensor(orig_action))
# mir_mir_action = sym_env.mirror_action(mir_action)
# print("orig action: ", orig_action)
# print("mir action: ", mir_action.numpy())
# print("mir mir action: ", mir_mir_action.numpy())
# print("action diff: ", orig_action - mir_mir_action.numpy())
# print("action diff norm: ", np.linalg.norm(orig_action - mir_mir_action.numpy()))
# exit()

total_time = env.trajectory.time.shape[0]
curr_time = 0
mirror = False
curr_qpos = copy.copy(env.trajectory.qpos[curr_time, :])
env.sim.set_qpos(curr_qpos)
# print("qpos height: ", curr_qpos[2])
env.step_simulation(np.zeros(10))
# for i in range(10):
#     env.sim.set_qpos(curr_qpos)
#     env.cassie_state = env.sim.step_pd(pd_in_t())
curr_obs = env.get_full_state()
mir_obs = sym_env.mirror_clock_observation(torch.Tensor(curr_obs), env.clock_inds)
mir_mir_obs = sym_env.mirror_clock_observation(mir_obs, env.clock_inds)
print("obs diff: ", curr_obs - mir_mir_obs.numpy())
print("obs diff norm: ", np.linalg.norm(curr_obs - mir_mir_obs.numpy()))
np.set_printoptions(threshold=np.inf)
foo = np.random.rand(4)
foo_mir_matrix = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]])
foo_mir = foo_mir_matrix @ foo
foo_mir_mir = foo_mir_matrix @ foo_mir
print("foo: ", foo)
print("foo mir: ", foo_mir)
print("foo diff: ", foo - foo_mir_mir)
foo_tensor = torch.Tensor(foo)
print("foo tensor diff: ", torch.Tensor(foo) - foo_tensor)

exit()
# print("obs height: ", curr_obs[0])
render_state = env.render()
old_settings = termios.tcgetattr(sys.stdin)
try:
    tty.setcbreak(sys.stdin.fileno())
    while render_state:
        if isData():
            c = sys.stdin.read(1)
            if c == "m":
                mirror = not mirror
                print("doing mirror state: ", mirror)
                if mirror:
                    # print("orig qpos: ", curr_qpos[2])
                    # print("orig obs: ", curr_obs[0])
                    # print("terrain height: ", env.cassie_state.terrain.height)
                    curr_obs = sym_env.mirror_clock_observation(torch.Tensor(curr_obs), env.clock_inds)
                    curr_obs = curr_obs.numpy()
                    # curr_obs[0] += env.cassie_state.terrain.height      # height adjust
                    # print("mirror obs: ", curr_obs[0])
                    curr_qpos = set_qpos_from_obs(curr_qpos, curr_obs)
                    # print("after qpos: ", curr_qpos[2])
                    new_obs = env.get_full_state()
                    # print("obs diff: ", new_obs - curr_obs)
                else:
                    print("resetting qpos")
                    curr_qpos = copy.copy(env.trajectory.qpos[curr_time, :])

                env.sim.set_qpos(curr_qpos)
                print("qpos height: ", curr_qpos[2])
                for i in range(1):
                    env.cassie_state = env.sim.step_pd(get_u(curr_qpos[pos_idx]))
                curr_obs = env.get_full_state()
                print("obs height: ", curr_obs[0])

            elif c == "d":
                print("Stepping forward 10 timesteps")
                curr_time += 10
                curr_qpos = env.trajectory.qpos[curr_time, :]
                env.sim.set_qpos(curr_qpos)
                env.cassie_state = env.sim.step_pd(pd_in_t())
                curr_obs = env.get_full_state()
                print("height diff: ", curr_qpos[2] - curr_obs[0])
                print("obs height: ", curr_obs[0])
            elif c == "a":
                print("Stepping backward 10 timesteps")
                curr_time -= 10
                curr_qpos = env.trajectory.qpos[curr_time, :]
                env.sim.set_qpos(curr_qpos)
                env.cassie_state = env.sim.step_pd(get_u(curr_qpos[pos_idx]))
                curr_obs = env.get_full_state()
            else:
                pass
        
        render_state = env.render()
finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
