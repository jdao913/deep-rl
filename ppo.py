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

#import gym
import torch

import numpy as np
import os
import time
import functools

#TODO: remove reliance on: Monitor, DummyVecEnv, VecNormalized
# def make_env(env_id, seed, rank, log_dir):
#     def _thunk(log=True):
#         env = gym.make(env_id)
#         env.seed(seed + rank)
#         filename = os.path.join(log_dir,os.path.join(log_dir,str(rank))) \
#                    if log else None

#         #env = bench.Monitor(env, filename, allow_early_resets=True)
#         return env

#     return _thunk

def make_cassie_env(*args, **kwargs):
    def _thunk():
        return CassieEnv(*args, **kwargs)
    return _thunk

parser = argparse.ArgumentParser()

PPO.add_arguments(parser)

parser.add_argument("--seed", type=int, default=1234567,
                    help="RNG seed")
parser.add_argument("--logdir", type=str, default="/tmp/rl/experiments/",
                    help="Where to log diagnostics to")
parser.add_argument("--name", type=str, default="model")

args = parser.parse_args()

args.minibatch_size = 2048
args.lr = 1e-4
args.epochs = 3
# args.epochs = 5
args.num_procs = 6
args.num_steps = 12000 // args.num_procs
args.max_traj_len = 300
args.seed = int(time.time())
args.max_grad_norm = 0.05
args.use_gae = False
args.state_est = True
args.mirror = False
ags.logdir = "./logs/test/"
args.log_type = "tensorboard"

args.name = "fwrd_walk_symmetry"
print("number of procs:", args.num_procs)

if __name__ == "__main__":
    torch.set_num_threads(1) # see: https://github.com/pytorch/pytorch/issues/13757 

    #env_fn = make_env("Walker2d-v1", args.seed, 1337, "/tmp/gym/rl/")

    # env_fn = make_cassie_env("walking", clock_based=True)
    # env_fn = functools.partial(CassieEnv_speed, "walking", clock_based=True, state_est=True)
    # env_fn = functools.partial(CassieEnv_nodelta, "walking", clock_based=True, state_est=False)
    # env_fn = functools.partial(CassieEnv_speed_dfreq, "walking", clock_based=True, state_est=False)
    env_fn = functools.partial(CassieEnv_speed_no_delta_neutral_foot, "walking", clock_based=True, state_est=args.state_est)
    # env_fn = functools.partial(CassieEnv_speed_sidestep, "walking", clock_based=True, state_est=args.state_est)
    # env_fn = functools.partial(CassieEnv_stand, state_est=False)

    obs_dim = env_fn().observation_space.shape[0] # TODO: could make obs and ac space static properties
    action_dim = env_fn().action_space.shape[0]

    if args.mirror:
        if args.state_est:
            # with state estimator
            mirror_obs = [0.1, 1, 2, 3, 4, -10, -11, 12, 13, 14, -5, -6, 7, 8, 9, 15,
                        16, 17, 18, 19, 20, -26, -27, 28, 29, 30, -21, -22, 23, 24,
                        25, 31, 32, 33, 37, 38, 39, 34, 35, 36, 43, 44, 45, 40, 41, 42]
        else:
            # without state estimator
            mirror_obs = [0.1, 1, 2, 3, 4, 5, -13, -14, 15, 16, 17, 18, 19, -6, -7,
                        8, 9, 10, 11, 12, 20, 21, 22, 23, 24, 25, -33, -34, 35, 36,
                        37, 38, 39, -26, -27, 28, 29, 30, 31, 32]
        
        mirror_obs += [i for i in range(obs_dim - env_fn().ext_size, obs_dim)]   
        env_fn = functools.partial(SymmetricEnv, env_fn, mirrored_obs=mirror_obs, mirrored_act = [-5, -6, 7, 8, 9, -0.1, -1, 2, 3, 4])

    # Make a new policy
    policy = GaussianMLP(obs_dim, action_dim, nonlinearity="relu", init_std=np.exp(-2), learn_std=False)
    
    # Load previous policy
    # policy = torch.load("./trained_models/sidestep_StateEst_footxypenaltysmall_forcepenalty_footorient_limittargs_speed-05-1_side03_freq1.pt")
    policy.train(0)

    normalizer = PreNormalizer(iter=10000, noise_std=2, policy=policy, online=False)
    # normalizer = None

    if args.mirror:
        algo = MirrorPPO(args=vars(args))
        # algo = Mirror_PPO_ADAM_adapt(args=vars(args))
    else:
        algo = PPO(args=vars(args))
        # algo = PPO_ADAM_adapt(args=vars(args))
    #with torch.autograd.detect_anomaly():
    # TODO: make log, monitor and render command line arguments
    # TODO: make algos take in a dictionary or list of quantities to log (e.g. reward, entropy, kl div etc)
    run_experiment(
        algo=algo,
        policy=policy,
        env_fn=env_fn,
        normalizer=normalizer,
        args=args,
        log=True,
        log_type=args.log_type,
        monitor=True,
        render=False # NOTE: CassieVis() hangs when launched in seperate thread. BUG?
                    # Also, waitpid() hangs on patrick's desktop in mp.Process. BUG?
    )
