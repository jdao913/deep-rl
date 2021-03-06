"""Python file for automatically running experiments from command line."""
import argparse

#from baselines import bench

from rl.utils import run_experiment
from rl.policies import GaussianMLP, BetaMLP
from rl.algos import PPO, MirrorPPO
from rl.envs.wrappers import SymmetricEnv

from rl.envs.normalize import PreNormalizer, get_normalization_params

# NOTE: importing cassie for some reason breaks openai gym, BUG ?
from cassie import CassieEnv, CassieTSEnv
from cassie.no_delta_env import CassieEnv_nodelta
from cassie.speed_env import CassieEnv_speed
from cassie.speed_double_freq_env import CassieEnv_speed_dfreq
from cassie.speed_no_delta_env import CassieEnv_speed_no_delta

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

args.batch_size = 256
args.lr = 1e-4
args.epochs = 3
# args.epochs = 5
args.num_steps = 30
args.seed = int(time.time())
args.max_grad_norm = 0.05
args.use_gae = False

args.name = "dfreq_max_speed_orient_symmetric"
args.num_procs = 2
print("number of procs:", args.num_procs)

if __name__ == "__main__":
    torch.set_num_threads(1) # see: https://github.com/pytorch/pytorch/issues/13757 

    #env_fn = make_env("Walker2d-v1", args.seed, 1337, "/tmp/gym/rl/")

    # env_fn = make_cassie_env("walking", clock_based=True)
    # env_fn = functools.partial(CassieEnv_speed, "walking", clock_based=True, state_est=False)
    # env_fn = functools.partial(CassieEnv_nodelta, "walking", clock_based=True, state_est=False)
    env_fn = functools.partial(CassieEnv_speed_dfreq, "walking", clock_based = True, state_est = False)
    args.env = "speed_dfreq"

    # env_fn = CassieTSEnv

    #env.seed(args.seed)
    #torch.manual_seed(args.seed)

    obs_dim = env_fn().observation_space.shape[0] # TODO: could make obs and ac space static properties
    action_dim = env_fn().action_space.shape[0]
    env_fn = functools.partial(SymmetricEnv, env_fn, mirrored_obs=[0, 1, 2, 3, 4, 5, -13, -14, 15, 16, 17,
                                        18, 19, -6, -7, 8, 9, 10, 11, 12, 20, 21, 22, 23, 24, 25, -33,
                                        -34, 35, 36, 37, 38, 39, -26, -27, 28, 29, 30, 31, 32, 40, 41, 42],
                                        mirrored_act = [0,1,2,3,4,5,6,7,8,9])

    # policy = GaussianMLP(obs_dim, action_dim, nonlinearity="relu", init_std=np.exp(-2), learn_std=False, bounded = False)
    # policy.obs_mean, policy.obs_std = map(torch.Tensor, get_normalization_params(iters=10000, noise_std=1, policy=policy, env_fn=env_fn))

    # Load previous policy
    policy = torch.load("./trained_models/dfreq_max_speed_orient_new.pt")
    policy.train(0)

    # normalizer = PreNormalizer(iter=10000, noise_std=1, policy=policy, online=False,)

    # algo = PPO(args=vars(args))
    algo = MirrorPPO(args = vars(args))

    #with torch.autograd.detect_anomaly():
    # TODO: make log, monitor and render command line arguments
    # TODO: make algos take in a dictionary or list of quantities to log (e.g. reward, entropy, kl div etc)
    run_experiment(
        algo=algo,
        policy=policy,
        env_fn=env_fn,
        args=args,
        log=True,
        monitor=True,
        render=False # NOTE: CassieVis() hangs when launched in seperate thread. BUG?
                    # Also, waitpid() hangs on patrick's desktop in mp.Process. BUG?
    )
