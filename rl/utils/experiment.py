import atexit, os
import os.path as osp
from subprocess import Popen
from functools import partial
import torch.multiprocessing as mp
from .render import renderloop
from .logging import Logger
from rl.envs import Normalize, Vectorize
from torch.utils.tensorboard import SummaryWriter


def run_experiment(algo, policy, env_fn, args, normalizer=None, log=True, log_type="tensorboard", monitor=False, render=False):
    if log:
        if log_type == "Visdom":
            logger = Logger(args, viz=monitor)
        elif log_type == "tensorboard":
            log_path = args.logdir + args.name+"/"
            logger = SummaryWriter(log_path, flush_secs=0.1)
        else:
            print("Error: Logger type unknown")
            exit()
    else:
        logger = None
    print("logger type: ", type(logger))
    # exit()
    # HOTFIX for Patrick's desktop: (MP is buggy on it for some reason)

    if render:
        policy.share_memory()

        train_p = mp.Process(target=algo.train,
                         args=(env_fn, policy, args.n_itr, normalizer),
                         kwargs=dict(logger=logger))
        train_p.start()

        # TODO: add normalize as a commandline argument
        renv_fn = partial(env_fn)

        renv = Normalize(Vectorize([renv_fn]))
        render_p = mp.Process(target=renderloop,
                              args=(renv, policy))
        render_p.start()

        train_p.join()
        render_p.join()
    
    else:
        print("logger: ", logger)
        algo.train(env_fn, policy, args.n_itr, normalizer, logger=logger)
