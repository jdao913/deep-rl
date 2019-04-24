import torch
from torch.autograd import Variable
import time
import numpy as np

@torch.no_grad()
def renderpolicy(env, policy, deterministic=False, speedup=1, dt=0.05):
    state = torch.Tensor(env.reset_for_test())
    # qpos = env.sim.qpos()
    # qpos[7] = .5
    # qpos[21] = -.5
    # qpos[7] = 0
    # qpos[8] = 0
    # qpos[21] = 0
    # qpos[22] = 0
    # env.sim.set_qpos(qpos)
    # obs = env.get_full_state()
    # mirrored_matrix = get_symmetry_matrix([0, 1, 2, 3, 4, 5, -13, 14, 15, 16, 17,
                                        # 18, 19, -6, 7, 8, 9, 10, 11, 12, 20, 21, 22, 23, 24, 25, 33,
                                        # 34, 35, 36, 37, 38, 39, 26, 27, 28, 29, 30, 31, 32, 40, 41, 42])
    # obs_mirror = obs @ mirrored_matrix
    # env.set_state(obs_mirror, iters=3000)
    # env.sim.set_qpos(qpos)

    render_state = env.render()
    print("render_state:", render_state)
    env.speed = 2
    env.phase_add = 2
    while render_state:
        if (not env.vis.ispaused()):
            # _, action = policy.act(state, deterministic)
            # action = action.data[0].numpy()
            
            _, action = policy.evaluate(state)
            action = action.mean.data[0].numpy()

            state, reward, done, _ = env.step(action)
            # print("speed: ", env.sim.qvel()[0])

            # if done:
                # state = env.reset()

            state = torch.Tensor(state)

        render_state = env.render()
        time.sleep(dt / speedup)

def renderloop(env, policy, deterministic=False, speedup=1):
    while True:
        renderpolicy(env, policy, deterministic, speedup)

def get_symmetry_matrix(mirrored):
    numel = len(mirrored)
    print("numel: ", numel)
    mat = np.zeros((numel, numel))

    for (i, j) in zip(np.arange(numel), np.abs(mirrored)):
        mat[i, j] = np.sign(mirrored[i])

    return mat 