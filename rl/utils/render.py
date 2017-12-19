import torch
from torch.autograd import Variable
import time

def renderpolicy(env, policy, explore=True, speedup=1, dt=0.05):
    state = torch.Tensor(env.reset())
    while True:
        _, action = policy.act(Variable(state, volatile=True))

        state, reward, done, _ = env.step(action.data.numpy())

        if done:
            state = env.reset()

        state = torch.Tensor(state)

        if hasattr(env, "render"):
            env.render()
        else:
            env.venv.envs[0].render()

        time.sleep(dt / speedup)

def renderloop(env, policy, explore=False, speedup=1):
    while True:
        renderpolicy(env, policy, explore, speedup)