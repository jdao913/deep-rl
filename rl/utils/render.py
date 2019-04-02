import torch
from torch.autograd import Variable
import time

@torch.no_grad()
def renderpolicy(env, policy, deterministic=False, speedup=1, dt=0.05):
    state = torch.Tensor(env.reset_for_test())
    render_state = env.render()
    print("render_state:", render_state)
    while render_state:
        if (not env.vis.ispaused()):
            _, action = policy.act(state, deterministic)
            # print("action: ", action.data.numpy())

            state, reward, done, _ = env.step(action.data[0].numpy())

            if done:
                state = env.reset()

            state = torch.Tensor(state)

        render_state = env.render()
        time.sleep(dt / speedup)

def renderloop(env, policy, deterministic=False, speedup=1):
    while True:
        renderpolicy(env, policy, deterministic, speedup)