import numpy as np

def max_speed_reward(self):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())
    reward = np.sign(qvel[0])*qvel[0]**2

    return reward

def speedmatch_reward(self):
    qpos = np.copy(self.sim.qpos())
    qvel = np.copy(self.sim.qvel())

    forward_diff = np.abs(qvel[0] - self.speed)
    orient_diff = np.linalg.norm(qpos[3:7] - np.array([1, 0, 0, 0]))
    y_vel = np.abs(qvel[1])
    if forward_diff < 0.05:
        forward_diff = 0
    if y_vel < 0.03:
        y_vel = 0
    straight_diff = np.abs(qpos[1])
    if straight_diff < 0.05:
        straight_diff = 0
    reward = .5*np.exp(-forward_diff) + .15*np.exp(-orient_diff) + .1*np.exp(-y_vel) + .25 * np.exp(-straight_diff)

    return reward