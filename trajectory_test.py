from cassie.trajectory import CassieTrajectory
import numpy as np
import matplotlib.pyplot as plt

traj = CassieTrajectory("./cassie/trajectory/stepdata.bin")

left_shin = traj.qpos[:, 15]
right_shin = traj.qpos[:, 29]

fig, ax = plt.subplots(2, 1, figsize=(10, 4))
ax[0].plot(left_shin)
ax[1].plot(right_shin)
ax[0].set_title("left shin")
ax[1].set_title("right shin")
plt.tight_layout()
plt.show()

