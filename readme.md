# Pytorch RL

This is a small, modular library that contains some implementations of continuous reinforcement learning algorithms. Fully compatible with OpenAI gym. This version of the library does not use Ray and uses regular Python OpenMP for multiprocessing

## Usage Guide

### Installing MuJoCo
1. MuJoCo is required for the usage of the library. MuJoCo can be downloaded [here](https://www.roboti.us/index.html). The version of the libcassiemujoco library that this repo uses required mujoco200_linux.
2. Once you have downloaded MuJoCo, place the `mujoco200_linux` folder in a directory called `.mujoco` in your home directory. It is also recommended that you place you MuJoCo license key (`mjkey.txt`) in this directory as well. 
3. Define the path to `mjkey.txt` in your `.bashrc` file as a environment variable called `MUJOCO_KEY_PATH`, e.g. `export MUJOCO_KEY_PATH=~/.mujoco/mjkey.txt`
Note: For further details on the usage of MuJoCo and to make changes to the functionality of the `libcassiemujoco.so` library check out [cassie-mujoco-sim](https://github.com/osudrl/cassie-mujoco-sim/tree/mujoco200)

### Installing Python Packages
This library requires the following python packages:
*Pytorch
*Numpy
*Matplotlib
*Visdom
Install the packages using pip or conda if using a conda environment.


### Live Monitoring With Visdom
Run ```$ python -m visdom.server``` then navigate to ```http://localhost:8097/```` in your browser. This will display graphs of the testing reward, batch reward, mean episode length, mean KL divergenve, and mean entropy. The graphs will update after every iteration and are useful to monitoring training progress.

The visdom process must be kept alive for the graphs to update. Thus it is often convenient to start the visdom server in a [tmux](https://hackernoon.com/a-gentle-introduction-to-tmux-8d784c404340) or [screen](https://help.ubuntu.com/community/Screen) session. This is also extremely useful when running experiments remotely as you then don't have to keep an ssh connection alive through the whole process.

If you're running experiments on a remote server and want to view the visdom output on your local machine, I recommend just using port forwarding, i.e. run the following on your local machine once you've started visdom on your remote server.

`ssh -NfL localhost:"local port you want to forward to":localhost:8097 "user@remote server"`

Requires matplotlib >= 2.0

### Running Experiments
To run an experiment run the `ppo.py` file. By default this should train a no delta walking policy. The parameters of training can be changed by editing the `ppo.py` file. Training hyperparameters can be changed by changing the fields of `args`. For further explanation of each argument field check the PPO class in `rl/algos/ppo.py`. To change the environment for training (i.e. delta, no delta, single freq, double freq) change the env_fn variable. Each environment has the `clock_based` and `state_est` flag. Setting them to True sets the policy to input just a clock instead of the whole expert trajectory and to use the output of the state estimator as the input to the policy (as opposed to the true MuJoCo state) respectively. 

To train a symmetric policy (add a symmetry loss when training) uncomment the `env_fn = functools.partial(SymmetricEnv, ....` line. Note that there are two versions of the line, one for use with state estimation and one for without (they should be labelled). 

You can then choose whether to use a new random policy or to start training from a previously trained policy. Note that if you are starting from a previously trained policy you should set normalizer to `None`.
At the end of the file you choose what algorithm you want to use for training (the `algo` variable). `MirrorPPO` should be used only for symmetric policies, otherwise use `PPO`. 

If you find that you are getting KL divergence spikes that is destroying learning (this can often happen when starting training from a previous policy) you can use ADAM_adapt variants of PPO and MirrorPPO, which places a hard constraint on the KL divergence by doing a line search for an appropriate step size if an update causes the KL divergence is too big. Note that this does slow down training however. 

### Modifying Environments
To specify a reward function and initialization conditions, you need to edit the environment files themselves. The reward function is defined in the `compute_reward` function, and the initialization is defined in the `reset` function. Termination conditions can also be changed in the `step` function. 

To change what MuJoCo model file is loaded (i.e. regular cassie vs. stiff spring cassie), change the model file that is passed to the `cassie_mujoco_init` function on line 24 of `cassiemujoco.py`

### Visualizing Policies
Once a policy is trained, you can visualize using the `renderpol.py` file. Load up the correct environment and policy and run the file. A full description of the interactive functionality of the visualization can be found in [cassie-mujoco-sim](https://github.com/osudrl/cassie-mujoco-sim/tree/mujoco200). The speed input and clock frequency can be set realtime by using the arrow keys. The left and right arrows decrement and increment speed by 0.1 respectively, and the down and up arrows decrement and increment phase_add (stepping frequency) by 0.1 respectively. 

NOTE: Due to a bug, if any other key besides the arrow keys is pressed, functionality will break down. This is a function of how keyboard events are being polled and that in python the arrow keys are defined by three characters, and thus three characters have to be read in. This will hopefully be fixed in the future.

Also note that in order for the keys to work, you need have selected the terminal window that is running `renderpol.py`, not be selecting the MuJoCo visualization window. In order to use MuJoCo visualization key press functionalities (like pause and reset) you need to switch back to the MuJoCo window.

You also have the option to render an ensemble of policies. In this case, load up the policies in the ensemble and put them in a list. Then pass the list to the `rendermultipolicy_speedinput` function (defined in `rl/utils/render.py`). This will execute the averaged output of all the policies in the list

### Plotting Policies
`plotData.py` will make the following plots:
* A state plot which includes the height, speed, and z positions of the left and right feet
* A torque plot of the torque commands sent to the motors
* A GRF plot of the ground reaction forces for each foot
* A PD target plot which plots the position target command for each motor
* An action plot, which plots the direct output of the policy.

`plotData.py` functions pretty much the same way as `renderpol.py` in that you have to load up the desired environment and policy. There are also flags that must be set to True if you are plotting a no delta policy or if you are plotting an ensemble of policies. 


### Notes

Troubleshooting: There is a segfault known error that will occasionally occur when visualizing policies. I have tracked this error to be thrown by the mj_render function in libcassiemujoco.so, but have not figured it out further since no one has access to the MuJoCo source code. A workaround that I have found to usually work is to make a copy of the policy file:
```
import torch

temp = torch.load("policy_file")
torch.save(temp, "policy_file_copy")
```
and then load the copy of the policy in `renderpol.py`