from .cassiemujoco import pd_in_t, state_out_t, CassieSim, CassieVis

from .trajectory import CassieTrajectory
from cassie.quaternion_function import *
from .rewards import *

from math import floor

import numpy as np 
import os
import random

import pickle

class CassieIKTrajectory:
    def __init__(self, filepath):
        with open(filepath, "rb") as f:
            trajectory = pickle.load(f)

        self.qpos = np.copy(trajectory["qpos"])
        self.qvel = np.copy(trajectory["qvel"])
        #self.foot =
    
    def __len__(self):
        return len(self.qpos)

class CassieEnv_speed_no_delta_neutral_foot:
    def __init__(self, traj, simrate=60, clock_based=False, state_est=False):
        self.sim = CassieSim("./cassie/cassiemujoco/cassie.xml")
        self.vis = None

        self.clock_based = clock_based
        self.state_est = state_est

        if clock_based:
            self.observation_space = np.zeros(42 + 1)
            if self.state_est:
                self.observation_space = np.zeros(48 + 1)       # Size for use with state est
            self.ext_size = 3       # Size of ext_state in get_full_state()
        else:
            self.observation_space = np.zeros(80)
            if self.state_est:
                self.observation_space = np.zeros(86)       # Size for use with state est
            self.ext_size = 1       # Size of ext_state in get_full_state()
        self.action_space      = np.zeros(10)

        dirname = os.path.dirname(__file__)
        if traj == "walking":
            traj_path = os.path.join(dirname, "trajectory", "stepdata.bin")

        elif traj == "stepping":
            # traj_path = os.path.join(dirname, "trajectory", "spline_stepping_traj.pkl")
            traj_path = os.path.join(dirname, "trajectory", "more-poses-trial.bin")

        # self.trajectory = CassieIKTrajectory(traj_path)
        self.trajectory = CassieTrajectory(traj_path)

        self.P = np.array([100,  100,  88,  96,  50]) 
        self.D = np.array([10.0, 10.0, 8.0, 9.6, 5.0])

        self.u = pd_in_t()

        # TODO: should probably initialize this to current state
        self.cassie_state = state_out_t()

        self.simrate = simrate # simulate X mujoco steps with same pd target
                               # 60 brings simulation from 2000Hz to roughly 30Hz

        self.time    = 0 # number of time steps in current episode
        self.phase   = 0 # portion of the phase the robot is in
        self.counter = 0 # number of phase cycles completed in episode

        # NOTE: a reference trajectory represents ONE phase cycle

        # should be floor(len(traj) / simrate) - 1
        # should be VERY cautious here because wrapping around trajectory
        # badly can cause assymetrical/bad gaits
        self.phaselen = floor(len(self.trajectory) / self.simrate) - 1

        # see include/cassiemujoco.h for meaning of these indices
        self.pos_idx = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
        self.vel_idx = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]

        #### Dynamics Randomization ####
        self.dynamics_rand = False
        # Record default dynamics parameters
        if self.dynamics_rand:
            self.default_damping = self.sim.get_dof_damping()
            self.default_mass = self.sim.get_body_mass()
            self.default_ipos = self.sim.get_body_ipos()
            self.default_fric = self.sim.get_geom_friction()

            weak_factor = 0.5
            strong_factor = 0.5

            pelvis_damp_range = [[self.default_damping[0], self.default_damping[0]], 
                                [self.default_damping[1], self.default_damping[1]], 
                                [self.default_damping[2], self.default_damping[2]], 
                                [self.default_damping[3], self.default_damping[3]], 
                                [self.default_damping[4], self.default_damping[4]], 
                                [self.default_damping[5], self.default_damping[5]]] 

            hip_damp_range = [[self.default_damping[6]*weak_factor, self.default_damping[6]*strong_factor],
                            [self.default_damping[7]*weak_factor, self.default_damping[7]*strong_factor],
                            [self.default_damping[8]*weak_factor, self.default_damping[8]*strong_factor]]  # 6->8 and 19->21

            achilles_damp_range = [[self.default_damping[9]*weak_factor,  self.default_damping[9]*strong_factor],
                                    [self.default_damping[10]*weak_factor, self.default_damping[10]*strong_factor], 
                                    [self.default_damping[11]*weak_factor, self.default_damping[11]*strong_factor]] # 9->11 and 22->24

            knee_damp_range     = [[self.default_damping[12]*weak_factor, self.default_damping[12]*strong_factor]]   # 12 and 25
            shin_damp_range     = [[self.default_damping[13]*weak_factor, self.default_damping[13]*strong_factor]]   # 13 and 26
            tarsus_damp_range   = [[self.default_damping[14], self.default_damping[14]]]             # 14 and 27
            heel_damp_range     = [[self.default_damping[15], self.default_damping[15]]]                           # 15 and 28
            fcrank_damp_range   = [[self.default_damping[16]*weak_factor, self.default_damping[16]*strong_factor]]   # 16 and 29
            prod_damp_range     = [[self.default_damping[17], self.default_damping[17]]]                           # 17 and 30
            foot_damp_range     = [[self.default_damping[18]*weak_factor, self.default_damping[18]*strong_factor]]   # 18 and 31

            side_damp = hip_damp_range + achilles_damp_range + knee_damp_range + shin_damp_range + tarsus_damp_range + heel_damp_range + fcrank_damp_range + prod_damp_range + foot_damp_range
            self.damp_range = pelvis_damp_range + side_damp + side_damp

            hi = 1.1
            lo = 0.9
            m = self.default_mass
            pelvis_mass_range      = [[lo*m[1],  hi*m[1]]]  # 1
            hip_mass_range         = [[lo*m[2],  hi*m[2]],  # 2->4 and 14->16
                                    [lo*m[3],  hi*m[3]], 
                                    [lo*m[4],  hi*m[4]]] 

            achilles_mass_range    = [[lo*m[5],  hi*m[5]]]  # 5 and 17
            knee_mass_range        = [[lo*m[6],  hi*m[6]]]  # 6 and 18
            knee_spring_mass_range = [[lo*m[7],  hi*m[7]]]  # 7 and 19
            shin_mass_range        = [[lo*m[8],  hi*m[8]]]  # 8 and 20
            tarsus_mass_range      = [[lo*m[9],  hi*m[9]]]  # 9 and 21
            heel_spring_mass_range = [[lo*m[10], hi*m[10]]] # 10 and 22
            fcrank_mass_range      = [[lo*m[11], hi*m[11]]] # 11 and 23
            prod_mass_range        = [[lo*m[12], hi*m[12]]] # 12 and 24
            foot_mass_range        = [[lo*m[13], hi*m[13]]] # 13 and 25

            side_mass = hip_mass_range + achilles_mass_range \
                        + knee_mass_range + knee_spring_mass_range \
                        + shin_mass_range + tarsus_mass_range \
                        + heel_spring_mass_range + fcrank_mass_range \
                        + prod_mass_range + foot_mass_range

            self.mass_range = [[0, 0]] + pelvis_mass_range + side_mass + side_mass

        self.speed = 1
        # maybe make ref traj only send relevant idxs?
        ref_pos, ref_vel = self.get_ref_state(self.phase)
        self.phase_add = 1
        if self.state_est:
            self.clock_inds = [46, 47]
        else:
            self.clock_inds = [40, 41] 

    def step_simulation(self, action):

        real_action = action
        offset = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])
        real_action = real_action + offset
        # real_action[4] += -1.5968
        # real_action[9] += -1.5968
        
        # target = action + ref_pos[self.pos_idx]
        
        self.u = pd_in_t()
        for i in range(5):
            # TODO: move setting gains out of the loop?
            # maybe write a wrapper for pd_in_t ?
            self.u.leftLeg.motorPd.pGain[i]  = self.P[i]
            self.u.rightLeg.motorPd.pGain[i] = self.P[i]

            self.u.leftLeg.motorPd.dGain[i]  = self.D[i]
            self.u.rightLeg.motorPd.dGain[i] = self.D[i]

            self.u.leftLeg.motorPd.torque[i]  = 0 # Feedforward torque
            self.u.rightLeg.motorPd.torque[i] = 0 

            self.u.leftLeg.motorPd.pTarget[i]  = real_action[i]
            self.u.rightLeg.motorPd.pTarget[i] = real_action[i + 5]

            self.u.leftLeg.motorPd.dTarget[i]  = 0
            self.u.rightLeg.motorPd.dTarget[i] = 0

        self.cassie_state = self.sim.step_pd(self.u)

    def step(self, action):
        for _ in range(self.simrate):
            self.step_simulation(action)

        height = self.sim.qpos()[2]

        self.time  += 1
        self.phase += self.phase_add

        if self.phase > self.phaselen:
            self.phase = 0
            self.counter += 1

        # Early termination
        done = not(height > 0.4 and height < 3.0)

        reward = self.compute_reward()

        # TODO: make 0.3 a variable/more transparent
        if reward < 0.3:
            done = True

        return self.get_full_state(), reward, done, {}

    def reset(self):
        self.phase = random.randint(0, self.phaselen)
        self.time = 0
        self.counter = 0

        orientation = random.randint(-10, 10) * np.pi / 25
        quaternion = euler2quat(z=orientation, y=0, x=0)
        qpos, qvel = self.get_ref_state(self.phase)
        qpos[3:7] = quaternion
        # qpos[2] -= .1

        self.sim.set_qpos(qpos)
        self.sim.set_qvel(qvel)

        # Need to reset u? Or better way to reset cassie_state than taking step
        self.cassie_state = self.sim.step_pd(self.u)

        
        self.speed = (random.randint(0, 10)) / 10
        self.phase_add = 1# + random.random()

        if self.dynamics_rand:
            #### Dynamics Randomization ####
            damp_noise = [np.random.uniform(a, b) for a, b in self.damp_range]
            mass_noise = [np.random.uniform(a, b) for a, b in self.mass_range]
            # com_noise = [0, 0, 0] + [np.random.uniform(self.delta_x_min, self.delta_x_min)] + [np.random.uniform(self.delta_y_min, self.delta_y_max)] + [0] + list(self.default_ipos[6:])
            fric_noise = [np.random.uniform(0.6, 1.2), np.random.uniform(1e-4, 1e-2), np.random.uniform(5e-5, 5e-4)]
            self.sim.set_dof_damping(np.clip(damp_noise, 0, None))
            self.sim.set_body_mass(np.clip(mass_noise, 0, None))
            # self.sim.set_body_ipos(com_noise)
            self.sim.set_geom_friction(np.clip(fric_noise, 0, None), "floor")
            self.sim.set_const()

        return self.get_full_state()

    # used for plotting against the reference trajectory
    def reset_for_test(self):
        self.phase = 0
        self.time = 0
        self.counter = 0
        self.speed = 1

        qpos, qvel = self.get_ref_state(self.phase)

        self.sim.set_qpos(qpos)
        self.sim.set_qvel(qvel)

        # Need to reset u? Or better way to reset cassie_state than taking step
        self.cassie_state = self.sim.step_pd(self.u)

        if self.dynamics_rand:
            self.sim.set_dof_damping(self.default_damping)
            self.sim.set_body_mass(self.default_mass)
            # self.sim.set_body_ipos(self.default_ipos)
            self.sim.set_geom_friction(self.default_fric)
            self.sim.set_const()

        return self.get_full_state()
    
    def set_joint_pos(self, jpos, fbpos=None, iters=5000):
        """
        Kind of hackish. 
        This takes a floating base position and some joint positions
        and abuses the MuJoCo solver to get the constrained forward
        kinematics. 

        There might be a better way to do this, e.g. using mj_kinematics
        """

        # actuated joint indices
        joint_idx = [7, 8, 9, 14, 20,
                     21, 22, 23, 28, 34]

        # floating base indices
        fb_idx = [0, 1, 2, 3, 4, 5, 6]

        for _ in range(iters):
            qpos = np.copy(self.sim.qpos())
            qvel = np.copy(self.sim.qvel())

            qpos[joint_idx] = jpos

            if fbpos is not None:
                qpos[fb_idx] = fbpos

            self.sim.set_qpos(qpos)
            self.sim.set_qvel(0 * qvel)

            self.sim.step_pd(pd_in_t())


    # NOTE: this reward is slightly different from the one in Xie et al
    # see notes for details
    def compute_reward(self):

        # reward = trajmatch_reward(self)
        reward = speedmatch_reward(self)

        return reward

    # get the corresponding state from the reference trajectory for the current phase
    def get_ref_state(self, phase=None):
        if phase is None:
            phase = self.phase

        if phase > self.phaselen:
            phase = 0

        pos = np.copy(self.trajectory.qpos[phase * self.simrate])

        # this is just setting the x to where it "should" be given the number
        # of cycles
        # pos[0] += (self.trajectory.qpos[-1, 0] - self.trajectory.qpos[0, 0]) * self.counter
        
        # ^ should only matter for COM error calculation,
        # gets dropped out of state variable for input reasons

        ###### Setting variable speed  #########
        pos[0] *= self.speed
        pos[0] += (self.trajectory.qpos[-1, 0]- self.trajectory.qpos[0, 0])* self.counter * self.speed
        ######                          ########

        # setting lateral distance target to 0?
        # regardless of reference trajectory?
        pos[1] = 0

        vel = np.copy(self.trajectory.qvel[phase * self.simrate])
        vel[0] *= self.speed

        return pos, vel

    def get_full_state(self):
        qpos = np.copy(self.sim.qpos())
        qvel = np.copy(self.sim.qvel()) 

        # TODO: maybe convert to set subtraction for clarity
        # {i for i in range(35)} - 
        # {0, 10, 11, 12, 13, 17, 18, 19, 24, 25, 26, 27, 31, 32, 33}

        # this is everything except pelvis x and qw, achilles rod quaternions, 
        # and heel spring/foot crank/plantar rod angles
        # note: x is forward dist, y is lateral dist, z is height

        # makes sense to always exclude x because it is in global coordinates and
        # irrelevant to phase-based control. Z is inherently invariant to
        # trajectory despite being global coord. Y is only invariant to straight
        # line trajectories.

        # [ 0] Pelvis y
        # [ 1] Pelvis z
        # [ 2] Pelvis orientation qw
        # [ 3] Pelvis orientation qx
        # [ 4] Pelvis orientation qy
        # [ 5] Pelvis orientation qz
        # [ 6] Left hip roll         (Motor [0])
        # [ 7] Left hip yaw          (Motor [1])
        # [ 8] Left hip pitch        (Motor [2])
        # [ 9] Left knee             (Motor [3])
        # [10] Left shin                        (Joint [0])
        # [11] Left tarsus                      (Joint [1])
        # [12] Left foot             (Motor [4], Joint [2])
        # [13] Right hip roll        (Motor [5])
        # [14] Right hip yaw         (Motor [6])
        # [15] Right hip pitch       (Motor [7])
        # [16] Right knee            (Motor [8])
        # [17] Right shin                       (Joint [3])
        # [18] Right tarsus                     (Joint [4])
        # [19] Right foot            (Motor [9], Joint [5])
        pos_index = np.array([1,2,3,4,5,6,7,8,9,14,15,16,20,21,22,23,28,29,30,34])

        # [ 0] Pelvis x
        # [ 1] Pelvis y
        # [ 2] Pelvis z
        # [ 3] Pelvis orientation wx
        # [ 4] Pelvis orientation wy
        # [ 5] Pelvis orientation wz
        # [ 6] Left hip roll         (Motor [0])
        # [ 7] Left hip yaw          (Motor [1])
        # [ 8] Left hip pitch        (Motor [2])
        # [ 9] Left knee             (Motor [3])
        # [10] Left shin                        (Joint [0])
        # [11] Left tarsus                      (Joint [1])
        # [12] Left foot             (Motor [4], Joint [2])
        # [13] Right hip roll        (Motor [5])
        # [14] Right hip yaw         (Motor [6])
        # [15] Right hip pitch       (Motor [7])
        # [16] Right knee            (Motor [8])
        # [17] Right shin                       (Joint [3])
        # [18] Right tarsus                     (Joint [4])
        # [19] Right foot            (Motor [9], Joint [5])
        vel_index = np.array([0,1,2,3,4,5,6,7,8,12,13,14,18,19,20,21,25,26,27,31])

        clock = [np.sin(2 * np.pi *  self.phase / self.phaselen),
                    np.cos(2 * np.pi *  self.phase / self.phaselen)]
        
        ext_state = np.concatenate((clock, [self.speed]))

        # Use state estimator
        robot_state = np.concatenate([
            [self.cassie_state.pelvis.position[2] - self.cassie_state.terrain.height], # pelvis height
            self.cassie_state.pelvis.orientation[:],                                 # pelvis orientation
            self.cassie_state.motor.position[:],                                     # actuated joint positions

            self.cassie_state.pelvis.translationalVelocity[:],                       # pelvis translational velocity
            self.cassie_state.pelvis.rotationalVelocity[:],                          # pelvis rotational velocity 
            self.cassie_state.motor.velocity[:],                                     # actuated joint velocities

            self.cassie_state.pelvis.translationalAcceleration[:],                   # pelvis translational acceleration
            
            self.cassie_state.joint.position[:],                                     # unactuated joint positions
            self.cassie_state.joint.velocity[:]                                      # unactuated joint velocities
        ])
        if self.state_est:
            return np.concatenate([robot_state,  
                               ext_state])
        else:
            return np.concatenate([qpos[pos_index], 
                               qvel[vel_index], 
                               ext_state])

    def render(self):
        if self.vis is None:
            self.vis = CassieVis(self.sim, "./cassie/cassiemujoco/cassie.xml")

        return self.vis.draw(self.sim)