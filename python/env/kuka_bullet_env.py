
import time
import numpy as np
import pybullet as p
from bullet_utils.env import BulletEnvWithGround
from robot_properties_kuka.iiwaWrapper import IiwaRobot, IiwaConfig

import pinocchio as pin

from matplotlib import pyplot as plt


class KukaBulletEnv:

    def __init__(self):

        # Create a Pybullet simulation environment
        self.env = BulletEnvWithGround(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
        p.resetDebugVisualizerCamera(1.5, 70, -20, (0.0, 0.0, 0.2)) 
        # Create a robot instance. This initializes the simulator as well.
        self.robot = IiwaRobot()
        self.pinModel = self.robot.pin_robot.model
        self.pinData = self.robot.pin_robot.data
        self.env.add_robot(self.robot)

        self.nq, self.nv = self.pinModel.nq, self.pinModel.nv

        # Reset the robot to some initial state.
        q0 = np.matrix(IiwaConfig.initial_configuration).T
        dq0 = np.matrix(IiwaConfig.initial_velocity).T
        self.robot.reset_state(q0, dq0)

        # for plotting
        self.q = []
        self.v = []
        self.tau = []
        self.plan = []

    def reset_robot(self, q0, dq0):
        self.robot.reset_state(q0, dq0)

    def get_state(self):
        q, dq = self.robot.get_state()
        return q, dq

    def set_gains(self, kp, kd):
        self.kp = kp
        self.kd = kd

    def send_id_command(self, q_des, v_des, a_des):
        """
        This function computes the ID torques and sends it to the robot
        Input:
            q_des : desired joint position
            v_des : desired joint velocity
            a_des : desired joint acceleration
        """

        q, v = self.get_state()
        self.q.append(q)
        self.v.append(v)

        tau = np.reshape(pin.rnea(self.pinModel, self.pinData, q, v, a_des), (self.nv,))
        tau_gain = -self.kp*(np.subtract(q.T, q_des.T)) - self.kd*(np.subtract(v.T, v_des.T))
        tau_total = np.reshape((tau_gain + tau), (7,)).T

        self.tau.append(tau_total)
        self.robot.send_joint_command(tau_total)

        # Step the simulator.
        self.env.step(sleep=False) # You can sleep here if you want to slow down the replay

    def plot(self):

        self.q = np.array(self.q)
        self.v = np.array(self.v)
        self.tau = np.array(self.tau)
        t = 0.001*np.arange(0, len(self.tau))

        ind = np.where(np.array(self.plan) == 1)
        fig, ax = plt.subplots(self.nq,1, sharex = True)
        for i in range(self.nq):
            ax[i].plot(t, self.q[:,i], label = "joint nb - " + str(i))
            # ax[i].scatter(t[ind], self.q[:,i][ind], color = "red")
            ax[i].legend()
            ax[i].grid()
        fig.suptitle("joint positions")

        fig, ax2 = plt.subplots(self.nq,1, sharex = True)
        for i in range(self.nq):
            ax2[i].plot(t, self.v[:,i], label = "joint nb - " + str(i))
            # ax2[i].scatter(t[ind], self.v[:,i][ind], color = "red")
            ax2[i].legend()
            ax2[i].grid()
        fig.suptitle("joint velocities")


        fig, ax3 = plt.subplots(self.nq,1, sharex = True)
        for i in range(self.nq):
            ax3[i].plot(t, self.tau[:,i], label = "joint nb - " + str(i))
            # ax3[i].scatter(t[ind], self.tau[:,i][ind], color = "red")
            ax3[i].legend()
            ax3[i].grid()
        fig.suptitle("joint torques")
        plt.show()