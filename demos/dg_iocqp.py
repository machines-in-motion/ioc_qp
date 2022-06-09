## This file contains the ioc qp setup wrapped with dynamic graph head
## Author : Avadesh Meduri
## Date : 21/03/2022
import pathlib
import os
python_path = pathlib.Path('.').absolute().parent/'python'
os.sys.path.insert(1, str(python_path))

import numpy as np
import pinocchio as pin
import torch
from torch.autograd import Function
from torch.nn import functional as F
from vocam.vocam_forward_pass import IOCForwardPass, rt_IOCForwardPass
from multiprocessing import Process, Pipe
import scipy.signal as signal
from scipy.signal import butter, lfilter

import time
import pybullet as p

x_des_arr = np.array([[0.4, -0.4, 0.7], [0.6, 0.4, 0.5]])


def butter_lowpass(highcut, order=2):
    return butter(order, highcut, btype='lowpass')

class DiffQPController:

    def __init__(self, head, robot_model, robot_data, nn_dir, mean, std, target = None, vicon_name = None, run_sim = False):
        """
        Input:
            head : thread head
            robot_model : pinocchio model
            robot_data : pinocchio data
            nn : trained neural network
            ioc : ioc QP
            mean : mean of the trained data (y_train)
            std : standard deviation of trained data (y_train)
        """

        self.run_sim = run_sim
        self.head = head
        self.pinModel = robot_model
        self.pinData = robot_data
        self.target = target
        self.nq = self.pinModel.nq

        self.nn_dir = nn_dir
        self.m = mean
        self.std = std
        self.n_col = 5
        # for sub processes
        # self.planner = IOCForwardPass(nn_dir, mean, std)

        self.parent_conn, self.child_conn = Pipe()
        self.sent = False

        self.count = 0 
        self.dt = 0.05
        self.inter = int(self.dt/0.001)

        self.vicon_name = vicon_name

        # for plotting
        self.x_des = np.zeros(3)
        self.ee_pos = np.zeros(3)
        self.f_id = self.pinModel.getFrameId("EE")

        self.joint_positions = head.get_sensor('joint_positions')
        self.joint_velocities = head.get_sensor("joint_velocities")
        if not self.run_sim:
            self.joint_torques = head.get_sensor("joint_torques_total")
            self.joint_ext_torques = head.get_sensor("joint_torques_external")


        # filter params
        self.set_vel_filter(0.02)
        self.filter_vel_z = [[0] for i in range(self.nq)]

    def set_vel_filter(self, percent):
        self.filter_vel_b = []
        self.filter_vel_a = []

        for i in range(self.nq):
            b, a = butter_lowpass(percent, order=1)
            self.filter_vel_b.append(b)
            self.filter_vel_a.append(a)

    def warmup(self, thread):

        # self.x_pred = self.planner.predict(self.joint_positions, self.joint_velocities, self.x_des)

        self.subp = Process(target=rt_IOCForwardPass, args=(self.child_conn, self.nn_dir, self.m, self.std))
        self.subp.start()

        q = self.joint_positions
        v = np.zeros_like(q)
        self.parent_conn.send((q, v, self.x_des))
        self.x_pred = self.parent_conn.recv()
    
        self.check = 0
        if not self.vicon_name or self.run_sim:
            self.prev_cube_pos,  _ = thread.vicon.get_state(self.vicon_name)
            self.prev_cube_pos = self.prev_cube_pos[0:3]

        self.op = 0

    def update_desired_position(self, x_des):
        self.x_des = x_des
        if self.target:
            p.resetBasePositionAndOrientation(self.target, x_des, (0,0,0,1))

    def set_gains(self, kp, kd):
        self.kp = kp
        self.kd = kd

    def get_cube_pos(self, thread):
        cube_pos, _ = thread.vicon.get_state(self.vicon_name)
        cube_pos = cube_pos[0:3]
        if np.linalg.norm(cube_pos) > 0:
            cube_pos[0] += 0.3 
            # cube_pos[1] -= 0.1
            self.prev_cube_pos = cube_pos

        else:
            cube_pos = self.prev_cube_pos

        return cube_pos

    def run(self, thread):

        t1 = time.time()

        q = self.joint_positions
        # self.v_fil = self.joint_velocities.copy()
        # for j in range(self.nq):
        #     self.v_fil[j], self.filter_vel_z[j] = signal.lfilter(
        #         self.filter_vel_b[j], self.filter_vel_a[j],
        #         [self.v_fil[j]], zi=self.filter_vel_z[j])
        # v = self.v_fil
        v = self.joint_velocities.copy()

        if not self.vicon_name or self.run_sim:
            x_des_tmp = np.array([[0.514, 0.3518, 0.1539], [0.5122, -0.3506,  0.1529]])
            if thread.ti % (7*1000) == 0:
                if self.op == 0:
                    self.op = 1
                else:
                    self.op = 0
            x_des = x_des_tmp[self.op]
        else:
            # x_des_tmp = np.array([[0.4094, -0.3638,  0.1958], [0.5066, 0.2694, 0.1690]])
            # if thread.ti % (15*1000) == 0:
            #     if self.op == 0:
            #         self.op = 1
            #     else:
            #         self.op = 0
            # x_des = x_des_tmp[self.op]

            x_des = self.get_cube_pos(thread)
        
        # print(x_des)
        self.update_desired_position(x_des)
        
        if thread.ti % int(self.dt*1000) == 0:
            count = self.count
            q_des = self.x_pred[count*3*self.nq:count*3*self.nq+self.nq]
            dq_des = self.x_pred[count*3*self.nq+self.nq:count*3*self.nq+2*self.nq]
            a_des = self.x_pred[count*3*self.nq + 2*self.nq:count*3*self.nq+3*self.nq]

            if self.count == self.n_col - 2 and thread.ti != 0 and not self.sent:
                # self.x_pred = self.planner.predict(q, v, self.x_des)
                self.parent_conn.send((q, v, self.x_des))
                self.sent = True
                # print("computing...")
            if self.parent_conn.poll():
                self.x_pred = self.parent_conn.recv()
                self.count = -1
                self.sent = False

            # print(count)
            count = self.count
            tmp = count + 1
            nq_des = self.x_pred[tmp*3*self.nq:tmp*3*self.nq+self.nq]
            ndq_des = self.x_pred[tmp*3*self.nq+self.nq:tmp*3*self.nq+2*self.nq]
            na_des = self.x_pred[tmp*3*self.nq + 2*self.nq:tmp*3*self.nq+3*self.nq]

            self.q_int = np.linspace(q_des, nq_des, self.inter)
            self.dq_int = np.linspace(dq_des, ndq_des, self.inter)
            self.a_int = np.linspace(a_des, na_des, self.inter)

            self.count = min(self.n_col - 2, self.count + 1)
            self.index = 0

        self.check += 1
        # controller
        self.q_des = self.q_int[self.index].T
        self.dq_des = self.dq_int[self.index].T
        self.a_des = self.a_int[self.index]
        
        tau = np.reshape(pin.rnea(self.pinModel, self.pinData, q, v, self.a_des), (self.nq,))
        self.gravity = np.reshape(pin.rnea(self.pinModel, self.pinData, q, np.zeros_like(q), np.zeros_like(q)), (self.nq,))
    
        tau_gain = -self.kp*(np.subtract(q.T, self.q_des)) - self.kd*(np.subtract(v.T, self.dq_des))
        self.tau_total = -1*np.reshape((tau_gain + tau), (7,)).T #only for plotting
        
        self.tau_in = np.reshape((tau_gain + tau), (7,)).T
        if not self.run_sim:
            self.tau_in -= self.gravity.copy()
                    
        self.index += 1
        t2 = time.time()

        self.time = t2 - t1
        # for plotting
        pin.forwardKinematics(self.pinModel, self.pinData, q, v, np.zeros_like(q))
        pin.updateFramePlacements(self.pinModel, self.pinData)
        self.ee_pos = self.pinData.oMf[self.f_id].translation.copy()

        pin.forwardKinematics(self.pinModel, self.pinData, self.q_des, self.dq_des, np.zeros_like(q))
        pin.updateFramePlacements(self.pinModel, self.pinData)
        self.ee_pos_des = self.pinData.oMf[self.f_id].translation

        self.head.set_control('ctrl_joint_torques', self.tau_in)     
        if not self.run_sim:   
            self.head.set_control('time_sent', np.array([thread.ti*1e-3]))
            self.head.set_control('desired_joint_positions', self.q_des)
