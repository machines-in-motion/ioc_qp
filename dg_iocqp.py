## This file contains the ioc qp setup wrapped with dynamic graph head
## Author : Avadesh Meduri
## Date : 21/03/2022

import numpy as np
import pinocchio as pin
import torch
from torch.autograd import Function
from torch.nn import functional as F
from inverse_qp import IOC, IOCForwardPass, subprocess_mpc_entry
from multiprocessing import Process, Pipe
import scipy.signal as signal
from scipy.signal import butter, lfilter

import time

def butter_lowpass(highcut, order=2):
    return butter(order, highcut, btype='lowpass')

class DiffQPController:

    def __init__(self, head, robot_model, robot_data, nn_dir, mean, std, vicon_name = None):
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

        self.head = head
        self.pinModel = robot_model
        self.pinData = robot_data
        
        self.nq = self.pinModel.nq

        self.nn_dir = nn_dir
        self.m = mean
        self.std = std
        self.n_col = 5
        # for sub processes
        self.parent_conn, self.child_conn = Pipe()
        self.sent = False

        self.count = 0 
        self.dt = 0.05
        self.inter = int(self.dt/0.001)

        # for plotting
        self.x_des = np.zeros(3)
        self.ee_pos = np.zeros(3)
        self.f_id = self.pinModel.getFrameId("EE")

        self.joint_positions = head.get_sensor('joint_positions')
        self.joint_velocities = head.get_sensor("joint_velocities")

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

        self.subp = Process(target=subprocess_mpc_entry, args=(self.child_conn, self.nn_dir, self.m, self.std))
        self.subp.start()

        q = self.joint_positions
        v = np.zeros_like(q)
        self.parent_conn.send((q, v, self.x_des))
        self.x_pred = self.parent_conn.recv()
    
        self.check = 0

    def update_desired_position(self, x_des):
        self.x_des = x_des
        
    def set_gains(self, kp, kd):
        self.kp = kp
        self.kd = kd

    def run(self, thread):

        t1 = time.time()

        q = self.joint_positions
        self.v_fil = self.joint_velocities.copy()
        for j in range(self.nq):
            self.v_fil[j], self.filter_vel_z[j] = signal.lfilter(
                self.filter_vel_b[j], self.filter_vel_a[j],
                [self.v_fil[j]], zi=self.filter_vel_z[j])
        v = self.v_fil

        if thread.ti % int(self.dt*1000) == 0:
            count = self.count
            q_des = self.x_pred[count*3*self.nq:count*3*self.nq+self.nq]
            dq_des = self.x_pred[count*3*self.nq+self.nq:count*3*self.nq+2*self.nq]
            a_des = self.x_pred[count*3*self.nq + 2*self.nq:count*3*self.nq+3*self.nq]

            if self.count == self.n_col - 2 and thread.ti != 0 and not self.sent:
                self.parent_conn.send((q, v, self.x_des))
                self.sent = True
                self.check = 0
        
        if self.parent_conn.poll():
            self.x_pred = self.parent_conn.recv()
            self.count = -1
            self.sent = False
            print(0.001*self.check)
            self.check = 0

        if thread.ti % int(self.dt*1000) == 0:

            count = self.count
            tmp = min(count + 1, self.n_col - 1)
            nq_des = self.x_pred[tmp*3*self.nq:tmp*3*self.nq+self.nq]
            ndq_des = self.x_pred[tmp*3*self.nq+self.nq:tmp*3*self.nq+2*self.nq]
            na_des = self.x_pred[tmp*3*self.nq + 2*self.nq:tmp*3*self.nq+3*self.nq]

            self.q_int = np.linspace(q_des, nq_des, self.inter)
            self.dq_int = np.linspace(dq_des, ndq_des, self.inter)
            self.a_int = np.linspace(a_des, na_des, self.inter)

            self.count = min(self.n_col - 1, self.count + 1)
            self.index = 0

        self.check += 1
        print(self.count)
        # controller

        q_des = self.q_int[self.index].T
        dq_des = self.dq_int[self.index].T
        a_des = self.a_int[self.index]
        
        tau = np.reshape(pin.rnea(self.pinModel, self.pinData, q, v, a_des), (self.nq,))
        tau_gain = -self.kp*(np.subtract(q.T, q_des)) - self.kd*(np.subtract(v.T, dq_des))
        tau_total = np.reshape((tau_gain + tau), (7,)).T
        self.index += 1
        t2 = time.time()

        self.time = t2 - t1
        # for plotting
        self.ee_pos = self.pinData.oMf[self.f_id].translation

        self.head.set_control('ctrl_joint_torques', tau_total)
