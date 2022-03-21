## This file contains the ioc qp setup wrapped with dynamic graph head
## Author : Avadesh Meduri
## Date : 21/03/2022

import numpy as np
import pinocchio as pin
import torch
from torch.autograd import Function
from torch.nn import functional as F
from inverse_qp import IOC, IOCForwardPass

import time

class DiffQPController:

    def __init__(self, head, robot_model, robot_data, nn_dir, mean, std, vicon_name = None):
        """
        Input:
            head : thread head
            robot_model : pinocchio model
            robot_data : pinocchio data
            nn : trained neural network
            ioc : ioc QP
            nn_dir : directory for NN weights
            mean : mean of the trained data (y_train)
            std : standard deviation of trained data (y_train)
        """

        self.head = head
        self.pinModel = robot_model
        self.pinData = robot_data

        self.nq = 7
        self.n_col = 5
        u_max = [2.5,2.5,2.5, 1.5, 1.5, 1.5, 1.0]
        self.dt = 0.05

        self.ioc = IOC(self.n_col, self.nq, u_max, self.dt, eps = 1.0, isvec=True)

        self.n_vars = self.ioc.n_vars
        self.state = torch.zeros(2*self.nq + 3) # q, v, x_des
        self.count = 0 
        self.inter = int(self.dt/0.001)

        self.nn = Net(2*self.nq + 3, 2*self.n_vars)
        self.nn.load_state_dict(torch.load(nn_dir))

        self.mean = mean
        self.std = std


        self.joint_positions = head.get_sensor('joint_positions')
        self.joint_velocities = head.get_sensor("joint_velocities")

    def warmup(self, thread):
        self.x_pred = self.compute_plan()
        # pass

    def update_desired_position(self, x_des):
        self.state[-3:] = torch.tensor(x_des)

    def compute_plan(self):
        
        self.state[0:self.nq] = torch.tensor(self.joint_positions)
        self.state[self.nq:2*self.nq] = torch.tensor(self.joint_velocities)
        pred_norm = self.nn(self.state)
        pred = pred_norm * self.std + self.mean
        
        self.ioc.weight = torch.nn.Parameter(pred[0:self.n_vars])
        self.ioc.x_nom = torch.nn.Parameter(pred[self.n_vars:])

        x_pred = self.ioc(self.state[:-3]) 
        x_pred = x_pred.detach().numpy()

        return x_pred
        
    def set_gains(self, kp, kd):
        self.kp = kp
        self.kd = kd

    def run(self, thread):
        
        if thread.ti % int(self.dt*1000) == 0:
            count = self.count
            # if thread.ti == 0:

            q_des = self.x_pred[count*3*self.nq:count*3*self.nq+self.nq]
            dq_des = self.x_pred[count*3*self.nq+self.nq:count*3*self.nq+2*self.nq]
            a_des = self.x_pred[count*3*self.nq + 2*self.nq:count*3*self.nq+3*self.nq]

            if self.count == self.n_col - 1 and thread.ti != 0:
                self.x_pred = self.compute_plan()
                self.count = -1

            count = self.count
            tmp = count + 1
            nq_des = self.x_pred[tmp*3*self.nq:tmp*3*self.nq+self.nq]
            ndq_des = self.x_pred[tmp*3*self.nq+self.nq:tmp*3*self.nq+2*self.nq]
            na_des = self.x_pred[tmp*3*self.nq + 2*self.nq:tmp*3*self.nq+3*self.nq]

            self.q_int = np.linspace(q_des, nq_des, self.inter)
            self.dq_int = np.linspace(dq_des, ndq_des, self.inter)
            self.a_int = np.linspace(a_des, na_des, self.inter)

            self.count += 1
            self.index = 0

        # controller
        q = self.joint_positions
        v = self.joint_velocities
        tau = np.reshape(pin.rnea(self.pinModel, self.pinData, q, v, self.a_int[self.index]), (self.nq,))
        tau_gain = -self.kp*(np.subtract(q.T, self.q_int[self.index].T)) - self.kd*(np.subtract(v.T, self.dq_int[self.index].T))
        tau_total = np.reshape((tau_gain + tau), (7,)).T

        self.index += 1
        t2 = time.time()
        self.head.set_control('ctrl_joint_torques', tau_total)



class Net(torch.nn.Module):

    def __init__(self, inp_size, out_size):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(inp_size, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.out = torch.nn.Linear(512, out_size)

    def forward(self, x):
       
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.out(x)
        return x