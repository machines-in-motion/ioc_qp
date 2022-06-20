## Contains the class that implementes the online adaptaion of weights on the robot without vision
## Author : Avadesh Meduri
## Date : 3/05/2022 

import time
import numpy as np
import torch
from torch.nn import functional as F

from vocam.nets import Net
from vocam.inverse_qp import IOC

class IOCForwardPassWithoutVision:

    def __init__(self, nn, m = None, std = None, u_max =  [2.5,2.5,2.5, 1.5, 1.5, 1.5, 1.0]):
        """
        Input:
            nn : nn class
            ioc : ioc QP
            m : mean of the trained data (y_train)
            std : standard deviation of trained data (y_train)
            u_max : max acceleration allowed in each joints
        """

        self.nq = 7
        self.n_col = 5
        self.state = np.zeros(2*self.nq)

        self.dt = 0.05

        self.ioc = IOC(self.n_col, self.nq, u_max, self.dt, eps = 1.0, isvec=True)
        self.m = m
        self.std = std
        self.n_vars = self.ioc.n_vars
        self.nn = nn

    def predict_obstacle(self, q, dq, x_des, obstacle=True):

        nq = self.ioc.nq
        n_vars = self.ioc.n_vars
        state = np.zeros(2*nq)
        state[0:nq] = q
        state[nq:] = dq

        if obstacle:
            x_input = torch.hstack((torch.tensor(state), torch.tensor(x_des), 0.5 * torch.ones(3))).float()
        else:
            x_input = torch.hstack((torch.tensor(state), torch.tensor(x_des), 0.05 * torch.zeros(3))).float()

        
        pred_norm = self.nn(x_input)
        if torch.is_tensor(self.std):
            pred = pred_norm * self.std + self.m
        else:
            pred = pred_norm
        
        pred = torch.squeeze(pred)

        # # if not self.ioc.isvec:
        #     self.ioc.weight = torch.nn.Parameter(torch.reshape(pred[0:n_vars**2], (n_vars, n_vars)))
        #     self.ioc.x_nom = torch.nn.Parameter(pred[n_vars**2:])
        # else:
        self.ioc.weight = torch.nn.Parameter(pred[0:n_vars])
        self.ioc.x_nom = torch.nn.Parameter(pred[n_vars:])

        x_pred = self.ioc(state)
        x_pred = x_pred.detach().numpy()


        return x_pred
        
    def predict(self, q, dq, x_des):

        nq = self.ioc.nq
        n_vars = self.ioc.n_vars
        state = np.zeros(2*nq)
        state[0:nq] = q
        state[nq:] = dq
        x_input = torch.hstack((torch.tensor(state), torch.tensor(x_des))).float()
        
        pred_norm = self.nn(x_input)
        if torch.is_tensor(self.std):
            pred = pred_norm * self.std + self.m
        else:
            pred = pred_norm
        
        pred = torch.squeeze(pred)

        # # if not self.ioc.isvec:
        #     self.ioc.weight = torch.nn.Parameter(torch.reshape(pred[0:n_vars**2], (n_vars, n_vars)))
        #     self.ioc.x_nom = torch.nn.Parameter(pred[n_vars**2:])
        # else:
        self.ioc.weight = torch.nn.Parameter(pred[0:n_vars])
        self.ioc.x_nom = torch.nn.Parameter(pred[n_vars:])

        x_pred = self.ioc(state)
        x_pred = x_pred.detach().numpy()


        return x_pred

    def predict_encoder(self, state, encoding):

        x_in = torch.hstack((torch.tensor(state)[None,], encoding)).float()
        pred = torch.squeeze(self.nn(x_in))
        n_vars = self.ioc.n_vars
        self.ioc.weight = torch.nn.Parameter(pred[0:n_vars])
        self.ioc.x_nom = torch.nn.Parameter(pred[n_vars:])

        x_pred = self.ioc(state) 
        x_pred = x_pred.detach().numpy()

        return x_pred

    def predict_rt(self, child_conn):
        while True:
            q, dq, x_des = child_conn.recv()

            t1 = time.time()
            x_pred = self.predict(q, dq, x_des)
            t2 = time.time()
            child_conn.send((x_pred))
            # print("compute time", t2 - t1)


## Call this function for multi threading
def rt_IOCForwardPassWithoutVision(channel, nn_dir, mean, std, u_max = [2.5,2.5,2.5, 1.5, 1.5, 1.5, 1.0]):
    planner = IOCForwardPassWithoutVision(nn_dir, mean, std, u_max)
    planner.predict_rt(channel)


