## This class usses diff qp to compute the weights (IOC)
## Author : Avadesh Meduri
## Date : 22/02/2022

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from inverse_kinematics import InverseKinematics
from diff_qp import DiffQP

class IOC(torch.nn.Module):
    
    def __init__(self, n : int, nq : int, tau_lim : list, dt : float = 0.05, eps : float = 0.1, isvec = False):
        
        """
        This class regresses to compute the weights of the MPC/IK problem given a loss
        Input:
            n : number of collocation points
            nq : number of joints/DOF in the system
            tau_lim: torque limits (len(tau_lim) == nq)
            dt : discretization of time
            eps : epsillon value added to make sure the cost matrix is Postive definite
            isvec : sets weight as a vector for Q matrix
        """
        
        super(IOC, self).__init__()
        
        self.eps = eps
        self.nq = nq
        self.isvec = isvec
        self.n_vars = 3*nq*n+2*nq

        problem = InverseKinematics(n, nq, tau_lim, dt)
        
        self.A, self.b, self.G, self.h = problem.create_matrices_nn()
        self.R = self.eps * torch.eye(self.n_vars)
        
        if self.isvec:
            self.Q = torch.zeros((self.n_vars, self.n_vars, self.n_vars), dtype = torch.float)
            for i in range(self.n_vars):
                self.Q[i,i,i] = 1.0
        
            self.weight = torch.nn.Parameter(torch.rand(self.n_vars, dtype = torch.float))
        else:
            self.weight = torch.nn.Parameter(torch.tril(torch.rand((self.n_vars, self.n_vars), dtype = torch.float)))
        
        self.x_nom = torch.nn.Parameter(0.005*torch.ones(self.n_vars, dtype = torch.float))
    

    def forward(self, x_init):

        # makes the weight positive definite
        psd_weight = F.relu(self.weight)
        
        if self.isvec:
            self.Q_torch = 0.5*torch.matmul(self.Q, psd_weight) + self.R
        else:
            self.Q_torch = 0.5*psd_weight.mm(psd_weight.t()) + self.R
    
        q = -1 * torch.matmul(self.Q_torch, self.x_nom)
        
        self.b[-2*self.nq:] = x_init
        
        x_opt = DiffQP.apply(self.Q_torch, q, self.G, self.h, self.A, self.b)
        
        return x_opt



class IOCForwardPass:

    def __init__(self, nn, ioc, m, std):
        """
        Input:
            nn : trained neural network
            ioc : ioc QP
            m : mean of the trained data (y_train)
            std : standard deviation of trained data (y_train)
        """
        
        self.nn = nn
        self.ioc = ioc
        self.m = m
        self.std = std

    def predict(self, q, dq, x_des):

        nq = self.ioc.nq
        n_vars = self.ioc.n_vars
        state = np.zeros(2*nq)
        state[0:nq] = q
        state[nq:] = dq
        x_input = torch.hstack((torch.tensor(state), torch.tensor(x_des))).float()
        pred_norm = self.nn(x_input)
        pred = pred_norm * self.std + self.m

        if not self.ioc.isvec:
            self.ioc.weight = torch.nn.Parameter(torch.reshape(pred[0:n_vars**2], (n_vars, n_vars)))
            self.ioc.x_nom = torch.nn.Parameter(pred[n_vars**2:])
        else:
            self.ioc.weight = torch.nn.Parameter(pred[0:n_vars])
            self.ioc.x_nom = torch.nn.Parameter(pred[n_vars:])

        x_pred = self.ioc(state) 
        x_pred = x_pred.detach().numpy()

        return x_pred

