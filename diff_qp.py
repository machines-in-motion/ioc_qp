## This an implementation of the differntiable QP paper of brandon Amos
## Author : Avadesh Meduri
## Date : 15/02/2022

import numba
import numpy as np

import torch
from torch.autograd import Function
from torch.nn import functional as F
from solver import quadprog_solve_qp

import time

class DiffQP(Function):


    @staticmethod
    def forward(ctx, Q, q, G, h, A, b):
    
        Q = Q.double().detach().numpy()
        q = q.double().detach().numpy()
        t1 = time.time()
        x_opt, lag = quadprog_solve_qp(Q, q, G, h, A, b)
        t2 = time.time()
        # print("solve time:", t2 - t1)

        x_opt, lag = torch.tensor(x_opt), torch.tensor(lag)
        
        ctx.save_for_backward(torch.tensor(Q), torch.tensor(A), torch.tensor(G), torch.tensor(h), x_opt, lag)

        return x_opt
    
    @staticmethod
    def backward(ctx, grad):
        
        t1 = time.time()

        Q, A, G, h, x_opt, lag = ctx.saved_tensors
    
        # creating kkt matrix
        t5 = time.time()
        
        D_lam = torch.diag(lag) #Diagona(lagrange multipliers for inequality constriants)
        D_g = torch.diag(np.matmul(G, x_opt) - h) #Diagonal(G*x^{*} - h)
        
        kkt = torch.zeros((Q.shape[0] + G.shape[0] + A.shape[0], Q.shape[1] + D_g.shape[1] + A.shape[0]))
        kkt[:Q.shape[0], :Q.shape[0]] = Q
        kkt[:Q.shape[0], Q.shape[0]:Q.shape[0]+D_g.shape[1]] = torch.matmul(G.T, D_lam)
        kkt[:Q.shape[0], Q.shape[0]+D_g.shape[1]:] = A.T
        kkt[Q.shape[0]: Q.shape[0] + G.shape[0], :G.shape[1]] = G
        kkt[Q.shape[0]: Q.shape[0] + G.shape[0], G.shape[1] : G.shape[1]+ D_g.shape[1]] = D_g
        kkt[-A.shape[0]:, :A.shape[1]] = A

        t6 = time.time()

        t3 = time.time()
        sol = torch.linalg.solve(kkt.double(), torch.hstack((grad.double(), torch.zeros(A.shape[0]+G.shape[0]))))
        
        t4 = time.time()

        dz = sol[:Q.shape[0]]
        
        # The minus comes because of the Quadprog wrapper adding minus on the matrices
        dl_dq = -dz

        dz = dz.reshape(-1,1)
        x_opt = x_opt.reshape(-1,1)
        dl_dQ = -0.5*(torch.matmul(dz, x_opt.T) + torch.matmul(x_opt, dz.T))
                
        t2 = time.time()
        # print("backward time:", t2 - t1)
        # print("backward linalg time:", t4 - t3)

        # print("kkt :", t6 - t5)
        # print("rest :", t2 - t4)

        return dl_dQ, dl_dq, None, None, None, None
        
        