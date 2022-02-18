## This an implementation of the differntiable QP paper of brandon Amos
## Author : Avadesh Meduri
## Date : 15/02/2022

import numpy as np

import torch
from torch.autograd import Function
from torch.nn import functional as F
from solver import quadprog_solve_qp

class DiffQP(Function):
    
    @staticmethod
    def forward(ctx, Q, q, G, h, A, b):
    
        Q = Q.double().detach().numpy()
        q = q.double().detach().numpy()
        x_opt, lag = quadprog_solve_qp(Q, q, G, h, A, b)
        x_opt, lag = torch.tensor(x_opt), torch.tensor(lag)
        
        ctx.save_for_backward(torch.tensor(Q), torch.tensor(A), torch.tensor(G), torch.tensor(h), x_opt, lag)

        return x_opt
    
    @staticmethod
    def backward(ctx, grad):
        
        Q, A, G, h, x_opt, lag = ctx.saved_tensors
        Q, A, G, h, x_opt, lag = Q.detach().numpy(), A.detach().numpy(), G.detach().numpy(),\
                                    h.detach().numpy(), x_opt.detach().numpy(), lag.detach().numpy()
        # creating kkt matrix
        assert len(lag) == G.shape[0]
        
        D_lam = np.diag(lag) #Diagona(lagrange multipliers for inequality constriants)
        D_g = np.diag(np.matmul(G, x_opt) - h) #Diagonal(G*x^{*} - h)
        
        kkt = np.block([[Q, np.matmul(G.T,D_lam), A.T],
                        [G, D_g, np.zeros((G.shape[0], A.shape[0]))],
                        [A, np.zeros((A.shape[0], D_g.shape[1])), np.zeros((A.shape[0], A.shape[0]))]])
        
        
        
        sol = np.linalg.solve(kkt, np.hstack((grad.detach().numpy(), np.zeros(A.shape[0]+G.shape[0]))))
        dz = sol[:Q.shape[0]]
        
        # The minus comes because of the Quadprog wrapper adding minus on the matrices
        dl_dq = -torch.tensor(dz, dtype = torch.double)

        dz = dz.reshape(-1,1)
        x_opt = x_opt.reshape(-1,1)
        dl_dQ = -torch.tensor(0.5*(np.matmul(dz, x_opt.T) + np.matmul(x_opt, dz.T)), dtype = torch.double)
                
        return dl_dQ, dl_dq, None, None, None, None
        
        