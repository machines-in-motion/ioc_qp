## This class usses diff qp to compute the weights (IOC)
## Author : Avadesh Meduri
## Date : 22/02/2022
import time
import numpy as np
from pandas import isna
import torch
import torch.nn as nn
from torch.nn import functional as F

from inverse_kinematics import InverseKinematics
from diff_qp import DiffQP
from torchvision.transforms import ToTensor, ToPILImage
from dynamic_graph_head import ImageLogger 

from multiprocessing import Process, Pipe


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
        self.n_col = problem.n
        self.dt = problem.dt
        
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

    def __init__(self, nn_dir, m, std, cnn_dir = None):
        """
        Input:
            nn_dir : directory for NN weights
            ioc : ioc QP
            m : mean of the trained data (y_train)
            std : standard deviation of trained data (y_train)
            cnn_dir : directory of cnn weights
        """

        self.nq = 7
        self.n_col = 5
        self.state = np.zeros(2*self.nq)

        u_max = [2.5,2.5,2.5, 1.5, 1.5, 1.5, 1.0]
        self.dt = 0.05

        self.ioc = IOC(self.n_col, self.nq, u_max, self.dt, eps = 1.0, isvec=True)
        self.m = m
        self.std = std
        self.n_vars = self.ioc.n_vars
        self.nn = Net(2*self.nq + 3, 2*self.n_vars)
        self.nn.load_state_dict(torch.load(nn_dir, map_location=torch.device('cpu')))
        
        if cnn_dir:
            self.cnet = C_Net()
            self.cnet.load_state_dict(torch.load(cnn_dir, map_location=torch.device('cpu')))

        # for data recording
        self.img_par, self.img_child = Pipe()
        self.data = {"color_image": [], "depth_image": [], "position": []}
        self.subp = Process(target=ImageLogger, args=(["color_image", "depth_image", "position"], "data4", 0.1, \
                    self.img_child))
        self.subp.start()
        self.ti = 0

    def predict(self, q, dq, x_des):

        nq = self.ioc.nq
        n_vars = self.ioc.n_vars
        state = np.zeros(2*nq)
        state[0:nq] = q
        state[nq:] = dq
        x_input = torch.hstack((torch.tensor(state), torch.tensor(x_des))).float()
        pred_norm = self.nn(x_input)
        pred = pred_norm * self.std + self.m

        # # if not self.ioc.isvec:
        #     self.ioc.weight = torch.nn.Parameter(torch.reshape(pred[0:n_vars**2], (n_vars, n_vars)))
        #     self.ioc.x_nom = torch.nn.Parameter(pred[n_vars**2:])
        # else:
        self.ioc.weight = torch.nn.Parameter(pred[0:n_vars])
        self.ioc.x_nom = torch.nn.Parameter(pred[n_vars:])

        x_pred = self.ioc(state) 
        x_pred = x_pred.detach().numpy()

        return x_pred

    def cnn_predict(self, image):
        image = ToTensor()(image.astype(np.float))[None,:,:,:].float()
        pred = self.cnet(image)

        return pred.detach().numpy()

    def predict_rt(self, child_conn):
        while True:
            q, dq, x_des, image = child_conn.recv()
            t1 = time.time()
            if isinstance(image, np.ndarray):
                # print("predecting..")
                # c_pred = self.cnn_predict(image)
                # print(c_pred, x_des, np.linalg.norm(c_pred - x_des))
                
                self.data["color_image"] = image[:,:,:3]
                self.data["depth_image"] = image[:,:,3]
                self.data["position"] = x_des
                self.img_par.send((self.data, self.ti))
                self.ti += 1

            x_pred = self.predict(q, dq, x_des)
            t2 = time.time()
            child_conn.send((x_pred))
            # print("compute time", t2 - t1)


def subprocess_mpc_entry(channel, nn_dir, mean, std, cnn_dir):
    planner = IOCForwardPass(nn_dir, mean, std, cnn_dir)
    planner.predict_rt(channel)

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

class C_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(293904, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
#         print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x