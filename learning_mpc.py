## This is a test of reaching task with kuka in simulation
## Author : Avadesh Meduri
## Date : 1/03/2022

import numpy as np
from kuka_bullet_env import KukaBulletEnv
import torch
from torch.autograd import Function
from torch.nn import functional as F

from inverse_qp import IOC
import pybullet as p

import time

class Net(torch.nn.Module):

    def __init__(self, inp_size, out_size):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        
        self.fc1 = torch.nn.Linear(inp_size, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.out = torch.nn.Linear(512, out_size)

    def forward(self, x):
       
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.out(x))
        return x

x_init = np.zeros(14)
nq = 7
dt = 0.05
n_col = 5
u_max = [2.5,2.5,2.5, 1.5, 1.5, 1.5, 1.0]

lr = 1e-1
eps = 80
isvec = True

ioc = IOC(n_col, nq, u_max, dt, eps = 1.0, isvec=isvec)
n_vars = 3*nq*n_col + 2*nq

# loading mean and std
# m = torch.load("./data/mean.pt")
# std = torch.load("./data/std.pt")

# loading model
nn = Net(2*nq + 3, 2*n_vars)
nn.load_state_dict(torch.load("./models/test1"))

x_des_arr = np.array([[0.5, -0.4, 0.4], [0.6, 0.4, 0.7]])

robot = KukaBulletEnv()
robot.set_gains(1.5, 0.05)

q_des = np.array( [1.3737, 0.9711, 1.6139, 1.2188, 1.5669, 0.1236, 0.2565])

q_init =  q_des + 0.3*(np.random.rand(len(q_des)) - 0.5)*2
robot.reset_robot(q_init, np.zeros_like(q_des))

count = 0
state = np.zeros(2*nq)
eps = 50

# robot.robot.start_recording("./test.mp4")
target = p.loadURDF("/home/ameduri/devel/workspace/dif_ddp/sphere.urdf", [0,0,0])

for k in range(10):

    x_des = x_des_arr[np.random.randint(len(x_des_arr))]
    p.resetBasePositionAndOrientation(target, x_des, (0,0,0,1))
    print("new k")
    for i in range(eps):
        
        print("running feedback number : " + str(i),  end = '\r', flush = True )
        q, dq = robot.get_state()

        state[0:nq] = q
        state[nq:] = dq
        x_input = torch.hstack((torch.tensor(state), torch.tensor(x_des))).float()
        pred = nn(x_input)
        # pred = pred_norm * std + m

        if not isvec:
            ioc.weight = torch.nn.Parameter(torch.reshape(pred[0:n_vars**2], (n_vars, n_vars)))
            ioc.x_nom = torch.nn.Parameter(pred[n_vars**2:])
        else:
            ioc.weight = torch.nn.Parameter(pred[0:n_vars])
            ioc.x_nom = torch.nn.Parameter(pred[n_vars:])

        x_pred = ioc(state) 
        x_pred = x_pred.detach().numpy()

        for count in range(int(n_col)):

            q_des = x_pred[count*3*nq:count*3*nq+nq]
            dq_des = x_pred[count*3*nq+nq:count*3*nq+2*nq]
            a_des = x_pred[count*3*nq + 2*nq:count*3*nq+3*nq]
            for i in range(int(dt/0.001)):
                robot.send_id_command(q_des, dq_des, a_des)
                time.sleep(0.0005)

robot.plot()
# robot.robot.stop_recording("./test.mp4")
