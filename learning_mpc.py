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
        
        self.fc1 = torch.nn.Linear(inp_size, 120)
        self.fc2 = torch.nn.Linear(120, 120)
        self.out = torch.nn.Linear(120, out_size)

    def forward(self, x):
       
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

x_init = np.zeros(14)
n_col = 10
nq = 7
dt = 0.05
u_max = nq*[5,]

lr = 1e-2
eps = 80
ioc = IOC(n_col, nq, u_max, dt, eps = 1.0, isvec=False)
n_vars = 3*nq*n_col + 2*nq

# loading model
nn = Net(2*nq + 3, n_vars**2 + n_vars)
nn.load_state_dict(torch.load("./models/test2"))

x_des = torch.tensor([-0.4712,  0.5082,  0.9357])

robot = KukaBulletEnv()
robot.set_gains(2.5, 0.1)

q_des = np.array([-0.58904862, -0.58904862, -0.58904862,  0.58904862,  0.        ,
        0.        ,  0.        ])

q_init =  q_des + (np.pi/4.0)*(np.random.rand(len(q_des)) - 0.5)*2
robot.reset_robot(q_init, np.zeros_like(q_des))

count = 0
state = np.zeros(2*nq)
eps = 10

p.loadURDF("/home/ameduri/devel/workspace/dif_ddp/sphere.urdf", x_des)

robot.robot.start_recording("./test.mp4")
for i in range(eps):
    
    print("running feedback number : " + str(i),  end = '\r', flush = True )
    q, dq = robot.get_state()

    state[0:nq] = q
    state[nq:] = dq
    x_input = torch.hstack((torch.tensor(state), x_des)).float()
    pred = nn(x_input)

    ioc.weight = torch.nn.Parameter(torch.reshape(pred[0:n_vars**2], (n_vars, n_vars)))
    ioc.x_nom = torch.nn.Parameter(pred[n_vars**2:])

    x_pred = ioc(state) 
    x_pred = x_pred.detach().numpy()

    for count in range(n_col):

        q_des = x_pred[count*3*nq:count*3*nq+nq]
        dq_des = x_pred[count*3*nq+nq:count*3*nq+2*nq]
        a_des = x_pred[count*3*nq + 2*nq:count*3*nq+3*nq]
        for i in range(int(dt/0.001)):
            robot.send_id_command(q_des, dq_des, a_des)
            time.sleep(0.001)

# robot.plot()
robot.robot.stop_recording("./test.mp4")
