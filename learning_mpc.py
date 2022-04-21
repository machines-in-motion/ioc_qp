## This is a test of reaching task with kuka in simulation
## Author : Avadesh Meduri
## Date : 1/03/2022

import numpy as np
from kuka_bullet_env import KukaBulletEnv
import torch
from torch.autograd import Function
from torch.nn import functional as F

from inverse_qp import IOC, IOCForwardPass
import pybullet as p
from robot_properties_kuka.config import IiwaConfig

import time

robot = IiwaConfig.buildRobotWrapper()
model, data = robot.model, robot.data
f_id = model.getFrameId("EE")

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
m = torch.load("./data/mean.pt")
std = torch.load("./data/std.pt")

# loading forward pass class
iocfp = IOCForwardPass("./models/test2", m, std)

x_des_arr = np.array([[0.5, -0.4, 0.4], [0.6, 0.4, 0.7], [0.3, -0.4, 0.6], [0.2, 0.6, 0.1]])

robot = KukaBulletEnv()
robot.set_gains(1.5, 0.05)

q_des = np.array( [1.3737, 0.9711, 1.6139, 1.2188, 1.5669, 0.1236, 0.2565])

q_init =  q_des + 0.3*(np.random.rand(len(q_des)) - 0.5)*2
robot.reset_robot(q_init, np.zeros_like(q_des))

count = 0
state = np.zeros(2*nq)
eps = 25
nb_switches = 5
count = 0

# robot.robot.start_recording("./test.mp4")
target = p.loadURDF("./sphere.urdf", [0,0,0])


for v in range(nb_switches*n_col*eps) :

    q, dq = robot.get_state()
    if v % (n_col*eps) == 0:
        x_des = x_des_arr[np.random.randint(len(x_des_arr))]
        p.resetBasePositionAndOrientation(target, x_des, (0,0,0,1))
        # print("running feedback number : " + str(j),  end = '\r', flush = True )
        robot.plan.append(1)
    
    if v == 0:
        x_pred = iocfp.predict(q, dq, x_des)


    q_des = x_pred[count*3*nq:count*3*nq+nq]
    dq_des = x_pred[count*3*nq+nq:count*3*nq+2*nq]
    a_des = x_pred[count*3*nq + 2*nq:count*3*nq+3*nq]

    if count == n_col-1:
        x_pred = iocfp.predict(q, dq, x_des)
        robot.plan.append(1)
        count = -1
        
    tmp = count + 1
    nq_des = x_pred[tmp*3*nq:tmp*3*nq+nq]
    ndq_des = x_pred[tmp*3*nq+nq:tmp*3*nq+2*nq]
    na_des = x_pred[tmp*3*nq + 2*nq:tmp*3*nq+3*nq]
    
    q_int = np.linspace(q_des, nq_des, int(dt/0.001))
    dq_int = np.linspace(dq_des, ndq_des, int(dt/0.001))
    a_int = np.linspace(a_des, na_des, int(dt/0.001))
    
    count += 1
    for i in range(int(dt/0.001)):
        if count == n_col -1 and i == 0:
            pass
        else:
            robot.plan.append(0)
        robot.send_id_command(q_int[i], dq_int[i], a_int[i])
        time.sleep(0.0005)

robot.plot()
# robot.robot.stop_recording("./test.mp4")
