## This is a test of reaching task with kuka in simulation
## Author : Avadesh Meduri
## Date : 1/03/2022

import pathlib
import os
python_path = pathlib.Path('.').absolute().parent/'python'
os.sys.path.insert(1, str(python_path))

import numpy as np
import torch
from torch.nn import functional as F

from vocam.forward_pass import IOCForwardPassWithoutVision, rt_IOCForwardPassWithoutVision
from vocam.nets import Net
from env.kuka_bullet_env import KukaBulletEnv

import pybullet as p
from robot_properties_kuka.config import IiwaConfig

import time
from multiprocessing import Process, Pipe


robot = IiwaConfig.buildRobotWrapper()
model, data = robot.model, robot.data

x_init = np.zeros(14)
nq = 7
dt = 0.05
n_col = 5
u_max = [3.5,4.5,2.5, 2.5, 1.5, 1.5, 1.0]
n_vars = 3*nq*n_col+2*nq

# loading forward pass class
use_nn = False

if os.getlogin() == "ameduri" and use_nn:
    print("using nn")
    nn_dir = "../models/model2"
    nn = Net(2*nq + 3, 2*n_vars)
    nn.load_state_dict(torch.load(nn_dir))   
    # loading mean and std
    m = torch.load("../data/mean.pt")
    std = torch.load("../data/std.pt")

    x_train = torch.load("../data/x_train3.pt")


else:
    print("using qpnet")
    from vocam.qpnet import QPNet
    nn_dir = "../models/qpnet_89.pt"
    nn = QPNet(2*nq + 3, 2*n_vars).eval()
    nn.load(nn_dir)

    x_train = torch.load("../data/x_train8.pt")

    # data_train = torch.load("../data/data_100_50.pt")
    # unzipped = list(zip(*data_train))
    # x_train = torch.vstack([*unzipped[0]])
    # y_train = torch.vstack([*unzipped[1]])

rt = False
if not rt:
    iocfp = IOCForwardPassWithoutVision(nn, u_max=u_max)
else:
    ## if real time (computation will be done in a parallel thread)
    parent_conn, child_conn = Pipe()
    subp = Process(target=rt_IOCForwardPassWithoutVision, args=(child_conn, nn_dir, m, std, u_max))
    subp.start()


i = np.random.randint(2)
x_des = x_train[i][-3:].detach().numpy()
x_des_arr = np.array([[0.5, 0.4, 0.7], [0.6, 0.4, 0.5], [0.5, -0.4, 0.7], [0.6, -0.4, 0.5]])


robot = KukaBulletEnv()
robot.set_gains(1.5, 0.05)

q_init = x_train[i][0:7].detach().numpy()
v_init = x_train[i][7:14].detach().numpy()

robot.reset_robot(q_init, v_init)

count = 0
state = np.zeros(2*nq)
eps = 15
nb_switches = 5
count = 0
pln_freq = n_col - 2
lag = 1

# robot.robot.start_recording("./test.mp4")
target = p.loadURDF("./sphere.urdf", [0,0,0])

for v in range(nb_switches*n_col*eps) :

    q, dq = robot.get_state()

    if v % (n_col*eps) == 0:
        x_des = x_des_arr[np.random.randint(len(x_des_arr))]
        p.resetBasePositionAndOrientation(target, x_des, (0,0,0,1))
        robot.plan.append(1)
    
    if v == 0:
        x_pred = iocfp.predict(q, dq, x_des)    
    
    q_des = x_pred[count*3*nq:count*3*nq+nq]
    dq_des = x_pred[count*3*nq+nq:count*3*nq+2*nq]
    a_des = x_pred[count*3*nq + 2*nq:count*3*nq+3*nq]

    if count == pln_freq and not rt:
        x_pred_wait = iocfp.predict(q, dq, x_des)
        robot.plan.append(1)
    
    if count == pln_freq + lag:
        x_pred = x_pred_wait
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
