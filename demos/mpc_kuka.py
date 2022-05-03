## This is a test of reaching task with kuka in simulation
## Author : Avadesh Meduri
## Date : 1/03/2022

import sys
sys.path.append("/home/ameduri/pydevel/ioc_qp/")

import numpy as np
import torch
from torch.nn import functional as F

from python.vocam.forward_pass import IOCForwardPassWithoutVision, rt_IOCForwardPassWithoutVision
from python.env.kuka_bullet_env import KukaBulletEnv

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
u_max = [2.5,2.5,2.5, 1.5, 1.5, 1.5, 1.0]

# loading mean and std
m = torch.load("../data/mean.pt")
std = torch.load("../data/std.pt")

# loading forward pass class
nn_dir = "../models/test4"
rt = False
if not rt:
    iocfp = IOCForwardPassWithoutVision(nn_dir, m, std, u_max)
else:
    ## if real time (computation will be done in a parallel thread)
    parent_conn, child_conn = Pipe()
    subp = Process(target=rt_IOCForwardPassWithoutVision, args=(child_conn, nn_dir, m, std, u_max))
    subp.start()

x_train = torch.load("../data/x_train1.pt")
i = 0
# x_des = x_train[i][-3:].detach().numpy()
x_des_arr = np.array([[0.5, -0.4, 0.7], [0.6, 0.4, 0.5]])


robot = KukaBulletEnv()
robot.set_gains(1.5, 0.05)

q_init = x_train[i][0:7].detach().numpy()
v_init = x_train[i][7:14].detach().numpy()

robot.reset_robot(q_init, v_init)

count = 0
state = np.zeros(2*nq)
eps = 10
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
        if not rt:
            x_pred = iocfp.predict(q, dq, x_des)    
        else:
            parent_conn.send((q, v, x_des))
            x_pred = parent_conn.recv()

    q_des = x_pred[count*3*nq:count*3*nq+nq]
    dq_des = x_pred[count*3*nq+nq:count*3*nq+2*nq]
    a_des = x_pred[count*3*nq + 2*nq:count*3*nq+3*nq]

    if count == n_col-1 and not rt:
        x_pred = iocfp.predict(q, dq, x_des)
        robot.plan.append(1)
        count = -1
    
    elif count == n_col - 2 and rt:
        parent_conn.send((q, v, x_des))

    if rt and v != 0:
        if parent_conn.poll():
            print("yes", count)
            x_pred = parent_conn.recv()
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
