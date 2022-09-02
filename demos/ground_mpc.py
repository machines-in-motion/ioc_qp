## This is a test of reaching task with kuka in simulation
## Author : Avadesh Meduri
## Date : 1/03/2022

import pathlib
import os
python_path = pathlib.Path('.').absolute().parent/'python'
os.sys.path.insert(1, str(python_path))

import pinocchio as pin
import numpy as np
from env.kuka_bullet_env import KukaBulletEnv
import torch
from torch.autograd import Function
from torch.nn import functional as F

from vocam.forward_pass import IOCForwardPassWithoutVision, rt_IOCForwardPassWithoutVision
from vocam.inverse_qp import IOC

import pybullet as p
from robot_properties_kuka.config import IiwaConfig

import time
from vocam.diff_pin_costs import DiffFrameTranslationCost, DiffFrameVelocityCost, DiffFramePlacementCost
from scipy.spatial.transform import Rotation as R


robot = IiwaConfig.buildRobotWrapper()
model, data = robot.model, robot.data
f_id = model.getFrameId("EE")
 
dtc = DiffFrameTranslationCost.apply
dvc = DiffFrameVelocityCost.apply
dfc = DiffFramePlacementCost.apply

ori_mat = pin.utils.rpyToMatrix(-np.pi/4, np.pi/2, 0)
# ori_mat = pin.utils.rpyToMatrix(-np.pi/4, np.pi/4, 0)
# ori_mat = pin.utils.rpyToMatrix(np.pi/6, np.pi/2, -np.pi/6)

quat = R.from_matrix(ori_mat).as_quat()

def quadratic_loss(q_pred, goal, nq, n_col):
    
    O_des = np.array(ori_mat)
#     error = doc(q_pred[-2*nq:], model, data, f_id, O_des)
#       wt = 1e1*np.eye(3)
  
    M_des = np.eye(4,4)
    M_des[0:3,0:3] = O_des
    M_des[0:3,3] = goal
    wt = 3.5e2*np.eye(6)
    wt[3,3] = 2e2
    wt[4,4] = 2e2
    wt[5,5] = 2e2
    error = dfc(q_pred[-2*nq:], model, data, f_id, M_des)
    
    loss = error.t()@torch.tensor(wt)@error
    
    
    loss += 2.5e0*torch.linalg.norm(dvc(q_pred[-2*nq:], torch.zeros(nq), model, data, f_id)) # asking for zero velocity
    loss += 1e-3*torch.linalg.norm(q_pred[-2*nq:-nq]) # joint regularization
    
    for i in range(n_col):    
        loss += 2e0 * torch.linalg.norm(dtc(q_pred[(3*i)*nq: (3*i+2)*nq], model, data, f_id) - goal)
        loss += 5e-1*torch.linalg.norm(dvc(q_pred[(3*i)*nq: (3*i+2)*nq], q_pred[(3*i+2)*nq:(3*i+3)*nq], model, data, f_id)) # asking for zero velocity
        loss += 1e-2*torch.linalg.norm(q_pred[(3*i+2)*nq: (3*i+3)*nq]) # control regularization
        loss += 2e-1*torch.linalg.norm(q_pred[(3*i+1)*nq: (3*i+2)*nq]) # velocity regularization
        loss += 5e-3*torch.linalg.norm(q_pred[(3*i)*nq: (3*i+1)*nq]) # joint regularization
        loss += 2e-3*torch.linalg.norm(q_pred[(3*i)*nq+3: (3*i+1)*nq])
        loss += 4e-3*torch.linalg.norm(q_pred[(3*i)*nq+5])
        
        if i < n_col - 1:
            loss += 5e-2*torch.linalg.norm(torch.subtract(q_pred[(3*i+2)*nq: (3*i+3)*nq], \
                                                          q_pred[(3*i+5)*nq: (3*i+6)*nq]))

    return loss

def regress(q, dq):
    ioc = IOC(n_col, nq, u_max, 0.05, eps = 1.0, isvec=isvec)
    optimizer = torch.optim.Adam(ioc.parameters(), lr=lr)
    o = 0
    loss = 1000.
    old_loss = 10000.
    x_init = np.zeros(2*nq)
    x_init[:nq] = q
    x_init[nq:] = dq
    while loss > 0.03 and o < 80 and abs(old_loss - loss) > 5e-4:
        x_pred = ioc(x_init) 
        old_loss = loss
        loss = quadratic_loss(x_pred, torch.tensor(x_des), nq, n_col)
        print(" Iteration :" + str(o) + "/" + str(80) +  " loss is : " + str(loss.detach().numpy()), end = '\r', flush = True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        o += 1

    x_pred = ioc(x_init).detach().numpy()
    return x_pred

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

x_des_arr = np.array([[0.6, 0.2, 0.5], [0.6, 0.2, 0.5]])

robot = KukaBulletEnv()
robot.set_gains(2.5, 0.1)

q_des = np.array( [1.3737, 0.9711, 1.6139, 1.2188, 1.5669, 0.1236, 0.2565])

q_init =  q_des + 0.3*(np.random.rand(len(q_des)) - 0.5)*2
robot.reset_robot(q_init, np.zeros_like(q_des))

count = 0
state = np.zeros(2*nq)
eps = 22
nb_switches = 1
count = 0
pln_freq = n_col-1
lag = 0

target = p.loadURDF("./sphere.urdf", [0,0,0])
q_arr = []
dq_arr = []

for v in range(nb_switches*n_col*eps) :

    q, dq = robot.get_state()

    if v % (n_col*eps) == 0:
        # x_des = x_des_arr[np.random.randint(len(x_des_arr))]
        x_des = x_des_arr[0]
        p.resetBasePositionAndOrientation(target, x_des, quat)
        # print("running feedback number : " + str(j),  end = '\r', flush = True )
        robot.plan.append(1)

    if v == 0:
        x_pred = regress(q, dq)


    q_des = x_pred[count*3*nq:count*3*nq+nq]
    dq_des = x_pred[count*3*nq+nq:count*3*nq+2*nq]
    a_des = x_pred[count*3*nq + 2*nq:count*3*nq+3*nq]

    if count == pln_freq:
        x_pred_wait = regress(q, dq)
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
        # time.sleep(0.0005)
        q, dq = robot.get_state()
        q_arr.append(q)
        dq_arr.append(dq)


robot.robot.start_recording("./ori1.mp4")
for i in range(len(q_arr)):
    robot.reset_robot(q_arr[i], dq_arr[i])
    time.sleep(0.001)

# robot.plot()
# robot.robot.stop_recording("./test.mp4")
