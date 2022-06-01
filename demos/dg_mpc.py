import pathlib
import os
python_path = pathlib.Path('.').absolute().parent/'python'
os.sys.path.insert(1, str(python_path))

import torch
from dg_iocqp import DiffQPController
import pybullet as p
import numpy as np
from dynamic_graph_head import ThreadHead, Vicon, SimHead, HoldPDController

import time
from vocam.nets import Net

import dynamic_graph_manager_cpp_bindings
from robot_properties_kuka.config import IiwaConfig
from robot_properties_kuka.iiwaWrapper import IiwaRobot
from bullet_utils.env import BulletEnvWithGround

from mim_data_utils import DataReader

run_sim = False

# x_des_arr = np.array([[0.5, -0.5, 0.4], [0.5, 0.4, 0.6], [0.4, -0.4, 0.4], [0.7, 0.4, 0.5]])
# x_des = x_des_arr[1]
x_train = torch.load("../data/x_train8.pt")
# i = np.random.randint(0))
i = 0
x_des = x_train[i][-3:].detach().numpy()

if run_sim:
    env = BulletEnvWithGround(p.GUI)

    robot = IiwaRobot()

    env.add_robot(robot)
    
    q_init = x_train[i][0:7].detach().numpy()
    # q_init += 0.5*(np.random.rand(len(q_init)) - 0.5)*2 
    v_init = x_train[i][7:14].detach().numpy()
    
    # reader = DataReader('test.mds')
    # q_init = reader.data['joint_positions'][0]
    # v_init = reader.data['joint_velocities'][0]
    # print(q_init, v_init)
    robot.reset_state(q_init, v_init)
   
    target = p.loadURDF("./sphere.urdf", [0,0,0])

    head = SimHead(robot, with_sliders=False)

else:

    path = "/home/ameduri/devel/workspace/install/robot_properties_kuka/lib/python3.8/site-packages/robot_properties_kuka/robot_properties_kuka/dynamic_graph_manager/dgm_parameters_iiwa.yaml"
    head = dynamic_graph_manager_cpp_bindings.DGMHead(path)
    target = None
    env = None

pin_robot = IiwaConfig.buildRobotWrapper()

# loading mean and std
# m = torch.load("../data/mean.pt")
# std = torch.load("../data/std.pt")
m = None
std = None
nq = 7
n_col = 5
n_vars = 3*nq*n_col+2*nq

# nn_dir = "/home/ameduri/pydevel/ioc_qp/models/qpnet_77.pt"
nn_dir = "/home/ameduri/pydevel/ioc_qp/models/qpnet_91.pt"
# nn_dir = "/home/ameduri/pydevel/ioc_qp/models/model2"

ctrl = DiffQPController(head, pin_robot.model, pin_robot.data, nn_dir, m, std, vicon_name = "cube10/cube10", target = target, run_sim = run_sim)
ctrl.update_desired_position(x_des)
if not run_sim:
    # kp = np.array([250.0, 250.0, 250.0, 250.0, 180.0, 30.0, 30.0])
    # kd = np.array([15.0, 15.0, 18.0, 18.0, 18.0, 5.0, 5.0])
    # kp = np.array([30.0, 30.0, 30.0, 20.0, 20.0, 10.0, 10.0])
    # kd = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0])
    kp = np.array([10.0, 10.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    kd = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    ctrl.set_gains(kp, kd)
   
else:
    ctrl.set_gains(1.5, 0.05)
thread_head = ThreadHead(
    0.001, # dt.
    HoldPDController(head, 50., 0.5, with_sliders=False), # Safety controllers.
    head, # Heads to read / write from.
    [('vicon', Vicon('172.24.117.119:801', ['cube10/cube10']))
    ], 
    env # Environment to step.
)


thread_head.switch_controllers(ctrl)
if run_sim:
    # thread_head.start_logging(6, "test.mds")
    thread_head.sim_run_timed(200000)
    # thread_head.stop_logging()

else:
    thread_head.start()
    thread_head.start_logging(30, "test.mds")
    # time.sleep(30)
    # thread_head.plot_timing()

