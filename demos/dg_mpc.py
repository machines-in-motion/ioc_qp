
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

import dynamic_graph_manager_cpp_bindings
from robot_properties_kuka.config import IiwaConfig
from robot_properties_kuka.iiwaWrapper import IiwaRobot
from bullet_utils.env import BulletEnvWithGround

from mim_data_utils import DataReader

run_sim = True

# x_des_arr = np.array([[0.5, -0.5, 0.4], [0.5, 0.4, 0.6], [0.4, -0.4, 0.4], [0.7, 0.4, 0.5]])
# x_des = x_des_arr[1]
x_train = torch.load("../data/x_train1.pt")
i = np.random.randint(len(x_train))
x_des = x_train[i][-3:].detach().numpy()

if run_sim:
    env = BulletEnvWithGround(p.GUI)

    robot = IiwaRobot()

    env.add_robot(robot)
    
    q_init = x_train[i][0:7].detach().numpy()
    q_init += 0.5*(np.random.rand(len(q_init)) - 0.5)*2 
    v_init = x_train[i][7:14].detach().numpy()
    
    # reader = DataReader('test.mds')
    # q_init = reader.data['joint_positions'][0]
    # v_init = reader.data['joint_velocities'][0]
    # print(q_init, v_init)
    robot.reset_state(q_init, v_init)
   
    target = p.loadURDF("./sphere.urdf", [0,0,0])

    head = SimHead(robot, with_sliders=False)

else:

    head = dynamic_graph_manager_cpp_bindings.DGMHead(IiwaConfig.yaml_path)
    target = None
    env = None

pin_robot = IiwaConfig.buildRobotWrapper()

# loading mean and std
m = torch.load("../data/mean.pt")
std = torch.load("../data/std.pt")

ctrl = DiffQPController(head, pin_robot.model, pin_robot.data, "../models/test4", m, std, vicon_name = "cube10/cube10", target = target, run_sim = run_sim)
ctrl.update_desired_position(x_des)
if not run_sim:
    kp = np.array([250.0, 250.0, 250.0, 250.0, 180.0, 30.0, 30.0])
    kd = np.array([15.0, 15.0, 18.0, 18.0, 18.0, 5.0, 5.0])
    ctrl.set_gains(kp, kd)
    thread_head = ThreadHead(
        0.001, # dt.
        HoldPDController(head, 50., 0.5, with_sliders=False), # Safety controllers.
        head, # Heads to read / write from.
        [('vicon', Vicon('172.24.117.119:801', ['cube10/cube10']))
        ], 
        env # Environment to step.
    )
else:
    ctrl.set_gains(1.0, 0.05)
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
    thread_head.sim_run_timed(150000)
    # thread_head.stop_logging()

else:
    thread_head.start()
    # thread_head.start_logging(15, "test.mds")