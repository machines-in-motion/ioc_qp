

import torch
from dg_iocqp import DiffQPController
import pybullet as p
import numpy as np
from dynamic_graph_head import ThreadHead, SimHead, HoldPDController

import time

import dynamic_graph_manager_cpp_bindings
from robot_properties_kuka.config import IiwaConfig
from robot_properties_kuka.iiwaWrapper import IiwaRobot
from bullet_utils.env import BulletEnvWithGround

env = BulletEnvWithGround(p.GUI)

robot = IiwaRobot()
pin_robot = IiwaConfig.buildRobotWrapper()

env.add_robot(robot)
q_des = np.array( [1.3737, 0.9711, 1.6139, 1.2188, 1.5669, 0.1236, 0.2565])
q_init =  q_des + 0.3*(np.random.rand(len(q_des)) - 0.5)*2
robot.reset_state(q_init, np.zeros_like(q_init))

head = SimHead(robot, with_sliders=False)
thread_head = ThreadHead(
    0.001, # dt.
    HoldPDController(head, 50., 0.5, False), # Safety controllers.
    head, # Heads to read / write from.
    [], 
    env # Environment to step.
)

x_des_arr = np.array([[0.5, -0.4, 0.4], [0.6, 0.4, 0.7], [0.3, -0.4, 0.6], [0.2, 0.6, 0.1]])
x_des = x_des_arr[2]

target = p.loadURDF("./sphere.urdf", [0,0,0])
p.resetBasePositionAndOrientation(target, x_des, (0,0,0,1))

# loading mean and std
m = torch.load("./data/mean.pt")
std = torch.load("./data/std.pt")

ctrl = DiffQPController(head, pin_robot.model, pin_robot.data, "./models/test1", m, std)
ctrl.update_desired_position(x_des)
ctrl.set_gains(1.5, 0.05)

thread_head.switch_controllers(ctrl)
thread_head.sim_run_timed(100000)
