## This is a demo to run vicon and the intelsense real camera live
## Author : Avadesh Meduri
## Date : 8/04/2021


import time
from dynamic_graph_head import ThreadHead, ImageLogger, HoldPDController, Vicon, VisionSensor
import dynamic_graph_manager_cpp_bindings
from robot_properties_kuka.config import IiwaConfig
import pinocchio as pin
import numpy as np
import cv2

from multiprocessing import Process, Pipe

import numpy as np

pin_robot = IiwaConfig.buildRobotWrapper()

class ConstantTorque:
    def __init__(self, head, robot_model, robot_data):
        self.head = head
        
        self.pinModel = robot_model
        self.pinData = robot_data

        self.joint_positions = head.get_sensor('joint_positions')
        self.joint_velocities = head.get_sensor("joint_velocities")
        self.joint_torques = head.get_sensor("joint_torques_total")
        self.joint_ext_torques = head.get_sensor("joint_torques_external")      
    
        self.parent_conn, self.child_conn = Pipe()
        self.data = {"color_image": [], "depth_image": [], "position": []}

    def warmup(self, thread_head):
        self.subp = Process(target=ImageLogger, args=(["color_image", "depth_image", "position"], "data21", 3.0, self.child_conn))
        self.subp.start()
        self.init = self.joint_positions.copy()

        pass

    def run(self, thread):
        t1 = time.time()
        q = self.joint_positions
        v = self.joint_velocities
        pos, vel = thread.vicon.get_state('cube10/cube10')
        self.color_image, self.depth_image = thread.camera.get_image()
        self.data["color_image"] = self.color_image
        self.data["depth_image"] = self.depth_image
        self.data["position"] = pos[0:3]
        self.parent_conn.send((self.data, thread.ti))

        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', self.color_image)
        cv2.waitKey(1)
        ti = thread.ti
        self.des_position = self.init.copy()
        self.des_position[0] += 0.2*np.sin(0.001*np.pi*ti)
        self.des_position[1] += 0.2*np.sin(0.001*np.pi*ti)
        self.tau = (
            40 * (self.des_position - self.joint_positions)
            - 0.5 * self.joint_velocities
        )
        pin.forwardKinematics(self.pinModel, self.pinData, q, v, np.zeros_like(q))
        pin.updateFramePlacements(self.pinModel, self.pinData)
        
        self.g_comp = pin.rnea(self.pinModel, self.pinData, q, v, np.zeros_like(q))
        self.head.set_control('ctrl_joint_torques', self.tau)

        t2 = time.time()

# Create the dgm communication and instantiate the controllers.
head = dynamic_graph_manager_cpp_bindings.DGMHead(IiwaConfig.yaml_path)


# Create the controllers.
hold_pd_controller = HoldPDController(head, 50., 0.5, with_sliders=False)
tau_ctrl = ConstantTorque(head, pin_robot.model, pin_robot.data)

thread_head = ThreadHead(
    0.001, # Run controller at 1000 Hz.
    hold_pd_controller,
    head,
    [('vicon', Vicon('172.24.117.119:801', ['cube10/cube10'])),
     ('camera', VisionSensor())
    ]
)

thread_head.switch_controllers(tau_ctrl)

# Start the parallel processing.
thread_head.start()

