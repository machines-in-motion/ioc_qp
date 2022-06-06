## This is a demo to run vicon and the intelsense real camera live
## Author : Avadesh Meduri
## Date : 8/04/2021


import time
import dynamic_graph_manager_cpp_bindings
from dynamic_graph_head import ThreadHead, HoldPDController, Vicon, VisionSensor
from logger import ImageLogger
from robot_properties_kuka.config import IiwaConfig
import pinocchio as pin
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.transforms import ToTensor
from torchvision import transforms
from torchvision.transforms import ToTensor, ToPILImage, Resize
from matplotlib import pyplot as plt
from skimage.io import imread

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
        self.subp = Process(target=ImageLogger, args=(["color_image", "depth_image", "position"], "data15", 10.0, self.child_conn))
        self.subp.start()
        self.init_pos = self.joint_positions.copy()

        pass

    def predict_cnn(self):
        with torch.no_grad():
        
            c_image = ToTensor()((imread("image_data/" + "color_" + str(1) + ".jpg")))
            d_image = ToTensor()((imread("image_data/" + "depth_" + str(1) + ".jpg")))
            image = torch.vstack((c_image, d_image))
            image = transforms.functional.crop(image, 50, 100, 180, 180)
            image = image[None,:,:,:]
            # pred = self.cnet(image)
            self.image = image
        return None
        # return (pred*self.std + self.mean).numpy()[0]

    def run(self, thread):
        t1 = time.time()
        q = self.joint_positions
        v = self.joint_velocities
        pos, vel = thread.vicon.get_state('cube10/cube10')
        self.color_image, self.depth_image = thread.camera.get_image()
        cv2.imwrite("./image_data/" + "color_" + str(1) + ".jpg", self.color_image)
        cv2.imwrite("./image_data/" + "depth_" + str(1) + ".jpg", self.depth_image)
        pred = self.predict_cnn()
        # print(pred, pos[0:3], np.linalg.norm(pred - pos[0:3]))

        self.data["color_image"] = self.color_image
        self.data["depth_image"] = self.depth_image
        self.data["position"] = pos[0:3]
        self.parent_conn.send((self.data, thread.ti))
        # box = ToPILImage()(self.image[0][0:3])
        # opencvImage = cv2.cvtColor(np.array(box), cv2.COLOR_RGB2BGR)

        # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('RealSense', opencvImage)
        # cv2.waitKey(1)

        ti = thread.ti
        self.Kp = np.array([10.0, 10.0, 3.0, 1.0, 1.0, 1.0, 1.0])
        self.Kd = np.array([2.5, 2.5, 2.5, 2.5, 2.5, 0.5, 0.5])

        self.des_position = self.init_pos.copy()   
        self.des_position[0] = 0.3*np.sin(4.5*np.pi*thread.ti/1000.) + np.pi/18
        self.des_position[2] = 0.35*np.sin(5.5*np.pi*thread.ti/1000. + np.pi/5.0) + self.init_pos[2]  
        self.des_position[3] = 0.45*np.cos(4.5*np.pi*thread.ti/1000. + np.pi/3.0) + 1.6*self.init_pos[3]  
        self.des_position[4] = 0.2*np.sin(3.5*np.pi*thread.ti/1000. + np.pi/7.0) + self.init_pos[4]  
        self.tau = (
            self.Kp * (self.des_position - self.joint_positions)
            - self.Kd * self.joint_velocities
        )

        self.tau = np.clip(self.tau, -15, 15)

        pin.forwardKinematics(self.pinModel, self.pinData, q, v, np.zeros_like(q))
        pin.updateFramePlacements(self.pinModel, self.pinData)
        
        self.g_comp = pin.rnea(self.pinModel, self.pinData, q, v, np.zeros_like(q))
        self.head.set_control('ctrl_joint_torques', self.tau)
        self.head.set_control('desired_joint_positions', self.des_position)

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

