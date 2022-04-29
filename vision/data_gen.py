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

class C_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv11 = nn.Conv2d(4, 64, 3)
        self.conv12 = nn.Conv2d(64, 64, 3)

        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv21 = nn.Conv2d(64, 128, 3)
        self.conv22 = nn.Conv2d(128, 128, 3)

        self.conv31 = nn.Conv2d(128, 256, 3)
        self.conv32 = nn.Conv2d(256, 256, 3)
        
        self.conv41 = nn.Conv2d(256, 512, 3)
        self.conv42 = nn.Conv2d(512, 512, 3)
        
        self.fc1 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 3)
#         self.fc3 = nn.Linear(128, 3)

    def forward(self, x):
        x = F.relu(self.conv11(x))
        x = self.pool(F.relu(self.conv12(x)))
        x = F.relu(self.conv21(x))
        x = self.pool(F.relu(self.conv22(x)))
        
        x = self.pool(F.relu(self.conv31(x)))
        x = self.pool(F.relu(self.conv32(x)))
        
        x = F.relu(self.conv41(x))
        x = self.pool(F.relu(self.conv42(x)))
            
        x = torch.flatten(x, 1) # flatten all dimensions except batch
#         print(x.shape)
        x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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
        
        self.cnet = C_Net()
        self.cnet.load_state_dict(torch.load("./models/cnn3"))
        self.mean = np.array([0.3186, 0.0425, 0.2636])
        self.std = np.array([0.1430, 0.1926, 0.1375])


    def warmup(self, thread_head):
        # self.subp = Process(target=ImageLogger, args=(["color_image", "depth_image", "position"], "data15", 0.4, self.child_conn))
        # self.subp.start()
        self.init = self.joint_positions.copy()

        pass

    def predict_cnn(self):
        with torch.no_grad():
        
            c_image = ToTensor()((imread("image_data/" + "/color_" + str(1) + ".jpg")))
            d_image = ToTensor()((imread("image_data/" + "/depth_" + str(1) + ".jpg")))
            image = torch.vstack((c_image, d_image))
            image = transforms.functional.crop(image, 0, 100, 150, 150)
            image = image[None,:,:,:]
            pred = self.cnet(image)
            self.image = image
        
        return (pred*self.std + self.mean).numpy()[0]

    def run(self, thread):
        t1 = time.time()
        q = self.joint_positions
        v = self.joint_velocities
        pos, vel = thread.vicon.get_state('cube10/cube10')
        self.color_image, self.depth_image = thread.camera.get_image()
        cv2.imwrite("image_data/" + "/color_" + str(1) + ".jpg", self.color_image)
        cv2.imwrite("image_data/" + "/depth_" + str(1) + ".jpg", self.depth_image)

        pred = self.predict_cnn(self.color_image, self.depth_image)
        print(pred, pos[0:3], np.linalg.norm(pred - pos[0:3]))
        
        # self.data["color_image"] = self.color_image
        # self.data["depth_image"] = self.depth_image
        # self.data["position"] = pos[0:3]
        # self.parent_conn.send((self.data, thread.ti))

        box = ToPILImage()(self.image[0][0:3])
        opencvImage = cv2.cvtColor(np.array(box), cv2.COLOR_RGB2BGR)

        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', opencvImage)
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

