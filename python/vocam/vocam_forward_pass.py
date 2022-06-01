## Contains the class that implementes the online adaptaion of weights on the real robot
## Author : Avadesh Meduri
## Date : 3/05/2022 

from csv import excel_tab
import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.transforms import ToTensor
from torchvision import transforms
import cv2
from multiprocessing import Process, Pipe
from skimage.io import imread

from vocam.diff_qp import DiffQP
from vocam.nets import C_Net, Net, C_Net_encoder
from vocam.inverse_qp import IOC
from vocam.qpnet import QPNet

try:
    from dynamic_graph_head import ImageLogger, VisionSensor
except:
    data_coll = False

class IOCForwardPass:

    def __init__(self, nn_dir, m, std, collect_data = False, vision_based = True):
        """
        Input:
            nn_dir : directory for NN weights
            ioc : ioc QP
            m : mean of the trained data (y_train)
            std : standard deviation of trained data (y_train)
            collect_data : collects data
            vison_based : uses cnn to predict cube location
        """

        self.nq = 7
        self.n_col = 5
        self.state = np.zeros(2*self.nq)

        u_max = [2.5,2.5,2.5, 1.5, 1.5, 1.5, 1.0]
        self.dt = 0.05

        self.ioc = IOC(self.n_col, self.nq, u_max, self.dt, eps = 1.0, isvec=True)
        self.mean = m
        self.std = std
        self.n_vars = self.ioc.n_vars
        try:
            self.nn = Net(2*self.nq + 3, 2*self.n_vars)
            self.nn.load_state_dict(torch.load(nn_dir))
            self.no_norm = False
            print("using nn")
        except:
            print("using qp net")
            self.nn = QPNet(2*self.nq + 3, 2*self.n_vars).eval()
            self.nn.load(nn_dir)
            self.no_norm = True

        self.vision_based = vision_based
        self.end_to_end = True
        if self.vision_based:
            self.cnet = C_Net_encoder()
            self.cnet.load_state_dict(torch.load("/home/ameduri/pydevel/ioc_qp/vision/models/cnn2", map_location=torch.device('cpu')))
            self.c_mean = np.array([0.3068, 0.1732, 0.4015])
            self.c_std = np.array([0.1402, 0.2325, 0.1624])

            self.camera = VisionSensor()
            self.camera.update(None)
        
        if self.end_to_end:
            nn_dir = "../models/e2eNet1"
            self.encoder_net= QPNet(2*self.nq + 512, 2*self.n_vars).eval()
            self.encoder_net.load(nn_dir)

        self.collect_data = collect_data
        if self.collect_data:
            self.data = {"color_image": [], "depth_image": [], "position": []}
            self.img_par, self.img_child = Pipe()
            self.subp = Process(target=ImageLogger, args=(["color_image", "depth_image", "position"], "data21", 1.5, self.img_child))
            self.subp.start()
            self.ctr = 0

    def predict(self, q, dq, x_des):

        nq = self.ioc.nq
        n_vars = self.ioc.n_vars
        state = np.zeros(2*nq)
        state[0:nq] = q
        state[nq:] = dq
        x_input = torch.hstack((torch.tensor(state), torch.tensor(x_des))).float()
        if not self.no_norm:
            pred_norm = self.nn(x_input)
            pred = pred_norm
            # print(pred.shape, self.std.shape, self.m.shape)
            # pred = pred_norm * self.std + self.mean
        else:
            pred_norm = torch.squeeze(self.nn(x_input))
            pred = pred_norm
        # # if not self.ioc.isvec:
        #     self.ioc.weight = torch.nn.Parameter(torch.reshape(pred[0:n_vars**2], (n_vars, n_vars)))
        #     self.ioc.x_nom = torch.nn.Parameter(pred[n_vars**2:])
        # else:
        self.ioc.weight = torch.nn.Parameter(pred[0:n_vars])
        self.ioc.x_nom = torch.nn.Parameter(pred[n_vars:])

        x_pred = self.ioc(state) 
        x_pred = x_pred.detach().numpy()

        return x_pred

    def predict_cnn(self):
        with torch.no_grad():
            c_image = ToTensor()((imread("." + "/color_" + str(1) + ".jpg")))
            d_image = ToTensor()((imread("." + "/depth_" + str(1) + ".jpg")))
            image = torch.vstack((c_image, d_image))
            image = transforms.functional.crop(image, 50, 100, 180, 180)

            image = image[None,:,:,:]
            pred, enc = self.cnet(image)

        return (pred*self.c_std + self.c_mean).numpy()[0], enc

    def predict_encoder(self, q, dq, encoding):
        
        nq = self.ioc.nq
        n_vars = self.ioc.n_vars
        state = np.zeros(2*nq)
        state[0:nq] = q
        state[nq:] = dq
        
        x_in = torch.hstack((torch.tensor(state)[None,], encoding)).float()
        pred = torch.squeeze(self.encoder_net(x_in))
        n_vars = self.ioc.n_vars
        self.ioc.weight = torch.nn.Parameter(pred[0:n_vars])
        self.ioc.x_nom = torch.nn.Parameter(pred[n_vars:])

        x_pred = self.ioc(state) 
        x_pred = x_pred.detach().numpy()

        return x_pred

    def predict_rt(self, child_conn):
        while True:
            q, dq, x_des = child_conn.recv()

            # if self.collect_data:
            #     self.color_image, self.depth_image = self.camera.get_image()
            #     self.data["color_image"] = self.color_image
            #     self.data["depth_image"] = self.depth_image
            #     self.data["position"] = x_des
            #     self.img_par.send((self.data, self.ctr))
            #     self.ctr += 1

            if self.vision_based:
                self.color_image, self.depth_image = self.camera.get_image()
                cv2.imwrite("." + "/color_" + str(1) + ".jpg", self.color_image)
                cv2.imwrite("." + "/depth_" + str(1) + ".jpg", self.depth_image)

                pred, enc = self.predict_cnn()
                pred[0] += 0.3
                # print(pred, x_des, np.linalg.norm(pred - x_des))
                x_des = pred

            if self.end_to_end and self.vision_based:
                x_pred = self.predict_encoder(q, dq, enc)
                
            else:
                t1 = time.time()
                x_pred = self.predict(q, dq, x_des)
                t2 = time.time()
            
            child_conn.send((x_pred))
            # print("compute time", t2 - t1)

def rt_IOCForwardPass(channel, nn_dir, mean, std):
    planner = IOCForwardPass(nn_dir, mean, std)
    planner.predict_rt(channel)
