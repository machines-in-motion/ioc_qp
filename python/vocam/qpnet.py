
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import functional as F
from torch.utils.data import DataLoader

from vocam.diff_pin_costs import DiffFrameTranslationCost, DiffFrameVelocityCost
from vocam.inverse_qp import IOC

import pinocchio as pin
import numpy as np
import meshcat.transformations as tf
import meshcat.geometry as g

from tqdm import trange
import time

class DataUtils(object):
    def __init__(self, robot, config, viz=None):
        self.robot = robot
        self.model, self.data = robot.model, robot.data
        self.f_id = self.model.getFrameId("EE")
        self.config = config
        self.viz = viz        


    def generate(self, buffer_size):
        self.x_train_init = torch.zeros((buffer_size, len(self.config.x_init)))
        self.x_train_des = torch.zeros((buffer_size, 3))

        if not self.config.isvec:
            self.y_train = torch.zeros((buffer_size, 
                                        self.config.n_vars**2 + self.config.n_vars))
        else:
            self.y_train = torch.zeros((buffer_size, 
                                        2*self.config.n_vars))

        all_x_des = []
        n_col, nq, nv = self.config.n_col, self.config.nq, self.config.nv
        u_max, dt = self.config.u_max, self.config.dt

        for k in trange(buffer_size):
            ioc = IOC(n_col, nq, u_max, dt, eps=1.0, isvec=self.config.isvec)
            optimizer = torch.optim.Adam(ioc.parameters(), lr=self.config.lr_qp)

            if k % 32 == 0:
                x_init = np.zeros(2 * nq)
                if k < 1:
                    x_des = torch.tensor([0.6, 0.4, 0.7])
                    x_init[:nq] = np.array([0.0, 0.3, 0.0, -0.8, -0.6,  0.0, 0.0])
                else:
                    x_des = self.sample_next_location(x_des.detach())
                    x_init[:nq] = x_pred[-2*nq:-nq] + 0.4 * (np.random.rand(nq) - 0.5)
            
                x_init[nq:] = 0.8 * (np.random.rand(nv) - 0.5) 
                all_x_des.append(x_des.detach().clone())
                    
                if self.viz is not None:
                    self.viz.display(x_init[:nq])
                    self.viz.viewer["box"].set_object(g.Sphere(0.05), 
                                    g.MeshLambertMaterial(
                                    color=0xff22dd,
                                    reflectivity=0.8))
                    self.viz.viewer["box"].set_transform(tf.translation_matrix(x_des.detach().numpy()))
            else:
                x_init = x_pred[-2*nq:]

            self.x_train_init[k] = torch.tensor(x_init)
            self.x_train_des[k] = x_des
            
            i = 0
            loss = 1000.
            old_loss = 10000.
            
            while loss > 0.03 and i < self.config.max_eps and abs(old_loss - loss) > 5e-4:
                x_pred = ioc(x_init) 
                old_loss = loss
                loss = self.task_loss(x_pred, x_des, nq, n_col)
        #         print("Index :" + str(k) + "/" + str(buffer_size) + " Iteration :" + str(i) + "/" + str(max_eps) +  " loss is : " + str(loss.detach().numpy()), end = '\r', flush = True)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                i += 1
            
            x_pred = ioc(x_init).detach().numpy()
            if self.viz is not None:
                for i in range(n_col+1):
                    q = x_pred[3*nq*i:3*nq*i + nq]
                    dq = x_pred[3*nq*i + nq:3*nq*i + 2*nq]

                    pin.forwardKinematics(self.model, self.data, q, dq, np.zeros(nv))
                    pin.updateFramePlacements(self.model, self.data)
                    self.viz.display(q)
                    time.sleep(0.01)
            # storing the weights and x_nom
            self.y_train[k] = torch.hstack((ioc.weight.flatten(), ioc.x_nom))

        self.x_train = torch.hstack((self.x_train_init, self.x_train_des)).float()
        self.y_train = self.y_train.detach().float()

    def sample_next_location(self, curr_location):
        lb = torch.tensor([0.4,  0.4, 0.0])
        ub = torch.tensor([1.0,  1.0, 1.1])
        diff_range = 1.6
        dist_ub = 0.8
        dist_lb = 0.4

        while True:
            diff = diff_range * (torch.rand(3) - 0.5)
            next_location = curr_location + diff
            if (all(next_location >= lb) 
                and all(next_location <= ub) 
                and torch.linalg.norm(next_location) >= dist_lb
                and torch.linalg.norm(next_location) <= dist_ub): 
                break
        return next_location

    def task_loss(self, q_pred, x_des, nq, n_col):
        dtc = DiffFrameTranslationCost.apply
        dvc = DiffFrameVelocityCost.apply
        model, data, f_id = self.model, self.data, self.f_id

        loss = 3.5e1*torch.linalg.norm(dtc(q_pred[-2*nq:], model, data, f_id) - x_des)
        loss += 1.5e0*torch.linalg.norm(dvc(q_pred[-2*nq:], torch.zeros(nq), model, data, f_id)) # asking for zero velocity
        loss += 5e-3*torch.linalg.norm(q_pred[-2*nq:-nq]) # joint regularization
        
        for i in range(n_col):    
            loss += 1e0*torch.linalg.norm(dtc(q_pred[(3*i)*nq: (3*i+2)*nq], model, data, f_id) - x_des)
            loss += 5e-1*torch.linalg.norm(dvc(q_pred[(3*i)*nq: (3*i+2)*nq], q_pred[(3*i+2)*nq:(3*i+3)*nq], model, data, f_id)) # asking for zero velocity
            loss += 1e-2*torch.linalg.norm(q_pred[(3*i+2)*nq: (3*i+3)*nq]) # control regularization
            loss += 2e-1*torch.linalg.norm(q_pred[(3*i+1)*nq: (3*i+2)*nq]) # velocity regularization
            loss += 3e-3*torch.linalg.norm(q_pred[(3*i)*nq: (3*i+1)*nq]) # joint regularization
            
            if i < n_col - 1:
                loss += 5e-2*torch.linalg.norm(torch.subtract(q_pred[(3*i+2)*nq: (3*i+3)*nq], \
                                                              q_pred[(3*i+5)*nq: (3*i+6)*nq]))
            if i == n_col - 1:
                # terminal joint velocity regularization
                loss += 2e-1*torch.linalg.norm(q_pred[(3*i+1)*nq: (3*i+2)*nq]) 

        return loss

    def save(self, path):
        self.data_train = list(zip(self.x_train, self.y_train))
        torch.save(self.data_train, path)

    def load(self, path):
        self.data_train = torch.load(path)
        unzipped = list(zip(*self.data_train))
        self.x_train = torch.vstack([*unzipped[0]])
        self.y_train = torch.vstack([*unzipped[1]])

class QPNet(nn.Module):
    def __init__(self, input_size, output_size):
        # game params
        super().__init__()
        
        self.swish = nn.SiLU()
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 1024)
        self.bn2 = nn.BatchNorm1d(1024)

        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)

        self.dropout = nn.Dropout(p=0.5)

        self.out = nn.Linear(512, output_size)
        
    def forward(self, x):
        if len(x.shape) == 1:
            x = x[None, :]
        x = self.swish(self.bn1(self.fc1(x)))
        x = self.dropout(self.swish(self.bn2(self.fc2(x)))) 
        x = self.swish(self.bn3(self.fc3(x)))
        x = self.out(x)

        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

def train(network, criterion, optimizer, dataloader, device):
    network.train()
    all_loss = []
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_pred = network(x)
        loss = criterion(y_pred, y)
        all_loss.append(loss.cpu().detach().numpy())
        loss.backward()
        optimizer.step()

    return np.mean(all_loss)

def test(network, criterion, dataloader, device):
    network.eval()
    all_loss = []
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        y_pred = network(x)
        loss = criterion(y_pred, y)
        all_loss.append(loss.cpu().detach().numpy())

    return np.mean(all_loss)

    