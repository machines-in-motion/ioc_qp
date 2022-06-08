
import torch
import torch.nn as nn

from vocam.inverse_qp import IOC

import pinocchio as pin
import numpy as np
import meshcat.transformations as tf
import meshcat.geometry as g

from tqdm import trange
import time

class DataUtils(object):
    def __init__(self, robot, task_loss, config, viz=None):
        self.robot = robot
        self.model, self.data = robot.model, robot.data
        self.f_id = self.model.getFrameId("EE")
        self.config = config
        self.viz = viz
        self.task_loss = task_loss

    def save(self, path):
        self.data_train = list(zip(self.x_train, self.y_train))
        torch.save(self.data_train, path)

    def load(self, path):
        self.data_train = torch.load(path)
        unzipped = list(zip(*self.data_train))
        self.x_train = torch.vstack([*unzipped[0]])
        self.y_train = torch.vstack([*unzipped[1]])
    
    def visualize(self, task_idx, task_horizon=None):
        if task_horizon is None:
            task_horizon = self.config.task_horizon
        start = task_idx * task_horizon
        end = start + task_horizon
        q = self.x_train[start:end, :self.config.nq].detach().cpu().numpy()
        goal = self.x_train[start, -3:].detach().cpu().numpy()
        if self.viz is not None:
            self.viz.display(q[0])
            self.viz.viewer["box"].set_object(g.Sphere(0.05), 
                                              g.MeshLambertMaterial(
                                              color=0xff22dd,
                                              reflectivity=0.8))
            self.viz.viewer["box"].set_transform(tf.translation_matrix(goal))
            time.sleep(0.5) 
            for n in range(len(q)):
                self.viz.display(q[n])
                time.sleep(0.05)

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

    def load(self, path, device='cpu'):
        self.load_state_dict(torch.load(path, map_location=torch.device(device)))


class QPNetObstacle(QPNet):
    def __init__(self, input_size, output_size, obstacle_size):
        super().__init__(input_size, output_size)
        self.fc1_obs = nn.Linear(obstacle_size, 512)
        self.bn1_obs = nn.BatchNorm1d(512)
        self.fc2_obs = nn.Linear(512, 512)
        self.bn2_obs = nn.BatchNorm1d(512)

        self.fc1 = nn.Linear(input_size + 512, 512)
        self.bn1 = nn.BatchNorm1d(512)

        self.input_size = input_size
        self.obstacle_size = obstacle_size

    def forward(self, x):
        if len(x.shape) == 1:
            x = x[None, :]
        obs = x[:, self.input_size:]
        obs = self.swish(self.bn1_obs(self.fc1_obs(obs)))
        obs = self.dropout(self.swish(self.bn2_obs(self.fc2_obs(obs))))

        x = torch.hstack((x[:, :self.input_size], obs))
        x = self.swish(self.bn1(self.fc1(x)))
        x = self.dropout(self.swish(self.bn2(self.fc2(x)))) 
        x = self.swish(self.bn3(self.fc3(x)))
        x = self.out(x)

        return x

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

    def load(self, path, device='cpu'):
        self.load_state_dict(torch.load(path, map_location=torch.device(device)))


class QPNetObstacle(QPNet):
    def __init__(self, input_size, output_size, obstacle_size):
        super().__init__(input_size, output_size)
        self.fc1_obs = nn.Linear(obstacle_size, 512)
        self.bn1_obs = nn.BatchNorm1d(512)
        self.fc2_obs = nn.Linear(512, 512)
        self.bn2_obs = nn.BatchNorm1d(512)

        self.fc1 = nn.Linear(input_size + 512, 512)
        self.bn1 = nn.BatchNorm1d(512)

        self.input_size = input_size
        self.obstacle_size = obstacle_size

    def forward(self, x):
        if len(x.shape) == 1:
            x = x[None, :]
        obs = x[:, self.input_size:]
        obs = self.swish(self.bn1_obs(self.fc1_obs(obs)))
        obs = self.dropout(self.swish(self.bn2_obs(self.fc2_obs(obs))))

        x = torch.hstack((x[:, :self.input_size], obs))
        x = self.swish(self.bn1(self.fc1(x)))
        x = self.dropout(self.swish(self.bn2(self.fc2(x)))) 
        x = self.swish(self.bn3(self.fc3(x)))
        x = self.out(x)

        return x
