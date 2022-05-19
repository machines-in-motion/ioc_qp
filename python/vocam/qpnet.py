
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

    def generate(self, n_tasks):
        n_col, nq, nv = self.config.n_col, self.config.nq, self.config.nv
        u_max, dt = self.config.u_max, self.config.dt
        X = []
        Y = []

        restart = True
        for i in trange(n_tasks):
            x_init = np.array(self.config.x_init)
            x_init[:nq] += self.config.q_noise * (np.random.rand(nv) - 0.5)
            x_init[nq:] = self.config.dq_noise * (np.random.rand(nv) - 0.5)

            if restart:
                # default goal position
                rand_idx = np.random.randint(len(self.config.default_goals))
                goal = torch.tensor(self.config.default_goals[rand_idx])
                goal += self.config.goal_noise * (torch.rand(3) - 0.5)
            else:
                goal = self.sample_next_location(goal.detach())
                x_init[:nq] = x_pred[-2*nq:-nq]
                
            if self.viz is not None:
                self.viz.display(x_init[:nq])
                self.viz.viewer["box"].set_object(g.Sphere(0.05), 
                                g.MeshLambertMaterial(
                                color=0xff22dd,
                                reflectivity=0.8))
                self.viz.viewer["box"].set_transform(tf.translation_matrix(goal.detach().numpy()))                

            
            # allocate memory for the data from the i-th task
            Xi = torch.zeros((self.config.task_horizon, 
                              len(self.config.x_init) + 3))
            if not self.config.isvec:
                Yi = torch.zeros((self.config.task_horizon,
                              self.config.n_vars**2 + self.config.n_vars))
            else:
                Yi = torch.zeros((self.config.task_horizon, 
                              2*self.config.n_vars))
            # MPC loop
            for j in range(self.config.task_horizon):
                ioc = IOC(n_col, nq, u_max, dt, eps=1.0, isvec=self.config.isvec)
                optimizer = torch.optim.Adam(ioc.parameters(), 
                                             lr=self.config.lr_qp)
                
                if j >= 1:
                    x_init = x_pred[-2 * nq:]

                old_loss = torch.inf
                for _ in range(self.config.max_it):
                    x_pred = ioc(x_init) 
                    loss = self.task_loss(self.robot, x_pred, goal, nq, n_col)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if (loss < self.config.loss_threshold or
                        abs(old_loss - loss) < self.config.convergence_threshold):
                        break
                    else:
                        old_loss = loss.detach().clone()
                
                x_pred = ioc(x_init).detach().numpy()

                if self.viz is not None:
                    for n in range(n_col + 1):
                        q = x_pred[3*nq*n:3*nq*n + nq]
                        dq = x_pred[3*nq*n+nq:3*nq*n+2*nq]
                        self.viz.display(q)

                # storing the weights and x_nom
                Xi[j] = torch.hstack((torch.tensor(x_init), goal)).detach().float()
                Yi[j] = torch.hstack((ioc.weight.detach().clone().flatten(), 
                                      ioc.x_nom.detach().clone()))
    
            q = x_pred[-2*nq:-nq]
            dq = x_pred[-nq:]
            pin.forwardKinematics(self.model, self.data, q, dq, np.zeros(nv))
            pin.updateFramePlacements(self.model, self.data)
            dist = np.linalg.norm(self.data.oMf[self.f_id].translation - goal.detach().numpy())
            
            if dist <= self.config.distance_threshold:
                # only store successful task executions
                X.append(Xi)
                Y.append(Yi)
                if (i + 1) % self.config.n_restart == 0:
                    restart = True
                else:
                    restart = False
            else:
                print(dist)
                restart = True

        self.x_train = torch.vstack(X)
        self.y_train = torch.vstack(Y)

    def sample_next_location(self, curr_location):
        lb = torch.tensor(self.config.lb)
        ub = torch.tensor(self.config.ub)
        diff_range = self.config.diff_range
        dist_ub = self.config.dist_ub
        dist_lb = self.config.dist_lb

        while True:
            diff = diff_range * (torch.rand(3) - 0.5)
            next_location = curr_location + diff
            flipped_sign_x = (next_location[0] * curr_location[0] < 0)
            flipped_sign_y = (next_location[1] * curr_location[1] < 0)
            if (all(next_location >= lb) 
                and all(next_location <= ub)
                and torch.linalg.norm(diff) >= diff_range / 4
                and torch.linalg.norm(diff) <= diff_range / 2
                and torch.linalg.norm(next_location[:2]) >= dist_lb
                and torch.linalg.norm(next_location[:2]) <= dist_ub
                and (not flipped_sign_x or not flipped_sign_y)): 
                break
        return next_location

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

    def generate2(self, n_tasks):
        n_col, nq, nv = self.config.n_col, self.config.nq, self.config.nv
        u_max, dt = self.config.u_max, self.config.dt
        X = []
        Y = []

        for i in trange(n_tasks):
            # generate random goal location
            r = self.config.r[0] + self.config.r[1]*np.random.rand(1)
            z = self.config.z[0] + self.config.z[1]*np.random.randint(1)
            theta = self.config.theta[0] + self.config.theta[1]*np.random.rand(1)
            goal = torch.squeeze(torch.tensor([r*np.sin(theta), 
                                               r*np.cos(theta), 
                                               z]))
            
            # generate random robot configuration
            if i == 0 or np.random.randint(self.config.n_restart) == 0:
                x_init = np.array(self.config.x_init)
                x_init[:nq] += self.config.q_noise * (np.random.rand(nv) - 0.5)
                x_init[nq:] = self.config.dq_noise * (np.random.rand(nv) - 0.5)
                x_init[0] -= 2*0.5*(np.random.rand(1) - 0.5)
                x_init[2] -= 2*0.3*(np.random.rand(1) - 0.5)
            
            # visualization
            if self.viz is not None:
                self.viz.display(x_init[:nq])
                self.viz.viewer["box"].set_object(g.Sphere(0.05), 
                                g.MeshLambertMaterial(
                                color=0xff22dd,
                                reflectivity=0.8))
                self.viz.viewer["box"].set_transform(tf.translation_matrix(goal.detach().numpy()))                

            
            # allocate memory for the data from the i-th task
            Xi = torch.zeros((self.config.task_horizon, 
                              len(self.config.x_init) + 3))
            if not self.config.isvec:
                Yi = torch.zeros((self.config.task_horizon,
                              self.config.n_vars**2 + self.config.n_vars))
            else:
                Yi = torch.zeros((self.config.task_horizon, 
                              2*self.config.n_vars))
            # MPC loop
            for j in range(self.config.task_horizon):
                ioc = IOC(n_col, nq, u_max, dt, eps=1.0, isvec=self.config.isvec)
                optimizer = torch.optim.Adam(ioc.parameters(), 
                                             lr=self.config.lr_qp)
                
                if j >= 1:
                    x_init = x_pred[-2 * nq:]

                old_loss = torch.inf
                for _ in range(self.config.max_it):
                    x_pred = ioc(x_init) 
                    loss = self.task_loss(self.robot, x_pred, goal, nq, n_col)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if (loss < self.config.loss_threshold or
                        abs(old_loss - loss) < self.config.convergence_threshold):
                        break
                    else:
                        old_loss = loss.detach().clone()
                
                x_pred = ioc(x_init).detach().numpy()

                if self.viz is not None:
                    for n in range(n_col + 1):
                        q = x_pred[3*nq*n:3*nq*n + nq]
                        dq = x_pred[3*nq*n+nq:3*nq*n+2*nq]
                        self.viz.display(q)

                # storing the weights and x_nom
                Xi[j] = torch.hstack((torch.tensor(x_init), goal)).detach().float()
                Yi[j] = torch.hstack((ioc.weight.detach().clone().flatten(), 
                                      ioc.x_nom.detach().clone()))
    
            q = x_pred[-2*nq:-nq]
            dq = x_pred[-nq:]
            pin.forwardKinematics(self.model, self.data, q, dq, np.zeros(nv))
            pin.updateFramePlacements(self.model, self.data)
            dist = np.linalg.norm(self.data.oMf[self.f_id].translation - goal.detach().numpy())
            
            if dist <= self.config.distance_threshold:
                # only store successful task executions
                X.append(Xi)
                Y.append(Yi)
            else:
                print(dist)

        self.x_train = torch.vstack(X)
        self.y_train = torch.vstack(Y)

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

    