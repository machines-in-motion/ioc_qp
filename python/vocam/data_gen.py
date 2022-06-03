
import torch
import torch.nn as nn

from vocam.inverse_qp import IOC

import pinocchio as pin
import numpy as np
import meshcat.transformations as tf
import meshcat.geometry as g

from tqdm import trange

def generate_obstacle(n_tasks, task_loss, robot, config, viz=None):
    n_col, nq, nv = config.n_col, config.nq, config.nv
    u_max, dt = config.u_max, config.dt
    n_obs = config.n_obs
    X = []
    Y = []

    for i in trange(n_tasks):
        # generate random goal location
        r = 0.3*np.random.rand(1) + 0.5
        if np.random.randint(2) == 0:
            theta = 0.1*np.pi*(np.random.rand(1)) + 0.25*np.pi
        else:
            theta = 0.1*np.pi*(np.random.rand(1)) + 0.65*np.pi
    
        goal = torch.squeeze(torch.tensor([r*np.sin(theta), r*np.cos(theta), 0.15*np.random.rand(1)+0.15]))

        # generate obstacles (for now only one fixed obstacle)
        obs_rad = config.obs_rad
        obs_pos_arr = []
        for l in range(n_obs):
            r = 0.0*np.random.rand(1) + 0.5
            theta = 0.0*np.pi*(np.random.rand(1) - 0.5) + 0.5*np.pi
            ht = 0.0*np.random.rand(1) 
            obs_pos = torch.squeeze(torch.tensor([r*np.sin(theta), r*np.cos(theta), ht]))
            obs_pos_arr.append(obs_pos)
        
        # generate random robot configuration
        if i == 0 or np.random.randint(config.n_restart) == 0:
            x_init = np.array(config.x_init)
            x_init[:nq] += config.q_noise * (np.random.rand(nv) - 0.5)
            x_init[nq:] = config.dq_noise * (np.random.rand(nv) - 0.5)
            x_init[0] -= 2*0.5*(np.random.rand(1) - 0.5)
            x_init[2] -= 2*0.3*(np.random.rand(1) - 0.5)
        
        # visualization
        if viz is not None:
            viz.display(x_init[:nq])
            viz.viewer["box"].set_object(g.Sphere(0.05), 
                            g.MeshLambertMaterial(
                            color=0xff22dd,
                            reflectivity=0.8))
            viz.viewer["box"].set_transform(tf.translation_matrix(goal.detach().numpy()))
            for obs_pos in obs_pos_arr:                
                viz.viewer["obstacle_" + str(l)].set_object(g.Ellipsoid(obs_rad), 
                                                            g.MeshLambertMaterial(
                                                            color=0x22ddff,
                                                            reflectivity=0.8))
                viz.viewer["obstacle_" + str(l)].set_transform(tf.translation_matrix(obs_pos.detach().numpy()))
            
        # allocate memory for the data from the i-th task
        Xi = torch.zeros((config.task_horizon, len(config.x_init) + 3))
        if not config.isvec:
            Yi = torch.zeros((config.task_horizon,
                            config.n_vars**2 + config.n_vars))
        else:
            Yi = torch.zeros((config.task_horizon, 
                            2*config.n_vars))
        # MPC loop
        for j in range(config.task_horizon):
            ioc = IOC(n_col, nq, u_max, dt, eps=1.0, isvec=config.isvec)
            optimizer = torch.optim.Adam(ioc.parameters(), 
                                            lr=config.lr_qp)
            
            if j >= 1:
                x_init = x_pred[-2 * nq:]

            old_loss = torch.inf
            for _ in range(config.max_it):
                x_pred = ioc(x_init) 
                loss = task_loss(x_pred, goal, obs_pos_arr, config)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (loss < config.loss_threshold or
                    abs(old_loss - loss) < config.convergence_threshold):
                    break
                else:
                    old_loss = loss.detach().clone()
            
            x_pred = ioc(x_init).detach().numpy()

            if viz is not None:
                for n in range(n_col + 1):
                    q = x_pred[3*nq*n:3*nq*n + nq]
                    dq = x_pred[3*nq*n+nq:3*nq*n+2*nq]
                    viz.display(q)

            # storing the weights and x_nom
            Xi[j] = torch.hstack((torch.tensor(x_init), goal)).detach().float()
            Yi[j] = torch.hstack((ioc.weight.detach().clone().flatten(), 
                                    ioc.x_nom.detach().clone()))

        q = x_pred[-2*nq:-nq]
        dq = x_pred[-nq:]
        pin.forwardKinematics(robot.model, robot.data, q, dq, np.zeros(nv))
        pin.updateFramePlacements(robot.model, robot.data)
        dist = np.linalg.norm(robot.data.oMf[config.f_id].translation - goal.detach().numpy())
        
        if dist <= config.distance_threshold:
            # only store successful task executions
            X.append(Xi)
            Y.append(Yi)
        else:
            print(dist)

    x_train = torch.vstack(X)
    y_train = torch.vstack(Y)

    return x_train, y_train