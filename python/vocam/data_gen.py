
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
    restart = True

    for i in trange(n_tasks):
        # generate random goal location
        if False:
            r = config.r_range[0] + (config.r_range[1]-config.r_range[0])*np.random.rand()
            z = config.z_range[0] + (config.z_range[1]-config.z_range[0])*np.random.rand()
            th = config.th_range[0] + (config.th_range[1]-config.th_range[0])*np.random.rand()
            goal = torch.tensor([r*np.sin(th), r*np.cos(th), z])

        else:
            r = config.r_range[0] + (config.r_range[1]-config.r_range[0])*np.random.rand()
            if np.random.randint(2) == 0:
                th = 0.1*np.pi*(np.random.rand()) + 0.25*np.pi
            else:
                th = 0.1*np.pi*(np.random.rand()) + 0.65*np.pi
        
            goal = torch.tensor([r*np.sin(th), r*np.cos(th), 0.15*np.random.rand()+0.15])
            
        
        if restart or np.random.randint(config.n_restart) == 0:
            # generate random robot configuration
            x_init = np.array(config.x_init)
            x_init[:nq] += config.q_noise * (np.random.rand(nv) - 0.5)
            x_init[nq:] = config.dq_noise * (np.random.rand(nv) - 0.5)
            x_init[0] -= 2*0.5*(np.random.rand() - 0.5)
            x_init[2] -= 2*0.3*(np.random.rand() - 0.5)

            # generate obstacles (for now only one fixed obstacle)
            obs_rad = np.array(config.obs_rad)
            obs_height = np.random.choice([0.55, 0.55, 0.55, 0.55]) + np.random.rand()*0.05
            obs_rad[2] = obs_height

            obs_pos_arr = []
            for l in range(n_obs):
                r = 0.0*np.random.rand() + 0.5
                th = 0.0*np.pi*(np.random.rand() - 0.5) + 0.5*np.pi
                ht = 0.0*np.random.rand() 
                obs_pos = torch.tensor([r*np.sin(th), r*np.cos(th), ht])
                obs_pos_arr.append(obs_pos)
        
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
        Xi = torch.zeros((config.task_horizon, len(config.x_init) + 6))
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

            q = x_init[:nq]
            dq = x_init[nq:]
            pin.forwardKinematics(robot.model, robot.data, q, dq, np.zeros(nv))
            pin.updateFramePlacements(robot.model, robot.data)
            ee_pos = robot.data.oMf[config.f_id].translation

            if j == 0:
                A, B = ee_pos[1], goal.detach().numpy()[1]
                threshold = -B / (A-B)
                crossing = (threshold > 0)
                crossed = False
                via_point = np.zeros(3)
                via_point[:2] = threshold*ee_pos[:2] + (1-threshold)*goal.detach().numpy()[:2]
                via_point[2] = obs_height + 0.05                   

            progress = (ee_pos[1] - B)/(A-B)
            if (progress > threshold
                and crossing
                and not crossed):
                goal_temp = torch.tensor(via_point)
            else:
                goal_temp = goal.clone()
                crossed = True

            old_loss = torch.inf
            for _ in range(config.max_it):
                x_pred = ioc(x_init) 
                loss = task_loss(x_pred, goal_temp, obs_pos_arr, obs_rad, config)
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
            Xi[j] = torch.hstack((torch.tensor(x_init), 
                                  goal, 
                                  obs_height*torch.ones(3))).detach().float()
            Yi[j] = torch.hstack((ioc.weight.detach().clone().flatten(), 
                                  ioc.x_nom.detach().clone()))

        q = x_pred[-2*nq:-nq]
        dq = x_pred[-nq:]
        pin.forwardKinematics(robot.model, robot.data, q, dq, np.zeros(nv))
        pin.updateFramePlacements(robot.model, robot.data)
        ee_pos = robot.data.oMf[config.f_id].translation
        dist = np.linalg.norm(ee_pos - goal.detach().numpy())

        dist_obs = ee_pos - obs_pos_arr[0].detach().numpy()
        A = np.diag(1 / (obs_rad ** 2))
        sd_ellipsoid = np.sqrt(dist_obs @ A @ dist_obs) - 1.0
        
        if dist <= config.distance_threshold or sd_ellipsoid < 0:
            # only store successful task executions
            X.append(Xi)
            Y.append(Yi)
            restart = False
        else:
            print(dist)
            print(sd_ellipsoid)
            restart = True

    x_train = torch.vstack(X)
    y_train = torch.vstack(Y)

    return x_train, y_train


def generate_reaching(n_tasks, task_loss, robot, config, viz=None):
    n_col, nq, nv = config.n_col, config.nq, config.nv
    u_max, dt = config.u_max, config.dt
    X = []
    Y = []

    for i in trange(n_tasks):
        # generate random goal location
        r = config.r_range[0] + config.r_range[1]*np.random.rand()
        z = config.z_range[0] + config.z_range[1]*np.random.rand()
        th = config.th_range[0] + config.th_range[1]*np.random.rand()
        goal = torch.tensor([r*np.sin(th), r*np.cos(th), z])
        
        # generate random robot configuration
        if i == 0 or np.random.randint(config.n_restart) == 0:
            x_init = np.array(config.x_init)
            x_init[:nq] += config.q_noise * (np.random.rand(nv) - 0.5)
            x_init[nq:] = config.dq_noise * (np.random.rand(nv) - 0.5)
            x_init[0] -= 2*0.5*(np.random.rand() - 0.5)
            x_init[2] -= 2*0.3*(np.random.rand() - 0.5)
        
        # visualization
        if viz is not None:
            viz.display(x_init[:nq])
            viz.viewer["box"].set_object(g.Sphere(0.05), 
                            g.MeshLambertMaterial(
                            color=0xff22dd,
                            reflectivity=0.8))
            viz.viewer["box"].set_transform(tf.translation_matrix(goal.detach().numpy()))                

        
        # allocate memory for the data from the i-th task
        Xi = torch.zeros((config.task_horizon, 
                            len(config.x_init) + 3))
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
                loss = task_loss(robot, x_pred, goal, nq, n_col)
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