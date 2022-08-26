## This file contains different pinocchio robot costs with pytorch compatibility
## Author : Avadesh Meduri
## Date : 25/02/2022

import numpy as np
import pinocchio as pin

import torch
from torch.autograd import Function
from torch.nn import functional as F

class DiffFrameTranslationCost(Function):
    """
    This cost provides gradients wrt joint position and velocity for the final end effector/frame translation
    """
    @staticmethod
    def forward(ctx, state, model, data, f_id):
        """
        Input:
            state : vector (q, dq)
            model : pinocchio robot model
            data : pinocchio robot data
            f_id : frame id for which derivatives are desired
        """
        state = state.double().detach().numpy()
        nq, nv = model.nq, model.nv
        pin.forwardKinematics(model, data, state[0:nq], state[nq:nq + nv], np.zeros(nv))
        pin.updateFramePlacements(model, data)
        
        J = pin.computeFrameJacobian(model, data, state[0:nq], f_id, pin.ReferenceFrame.LOCAL)
        J_rot = np.matmul(data.oMf[f_id].rotation,J[0:3])
        
        J_rot_torch = torch.tensor(J_rot, dtype = torch.double)
        ctx.J = J_rot_torch
        
        return torch.tensor(data.oMf[f_id].translation)

    @staticmethod
    def backward(ctx, grad):
        
        jac_rot_torch = ctx.J
        fk_q = jac_rot_torch.t()@grad # derivative wrt joint positions
        fk_dq = torch.zeros(jac_rot_torch.shape[1]) # derivative wrt joint velocities
        
        return torch.hstack((fk_q, fk_dq)), None, None, None
        

class DiffFramePlacementCost(Function):
    """
    This cost provides gradients wrt joint position and velocity for the final end effector/frame placement
    """
    @staticmethod
    def forward(ctx, state, model, data, f_id, M_des):
        """
        Input:
            state : vector (q, dq)
            model : pinocchio robot model
            data : pinocchio robot data
            f_id : frame id for which derivatives are desired
        """
        state = state.double().detach().numpy()
        nq, nv = model.nq, model.nv
        pin.forwardKinematics(model, data, state[0:nq], state[nq:nq + nv], np.zeros(nv))
        pin.updateFramePlacements(model, data)
        
        J = pin.computeFrameJacobian(model, data, state[0:nq], f_id, pin.ReferenceFrame.LOCAL)
        J_rot = np.matmul(data.oMf[f_id].rotation,J[0:3])
        
        J_rot_torch = torch.tensor(J_rot, dtype = torch.double)
        ctx.J = J_rot_torch
        
        return torch.tensor(data.oMf[f_id].translation)

    @staticmethod
    def backward(ctx, grad):
        
        jac_rot_torch = ctx.J
        fk_q = jac_rot_torch.t()@grad # derivative wrt joint positions
        fk_dq = torch.zeros(jac_rot_torch.shape[1]) # derivative wrt joint velocities
        
        return torch.hstack((fk_q, fk_dq)), None, None, None
        


class DiffFrameVelocityCost(Function):
    """
    This cost provides gradients wrt joint position and velocity for the final end effector/frame velocity
    """
    @staticmethod
    def forward(ctx, state, accel, model, data, f_id):
        """
        Input:
            state : vector (q, dq, a)
            accel : acceleration of joints
            model : pinocchio robot model
            data : pinocchio robot data
            f_id : frame id for which derivatives are desired
        """
        state = state.double().detach().numpy()
        accel = accel.double().detach().numpy()
        nq, nv = model.nq, model.nv
        pin.computeForwardKinematicsDerivatives(model, data, state[0:nq], state[nq:nq + nv], accel)
        pin.updateFramePlacements(model, data)
        
        vel = pin.getFrameVelocity(model, data, f_id, pin.ReferenceFrame.WORLD)
        dv_dq, dv_ddq = pin.getFrameVelocityDerivatives(model,data,f_id,pin.ReferenceFrame.WORLD)
        
        ctx.der = torch.hstack((torch.tensor(dv_dq), torch.tensor(dv_ddq)))
        
        return torch.tensor(np.array(vel))

    @staticmethod
    def backward(ctx, grad):
        
        der = ctx.der # derivatives wrt to velocity
        return der.t()@grad, None, None, None, None

