## This file contains the pinocchio based kinematic models
## of robots
## Author : Avadesh Meduri
## Date : 10/02/2021

import pinocchio as pin 
import numpy as np  

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

def TwoDofManipulator():
# initialising robot
    model = pin.Model()
    HFE_id = model.addJoint(0, pin.JointModelRZ(), pin.SE3.Identity(), "HFE_joint")
    model.addFrame(pin.Frame("HFE", HFE_id, 0 , pin.SE3.Identity(),  pin.FrameType.FIXED_JOINT))

    frame_loc = np.eye(4)
    frame_loc[:,3][1] = -1.0
    KFE_id = model.addJoint(HFE_id, pin.JointModelRZ(), pin.SE3(frame_loc), "KFE_joint")
    model.addFrame(pin.Frame("KFE", KFE_id, 0 , pin.SE3.Identity(),  pin.FrameType.FIXED_JOINT))

    model.addFrame(pin.Frame("FOOT", KFE_id, 0 , pin.SE3(frame_loc), pin.FrameType.FIXED_JOINT))

    data = model.createData()

    return model, data


def two_dof_vis(x_traj, x_des, model, data):
        '''
        This function simulates the kinematics trajectory 
        returned by ilqr
        Input:
            x_traj : trajectory returned by ilqr
        '''
        fig = plt.figure()
        l1 = 1.0
        l2 = 1.0
        ax = plt.axes(xlim=(-l1 - l2 -1, l1 + l2 + 1), ylim=(-l1 - l2 -1, l1 + l2 + 1))
        text_str = "Two Dof Manipulator Animation"
        arm1, = ax.plot([], [], lw=4)
        arm2, = ax.plot([], [], lw=4)
        base, = ax.plot([], [], 'o', color='black')
        joint, = ax.plot([], [], 'o', color='green')
        hand, = ax.plot([], [], 'o', color='pink')
        des, = ax.plot([], [], 'x', color='red')
        
        def init():
            arm1.set_data([], [])
            arm2.set_data([], [])
            base.set_data([], [])
            joint.set_data([], [])
            hand.set_data([], [])
            
            des.set_data([], [])
            
            return arm1, arm2, base, joint, hand, des
        
        def animate(i):
            q = x_traj[:,i][0:model.nq]
            v = x_traj[:,i][model.nq:]

            a = np.zeros(model.nv)
            
            pin.forwardKinematics(model, data, q, v, a)
            pin.updateFramePlacements(model, data)
            
            joint_x = data.oMf[2].translation[0] 
            joint_y = data.oMf[2].translation[1]
            
            hand_x = data.oMf[3].translation[0]
            hand_y = data.oMf[3].translation[1]

            base.set_data([0, 0])
            arm1.set_data([0,joint_x], [0,joint_y])
            joint.set_data([joint_x, joint_y])
            arm2.set_data([joint_x, hand_x], [joint_y, hand_y])
            hand.set_data([hand_x, hand_y])
            des.set_data(x_des[0:2])
            
            return base, arm1, joint, arm2, hand
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=15,
        verticalalignment='top', bbox=props)
        ax.grid()
        anim = FuncAnimation(fig, animate, init_func=init,
                                       frames=x_traj.shape[1])
        plt.close()
        anim.to_html5_video()
        return anim