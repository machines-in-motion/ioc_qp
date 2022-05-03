## This class creates the matrices for the inverse kinematics for an arbitraty 
## DOF system. The matrices are constraints (dyn and inequality constraints)
## NOTE : This is designed for fixed base systems (nq = nv)
## Author : Avadesh Meduri
## Date : 22/02/2022


import numpy as np 

class InverseKinematics:

    def __init__(self, n : int, nq : int, tau_lim : list, dt : float = 0.05):
        """
        Input:
            n : number of colocation points
            nq : number of joints/DOF in the system
            tau_lim: torque limits (len(tau_lim) == nq)
            dt : discretization of time
        """
        self.n = n
        self.nq = nq
        assert len(tau_lim) == self.nq
        self.tau_lim = tau_lim
        self.dt = dt

        zeros = np.zeros((self.nq, self.nq))
        eye = np.eye(self.nq)
        self.A_dyn = np.block([[eye, zeros, zeros, -1*eye, self.dt*eye],
                               [zeros, eye, self.dt*eye, zeros, -1*eye ]])
        self.b_dyn = np.zeros(2*self.nq)

        n_vars = 3*self.nq*self.n + 2*self.nq
        self.A = np.zeros(((self.n+1)*(2*self.nq), n_vars))
        self.b = np.zeros((self.n+1)*(2*self.nq))
        self.G = np.zeros((2*self.nq*self.n, n_vars))
        self.h = np.tile(tau_lim, 2*self.n)

        self.Q = np.zeros((n_vars, n_vars))
        self.q = np.zeros(n_vars)

    def create_matrices(self, q_init, q_des, wt, wt_ter):
        """
        This function creates the matrices for solving the IK problem
        Input:
            q_init : initial state of the sytem (q, dq)
            q_des : desired state of the system (q, dq)
            wt : running weights
            wt_ter : terminal weights
        """

        wt_arr = np.eye(3*self.nq)
        np.fill_diagonal(wt_arr, wt)

        for i in range(self.n):

            self.A[2*self.nq*i: 2*self.nq*(i+1), 3*self.nq*i: 3*self.nq*i + 5*self.nq] = self.A_dyn.copy()
            self.G[self.nq*(2*i):self.nq*(2*i+1), self.nq*(3*i+2):self.nq*(3*i+3)] = np.eye(self.nq)
            self.G[self.nq*(2*i+1):self.nq*(2*i+2), self.nq*(3*i+2):self.nq*(3*i+3)] = -np.eye(self.nq)
            self.Q[3*self.nq*i:3*self.nq*(i+1), 3*self.nq*i:3*self.nq*(i+1)] = wt_arr

        # intial constraints
        self.A[-2*self.nq:, 0:2*self.nq] = np.eye(2*self.nq)
        self.b[-2*self.nq:] = q_init

        tmp = np.eye(2*self.nq)
        np.fill_diagonal(tmp, wt_ter)
        self.Q[-2*self.nq:, -2*self.nq:] = tmp
        self.q[-2*self.nq:] = -2*np.multiply(wt_ter, q_des)

        return self.Q, self.q, self.A, self.b, self.G, self.h
        
    def create_matrices_nn(self):
        """
        This function creates matrices for the differenitable QP setup
        """                            

        for i in range(self.n):

            self.A[2*self.nq*i: 2*self.nq*(i+1), 3*self.nq*i: 3*self.nq*i + 5*self.nq] = self.A_dyn.copy()
            self.G[self.nq*(2*i):self.nq*(2*i+1), self.nq*(3*i+2):self.nq*(3*i+3)] = np.eye(self.nq)
            self.G[self.nq*(2*i+1):self.nq*(2*i+2), self.nq*(3*i+2):self.nq*(3*i+3)] = -np.eye(self.nq)


        self.A[-2*self.nq:, 0:2*self.nq] = np.eye(2*self.nq)

        return self.A, self.b, self.G, self.h
