import numpy as np

class TwoDPointMass:
    
    def __init__(self, n, u_max, dt = 0.01):
        """
        This class creates the matrices for a 2d unit mass system 
        system moving from an initial condition to a final condition
        Input:
            mass : mass of double integrator
            n : number of collocation points
            u_max : torque limit
            dt : discretization
        """
        
        self.n = n
        self.dt = dt
        self.u_max = u_max
        
        self.A_dyn = np.matrix([[1, 0, 0, 0, 0, 0, -1, 0, self.dt, 0],
                                [0, 1, 0, 0, 0, 0, 0, -1, 0, self.dt], 
                                [0, 0, 1, 0, self.dt, 0, 0, 0, -1, 0],
                                [0, 0, 0, 1, 0, self.dt, 0, 0, 0, -1]])

        self.b_dyn = np.matrix([0, 0, 0, 0])
        
        self.A = np.zeros((4*n+4,6*n+4))
        self.b = np.zeros(4*n+4)
        self.G = np.zeros((4*n, 6*n+4))
        self.h = u_max*np.ones(4*n)
        
        self.Q = np.zeros((6*n+4,6*n+4))
        self.R = np.zeros_like(self.Q)
        self.q = np.zeros((6*n+4))

    def create_matrices(self, x_init, x_des, wt, wt_ter):
        
        wt_arr = np.eye(6)
        np.fill_diagonal(wt_arr, wt)

        for i in range(0, self.n):
            self.A[4*i:4*(i+1), 6*i:6*i+10] = self.A_dyn.copy()
            self.G[4*i:4*i+2, 6*(i)+4] = [1,-1]
            self.G[4*i+2:4*i+4, 6*(i)+5] = [1,-1]
            self.Q[6*i:6*i+6, 6*i:6*i+6] = wt_arr

        self.A[-4:, 0:4] = np.eye(4)
        self.b[-4:] = x_init

        tmp = np.eye(4)
        np.fill_diagonal(tmp, wt_ter)
        self.Q[-4:, -4:] = tmp
        self.q[-4:] = -2*np.multiply(wt_ter,x_des)
        
        return self.Q, self.q, self.A, self.b, self.G, self.h 
    
    def create_matrices_nn(self):
        
        """
        This function creates matrices for the differentiable QP module
        """

        for i in range(0, self.n):
            self.A[4*i:4*(i+1), 6*i:6*i+10] = self.A_dyn.copy()
            self.G[4*i:4*i+2, 6*(i)+4] = [1,-1]
            self.G[4*i+2:4*i+4, 6*(i)+5] = [1,-1]

        self.A[-4:, 0:4] = np.eye(4)

        return self.A, self.b, self.G, self.h 