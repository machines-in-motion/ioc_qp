import numpy as np

class DoubleIntegrator:
    
    def __init__(self, n, u_max, dt = 0.01):
        """
        This class creates the matrices for a unit massdouble integrator 
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
        
        self.A_dyn = np.matrix([[1, 0, 0, -1, self.dt],
                              [0, 1, self.dt, 0, -1]])
        self.b_dyn = np.matrix([0,0])
        
        self.A = np.zeros((2*n+2,3*n+2))
        self.b = np.zeros(2*n+2)
        self.G = np.zeros((2*n, 3*n+2))
        self.h = u_max*np.ones(2*n)
        
        self.Q = np.zeros((3*n+2,3*n+2))
        self.R = np.zeros_like(self.Q)
        self.q = np.zeros((3*n+2))
        
    def create_matrices(self, x_init, x_des, wt, wt_ter):
        
        for i in range(0, self.n):
            self.A[2*i:2*(i+1), 3*i:3*i+5] = self.A_dyn.copy()
            self.G[2*i:2*(i+1), 3*(i)+2] = [1,-1]
            self.Q[3*i:3*i+3, 3*i:3*i+3] = np.matrix([   [wt[0], 0, 0], 
                                                         [0, wt[1], 0],
                                                         [0, 0, wt[2]]])

        self.A[-2:, 0:2] = np.eye(2)
        self.b[-2:] = x_init

        self.Q[-2:, -2:] = np.matrix([[wt_ter[0], 0], 
                                      [0, wt_ter[1]]])
        self.q[-2:] = -2*np.multiply(wt_ter,x_des)
        
        return self.Q, self.q, self.A, self.b, self.G, self.h 

    def create_matrices_nn(self):
        
        """
        This function creates matrices for the differentiable QP module
        """

        for i in range(0, self.n):
            self.A[2*i:2*(i+1), 3*i:3*i+5] = self.A_dyn.copy()
            self.G[2*i:2*(i+1), 3*(i)+2] = [1,-1]
            
        self.A[-2:, 0:2] = np.eye(2)
        
        return self.A, self.b, self.G, self.h 