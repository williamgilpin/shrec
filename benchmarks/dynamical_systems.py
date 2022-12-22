import numpy as np
import warnings
import os

def load_data():
    """Load precomputed data for the Lorenz system"""
    dirname = os.path.dirname(__file__)
    X_measure = np.load(dirname + "/data/lorenz_data.npy", allow_pickle=True)
    y_driver = np.load(dirname + "/data/lorenz_driver.npy", allow_pickle=True)
    return X_measure, y_driver

class DrivenLorenz:
    """
    An ensemble of Lorenz systems driven by an external signal

    Parameters
        random_state (int): seed for the random number generator
        driver (str): type of driver, either "rossler" or "periodic"
    """
    def __init__(self, random_state=None, driver="rossler"):
        
        # driver properties
        self.ad = 0.5
        self.n = 5.3
        self.r = 1

        if driver == "rossler":
            self.rhs_driver = self._rhs_rossler
        elif driver == "periodic":
            self.rhs_driver= self._rhs_periodic
        else:
            warnings.warn("Unknown driver type, defauling to Rossler")
            self.rhs_driver= self._rhs_rossler
        
        # response properties
#         self.ar = 1.2
#         self.mu = 8.53
#         self.w = 0.63
        self.rho = 28
        self.beta = 2.667
        self.sigma = 10
        
        self.n_drive = 3
        self.n_response = 3
        
        ## rossler
        self.a = 0.2
        self.b = 0.2
        self.c = 5.7
        
        np.random.seed(random_state)
        self.n_sys = 24
        self.rho = 28 * (1 + 0.5*(np.random.random(self.n_sys) - 0.5))
        self.beta = 2.667 * (1 + 0.1*(np.random.random(self.n_sys) - 0.5))
        self.sigma = 10 * (1 + 0.1*(np.random.random(self.n_sys) - 0.5))
        

        self.n_sys = 12 * 3 * 2 * 4
        self.rho = 28 * (1 + 1 + 0.5*(np.random.random(self.n_sys) - 0.5))
        self.beta = 2.667 * (1 + 0.1*(np.random.random(self.n_sys) - 0.5))
        self.sigma = 10 * (1 + 0.1*(np.random.random(self.n_sys) - 0.5))
        
        
        
        self.rho = 28 * (1 + 1 + 0.1 * (np.random.random(self.n_sys) - 0.5))
        self.beta = 2.667 * (1 + 0.1 * (np.random.random(self.n_sys) - 0.5))
        self.sigma = 10 * (1 + 2 * (np.random.random(self.n_sys) - 0.5))
        
        
        self.rho = 28 * (1 + 5 * (np.random.random(self.n_sys)))
        self.beta = 2.667 * (1 +  0.1 * (np.random.random(self.n_sys)))
        self.sigma = 10 * (1 + 20 * (np.random.random(self.n_sys) ))
        
        
        
        self.rho = 28 * (1 + 2 * 5 * (np.random.random(self.n_sys)))
        self.beta = 2.667 * (1 +  2 * 1 * (np.random.random(self.n_sys)))
        self.sigma = 10 * (1 + 2 * 10 * (np.random.random(self.n_sys) ))
        

    def _rhs_periodic(self, t, X):
        """Simple periodic driver"""
        x, y, z = X
        a, n, r = self.ad, self.n, self.r
        xdot = a * 15 * np.sin(t / 2) - x
        ydot =  a * 15 * np.sin(t / 2) - y
        zdot =  a * 15 * np.sin(t / 2) - z
        return xdot, ydot, zdot

    def _rhs_rossler(self, t, X):
        """Rossler driving (aperiodic)"""
        x, y, z = X
        a, b, c = self.a, self.b, self.c
        xdot = -y - z
        ydot = x + a * y
        zdot = b + z * (x - c)
        return xdot * 0.5, ydot * 0.5, zdot * 0.5
    
    def rhs_response_ensemble(self, t, X):
        """Response system

        Args:
            t (float): time
            X (np.ndarray): state vector

        Returns:
            np.ndarray: derivative of the state vector
        """
        
        Xd = X[:self.n_drive]
        Xr = X[self.n_drive:]
        
        xd, yd, zd = Xd
        x, y, z = Xr[:self.n_sys], Xr[self.n_sys:2*self.n_sys], Xr[2 * self.n_sys:]

        xdot = self.sigma * (y - x) + self.ar * xd
        ydot = x * (self.rho - z) - y # - self.ar * xd
        zdot = x * y - self.beta * z
        return np.hstack([xdot, ydot, zdot])
    
    def rhs(self, t, X):
        """Full system

        Args:
            t (float): time
            X (np.ndarray): state vector

        Returns:
            np.ndarray: derivative of the state vector
        """
        return [*self.rhs_driver(t, X[:self.n_drive]), *self.rhs_response_ensemble(t, X)]

