class StreamingKalmanFilter():
    def __init__(self):
        self.A = None
        self.C = None
        self.W = None
        self.Q = None
      
    def train_model(self, neural, kin):
        if self.A is not None:
            raise ValueError("Tried to train_model a model that's already trained ")
        ## neural should have shape [timepoints, features] and kin should have shape [timepoints, behavioral dimensions]
        Y = neural.T
        X = kin.T
        ## Train Kalman filter by calculating Theta1 and K
        XT = np.transpose(X)
        ## Calculate A
        # A = (X[:, 1:] @ XT[:-1, :]) @ utils.inverse_singular(X[:, 0:-1] @ XT[0:-1, :])
        a = X[:, 1:] @ XT[:-1, :]
        A = (X[:, 1:] @ XT[:-1, :]) @ np.linalg.pinv(X[:, 0:-1] @ XT[0:-1, :])
        ## Calculate C
        # C = Y @ XT @ utils.inverse_singular(X @ XT)
        C = Y @ XT @ np.linalg.pinv(X @ XT)
        # Find W
        w = X[:, 1:] - A @ X[:, :-1]
        W = (w @ np.transpose(w)) / (np.shape(X)[1] - 1)
        # Find Q
        q = Y - C @ X
        Q = (q @ np.transpose(q)) / (np.shape(X)[1])
        self.A = A
        self.C = C
        self.W = W
        self.Q = Q
        self.Pt = self.W
        self.xlast = np.zeros(X.shape[0])
      
    def run_model(self, Y):
        Y = Y.T
        xlast_p = self.A @ self.xlast #X_corr[:, i-1]
        plast = self.A @ self.Pt @ self.A.T + self.W
        Kt = plast @ self.C.T @ np.linalg.pinv(self.C @ plast @ self.C.T + self.Q)
        X_corr = xlast_p +Kt @(Y - self.C @ xlast_p)
        self.Pt = (np.eye(self.C.shape[1]) - Kt @ self.C) @ plast
        self.xlast = X_corr
        return X_corr

    
class StreamingSteadyStateKalmanFilter(Model):
    def __init__(self):
        self.K = None
        self.theta1 = None
               
    def train_model(self, neural, kin):
        
        if self.K is not None:
            raise ValueError("Tried to train_model a model that's already trained ")
        
        Y = neural.T
        X = kin.T

        ## Train Kalman filter by calculating Theta1 and K
        XT = np.transpose(X)
        
        ## Calculate A
        A = (X[:, 1:] @ XT[:-1, :]) @ utils.inverse_singular(X[:, 0:-1] @ XT[0:-1, :])

        ## Calculate C 
        C = Y @ XT @ utils.inverse_singular(X @ XT)
        
        # Find W 
        w = X[:, 1:] - A @ X[:, :-1]
        W = (w @ np.transpose(w)) / (np.shape(X)[1] - 1)

        # Find Q
        q = Y - C @ X
        Q = (q @ np.transpose(q)) / (np.shape(X)[1])
        
        #Find steady state K
        m = np.shape(X)[0]
        P = W
        CT = np.transpose(C)
        
        matrix = C @ W @ CT + Q
        
        KOld = P @ CT @ utils.inverse_singular(matrix)
        P = (np.identity(m) - KOld @ C) @ P
        dif = KOld
        nRec = 0
        while True:
            nRec +=1
            P = A @ P @ np.transpose(A) + W
            matrix = C @ P @ CT + Q
            K = P @ CT @ utils.inverse_singular(matrix) 
            dif = abs(K-KOld)
            KOld = K
            P = (np.identity(m) - K @ C) @ P
            if (dif < 1E-16).all():
                break
 
            if nRec > 3000:
                print("more than 3000")
                break
            
        print(f"Num iterations to get steady state K {nRec}")
        print(np.max(dif))
        theta = A - K @ C @ A 

        self.theta1 = theta
        self.K = K
        self.xlast = np.zeros(X.shape[0])
  
    def run_model(self, neural):
        Y = neural.T

        m = np.shape(self.xlast)[0]

        theta2 = self.K @ Y.reshape((np.shape(Y)[0], 1))
        a = self.theta1 @ self.xlast.reshape((m, 1)) + theta2
        self.xlast = np.reshape(a, (m))
        
        return self.xlast.T
