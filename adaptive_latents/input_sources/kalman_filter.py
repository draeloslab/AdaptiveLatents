import numpy as np


class KalmanFilter:
    def __init__(self, use_steady_state_k=False, subtract_means=True):
        self.use_steady_state_K = use_steady_state_k
        self.subtract_means = subtract_means

        self.A = None  # state transitions
        self.C = None  # link between states and observations
        self.Q = None  # state noise
        self.W = None  # observation noise

        self.steady_state_K = None

        self.state_var = None
        self.state = None

    def fit(self, X, Y):
        X_mean, Y_mean = (X.mean(axis=0), Y.mean(axis=0)) if self.subtract_means else (0,0)

        X = X - X_mean
        Y = Y - Y_mean

        A, _, _, _ = np.linalg.lstsq(X[:-1], X[1:])

        C, _, _, _ = np.linalg.lstsq(X, Y)

        w = X[1:] - X[:-1] @ A
        W = (w.T @ w) / (X.shape[1] - 1)

        q = Y - X @ C
        Q = (q.T @ q) / (X.shape[1])

        # model variables
        self.A = A
        self.C = C
        self.W = W
        self.Q = Q
        self.X_mean = X_mean
        self.Y_mean = Y_mean

        if self.use_steady_state_K:
            m = X.shape[1]
            P = W
            matrix = C.T @ P @ C + Q

            K_old = P @ C @ np.linalg.pinv(matrix)
            P = (np.eye(m) - C @ K_old.T) @ P
            for i in range(3000):
                P = A @ P @ A.T + W
                matrix = C.T @ P @ C + Q
                K = P @ C @ np.linalg.pinv(matrix)
                P = (np.eye(m) - C @ K.T) @ P

                dif = np.abs(K - K_old)
                K_old = K
                if (dif < 1E-16).all():
                    break
            self.steady_state_K = K

        # state variables
        self.state = np.zeros_like(X[-1:])
        self.state_var = self.W

    def step(self, Y=None):
        state = self.state @ self.A
        state_var = self.A @ self.state_var @ self.A.T + self.W

        if Y is not None:
            Y = Y - self.Y_mean
            if not self.use_steady_state_K:
                kalman_gain = state_var @ self.C @ np.linalg.pinv(self.C.T @ state_var @ self.C + self.Q)
            else:
                kalman_gain = self.steady_state_K
            state = state + (Y - state @ self.C) @ kalman_gain.T
            state_var = (np.eye(self.C.shape[0]) - self.C @ kalman_gain.T) @ state_var

        self.state = state
        self.state_var = state_var
        return state + self.X_mean

    def predict(self, n_steps, initial_state=None, initial_state_var=None):
        old_state, old_var = self.state, self.state_var  # TODO: I don't like saving the state like this

        if initial_state is not None:
            self.state = initial_state - self.X_mean
        if initial_state_var is None:
            self.state_var = self.W

        prediction = np.zeros((n_steps, self.A.shape[0])) * np.nan
        for i in range(n_steps):
            prediction[i] = self.step()

        self.state, self.state_var = old_state, old_var
        return prediction


if __name__ == '__main__':
    rng = np.random.default_rng()

    import matplotlib.pyplot as plt
    from scipy.stats import special_ortho_group

    samples_per_second = 20
    seconds_per_rotation = 3
    total_seconds = 12

    t = np.linspace(0, total_seconds / seconds_per_rotation * np.pi * 2, total_seconds * samples_per_second + 1)
    X = np.vstack((np.cos(t), np.sin(t))).T
    X = X + 10
    Y = X @ special_ortho_group(dim=20, seed=rng).rvs()[:, :2].T


    kf = KalmanFilter(use_steady_state_k=False, subtract_means=True)
    kf.fit(X, Y)

    plt.plot(X[:, 0], X[:, 1])

    initial_point = X[-1:]
    X_hat = kf.predict(n_steps=samples_per_second * seconds_per_rotation, initial_state=initial_point)


    one_step_distance = np.linalg.norm(X[0] - X[1])
    def close(a, b, radius):
        return np.linalg.norm(a - b) < radius


    plt.plot([initial_point[0, 0], X_hat[0, 0]], [initial_point[0, 1], X_hat[0, 1]], '--.', color='C1')
    plt.plot(X_hat[:, 0], X_hat[:, 1], '.-')

    assert close(X_hat[-1], initial_point, 2 * one_step_distance)
    assert not close(X_hat[X_hat.shape[0] // 2], initial_point, 2 * one_step_distance)

    initial_point = np.array([[10, 10]])
    X_hat = kf.predict(initial_state=initial_point, n_steps=samples_per_second * seconds_per_rotation)
    plt.plot([initial_point[0, 0], X_hat[0, 0]], [initial_point[0, 1], X_hat[0, 1]], '--.', color='C2')
    plt.plot(X_hat[:, 0], X_hat[:, 1], '.-')
    assert close(X_hat[-1], initial_point, .1 * one_step_distance)
    assert close(X_hat[X_hat.shape[0] // 2], initial_point, .3)
    plt.axis('equal')
    plt.show()