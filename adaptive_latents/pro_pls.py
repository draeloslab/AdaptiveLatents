import numpy as np
import scipy.linalg
import adaptive_latents as al
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression

class BaseProPLS:
    def __init__(self, k=10, decay_alpha=1):
        self.k = k
        self.decay_alpha = decay_alpha

        self.u = None
        self.s = None
        self.vh = None
        self.n_samples_observed = 0

    def initialize(self, x, y):
        d_x = x.shape[1]
        d_y = y.shape[1]
        assert min(d_x, d_y) >= self.k, "k size doesn't make sense"

        self.u = np.zeros((d_x, self.k))
        self.s = np.zeros((self.k, self.k))
        self.vh = np.zeros((self.k, d_y))
        self.n_samples_observed = 0
        self.update(x, y) # TODO: check semantics of initialization

    def add_new_input_channels(self, n_x=0, n_y=0):
        if n_x:
            self.u = np.vstack([self.u, np.zeros(shape=(n_x, self.u.shape[1]))])
        if n_y:
            self.vh = np.hstack([self.vh, np.zeros((self.vh.shape[0], n_y))])

    def update(self, x, y):
        # decompose x into parallel and orthogonal components
        x_hat = x @ self.u
        x_orth = x - x_hat @ self.u.T
        r_x_orth, q_x_orth = scipy.linalg.rq(x_orth, mode='economic')

        # decompose y into parallel and orthogonal components
        y_hat = y @ self.vh.T
        y_orth = y - y_hat @ self.vh
        r_y_orth, q_y_orth = scipy.linalg.rq(y_orth, mode='economic')

        # decay old s information
        new_s = self.s * self.decay_alpha

        # construct the new svd
        new_u = np.hstack([self.u, q_x_orth.T])
        new_s = np.block([
            [new_s + x_hat.T@y_hat, x_hat.T@r_y_orth],
            [r_x_orth.T@y_hat, r_x_orth.T@r_y_orth]
        ])
        new_vh = np.vstack([self.vh, q_y_orth])

        # diagonalize the new svd
        u_tilde, s_tilde, vh_tilde = np.linalg.svd(new_s)

        # drop the smallest-covariance dimensions from our new svd
        new_u = (new_u @ u_tilde)[:,:self.k]
        new_s = np.diag(s_tilde[:self.k])
        new_vh = (vh_tilde @ new_vh)[:self.k]

        # align the new svd to the previous u and vh matrices
        result = np.linalg.svd(new_u.T @ self.u)
        R_u = result[0] @ result[2]
        self.u = new_u @ R_u

        result = np.linalg.svd(new_vh @ self.vh.T)
        R_v = result[2].T @ result[0].T
        self.vh = R_v @ new_vh

        self.s = R_u.T @ new_s @ R_v.T

    def project(self, *, x=None, y=None):
        x_proj = None
        y_proj = None

        if x is not None:
            x_proj = x @ self.u
        if y is not None:
            y_proj = y @ self.vh.T

        return tuple(filter(lambda z: z is not None, [x_proj, y_proj]))

    def get_cross_covariance(self):
        return self.u @ self.s @ self.vh



def row_version():
    rng = np.random.default_rng()
    base_d = 5
    high_d = (10, 9)
    n_points = 100
    X = rng.normal(size=(n_points, base_d))
    Y = rng.normal(size=(n_points, base_d))

    X = np.hstack([X, np.zeros((n_points, high_d[0] - base_d))])
    Y = np.hstack([Y, np.zeros((n_points, high_d[1] - base_d))])


    step = 3

    pls = BaseProPLS(k=5)
    for i in range(0,n_points,step):
        x = X[i:i+step]
        y = Y[i:i+step]

        if i == 0:
            pls.initialize(x, y)
        else:
            pls.update(x, y)

    assert np.allclose(X.T@Y, pls.get_cross_covariance())

def actually_works():
    rng = np.random.default_rng()
    high_d = (10, 9)
    n_points = 1000
    common_d = 2

    X = rng.normal(size=(n_points, high_d[0]))
    Y = rng.normal(size=(n_points, high_d[1]))

    common = rng.normal(size=(n_points, common_d))
    snr = 1

    Y[:,:common_d] = (snr * common + rng.normal(size=(n_points, common_d)))/np.sqrt(1 + snr**2)
    X[:,:common_d] = (snr * common + rng.normal(size=(n_points, common_d)))/np.sqrt(1 + snr**2)

    x_common_basis = np.eye(high_d[0])[:,:common_d]

    pls = BaseProPLS(k=2)

    step = 1
    errors = []
    for i in range(0,n_points,step):
        x = X[i:i+step]
        y = Y[i:i+step]

        if i == 0:
            pls.initialize(x, y)
        else:
            pls.update(x, y)

        errors.append(np.abs(al.utils.principle_angles(pls.u, x_common_basis)).sum())
    plt.plot(errors)

    sk_pls = PLSRegression(n_components=pls.k).fit(X, Y)
    offline_error = np.abs(al.utils.principle_angles(sk_pls.x_weights_, x_common_basis)).sum()
    plt.axhline(offline_error, color='k', linestyle='--')

    plt.semilogy()
    plt.show()




def align_to(A, B):
    # R = argmin(lambda omega: norm(omega @ A - B))
    A, B = A.T, B.T
    C = A @ B.T
    u, s, vh = np.linalg.svd(C)
    R = vh.T @ u.T
    return (R @ A).T, B.T, R.T


def column_version():
    rng = np.random.default_rng()
    base_d = 5
    high_d = 10
    n_points = 100
    max_rank = 6
    X = rng.normal(size=(base_d, n_points))
    Y = rng.normal(size=(base_d, n_points))

    X = np.vstack([X, np.zeros((high_d - base_d, n_points))])
    Y = np.vstack([Y, np.zeros((high_d - base_d, n_points))])

    cov_old_way = np.zeros((high_d, high_d))

    u, s, v = np.zeros((high_d,max_rank)), np.zeros((max_rank, max_rank)), np.zeros((high_d,max_rank))
    step = 1
    for i in range(0,n_points,step):
        x = X[:,i:i+step]
        y = Y[:,i:i+step]

        cov_old_way = cov_old_way + x @ y.T

        x_hat = u.T @ x
        x_orth = x - u @ x_hat
        q_x_orth, r_x_orth = scipy.linalg.qr(x_orth, mode='economic')

        y_hat = v.T @ y
        y_orth = y - v @ y_hat
        q_y_orth, r_y_orth = scipy.linalg.qr(y_orth, mode='economic')

        new_u = np.hstack([u, q_x_orth])
        new_s = np.block([
            [s + x_hat@y_hat.T, x_hat@r_y_orth.T],
            [r_x_orth@y_hat.T, r_x_orth@r_y_orth.T]
        ])
        new_v = np.vstack([v.T, q_y_orth.T])

        u_tilde, s_tilde, vh_tilde = np.linalg.svd(new_s)
        new_u = (new_u @ u_tilde)[:,:max_rank]
        new_s = np.diag(s_tilde[:max_rank])
        new_v = (new_v.T @ vh_tilde.T)[:,:max_rank]

        if not 'align':
            u = new_u
            s = new_s
            v = new_v
        elif 'align':
            u, old_u, R_u = align_to(new_u, u) # u = new_u @ R_u
            v, old_v, R_v = align_to(new_v, v)
            s = R_u.T @ new_s @ R_v

        # old_u, old_s, old_vh = np.linalg.svd(cov_old_way)
        # print(old_s)
        # print(np.diag(new_s))



    assert np.allclose(X@Y.T, (u@s)@v.T)

if __name__ == '__main__':
    actually_works()
    column_version()