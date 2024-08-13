import numpy as np
rng = np.random.default_rng()

def align_to(A, B):
    # R = argmin(lambda omega: norm(omega @ A - B))
    A, B = A.T, B.T
    C = A @ B.T
    u, s, vh = np.linalg.svd(C)
    R = vh.T @ u.T
    return (R @ A).T, R.T


if __name__ == '__main__':
    base_d = 5
    high_d = 10
    n_points = 20
    max_rank = 6
    X = rng.normal(size=(base_d, n_points))
    Y = rng.normal(size=(base_d, n_points))

    X = np.vstack([X, np.zeros((high_d - base_d, n_points))])
    Y = np.vstack([Y, np.zeros((high_d - base_d, n_points))])

    cov_old_way = np.zeros((high_d, high_d))

    u, s, v = np.zeros((high_d,max_rank)), np.zeros((max_rank, max_rank)), np.zeros((high_d,max_rank))
    step = 3
    for i in range(0,20,step):
        x = X[:,i:i+step]
        y = Y[:,i:i+step]

        cov_old_way = cov_old_way + x @ y.T

        x_hat = u.T @ x
        x_orth = x - u @ x_hat
        q_x_orth, r_x_orth = np.linalg.qr(x_orth)

        y_hat = v.T @ y
        y_orth = y - v @ y_hat
        q_y_orth, r_y_orth = np.linalg.qr(y_orth)

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

        if not "align":
            u = new_u
            s = new_s
            v = new_v
        else:
            u, R_u = align_to(new_u, u) # u = new_u @ R_u
            v, R_v = align_to(new_v, v)
            s = R_u.T @ new_s @ R_v

        # old_u, old_s, old_vh = np.linalg.svd(cov_old_way)
        # print(old_s)
        # print(np.diag(new_s))



    assert np.allclose(X@Y.T, (u@s)@v.T)
