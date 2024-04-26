import numpy as np
from scipy.linalg import block_diag
from adaptive_latents.regressions import VanillaOnlineRegressor
import matplotlib.pyplot as plt
import warnings
from scipy.stats import special_ortho_group

def make_H(d):
    h = []
    for i in range(0,d):
        for j in range(0,i):
            a = np.zeros((d,d))
            a[i,j] = 1
            a[j,i] = -1
            h.append(a.flatten())
    return np.column_stack(h)

def make_X_tilde(X, order='C'):
    m, n = X.shape
    match order:
        case 'C':
            X_tilde = np.zeros(shape=(m * n, n * n))
            for i in range(m):
                for j in range(n):
                    X_tilde[i * n + j, j * n:(j + 1) * n] = X[i]
        case 'F':
            X_tilde = block_diag(*[X] * n)
        case _:
            raise Exception("Input must be 'C' or 'F'")

    return X_tilde

def X_and_X_dot_from_data(X_all):
    """note: this is technically off-by-one for the way I norally think about it, but it's causal"""
    # todo: is this necessarily off-by-one?
    X_dot = np.diff(X_all,axis=0)
    X = X_all[1:]
    return X, X_dot

def U_from_beta(beta, n, last_U=None):
    # todo: you can derive n from beta
    assert n % 2 == 0
    if np.any(np.isnan(beta)):
        return np.zeros((n,n)) * np.nan
    H = make_H(n)
    sksym = (H@beta.ravel()).reshape(n,n)
    evals, evecs = np.linalg.eig(sksym)
    idx = np.argsort(np.abs(np.imag(evals)) + 1j * np.imag(evals))[::-1]
    evals, evecs = evals[idx], evecs[:,idx]

    U = np.zeros((n,n))
    for i in range(n//2):
        v1 = evecs[:,i*2]
        v2 = evecs[:,i*2+1]
        if np.sign(np.real(v1[0])) != np.sign(np.real(v2[0])):
            v2 *= -1
        assert np.allclose(np.real(v1), np.real(v2))
        u1 = v1 + v2
        u2 = 1j * (v1 - v2)
        u1 /= np.linalg.norm(u1)
        u2 /= np.linalg.norm(u2)
        assert np.allclose(np.imag(u1),0)
        assert np.allclose(np.imag(u2),0)
        U[:,i*2] = np.real(u1)
        U[:,i*2+1] = np.real(u2)
        if last_U is not None and np.all(~np.isnan(last_U)):
            U[:, (i * 2): (i * 2 +2)], _ = align_column_spaces(U[:, (i * 2): (i * 2 +2)], last_U[:, (i * 2): (i * 2 +2)])
    return U

def jpca_data(X):
    if X.shape[1]> 16:
        warnings.warn("this jpca implementation assumes low-d input")
    X, X_dot = X_and_X_dot_from_data(X)
    n = X.shape[1]
    H = make_H(n)
    reg = VanillaOnlineRegressor(input_d=H.shape[1], output_d=1, init_min_ratio=2)

    U = None
    observations = []
    for i in range(X.shape[0]):
        rows = make_X_tilde(X[i, None]) @ H
        for j in range(n):
            x = rows[j]
            y = X_dot[i, j]
            reg.observe(x, y)
        U = U_from_beta(reg.get_beta(), n, U)
        observations.append(X[i] @ U)

    return np.array(observations), U


def align_column_spaces(A, B):
    # R = argmin(lambda omega: norm(omega @ A - B))
    A, B = A.T, B.T
    C = A @ B.T
    u, s, vh = np.linalg.svd(C)
    R = vh.T @ u.T
    return (R @ A).T, (B).T

def generate_circle_embedded_in_high_d(rng, m=1000, n=4, stddev=1, shape=(10,10)):
    t = np.linspace(0, m/50*np.pi*2, m+1)
    circle = np.column_stack([np.cos(t),np.sin(t)]) @ np.diag(shape)
    C = special_ortho_group(dim=n,seed=rng).rvs()[:,:2]
    X_all = (circle @ C.T) + rng.normal(size=(m+1,n))*stddev
    X, X_dot = X_and_X_dot_from_data(X_all)
    return X, X_dot, dict(C=C)

# def generate_by_circle(rng, m=1000, n=4):
#     t = np.linspace(0, m/50*np.pi*2, m+1)
#     circle = np.column_stack([np.cos(t),np.sin(t)]) @ np.diag([20,10])
#     C = special_ortho_group(dim=n,seed=rng).rvs()[:,:2]
#     X_all = (circle @ C.T) + rng.normal(size=(m+1,n))
#     X, X_dot = from_data(X_all)
#     return X, X_dot, dict(C=C)
