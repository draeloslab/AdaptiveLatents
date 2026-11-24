import numpy as np
from adaptive_latents import ArrayWithTime, proPLS, proSVD
from adaptive_latents.utils import principle_angles, column_space_distance, is_orthonormal
from scipy.stats import special_ortho_group
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

#%%
# lower is better
def prosvd_residual_objective(A, Q):
    assert is_orthonormal(Q)
    original = A
    reconstructed = (A @ Q) @ Q.T
    return np.linalg.norm(original - reconstructed)**2

def pls_residual_objective(A, B, U, V):
    assert is_orthonormal(U) and is_orthonormal(V)
    return np.linalg.norm((A @ U) - (B @ V))**2

#%%
# higher is better
def prosvd_variance_objective(A, Q):
    # technically this is the PCA objective, but it's more comparable to the PLS objective
    assert is_orthonormal(Q)
    return np.trace(Q.T @ A.T @ A @ Q)

def pls_variance_objective(A, B, U, V):
    assert is_orthonormal(U) and is_orthonormal(V)
    return np.trace(U.T @ A.T @ B @ V)

# %% generate synthetic data
A = rng.uniform(size=(100, 10)) * np.arange(10)
B = A[:,:5] + rng.uniform(size=(A.shape[0], 5)) * .1
k = 2

# %% train proSVD
pro = proSVD(k=k, log_level=2)
pro.offline_run_on(A)

prosvd_residual_objective(A, pro.Q)

# %% train proPLS
pls = proPLS(k=k)
pls.offline_run_on([A,B])

pls_residual_objective(A, B, pls.u, pls.vh.T)
pls_variance_objective(A, B, pls.u, pls.vh.T)

# %% generate offline fits

# to fairly compare online and offline, we need the algorithms to be fit on the same data
offline_svds = []
for i in range(k,len(A)):

    u, s, vh = np.linalg.svd(A[:i], full_matrices=False)
    offline_svd_Q = vh[:k].T

    offline_svds.append(ArrayWithTime(offline_svd_Q, t=i))


u, s, vh = np.linalg.svd(A.T @ B, full_matrices=False)
offline_pls_U = u[:,:k]
offline_pls_V = vh[:k].T

# %% generate random subspaces to compare to
# We aren't doing a formal permuation test here, but we're comparing to a similar null distriubtion
# https://www.jwilber.me/permutationtest/

Q_nulls = []
for _ in range(1000): # feel free to use more, the histogram will be smoother
    Q_null = special_ortho_group.rvs(random_state=rng,dim=A.shape[1])[:, :k]
    Q_nulls.append(Q_null)


# %% distance to offline SVD
fig, ax = plt.subplots()
ax.axvline(column_space_distance(offline_svd_Q, offline_svd_Q), color='k', linestyle='-', label='self distance')

ax.hist([column_space_distance(Q_null, offline_svd_Q) for Q_null in Q_nulls], bins=100, density=True, label='random subspaces')
ax.axvline(column_space_distance(pro.Q, offline_svd_Q), color='r', linestyle='-', label='proSVD')
ax.axvline(column_space_distance(pls.u, offline_svd_Q), color='r', linestyle='--', label='proPLS')
ax.set_title('Grassmannian distance to offline SVD column space')
ax.set_xlabel('distance metric')
ax.set_ylabel('density')
ax.legend()

# %% distance to final offline SVD over time
fig, axs = plt.subplots(ncols=2, figsize=(10,5))

for ax, column_space_method in zip(axs, ['grassmann', 'frobenius']):
    for name, Q_series in [('proSVD', pro.log['Q']), ('offline SVD', offline_svds)]:
        distances = [column_space_distance(Q_t, offline_svd_Q, method=column_space_method) for Q_t in Q_series]
        t = [Q_t.t for Q_t in Q_series]
        ax.plot(t, distances, label=name)

    ax.axhline(column_space_distance(offline_svd_Q, offline_svd_Q), color='k', linestyle='-')

    ax.set_title(f'Distance to final offline SVD ({column_space_method})')
    ax.set_xlabel('sample in dataset')
    ax.set_ylabel('distance')
ax.legend()

# %% distance to offline PLS

# %% distance to final offline PLS over time

# %% stability over time for proSVD

# %% stability over time for proPLS

# %% principle angles between proSVD and proPLS spaces (in degrees)

# %% histogram of the objective value of all the spaces for proSVD's variance objective function

# %% something about proSVD's variance objective function over time

# %% I'm interested in proSVD's two objective functions; please make a scatterplot of their values against each other

# %% histogram of the objective value of all the spaces for proPLS's variance objective function
# To evaluate some of the spaces, you'll need to choose a V; both  offline_pls_V and pls.vh.T are valid choices.

# %% feel free to add any missing analyses here; once you've run this on the synthetic data, run it on the real data