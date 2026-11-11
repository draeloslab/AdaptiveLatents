import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
from adaptive_latents import ArrayWithTime
from adaptive_latents.plotting_functions import AnimationManager, plot_history_with_tail
from scipy.optimize import newton
from tqdm import tqdm
import gudhi
from gudhi.sklearn import RipsPersistence
from scipy.stats import qmc
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from scipy.stats import special_ortho_group
from adaptive_latents.utils import principle_angles

from adaptive_latents import proSVD, proPLS

rng = np.random.default_rng(0)

# %%

n_seconds = .15 * 60
t = np.arange(0, n_seconds, 0.01)  # 100 Hz sampling rate

maze_size = 2.0  # meters

n_non_grid_cells = 100
n_grid_cells = 100

tunings = rng.uniform(0, maze_size, size=(n_grid_cells,2))
tunings = tunings[np.argsort(tunings[:, 0]), :]


# %%
fig, ax = plt.subplots()
ax.scatter(tunings[:,0], tunings[:,1])
ax.axis('scaled')

# %%
scale = 3*np.pi
def cannonical_tuning_curve_f(x, x0, b=0, kappa=1.5, scale=scale):
    k_1 = scale * np.array([1, 0])
    k_2 =  scale * np.array([1, np.sqrt(3)]) / 2
    k_3 =  scale * np.array([-1, np.sqrt(3)]) / 2
    xx1 = k_1 @ (x - x0) % (2 * np.pi)
    xx2 = k_2 @ (x - x0) % (2 * np.pi)
    xx3 = k_3 @ (x - x0) % (2 * np.pi)
    r = np.exp(b + kappa * (np.cos(xx1) + np.cos(xx2) + np.cos(xx3)))
    return r / np.exp(b + 3 * kappa)

# %%
fig, ax = plt.subplots()

samples = np.linspace(0, maze_size, 100)
samples = np.stack(np.meshgrid(samples, samples), axis=-1).reshape(-1, 2)
cannonical_tuning_curve = np.array([cannonical_tuning_curve_f(sample, (0,0)) for sample in samples])

ax.imshow(cannonical_tuning_curve.reshape(100, 100), extent=(0, maze_size, 0, maze_size), origin='lower')

ax.axis('scaled')

# %%

position = rng.uniform(0, maze_size, size=(t.size, 2))

# %%
fig, ax = plt.subplots()
ax.scatter(position[:, 0], position[:, 1])
ax.axis('scaled')


# %%

grid_firing_rates = [[cannonical_tuning_curve_f(location, tuning) for tuning in tunings] for location in tqdm(position)]
grid_firing_rates = np.array(grid_firing_rates)

non_grid_firing_rates = [rng.choice(cannonical_tuning_curve, size=len(t), replace=True) * 1.3 for _ in range(n_non_grid_cells)]
non_grid_firing_rates = np.array(non_grid_firing_rates).T.reshape(len(t), n_non_grid_cells)

# %%
x_no_noise = ArrayWithTime(np.hstack([grid_firing_rates, non_grid_firing_rates]), t)

x = x_no_noise + rng.normal(scale=0.01, size=x_no_noise.shape)

# %%

u_1 = np.array([1, 0])
u_2 = np.array([1, np.sqrt(3)]) / 2
u_3 = np.array([-1, np.sqrt(3)]) / 2
basis = np.column_stack([u_1, u_2, u_3])


# %%
phases = position @ basis * scale
embedding = np.column_stack([np.cos(phases), np.sin(phases)])

# %%
alternate_scales = np.linspace(0, 3*scale, 40)
alternate_rotations = np.linspace(0, np.pi/3, 40)

scores = []
scales = []
rotations = []
for r in tqdm(alternate_rotations):
    rotation = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
    for s in alternate_scales:
        phases = position @ rotation @ basis * s
        scale_embedding = np.column_stack([np.cos(phases), np.sin(phases)])
        scale_embedding = scale_embedding

        reg = LinearRegression(fit_intercept=True)
        reg.fit(scale_embedding, x)
        scores.append(reg.score(scale_embedding, x))
        scales.append(s)
        rotations.append(r)
# %%

fig, ax = plt.subplots()
ax.scatter(scales, rotations, c=np.log(scores), cmap='viridis')
# ax.axvline(scale)

# %%
pca = PCA(n_components=6)
pca.fit(x)
# %%
cca = CCA(n_components=6)

cca.fit(x, embedding)

x_projected = cca.predict(x)


# %%
fig, axs = plt.subplots(nrows=2)

axs[0].plot(pca.components_.T)
axs[1].plot(cca.x_weights_)

principle_angles(pca.components_.T, cca.x_weights_) * 180 / np.pi
# %% filled cube
point_cloud = rng.uniform(0, maze_size, size=(500, 3))


#%% ring
theta = np.linspace(0, 2 * np.pi, 50)
point_cloud = np.array([[np.cos(t1), np.sin(t1), 0] for t1 in theta])

#%% sphere
point_cloud = rng.normal(size=(500, 3))
point_cloud = point_cloud / np.linalg.norm(point_cloud, axis=1, keepdims=True)

#%% disk
point_cloud = rng.uniform(0, 2*np.pi, size=(500, 2))

phi = point_cloud[:, 0]
theta = point_cloud[:, 1]
point_cloud = np.column_stack([np.cos(theta)*np.sin(phi), np.sin(theta)*np.sin(phi), np.sin(phi) * 0])

#%% quasirandom sampled sphere
point_cloud = qmc.Sobol(d=2, scramble=True, seed=0).random(n=512)

phi = np.acos(2*point_cloud[:, 1]-1)
theta = point_cloud[:, 0] * 2 * np.pi

point_cloud = np.column_stack([np.cos(theta)*np.sin(phi), np.sin(theta)*np.sin(phi), np.cos(phi)])

#%% torus

theta = np.linspace(0, 2 * np.pi, 25)

point_cloud = []
for t1 in theta:
    for t2 in theta:
        point = np.array([np.cos(t1), np.sin(t1), 0]) * 2
        point = point + np.array([np.cos(t1) * np.cos(t2), np.sin(t1) * np.cos(t2), np.sin(t2)])
        point_cloud.append(point)

point_cloud = np.array(point_cloud)

#%% sobol torus

R = 2 # major radius
r = 1 # minor radius

point_cloud = qmc.Sobol(d=2, scramble=True, seed=0).random(n=512)
# point_cloud = rng.uniform(0, 1, size=(512, 2))
theta = point_cloud[:, 0] * 2 * np.pi
phi = np.zeros_like(theta)
for i in range(len(phi)):
    def f(p):
        return p + (r/R) * np.sin(p) - point_cloud[i, 1] * 2 * np.pi
    def f_prime(p):
        return 1 + (r/R) * np.cos(p)
    guess = point_cloud[i, 1] * 2 * np.pi

    phi[i] = newton(f, guess, fprime=f_prime)

xx = (R + r*np.cos(phi)) * np.cos(theta)
y = (R + r*np.cos(phi)) * np.sin(theta)
z = r * np.sin(phi)

point_cloud = np.column_stack([xx, y, z])


#%% 3d plot

fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])
ax.axis('equal')

# %%
# point_cloud = x_no_noise
# point_cloud = x
# point_cloud = embedding
point_cloud = x_projected

# %%
sample_size = min(700, len(point_cloud))
sample_indices = rng.choice(len(point_cloud), size=sample_size, replace=False)
subsampled_point_cloud = point_cloud[sample_indices]

# Limit the Rips complex to a local scale to keep the number of simplices manageable.
distance_matrix = squareform(pdist(subsampled_point_cloud))
np.fill_diagonal(distance_matrix, np.inf)
nearest_neighbor_distances = distance_matrix.min(axis=1)
max_edge_length = 10 * np.quantile(nearest_neighbor_distances, 0.9)


diagrams = RipsPersistence(
    homology_dimensions=[0, 1, 2],
    threshold=max_edge_length,
).fit_transform([subsampled_point_cloud])[0]

gudhi.plot_persistence_diagram(diagrams)
plt.show()