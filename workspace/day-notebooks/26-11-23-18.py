# %%
from adaptive_latents.input_sources.datasets import Gardner22Dataset, ArrayWithTime
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d, correlate2d
from skimage.feature import peak_local_max
from tqdm import tqdm
from adaptive_latents.utils import rotation_matrix

# %%
d = Gardner22Dataset()

# %%
d.neural_data.shape
# %%
pos, fr = ArrayWithTime.align_indices(d.position, d.neural_data)
# %%
i = 11
s = fr[:, i] > 0
plt.scatter(pos[s, 0], pos[s, 1], c=fr[s, i], s=1)
plt.axis('equal')
# %%
i = 17
s = fr[:, i] > 0
x_edges, y_edges = np.linspace(d.position[:, 0].min(), d.position[:, 0].max(), 101), np.linspace(d.position[:, 1].min(), d.position[:, 1].max(), 101)

occupancy = np.histogram2d(pos[:, 0], pos[:, 1], bins=[x_edges, y_edges])[0]
occupancy[occupancy < 1] = 1e-10
spiking = np.histogram2d(pos[:, 0], pos[:, 1], bins=[x_edges, y_edges], weights=fr[:,i])[0]

tuning_curve = gaussian_filter(spiking/occupancy, 1)

fig, axs = plt.subplots(1, 2)

autocorrelation = correlate2d(tuning_curve, tuning_curve, mode='same')

axs[0].matshow(tuning_curve)

axs[1].matshow(autocorrelation)
peaks = peak_local_max(autocorrelation, min_distance=10)
for x, y in peaks:
    axs[1].scatter(y, x, c='r', s=10)
axs[1].scatter(peaks[0,1], peaks[0,0], c='r', s=40)

tc_peaks = peak_local_max(tuning_curve, min_distance=10)
# for x, y in tc_peaks:
#     axs[0].scatter(y, x, c='r', s=10)


scale_px = np.linalg.norm(peaks[1:] - peaks[0], axis=1).mean()
rotation = np.atan2(*(peaks[1] - peaks[0])[::-1])

theta = np.linspace(0, 2 * np.pi, 100)
axs[0].plot(np.cos(theta) * scale_px + tc_peaks[0,1], np.sin(theta) * scale_px + tc_peaks[0, 0], c='r', lw=2)

r = np.linspace(0,1,2)
axs[0].plot(tc_peaks[0,1] + np.cos(rotation) * r * scale_px, tc_peaks[0,0] + np.sin(rotation) * r * scale_px, '.-r')


print(scale_px)
scale_m = (scale_px * (x_edges[1] - x_edges[0]))

# %%
u_1 = np.array([1, 0])
u_2 = np.array([1, np.sqrt(3)]) / 2
u_3 = np.array([-1, np.sqrt(3)]) / 2
basis = np.column_stack([u_1, u_2, u_3])

rotation_matrix = np.array([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]])

rotated_basis = -rotation_matrix @ basis

plt.scatter(pos[s, 0], pos[s, 1], c=fr[s, i], s=1)
j = np.argmax(fr[s, i])
for k in range(3):
    plt.plot(pos[s, 0][j] + np.array([0, 1, 2,]) * rotated_basis[0, k] * scale_m, pos[s, 1][j] + np.array([0, 1, 2,]) * rotated_basis[1, k] * scale_m, '.-r')

# %%
def embed(position, scale_m, rotation):
    u_1 = np.array([1, 0])
    u_2 = np.array([1, np.sqrt(3)]) / 2
    u_3 = np.array([-1, np.sqrt(3)]) / 2
    basis = np.column_stack([u_1, u_2, u_3])
    rotation_matrix = np.array([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]])
    phases = position @ rotation_matrix @ basis * 2*np.pi/scale_m
    embedding = np.column_stack([np.cos(phases), np.sin(phases)])
    return embedding

embedding = embed(pos, scale_m, rotation)

# %%

alternate_scales = np.linspace(0, 2*2*np.pi/scale_m, 50)
alternate_rotations = np.linspace(0, np.pi/3, 50)

scores = []
scales = []
rotations = []
pbar = tqdm(total=len(alternate_rotations) * len(alternate_scales))
for r in alternate_rotations:
    rotation = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
    for sc in alternate_scales:
        phases = pos @ rotation @ basis * sc
        scale_embedding = np.column_stack([np.cos(phases), np.sin(phases)])
        scale_embedding = scale_embedding

        if not 'use_sklearn':
            reg = LinearRegression(fit_intercept=True)
            reg.fit(scale_embedding, fr)
            scores.append(reg.score(scale_embedding, fr))
        else:

            X = scale_embedding - scale_embedding.mean(axis=0)
            Y = fr - fr.mean(axis=0)
            tss = np.sum(Y * Y, axis=0)

            xtx = X.T @ X
            xty = X.T @ Y
            coef = np.linalg.pinv(xtx) @ xty

            r2 = np.divide(
                np.sum(xty * coef, axis=0),
                tss,
                out=np.ones_like(tss),  # constant targets score 1 with an intercept
                where=tss > 0,
            )
            scores.append(r2.mean())

        scales.append(sc)
        rotations.append(r)
        pbar.update(1)
pbar.close()
rotations = np.asarray(rotations)
scales = np.asarray(scales)
scores = np.asarray(scores)
np.savez('/home/jgould/Downloads/scale_rotation_scores.npz', scales=scales, rotations=rotations, scores=scores)
# %%

# %%
score_grid = np.log(np.asarray(scores).reshape(len(alternate_rotations), len(alternate_scales)))
best_score = np.argmax(score_grid, keepdims=True)

scale_step = alternate_scales[1] - alternate_scales[0]
rotation_step = alternate_rotations[1] - alternate_rotations[0]
scale_edges = np.linspace(
    alternate_scales[0] - scale_step / 2,
    alternate_scales[-1] + scale_step / 2,
    len(alternate_scales) + 1,
)
rotation_edges = np.linspace(
    alternate_rotations[0] - rotation_step / 2,
    alternate_rotations[-1] + rotation_step / 2,
    len(alternate_rotations) + 1,
)
scale_grid, rotation_grid = np.meshgrid(scale_edges, rotation_edges)

vmin = None

fig, axs = plt.subplots(ncols=4, figsize=(12, 3), constrained_layout=True)
mesh = axs[0].pcolormesh(scale_grid, rotation_grid, score_grid, cmap='viridis', shading='flat', vmin=vmin)
axs[0].scatter(scales[best_score], rotations[best_score], c='r', s=20)
axs[1].pcolormesh(
    np.cos(rotation_grid) * scale_grid,
    np.sin(rotation_grid) * scale_grid,
    score_grid,
    cmap='viridis',
    shading='flat',
    vmin=vmin,
)
for i in range(6):
    mesh = axs[2].pcolormesh(
        np.cos(rotation_grid + np.pi / 3 * i) * scale_grid,
        np.sin(rotation_grid + np.pi / 3 * i) * scale_grid,
        score_grid,
        cmap='viridis',
        shading='flat',
        vmin=vmin,
    )
axs[3].pcolormesh(
    np.cos(rotation_grid * 6) * scale_grid,
    np.sin(rotation_grid * 6) * scale_grid,
    score_grid,
    cmap='viridis',
    shading='flat',
    vmin=vmin,
)
fig.colorbar(mesh, ax=axs, label='log score')

axs[0].set_title('Scale vs rotation')
axs[1].set_title('Scale vs rotation, polar')
axs[2].set_title('Scale vs rotation, polar tiled')
axs[3].set_title('Scale vs rotation, polar stretched')

#%%
r = np.squeeze(rotations[best_score])

rotation = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
phases = pos @ rotation @ basis * scales[best_score]
scale_embedding = np.column_stack([np.cos(phases), np.sin(phases)])
scale_embedding = scale_embedding

reg = LinearRegression(fit_intercept=True)
reg.fit(scale_embedding, fr)
fr_hat = reg.predict(scale_embedding)

#%%
i = 12
fig, axs = plt.subplots(ncols=2)
s = fr[:, i] > 0
axs[0].scatter(pos[s, 0], pos[s, 1], c=fr[s, i], s=1)

s = fr_hat[:, i] > 0
axs[1].scatter(pos[s, 0], pos[s, 1], c=fr_hat[s, i], s=1)


for ax in axs:
    ax.axis('equal')

#%%

def fit_predict(pos, s, r, k, b, offset, fr):
    phases = (pos - offset) @ rotation_matrix(r) @ basis * s
    return np.exp(np.cos(phases).sum(axis=1) * k + b)



i = 12

fr_hat = fit_predict(pos, s=11.5, r= 5/20, k=.3, b=0,offset=np.array([1,1])/10, fr=fr[:,i])


fig, axs = plt.subplots(ncols=2)
s = fr[:, i] > 0
axs[0].scatter(pos[s, 0], pos[s, 1], c=fr[s, i], s=1)

s = fr_hat > 0
axs[1].scatter(pos[s, 0], pos[s, 1], c=fr_hat[s], s=1)


for ax in axs:
    ax.axis('equal')
