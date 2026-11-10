import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from adaptive_latents import ArrayWithTime
from adaptive_latents.plotting_functions import AnimationManager, plot_history_with_tail

from adaptive_latents import proSVD, proPLS, ArrayWithTime, KernelSmoother
from adaptive_latents.utils import resample_matched_timeseries

from adaptive_latents.input_sources.datasets import Duszkiewicz26Dataset

rng = np.random.default_rng(0)

data = Duszkiewicz26Dataset(9, bin_width=0.03)
nd = data.neural_data # (371413, 185) I think means 185 neurons?
hd = data.head_direction # (303775, 1)

# %% Visualize the head direction over time
fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
ax.plot(hd.t, hd)
ax.set(
    xlabel="Time (?)",
    ylabel="Head direction (radians)",
    title="Head direction over time",
)
for t in data.epochs.stop_time:
    ax.axvline(x=t, color='r')

# %% Visualize the distribution of head directions

fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
ax.hist(hd, bins=50, density=True)
ax.set(
    xlabel="Head direction (radians)",
    ylabel="Density",
    title="Distribution of simulated head direction",
)

# %% Visualize the neural responses

fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
# Maybe just plot three neurons for now
# for cell in range(np.shape(nd)[1]):
for cell in range(3):
    ax.plot(nd[:,cell].t,nd[:,cell])
# opposite_cells = (0, n_head_direction_cells // 2)
# for cell in opposite_cells:
#     ax.plot(t, responses[:, cell], label=f"Head-direction cell {cell + 1}")

ax.set(
    xlabel="Time (s)",
    ylabel="Neural response",
    title="Neural Responses over Time",
)
ax.legend(title="Cell")

print(data.epochs.tags) # We'll want to select tags "wake_square" and "wake_triangle"

def filter_for_good_times(a):
    squareIdx = pd.Index(data.epochs.tags).get_loc("wake_square")
    wake_square = (data.epochs.start_time[squareIdx] < a.t) & (a.t < data.epochs.stop_time[squareIdx])
    try:
        triangleIdx = pd.Index(data.epochs.tags).get_loc("wake_triangle")
        wake_triangle = (data.epochs.start_time[triangleIdx] < a.t) & (a.t < data.epochs.stop_time[triangleIdx])
    except KeyError:
        # Should guarantee a full array of false
        wake_triangle = a.t < 0
    return a.slice(wake_square | wake_triangle)

raw_x = filter_for_good_times(data.neural_data)
smoother = KernelSmoother(tau=.1/raw_x.dt) # TODO: TUNE THIS

smooth_x = smoother.offline_run_on([raw_x]) # smoothing it
smooth_x = smooth_x - smooth_x.mean(axis=0) # centering data

raw_y = filter_for_good_times(data.behavioral_data)
resample_y = resample_matched_timeseries(raw_y, raw_y.t, smooth_x.t) # Matches y t's to x t's

fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
ax.plot(raw_x.t, raw_x[:,0] - raw_x[:,0].mean(), '-b', label="Raw Neural Data")
ax.plot(smooth_x.t, smooth_x[:,0], '-r', label="Smoothed Neural Data")

fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
ax.plot(resample_y, smooth_x[:,0], '.')
ax.set(
    xlabel="Head Direction (radians)",
    ylabel="Smoothed Spike Data"
)

# These are all places where the behavioral data drops out to nan
# The way to fix would be to just remove all of the nan points entirely
# Interpolation doesn't seem appropriate here since most of these are improbably fast jumps in angle
times_to_check = [3761.91425, 3761.94425, 3761.9742499999998, 3762.00425, 3762.0342499999997, 3762.06425, 3762.12425, 3762.1542499999996, 3762.18425, 3762.21425, 3930.9942499999997, 4549.50425, 5276.07425, 5279.854249999999, 9907.47425, 10331.34425, 10751.52425, 11346.874249999999, 12083.25425]
for t in times_to_check:
    print(resample_y[resample_y.time_to_sample(t)])
    
keepIdx = np.logical_not(np.isnan(resample_y))

y = resample_y[keepIdx]
y.t = resample_y.t[np.reshape(keepIdx,(-1))]
# Can I assume that x is in time order???
x = smooth_x[np.reshape(keepIdx,(-1)),:]
x.y = smooth_x.t[np.reshape(keepIdx,(-1))]

print(np.shape(y))
print(np.shape(x))

fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
ax.plot(y.t, y)

svd = proSVD(k=2, log_level=2)

latents_prosvd = svd.offline_fit_then_transform(smooth_x) # this is more stable than offline_run_on, but offline_run_on is more realistic for what we'd actually use
# latents = pro.offline_run_on(responses)

# %%

theta = np.linspace(0, 2*np.pi, 500) # for drawing a circle
hd_bins = np.linspace(0, 2*np.pi, 51) # bin edges for head direction colors
c = y.flatten() * np.nan
bar_c = theta * np.nan
for bin_i in range(len(hd_bins) - 1):
    s = (hd_bins[bin_i] <= y) & (y < hd_bins[bin_i + 1])
    s = s.flatten()
    c[s] = bin_i

    s = (hd_bins[bin_i] <= theta) & (theta < hd_bins[bin_i + 1])
    s = s.flatten()
    bar_c[s] = bin_i

cmap = 'hsv' # also try 'twilight' or 'twilight_shifted'

# %%

fig, ax = plt.subplots()

ax.scatter(latents_prosvd[:, 0], latents_prosvd[:, 1], c=c, s=1, cmap=cmap)

ax.scatter(np.cos(theta) + 4.5, np.sin(theta) + 2.2, c=bar_c, s=15, cmap=cmap) # this is like a colorbar for the head direction
ax.axis('scaled')

# %%

pls = proPLS(k=2, log_level=2)

y_circular_basis = np.transpose(np.stack([np.sin(y.reshape(-1)), np.cos(y.reshape(-1))]))
print(np.shape(y_circular_basis))
y_circular_basis = ArrayWithTime(y_circular_basis, y.t)

latents_propls = pls.offline_fit_then_transform([x, y_circular_basis])

# %%

fig, ax = plt.subplots()

# This is using the same colorbins as defined above, may or may not be valid given proPLS takes in both x and y
ax.scatter(latents_propls[:, 0], latents_propls[:, 1], c=c, s=1, cmap=cmap)

ax.scatter(np.cos(theta) + 4.5, np.sin(theta) + 2.2, c=bar_c, s=15, cmap=cmap) # this is like a colorbar for the head direction
ax.axis('scaled')
plt.show()

# %%

pls = proPLS(k=2, log_level=2)

# instead of representing the head direction as a single angle like above, represent it as a 2d vector on the unit circle (cos(theta), sin(theta))
hd_circular_basis = np.transpose(np.stack([np.sin(hd.reshape(-1)), np.cos(hd.reshape(-1))]))
print(np.shape(hd_circular_basis))
hd_circular_basis = ArrayWithTime(hd_circular_basis, t) # this is necessary to make the timestamps line up

latents = pls.offline_fit_then_transform([responses, hd_circular_basis])

# %% Plot the latents over time
fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
ax.plot(latents.t, latents) # note the use of the t attribute of the ArrayWithTime latents
ax.set(
    xlabel="",
    ylabel="",
    title="",
)

# %% Plot a scatterplot of latent 1 against latent 2
plt.plot(latents[:, 0], latents[:, 1], '.')

# %% Plot a scatterplot of latent 1 against latent 2 in black, and overlay the last 100 points with a line
fig2, ax2 = plt.subplots()
ax2.plot(latents[:,0], latents[:,1],'k.')
ax2.plot(latents[np.shape(latents)[0]-100:np.shape(latents)[0],0], latents[np.shape(latents)[0]-100:np.shape(latents)[0],1])
ax2.set_title("Last 100 points")

# %% Plot a scatterplot of latent 1 against latent 2 in black, and overlay the last 5 seconds with a line
# hint: use ArrayWithTime.slice_by_time(slice(start_t, end_t))
slicedLatents = latents.slice_by_time(slice(t.max()-5,t.max()))
fig3, ax3 = plt.subplots()
ax3.plot(latents[:,0], latents[:,1],'k.')
ax3.plot(slicedLatents[:,0], slicedLatents[:,1])
ax3.set_title("Last 5 seconds")
plt.show()

# %% Feel free to play around with this

with AnimationManager(outdir='.', filename_stem='latent_trajectory', filetype='mp4', fps=30) as am:
    for current_t in np.linspace(n_seconds-60, n_seconds, 3*60+1):
        for ax in am.axs.flatten():
            ax.cla()
        ax = am.axs[0,0]

        plot_history_with_tail(ax=ax, data=latents, current_t=current_t, tail_length=1)

        am.grab_frame()


# %% Plot the weights of u per each neuron, for each latent dimension
# You'll get two lines, one for each latent dimension. The x-axis should be the neuron index, and the y-axis should be the weight of that neuron in the projection.

# %% Plot the norm of the projection weights for each neuron.
# The x-axis should be the neuron index, and the y-axis should be the norm of the projection weights.
# Is there a pattern for head direction cells vs non-head direction cells?
# Can you classify them?


# %% Use the AnimationManager to make an animation of the latent trajectory over time, with a moving point that shows the current position in the latent space, and a line that shows the trajectory over the last 5 seconds.



# %% You don't have to understand all of this, I just think it's interesting.

subspace_cov = np.cov(pls.project(x=pls.project(x=responses), project_up=True).T)
full_d_cov = np.cov(responses.T)
no_noise_cov = np.cov(no_noise_responses.T)

true_tuning_structure = np.cov([[cannonical_tuning_curve(th - hd) for th in theta] for hd in hd_tunings])
true_cov_matrix = np.block([[true_tuning_structure, np.zeros((n_head_direction_cells, n_non_hd_cells))],
                            [np.zeros((n_non_hd_cells, n_head_direction_cells)), np.eye(n_non_hd_cells) * true_tuning_structure[0,0] * 1.2**2]])


fig, axs = plt.subplots(figsize=(16, 3), ncols=4, layout="constrained")
covariance_matrices = (
    (subspace_cov, "Covariance in projected subspace"),
    (full_d_cov, "Covariance of noisy responses"),
    (no_noise_cov, "Covariance of noise-free responses"),
    (true_cov_matrix, "Theoretical covariance"),
)
for ax, (matrix, title) in zip(axs, covariance_matrices):
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set(
        xlabel="Neuron index",
        ylabel="Neuron index",
        title=title,
    )
    fig.colorbar(image, ax=ax, label="Covariance")
    ax.axis('scaled')
plt.show()

# %%

fig, axs = plt.subplots(figsize=(12, 3), ncols=3, layout="constrained")
covariance_errors = (
    (subspace_cov - full_d_cov, "Error due to dimensionality reduction"),
    (full_d_cov - no_noise_cov, "Error due to observation noise"),
    (no_noise_cov - true_cov_matrix, "Error due to head-direction sampling"),
)

for ax, (matrix, title) in zip(axs, covariance_errors):
    image = ax.imshow(matrix, aspect="auto", cmap="coolwarm")
    ax.set(
        xlabel="Neuron index",
        ylabel="Neuron index",
        title=title,
    )
    fig.colorbar(image, ax=ax, label="Covariance difference")
    ax.axis('scaled')
plt.show()
