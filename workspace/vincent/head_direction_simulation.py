import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from adaptive_latents import ArrayWithTime
from adaptive_latents.plotting_functions import AnimationManager, plot_history_with_tail

from adaptive_latents import proSVD, proPLS

rng = np.random.default_rng(0)

# %% We'll do a simple simulation of jointly generated neural and behavioral data

n_seconds = 5 * 60
t = np.arange(0, n_seconds, 0.01)  # 100 Hz sampling rate

n_non_hd_cells = 10
n_head_direction_cells = 10
hd_tunings = np.linspace(0, 2 * np.pi, n_head_direction_cells, endpoint=False)  # preferred head direction for each cell

cannonical_tuning_curve = stats.vonmises(kappa=4).pdf  # von Mises distribution with concentration parameter kappa=4
def hd_tuning_curve(theta, preferred_direction):
    return cannonical_tuning_curve(theta - preferred_direction)

# %% Visualize the head direction tuning curves
theta = np.linspace(0, 2 * np.pi, 361)
theta_degrees = np.rad2deg(theta)

fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
for cell, preferred_direction in enumerate(hd_tunings):
    ax.plot(
        theta_degrees,
        hd_tuning_curve(theta, preferred_direction),
        label=f"Cell {cell + 1}",
    )

ax.set(
    xlim=(0, 360),
    xticks=np.arange(0, 361, 45),
    xlabel="Head direction (degrees)",
    ylabel="Tuning curve response",
    title="Head direction tuning curves",
)
ax.legend(title="Neuron", loc="upper left")
plt.show()

# %% Simulate head direction over time

smoothing_kernel = np.ones(100)
smoothing_kernel = smoothing_kernel / np.sum(smoothing_kernel)
hd = np.convolve(np.cumsum(rng.normal(0, 0.05, size=t.shape) + 0.0001), smoothing_kernel, mode='same') % (2 * np.pi)  # random walk in head direction
hd = ArrayWithTime(hd, t)

# %% Visualize the simulated head direction over time
fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
ax.plot(t, hd)
ax.set(
    xlabel="Time (s)",
    ylabel="Head direction (radians)",
    title="Simulated head direction over time",
)
plt.show()

# %% Visualize the distribution of simulated head directions

fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
ax.hist(hd, bins=50, density=True)
ax.set(
    xlabel="Head direction (radians)",
    ylabel="Density",
    title="Distribution of simulated head direction",
)
plt.show()

# %% Simulate neural responses based on head direction

no_noise_responses = np.zeros((len(t), n_non_hd_cells + n_head_direction_cells))
for i in range(n_head_direction_cells):
    no_noise_responses[:, i] = hd_tuning_curve(hd, hd_tunings[i])


tuning_curve_to_sample = cannonical_tuning_curve(np.linspace(0, 2 * np.pi, 200)) * 1.2
for i in range(n_head_direction_cells, n_non_hd_cells + n_head_direction_cells):
    no_noise_responses[:, i] = rng.choice(tuning_curve_to_sample, size=len(t), replace=True)

responses = no_noise_responses + rng.normal(0, 0.05, size=no_noise_responses.shape)
responses = ArrayWithTime(responses, t)
responses = responses - responses.mean(axis=0)  # center the responses

# %% Visualize the simulated neural responses

fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
opposite_cells = (0, n_head_direction_cells // 2)
for cell in opposite_cells:
    ax.plot(t, responses[:, cell], label=f"Head-direction cell {cell + 1}")

ax.set(
    xlabel="Time (s)",
    ylabel="Neural response",
    title="Simulated neural responses over time",
)
ax.legend(title="Cell")
plt.show()


# %% Visualize the simulated neural responses as a heatmap

fig, ax = plt.subplots(figsize=(5, 9), layout="constrained")
image = ax.imshow(
    responses,
    aspect="auto",
    cmap="viridis",
    interpolation='nearest',
    extent=(0, responses.shape[1], t[-1], t[0]),
)
ax.set(
    xlabel="Neuron index",
    ylabel="Time (s)",
    title="Simulated neural responses",
)
fig.colorbar(image, ax=ax, label="Neural response")
plt.show()

# %%

pro = proSVD(k=2, log_level=2)

latents = pro.offline_fit_then_transform(responses) # this is more stable than offline_run_on, but offline_run_on is more realistic for what we'd actually use
# latents = pro.offline_run_on(responses)

# %%

fig, ax = plt.subplots(figsize=(5, 5), layout="constrained")
ax.plot(latents[:, 0], latents[:, 1], '.')
ax.set(
    xlabel="Latent 1",
    ylabel="Latent 2",
    title="SVD latent-space trajectory",
    aspect="equal",
)
plt.show()

# Note how we don't get a ring, even though there's a ring structure in the latent space.


# %%

pls = proPLS(k=1, log_level=2)

latents = pls.offline_fit_then_transform([responses, hd.reshape(-1, 1)])

# %%

fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
ax.plot(latents)
ax.plot(hd / hd.var())
ax.set(
    xlabel="Time (samples)",
    ylabel="Latent value",
    title="PLS latent over time (head direction in radians)",
)
ax.legend(["Latent 1"])
plt.show()

# %%

pls = proPLS(k=2, log_level=2)

# instead of representing the head direction as a single angle like above, represent it as a 2d vector on the unit circle (cos(theta), sin(theta))
hd_circular_basis = ...
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

# %% Plot a scatterplot of latent 1 against latent 2 in black, and overlay the last 5 seconds with a line
# hint: use ArrayWithTime.slice_by_time(slice(start_t, end_t))

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
