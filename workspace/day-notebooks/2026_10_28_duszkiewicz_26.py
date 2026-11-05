import numpy as np
from adaptive_latents.input_sources.datasets import Duszkiewicz26Dataset
import matplotlib.pyplot as plt

from scipy import stats

from adaptive_latents import proSVD, proPLS, ArrayWithTime, KernelSmoother
from adaptive_latents.utils import resample_matched_timeseries
rng = np.random.default_rng(0)

# TODO: add stagger or shuffle or reverse controls

# %% 1

d = Duszkiewicz26Dataset(9, bin_width=0.03)

# %%
# Note that there is a gap in the head direction recording.

plt.plot(d.neural_data.t, d.neural_data[:,0])
plt.plot(d.behavioral_data.t, d.behavioral_data)

for t in d.epochs.stop_time:
    plt.axvline(x=t, color='r')

# %% 2
def filter_for_good_times(a):
    # TODO: make this more robust, it currently ignores the `tags` field of the epochs df
    s1 = (d.epochs.start_time[1] < a.t) & (a.t < d.epochs.stop_time[1])
    s2 = (d.epochs.start_time[3] < a.t) & (a.t < d.epochs.stop_time[3])
    return a.slice(s1 | s2)

unsmoothed_x = filter_for_good_times(d.neural_data)

smoother = KernelSmoother(tau=.1/unsmoothed_x.dt) # TODO: tune this smoothing constant

x = smoother.offline_run_on([unsmoothed_x])
y = filter_for_good_times(d.behavioral_data)

x = x - x.mean(axis=0)

y_resampled = resample_matched_timeseries(y, y.t, x.t)

# %%

# Zoom in on this plot to compare the unsmoothed and smoothed neural data.
# Does it look reasonable? Try changing the smoothing constant in the KernelSmoother above and see how it affects the smoothed data.

plt.plot(unsmoothed_x.t, unsmoothed_x[:,0] - unsmoothed_x[:,0].mean(), '-')
plt.plot(x.t, x[:,0], '-')
# plt.xlim([5000, 5020])


# %%

# What does this plot show? What does it mean?
plt.plot(y_resampled, x[:,0], '.')


# %%

# Something goes wrong at these times (specifically for dataset 9). What is it?
# Can you fix it?

times_to_check = [3761.91425, 3761.94425, 3761.9742499999998, 3762.00425, 3762.0342499999997, 3762.06425, 3762.12425, 3762.1542499999996, 3762.18425, 3762.21425, 3930.9942499999997, 4549.50425, 5276.07425, 5279.854249999999, 9907.47425, 10331.34425, 10751.52425, 11346.874249999999, 12083.25425]


# %% 3

svd = proSVD(k=2)
latents_prosvd = svd.offline_fit_then_transform(x)

# %%

theta = np.linspace(0, 2*np.pi, 500) # for drawing a circle
hd_bins = np.linspace(0, 2*np.pi, 51) # bin edges for head direction colors
c = y_resampled.flatten() * np.nan
bar_c = theta * np.nan
for bin_i in range(len(hd_bins) - 1):
    s = (hd_bins[bin_i] <= y_resampled) & (y_resampled < hd_bins[bin_i + 1])
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

plt.plot(svd.Q)

# %% 4

hd_circular_basis = np.column_stack((np.cos(y), np.sin(y)))
hd_circular_basis = ArrayWithTime(hd_circular_basis, y.t)

pls = proPLS(k=2)

latents_pls = pls.offline_fit_then_transform([x, hd_circular_basis])



# %% 5



fig, axs = plt.subplots(figsize=(10, 10), ncols=2, nrows=2, layout="constrained", sharex='row', sharey='row')

for axs, latents, row_i in zip(axs, [latents_pls, latents_prosvd], [0, 1]):
    p = rng.permutation(latents.shape[0])

    ax = axs[0]
    scatter = ax.scatter(latents[p, 0], latents[p, 1], c=c[p], s=1, cmap=cmap)

    if row_i == 0:
        ax.scatter(np.cos(theta) + 4, np.sin(theta) + 2, c=bar_c, s=15, cmap=cmap)
        ax.axis('scaled')

    ax = axs[1]
    ax.hexbin(latents[:, 0], latents[:, 1], bins='log')
    ax.axis('scaled')

# %%
fig, ax = plt.subplots()

i = 17
ax.scatter(latents_pls[:, 0], latents_pls[:, 1], color='k', s=1)
ax.scatter(latents_pls[c==i, 0], latents_pls[c==i, 1], color='C0', s=1)

points = np.asarray(latents_pls[c == i])
mu = points.mean(axis=0)
sigma = np.cov(points, rowvar=False)

centered = points - mu
squared_mahalanobis = np.einsum('ij,jk,ki->i', centered, np.linalg.pinv(sigma), centered.T)

# for Gaussian data, squared Mahalanobis distance follows a chi-squared distribution with d degrees of freedom
is_outlier = squared_mahalanobis > stats.chi2.ppf(0.9999, df=points.shape[1])

ax.scatter(points[is_outlier, 0], points[is_outlier, 1], color='C1')

# %%
plt.plot(y.t, y)
for t in latents_pls.t[c == i][is_outlier]:
    plt.axvline(x=t, color='C1')
    print(t)

# %%
plt.hist(y)
# %%

plt.plot(pls.u)

# %%

tuning_weight = np.linalg.norm(pls.u, axis=1)
plt.plot(np.sort(tuning_weight)[::-1])
np.argsort(tuning_weight)