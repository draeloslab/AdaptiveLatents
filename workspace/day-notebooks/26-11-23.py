import numpy as np
import matplotlib.pyplot as plt
import colorsys
import matplotlib.colors as mcolors
import addcopyfighandler

# %%
u_1 = np.array([1, 0])
u_2 = np.array([1, np.sqrt(3)]) / 2
u_3 = np.array([-1, np.sqrt(3)]) / 2

basis = np.column_stack([u_1, u_2, u_3])

# %%
X, Y = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))

waves = np.cos(np.column_stack([X.ravel(), Y.ravel()]) @ basis)

fig, axs = plt.subplots(ncols=4, figsize=(12, 3), constrained_layout=True)

for i in range(3):
    axs[i].pcolormesh(X, Y, waves[:, i].reshape(X.shape))

axs[3].pcolormesh(X, Y, np.sum(waves, axis=1).reshape(X.shape))

for i in range(4):
    axs[i].set_aspect('equal')


axs[0].set_title('cos of wave 1')
axs[1].set_title('cos of wave 2')
axs[2].set_title('cos of wave 3')
axs[3].set_title('Sum of cosines of waves')
# %%
X, Y = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))

waves = np.column_stack([X.ravel(), Y.ravel()]) @ basis % (2 * np.pi)

fig, axs = plt.subplots(ncols=4, figsize=(12, 3), constrained_layout=True)

for i in range(3):
    axs[i].pcolormesh(X, Y, waves[:, i].reshape(X.shape), cmap='twilight')

def phase_rgb(phase1, phase2):
    rgb = mcolors.hsv_to_rgb(np.column_stack([phase1 % (2 * np.pi) / (2 * np.pi), np.ones_like(phase1), phase2 % (2 * np.pi) / (4 * np.pi) + .5]))
    return rgb.reshape(*X.shape, 3)


for ax, phase1, phase2 in [
    (axs[3], waves[:, 0], waves[:, 1]),
    # (axs[4], X.ravel(), Y.ravel()),
]:
    ax.imshow(
        phase_rgb(phase1, phase2),
        extent=(X.min(), X.max(), Y.min(), Y.max()),
        origin="lower",
        interpolation="nearest",
    )

for i in range(4):
    axs[i].set_aspect('equal')

axs[0].set_title('Wave 1 phase')
axs[1].set_title('Wave 2 phase')
axs[2].set_title('Wave 3 phase')
axs[3].set_title('2D phase from waves')
axs[3].set_title('2D phase from waves')


HD = np.linspace(0, 2 * np.pi, 100)


u_1 = np.array([1, 0])
u_2 = np.array([1, np.sqrt(3)]) / 2
u_3 = np.array([-1, np.sqrt(3)]) / 2
basis = np.column_stack([u_1, u_2, u_3])

waves = np.column_stack([X.ravel(), Y.ravel()])
np.column_stack([np.cos(waves @ basis), np.sin(waves @ basis)])

np.column_stack([np.cos(HD), np.sin(HD)])