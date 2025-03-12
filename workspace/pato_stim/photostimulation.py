from math import atan2

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA

from adaptive_latents.prosvd import proSVD


def plot_photostim(neuron_ID, method, spike_train_matrix, photostim_matrix, df):
    # Find the first and last frame number for the given neuron_ID
    neuron_data = df[df['neuron_ID'] == neuron_ID]
    first_frame = neuron_data['frame_number'].min()
    last_frame = neuron_data['frame_number'].max()
    
    # Adjust the photostim window based on the frame numbers
    C_stims = spike_train_matrix[:, :1147]
    C_photostims = spike_train_matrix[:, first_frame:last_frame]
    
    if method.lower() == 'pca':
        # Use PC of visual stimuli to plot photostimuli

        # Perform PCA to reduce dimensions
        pca_visual_stim = PCA(n_components=3)
        pca_visual_stim.fit_transform(C_stims.T)

        principal_components_photo = pca_visual_stim.transform(C_photostims.T)  # use transform instead of fit_transform
        threeDim_C = principal_components_photo[:, :3]

    elif method.lower() == 'prosvd':
        # Use proSVD on the photostim data
        pro = proSVD(k=3) # like the PCA() step
        pro.partial_fit(C_stims.T) # like the fit_transform step
        threeDim_C = pro.transform(C_photostims.T) # like the transform step

    else:
        raise ValueError("Method must be either 'PCA' or 'proSVD'")
    
    """Or use for proSVD:
    elif method.lower() == 'prosvd':
        # Use proSVD on the photostim data
        threeDim_C = prosvd_data(C_photostims.T, 3, 20, centering=True)"""
    
    # Extract the corresponding five values of the first column of photostim corresponding to the neuron of interest
    neuron_indices = df[df['neuron_ID'] == neuron_ID].index[:5]
    photostim_indices = np.clip(photostim_matrix[neuron_indices, 0].astype(int) - first_frame, 0, len(threeDim_C) - 1)  # Ensure indices are within bounds

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the trajectories of the processed data
    ax.plot(threeDim_C[:, 0], threeDim_C[:, 1], threeDim_C[:, 2], marker='o')

    # Add arrows to indicate the flow of time using quiver
    for i in range(len(threeDim_C) - 1):
        ax.quiver(threeDim_C[i, 0], threeDim_C[i, 1], threeDim_C[i, 2],
                  threeDim_C[i+1, 0] - threeDim_C[i, 0], threeDim_C[i+1, 1] - threeDim_C[i, 1], threeDim_C[i+1, 2] - threeDim_C[i, 2],
                  color='blue', arrow_length_ratio=0.1)

    # Add red dots at the time steps corresponding to the values in photostim[:5, 0]
    for idx, photostim_index in enumerate(photostim_indices):
        ax.scatter(threeDim_C[photostim_index, 0], threeDim_C[photostim_index, 1], threeDim_C[photostim_index, 2], color='red', s=50, zorder=5)
        ax.text(threeDim_C[photostim_index, 0], threeDim_C[photostim_index, 1], threeDim_C[photostim_index, 2], str(idx+1), color='black', fontsize=12)

    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    
    if method.lower() == 'pca':
        ax.set_title(f'Photostim trajectories with PCA // Neuron {neuron_ID}')
    elif method.lower() == 'prosvd':
        ax.set_title(f'Photostim trajectories with proSVD // Neuron {neuron_ID}')

    plt.show()


def extract_predictions(brs, offset):
    all_predictions = []

    for br in brs:
        predictions = br.h.log_pred_p[offset]
        all_predictions.append(predictions)

    return all_predictions

def rank_neurons(stim, C, photostim, predictions_list):
    # Create DataFrame for stim
    stim_columns = ['frame_number', 'ignore', 'angle_of_motion_L', 'angle_of_motion_R', 'timestamp']
    stim_df = pd.DataFrame(stim, columns=stim_columns)

    # Create DataFrame for C, where each column after the first represents a neuron
    neuron_ids = [f'neuron_{i}' for i in range(C.shape[0])]
    time_points = [f'frame_{i}' for i in range(C.shape[1])]
    C_df = pd.DataFrame(C.T, columns=neuron_ids, index=time_points)

    # Create DataFrame for photostim
    photostim_columns = ['frame_number', 'stim_counter', 'neuron_ID', 'position_X', 'position_Y']
    photostim_df = pd.DataFrame(photostim, columns=photostim_columns)

    # Handle neurons with two trials in the data frame: 14 and 98 turn into 15 and 99 in the second trial
    neuron_position_counts = photostim_df.groupby('neuron_ID')['position_X'].count()

    # Filter neurons with more than 6 occurrences
    neurons_with_many_positions = neuron_position_counts[neuron_position_counts > 6].index

    # Filter the DataFrame to include only these neurons
    filtered_neurons_df = photostim_df[photostim_df['neuron_ID'].isin(neurons_with_many_positions)]

    # Function to modify the neuron ID for the second group of 5 occurrences
    def modify_neuron_ids(df):
        modified_ids = df.copy()
        neuron_groups = modified_ids.groupby('neuron_ID')
        
        for neuron_id, group in neuron_groups:
            # Split the group into sets of 5
            for i in range(1, len(group) // 5 + 1):
                start_idx = i * 5
                if start_idx < len(group):
                    modified_ids.loc[group.index[start_idx:start_idx + 5], 'neuron_ID'] = str(int(neuron_id) + 1)
        
        return modified_ids

    # Apply the function to modify the neuron IDs in the original DataFrame
    modified_photostim_df = modify_neuron_ids(photostim_df)
    # Drop the last row by index (we only have one stimulation for the last neuron)
    modified_photostim_df = modified_photostim_df.drop(modified_photostim_df.index[-1])

    # Ensure the frame_number and neuron_ID are of integer type
    modified_photostim_df['frame_number'] = modified_photostim_df['frame_number'].astype(int)
    modified_photostim_df['neuron_ID'] = modified_photostim_df['neuron_ID'].astype(int)

    # Sort the df by 'frame_number'
    modified_photostim_df = modified_photostim_df.sort_values(by='frame_number').reset_index(drop=True)

    # Create a dictionary to store the results
    neuron_predictions = {}

    # Initialize variables to keep track of the current neuron and the starting index
    current_neuron = None
    start_idx = 0

    # Iterate over the sorted DataFrame
    for i in range(len(modified_photostim_df)):
        row = modified_photostim_df.iloc[i]
        frame_number = row['frame_number']
        neuron_id = row['neuron_ID']
        
        # Ensure frame_number is an integer
        frame_number = int(frame_number)
        
        # Check if we have moved to a different neuron
        if neuron_id != current_neuron:
            # If it's not the first neuron, save the previous neuron's predictions
            if current_neuron is not None:
                neuron_predictions[current_neuron].append(predictions_list[0][start_idx:frame_number])
            
            # Update the current neuron and initialize its list if necessary
            current_neuron = neuron_id
            if current_neuron not in neuron_predictions:
                neuron_predictions[current_neuron] = []
            
            # Update the start index to the current frame number
            start_idx = frame_number

    # Add the last neuron's predictions
    neuron_predictions[current_neuron].append(predictions_list[0][start_idx:])

    # Calculate and display the average of the prediction values for each neuron
    neuron_averages = {neuron_id: sum(values[0]) / len(values[0]) for neuron_id, values in neuron_predictions.items() if len(values[0]) > 0}

    # Calculate the drop in prediction value for each neuron compared to the neuron before it in the DataFrame
    neuron_drops = {}
    previous_neuron_id = None
    previous_avg = None

    # Iterate through the original order in the DataFrame
    for neuron_id in modified_photostim_df['neuron_ID'].unique():
        if neuron_id in neuron_averages:
            current_avg = neuron_averages[neuron_id]
            if previous_avg is not None:
                neuron_drops[neuron_id] = previous_avg - current_avg
            previous_neuron_id = neuron_id
            previous_avg = current_avg

    # Rank the neurons based on the drop in prediction value
    ranked_neurons = sorted(neuron_drops.items(), key=lambda x: x[1], reverse=True)

    # Create a DataFrame from the ranked neurons
    ranked_df = pd.DataFrame(ranked_neurons, columns=['neuron_ID', 'drop_value'])
    ranked_df['ranking'] = range(1, len(ranked_df) + 1)

    # Merge the rankings back into the modified DataFrame
    modified_photostim_df = pd.merge(modified_photostim_df, ranked_df, on='neuron_ID', how='left')

    # Add the average drop value to the modified DataFrame
    modified_photostim_df['average_PP'] = modified_photostim_df['neuron_ID'].map(neuron_averages)

    return modified_photostim_df

def bw_animation(data, A, mu, L, n_obs, save_path='fishBW_animation.mp4', fps=10):
    # Create the figure and axis
    fig, axs = plt.subplots(figsize=(8, 6))

    # Initial plot setup
    axs.plot(data[:, 0], data[:, 1], color='gray', alpha=0.8)
    scatter = axs.scatter([], [], c='k', zorder=10)
    ellipses = []

    # Initialize plot elements
    def init():
        scatter.set_offsets(np.empty((0, 2)))
        for ellipse in ellipses:
            ellipse.remove()
        ellipses.clear()
        return scatter,

    # Update function for animation
    def update(frame):
        n = frame
        mask = n_obs >= 0.5
        if mask[n] and n_obs[n] > 0.2:
            el = np.linalg.inv(L[n])
            sig = el.T @ el
            u, s, v = np.linalg.svd(sig)
            width, height = np.sqrt(s[0]) * 3, np.sqrt(s[1]) * 3
            angle = atan2(v[0, 1], v[0, 0]) * 360 / (2 * np.pi)
            ellipse = Ellipse((mu[n, 0], mu[n, 1]), width, height, angle=angle, zorder=8)
            ellipse.set_alpha(0.4)
            ellipse.set_clip_box(axs.bbox)
            ellipse.set_facecolor('#ed6713')
            axs.add_artist(ellipse)
            ellipses.append(ellipse)

        # Update scatter plot
        scatter.set_offsets(mu[mask][:n + 1, :2])
        return scatter,

    # Create the animation
    ani = FuncAnimation(fig, update, frames=np.arange(len(A)), init_func=init, blit=True, repeat=False)

    in1, in2 = -0.15, 1
    axs.text(in1, in2,s='a', transform=axs.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    # Save or display the animation
    ani.save(save_path, writer='ffmpeg', fps=fps)  # Save the animation
    plt.show()
