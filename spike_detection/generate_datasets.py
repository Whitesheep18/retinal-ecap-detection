import os
import numpy as np
import json
from src.make_dataset import  make_dataset_stim
import sys

idx = int(sys.argv[1])

snr_range = [10,20,30]
snr_value = snr_range[idx-1]
ME_value = 10
spontaneous_firing_Hz_value = 100
folder_name = f"DS-{snr_value}-{ME_value}-{spontaneous_firing_Hz_value}"

# Define the base directory for saving data
base_directory = 'data_simulated'
folder_path = os.path.join(base_directory, folder_name)
os.makedirs(folder_path, exist_ok=True)

# Define the parameters
params = {
    "num_cells_list": [0, 50],
    "white_SNR_dB_list": [snr_value],
    "mains_SNR_dB_list": [10],
    "spontaneous_firing_Hz_list": [100],
    "AP_amplitude_std_pct_list": [1, 10, 20]
}

# Save parameters to a JSON file
with open(os.path.join(folder_path, 'params.json'), 'w') as params_file:
    json.dump(params, params_file, indent=4)

# Generate the dataset
X, y_reg = make_dataset_stim(**params)

# Save X and y_reg to files
np.save(os.path.join(folder_path, 'X.npy'), X)
np.save(os.path.join(folder_path, 'y_reg.npy'), y_reg)

