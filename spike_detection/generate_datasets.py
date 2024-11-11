import os
import numpy as np
import json
from src.make_dataset import  make_dataset_stim
import sys

idx = int(sys.argv[1])

snr_range = [-10,0,10,20,50]
snr_value = snr_range[idx-1]
ME_value = 10
spontaneous_firing_Hz_value = 100
folder_name = f"DS_{snr_value}_{ME_value}_{spontaneous_firing_Hz_value}"

# Define the base directory for saving data
base_directory = 'simulated_data'
folder_path = os.path.join(base_directory, folder_name)
os.makedirs(folder_path, exist_ok=True)

# Define the parameters
N = 2000 # total dataset size
num_cells_list = [0, 50]
ME_amplitude_scaler_list = [10]
spontaneous_firing_Hz_list = [100]
AP_amplitude_std_pct_list = [1, 10, 20]
num_comb = len(num_cells_list) + len(ME_amplitude_scaler_list) + len(spontaneous_firing_Hz_list) + len(AP_amplitude_std_pct_list)

params = {
    "n": N//(num_comb), # num samples per combination
    "num_cells_list": num_cells_list,
    "white_SNR_dB_list": [snr_value], # len is no datasets
    "ME_amplitude_scaler_list": ME_amplitude_scaler_list,
    "spontaneous_firing_Hz_list": spontaneous_firing_Hz_list,
    "AP_amplitude_std_pct_list": AP_amplitude_std_pct_list,
    "first_AP_stim_lambda_ms": 0.2,
    "AP_length_mean_std_ms": [5, 1],
    "SA_amplitude_mean_std_pct": [1, 0.1],
    "spike_train_start_lambda_ms": 1,
    "spike_train_rate_lambda": 3,
    "inter_spike_train_interval_lambda_ms": 5,
    "CAP_jitter_mean_std_ms": [1, 0.1],
    "template_jitter_ms": 1,
    "window_size": 2700,
}

# Save parameters to a JSON file
with open(os.path.join(folder_path, 'params.json'), 'w') as params_file:
    json.dump(params, params_file, indent=4)

# Generate the dataset
X, y_reg = make_dataset_stim(**params)

# Save X and y_reg to files
np.save(os.path.join(folder_path, 'X.npy'), X)
np.save(os.path.join(folder_path, 'y_reg.npy'), y_reg)

