import os
import numpy as np
import json
from src.make_dataset import  make_dataset_stim
import sys

idx = int(sys.argv[1])

white_snr_range = [-10,0,10,20,50,80]
me_snr_range = [0, 10]

white_snr_value = white_snr_range[(idx-1)%len(white_snr_range)]
white_ME_value = white_snr_range[(idx-1)//len(white_snr_range)]

spontaneous_firing_Hz_value = 10
folder_name = f"DS_{white_snr_value}_{white_ME_value}_{spontaneous_firing_Hz_value}"

# Define the base directory for saving data
base_directory = 'simulated_data'
folder_path = os.path.join(base_directory, folder_name)
os.makedirs(folder_path, exist_ok=True)

# Define the parameters
N = 2000 # total dataset size
num_cells_list = [0, 50]
ME_SNR_dB_list = [white_ME_value]
spontaneous_firing_Hz_list = [spontaneous_firing_Hz_value]
AP_amplitude_std_pct_list = [1, 10]
num_comb = len(num_cells_list) * len(ME_SNR_dB_list) * len(spontaneous_firing_Hz_list) * len(AP_amplitude_std_pct_list)
num_samples_per_comb = N//(num_comb)

print(f'Generating {num_samples_per_comb} samples per comb, ie. {num_samples_per_comb*num_comb} samples in total - close to {N}')

params = {
    "n": num_samples_per_comb,
    "num_cells_list": num_cells_list,
    "white_SNR_dB_list": [white_snr_value], # len is no datasets
    "ME_SNR_dB_list": ME_SNR_dB_list,
    "spontaneous_firing_Hz_list": spontaneous_firing_Hz_list,
    "AP_amplitude_std_pct_list": AP_amplitude_std_pct_list,
    "first_AP_stim_lambda_ms": 0.2,
    "AP_length_mean_std_ms": [5, 1],
    "SA_amplitude_mean_std_pct": [1, 0.1],
    "spike_train_start_lambda_ms": 1,
    "spike_train_rate_lambda": 1,
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

