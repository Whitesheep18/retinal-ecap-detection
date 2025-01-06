import os
import numpy as np
import json
import sys
from src.recording_generator import RecordingGenerator
np.random.seed(42)

def get_noised_recording_stim(rec, num_stim = 15, num_samples = 300, window_size = 2700, white_SNR_dB=20, ME_SNR_dB=0):
    """
    rec: RecordingGenerator 
    """
    SAs, SA_indexes, APs, AP_indexes, is_spike, amount_spike, data = rec.generate(num_stim, verbose=0)
    white_noise = rec.create_white_noise(data, SNR_dB=white_SNR_dB)
    me_noise = rec.create_mains_electricity_noise(data, SNR_dB=ME_SNR_dB)
    noised_data = data + white_noise + me_noise

    SA_ends = [idx[-1] for idx in SA_indexes]

    X, y_reg = np.zeros((num_samples, window_size)), np.zeros(num_samples)
    i = 0
    while i < num_samples:
        for SA_end_idx in range(len(SA_ends)):
            if i >= num_samples: break
            window_start = SA_ends[SA_end_idx]+1
            X[i] = noised_data[window_start:window_start+window_size]
            # count spikes fully contained in this window
            y_reg[i] = len([x for x in AP_indexes if x[0] >= window_start and x[-1] <= window_start+window_size])
            i += 1

    return X, y_reg

def make_dataset_stim(
    n=20,
    AP_length_mean_std_ms=[5, 1],
    AP_amplitude_std_pct_list=[1],  # List of amplitude mean values
    SA_amplitude_mean_std_pct=[1, 0.1],
    num_cells_list=[50],
    spike_train_start_lambda_ms=1,
    spike_train_rate_lambda=1,
    inter_spike_train_interval_lambda_ms=5,
    CAP_jitter_mean_std_ms=[1, 0.1],
    template_jitter_ms=1,
    window_size=2700,
    white_SNR_dB_list=[20],
    ME_SNR_dB_list=[1],
    spontaneous_firing_Hz_list=[10]):

    n_cells_spike = n
    
    X_list, y_reg_list = [], []

    for num_cells in num_cells_list:
        for white_SNR_dB in white_SNR_dB_list:
            for ME_SNR_dB in ME_SNR_dB_list:
                for spontaneous_firing_Hz in spontaneous_firing_Hz_list:
                    for AP_std_pct in AP_amplitude_std_pct_list:
                        AP_amplitude_mean_std_pct = [1, AP_std_pct]


                        rec_cells = RecordingGenerator(
                            AP_length_mean_std_ms=AP_length_mean_std_ms,
                            AP_amplitude_mean_std_pct=AP_amplitude_mean_std_pct,
                            SA_amplitude_mean_std_pct=SA_amplitude_mean_std_pct,
                            num_cells=num_cells,
                            spike_train_start_lambda_ms=spike_train_start_lambda_ms,
                            spike_train_rate_lambda=spike_train_rate_lambda,
                            inter_spike_train_interval_lambda_ms=inter_spike_train_interval_lambda_ms,
                            CAP_jitter_mean_std_ms=CAP_jitter_mean_std_ms,
                            template_jitter_ms=template_jitter_ms,
                            spontaneous_firing_rate_Hz=spontaneous_firing_Hz
                        )
                        
                        # Generate data samples with spikes and without spikes
                        X_cells, y_reg_cells = get_noised_recording_stim(
                            rec_cells, num_stim=n_cells_spike, num_samples=n_cells_spike,
                            window_size=window_size, white_SNR_dB=white_SNR_dB,
                            ME_SNR_dB=ME_SNR_dB
                        )

                        # Collect samples for each amplitude level in lists
                        X_list.append(X_cells)
                        y_reg_list.append(y_reg_cells)

    # Final dataset concatenation across different amplitude levels
    X = np.concatenate(X_list, axis=0)
    y_reg = np.concatenate(y_reg_list, axis=0)

    print("shape X generation", X.shape)

    return X, y_reg


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("Provide dataset idx. Usage python generate_datasets.py <IDX>")
    
    idx = int(sys.argv[1])

    white_snr_range = [-10,0,10,20,50,80]
    ME_snr_range = [10, 80]

    white_snr_idx = (idx-1)%len(white_snr_range)
    ME_snr_idx = (idx-1)//len(white_snr_range)

    white_snr_value = white_snr_range[white_snr_idx]
    ME_snr_value = ME_snr_range[ME_snr_idx]

    print(f"IDX: {idx}")
    print(f"SNR: idx {white_snr_idx}, value {white_snr_value}")
    print(f"ME: idx {ME_snr_idx}, value {ME_snr_value}")

    spontaneous_firing_Hz_value = 10
    folder_name = f"DS_{white_snr_value}_{ME_snr_value}_{spontaneous_firing_Hz_value}"

    # Define the base directory for saving data
    base_directory = 'simulated_data'
    folder_path = os.path.join(base_directory, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # Define the parameters
    N = 4000 # total dataset size
    num_cells_list = [0, 30, 50, 70]
    ME_SNR_dB_list = [ME_snr_value]
    spontaneous_firing_Hz_list = [spontaneous_firing_Hz_value]
    AP_amplitude_std_pct_list = [1]
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
    np.random.seed(42)
    X, y_reg = make_dataset_stim(**params)

    # Save X and y_reg to files
    np.save(os.path.join(folder_path, 'X.npy'), X)
    np.save(os.path.join(folder_path, 'y_reg.npy'), y_reg)

