import numpy as np
from src.recording_generator import RecordingGenerator

def get_noised_recording_stim(rec, num_stim = 15, num_samples = 300, window_size = 2700, white_SNR_dB=20, ME_amplitude_scaler=1, spontaneous_firing_Hz=1000):
    """
    rec: RecordingGenerator 
    """
    SAs, SA_indexes, APs, AP_indexes, is_spike, amount_spike, data = rec.generate(num_stim, verbose=0)
    noised_data = rec.add_white_noise(data, SNR_dB=white_SNR_dB)
    noised_data = rec.add_mains_electricity_noise(noised_data, amplitude_scaler=ME_amplitude_scaler)

    SA_ends = [idx[-1] for idx in SA_indexes]

    X, y_reg = np.zeros((num_samples, window_size)), np.zeros(num_samples)
    i = 0
    while i < num_samples:
        for SA_end_idx in range(len(SA_ends)):
            if i >= num_samples: break
            window_start = SA_ends[SA_end_idx]
            X[i] = noised_data[window_start:window_start+window_size]
            # count spikes fully contained in this window
            y_reg[i] = len([x for x in AP_indexes if x[0] >= window_start and x[-1] <= window_start+window_size])
            i += 1

    return X, y_reg

def make_dataset_stim(
    n=20,
    first_AP_stim_lambda_ms=0.2,
    AP_length_mean_std_ms=[5, 1],
    AP_amplitude_std_pct_list=[1],  # List of amplitude mean values
    SA_amplitude_mean_std_pct=[1, 0.1],
    num_cells_list=[50],
    spike_train_start_lambda_ms=1,
    spike_train_rate_lambda=3,
    inter_spike_train_interval_lambda_ms=5,
    CAP_jitter_mean_std_ms=[1, 0.1],
    template_jitter_ms=1,
    window_size=2700,
    white_SNR_dB_list=[20],
    ME_amplitude_scaler_list=[1],
    spontaneous_firing_Hz_list=[100]):

    n_cells_spike = n
    
    X_list, y_reg_list = [], []

    for num_cells in num_cells_list:
        for white_SNR_dB in white_SNR_dB_list:
            for ME_amplitude_scaler in ME_amplitude_scaler_list:
                for spontaneous_firing_Hz in spontaneous_firing_Hz_list:
                    for AP_std_pct in AP_amplitude_std_pct_list:
                        AP_amplitude_mean_std_pct = [1, AP_std_pct]


                        rec_cells = RecordingGenerator(
                            first_AP_stim_lambda_ms=first_AP_stim_lambda_ms,
                            AP_length_mean_std_ms=AP_length_mean_std_ms,
                            AP_amplitude_mean_std_pct=AP_amplitude_mean_std_pct,
                            SA_amplitude_mean_std_pct=SA_amplitude_mean_std_pct,
                            num_cells=num_cells,
                            spike_train_start_lambda_ms=spike_train_start_lambda_ms,
                            spike_train_rate_lambda=spike_train_rate_lambda,
                            inter_spike_train_interval_lambda_ms=inter_spike_train_interval_lambda_ms,
                            CAP_jitter_mean_std_ms=CAP_jitter_mean_std_ms,
                            template_jitter_ms=template_jitter_ms
                        )
                        
                        # Generate data samples with spikes and without spikes
                        X_cells, y_reg_cells = get_noised_recording_stim(
                            rec_cells, num_stim=n_cells_spike, num_samples=n_cells_spike,
                            window_size=window_size, white_SNR_dB=white_SNR_dB,
                            ME_amplitude_scaler=ME_amplitude_scaler, spontaneous_firing_Hz=spontaneous_firing_Hz
                        )

                        # Collect samples for each amplitude level in lists
                        X_list.append(X_cells)
                        y_reg_list.append(y_reg_cells)

    # Final dataset concatenation across different amplitude levels
    X = np.concatenate(X_list, axis=0)
    y_reg = np.concatenate(y_reg_list, axis=0)

    return X, y_reg

X, y_reg = make_dataset_stim(num_cells_list=[0, 50],white_SNR_dB_list=[10,20,50],   
                             ME_amplitude_scaler_list=[1, 2], spontaneous_firing_Hz_list=[100,1000],   
                             AP_amplitude_std_pct_list=[1, 10, 20])
