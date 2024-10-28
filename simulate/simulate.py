import numpy as np
import matplotlib.pyplot as plt
from src.recording_generator import RecordingGenerator



if __name__ == "__main__":
    AP_mean = 5

    rec = RecordingGenerator(
        first_AP_stim_lambda_ms = 0.2,
        AP_length_mean_std_ms = [5, 1],
        AP_amplitude_mean_std_pct = [AP_mean, 0.5],
        SA_amplitude_mean_std_pct = [1, 0.1],
        num_cells = 50,
        spike_train_start_lambda_ms = 1,
        spike_train_rate_lambda = 3,
        inter_spike_train_interval_lambda_ms = 5,
        CAP_jitter_mean_std_ms = [1, 0.1],
        template_jitter_ms = 1, 
        )
    
    SAs, SA_indexes, APs, AP_indexes, is_spike, amount_spike, data = rec.generate(20, verbose=0)
    noised_data = rec.add_white_noise(data, SNR_dB=20)
    noised_data = rec.add_mains_electricity_noise(noised_data, SNR_dB=20)
    noised_data = rec.add_spontaneous_spikes(noised_data, firing_Hz=1000)

    rec_no_cells = RecordingGenerator(
        first_AP_stim_lambda_ms = 0.2,
        AP_length_mean_std_ms = [5, 1],
        AP_amplitude_mean_std_pct = [AP_mean, 0.5],
        SA_amplitude_mean_std_pct = [1, 0.1],
        num_cells = 0,
        spike_train_start_lambda_ms = 1,
        spike_train_rate_lambda = 3,
        inter_spike_train_interval_lambda_ms = 5,
        CAP_jitter_mean_std_ms = [1, 0.1],
        template_jitter_ms = 1, 
        )
    
    SAs_no_cells, SA_indexes_no_cells, APs_no_cells, AP_indexes_no_cells, is_spike_no_cells, amount_spike_no_cells, data_no_cells = rec_no_cells.generate(10, verbose=0)
    noised_data_no_cells = rec_no_cells.add_white_noise(data_no_cells, SNR_dB=20)
    noised_data_no_cells = rec_no_cells.add_mains_electricity_noise(noised_data_no_cells, SNR_dB=20)
    noised_data_no_cells = rec_no_cells.add_spontaneous_spikes(noised_data_no_cells, firing_Hz=1000)

    plt.plot(noised_data, label='noised_data')
    plt.plot(data, label='data', color='orange')
    plt.legend()
    plt.show()

    plt.scatter(np.arange(len(data)), data, color=['olive' if x else 'pink' for x in is_spike], s=3)
    plt.title('Data without noise, spiked index marked green')
    plt.show()

    plt.plot(amount_spike)
    plt.title('Amount of spikes present at an index')
    plt.show()

    # order APs and AP_indexes by start index in AP_indexes

    #order = np.argsort([idx[0] for idx in AP_indexes])
    #APs = [APs[i] for i in order]
    #AP_indexes = [AP_indexes[i] for i in order]
    #AP_indexes_start = [idx[0] for idx in AP_indexes]
    #AP_indexes_end = [idx[-1] for idx in AP_indexes]

    # get samples for training 

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    SA_ends = [idx[-1] for idx in SA_indexes]

    SA_size = len(SA_indexes[0])
    num_samples = 2000
    window_size = 200
    window_offset = 50
    X, y_class, y_reg = np.zeros((num_samples, window_size)), np.zeros(num_samples), np.zeros(num_samples)

    # Set desired counts
    spiking_samples_target = 1000
    non_spiking_samples_target = 500
    no_cells_samples_target = 500

    # Initialize counters
    spiking_count = 0
    non_spiking_count = 0
    no_cell_count = 0
    i = 0

    while spiking_count < spiking_samples_target or non_spiking_count < non_spiking_samples_target:
        for SA_end_idx in range(len(SA_ends) - 1):
            if spiking_count >= spiking_samples_target and non_spiking_count >= non_spiking_samples_target:
                break  # Exit once targets are reached
            
            window_start = SA_ends[SA_end_idx]

            # Process windows within the segment before the next stimulus
            while window_start < SA_ends[SA_end_idx + 1] - SA_size - window_size:
                if spiking_count >= spiking_samples_target and non_spiking_count >= non_spiking_samples_target:
                    break

                # Extract the data for the current window
                X[i] = noised_data[window_start:window_start + window_size]
                spike_percentage = is_spike[window_start:window_start + window_size].mean()
                
                # Check if the window is classified as "spiking" (more than 90% spikes)
                if spike_percentage > 0.90 and spiking_count < spiking_samples_target:
                    y_class[i] = 1  # spiking window
                    spiking_count += 1
                    axs[0].plot(X[i], alpha=0.2, color='olive')
                elif spike_percentage <= 0.90 and non_spiking_count < non_spiking_samples_target:
                    y_class[i] = 0  # non-spiking window
                    non_spiking_count += 1
                    axs[1].plot(X[i], alpha=0.2, color='pink')
                else:
                    # If we have enough samples of the current type, skip this window
                    window_start += window_offset
                    continue
                
                # Count spikes fully contained in this window for regression target
                y_reg[i] = len([x for x in AP_indexes if x[0] >= window_start and x[-1] <= window_start + window_size])
                
                window_start += window_offset
                i += 1
    
    # Add samples without cells
    SA_ends_no_cells = [idx[-1] for idx in SA_indexes_no_cells]
    while no_cell_count < no_cells_samples_target:
        for SA_end_idx in range(len(SA_ends_no_cells) - 1):
            if no_cell_count >= no_cells_samples_target:
                break  # Exit once targets are reached
            
            window_start = SA_ends[SA_end_idx]

            # Process windows within the segment before the next stimulus
            while window_start < len(noised_data) - SA_size - window_size:
                if no_cell_count >= no_cells_samples_target:
                    break

                # Extract the data for the current window
                X[i] = noised_data_no_cells[window_start:window_start + window_size]
                y_class[i] = 0  # non-spiking window
                no_cell_count += 1
                axs[1].plot(X[i], alpha=0.2, color='pink')

      


    print("Samples found:", len(y_class))
    print("number of positive samples:", y_class.sum())
    

    axs[0].set_ylim(-800, 800)
    axs[0].set_title('Spike')
    axs[1].set_ylim(-800, 800)
    axs[1].set_title('Not spike')
    plt.show()

    np.save(f'X_AP_level={AP_mean}.npy', X)
    np.save(f'y_class_AP_level={AP_mean}.npy', y_class)
    np.save(f'x_reg_AP_level={AP_mean}.npy', y_reg)
    
    plt.plot(y_reg)
    plt.xlabel('Window id')
    plt.ylabel('Num. fully contained spikes')
    plt.show()
        
            
