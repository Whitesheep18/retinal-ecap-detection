import numpy as np
import matplotlib.pyplot as plt
from src.recording_generator import RecordingGenerator



if __name__ == "__main__":
                
    rec = RecordingGenerator(
        first_AP_stim_lambda_ms = 0.2,
        AP_length_mean_std_ms = [5, 1],
        AP_amplitude_mean_std_pct = [10, 0.5],
        SA_amplitude_mean_std_pct = [1, 0.1],
        num_cells = 50,
        spike_train_start_lambda_ms = 1,
        spike_train_rate_lambda = 3,
        inter_spike_train_interval_lambda_ms = 5,
        CAP_jitter_mean_std_ms = [1, 0.1],
        template_jitter_ms = 1, 
        )
    
    SAs, SA_indexes, APs, AP_indexes, is_spike, amount_spike, data = rec.generate(3, verbose=0)
    noised_data = rec.add_white_noise(data, SNR_dB=20)
    noised_data = rec.add_mains_electricity_noise(noised_data, SNR_dB=20)
    noised_data = rec.add_spontaneous_spikes(noised_data, firing_Hz=1000)

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
    num_samples = 300
    window_size = 200
    window_offset = 50
    X, y_class, y_reg = np.zeros((num_samples, window_size)), np.zeros(num_samples), np.zeros(num_samples)
    i = 0
    while i < num_samples:
        for SA_end_idx in range(len(SA_ends)-1):
            if i >= num_samples: break
            window_start = SA_ends[SA_end_idx]

            # while the window fits in this segment (so before the next stimulus)
            while window_start < SA_ends[SA_end_idx+1]-SA_size-window_size:
                if i >= num_samples: break
                X[i] = noised_data[window_start:window_start+window_size]
                # if more than 90% of the samples in this window are spikes then it is a spiking window
                y_class[i] = is_spike[window_start: window_start+window_size].mean() > 0.90 
                # count spikes fully contained in this window
                y_reg[i] = len([x for x in AP_indexes if x[0] >= window_start and x[-1] <= window_start+window_size]) # TODO: this might be slow
                if y_reg[i]:
                    axs[0].plot(X[i], alpha=0.2, color='olive')
                else:
                    axs[1].plot(X[i], alpha=0.2, color='pink')
                window_start += window_offset
                i += 1

    print("Samples found:", i)

    axs[0].set_ylim(-800, 800)
    axs[0].set_title('Spike')
    axs[1].set_ylim(-800, 800)
    axs[1].set_title('Not spike')
    plt.show()

    np.save('X.npy', X)
    np.save('y_class.npy', y_class)
    np.save('x_reg.npy', y_reg)
    
    plt.plot(y_reg)
    plt.xlabel('Window id')
    plt.ylabel('Num. fully contained spikes')
    plt.show()
        
            
