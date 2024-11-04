import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import copy
from src.utils import get_template


class RecordingGenerator():
    def __init__(self, 
                 first_AP_stim_lambda_ms = 1, # Exponential
                 AP_length_mean_std_ms = [2, 0.1], # Gaussian
                 AP_amplitude_mean_std_pct = [1, 1], # Gaussian
                 SA_amplitude_mean_std_pct = [1, 0.1], # Gaussian
                 num_cells = 10, # Poisson # num_spike trains
                 spike_train_start_lambda_ms = 1, # Exponential # first spike after begining of SA
                 spike_train_rate_lambda = 1, # Poisson # number of spikes per spike train
                 inter_spike_train_interval_lambda_ms = 1, # Exponential
                 CAP_jitter_mean_std_ms = [1, 0.1], # Gaussian
                 template_jitter_ms = 0.01, # Uniform (width)
                 spontaneous_firing_rate_Hz = 1000, # Poisson
                 SA_templates = None,
                 AP_templates = None,
                 ME_template  = None
                 ):
        self.first_AP_stim_lambda_ms = first_AP_stim_lambda_ms
        self.AP_length_mean_std_ms = AP_length_mean_std_ms
        self.AP_amplitude_mean_std_pct = AP_amplitude_mean_std_pct
        self.SA_amplitude_mean_std_pct = SA_amplitude_mean_std_pct
        self.num_cells = num_cells
        self.spike_train_start_lambda_ms = spike_train_start_lambda_ms
        self.spike_train_rate_lambda = spike_train_rate_lambda
        self.inter_spike_train_interval_lambda_ms = inter_spike_train_interval_lambda_ms
        self.CAP_jitter_mean_std_ms = CAP_jitter_mean_std_ms
        self.template_jitter_ms = template_jitter_ms
        self.spontaneous_firing_rate_Hz = spontaneous_firing_rate_Hz

        self.fs = 30_000 # 30 kHz

        self.SA_templates = get_template('SA') if SA_templates is None else self._set_template(SA_templates) # 300 points, 30 kHz
        self.AP_templates = get_template('AP') if AP_templates is None else self._set_template(AP_templates) # variable sampling rate
        self.ME_template  = get_template('ME') if ME_template is None else self._set_template(ME_template)  # 600 points, 30 kHz   

        # should start with 0 and end with 0
        self.AP_templates = np.concatenate([np.zeros((len(self.AP_templates), 1)), 
                                            self.AP_templates, 
                                            np.zeros((len(self.AP_templates), 1))], axis=1)
        
        self.SA_length = self.SA_templates.shape[1]
        self.AP_length = self.AP_templates.shape[1]

    def _set_template(self, template):
        if isinstance(template, str):
            return np.load(template)
        else:
            return template        

    def interp_template(self, template, template_length_ms):

        time_old_ms = np.linspace(0, template_length_ms, len(template))
        
        num_points = int(self.fs*template_length_ms//1000)
        random_jitter = np.random.uniform(-self.template_jitter_ms/2, self.template_jitter_ms/2)
        time_new_ms = np.linspace(self.template_jitter_ms/2+random_jitter, 
                                template_length_ms-self.template_jitter_ms/2 + random_jitter, 
                                num_points)

        interp_func = interp1d(time_old_ms, template, kind='cubic')
        interp_template = interp_func(time_new_ms)
        return interp_template, time_new_ms
    
    def generate(self, num_stimuli=1,verbose=0):
        """
        Generates a recording with num_stimuli stimuli

        Returns:
        - values: stimuli and spikes
        - indexes: indexes of the values in the data array
        """
        segment_length=3000

        data = np.zeros(num_stimuli*segment_length)
        is_spike = np.zeros(num_stimuli*segment_length)
        amount_spikes = np.zeros(num_stimuli*segment_length)

        SAs, APs, SA_indexes, AP_indexes = [], [], [], []
        SA_template = self.SA_templates[np.random.choice(len(self.SA_templates)), :]

        for stimulus_idx in range(num_stimuli):
            if verbose: print(f"Stimulus {stimulus_idx}")
            
            SA, _ = self.interp_template(SA_template, 10) # add time jitter
            SA_amplitude = np.random.normal(*self.SA_amplitude_mean_std_pct)
            SA *= SA_amplitude
            
            start_SA = stimulus_idx*segment_length
            end_SA = start_SA + self.SA_length
            idxs = np.arange(start_SA, end_SA)
            data[idxs[0]: idxs[-1]+1] += SA

            SAs.append(SA)
            SA_indexes.append(idxs)

            res = self.add_spontaneous_spikes(data[start_SA: start_SA+self.SA_length], firing_Hz=self.spontaneous_firing_rate_Hz, return_APs=True)
            APs += res[0]
            AP_indexes += res[1]
            data[start_SA: start_SA+self.SA_length] = res[2]

            num_cells = np.random.poisson(self.num_cells)
            for cell_idx in range(num_cells):
                if verbose: print(f"Spike train {cell_idx}")

                start_idx = int(np.ceil(np.random.exponential(self.spike_train_start_lambda_ms) / 1000 * self.fs) + end_SA)
                num_spikes = np.random.poisson(self.spike_train_rate_lambda)
                AP_template = self.AP_templates[np.random.choice(len(self.AP_templates)), :]

                for spike_idx in range(num_spikes):
                    if verbose: print(f"Spike {spike_idx}")

                    AP_template_length_ms = np.random.normal(*self.AP_length_mean_std_ms)
                    AP_template_length_ms = np.max([AP_template_length_ms, self.template_jitter_ms*2])
                    AP, _ = self.interp_template(AP_template, AP_template_length_ms)
                    AP_amplitude = np.random.normal(*self.AP_amplitude_mean_std_pct)
                    AP *= AP_amplitude

                    if start_idx + len(AP) < num_stimuli*segment_length:
                        data[start_idx: start_idx + len(AP)] += AP
                        is_spike[start_idx: start_idx + len(AP)] = 1
                        amount_spikes[start_idx: start_idx + len(AP)] += 1

                    APs.append(AP)
                    AP_indexes.append(np.arange(start_idx, start_idx + len(AP)))

                    start_idx += len(AP)

                    # TOASK: make it fx beta so that intervals are more alike?
                    inter_spike_train_time = np.random.exponential(self.inter_spike_train_interval_lambda_ms) 
                    inter_spike_train_idx_offset = int(inter_spike_train_time / 1000 * self.fs)
                    start_idx += inter_spike_train_idx_offset

        return SAs, SA_indexes, APs, AP_indexes, is_spike, amount_spikes, data
    
    def add_white_noise(self, data, SNR_dB=10):
        """
        Adds white noise to the data with a given SNR in dB
        """
        signal_power = np.mean(data**2)
        noise_power = signal_power / 10**(SNR_dB/10)
        noise = np.random.normal(0, np.sqrt(noise_power), len(data))
        return data + noise
    
    def add_mains_electricity_noise(self, data, ME_template=None, amplitude_scaler=1):
        """
        Adds mains electricity noise to the data
        """

        if ME_template is None:
            ME_template = self.ME_template
        ME = np.tile(ME_template.flatten(), len(data)//len(ME_template) + 1)
        ME = np.roll(ME, np.random.randint(len(ME))) # random phase
        ME = ME[:len(data)]
        ME *= amplitude_scaler
        ME, _ = self.interp_template(ME, len(ME)/self.fs*1000) # jitter
        
        return data + ME
    
    def add_spontaneous_spikes(self, data, firing_Hz=1, return_APs=False):
        """
        Adds spontaneous spikes to the data
        """
        APs, AP_indexes = [], []
        num_spikes = np.random.poisson(firing_Hz * len(data) / self.fs)
        for _ in range(num_spikes):
            AP_template = self.AP_templates[np.random.choice(len(self.AP_templates)), :]
            AP_template_length_ms = np.random.normal(*self.AP_length_mean_std_ms)
            AP_template_length_ms = np.max([AP_template_length_ms, self.template_jitter_ms*2])
            AP, _ = self.interp_template(AP_template, AP_template_length_ms)
            AP_amplitude = np.random.normal(*self.AP_amplitude_mean_std_pct)
            AP *= AP_amplitude
            start_idx = np.random.randint(len(data) - len(AP))
            data[start_idx: start_idx + len(AP)] += AP
            if return_APs:
                APs.append(AP)
                AP_indexes.append(np.arange(start_idx, start_idx + len(AP)))
        if return_APs:
            return APs, AP_indexes, data
        return data
    
if __name__ == "__main__":
                
    rec = RecordingGenerator(
        first_AP_stim_lambda_ms = 0.2,
        AP_length_mean_std_ms = [5, 1],
        AP_amplitude_mean_std_pct = [1, 20],
        SA_amplitude_mean_std_pct = [1, 0.1],
        num_cells = 50,
        spike_train_start_lambda_ms = 1,
        spike_train_rate_lambda = 3,
        inter_spike_train_interval_lambda_ms = 5,
        CAP_jitter_mean_std_ms = [1, 0.1],
        template_jitter_ms = 1, 
        spontaneous_firing_rate_Hz=1000
        )
    
    SAs, SA_indexes, APs, AP_indexes, is_spike, amount_spike, data = rec.generate(2, verbose=0)
    noised_data = rec.add_white_noise(data, SNR_dB=10)
    noised_data = rec.add_mains_electricity_noise(noised_data, amplitude_scaler=1)

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
    np.save('y_reg.npy', y_reg)
    
    plt.plot(y_reg)
    plt.xlabel('Window id')
    plt.ylabel('Num. fully contained spikes')
    plt.show()
        
            
