import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


class RecordingGenerator():
    def __init__(self, 
                 first_AP_stim_lambda_ms = 1, # Exponential
                 AP_length_mean_std_ms = [2, 0.1], # Gaussian
                 AP_amplitude_mean_std_pct = [1, 1], # Gaussian
                 num_cells = 10, # Poisson # num_spike trains
                 spike_train_start_lambda_ms = 1, # Exponential # first spike after begining of SA
                 spike_train_rate_lambda = 1, # Poisson # number of spikes per spike train
                 inter_spike_train_interval_lambda_ms = 1, # Exponential
                 CAP_jitter_mean_std_ms = [1, 0.1], # Gaussian
                 template_jitter_ms = 0.01, # Uniform (width)
                 SA_templates = 'SA_templates.npy',
                 AP_templates = 'AP_templates.npy',
                 ME_template  = 'ME_template.npy',
                 ):
        self.first_AP_stim_lambda_ms = first_AP_stim_lambda_ms
        self.AP_length_mean_std_ms = AP_length_mean_std_ms
        self.AP_amplitude_mean_std_pct = AP_amplitude_mean_std_pct
        self.num_cells = num_cells
        self.spike_train_start_lambda_ms = spike_train_start_lambda_ms
        self.spike_train_rate_lambda = spike_train_rate_lambda
        self.inter_spike_train_interval_lambda_ms = inter_spike_train_interval_lambda_ms
        self.CAP_jitter_mean_std_ms = CAP_jitter_mean_std_ms
        self.template_jitter_ms = template_jitter_ms

        self.fs = 30_000 # 30 kHz

        self.SA_templates = self._set_template(SA_templates) # 300 points, 30 kHz
        print(self.SA_templates.shape)
        self.AP_templates = self._set_template(AP_templates) # variable sampling rate
        print(self.AP_templates.shape)
        self.ME_template  = self._set_template(ME_template)  # 600 points, 30 kHz   
        print(self.ME_template.shape)     

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
        time_new_ms = np.linspace(self.template_jitter_ms+random_jitter, 
                                  template_length_ms-self.template_jitter_ms + random_jitter, 
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
        values = []
        indexes = []

        for i in range(num_stimuli):
            if verbose: print(f"Stimulus {i}")
            SA_template = self.SA_templates[np.random.choice(len(self.SA_templates)), :]
            SA, _ = self.interp_template(SA_template, 10) # add time jitter
            
            start_SA = i*segment_length
            end_SA = start_SA + self.SA_length
            idxs = np.arange(start_SA, end_SA)
            data[idxs[0]: idxs[-1]+1] += SA

            values.append(SA)
            indexes.append(idxs)

            num_cells = np.random.poisson(self.num_cells)
            for j in range(num_cells):
                if verbose: print(f"Spike train {j}")

                start_idx = int(np.random.exponential(self.spike_train_start_lambda_ms) / 1000 * self.fs) + end_SA
                num_spikes = np.random.poisson(self.spike_train_rate_lambda)
                AP_template = self.AP_templates[np.random.choice(len(self.AP_templates)), :]

                for k in range(num_spikes):
                    if verbose: print(f"Spike {k}")

                    AP_template_length_ms = np.random.normal(*self.AP_length_mean_std_ms)
                    AP, _ = self.interp_template(AP_template, AP_template_length_ms)
                    AP_amplitude = np.random.normal(*self.AP_amplitude_mean_std_pct)
                    AP *= AP_amplitude

                    if start_idx + len(AP) < num_stimuli*segment_length:
                        data[start_idx: start_idx + len(AP)] += AP

                    values.append(AP)
                    indexes.append(np.arange(start_idx, start_idx + len(AP)))

                    start_idx += len(AP)

                    # TOASK: make it fx beta so that intervals are more alike?
                    inter_spike_train_time = np.random.exponential(self.inter_spike_train_interval_lambda_ms) 
                    inter_spike_train_idx_offset = int(inter_spike_train_time / 1000 * self.fs)
                    start_idx += inter_spike_train_idx_offset

        return values, indexes, data
    
    def add_white_noise(self, data, SNR_dB=10):
        """
        Adds white noise to the data with a given SNR in dB
        """
        signal_power = np.mean(data**2)
        noise_power = signal_power / 10**(SNR_dB/10)
        noise = np.random.normal(0, np.sqrt(noise_power), len(data))
        return data + noise
    
    def add_mains_electricity_noise(self, data, ME_template=None, SNR_dB=10):
        """
        Adds mains electricity noise to the data
        """
        # TODO: interpolate ME_template
        if ME_template is None:
            ME_template = self.ME_template
        ME = np.tile(ME_template.flatten(), len(data)//len(ME_template) + 1)
        ME = ME[:len(data)]
        # amplify to match SNR
        ME_power = np.mean(ME**2)
        data_power = np.mean(data**2)
        ME *= np.sqrt(data_power / ME_power / 10**(SNR_dB/10))
        # random phase
        ME = np.roll(ME, np.random.randint(len(ME)))
        
        return data + ME
    
    def add_spontaneous_spikes(self, data, firing_Hz=1):
        """
        Adds spontaneous spikes to the data
        """
        num_spikes = np.random.poisson(firing_Hz * len(data) / self.fs)
        for _ in range(num_spikes):
            AP_template = self.AP_templates[np.random.choice(len(self.AP_templates)), :]
            AP_template_length_ms = np.random.normal(*self.AP_length_mean_std_ms)
            AP, _ = self.interp_template(AP_template, AP_template_length_ms)
            AP_amplitude = np.random.normal(*self.AP_amplitude_mean_std_pct)
            AP *= AP_amplitude
            start_idx = np.random.randint(len(data) - len(AP))
            data[start_idx: start_idx + len(AP)] += AP
        return data

if __name__ == "__main__":
                
    rec = RecordingGenerator(
        first_AP_stim_lambda_ms = 0.2,
        AP_length_mean_std_ms = [5, 1],
        AP_amplitude_mean_std_pct = [1, 0.5],
        num_cells = 50,
        spike_train_start_lambda_ms = 1,
        spike_train_rate_lambda = 3,
        inter_spike_train_interval_lambda_ms = 5,
        CAP_jitter_mean_std_ms = [1, 0.1],
        template_jitter_ms = 1, 
        )
    
    segments, segment_idxs, data = rec.generate(2, verbose=0)
    noised_data = rec.add_white_noise(data, SNR_dB=20)
    noised_data = rec.add_mains_electricity_noise(noised_data, SNR_dB=20)
    noised_data = rec.add_spontaneous_spikes(noised_data, firing_Hz=1000)

    #plt.plot(noised_data, label='noised_data')
    plt.plot(data, label='data', color='orange')
    plt.legend()
    plt.show()
