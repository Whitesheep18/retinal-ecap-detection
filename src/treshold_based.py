
import numpy as np


class TresholdBased():
    """
    Treshold-based spike-counting method. AKA "Jespers method". 

    Used in Kusumanchi, Pratik et al. (2024). “Electrical stimulation of neuroretinas with 3D pyrolytic carbon 
    electrodes.” In: ASC Biomaterials Science Engineering. Manuscript submitted for publication.
    """
    def __init__(self, noise_factor=4.5, noise_method='rms', patience = 30):
        self.noise_factor = noise_factor
        self.noise_method = noise_method
        self.patience = patience # refactory period as number of points after a peak
    
    def fit(self, x, y):
        "no training required. this is an unsupervised method."
        pass

    def _predict_stimulation(self, x):
        "x is one stimulation on one channel"
        if self.noise_method == 'rms':
            base_noise_level = np.sqrt(np.mean(x**2))
        elif callable(self.noise_method):
            base_noise_level = self.noise_method(x)
        else:
            raise Exception("Unknown noise method {self.noise_method}")
        
        candidate_peak_mask = np.abs(x) > (base_noise_level * self.noise_factor)
        candidate_peak_idxs = np.where(candidate_peak_mask)[0]

        for i in range(len(x)):
            if i in candidate_peak_idxs:
                refactory_period_mask = (candidate_peak_idxs > i) & (candidate_peak_idxs < i + self.patience)
                candidate_peak_idxs = candidate_peak_idxs[~refactory_period_mask]

        return len(candidate_peak_idxs)

    def predict(self, x):
        n = x.shape[0]
        y = np.zeros(n)
        for i in range(n):
            y[i] = self._predict_stimulation(x[i])
        return y

            

if __name__ == '__main__':
    import os
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import root_mean_squared_error

    dataset = 'simulated_data/DS_0_10_10'
    X = np.load(os.path.join(dataset, "X.npy"))
    y = np.load(os.path.join(dataset, "y_reg.npy"))

    model = TresholdBased()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    y_pred = model.predict(X_test)
    print("Total no. spikes:", y_pred.sum())
    print("RMSE:", root_mean_squared_error(y_test, y_pred))
    

