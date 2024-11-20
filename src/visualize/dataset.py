import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np

def plot_single_sample(dataset, idx):
    X = np.load(os.path.join(dataset, "X.npy"))
    y = np.load(os.path.join(dataset, "y_reg.npy"))

    plt.plot(X[idx])
    plt.title(f"Sample {idx} from dataset {dataset} Spike count: {y[idx]}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    dataset_name = dataset.split('/')[-1]
    filename = f'{dataset_name}_sample_{idx}_{timestamp}.png'
    if not os.path.exists('./plots'):
        os.mkdir('./plots')
    filepath = os.path.join('./plots', filename)
    plt.savefig(filepath)
    plt.close()  


def plot_random_sample(dataset, num_samples):
    X = np.load(os.path.join(dataset, "X.npy"))
    y = np.load(os.path.join(dataset, "y_reg.npy"))

    N, T = X.shape

    indexes = np.arange(N)
    np.random.shuffle(indexes)
    indexes = indexes[:num_samples]

    plt.figure(figsize=(15, 7))

    for i in range(num_samples):
        plt.plot(np.arange(T)+i*T, X[indexes[i]], color='tab:blue')
        plt.axvline(x=i*T, color='tab:orange')
        plt.text(i*T+100, X[indexes[i]].max(), f"y = {int(y[indexes[i]])}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    dataset_name = dataset.split('/')[-1]
    filename = f'{dataset_name}_{timestamp}.png'
    if not os.path.exists('./plots'):
        os.mkdir('./plots')
    filepath = os.path.join('./plots', filename)

    plt.title(f'{num_samples} samples from {dataset_name}')
    plt.savefig(filepath)
    plt.close()

def plot_ds_overview(path_to_datasets, num_samples):

    datasets = [x for x in os.listdir(path_to_datasets) if x.startswith('DS')]
    dataset_idx = 0
    ymax = 0
    ncols = 2
    nrows = int(np.ceil(len(datasets)/ncols))
    fig, axs = plt.subplots(nrows, ncols)
    fig.set_figheight(3*ncols*1.1)
    fig.set_figwidth(4*nrows*1.1)
    for i in range(nrows):
        for j in range(ncols):
            dataset = os.path.join(path_to_datasets, datasets[dataset_idx])
            X = np.load(os.path.join(dataset, "X.npy"))
            y = np.load(os.path.join(dataset, "y_reg.npy"))

            N, T = X.shape

            indexes = np.arange(N)
            np.random.shuffle(indexes)
            indexes = indexes[:num_samples]

            for k in range(num_samples):
                axs[i, j].plot(np.arange(T)+k*T, X[indexes[k]], color='tab:blue')
                axs[i, j].axvline(x=k*T, color='tab:orange')
                maximum = X[indexes[k]].max()
                ymax = maximum if maximum > ymax else ymax 
                axs[i, j].text(k*T+100, maximum, f"y = {int(y[indexes[k]])}")
            axs[i, j].set_title(datasets[dataset_idx])
            dataset_idx += 1

    for i in range(nrows):
        for j in range(ncols):
            axs[i, j].set_ylim(-ymax*1.2, ymax*1.2)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    filename = f'datasets_{timestamp}.png'
    if not os.path.exists('./plots'):
        os.mkdir('./plots')
    filepath = os.path.join('./plots', filename)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


if __name__ == "__main__":
    #plot_random_sample('simulated_data/DS_20_10_10', 2)
    plot_ds_overview('simulated_data', 5)