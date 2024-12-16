import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from src.utils import save_figure

def plot_loss(train_loss, validation_loss=None, title='Loss', id=''):
    """ if valid loss is None only plotting train loss """
    num_models = len(train_loss)
    colors = plt.cm.get_cmap('tab10', 10)  # Choose a colormap
    for i in range(num_models):
        plt.plot(np.arange(1,len(train_loss[i])+1), train_loss[i], label=f'Model {i+1}: train', c=colors(i))
        if validation_loss is not None:
            plt.plot(np.arange(1,len(validation_loss[i])+1), validation_loss[i], label=f'Model {i+1}: valid', c=colors(i), linestyle='--')
    plt.yscale('log')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    filename = f'loss_{id}_{timestamp}.png'
    save_figure(name=filename, figdir='./plots')