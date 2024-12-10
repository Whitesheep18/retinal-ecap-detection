import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pandas as pd
import ast
from utils import setup_plotting_environment, save_figure

def RMSE_SNR_plot(csv_file_path, me_level=None, y_range=(0, 10)):
    setup_plotting_environment()
    # Load the CSV file
    data = pd.read_csv(csv_file_path)


    # Plot SNR vs RMSE for each model
    plt.figure(figsize=(10, 6))
    for model in data['Model'].unique():
        model_data = data[data['Model'] == model]
        if me_level is not None:
            model_data = model_data[model_data['ME SNR'] == me_level]
        plt.scatter(model_data['White SNR'], model_data['RMSE'], label=model, s=100)


    # Labels, title, and legend
    plt.xlabel('White SNR')
    plt.ylabel('RMS')
    plt.title(f'White SNR vs. RMS for models {"ME level = "+str(me_level) if me_level is not None else ""}')
    plt.legend(title='Models')
    if y_range is not None:
        plt.ylim(*y_range)
    plt.grid(True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    filename = f'snr_rms_plot_{timestamp}.png'
    save_figure(name=filename, figdir='./plots')    


def pred_test_plot(csv_file_path, model_name, me_level=None):
    setup_plotting_environment()
    data = pd.read_csv(csv_file_path)
    
    filtered_data = data[data['Model'] == model_name]
    
    if filtered_data.empty:
        print(f"No data found for Model: {model_name}")
        return

    snr_levels = sorted(filtered_data['White SNR'].unique())
    colors = plt.cm.get_cmap('tab10', len(snr_levels))  # Choose a colormap
    
    plt.figure(figsize=(10, 6))
    for idx, snr_value in enumerate(snr_levels):
        snr_data = filtered_data[filtered_data['White SNR'] == snr_value]
        if me_level is not None:
            snr_data = snr_data[snr_data['ME SNR'] == me_level]

                
        y_pred = snr_data['y_pred'].apply(lambda x: [float(num) for num in x.split(',')])
        y_pred = [item for sublist in y_pred for item in sublist]
        y_test = snr_data['y_test'].apply(lambda x: [float(num) for num in x.split(',')])
        y_test = [item for sublist in y_test for item in sublist]
        plt.scatter(y_test, y_pred, color=colors(idx), label=f'White SNR {snr_value}', alpha=0.5)
    
    # Add labels and title
    plt.xlabel('Real values')
    plt.ylabel('Predicted Values')
    plt.title(f'Predictions for {model_name} {"ME level = "+str(me_level) if me_level is not None else ""}')
    plt.grid(True)
    plt.legend(title='White SNR levels')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    filename = f'{model_name}_y_test_y_pred_plot_{timestamp}.png'
    save_figure(name=filename, figdir='./plots')


def residual_plot(csv_file_path, model_name, snr_value, me_level=None):
    setup_plotting_environment()
    data = pd.read_csv(csv_file_path)
    filtered_data = data[(data['Model'] == model_name) & (data['White SNR'] == snr_value)]
    if me_level is not None:
        filtered_data = filtered_data[filtered_data['ME SNR'] == me_level]
    
    if filtered_data.empty:
        print(f"No data found for Model: {model_name} and White SNR: {snr_value}")
        return

    y_pred = filtered_data['y_pred'].apply(lambda x: [float(num) for num in x.split(',')])
    y_pred = [item for sublist in y_pred for item in sublist]
    y_test = filtered_data['y_test'].apply(lambda x: [float(num) for num in x.split(',')])
    y_test = [item for sublist in y_test for item in sublist]


    plt.figure(figsize=(10, 6))
    residuals = np.array(y_pred) - np.array(y_test)
    plt.scatter(y_pred, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot for {model_name} at White SNR = {snr_value}{", ME SNR = "+str(me_level) if me_level is not None else ""}')
    plt.grid(True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    filename = f'{model_name}_residual_plot_{snr_value}_{timestamp}.png'
    save_figure(name=filename, figdir='./plots')


def residual_plot_individual(y_individual, y_test):
    setup_plotting_environment()
        # Ensure y_test is a 1D array.
    y_test = np.asarray(y_test).flatten()

    # Ensure shapes match for subtraction.
    if y_test.shape[0] != y_individual.shape[0]:
        raise ValueError(f"Shape mismatch: y_test has shape {y_test.shape}, but y_individual has shape {y_individual.shape}")

    # Calculate residuals for each model.
    residuals = np.squeeze(y_individual) - y_test[:, np.newaxis]

    # Number of models.
    num_models = residuals.shape[1]

    # Define a list of colors for each model using a colormap
    colors = plt.cm.get_cmap('tab10', 10)  # Use tab20 for more distinct colors if you have more than 10 models
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot residuals for each model against y_test.
    for model_idx in range(num_models):
        plt.scatter(
            y_test,                     # True labels (x-axis)
            residuals[:, model_idx],    # Residuals for the current model (y-axis)
            label=f'Model {model_idx + 1}',
            alpha=0.7,
            s=10,                       # Marker size
            color=colors(model_idx)     # Use the color from the colormap
        )

    plt.axhline(0, color='black', linestyle='--', linewidth=1, label='Zero Residual')
    plt.title("Residuals of Individual Model Predictions vs. y_test")
    plt.xlabel("y_test (True Labels)")
    plt.ylabel("Residuals")
    plt.legend(loc='upper right', fontsize=8)
    plt.tight_layout()

    # Save the plot to file
    filename = 'res.png'
    save_figure(name=filename, figdir='./plots')



    


#RMSE_SNR_plot('spike_detection/results.csv', me_level=10, y_range=None)
#pred_test_plot('spike_detection/results.csv', 'DrCIFRegressor', me_level=30)
#residual_plot('spike_detection/results.csv', 'DrCIFRegressor', 80, me_level=30)
