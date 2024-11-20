import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pandas as pd
import ast

def RMSE_SNR_plot(csv_file_path):
    # Load the CSV file
    data = pd.read_csv(csv_file_path)


    # Plot SNR vs RMSE for each model
    plt.figure(figsize=(10, 6))
    for model in data['Model'].unique():
        model_data = data[data['Model'] == model]
        plt.scatter(model_data['SNR'], model_data['RMSE'], label=model, s=100)


    # Labels, title, and legend
    plt.xlabel('SNR')
    plt.ylabel('RMS')
    plt.title('SNR vs. RMS for Different Models')
    plt.legend(title='Models')
    plt.grid(True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    filename = f'snr_rms_plot_{timestamp}.png'
    filepath = os.path.join('./plots', filename)
    plt.savefig(filepath)
    plt.close()  


def pred_test_plot(csv_file_path, model_name):
    data = pd.read_csv(csv_file_path)
    
    filtered_data = data[data['Model'] == model_name]
    
    if filtered_data.empty:
        print(f"No data found for Model: {model_name}")
        return

    snr_levels = sorted(filtered_data['SNR'].unique())
    colors = plt.cm.get_cmap('tab10', len(snr_levels))  # Choose a colormap
    
    plt.figure(figsize=(10, 6))
    for idx, snr_value in enumerate(snr_levels):
        snr_data = filtered_data[filtered_data['SNR'] == snr_value]
        
        y_pred = snr_data['y_pred'].apply(lambda x: [float(num) for num in x.split(',')])
        y_pred = [item for sublist in y_pred for item in sublist]
        y_test = snr_data['y_test'].apply(lambda x: [float(num) for num in x.split(',')])
        y_test = [item for sublist in y_test for item in sublist]
        
        plt.scatter(y_test, y_pred, color=colors(idx), label=f'SNR {snr_value}', alpha=0.5)
    
    # Add labels and title
    plt.xlabel('Real values')
    plt.ylabel('Predicted Values')
    plt.title(f'Predictions for {model_name}')
    plt.grid(True)
    plt.legend(title='SNR levels')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    filename = f'{model_name}_y_test_y_pred_plot_{timestamp}.png'
    filepath = os.path.join('./plots', filename)
    plt.savefig(filepath)
    plt.close()


def residual_plot(csv_file_path, model_name, snr_value):
    data = pd.read_csv(csv_file_path)
    filtered_data = data[(data['Model'] == model_name) & (data['SNR'] == snr_value)]
    
    if filtered_data.empty:
        print(f"No data found for Model: {model_name} and SNR: {snr_value}")
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
    plt.title(f'Residual Plot for {model_name} at {snr_value}')
    plt.grid(True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    filename = f'{model_name}_residual_plot_{snr_value}_{timestamp}.png'
    filepath = os.path.join('./plots', filename)
    plt.savefig(filepath)
    plt.close()


    


# RMSE_SNR_plot('spike_detection/results_inception_plain.csv')
pred_test_plot('spike_detection/results_inception_plain.csv', 'InceptionTime')
# residual_plot('spike_detection/results.csv', 'InceptionTime', 10)