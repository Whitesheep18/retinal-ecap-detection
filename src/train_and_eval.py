import pickle
import datetime as dt
import os
import numpy as np
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
import csv

def train_and_eval(model, dataset, results, save_model_path, verbose=0, comment=''):

    model_name = model.__class__.__name__

    X = np.load(os.path.join(dataset, "X.npy"))
    y = np.load(os.path.join(dataset, "y_reg.npy"))
    SNR = dataset.split('_')[2]

    if verbose: print(f"Training model {model}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    
    if model_name == 'InceptionTime':
        model.fit(X_train, y_train, X_val, y_val)
    
    else: 
        model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)

    if verbose: print("Evaluating model")
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    if verbose: print(f"RMSE: {rmse}, R2: {r2}")
    results_dir = os.path.dirname(results)
    if results_dir and not os.path.exists(results_dir):
        os.makedirs(results_dir)

    file_exists = os.path.isfile(results)
    with open(results, mode='a', newline='') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow(["Date", "Model", "RMSE", "R2", "Dataset", "SNR", "y_pred", "y_test", "comment"])
        
        y_pred = ', '.join(map(str, y_pred))
        y_test = ', '.join(map(str, y_test))
        # Write the data row
        writer.writerow([
            dt.datetime.now(), 
            model_name, 
            rmse, 
            r2, 
            dataset.split('/')[-1],  # Get the dataset name from the path
            SNR, 
            y_pred,
            y_test, 
            comment
        ])
    
    if save_model_path != "False":
        if 'inception' not in model_name.lower():
            model_path = os.path.join(save_model_path, f"{model_name}_{os.path.basename(dataset)}.pkl")
            print('Saving model to', model_path)
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
        elif model_name == 'InceptionTime':
            model_path = os.path.join(save_model_path, f"{model_name}_{os.path.basename(dataset)}")
            model.save(model_path)

    