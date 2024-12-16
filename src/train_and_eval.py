import pickle
import datetime as dt
import os
import numpy as np
from sklearn.metrics import r2_score, root_mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
import csv
from sklearn.ensemble import RandomForestClassifier

def train_and_eval(model, dataset, results, save_model_path, verbose=0, comment=''):

    classifier = RandomForestClassifier(random_state=0)
    model_name = model.__class__.__name__

    X = np.load(os.path.join(dataset, "X.npy"))
    y_reg = np.load(os.path.join(dataset, "y_reg.npy"))
    y_class = [0 if value < 5 else 1 for value in y_reg]

    SNR = dataset.split('_')[2]
    ME = dataset.split('_')[3]

    if verbose: print(f"Training classification model {model}")

    X_train, X_temp, y_class_train, y_class_temp, y_reg_train, y_reg_temp = train_test_split(
        X, y_class, y_reg, test_size=0.3, random_state=42)
    X_val, X_test, y_class_val, y_class_test, y_reg_val, y_reg_test = train_test_split(
        X_temp, y_class_temp, y_reg_temp, test_size=0.5, random_state=42)

    classifier.fit(X_train, y_class_train)

    y_class_train_pred = classifier.predict(X_train)
    y_class_val_pred = classifier.predict(X_val)
    y_class_test_pred = classifier.predict(X_test)

    accuracy_train = accuracy_score(y_class_train, y_class_train_pred)
    accuracy = accuracy_score(y_class_test, y_class_test_pred)
    
    if verbose: print("Train and test Classifier Accuracy:", accuracy_train, accuracy)

    class1_indices_val = y_class_val_pred == 1
    X_class1_val = X_val[class1_indices_val]
    y_reg_class1_val = y_reg_val[class1_indices_val]

    class1_indices_train = np.array(y_class_train) == 1
    X_class1_train = X_train[class1_indices_train]
    y_reg_class1_train = y_reg_train[class1_indices_train]

    class1_indices_test = np.array(y_class_test_pred) == 1
    X_class1_test = X_test[class1_indices_test]
    y_reg_class1_test = y_reg_test[class1_indices_test]

    
    if model_name == 'InceptionTime':
        model.fit(X_class1_train, y_reg_class1_train, X_class1_val, y_reg_class1_val)
    else: 
        model.fit(X_class1_train, y_reg_class1_train)

    if  model_name == "AveragePrediction":
        y_pred_train = model.predict(y_reg_class1_train, X_class1_train)
        y_pred = model.predict(y_reg_class1_train, X_class1_test)
    elif model_name == "InceptionTime":
        y_pred_train, _ = model.predict(X_class1_train)
        y_pred, _ = model.predict(X_class1_test)
    else:
        y_pred_train = model.predict(X_class1_train)
        y_pred = model.predict(X_class1_test)

    if verbose: print("Evaluating model")

    r2_train = r2_score(y_reg_class1_train, y_pred_train)
    rmse_train = r2_score(y_reg_class1_train, y_pred_train)
                        
    r2 = r2_score(y_reg_class1_test, y_pred)
    rmse = root_mean_squared_error(y_reg_class1_test, y_pred)

    if verbose: print(f"Train and test RMSE: {rmse_train}, {rmse}, R2: {r2_train}, {r2}")

    file_exists = os.path.isfile(results)
    with open(results, mode='a', newline='') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow(["Date", "Model", "Dataset", "White SNR", "ME SNR", 
                             "Accuracy train", "Accuracy test", "RMSE train", "RMSE test", "R2 train", "R2 test",
                              "comment", "params",
                             "y_pred", "y_test"])
        
        y_pred = ', '.join(map(str, y_pred))
        y_test = ', '.join(map(str, y_reg_class1_test))
        # Write the data row
        writer.writerow([
            dt.datetime.now(), 
            model_name, 
            dataset.split('/')[-1],  # Get the dataset name from the path
            SNR,
            ME, 
            accuracy_train,
            accuracy,
            rmse_train,
            rmse, 
            r2_train,
            r2, 
            comment,
            model.get_params(),
            y_pred,
            y_test, 
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

    if model_name == 'InceptionTime':
        from src.visualize.training import plot_loss
        plot_loss(model.train_loss, model.valid_loss, title=f'Trained on {os.path.basename(dataset)}', id=comment)
