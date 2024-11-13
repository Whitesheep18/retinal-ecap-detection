import pickle
import datetime as dt
import os
import numpy as np
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

def train_and_eval(model, dataset, results, save_model_path, verbose=0, comment=''):

    model_name = model.__class__.__name__

    X = np.load(os.path.join(dataset, "X.npy"))
    y = np.load(os.path.join(dataset, "y_reg.npy"))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if verbose: print(f"Training model {model}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if verbose: print("Evaluating model")
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    if verbose: print(f"RMSE: {rmse}, R2: {r2}")
    if os.path.exists(results):
        with open(results, "a") as f:
            f.write(f"{dt.datetime.now()},{model_name},{rmse},{r2},{dataset.split('/')[-1]},{comment}\n")
    else:
        with open(results, "w") as f:
            f.write("Date,Model,RMSE,R2,Dataset,comment\n")
            f.write(f"{dt.datetime.now()},{model_name},{rmse},{r2},{dataset.split('/')[-1]},{comment}\n")

    
    if save_model_path != "False":
        if 'inception' not in model_name.lower():
            model_path = os.path.join(save_model_path, f"{model_name}_{os.path.basename(dataset)}.pkl")
            print('Saving model to', model_path)
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
        elif model_name == 'InceptionTime':
            model_path = os.path.join(save_model_path, f"{model_name}_{os.path.basename(dataset)}")
            model.save(model_path)

    