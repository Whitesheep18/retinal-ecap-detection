import pickle
import datetime as dt
import os
import numpy as np
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

def train_and_eval(model, dataset, results, save_model_path, verbose=0):

    model_name = model.__class__.__name__

    X = np.load(os.path.join(dataset, "X.npy"))
    y = np.load(os.path.join(dataset, "y_reg.npy"))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if verbose: print(f"Training model {model}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if verbose: print("Evaluating model")
    mse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    if verbose: print(f"MSE: {mse}, R2: {r2}")
    with open(results, "a") as f:
        f.write(f"{dt.datetime.now()},{model_name},{mse},{r2},{dataset.split('/')[-1]}\n")
    
    
    if save_model_path != "False" and model_name != "InceptionTimeRegressor":
        model_path = os.path.join(save_model_path, f"{model_name}_{os.path.basename(dataset)}.pkl")
        print('Saving model to', model_path)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    