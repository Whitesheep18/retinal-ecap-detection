

if __name__ == "__main__":
    import numpy as np
    import argparse
    import os
    import glob
    import sys
    import pickle
    import datetime as dt
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import root_mean_squared_error, r2_score

    parser = argparse.ArgumentParser(description='Train a model on the simulated data')
    parser.add_argument('--models', type=str, required=True, nargs='*', default='LinearRegression', help='model or models to train')
    parser.add_argument('--results', type=str, default='results.csv', help='results_file [.csv]')
    parser.add_argument('--save_model_path', type=str, default='False', help='wether to save the model as pickle [<path_to_model>/False]')
    parser.add_argument('--dataset', type=str, required=True, help='path to dataset folder')
    args = parser.parse_args()

    if args.dataset is None:
        print("Please provide a dataset path")
        sys.exit(1)

    X = np.load(os.path.join(args.dataset, "X.npy"))
    y = np.load(os.path.join(args.dataset, "y_reg.npy"))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for model in args.models:
        if model == "LinearRegression":
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
        elif model == "FreshPINCE":        
            from aeon.regression.feature_based import FreshPRINCERegressor
            model = FreshPRINCERegressor()
        elif model == "InceptionNet":
            from aeon.regression.feature_based import InceptionTimeRegressor
            model = InceptionTimeRegressor()
        else:
            print(f"Unknown model {model}")
            sys.exit(1)

        print(f"Training model {model}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print("Evaluating model")
        mse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"MSE: {mse}, R2: {r2}")
        with open(args.results, "a") as f:
            f.write(f"{dt.datetime.now()},{model},{mse},{r2}\n")
        
        if args.save_model_path != "False":
            model_path = os.path.join(args.save_model_path, f"{model}_{os.path.basename(args.dataset)}.pkl")
            print('Saving model to', model_path)
            with open(model_path, "wb") as f:
                pickle.dump(model, f)