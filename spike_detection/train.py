


if __name__ == "__main__":
    #python spike_detection/train.py --models LinearRegression --dataset DS_0_0_0
    import numpy as np
    import argparse
    import os
    import glob
    import sys
    import pickle
    import datetime as dt
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import root_mean_squared_error, r2_score
    from src.train_and_eval import train_and_eval

    parser = argparse.ArgumentParser(description='Train a model on the simulated data')
    parser.add_argument('--models', type=str, required=True, nargs='*', default='LinearRegression', help='model or models to train')
    parser.add_argument('--results', type=str, default='results.csv', help='results_file [.csv]')
    parser.add_argument('--save_model_path', type=str, default='False', help='wether to save the model as pickle [<path_to_model>/False]')
    parser.add_argument('--dataset', type=str, help='path to dataset folder')
    parser.add_argument('--dataset_idx', type=int, help='index of the dataset in the folder')
    args = parser.parse_args()

    if args.dataset_idx is not None:
        datasets = os.listdir('simulated_data')
        dataset_path = os.path.join('simulated_data', datasets[args.dataset_idx])
    elif args.dataset is not None:
        dataset_path = os.path.join('simulated_data', args.dataset)
    else:
        print("Either dataset or dataset_idx must be given")
        sys.exit(1)

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

        train_and_eval(model, dataset_path, args.results, args.save_model_path, verbose=1)

