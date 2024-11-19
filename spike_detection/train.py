


if __name__ == "__main__":
    #python spike_detection/train.py --models LinearRegression --dataset DS_0_0_0
    import numpy as np
    import argparse
    import os
    import glob
    import sys
    import pickle
    import aeon
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
    parser.add_argument('--verbose', type=int, default=1, help="print what's going on")
    parser.add_argument('--comment', type=str, default='', help="anything else (eg. jobid) you want to add")
    args = parser.parse_args()

    if args.dataset_idx is not None:
        datasets = [x for x in os.listdir('simulated_data') if x.startswith('DS')]
        dataset_path = os.path.join('simulated_data', datasets[args.dataset_idx-1])
    elif args.dataset is not None:
        dataset_path = os.path.join('simulated_data', args.dataset)
    else:
        print("Either dataset or dataset_idx must be given")
        sys.exit(1)

    for model in args.models:
        if model == "LinearRegression":
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
        elif model == "TresholdBased":
            from src.treshold_based import ThresholdBased
            model = ThresholdBased()
        elif model == "FreshPRINCE":        
            from aeon.regression.feature_based import FreshPRINCERegressor
            model = FreshPRINCERegressor(verbose=args.verbose, default_fc_parameters='efficient', n_estimators=100)
        elif model == "DrCIF":        
            from aeon.regression.interval_based import DrCIFRegressor
            model = DrCIFRegressor(n_estimators=10, min_interval_length= 100, random_state=0)
        elif model == "InceptionNet":
            from aeon.regression.deep_learning import InceptionTimeRegressor
            n_epochs = 300
            if args.save_model_path != 'False':
                model_path = os.path.join(args.save_model_path, f"{model}_{os.path.basename(dataset_path)}.pkl")
                model = InceptionTimeRegressor(verbose=args.verbose, file_path = model_path, save_best_model = True, n_epochs=n_epochs)
            else:
                model = InceptionTimeRegressor(verbose=args.verbose, n_epochs=n_epochs)
        elif model == "InceptionNetPytorch":
            from src.inception_time.model import InceptionTime
            n_epochs = 300
            if args.save_model_path != 'False':
                model = InceptionTime(verbose=args.verbose, epochs=n_epochs)
            else:
                model = InceptionTime(verbose=args.verbose, epochs=n_epochs)
        else:
            print(f"Unknown model {model}")
            sys.exit(1)

        train_and_eval(model, dataset_path, args.results, args.save_model_path, verbose=args.verbose, comment=args.comment)

