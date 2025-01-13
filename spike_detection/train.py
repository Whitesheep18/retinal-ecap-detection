


if __name__ == "__main__":
    #python spike_detection/train.py --models LinearRegression --dataset DS_0_0_0
    import argparse
    import os
    import sys
    from src.train_and_eval import train_and_eval
    from src.utils import sorting_key

    parser = argparse.ArgumentParser(description='Train a model on the simulated data')
    parser.add_argument('--classification_models', type=str, required=True, nargs='*', default='Filter', help='classification model or models to train')
    parser.add_argument('--models', type=str, required=True, nargs='*', default='LinearRegression', help='model or models to train')
    parser.add_argument('--results', type=str, default='results.csv', help='results_file [.csv]')
    parser.add_argument('--save_model_path', type=str, default='False', help='wether to save the model as pickle [<path_to_model>/False]')
    parser.add_argument('--dataset', type=str, help='path to dataset folder')
    parser.add_argument('--dataset_idx', type=int, help='index of the dataset in the folder')
    parser.add_argument('--verbose', type=int, default=1, help="print what's going on")
    parser.add_argument('--comment', type=str, default='', help="anything else (eg. jobid) you want to add")
    # inception time arguments
    parser.add_argument('--n_models', type=int, default=5, help='number of models in ensemble (original and ours)')
    parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs in InceptionTime (original and ours)')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate in InceptionTime (original and ours)')
    parser.add_argument('--dropout', type=float, default=0.0, help='portion of weights to forget in InceptionTime (ours)')
    parser.add_argument('--l2_penalty', type=float, default=0, help='l2 penalty in InceptionTime (ours)')
    parser.add_argument('--init_stride', type=int, default=2, help='rate of initial downsampling CNN in InceptionTime (ours big time)')
    args = parser.parse_args()

    if args.dataset_idx is not None:
        datasets = [x for x in os.listdir('simulated_data') if x.startswith('DS')]
        datasets = sorted(datasets, key=sorting_key)
        dataset_path = os.path.join('simulated_data', datasets[args.dataset_idx-1])
    elif args.dataset is not None:
        dataset_path = os.path.join('simulated_data', args.dataset)
    else:
        print("Either dataset or dataset_idx must be given")
        sys.exit(1)

    for classification_model in args.classification_models:
        if classification_model == "Fliter":
            classifier = classification_model
        elif classification_model == "RandomForestClassifier":
            from sklearn.ensemble import RandomForestClassifier
            classifier = RandomForestClassifier()
        elif classification_model == "HIVECOTEV2":
            from aeon.classification.hybrid import HIVECOTEV2
            classifier = HIVECOTEV2(verbose=1, time_limit_in_minutes = 1)
        else:
            print(f"Unknown classifcation model {classification_model}")
            sys.exit(1)


    for model in args.models:
        if model == "LinearRegression":
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
        elif model == "ThresholdBased":
            from src.threshold_based import ThresholdBased
            model = ThresholdBased()
        elif model == "FreshPRINCERegressor":        
            from aeon.regression.feature_based import FreshPRINCERegressor
            model = FreshPRINCERegressor(verbose=args.verbose, default_fc_parameters='efficient', n_estimators=100)
        elif model == "DrCIFRegressor":        
            from aeon.regression.interval_based import DrCIFRegressor
            model = DrCIFRegressor(n_estimators=10, min_interval_length= 100)
        elif model == "InceptionTimeE":
            from src.inception_time.model import InceptionTimeE
            model = InceptionTimeE(verbose=args.verbose, epochs=args.n_epochs, learning_rate=args.learning_rate, 
                                  dropout=args.dropout, l2_penalty=args.l2_penalty, init_stride=args.init_stride,
                                  n_models=args.n_models)
        elif model == "AveragePrediction":
            from src.average_method import AveragePrediction
            model = AveragePrediction()
        else:
            print(f"Unknown model {model}")
            sys.exit(1)

        train_and_eval(model, classifier, dataset_path, args.results, args.save_model_path, verbose=args.verbose, comment=args.comment)

